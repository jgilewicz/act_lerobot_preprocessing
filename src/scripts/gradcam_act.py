from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

from src.visualization import ACTGradCAM, draw_joint_overlay, overlay_cam_on_frame, tensor_to_rgb_uint8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Grad-CAM videos for ACT policies on offline episodes.")
    parser.add_argument("--policy-path", required=True, help="Local ACT checkpoint directory or HF repo id.")
    parser.add_argument("--dataset-id", required=True, help="LeRobot dataset id.")
    parser.add_argument("--dataset-root", default=None, help="Optional local dataset root.")
    parser.add_argument("--episode-index", type=int, required=True, help="Dataset episode index to replay.")
    parser.add_argument("--output-dir", required=True, help="Directory for gradcam.mp4 and gradcam_logs.npz.")
    parser.add_argument(
        "--image-key",
        default=None,
        help="Image feature key to visualize. Defaults to the first ACT image feature.",
    )
    parser.add_argument("--target-step", type=int, default=0, help="Action chunk step to explain.")
    parser.add_argument("--target-dim", type=int, required=True, help="Action dimension / joint to explain.")
    parser.add_argument("--alpha", type=float, default=0.45, help="Heatmap blend factor.")
    parser.add_argument("--fps", type=float, default=None, help="Output video FPS. Defaults to dataset FPS or 30.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame limit for debugging.")
    return parser.parse_args()


def _resolve_state_key(policy: ACTPolicy) -> str:
    for key, feature in policy.config.input_features.items():
        if feature.type is FeatureType.STATE:
            return key
    raise ValueError("ACT policy does not expose a robot state input feature.")


def _prepare_observation(sample: dict[str, Any], input_keys: list[str]) -> dict[str, Any]:
    missing = [key for key in input_keys if key not in sample]
    if missing:
        raise ValueError(f"Dataset sample is missing required policy inputs: {missing}")
    return {key: sample[key] for key in input_keys}


def _ensure_action_target(policy: ACTPolicy, target_step: int, target_dim: int) -> None:
    action_feature = policy.config.action_feature
    if action_feature is None:
        raise ValueError("ACT policy is missing an action output feature.")
    if target_step < 0 or target_step >= policy.config.chunk_size:
        raise ValueError(f"target_step={target_step} is out of range for chunk size {policy.config.chunk_size}.")
    if target_dim < 0 or target_dim >= action_feature.shape[0]:
        raise ValueError(f"target_dim={target_dim} is out of range for action dim {action_feature.shape[0]}.")


def _postprocess_action_chunk(postprocessor: Any, action_chunk: torch.Tensor) -> np.ndarray:
    processed = postprocessor(action_chunk)
    if isinstance(processed, torch.Tensor):
        return processed.detach().cpu().numpy()
    raise TypeError(f"Expected tensor from postprocessor, got {type(processed).__name__}.")


def _scalar_int(value: Any, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        return int(value.detach().cpu().item())
    if isinstance(value, np.ndarray):
        return int(value.item())
    return int(value)


def _resolve_dataset_root(dataset_id: str, dataset_root: str | None) -> Path | None:
    if not dataset_root:
        return None
    return Path(dataset_root) / dataset_id


def main() -> int:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    policy = ACTPolicy.from_pretrained(args.policy_path)
    if not isinstance(policy, ACTPolicy):
        raise TypeError(f"Expected ACTPolicy, got {type(policy).__name__}.")
    if not policy.config.vision_backbone.startswith("resnet"):
        raise ValueError(
            f"Grad-CAM only supports ResNet ACT backbones in v1, got {policy.config.vision_backbone!r}."
        )

    image_keys = list(policy.config.image_features.keys())
    if not image_keys:
        raise ValueError("ACT policy has no image features.")

    image_key = args.image_key or image_keys[0]
    if image_key not in image_keys:
        raise ValueError(f"Unknown image key {image_key!r}. Available: {image_keys}")
    print(f"Using image_key={image_key}")

    _ensure_action_target(policy, args.target_step, args.target_dim)
    state_key = _resolve_state_key(policy)

    dataset_root = _resolve_dataset_root(args.dataset_id, args.dataset_root)
    dataset = LeRobotDataset(args.dataset_id, root=dataset_root, episodes=[args.episode_index])
    if len(dataset) == 0:
        raise ValueError(f"Episode {args.episode_index} contains no frames.")

    fps = float(args.fps if args.fps is not None else getattr(dataset.meta, "fps", 30) or 30)
    preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=args.policy_path)

    first_sample = dataset[0]
    first_frame = tensor_to_rgb_uint8(first_sample[image_key])
    height, width = first_frame.shape[:2]

    video_path = output_dir / "gradcam.mp4"
    video_writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not video_writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {video_path}.")

    frame_indices: list[int] = []
    qpos_log: list[np.ndarray] = []
    action_chunk_log: list[np.ndarray] = []
    executed_action_log: list[np.ndarray] = []
    cam_log: list[np.ndarray] = []

    input_keys = list(policy.config.input_features.keys())

    with ACTGradCAM(policy) as gradcam:
        try:
            for local_index in range(len(dataset)):
                if args.max_frames is not None and local_index >= args.max_frames:
                    break

                sample = dataset[local_index]
                raw_frame = tensor_to_rgb_uint8(sample[image_key])
                observation = _prepare_observation(sample, input_keys)
                model_input = preprocessor(observation)

                cam, raw_actions = gradcam(
                    model_input,
                    image_keys=image_keys,
                    image_key=image_key,
                    target_step=args.target_step,
                    target_dim=args.target_dim,
                )
                action_chunk = _postprocess_action_chunk(postprocessor, raw_actions)[0]
                executed_action = action_chunk[args.target_step]
                qpos = sample[state_key].detach().cpu().numpy()

                frame_index = _scalar_int(sample.get("frame_index"), local_index)

                overlay = overlay_cam_on_frame(raw_frame, cam, alpha=args.alpha)
                overlay = draw_joint_overlay(
                    overlay,
                    frame_index=frame_index,
                    qpos=qpos,
                    target_step=args.target_step,
                    target_dim=args.target_dim,
                    target_value=float(executed_action[args.target_dim]),
                )
                video_writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

                frame_indices.append(frame_index)
                qpos_log.append(qpos)
                action_chunk_log.append(action_chunk)
                executed_action_log.append(executed_action)
                cam_log.append(cam)
        finally:
            video_writer.release()

    if not frame_indices:
        raise ValueError(f"Episode {args.episode_index} contains no renderable frames.")

    np.savez_compressed(
        output_dir / "gradcam_logs.npz",
        frame_index=np.asarray(frame_indices, dtype=np.int64),
        qpos=np.stack(qpos_log),
        action_chunk=np.stack(action_chunk_log),
        executed_action=np.stack(executed_action_log),
        cam=np.stack(cam_log),
        target_step=np.asarray(args.target_step, dtype=np.int64),
        target_dim=np.asarray(args.target_dim, dtype=np.int64),
        image_key=np.asarray(image_key),
    )
    print(f"Saved Grad-CAM video to {video_path}")
    print(f"Saved Grad-CAM logs to {output_dir / 'gradcam_logs.npz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
