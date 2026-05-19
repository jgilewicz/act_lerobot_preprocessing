from __future__ import annotations

from collections.abc import Mapping

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.constants import OBS_IMAGES


def _require_resnet_act_policy(policy: ACTPolicy) -> None:
    if not isinstance(policy, ACTPolicy):
        raise TypeError(f"Expected ACTPolicy, got {type(policy).__name__}.")
    if not policy.config.vision_backbone.startswith("resnet"):
        raise ValueError(
            f"Grad-CAM only supports ResNet ACT backbones in v1, got {policy.config.vision_backbone!r}."
        )


def resolve_act_target_layer(policy: ACTPolicy) -> nn.Module:
    _require_resnet_act_policy(policy)

    backbone = getattr(policy.model, "backbone", None)
    if backbone is None:
        raise ValueError("ACT policy does not expose a visual backbone.")

    try:
        return backbone["layer4"]
    except Exception as exc:
        raise ValueError("Could not resolve ResNet layer4 from ACT backbone.") from exc


class ACTGradCAM:
    def __init__(self, policy: ACTPolicy, target_layer: nn.Module | None = None):
        _require_resnet_act_policy(policy)
        self.policy = policy
        self.target_layer = target_layer or resolve_act_target_layer(policy)
        self.activations: list[Tensor] = []
        self.gradients: list[Tensor] = []

        self._fwd_hook = self.target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor) -> None:
        self.activations.append(output)

    def _save_gradient(
        self,
        _module: nn.Module,
        _grad_input: tuple[Tensor | None, ...],
        grad_output: tuple[Tensor | None, ...],
    ) -> None:
        grad = grad_output[0]
        if grad is not None:
            self.gradients.append(grad)

    def remove(self) -> None:
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    def __enter__(self) -> "ACTGradCAM":
        return self

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        self.remove()

    def __call__(
        self,
        batch: Mapping[str, Tensor],
        *,
        image_keys: list[str],
        image_key: str,
        target_step: int = 0,
        target_dim: int | None = None,
    ) -> tuple[np.ndarray, Tensor]:
        if image_key not in image_keys:
            raise ValueError(f"Unknown image key {image_key!r}. Available: {image_keys}")

        image_index = image_keys.index(image_key)
        model_batch = dict(batch)
        model_batch[OBS_IMAGES] = [model_batch[key] for key in image_keys]

        self.activations.clear()
        self.gradients.clear()
        self.policy.zero_grad(set_to_none=True)

        actions, _ = self.policy.model(model_batch)

        if target_step < 0 or target_step >= actions.shape[1]:
            raise ValueError(
                f"target_step={target_step} is out of range for chunk size {actions.shape[1]}."
            )
        if target_dim is not None and (target_dim < 0 or target_dim >= actions.shape[2]):
            raise ValueError(
                f"target_dim={target_dim} is out of range for action dim {actions.shape[2]}."
            )

        if target_dim is None:
            score = actions[:, target_step, :].norm(dim=-1).sum()
        else:
            score = actions[:, target_step, target_dim].sum()

        score.backward()

        if len(self.activations) != len(image_keys):
            raise RuntimeError(
                f"Expected {len(image_keys)} backbone activations, got {len(self.activations)}."
            )
        if len(self.gradients) != len(image_keys):
            raise RuntimeError(
                f"Expected {len(image_keys)} backbone gradients, got {len(self.gradients)}."
            )

        activations = self.activations[image_index]
        gradients = list(reversed(self.gradients))[image_index]

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1)
        cam = F.relu(cam)
        cam = cam[0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.detach().cpu().numpy(), actions.detach()


def tensor_to_rgb_uint8(image: Tensor | np.ndarray) -> np.ndarray:
    if isinstance(image, Tensor):
        image = image.detach().cpu().float().numpy()

    if image.ndim != 3:
        raise ValueError(f"Expected 3D image tensor, got shape {image.shape}.")

    if image.shape[0] in {1, 3} and image.shape[-1] not in {1, 3}:
        image = np.transpose(image, (1, 2, 0))

    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    elif image.shape[-1] != 3:
        raise ValueError(f"Expected 1 or 3 channels, got shape {image.shape}.")

    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)

    return image


def overlay_cam_on_frame(frame_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    height, width = frame_rgb.shape[:2]
    cam_resized = cv2.resize(cam, (width, height), interpolation=cv2.INTER_LINEAR)
    heatmap = np.uint8(np.clip(cam_resized, 0.0, 1.0) * 255)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = ((1.0 - alpha) * frame_rgb.astype(np.float32)) + (alpha * heatmap.astype(np.float32))
    return np.clip(overlay, 0, 255).astype(np.uint8)


def draw_joint_overlay(
    frame_rgb: np.ndarray,
    *,
    frame_index: int,
    qpos: np.ndarray,
    target_step: int,
    target_dim: int,
    target_value: float,
    max_joints: int = 8,
) -> np.ndarray:
    frame = frame_rgb.copy()
    qpos_text = ", ".join(f"{value:+.2f}" for value in qpos[:max_joints])
    if len(qpos) > max_joints:
        qpos_text += " ..."

    lines = [
        f"frame={frame_index}",
        f"target_step={target_step} target_dim={target_dim} target_value={target_value:+.3f}",
        f"qpos: {qpos_text}",
    ]

    y = 30
    for line in lines:
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 26

    return frame
