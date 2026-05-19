from dataclasses import dataclass
from typing import Optional
import cv2
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class ExperimentConfig:
    name: str
    vision_backbone: str
    pretrained_backbone_weights: Optional[str]
    use_vae: bool
    image_transform: str = "rgb"
    canny_low: int = 50
    canny_high: int = 150
    blur_kernel_size: int = 5
    downsample_size: Optional[int] = None
    steps: int = 50_000
    batch_size: int = 8
    lr: float = 5e-5
    grad_clip_norm: float = 0.5
    warmup_steps: int = 2_000
    lr_plateau_factor: float = 0.5
    lr_plateau_patience: int = 1_000
    min_lr: float = 1e-6
    log_freq: int = 100
    save_freq: int = 10_000


EXPERIMENTS: dict[str, ExperimentConfig] = {
    "baseline": ExperimentConfig(
        name="baseline",
        vision_backbone="resnet18",
        pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
        use_vae=True,
        image_transform="rgb",
    ),
    "baseline_no_vae": ExperimentConfig(
        name="baseline_no_vae",
        vision_backbone="resnet18",
        pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
        use_vae=False,
        image_transform="rgb",
    ),
    "canny": ExperimentConfig(
        name="canny",
        vision_backbone="resnet18",
        pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
        use_vae=True,
        image_transform="canny",
        canny_low=50,
        canny_high=150,
    ),
    "canny_no_vae": ExperimentConfig(
        name="canny_no_vae",
        vision_backbone="resnet18",
        pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
        use_vae=False,
        image_transform="canny",
        canny_low=50,
        canny_high=150,
    ),
    "grayscale": ExperimentConfig(
        name="grayscale",
        vision_backbone="resnet18",
        pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
        use_vae=True,
        image_transform="grayscale",
    ),
    "sobel": ExperimentConfig(
        name="sobel",
        vision_backbone="resnet18",
        pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
        use_vae=True,
        image_transform="sobel",
    ),
    "blur_canny": ExperimentConfig(
        name="blur_canny",
        vision_backbone="resnet18",
        pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
        use_vae=True,
        image_transform="blur_canny",
        canny_low=50,
        canny_high=150,
        blur_kernel_size=5,
    ),
    "downsample_84": ExperimentConfig(
        name="downsample_84",
        vision_backbone="resnet18",
        pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
        use_vae=True,
        image_transform="downsample",
        downsample_size=84,
    ),
}


def _apply_transform_to_rgb_uint8_image(
    image_rgb: np.ndarray, exp: ExperimentConfig
) -> np.ndarray:
    if exp.image_transform == "rgb":
        return image_rgb

    if exp.image_transform == "downsample":
        if exp.downsample_size is None:
            raise ValueError(f"Experiment '{exp.name}' is missing `downsample_size`.")
        return cv2.resize(
            image_rgb,
            (exp.downsample_size, exp.downsample_size),
            interpolation=cv2.INTER_LINEAR,
        )

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    if exp.image_transform == "grayscale":
        image_2d = gray
    elif exp.image_transform == "canny":
        image_2d = cv2.Canny(gray, exp.canny_low, exp.canny_high)
    elif exp.image_transform == "sobel":
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(grad_x, grad_y)
        image_2d = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )
    elif exp.image_transform == "blur_canny":
        kernel = max(1, exp.blur_kernel_size)
        if kernel % 2 == 0:
            kernel += 1
        blurred = cv2.GaussianBlur(gray, (kernel, kernel), 0)
        image_2d = cv2.Canny(blurred, exp.canny_low, exp.canny_high)
    else:
        raise ValueError(f"Unsupported image transform: {exp.image_transform}")

    return np.repeat(image_2d[:, :, None], 3, axis=2)


def apply_image_transform_to_array(
    image_rgb: np.ndarray, transform_name: str
) -> np.ndarray:
    if transform_name == "none":
        return image_rgb
    if transform_name not in EXPERIMENTS:
        raise ValueError(
            f"Unknown transform '{transform_name}'. Available: {sorted(EXPERIMENTS)}"
        )
    return _apply_transform_to_rgb_uint8_image(image_rgb, EXPERIMENTS[transform_name])


def get_transformed_image_size(
    height: int | None, width: int | None, transform_name: str
) -> tuple[int | None, int | None]:
    if transform_name == "none":
        return height, width

    if transform_name not in EXPERIMENTS:
        raise ValueError(
            f"Unknown transform '{transform_name}'. Available: {sorted(EXPERIMENTS)}"
        )

    exp = EXPERIMENTS[transform_name]
    if exp.image_transform == "downsample":
        if exp.downsample_size is None:
            raise ValueError(
                f"Experiment '{exp.name}' is missing `downsample_size`."
            )
        return exp.downsample_size, exp.downsample_size

    return height, width


def apply_image_transform(batch: dict, exp: ExperimentConfig) -> dict:
    if exp.image_transform == "rgb":
        return batch

    for key in list(batch.keys()):
        if "image" not in key:
            continue

        imgs = batch[key]
        if exp.image_transform == "downsample":
            if exp.downsample_size is None:
                raise ValueError(
                    f"Experiment '{exp.name}' is missing `downsample_size`."
                )
            batch[key] = F.interpolate(
                imgs,
                size=(exp.downsample_size, exp.downsample_size),
                mode="bilinear",
                align_corners=False,
            )
            continue

        B, _, H, W = imgs.shape
        out = torch.zeros(B, 3, H, W, dtype=imgs.dtype, device=imgs.device)
        for i in range(B):
            img_np = (imgs[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            transformed = _apply_transform_to_rgb_uint8_image(img_np, exp)
            out[i] = (
                torch.from_numpy(transformed)
                .to(device=imgs.device, dtype=imgs.dtype)
                .permute(2, 0, 1)
                / 255.0
            )

        batch[key] = out
    return batch
