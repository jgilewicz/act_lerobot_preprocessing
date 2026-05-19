from dataclasses import dataclass
from pathlib import Path

from lerobot.cameras.configs import ColorMode, Cv2Backends, Cv2Rotation
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

from src.config import EXPERIMENTS


@OpenCVCameraConfig.register_subclass("filtered_opencv")
@dataclass
class FilteredOpenCVCameraConfig(OpenCVCameraConfig):
    index_or_path: int | Path
    filter_name: str = "none"
    color_mode: ColorMode = ColorMode.RGB
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    warmup_s: int = 3
    fourcc: str | None = None
    backend: Cv2Backends = Cv2Backends.ANY
    connection_attempts: int = 3
    connection_retry_delay_s: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.filter_name != "none" and self.filter_name not in EXPERIMENTS:
            raise ValueError(
                f"Unknown filter_name '{self.filter_name}'. Available: ['none', {', '.join(sorted(EXPERIMENTS))}]"
            )
        if self.color_mode != ColorMode.RGB:
            raise ValueError("FilteredOpenCVCameraConfig requires color_mode='rgb'.")
        if self.connection_attempts < 1:
            raise ValueError("FilteredOpenCVCameraConfig requires connection_attempts >= 1.")
        if self.connection_retry_delay_s < 0:
            raise ValueError("FilteredOpenCVCameraConfig requires connection_retry_delay_s >= 0.")
