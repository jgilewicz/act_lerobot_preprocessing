import logging
import time
from typing import Any

from numpy.typing import NDArray

from lerobot.cameras.opencv.camera_opencv import OpenCVCamera

from src.config import apply_image_transform_to_array

from .configuration_filtered_opencv import FilteredOpenCVCameraConfig

logger = logging.getLogger(__name__)


class FilteredOpenCVCamera(OpenCVCamera):
    def __init__(self, config: FilteredOpenCVCameraConfig):
        super().__init__(config)
        self.config = config

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        return OpenCVCamera.find_cameras()

    def connect(self, warmup: bool = True) -> None:
        last_error: Exception | None = None

        for attempt in range(1, self.config.connection_attempts + 1):
            try:
                super().connect(warmup=warmup)
                return
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "%s connection attempt %s/%s failed: %s",
                    self,
                    attempt,
                    self.config.connection_attempts,
                    exc,
                )
                self._reset_capture()
                if attempt < self.config.connection_attempts and self.config.connection_retry_delay_s > 0:
                    time.sleep(self.config.connection_retry_delay_s)

        assert last_error is not None
        raise last_error

    def _reset_capture(self) -> None:
        if self.thread is not None:
            self._stop_read_thread()

        if self.videocapture is not None:
            self.videocapture.release()
            self.videocapture = None

    def _postprocess_image(self, image: NDArray[Any]) -> NDArray[Any]:
        processed = super()._postprocess_image(image)
        return apply_image_transform_to_array(processed, self.config.filter_name)
