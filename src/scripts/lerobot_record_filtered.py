import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cameras.filtered_opencv import FilteredOpenCVCameraConfig  # noqa: F401
from src.config import get_transformed_image_size

import lerobot.scripts.lerobot_record as lerobot_record_module
from lerobot.robots.so_follower.so_follower import SOFollower


@property
def _patched_camera_features(self: SOFollower) -> dict[str, tuple[int | None, int | None, int]]:
    features: dict[str, tuple[int | None, int | None, int]] = {}
    for cam_name in self.cameras:
        camera_cfg = self.config.cameras[cam_name]
        height, width = get_transformed_image_size(
            camera_cfg.height,
            camera_cfg.width,
            getattr(camera_cfg, "filter_name", "none"),
        )
        features[cam_name] = (height, width, 3)
    return features


SOFollower._cameras_ft = _patched_camera_features

main = lerobot_record_module.main


if __name__ == "__main__":
    raise SystemExit(main())
