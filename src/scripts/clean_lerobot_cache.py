import argparse
import shutil
from pathlib import Path

from lerobot.utils.constants import HF_LEROBOT_HOME


def _cache_path_for_repo(repo_id: str) -> Path:
    target = HF_LEROBOT_HOME / repo_id
    base = HF_LEROBOT_HOME.resolve()
    resolved = target.resolve()
    if not resolved.is_relative_to(base):
        raise ValueError(f"Refusing to remove path outside HF_LEROBOT_HOME: {resolved}")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id", help="LeRobot dataset repo id, e.g. user/eval_dataset")
    args = parser.parse_args()

    cache_path = _cache_path_for_repo(args.repo_id)
    if cache_path.exists():
        shutil.rmtree(cache_path)
        print(f"Removed existing LeRobot cache: {cache_path}")


if __name__ == "__main__":
    main()
