import os
import subprocess
import sys
from typing import Iterable


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    return int(value)


def _iter_model_filters(raw_pairs: str) -> Iterable[tuple[str, str]]:
    for raw_pair in raw_pairs.split():
        model, separator, filter_name = raw_pair.partition(":")
        if not separator or not model or not filter_name:
            raise ValueError(
                f"Invalid EVAL_MODEL_FILTERS entry '{raw_pair}'. Expected MODEL:FILTER."
            )
        yield model, filter_name


def main() -> int:
    eval_trials = _env_int("EVAL_TRIALS", 10)
    eval_dataset_repo = os.environ["EVAL_DATASET_REPO"]
    eval_model_filters = os.environ["EVAL_MODEL_FILTERS"]
    eval_episodes = _env_int("EVAL_ALL_EPISODES", 1)
    episode_time = _env_int("EVAL_ALL_EPISODE_TIME", 60)
    reset_time = _env_int("EVAL_ALL_RESET_TIME", 0)

    for trial in range(1, eval_trials + 1):
        for model, filter_name in _iter_model_filters(eval_model_filters):
            print(
                f"==> trial {trial}/{eval_trials} model={model} filter={filter_name}",
                flush=True,
            )
            dataset_repo = f"{eval_dataset_repo}_{filter_name}_t{trial}"
            command = [
                "make",
                "eval",
                f"MODEL={model}",
                f"FILTER={filter_name}",
                f"EVAL_EPISODES={eval_episodes}",
                f"EPISODE_TIME={episode_time}",
                f"RESET_TIME={reset_time}",
                f"EVAL_DATASET_REPO={dataset_repo}",
            ]
            subprocess.run(command, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
