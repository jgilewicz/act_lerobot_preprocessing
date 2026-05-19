# UM ACT LeRobot

Training [ACT](https://github.com/huggingface/lerobot) policies on a pick-and-lift task.

## Experiments

| Name | Backbone | VAE | Transform |
|------|----------|-----|-----------|
| `baseline` | ResNet18 | yes | `rgb` |
| `baseline_no_vae` | ResNet18 | no | `rgb` |
| `canny` | ResNet18 | yes | `canny` |
| `canny_no_vae` | ResNet18 | no | `canny` |
| `grayscale` | ResNet18 | yes | `grayscale` |
| `sobel` | ResNet18 | yes | `sobel` |
| `blur_canny` | ResNet18 | yes | `blur_canny` |
| `downsample_84` | ResNet18 | yes | `downsample_84` |

## Setup

```bash
pip install -e .
```

For SO100/SO101 follower eval on Feetech hardware, this repo also installs `feetech-servo-sdk` so `scservo_sdk` is available. If you already created `.venv` before this change, rerun:

```bash
pip install -e .
```

Copy `.env.example` to `.env` and fill in the values:

```
DATASET_ID=<hf-dataset-repo-id>        # e.g. username/dataset-name
POLICY_REPO_ID=<hf-model-repo-prefix>  # e.g. username/act — experiment name appended automatically
HF_TOKEN=hf_...                        # Hugging Face write token
```

Log in to Weights & Biases:

```bash
wandb login
```

## Running

```bash
make train EXP=baseline
make train-all
```

## Evaluation

Single eval runs use `src/scripts/eval.py`, which wraps the filtered camera recorder and returns the SO follower to `HOME_ACTION` after each run.

Eval with filtered camera input:

```bash
make eval MODEL=W1ndrunn3rr/act_pick_and_lift_v2_canny FILTER=canny ROBOT_PORT=/dev/ttyACM0 ROBOT_ID=my_robot
```

The main eval-related `Makefile` variables are:

| Variable | Purpose |
|----------|---------|
| `MODEL` | Policy repo or local policy path passed to `--policy.path` |
| `FILTER` | Camera transform name such as `none`, `canny`, `grayscale`, `blur_canny`, `downsample_84` |
| `EVAL_DATASET_REPO` | Output dataset repo id / local dataset name for the recorded eval |
| `EVAL_EPISODES` | Number of episodes to record in one `make eval` run |
| `EPISODE_TIME` | Per-episode evaluation time in seconds |
| `RESET_TIME` | Time between episodes in seconds |
| `HOME_ACTION` | JSON motor target used to return the follower to a safe pose after eval |
| `HOME_RETURN_TIME_S` | Duration of the home return motion |
| `HOME_HOLD_TIME_S` | Extra hold time after reaching the home pose |

Batch eval runs use `src/scripts/eval_all.py`, which iterates over `EVAL_MODEL_FILTERS` and `EVAL_TRIALS` and dispatches `make eval` for each pair:

```bash
make eval-all
```

Useful overrides:

```bash
make eval-all EVAL_TRIALS=3 EVAL_DATASET_REPO=my_eval_runs
make eval-all EVAL_MODEL_FILTERS="user/policy_a:canny user/policy_b:grayscale"
```

## Grad-CAM

Offline ACT Grad-CAM analysis is available through `src/scripts/gradcam_act.py`. It loads an ACT policy, replays one LeRobot dataset episode, renders a Grad-CAM MP4 for one selected camera stream, and saves aligned logs as `gradcam_logs.npz`.

```bash
python -m src.scripts.gradcam_act \
  --policy-path outputs/train/baseline/step_0050000 \
  --dataset-id W1ndrunn3rr/pick_and_lift_v2 \
  --episode-index 0 \
  --output-dir outputs/gradcam/baseline_ep0 \
  --target-dim 0
```

Useful options:

| Option | Purpose |
|--------|---------|
| `--image-key` | Selects which ACT image feature to visualize when multiple cameras are present |
| `--target-step` | Chooses which action step inside the ACT chunk to explain |
| `--target-dim` | Chooses which action dimension / joint to explain |
| `--fps` | Overrides output video FPS |
| `--max-frames` | Truncates the episode for faster debugging |
