PYTHON_BIN := $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

EXPERIMENTS := baseline baseline_no_vae canny canny_no_vae grayscale sobel blur_canny downsample_84

FOLLOWER_PORT ?= /dev/tty.usbmodem5AAF2630541
LEADER_PORT  ?= /dev/tty.usbmodem5AAF2634711
FOLLOWER_ID  ?= blue_follower
LEADER_ID    ?= blue_leader_2

DATASET_ID      ?= W1ndrunn3rr/pick_and_lift_v3
POLICY_ID       ?= W1ndrunn3rr/act_pick_and_lift_v3_baseline
DATASET_ROOT    ?= workspace/pick_and_lift_v3

EXP          ?= baseline
MODEL        ?= $(POLICY_ID)
FILTER       ?= none
ROBOT_TYPE   ?= so101_follower
ROBOT_PORT   ?= $(FOLLOWER_PORT)
ROBOT_ID     ?= $(FOLLOWER_ID)

CAMERA_INDEX             ?= 0
CAMERA_WIDTH             ?= 1280
CAMERA_HEIGHT            ?= 720
CAMERA_FPS               ?= 30
CAMERA_BACKEND           ?= 1200
CAMERA_WARMUP_S          ?= 3
CAMERA_CONNECTION_ATTEMPTS     ?= 3
CAMERA_CONNECTION_RETRY_DELAY_S ?= 1
CAMERA_FOURCC            ?=

EVAL_DATASET_REPO ?= W1ndrunn3rr/act_pick_and_lift_v3_baseline
EVAL_EPISODES     ?= 3
EVAL_TRIALS       ?= 10
EVAL_OVERWRITE    ?= true
EVAL_DISPLAY_DATA ?= false

EPISODE_TIME      ?= 30
RESET_TIME        ?= 10
TASK              ?= pick_and_lift

HOME_ACTION          ?= {"shoulder_pan":-2.5934065934065935,"shoulder_lift":-103.12087912087912,"elbow_flex":96.92307692307692,"wrist_flex":72.08791208791209,"wrist_roll":11.648351648351648,"gripper":0.9688581314878892}
HOME_RETURN_TIME_S   ?= 3
HOME_HOLD_TIME_S     ?= 0.5

RECORD_NUM_EPISODES   ?= 50
RECORD_EPISODE_TIME   ?= $(EPISODE_TIME)
RECORD_RESET_TIME     ?= $(RESET_TIME)
RECORD_TASK           ?= $(TASK)
RECORD_DATASET_ID     ?= $(DATASET_ID)
RECORD_DATASET_ROOT   ?= $(DATASET_ROOT)
RECORD_RESUME         ?= false
RECORD_PUSH_TO_HUB    ?= true
RECORD_DISPLAY_DATA   ?= false
# Camera front: 1920x1080 @ 30fps (external USB, index 0)
RECORD_CAM0_INDEX     ?= 0
RECORD_CAM0_WIDTH     ?= 1920
RECORD_CAM0_HEIGHT    ?= 1080
RECORD_CAM0_FPS       ?= 30
RECORD_CAM0_BACKEND   ?= 1200
RECORD_CAM0_WARMUP_S  ?= 15
# Camera side: 1920x1080 @ 30fps (external USB, index 1)
RECORD_CAM1_INDEX     ?= 1
RECORD_CAM1_WIDTH     ?= 1920
RECORD_CAM1_HEIGHT    ?= 1080
RECORD_CAM1_FPS       ?= 30
RECORD_CAM1_BACKEND   ?= 1200
RECORD_CAM1_WARMUP_S  ?= 15

GRADCAM_POLICY_PATH  ?= $(MODEL)
GRADCAM_DATASET_ID   ?= $(DATASET_ID)
GRADCAM_DATASET_ROOT ?= $(DATASET_ROOT)
GRADCAM_EPISODE      ?= 0
GRADCAM_OUTPUT_DIR   ?= outputs/gradcam/episode_$(GRADCAM_EPISODE)
GRADCAM_IMAGE_KEY    ?=
GRADCAM_TARGET_STEP  ?= 0
GRADCAM_TARGET_DIM   ?= 0
GRADCAM_ALPHA        ?= 0.45
GRADCAM_FPS          ?=
GRADCAM_MAX_FRAMES   ?=

EVAL_MODEL_FILTERS := \
	W1ndrunn3rr/act_pick_and_lift_v3_downsample_84:downsample_84 \
	W1ndrunn3rr/act_pick_and_lift_v3_blur_canny:blur_canny \
	W1ndrunn3rr/act_pick_and_lift_v3_grayscale:grayscale \
	W1ndrunn3rr/act_pick_and_lift_v3_canny_no_vae:canny_no_vae \
	W1ndrunn3rr/act_pick_and_lift_v3_canny:canny \
	W1ndrunn3rr/act_pick_and_lift_v3_baseline_no_vae:baseline_no_vae \
	W1ndrunn3rr/act_pick_and_lift_v2_baseline:baseline

.DEFAULT_GOAL := help

.PHONY: help record train train-all eval eval-direct eval-all gradcam eval-gradcam check-eval-deps

help:
	@echo ""
	@echo "Available commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*##"}; {printf "  \033[36mmake %-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Default values:"
	@echo "  MODEL=$(MODEL)"
	@echo "  ROBOT_PORT=$(ROBOT_PORT)"
	@echo "  ROBOT_ID=$(ROBOT_ID)"
	@echo ""

record: ## Record a teleoperation dataset  [RECORD_DATASET_ID=... RECORD_NUM_EPISODES=50 RECORD_PUSH_TO_HUB=false]
	lerobot-record \
		--robot.type=$(ROBOT_TYPE) \
		--robot.port=$(FOLLOWER_PORT) \
		--robot.id=$(FOLLOWER_ID) \
		--teleop.type=so101_leader \
		--teleop.port=$(LEADER_PORT) \
		--teleop.id=$(LEADER_ID) \
		--robot.cameras='{"front": {"type": "opencv", "index_or_path": $(RECORD_CAM0_INDEX), "width": $(RECORD_CAM0_WIDTH), "height": $(RECORD_CAM0_HEIGHT), "fps": $(RECORD_CAM0_FPS), "backend": $(RECORD_CAM0_BACKEND), "warmup_s": $(RECORD_CAM0_WARMUP_S)}, "side": {"type": "opencv", "index_or_path": $(RECORD_CAM1_INDEX), "width": $(RECORD_CAM1_WIDTH), "height": $(RECORD_CAM1_HEIGHT), "fps": $(RECORD_CAM1_FPS), "backend": $(RECORD_CAM1_BACKEND), "warmup_s": $(RECORD_CAM1_WARMUP_S)}}' \
		--display_data=$(RECORD_DISPLAY_DATA) \
		--dataset.repo_id=$(RECORD_DATASET_ID) \
		--dataset.root=$(RECORD_DATASET_ROOT) \
		--dataset.num_episodes=$(RECORD_NUM_EPISODES) \
		--dataset.single_task='$(RECORD_TASK)' \
		--dataset.push_to_hub=$(RECORD_PUSH_TO_HUB) \
		--resume=$(RECORD_RESUME) 


train: ## Train a single model  [EXP=baseline]
	accelerate launch \
		--num_processes=2 \
		--num_machines=1 \
		--mixed_precision=bf16 \
		--dynamo_backend=no \
		--multi_gpu \
		-m src.scripts.train $(EXP)

train-all: ## Train models for all experiments from EXPERIMENTS
	@for exp in $(EXPERIMENTS); do \
		echo "==> $$exp"; \
		accelerate launch --num_processes=1 -m src.scripts.train $$exp || exit $$?; \
	done

train-wcss: ## Train models for all experiments from EXPERIMENTS via SLURM array job
	@echo "==> Submitting $(words $(EXPERIMENTS)) experiments to SLURM array job..."
	@echo "#!/bin/bash" > .slurm_train_all.sh
	@echo "#SBATCH -N 1" >> .slurm_train_all.sh
	@echo "#SBATCH -c 8" >> .slurm_train_all.sh
	@echo "#SBATCH --mem=32gb" >> .slurm_train_all.sh
	@echo "#SBATCH --gres=gpu:1" >> .slurm_train_all.sh
	@echo "#SBATCH --time=12:00:00" >> .slurm_train_all.sh
	@echo "#SBATCH --job-name=train_all" >> .slurm_train_all.sh
	@echo "#SBATCH --output=wyniki_train_%A_%a.txt" >> .slurm_train_all.sh
	@echo "#SBATCH --array=1-$(words $(EXPERIMENTS))" >> .slurm_train_all.sh
	@echo "" >> .slurm_train_all.sh
	@echo "source /usr/local/sbin/modules.sh" >> .slurm_train_all.sh
	@echo "uv sync" >> .slurm_train_all.sh
	@echo "source .venv/bin/activate" >> .slurm_train_all.sh
	@echo "EXPERIMENTS=($(EXPERIMENTS))" >> .slurm_train_all.sh
	@echo 'EXP=$${EXPERIMENTS[$$SLURM_ARRAY_TASK_ID-1]}' >> .slurm_train_all.sh
	@echo 'echo "==> Running experiment: $$EXP"' >> .slurm_train_all.sh
	@echo 'accelerate launch --num_processes=1 -m src.scripts.train $$EXP' >> .slurm_train_all.sh
	@sbatch .slurm_train_all.sh

eval-direct: ## Run lerobot-record directly with 2 cameras + policy, no patching  [MODEL=... EVAL_EPISODES=3 EPISODE_TIME=2000]
	lerobot-record \
		--robot.type=$(ROBOT_TYPE) \
		--robot.port=$(ROBOT_PORT) \
		--robot.id=$(ROBOT_ID) \
		--robot.cameras='{"front": {"type": "opencv", "index_or_path": $(CAMERA_INDEX), "width": $(CAMERA_WIDTH), "height": $(CAMERA_HEIGHT), "fps": $(CAMERA_FPS), "backend": $(CAMERA_BACKEND), "warmup_s": $(CAMERA_WARMUP_S)}, "side": {"type": "opencv", "index_or_path": $(CAMERA_SIDE_INDEX), "width": $(CAMERA_SIDE_WIDTH), "height": $(CAMERA_SIDE_HEIGHT), "fps": $(CAMERA_SIDE_FPS), "backend": $(CAMERA_SIDE_BACKEND), "warmup_s": $(CAMERA_SIDE_WARMUP_S)}}' \
		--policy.path=$(MODEL) \
		--dataset.repo_id=local/eval_direct \
		--dataset.root=/tmp/lerobot_eval_direct \
		--dataset.num_episodes=$(EVAL_EPISODES) \
		--dataset.episode_time_s=$(EPISODE_TIME) \
		--dataset.reset_time_s=$(RESET_TIME) \
		--dataset.single_task=$(TASK) \
		--dataset.push_to_hub=false
	@rm -rf /tmp/lerobot_eval_direct

check-eval-deps: ## Check whether evaluation dependencies are installed
	@$(PYTHON_BIN) -c "import scservo_sdk" >/dev/null 2>&1 || \
		(echo "Missing dependency: scservo_sdk"; \
		echo "Reinstall the project dependencies:"; \
		echo "  $(PYTHON_BIN) -m pip install -e ."; \
		exit 1)

eval: check-eval-deps ## Evaluate a single model  [MODEL=... FILTER=canny]
	@test -n "$(MODEL)" || (echo "MODEL is required, e.g. make eval MODEL=user/policy FILTER=canny" && exit 1)
	@if [ "$(EVAL_OVERWRITE)" = "true" ]; then \
		$(PYTHON_BIN) -m src.scripts.clean_lerobot_cache $(EVAL_DATASET_REPO); \
	fi
	HOME_ACTION='$(HOME_ACTION)' \
	HOME_RETURN_TIME_S=$(HOME_RETURN_TIME_S) \
	HOME_HOLD_TIME_S=$(HOME_HOLD_TIME_S) \
	$(PYTHON_BIN) -m src.scripts.eval \
		--robot.type=$(ROBOT_TYPE) \
		--robot.port=$(ROBOT_PORT) \
		--robot.id=$(ROBOT_ID) \
		--robot.cameras='{front: {type: filtered_opencv, index_or_path: $(CAMERA_INDEX), width: $(CAMERA_WIDTH), height: $(CAMERA_HEIGHT), fps: $(CAMERA_FPS), backend: $(CAMERA_BACKEND), warmup_s: $(CAMERA_WARMUP_S), connection_attempts: $(CAMERA_CONNECTION_ATTEMPTS), connection_retry_delay_s: $(CAMERA_CONNECTION_RETRY_DELAY_S), fourcc: $(if $(strip $(CAMERA_FOURCC)),$(CAMERA_FOURCC),null), filter_name: $(FILTER)}}' \
		--display_data=$(EVAL_DISPLAY_DATA) \
		--dataset.repo_id=$(EVAL_DATASET_REPO) \
		--dataset.num_episodes=$(EVAL_EPISODES) \
		--dataset.episode_time_s=$(EPISODE_TIME) \
		--dataset.reset_time_s=$(RESET_TIME) \
		--dataset.single_task=$(TASK) \
		--dataset.push_to_hub=false \
		--policy.path=$(MODEL)

eval-all: check-eval-deps ## Evaluate all models from EVAL_MODEL_FILTERS
	@EVAL_TRIALS='$(EVAL_TRIALS)' \
	EVAL_MODEL_FILTERS='$(EVAL_MODEL_FILTERS)' \
	EVAL_DATASET_REPO='$(EVAL_DATASET_REPO)' \
	$(PYTHON_BIN) -m src.scripts.eval_all

gradcam: ## Render offline ACT Grad-CAM video  [GRADCAM_POLICY_PATH=... GRADCAM_TARGET_DIM=0]
	$(PYTHON_BIN) -m src.scripts.gradcam_act \
		--policy-path=$(GRADCAM_POLICY_PATH) \
		--dataset-id=$(GRADCAM_DATASET_ID) \
		--dataset-root=$(GRADCAM_DATASET_ROOT) \
		--episode-index=$(GRADCAM_EPISODE) \
		--output-dir=$(GRADCAM_OUTPUT_DIR) \
		--target-step=$(GRADCAM_TARGET_STEP) \
		--target-dim=$(GRADCAM_TARGET_DIM) \
		--alpha=$(GRADCAM_ALPHA) \
		$(if $(strip $(GRADCAM_IMAGE_KEY)),--image-key=$(GRADCAM_IMAGE_KEY)) \
		$(if $(strip $(GRADCAM_FPS)),--fps=$(GRADCAM_FPS)) \
		$(if $(strip $(GRADCAM_MAX_FRAMES)),--max-frames=$(GRADCAM_MAX_FRAMES))

eval-gradcam: check-eval-deps ## Run live eval, record dataset, then render Grad-CAM  [MODEL=... FILTER=... GRADCAM_TARGET_DIM=0]
	@$(MAKE) eval
	@$(MAKE) gradcam \
		GRADCAM_POLICY_PATH='$(MODEL)' \
		GRADCAM_DATASET_ID='$(EVAL_DATASET_REPO)' \
		GRADCAM_DATASET_ROOT= \
		GRADCAM_EPISODE=0

live-gradcam: ## Example command to run live eval and render Grad-CAM for the first episode and first action dimension
	make eval-gradcam \
	MODEL=W1ndrunn3rr/act_pick_and_lift_v2_baseline \
	FILTER=baseline \
	EVAL_DATASET_REPO=W1ndrunn3rr/eval_pick_and_lift_v2_baseline \
	GRADCAM_TARGET_DIM=0
