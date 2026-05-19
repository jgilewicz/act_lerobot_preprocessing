import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import lerobot.scripts.lerobot_record as lerobot_record_module
from lerobot.robots.so_follower.so_follower import SOFollower

from src.scripts import lerobot_record_filtered as lerobot_record_filtered_module

logger = logging.getLogger(__name__)

DEFAULT_HOME_ACTION: dict[str, float] = {
    "shoulder_pan": -2.5934065934065935,
    "shoulder_lift": -103.12087912087912,
    "elbow_flex": 96.92307692307692,
    "wrist_flex": 72.08791208791209,
    "wrist_roll": 11.648351648351648,
    "gripper": 0.9688581314878892,
}


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    return float(value)


def _load_home_action() -> dict[str, float]:
    raw_action = os.environ.get("HOME_ACTION")
    if not raw_action:
        return DEFAULT_HOME_ACTION.copy()

    parsed = json.loads(raw_action)
    if not isinstance(parsed, dict):
        raise ValueError("HOME_ACTION must be a JSON object.")

    action: dict[str, float] = {}
    for raw_key, raw_value in parsed.items():
        if not isinstance(raw_key, str):
            raise ValueError("HOME_ACTION keys must be motor names.")
        if not isinstance(raw_value, int | float):
            raise ValueError(f"HOME_ACTION value for '{raw_key}' must be numeric.")

        motor = raw_key.removesuffix(".pos")
        action[motor] = float(raw_value)

    return action


def _move_so_follower_home(robot: Any) -> None:
    if not _env_flag("HOME_ON_EXIT", True):
        return
    if not isinstance(robot, SOFollower) or not robot.is_connected:
        return

    home_action = _load_home_action()
    unknown_motors = set(home_action) - set(robot.bus.motors)
    if unknown_motors:
        raise ValueError(
            f"HOME_ACTION contains unknown motors: {sorted(unknown_motors)}. "
            f"Available: {sorted(robot.bus.motors)}"
        )

    current_position = robot.bus.sync_read("Present_Position")
    target_position = {
        motor: home_action.get(motor, current_position[motor])
        for motor in robot.bus.motors
    }

    return_time_s = max(0.0, _env_float("HOME_RETURN_TIME_S", 3.0))
    control_hz = max(1.0, _env_float("HOME_CONTROL_HZ", 30.0))
    steps = max(1, int(return_time_s * control_hz))

    logger.info("Moving %s to home configuration.", robot)
    for step in range(1, steps + 1):
        alpha = step / steps
        command = {
            motor: current_position[motor]
            + (target_position[motor] - current_position[motor]) * alpha
            for motor in target_position
        }
        robot.bus.sync_write("Goal_Position", command)
        time.sleep(1.0 / control_hz)

    robot.bus.sync_write("Goal_Position", target_position)
    hold_time_s = max(0.0, _env_float("HOME_HOLD_TIME_S", 0.5))
    if hold_time_s:
        time.sleep(hold_time_s)


_original_record_loop = lerobot_record_module.record_loop


def _record_loop_with_home_return(*args: Any, **kwargs: Any) -> Any:
    robot = kwargs.get("robot")
    if robot is None and args:
        robot = args[0]

    try:
        return _original_record_loop(*args, **kwargs)
    finally:
        try:
            _move_so_follower_home(robot)
        except Exception:
            logger.exception("Failed to move robot to home configuration after record loop.")


lerobot_record_module.record_loop = _record_loop_with_home_return

_original_disconnect = SOFollower.disconnect


def _disconnect_with_home_return(self: SOFollower) -> None:
    try:
        _move_so_follower_home(self)
    except Exception:
        logger.exception("Failed to move robot to home configuration before disconnect.")
    _original_disconnect(self)


SOFollower.disconnect = _disconnect_with_home_return

main = lerobot_record_filtered_module.main


if __name__ == "__main__":
    raise SystemExit(main())
