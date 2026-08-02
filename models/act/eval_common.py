"""Shared real-robot evaluation helpers for ACT policy implementations."""

from __future__ import annotations

import time

import numpy as np
from lerobot.datasets import LeRobotDataset

from utils.config import JOINT_NAMES


def read_joint_vector(robot) -> np.ndarray:
    raw = robot.bus.sync_read("Present_Position")
    return np.array([raw[name] for name in JOINT_NAMES], dtype=np.float32)


def joint_positions(robot) -> dict[str, float]:
    vector = read_joint_vector(robot)
    return {f"{name}.pos": float(vector[i]) for i, name in enumerate(JOINT_NAMES)}


def clear_stuck_port(robot) -> bool:
    port_handler = getattr(getattr(robot, "bus", None), "port_handler", None)
    if port_handler is None:
        return False
    port_handler.is_using = False
    return True


def dataset_start_statistics(repo_id: str, revision: str = "main") -> dict[str, np.ndarray]:
    dataset = LeRobotDataset(repo_id=repo_id, revision=revision)
    episode_indices = np.asarray(dataset.hf_dataset["episode_index"])
    frame_indices = np.asarray(dataset.hf_dataset["frame_index"])
    states = []
    for episode in sorted(np.unique(episode_indices)):
        indices = np.where(episode_indices == episode)[0]
        first = indices[np.argmin(frame_indices[indices])]
        states.append(np.asarray(dataset.hf_dataset["observation.state"][int(first)], dtype=np.float32))
    if not states:
        raise ValueError(f"dataset {repo_id}@{revision} has no episodes")
    start_states = np.stack(states)
    if start_states.shape[1] != len(JOINT_NAMES):
        raise ValueError(
            f"dataset start state dim={start_states.shape[1]}, "
            f"but OMX follower has {len(JOINT_NAMES)} joints"
        )
    return {
        "mean": start_states.mean(axis=0),
        "median": np.median(start_states, axis=0),
        "std": start_states.std(axis=0),
        "min": start_states.min(axis=0),
        "max": start_states.max(axis=0),
    }


def start_state_violations(
    current: dict[str, float],
    target: dict[str, float],
    std: dict[str, float],
    *,
    tolerance_floor: float,
    max_z: float,
) -> dict[str, dict[str, float]]:
    if not np.isfinite(tolerance_floor) or tolerance_floor < 0:
        raise ValueError("tolerance_floor must be non-negative and finite.")
    if not np.isfinite(max_z) or max_z < 0:
        raise ValueError("max_z must be non-negative and finite.")
    violations = {}
    for key, target_value in target.items():
        allowed = max(tolerance_floor, max_z * std[key])
        error = abs(current[key] - target_value)
        if error > allowed:
            violations[key] = {"error": error, "allowed": allowed}
    return violations


def move_to_joint_state(
    robot,
    *,
    target: np.ndarray,
    timeout: float,
    ramp_seconds: float,
    hz: float,
    max_delta: float,
    tolerance: float | np.ndarray | None,
    log_every: float = 0,
) -> dict:
    if timeout <= 0 or hz <= 0:
        raise ValueError("timeout and hz must be positive.")
    if ramp_seconds < 0:
        raise ValueError("ramp_seconds must be non-negative.")
    if ramp_seconds == 0 and max_delta <= 0:
        raise ValueError("ramp_seconds=0 requires max_delta > 0.")

    target = np.asarray(target, dtype=np.float32)
    if target.shape != (len(JOINT_NAMES),):
        raise ValueError(f"target shape {target.shape} does not match {len(JOINT_NAMES)} joints")
    tolerance_array = None
    if tolerance is not None:
        tolerance_array = np.asarray(tolerance, dtype=np.float32)
        if tolerance_array.ndim == 0:
            tolerance_array = np.full_like(target, float(tolerance_array))
        if (
            tolerance_array.shape != target.shape
            or not np.all(np.isfinite(tolerance_array))
            or np.any(tolerance_array <= 0)
        ):
            raise ValueError(f"tolerance must be positive and finite with shape {target.shape}")

    initial_state = read_joint_vector(robot)
    command = initial_state.copy()
    total_steps = max(1, int(np.ceil(timeout * hz)))
    ramp_steps = max(1, int(np.ceil(ramp_seconds * hz)))
    if max_delta > 0:
        ramp_steps = max(ramp_steps, int(np.ceil(np.max(np.abs(target - initial_state)) / max_delta)))
        total_steps = max(total_steps, ramp_steps)
    next_log = time.perf_counter()

    for step in range(total_steps):
        loop_start = time.perf_counter()
        live_state = read_joint_vector(robot)
        error = target - live_state
        if tolerance_array is not None and np.all(np.abs(error) <= tolerance_array):
            return _move_result(live_state, error, tolerance_array, reached=True)

        alpha = min(1.0, (step + 1) / ramp_steps)
        desired = initial_state + alpha * (target - initial_state)
        if max_delta > 0:
            command = command + np.clip(desired - command, -max_delta, max_delta)
        else:
            command = desired
        sent = robot.send_action(
            {f"{name}.pos": float(command[i]) for i, name in enumerate(JOINT_NAMES)}
        )
        command = np.array([sent[f"{name}.pos"] for name in JOINT_NAMES], dtype=np.float32)

        now = time.perf_counter()
        if log_every > 0 and now >= next_log:
            worst = int(np.argmax(np.abs(error)))
            print(
                f"start_state progress step={step} max_error={np.max(np.abs(error)):.3f} "
                f"worst={JOINT_NAMES[worst]} state={live_state[worst]:.3f} "
                f"target={target[worst]:.3f} sent={command[worst]:.3f}",
                flush=True,
            )
            next_log = now + log_every
        time.sleep(max(0.0, 1.0 / hz - (time.perf_counter() - loop_start)))

    final_state = read_joint_vector(robot)
    final_error = target - final_state
    reached = tolerance_array is None or bool(np.all(np.abs(final_error) <= tolerance_array))
    return _move_result(final_state, final_error, tolerance_array, reached=reached)


def _move_result(
    state: np.ndarray,
    error: np.ndarray,
    tolerance: np.ndarray | None,
    *,
    reached: bool,
) -> dict:
    worst = int(np.argmax(np.abs(error)))
    support_ratio = None if tolerance is None else float(np.max(np.abs(error) / tolerance))
    return {
        "state": state,
        "max_abs_error": float(np.max(np.abs(error))),
        "max_support_ratio": support_ratio,
        "worst_joint": JOINT_NAMES[worst],
        "reached": reached,
    }
