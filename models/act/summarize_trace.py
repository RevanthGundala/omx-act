"""Summarize JSONL traces saved by models.act.eval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_trace(path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    records = []
    with path.open() as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"{path} has no trace records")

    joint_names = records[0]["joint_names"]
    arrays = {
        key: np.asarray([record[key] for record in records], dtype=np.float32)
        for key in ("qpos", "qpos_normalized", "policy_action", "policy_action_normalized", "sent_action")
    }
    return joint_names, arrays


def summarize_trace(path: Path, saturation_threshold: float) -> None:
    joint_names, arrays = load_trace(path)
    qpos = arrays["qpos"]
    action = arrays["policy_action"]
    action_norm = arrays["policy_action_normalized"]
    sent_action = arrays["sent_action"]

    policy_delta = action - qpos
    sent_delta = sent_action - action
    step_delta = np.diff(action, axis=0) if action.shape[0] > 1 else np.zeros_like(action)

    print(f"\n{path}")
    print(f"steps={qpos.shape[0]} joints={len(joint_names)}")
    print(
        "joint                              "
        "state_rng action_rng mean|a-q| max|a_norm| sat% mean|sent-a| mean|da|"
    )
    for idx, name in enumerate(joint_names):
        state_range = float(qpos[:, idx].max() - qpos[:, idx].min())
        action_range = float(action[:, idx].max() - action[:, idx].min())
        mean_abs_policy_delta = float(np.mean(np.abs(policy_delta[:, idx])))
        max_abs_norm = float(np.max(np.abs(action_norm[:, idx])))
        saturation_pct = float(np.mean(np.abs(action_norm[:, idx]) >= saturation_threshold) * 100.0)
        mean_abs_sent_delta = float(np.mean(np.abs(sent_delta[:, idx])))
        mean_abs_step_delta = float(np.mean(np.abs(step_delta[:, idx]))) if step_delta.size else 0.0
        print(
            f"{name:<34} "
            f"{state_range:8.3f} {action_range:10.3f} {mean_abs_policy_delta:9.3f} "
            f"{max_abs_norm:11.3f} {saturation_pct:5.1f} {mean_abs_sent_delta:12.4f} "
            f"{mean_abs_step_delta:8.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--saturation-threshold", type=float, default=3.0)
    args = parser.parse_args()

    for trace in args.traces:
        summarize_trace(trace, args.saturation_threshold)


if __name__ == "__main__":
    main()
