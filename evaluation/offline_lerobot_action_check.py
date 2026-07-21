"""Offline first-action sanity check for exported LeRobot policies.

Runs a pretrained_model policy on a recorded dataset frame and compares the
first predicted action against the recorded action and current state. Use this
before live eval to catch policies that command far-away targets at t=0.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from lerobot.configs import PreTrainedConfig
from lerobot.datasets import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import models.act.lerobot_policy  # noqa: F401 -- register custom_act before config parsing
from utils.config import JOINT_NAMES, RECORD_DATASET_REPO_ID


def load_sample(dataset: LeRobotDataset, episode: int, frame_index: int):
    for idx in range(len(dataset)):
        row = dataset.hf_dataset[idx]
        if int(row["episode_index"]) == episode and int(row["frame_index"]) == frame_index:
            return idx, dataset[idx]
    raise ValueError(f"could not find episode={episode} frame_index={frame_index}")


def vector_by_joint(values) -> dict[str, float]:
    array = values.detach().cpu().numpy() if hasattr(values, "detach") else np.asarray(values)
    return {name: float(array[i]) for i, name in enumerate(JOINT_NAMES)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Check LeRobot policy first action on a recorded dataset frame.")
    parser.add_argument("--policy-path", required=True, help="Path to exported pretrained_model directory.")
    parser.add_argument("--dataset", default=RECORD_DATASET_REPO_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--policy-n-action-steps", type=int, default=None)
    parser.add_argument("--policy-temporal-ensemble-coeff", type=float, default=None)
    parser.add_argument(
        "--action-space",
        choices=["absolute", "delta"],
        default="absolute",
        help="If 'delta', the policy/dataset action is an offset from state; reconstruct "
        "absolute = state + action for both prediction and recorded action so the reported "
        "errors stay in absolute joint units (comparable across absolute and delta runs).",
    )
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_name)

    policy_path = Path(args.policy_path)
    policy_config = PreTrainedConfig.from_pretrained(policy_path)
    policy_config.device = device.type
    if args.policy_n_action_steps is not None:
        if not hasattr(policy_config, "n_action_steps"):
            raise ValueError(f"policy type {policy_config.type!r} does not support n_action_steps")
        policy_config.n_action_steps = args.policy_n_action_steps
    if args.policy_temporal_ensemble_coeff is not None:
        if not hasattr(policy_config, "temporal_ensemble_coeff"):
            raise ValueError(f"policy type {policy_config.type!r} does not support temporal_ensemble_coeff")
        policy_config.temporal_ensemble_coeff = args.policy_temporal_ensemble_coeff

    policy = get_policy_class(policy_config.type).from_pretrained(policy_path, config=policy_config)
    policy.to(device)
    policy.eval()
    policy.reset()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(policy_path),
        preprocessor_overrides={"device_processor": {"device": device.type}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )

    dataset = LeRobotDataset(args.dataset, revision=args.revision)
    dataset_index, sample = load_sample(dataset, args.episode, args.frame_index)
    obs = {"observation.state": sample["observation.state"]}
    for key in policy.config.input_features:
        if key.startswith("observation.images.") and key in sample:
            obs[key] = sample[key]

    with torch.inference_mode():
        action = postprocessor(policy.select_action(preprocessor(obs))).squeeze(0).cpu()

    state = sample["observation.state"].cpu()
    recorded_action = sample["action"].cpu()
    if args.action_space == "delta":
        # Dataset/policy actions are offsets from state; reconstruct absolute targets
        # so the reported errors are in absolute joint units and comparable to
        # absolute-action runs.
        action = action + state
        recorded_action = recorded_action + state
    action_minus_state = action - state
    action_minus_recorded = action - recorded_action

    result = {
        "policy_path": str(policy_path),
        "policy_type": policy.config.type,
        "dataset": args.dataset,
        "revision": args.revision,
        "action_space": args.action_space,
        "dataset_index": int(dataset_index),
        "episode": args.episode,
        "frame_index": args.frame_index,
        "device": str(device),
        "state": vector_by_joint(state),
        "recorded_action": vector_by_joint(recorded_action),
        "predicted_action": vector_by_joint(action),
        "predicted_minus_state": vector_by_joint(action_minus_state),
        "predicted_minus_recorded": vector_by_joint(action_minus_recorded),
        "max_abs_predicted_minus_state": float(action_minus_state.abs().max()),
        "max_abs_predicted_minus_recorded": float(action_minus_recorded.abs().max()),
    }

    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
