"""Compare ACT policies on task-relevant episode-start diagnostics.

Unlike aggregate normalized joint L1, this reports end-effector geometry,
gripper error, visual conditioning, and agreement between adjacent replans.
All policies receive exactly the same recorded observations.

Example:
    uv run python evaluation/compare_act_policies.py \
      --dataset RevanthGundala/014-place-sponge-in-bowl-diverse-50 \
      --policy "official=outputs/.../pretrained_model" \
      --policy "custom=outputs/.../pretrained_model" \
      --output outputs/dataset_audits/act_policy_comparison.json
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from lerobot.configs import PreTrainedConfig
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import models.act.lerobot_policy  # noqa: F401 -- register custom_act
from utils.config import JOINT_NAMES

OMX_DEGREES_PER_NORMALIZED_UNIT = 360.0 / 200.0


def _translation(x: float, y: float, z: float) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = (x, y, z)
    return transform


def _rotation_x(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.asarray(
        [
            [1, 0, 0, 0],
            [0, cosine, -sine, 0],
            [0, sine, cosine, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )


def _rotation_y(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.asarray(
        [
            [cosine, 0, sine, 0],
            [0, 1, 0, 0],
            [-sine, 0, cosine, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )


def _rotation_z(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.asarray(
        [
            [cosine, -sine, 0, 0],
            [sine, cosine, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )


def omx_end_effector_transforms(actions_degrees: np.ndarray) -> np.ndarray:
    """Return OMX end-effector transforms for (..., action_dim) joint targets.

    The first four joints use the official OpenMANIPULATOR-X URDF chain. The
    fifth OMX wrist-roll joint rotates around the tool axis and therefore
    changes orientation but not end-effector position.
    """
    actions = np.asarray(actions_degrees, dtype=np.float64)
    if actions.shape[-1] < 5:
        raise ValueError(f"Expected at least 5 arm joints, got shape {actions.shape}")

    radians = np.deg2rad(actions[..., :5])
    transforms = np.empty(actions.shape[:-1] + (4, 4), dtype=np.float64)
    for index in np.ndindex(actions.shape[:-1]):
        joint = radians[index]
        transforms[index] = (
            _translation(0.012, 0, 0)
            @ _rotation_z(joint[0])
            @ _translation(0, 0, 0.0595)
            @ _rotation_y(joint[1])
            @ _translation(0.024, 0, 0.128)
            @ _rotation_y(joint[2])
            @ _translation(0.124, 0, 0)
            @ _rotation_y(joint[3])
            @ _translation(0.126, 0, 0)
            @ _rotation_x(joint[4])
        )
    return transforms


def pose_errors(
    predicted_actions: np.ndarray,
    target_actions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return pose errors for stock OMX normalized joint values."""
    predicted = omx_end_effector_transforms(
        np.asarray(predicted_actions) * OMX_DEGREES_PER_NORMALIZED_UNIT
    )
    target = omx_end_effector_transforms(
        np.asarray(target_actions) * OMX_DEGREES_PER_NORMALIZED_UNIT
    )
    position_error_mm = np.linalg.norm(
        predicted[..., :3, 3] - target[..., :3, 3],
        axis=-1,
    ) * 1000
    relative_rotation = np.einsum(
        "...ji,...jk->...ik",
        target[..., :3, :3],
        predicted[..., :3, :3],
    )
    cosine = (np.trace(relative_rotation, axis1=-2, axis2=-1) - 1) / 2
    orientation_error_degrees = np.rad2deg(np.arccos(np.clip(cosine, -1, 1)))
    return position_error_mm, orientation_error_degrees


def _parse_policy(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--policy must be LABEL=PATH")
    label, raw_path = value.split("=", 1)
    path = Path(raw_path).expanduser()
    if not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("--policy must contain a non-empty LABEL and PATH")
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"Policy directory does not exist: {path}")
    return label.strip(), path


def _resolve_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def episode_start_frame_pairs(
    dataset: LeRobotDataset,
    max_episodes: int | None = None,
) -> list[tuple[int, int]]:
    indices_by_frame = {}
    episode_starts = []
    for dataset_index in range(len(dataset)):
        row = dataset.hf_dataset[dataset_index]
        episode_index = int(row["episode_index"])
        frame_index = int(row["frame_index"])
        key = (episode_index, frame_index)
        if key in indices_by_frame:
            raise ValueError(f"Dataset contains duplicate episode/frame key {key}")
        indices_by_frame[key] = dataset_index
        if frame_index == 0:
            episode_starts.append((episode_index, dataset_index))

    pairs = [
        (start_index, indices_by_frame[(episode_index, 1)])
        for episode_index, start_index in episode_starts
        if (episode_index, 1) in indices_by_frame
    ]
    if max_episodes is not None:
        if max_episodes <= 0:
            raise ValueError("--max-episodes must be positive")
        if len(pairs) > max_episodes:
            selected = np.linspace(0, len(pairs) - 1, max_episodes).round().astype(int)
            pairs = [pairs[index] for index in selected]
    if not pairs:
        raise ValueError("Dataset contains no episode starts with an in-episode frame 1")
    return pairs


def _predict_chunk(policy, preprocessor, postprocessor, image_keys, state_sample, image_sample):
    observation = {"observation.state": state_sample["observation.state"]}
    observation.update({key: image_sample[key] for key in image_keys})
    with torch.inference_mode():
        normalized = policy.predict_action_chunk(preprocessor(observation))
        return postprocessor(normalized).squeeze(0).cpu().numpy()


def _masked_mean(values: np.ndarray, valid: np.ndarray) -> float:
    return float(np.asarray(values)[valid].mean())


def _horizon_summary(values: np.ndarray, valid: np.ndarray) -> list[float | None]:
    summary = []
    for horizon in range(values.shape[1]):
        horizon_valid = valid[:, horizon]
        summary.append(
            float(values[horizon_valid, horizon].mean())
            if horizon_valid.any()
            else None
        )
    return summary


def _scene_tracking(
    predicted: np.ndarray,
    target: np.ndarray,
) -> dict[str, dict[str, float | None]]:
    result = {}
    for joint_index, joint_name in enumerate(JOINT_NAMES):
        prediction = predicted[:, -1, joint_index]
        truth = target[:, -1, joint_index]
        if prediction.std() > 1e-8 and truth.std() > 1e-8:
            correlation = float(np.corrcoef(prediction, truth)[0, 1])
        else:
            correlation = None
        result[joint_name] = {
            "target_std_normalized_units": float(truth.std()),
            "prediction_std_normalized_units": float(prediction.std()),
            "correlation": correlation,
            "mae_normalized_units": float(np.abs(prediction - truth).mean()),
        }
    return result


def evaluate_policy(
    label: str,
    path: Path,
    device: torch.device,
    starts: list[dict[str, Any]],
    next_frames: list[dict[str, Any]],
) -> dict[str, Any]:
    config = PreTrainedConfig.from_pretrained(path)
    config.device = device.type
    policy = get_policy_class(config.type).from_pretrained(path, config=config)
    policy.to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(path),
        preprocessor_overrides={"device_processor": {"device": device.type}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    image_keys = tuple(
        key
        for key in policy.config.input_features
        if key.startswith("observation.images.")
    )

    predictions = []
    image_swaps = []
    state_swaps = []
    next_predictions = []
    for sample_index, sample in enumerate(starts):
        other = starts[(sample_index + 1) % len(starts)]
        predictions.append(
            _predict_chunk(
                policy,
                preprocessor,
                postprocessor,
                image_keys,
                sample,
                sample,
            )
        )
        image_swaps.append(
            _predict_chunk(
                policy,
                preprocessor,
                postprocessor,
                image_keys,
                sample,
                other,
            )
        )
        state_swaps.append(
            _predict_chunk(
                policy,
                preprocessor,
                postprocessor,
                image_keys,
                other,
                sample,
            )
        )
        next_predictions.append(
            _predict_chunk(
                policy,
                preprocessor,
                postprocessor,
                image_keys,
                next_frames[sample_index],
                next_frames[sample_index],
            )
        )

    predicted = np.stack(predictions)
    image_swapped = np.stack(image_swaps)
    state_swapped = np.stack(state_swaps)
    next_predicted = np.stack(next_predictions)
    target = np.stack([sample["action"].cpu().numpy() for sample in starts])
    valid = np.stack([~sample["action_is_pad"].cpu().numpy() for sample in starts])
    next_valid = np.stack(
        [~sample["action_is_pad"].cpu().numpy() for sample in next_frames]
    )

    joint_error = np.abs(predicted - target)
    joint_valid = np.repeat(valid[:, :, None], joint_error.shape[-1], axis=2)
    position_error_mm, orientation_error_degrees = pose_errors(predicted, target)
    image_delta = np.linalg.norm(image_swapped - predicted, axis=2)
    state_delta = np.linalg.norm(state_swapped - predicted, axis=2)
    natural_delta = np.linalg.norm(np.roll(predicted, -1, axis=0) - predicted, axis=2)
    overlap_valid = valid[:, 1:] & next_valid[:, :-1]
    overlap_error = np.abs(predicted[:, 1:] - next_predicted[:, :-1])
    overlap_joint_valid = np.repeat(
        overlap_valid[:, :, None],
        overlap_error.shape[-1],
        axis=2,
    )

    result = {
        "label": label,
        "path": str(path),
        "policy_type": config.type,
        "joint_mae_normalized_units": _masked_mean(joint_error, joint_valid),
        "joint_mae_by_joint_normalized_units": {
            joint_name: _masked_mean(joint_error[..., index], valid)
            for index, joint_name in enumerate(JOINT_NAMES)
        },
        "joint_mae_by_horizon_normalized_units": _horizon_summary(
            joint_error.mean(axis=2),
            valid,
        ),
        "end_effector_position_mae_mm": _masked_mean(position_error_mm, valid),
        "end_effector_position_by_horizon_mm": _horizon_summary(
            position_error_mm,
            valid,
        ),
        "end_effector_orientation_mae_degrees": _masked_mean(
            orientation_error_degrees,
            valid,
        ),
        "end_effector_orientation_by_horizon_degrees": _horizon_summary(
            orientation_error_degrees,
            valid,
        ),
        "gripper_mae_normalized_units": _masked_mean(joint_error[..., -1], valid),
        "conditioning": {
            "image_swap_l2_per_step_normalized_units": _masked_mean(image_delta, valid),
            "state_swap_l2_per_step_normalized_units": _masked_mean(state_delta, valid),
            "natural_prediction_l2_per_step_normalized_units": _masked_mean(
                natural_delta,
                valid,
            ),
        },
        "replan_consistency": {
            "overlap_mae_normalized_units": _masked_mean(
                overlap_error,
                overlap_joint_valid,
            ),
            "aligned_first_target_mae_normalized_units": float(
                overlap_error[:, 0].mean()
            ),
            "overlap_mae_by_joint_normalized_units": {
                joint_name: _masked_mean(overlap_error[..., index], overlap_valid)
                for index, joint_name in enumerate(JOINT_NAMES)
            },
        },
        "last_horizon_scene_tracking": _scene_tracking(predicted, target),
    }

    del policy
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    return result


def _print_summary(results: list[dict[str, Any]]) -> None:
    print(
        f"{'policy':<24} {'joint MAE':>10} {'EE pos':>10} "
        f"{'EE orient':>10} {'gripper':>10} {'image/natural':>14} {'replan':>10}"
    )
    for result in results:
        conditioning = result["conditioning"]
        natural = conditioning["natural_prediction_l2_per_step_normalized_units"]
        image_fraction = conditioning["image_swap_l2_per_step_normalized_units"] / max(
            natural,
            1e-12,
        )
        print(
            f"{result['label']:<24} "
            f"{result['joint_mae_normalized_units']:>9.3f}u "
            f"{result['end_effector_position_mae_mm']:>8.2f}mm "
            f"{result['end_effector_orientation_mae_degrees']:>9.2f}° "
            f"{result['gripper_mae_normalized_units']:>9.3f}u "
            f"{image_fraction:>13.1%} "
            f"{result['replan_consistency']['overlap_mae_normalized_units']:>9.3f}u"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--policy",
        action="append",
        type=_parse_policy,
        required=True,
        metavar="LABEL=PATH",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    configs = [PreTrainedConfig.from_pretrained(path) for _, path in args.policy]
    chunk_sizes = {config.chunk_size for config in configs}
    if len(chunk_sizes) != 1:
        raise ValueError(f"Policies must use the same chunk size, got {chunk_sizes}")
    chunk_size = chunk_sizes.pop()

    metadata = LeRobotDatasetMetadata(args.dataset, revision=args.revision)
    dataset = LeRobotDataset(
        args.dataset,
        revision=args.revision,
        delta_timestamps={
            "action": [index / metadata.fps for index in range(chunk_size)]
        },
    )
    start_frame_pairs = episode_start_frame_pairs(dataset, args.max_episodes)
    starts = [dataset[start_index] for start_index, _ in start_frame_pairs]
    next_frames = [dataset[next_index] for _, next_index in start_frame_pairs]
    device = _resolve_device(args.device)
    results = [
        evaluate_policy(label, path, device, starts, next_frames)
        for label, path in args.policy
    ]
    _print_summary(results)

    output = {
        "dataset": args.dataset,
        "revision": args.revision,
        "sample_mode": "episode_starts",
        "num_episodes": len(starts),
        "chunk_size": chunk_size,
        "device": str(device),
        "results": results,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
