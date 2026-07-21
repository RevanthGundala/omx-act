"""Normalization round-trip + action-amplification check for LeRobot ACT policies.

Answers two questions that decide whether a stuck ACT overfit is a *pipeline*
problem or a *representation / under-training* problem:

  1. Is the action normalization invertible, and are the preprocessor
     normalizer stats identical to the postprocessor unnormalizer stats?
     If `denormalize(normalize(a)) != a`, there is a hard floor on raw action
     accuracy no matter how long you train.

  2. How much does a small *normalized* error blow up into *raw* joint units,
     under the current absolute-action representation vs a delta-action
     (action - state) representation? This quantifies the "0.05 normalized L1
     still misses the grasp" effect and whether delta actions would help.

Usage:
    uv run python evaluation/normalization_roundtrip_check.py \
        --policy-path outputs/modal_official_act_towel_ep1_novae_5k/005000/pretrained_model \
        --dataset RevanthGundala/009-place-towel-in-bowl-10fps-ep1-only \
        --ref-normalized-error 0.05
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config import JOINT_NAMES


PRE_NORMALIZER = "policy_preprocessor_step_3_normalizer_processor.safetensors"
POST_UNNORMALIZER = "policy_postprocessor_step_0_unnormalizer_processor.safetensors"


def _vec(values: torch.Tensor) -> np.ndarray:
    return values.detach().cpu().float().flatten().numpy()


def load_action_state_stats(safetensor_path: Path) -> dict[str, np.ndarray]:
    tensors = load_file(str(safetensor_path))
    out: dict[str, np.ndarray] = {}
    for key in (
        "action.mean",
        "action.std",
        "action.min",
        "action.max",
        "action.q01",
        "action.q10",
        "action.q50",
        "action.q90",
        "action.q99",
        "observation.state.mean",
        "observation.state.std",
    ):
        if key in tensors:
            out[key] = _vec(tensors[key])
    return out


def per_joint(values: np.ndarray) -> dict[str, float]:
    return {name: float(values[i]) for i, name in enumerate(JOINT_NAMES)}


def check_pre_post_consistency(pre: dict, post: dict) -> dict:
    report = {}
    for key in ("action.mean", "action.std"):
        diff = np.abs(pre[key] - post[key])
        report[key] = {
            "max_abs_diff": float(diff.max()),
            "identical": bool(diff.max() < 1e-6),
        }
    return report


def roundtrip_meanstd(samples: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """LeRobot MEAN_STD: normalize = (x - mean)/std, denormalize = x*std + mean."""
    normalized = (samples - mean) / std
    recovered = normalized * std + mean
    return np.abs(recovered - samples)


def build_sweep(stats: dict, n_random: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pieces = [
        stats[k][None, :]
        for k in ("action.min", "action.q01", "action.q10", "action.q50", "action.q90", "action.q99", "action.max")
        if k in stats
    ]
    lo, hi = stats["action.min"], stats["action.max"]
    uniform = rng.uniform(lo, hi, size=(n_random, lo.shape[0]))
    gaussian = rng.normal(stats["action.mean"], stats["action.std"], size=(n_random, lo.shape[0]))
    return np.concatenate(pieces + [uniform, gaussian], axis=0)


def load_dataset_deltas(dataset_repo: str, revision: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (actions, states) arrays [N, 6] from the dataset parquet, or None."""
    try:
        from lerobot.datasets import LeRobotDataset
    except Exception as exc:  # pragma: no cover
        print(f"[warn] could not import LeRobotDataset: {exc}")
        return None
    try:
        dataset = LeRobotDataset(dataset_repo, revision=revision)
        hf = dataset.hf_dataset
        actions = np.asarray(hf["action"], dtype=np.float64)
        states = np.asarray(hf["observation.state"], dtype=np.float64)
        return actions, states
    except Exception as exc:
        print(f"[warn] could not load dataset {dataset_repo!r}: {exc}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", required=True, help="Path to exported pretrained_model directory.")
    parser.add_argument("--dataset", default=None, help="Dataset repo id for delta-amplification stats (optional).")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--ref-normalized-error", type=float, default=0.05,
                        help="Reference normalized L1 error to translate into raw joint units.")
    parser.add_argument("--n-random", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    policy_path = Path(args.policy_path)
    pre = load_action_state_stats(policy_path / PRE_NORMALIZER)
    post = load_action_state_stats(policy_path / POST_UNNORMALIZER)

    result: dict = {"policy_path": str(policy_path)}

    # 1. Pre/post stat consistency.
    consistency = check_pre_post_consistency(pre, post)
    result["pre_post_consistency"] = consistency

    # 2. Round-trip invertibility on a sweep spanning the action range.
    sweep = build_sweep(pre, args.n_random, args.seed)
    rt_err = roundtrip_meanstd(sweep, pre["action.mean"], pre["action.std"])
    result["roundtrip"] = {
        "num_samples": int(sweep.shape[0]),
        "max_abs_error_overall": float(rt_err.max()),
        "mean_abs_error_overall": float(rt_err.mean()),
        "max_abs_error_per_joint": per_joint(rt_err.max(axis=0)),
    }

    # 3. Std amplification: what does ref-normalized-error map to in raw units?
    abs_std = pre["action.std"]
    ref = args.ref_normalized_error
    amplification: dict = {
        "ref_normalized_error": ref,
        "absolute_action_std": per_joint(abs_std),
        "raw_units_at_ref_absolute": per_joint(ref * abs_std),
    }

    deltas = load_dataset_deltas(args.dataset, args.revision) if args.dataset else None
    if deltas is not None:
        actions, states = deltas
        n = min(len(actions), len(states))
        delta = actions[:n] - states[:n]
        delta_std = delta.std(axis=0)
        abs_std_measured = actions[:n].std(axis=0)
        amplification["delta_action_std"] = per_joint(delta_std)
        amplification["absolute_action_std_measured"] = per_joint(abs_std_measured)
        amplification["delta_over_absolute_std_ratio"] = per_joint(delta_std / np.maximum(abs_std_measured, 1e-9))
        # If you train on deltas and reach the SAME normalized error, raw error
        # scales by delta_std instead of absolute_std.
        amplification["raw_units_at_ref_delta"] = per_joint(ref * delta_std)
        amplification["num_frames"] = int(n)

    result["amplification"] = amplification

    # 4. Verdict.
    rt_clean = result["roundtrip"]["max_abs_error_overall"] < 1e-3
    pre_post_ok = all(v["identical"] for v in consistency.values())
    verdict = []
    if rt_clean and pre_post_ok:
        verdict.append(
            "NORMALIZATION IS CLEAN: pre/post stats identical and round-trip error ~0. "
            "Any residual raw error is genuine model under-fit, NOT a normalization floor."
        )
    else:
        verdict.append(
            "NORMALIZATION SUSPECT: pre/post stats differ or round-trip is lossy. "
            "Fix this before any further training."
        )
    if deltas is not None:
        ratios = np.array(list(amplification["delta_over_absolute_std_ratio"].values()))
        verdict.append(
            f"Delta-action std is {ratios.mean():.2f}x the absolute-action std on average; "
            f"the same normalized error would yield ~{ratios.mean():.2f}x smaller raw error "
            "under a delta-action target."
        )
    result["verdict"] = verdict

    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
