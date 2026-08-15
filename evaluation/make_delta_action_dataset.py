"""Relabel a LeRobot dataset to *delta* actions: ``action[t] -= observation.state[t]``.

Motivation: with absolute next-joint-position actions, the action std is huge
(elbow ~30), so a small normalized error becomes several raw joint units. Delta
actions (``action - state``) have ~10x smaller std, so the same normalized L1
yields proportionally smaller raw error — directly improving one-step precision
for grasping. See ``evaluation/normalization_roundtrip_check.py``.

The relabel is purely per-frame, so it is temporal-ensemble compatible: each
delta target is a function of its own frame only. Train stock ``lerobot-train``
on the resulting dataset unchanged (fresh stats are computed over the deltas),
then run eval with ``--action-space delta`` so the live command is reconstructed
as ``command = current_state + policy_output``.

Example:
    uv run python evaluation/make_delta_action_dataset.py \
      --source-repo RevanthGundala/009-place-towel-in-bowl-10fps \
      --target-repo RevanthGundala/009-place-towel-in-bowl-10fps-delta \
      --root outputs/filtered_datasets/009-place-towel-in-bowl-10fps-delta \
      --push
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

import numpy as np
import torch
from huggingface_hub import get_token
from lerobot.datasets import LeRobotDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))



def tensor_to_frame_value(value):
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().numpy()
    return value


def frame_for_clone(sample: dict, feature_keys: list[str], state_key: str, action_key: str) -> dict:
    """Build a frame dict, relabeling the action as ``action - state``."""
    state = np.asarray(tensor_to_frame_value(sample[state_key]), dtype=np.float32).reshape(-1)
    frame = {}
    for key in feature_keys:
        value = tensor_to_frame_value(sample[key])
        if key.startswith("observation.images.") and isinstance(value, np.ndarray):
            if value.ndim == 3 and value.shape[0] in (1, 3):
                value = np.transpose(value, (1, 2, 0))
            if value.dtype != np.uint8 and value.max() <= 1.0:
                value = (value * 255).clip(0, 255).astype(np.uint8)
        elif key == action_key:
            action = np.asarray(value, dtype=np.float32).reshape(-1)
            if action.shape != state.shape:
                raise ValueError(
                    f"action shape {action.shape} != state shape {state.shape}; cannot form delta"
                )
            value = (action - state).astype(np.float32)
        frame[key] = value
    frame["task"] = sample["task"]
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Relabel a LeRobot dataset to delta actions.")
    parser.add_argument("--source-repo", required=True)
    parser.add_argument("--source-revision", default="main")
    parser.add_argument("--target-repo", required=True)
    parser.add_argument("--root", default=None, help="Local root for the relabeled dataset.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--no-videos", action="store_true", help="Store images instead of encoded videos.")
    parser.add_argument("--state-key", default="observation.state")
    parser.add_argument("--action-key", default="action")
    args = parser.parse_args()

    root = Path(args.root).expanduser() if args.root else None
    if root is not None and root.exists() and args.overwrite:
        shutil.rmtree(root)

    source = LeRobotDataset(args.source_repo, revision=args.source_revision, return_uint8=True)
    features = {
        key: value
        for key, value in source.features.items()
        if key not in {"timestamp", "frame_index", "episode_index", "index", "task_index"}
    }
    if args.action_key not in features or args.state_key not in features:
        raise KeyError(
            f"source dataset must contain {args.action_key!r} and {args.state_key!r}; "
            f"found {sorted(features)}"
        )

    target = LeRobotDataset.create(
        repo_id=args.target_repo,
        fps=getattr(source, "fps", source.meta.info["fps"]),
        features=features,
        root=root,
        robot_type=source.meta.info.get("robot_type"),
        use_videos=not args.no_videos,
    )

    feature_keys = list(features)
    source_by_episode: dict[int, list[int]] = {}
    for idx in range(len(source)):
        episode = int(source[idx]["episode_index"])
        source_by_episode.setdefault(episode, []).append(idx)

    max_abs_delta = 0.0
    source_abs_actions: list[np.ndarray] = []
    for output_episode, source_episode in enumerate(sorted(source_by_episode)):
        indices = source_by_episode[source_episode]
        print(
            f"Relabeling source episode {source_episode} -> target episode {output_episode} "
            f"({len(indices)} frames)"
        )
        for idx in indices:
            sample = source[idx]
            source_abs_actions.append(
                np.asarray(tensor_to_frame_value(sample[args.action_key]), dtype=np.float32).reshape(-1)
            )
            frame = frame_for_clone(sample, feature_keys, args.state_key, args.action_key)
            max_abs_delta = max(max_abs_delta, float(np.abs(frame[args.action_key]).max()))
            target.add_frame(frame)
        target.save_episode()
    target.finalize()

    print(f"Relabeled {len(source_by_episode)} episodes to {target.root}")
    print(f"max abs delta action across dataset: {max_abs_delta:.3f} joint units")

    reread = LeRobotDataset(args.target_repo, root=target.root, download_videos=False)
    action_std = np.asarray(reread.meta.stats[args.action_key]["std"]).reshape(-1)
    print(f"Read-back ok: {reread.num_episodes} episodes, {len(reread)} frames")
    print(f"delta action std per joint: {np.round(action_std, 4).tolist()}")

    # Correctness self-check: reconstructing absolute actions from the written
    # deltas (delta + state) must recover the original source actions. Guards
    # against joint-order, dtype, or storage-quantization regressions before any
    # expensive training run.
    if len(reread) != len(source_abs_actions):
        raise RuntimeError(
            f"frame count mismatch: target {len(reread)} vs source {len(source_abs_actions)}"
        )
    max_recon_error = 0.0
    reread_hf = reread.hf_dataset
    for i in range(len(reread)):
        row = reread_hf[i]
        delta = np.asarray(row[args.action_key], dtype=np.float64).reshape(-1)
        state = np.asarray(row[args.state_key], dtype=np.float64).reshape(-1)
        recon = delta + state
        max_recon_error = max(max_recon_error, float(np.abs(recon - source_abs_actions[i]).max()))
    print(f"reconstruction self-check: max |(delta+state) - source_action| = {max_recon_error:.3e}")
    if max_recon_error > 1e-3:
        raise RuntimeError(
            f"delta reconstruction error {max_recon_error:.3e} exceeds 1e-3; "
            "relabel/reconstruction is not invertible — do NOT train on this dataset."
        )
    print("reconstruction self-check PASSED (delta dataset is exactly invertible).")

    if args.push:
        if not get_token():
            raise RuntimeError(
                "No Hugging Face token found. Run `huggingface-cli login` or set HF_TOKEN."
            )
        target.push_to_hub(private=args.private)
        print(f"Pushed dataset to https://huggingface.co/datasets/{args.target_repo}")


if __name__ == "__main__":
    main()
