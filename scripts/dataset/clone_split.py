"""Clone selected episodes from one LeRobot dataset into a new dataset.

The input split JSON is produced by ``scripts/dataset/shortcut_audit.py`` and
contains ``keep_episodes`` / ``drop_episodes``. By default this script clones
the keep split locally; pass ``--push`` to upload to Hugging Face.

Example:
    uv run python scripts/dataset/clone_split.py \
      --source-repo RevanthGundala/007-place-penne-pasta-in-bowl \
      --target-repo RevanthGundala/007-place-penne-pasta-in-bowl-diverse75 \
      --split-json outputs/dataset_audits/recommended_split.json \
      --root outputs/filtered_datasets/007-place-penne-pasta-in-bowl-diverse75 \
      --push
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch
from huggingface_hub import HfApi
from lerobot.datasets import LeRobotDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))



def tensor_to_frame_value(value):
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().numpy()
    return value


def frame_for_clone(sample: dict, feature_keys: list[str]) -> dict:
    frame = {}
    for key in feature_keys:
        value = tensor_to_frame_value(sample[key])
        if key.startswith("observation.images.") and isinstance(value, np.ndarray):
            if value.ndim == 3 and value.shape[0] in (1, 3):
                value = np.transpose(value, (1, 2, 0))
            if value.dtype != np.uint8 and value.max() <= 1.0:
                value = (value * 255).clip(0, 255).astype(np.uint8)
        frame[key] = value
    frame["task"] = sample["task"]
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Clone selected LeRobot episodes into a new dataset.")
    parser.add_argument("--source-repo", required=True)
    parser.add_argument("--source-revision", default="main")
    parser.add_argument("--target-repo", required=True)
    parser.add_argument("--split-json", default="outputs/dataset_audits/recommended_split.json")
    parser.add_argument("--episodes-key", choices=["keep_episodes", "drop_episodes"], default="keep_episodes")
    parser.add_argument("--root", default=None, help="Local root for the cloned dataset.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--no-videos", action="store_true", help="Store images instead of encoded videos.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Keep every Nth source frame within each episode.")
    parser.add_argument("--target-fps", type=int, default=None, help="FPS for the cloned dataset. Defaults to source FPS.")
    args = parser.parse_args()
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be >= 1")

    split = json.loads(Path(args.split_json).read_text())
    episodes = [int(episode) for episode in split[args.episodes_key]]
    if not episodes:
        raise ValueError(f"{args.episodes_key} is empty in {args.split_json}")

    root = Path(args.root).expanduser() if args.root else None
    if root is not None and root.exists() and args.overwrite:
        shutil.rmtree(root)

    source = LeRobotDataset(args.source_repo, revision=args.source_revision, episodes=episodes, return_uint8=True)
    features = {
        key: value
        for key, value in source.features.items()
        if key not in {"timestamp", "frame_index", "episode_index", "index", "task_index"}
    }
    target = LeRobotDataset.create(
        repo_id=args.target_repo,
        fps=args.target_fps or getattr(source, "fps", source.meta.info["fps"]),
        features=features,
        root=root,
        robot_type=source.meta.info.get("robot_type"),
        use_videos=not args.no_videos,
    )

    feature_keys = list(features)
    source_by_episode: dict[int, list[int]] = {}
    for idx in range(len(source)):
        sample = source[idx]
        episode = int(sample["episode_index"])
        source_by_episode.setdefault(episode, []).append(idx)

    for output_episode, source_episode in enumerate(episodes):
        indices = source_by_episode.get(source_episode)
        if not indices:
            raise RuntimeError(f"episode {source_episode} not found in source subset")
        kept_indices = indices[:: args.frame_stride]
        print(
            f"Cloning source episode {source_episode} -> target episode {output_episode} "
            f"({len(kept_indices)}/{len(indices)} frames, stride={args.frame_stride})"
        )
        for idx in kept_indices:
            target.add_frame(frame_for_clone(source[idx], feature_keys))
        target.save_episode()
    target.finalize()

    print(f"Cloned {len(episodes)} episodes to {target.root}")

    # Quick sanity read-back to ensure metadata/indexing is valid.
    reread = LeRobotDataset(args.target_repo, root=target.root, download_videos=False)
    print(f"Read-back ok: {reread.num_episodes} episodes, {len(reread)} frames")

    if args.push:
        if not HfApi().token:
            raise RuntimeError("No Hugging Face token found. Run `huggingface-cli login` or set HF_TOKEN.")
        target.push_to_hub(private=args.private)
        print(f"Pushed dataset to https://huggingface.co/datasets/{args.target_repo}")


if __name__ == "__main__":
    main()
