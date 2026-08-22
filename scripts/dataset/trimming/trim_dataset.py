"""Produce a no-op-trimmed copy of a LeRobot v3.0 dataset, keeping the SAME v3.0 format (no
FastWAM-style format conversion) so both train/act/custom.py and stock lerobot-train can consume
the result directly by repo_id, exactly like the source dataset.

Trims only the leading/trailing near-static "settle" segment of each episode (see
scripts/dataset/trimming/noop_trim.py docstring for why this matters for ACT chunk quality). Uses the OFFICIAL
lerobot package for both reading (v3.0, handles the packed/chunked parquet+video format) and
writing (LeRobotDataset.create() -> add_frame() -> save_episode() -> finalize()).

Usage:
    uv run python -m scripts.dataset.trimming.trim_dataset \
        --repo-id RevanthGundala/011-place-sponge-in-bowl \
        --output-repo-id RevanthGundala/012-place-sponge-in-bowl \
        --output-dir outputs/datasets/012-place-sponge-in-bowl \
        [--max-episodes 2]   # smoke test on a subset first
        [--push]             # push to the HF Hub when done (omit to just build locally)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from scripts.dataset.trimming.noop_trim import find_noop_trim_bounds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="Source HF LeRobot dataset repo id (v3.0 format).")
    parser.add_argument("--output-repo-id", required=True, help="Target repo id for the trimmed dataset.")
    parser.add_argument("--output-dir", required=True, help="Local target directory (must not already exist).")
    parser.add_argument("--max-episodes", type=int, default=None, help="Convert only the first N episodes (smoke test).")
    parser.add_argument(
        "--noop-threshold-deg", type=float, default=0.1,
        help="Max per-joint |delta| (degrees) below which a frame-to-frame transition is a no-op. "
        "Set to 0 to disable trimming (useful as a passthrough sanity check).",
    )
    parser.add_argument(
        "--noop-margin-frames", type=int, default=4,
        help="Keep this many extra frames at the start/end of the detected no-op run.",
    )
    parser.add_argument("--push", action="store_true", help="Push the finished dataset to the HF Hub.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"--output-dir already exists: {output_dir}. Choose a fresh directory.")

    print(f"Probing source dataset metadata: {args.repo_id}")
    probe_ds = LeRobotDataset(args.repo_id, episodes=[0])
    fps = int(round(probe_ds.fps))
    total_episodes = probe_ds.meta.total_episodes
    camera_keys = sorted(
        key for key, ft in probe_ds.meta.features.items() if key.startswith("observation.images.")
    )
    state_names = probe_ds.meta.features["observation.state"].get("names")
    action_names = probe_ds.meta.features["action"].get("names")
    state_dim = int(probe_ds.meta.features["observation.state"]["shape"][0])
    action_dim = int(probe_ds.meta.features["action"]["shape"][0])
    height, width = probe_ds[0][camera_keys[0]].shape[-2:]
    print(f"  fps={fps} total_episodes={total_episodes} cameras={camera_keys}")
    print(f"  state_dim={state_dim} action_dim={action_dim} image_size=({height},{width})")

    num_episodes = total_episodes if args.max_episodes is None else min(args.max_episodes, total_episodes)
    print(f"Trimming {num_episodes}/{total_episodes} episodes -> {output_dir}")

    features = {
        "observation.state": {"dtype": "float32", "shape": (state_dim,), "names": state_names},
        "action": {"dtype": "float32", "shape": (action_dim,), "names": action_names},
    }
    for cam_key in camera_keys:
        features[cam_key] = {
            "dtype": "video",
            "shape": (3, height, width),
            "names": ["channels", "height", "width"],
        }

    target_ds = LeRobotDataset.create(
        repo_id=args.output_repo_id,
        fps=fps,
        features=features,
        root=output_dir,
        robot_type=probe_ds.meta.robot_type,
        use_videos=True,
    )

    total_orig_frames = 0
    total_kept_frames = 0
    per_episode_log = []

    for ep_idx in range(num_episodes):
        source_ds = probe_ds if ep_idx == 0 else LeRobotDataset(args.repo_id, episodes=[ep_idx])
        num_frames = len(source_ds)

        full_action = np.stack(
            [source_ds[i]["action"].numpy() for i in range(num_frames)]
        ).astype(np.float32)
        start, stop = find_noop_trim_bounds(full_action, args.noop_threshold_deg, args.noop_margin_frames)
        kept = stop - start
        total_orig_frames += num_frames
        total_kept_frames += kept
        per_episode_log.append((ep_idx, num_frames, start, num_frames - stop, kept))
        print(
            f"  episode {ep_idx}: {num_frames} frames "
            f"(trim lead={start}, trim trail={num_frames - stop}, keeping {kept})",
            flush=True,
        )

        for frame_idx in range(start, stop):
            item = source_ds[frame_idx]
            frame_dict = {
                "observation.state": item["observation.state"].numpy().astype(np.float32),
                "action": item["action"].numpy().astype(np.float32),
                "task": item["task"],
            }
            for cam_key in camera_keys:
                frame_dict[cam_key] = item[cam_key].numpy()  # CHW float32 [0,1]; add_frame handles conversion
            target_ds.add_frame(frame_dict)

        target_ds.save_episode()
        print(f"  episode {ep_idx}: saved.", flush=True)

    target_ds.finalize()
    print(f"\nDone. Trimmed dataset at: {output_dir}")
    print(f"total original frames: {total_orig_frames}")
    print(f"total kept frames: {total_kept_frames} "
          f"({100 * total_kept_frames / total_orig_frames:.1f}%)")
    print(f"total trimmed: {total_orig_frames - total_kept_frames} "
          f"({100 * (total_orig_frames - total_kept_frames) / total_orig_frames:.1f}%)")

    if args.push:
        print(f"\nPushing to hub: {args.output_repo_id}")
        target_ds.push_to_hub()
        print("Push complete.")
    else:
        print("\n--push not set; dataset left local only. Re-run with --push to publish it.")


if __name__ == "__main__":
    main()
