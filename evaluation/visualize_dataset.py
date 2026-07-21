"""Quick Rerun visualizer for LeRobot dataset episodes.

Examples:
    uv run python evaluation/visualize_dataset.py --episode 0
    uv run python evaluation/visualize_dataset.py --episodes all --visual-only
    uv run python evaluation/visualize_dataset.py --episodes 50:70 --grid --grid-cols 6
    uv run python evaluation/visualize_dataset.py --episodes all --play-grid --stride 10
"""

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
import rerun as rr
from lerobot.datasets.lerobot_dataset import LeRobotDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config import JOINT_NAMES, TRAIN_DATASET_REPO_ID


def parse_episodes(value: str | None, fallback: int) -> list[int] | None:
    if value is None:
        return [fallback]
    if value == "all":
        return None
    if ":" in value:
        start, end = value.split(":", 1)
        return list(range(int(start), int(end)))
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def scalar(value) -> int | float:
    return value.item() if hasattr(value, "item") else value


def image_to_uint8_hwc(img) -> np.ndarray:
    if hasattr(img, "numpy"):
        img = img.numpy()
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    return img


def resize_nearest(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    height, width = size
    ys = np.linspace(0, img.shape[0] - 1, height).astype(np.int64)
    xs = np.linspace(0, img.shape[1] - 1, width).astype(np.int64)
    return img[ys][:, xs]


def collect_episode_indices(ds: LeRobotDataset) -> dict[int, list[int]]:
    indices: dict[int, list[int]] = {}
    for i in range(len(ds)):
        sample = ds.hf_dataset[i]
        episode = int(scalar(sample["episode_index"]))
        indices.setdefault(episode, []).append(i)
    return indices


def detect_cameras(ds: LeRobotDataset, requested: str) -> list[str]:
    if requested != "all":
        return [name.strip() for name in requested.split(",") if name.strip()]
    if len(ds) == 0:
        return []
    sample = ds[0]
    prefix = "observation.images."
    return sorted(key.removeprefix(prefix) for key in sample if key.startswith(prefix))


def log_episode_grid(
    ds: LeRobotDataset,
    episodes: list[int] | None,
    cameras: list[str],
    samples_per_episode: int,
    grid_cols: int,
    thumb_width: int,
) -> None:
    first_indices = collect_episode_indices(ds)
    episode_ids = episodes if episodes is not None else sorted(first_indices)
    thumb_height = int(thumb_width * 0.75)
    cell_width = thumb_width * samples_per_episode
    cell_height = thumb_height

    for camera in cameras:
        rows = int(np.ceil(len(episode_ids) / grid_cols))
        sheet = np.zeros((rows * cell_height, grid_cols * cell_width, 3), dtype=np.uint8)
        key = f"observation.images.{camera}"
        for episode_idx, episode in enumerate(episode_ids):
            indices = first_indices.get(episode, [])
            if not indices:
                continue
            sample_positions = np.linspace(0, len(indices) - 1, samples_per_episode).astype(int)
            row = episode_idx // grid_cols
            col = episode_idx % grid_cols
            for sample_col, pos in enumerate(sample_positions):
                sample = ds[indices[pos]]
                if key not in sample:
                    continue
                img = resize_nearest(image_to_uint8_hwc(sample[key]), (thumb_height, thumb_width))
                y0 = row * cell_height
                x0 = col * cell_width + sample_col * thumb_width
                sheet[y0 : y0 + thumb_height, x0 : x0 + thumb_width] = img

        rr.log(f"grid/{camera}", rr.Image(sheet))
        print(f"Logged {camera} grid: {len(episode_ids)} episodes, {samples_per_episode} samples/episode")


def log_playable_grid(
    ds: LeRobotDataset,
    episodes: list[int] | None,
    cameras: list[str],
    grid_cols: int,
    thumb_width: int,
    stride: int,
) -> None:
    episode_indices = collect_episode_indices(ds)
    episode_ids = episodes if episodes is not None else sorted(episode_indices)
    thumb_height = int(thumb_width * 0.75)
    rows = int(np.ceil(len(episode_ids) / grid_cols))
    max_len = max(len(episode_indices[episode]) for episode in episode_ids)

    for frame in range(0, max_len, stride):
        rr.set_time("relative_frame", sequence=frame)
        for camera in cameras:
            sheet = np.full((rows * thumb_height, grid_cols * thumb_width, 3), 32, dtype=np.uint8)
            key = f"observation.images.{camera}"
            for episode_idx, episode in enumerate(episode_ids):
                indices = episode_indices.get(episode, [])
                if not indices:
                    continue
                sample = ds[indices[min(frame, len(indices) - 1)]]
                if key not in sample:
                    continue
                img = resize_nearest(image_to_uint8_hwc(sample[key]), (thumb_height, thumb_width))
                row = episode_idx // grid_cols
                col = episode_idx % grid_cols
                y0 = row * thumb_height
                x0 = col * thumb_width
                sheet[y0 : y0 + thumb_height, x0 : x0 + thumb_width] = img
            rr.log(f"play_grid/{camera}", rr.Image(sheet))

    print(
        f"Logged playable grid: {len(episode_ids)} episodes, "
        f"{max_len // stride + 1} timesteps, stride={stride}"
    )


def build_playable_grid_frame(
    ds: LeRobotDataset,
    episode_indices: dict[int, list[int]],
    episode_ids: list[int],
    camera: str,
    relative_frame: int,
    grid_cols: int,
    thumb_width: int,
) -> np.ndarray:
    thumb_height = int(thumb_width * 0.75)
    rows = int(np.ceil(len(episode_ids) / grid_cols))
    sheet = np.full((rows * thumb_height, grid_cols * thumb_width, 3), 32, dtype=np.uint8)
    key = f"observation.images.{camera}"
    for episode_idx, episode in enumerate(episode_ids):
        indices = episode_indices.get(episode, [])
        if not indices:
            continue
        sample = ds[indices[min(relative_frame, len(indices) - 1)]]
        if key not in sample:
            continue
        img = resize_nearest(image_to_uint8_hwc(sample[key]), (thumb_height, thumb_width))
        row = episode_idx // grid_cols
        col = episode_idx % grid_cols
        y0 = row * thumb_height
        x0 = col * thumb_width
        sheet[y0 : y0 + thumb_height, x0 : x0 + thumb_width] = img
    return sheet


def export_playable_grid_mp4(
    ds: LeRobotDataset,
    episodes: list[int] | None,
    cameras: list[str],
    grid_cols: int,
    thumb_width: int,
    stride: int,
    output_dir: Path,
    fps: float,
) -> None:
    episode_indices = collect_episode_indices(ds)
    episode_ids = episodes if episodes is not None else sorted(episode_indices)
    max_len = max(len(episode_indices[episode]) for episode in episode_ids)
    output_dir.mkdir(parents=True, exist_ok=True)

    for camera in cameras:
        output_path = output_dir / f"{camera}_grid.mp4"
        first_frame = build_playable_grid_frame(
            ds,
            episode_indices,
            episode_ids,
            camera,
            0,
            grid_cols,
            thumb_width,
        )
        height, width = first_frame.shape[:2]
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"failed to open video writer: {output_path}")
        try:
            for relative_frame in range(0, max_len, stride):
                frame = build_playable_grid_frame(
                    ds,
                    episode_indices,
                    episode_ids,
                    camera,
                    relative_frame,
                    grid_cols,
                    thumb_width,
                )
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
        print(f"Wrote {output_path} ({len(episode_ids)} episodes, stride={stride}, fps={fps:g})")


def export_episode_raw_mp4(
    ds: LeRobotDataset,
    episode: int,
    cameras: list[str],
    output_dir: Path,
    stride: int,
    fps: float,
) -> None:
    episode_indices = collect_episode_indices(ds)
    indices = episode_indices.get(episode)
    if not indices:
        raise ValueError(f"episode {episode} not found")
    output_dir.mkdir(parents=True, exist_ok=True)

    for camera in cameras:
        key = f"observation.images.{camera}"
        output_path = output_dir / f"{camera}_episode_{episode:04d}.mp4"
        writer = None
        frame_count = 0
        try:
            for i, dataset_index in enumerate(indices):
                if i % stride != 0:
                    continue
                sample = ds[dataset_index]
                if key not in sample:
                    continue
                frame = image_to_uint8_hwc(sample[key])
                if writer is None:
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(output_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"failed to open video writer: {output_path}")
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                frame_count += 1
        finally:
            if writer is not None:
                writer.release()
        print(f"Wrote {output_path} ({frame_count} raw frames, stride={stride}, fps={fps:g})")


def export_episode_stills(
    ds: LeRobotDataset,
    episode: int,
    cameras: list[str],
    output_dir: Path,
    frame: int,
) -> None:
    episode_indices = collect_episode_indices(ds)
    indices = episode_indices.get(episode)
    if not indices:
        raise ValueError(f"episode {episode} not found")
    sample = ds[indices[min(frame, len(indices) - 1)]]
    output_dir.mkdir(parents=True, exist_ok=True)
    for camera in cameras:
        key = f"observation.images.{camera}"
        if key not in sample:
            continue
        output_path = output_dir / f"dataset_episode_{episode:04d}_frame_{frame:05d}_{camera}.jpg"
        cv2.imwrite(str(output_path), cv2.cvtColor(image_to_uint8_hwc(sample[key]), cv2.COLOR_RGB2BGR))
        print(f"Wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=TRAIN_DATASET_REPO_ID)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--episodes", default=None, help="'all', 'start:end', or comma-separated episode ids")
    parser.add_argument("--cameras", default="wrist,top", help="Comma-separated camera names, or 'all'.")
    parser.add_argument("--visual-only", action="store_true", help="Only log camera images, not state/action scalars")
    parser.add_argument("--stride", type=int, default=1, help="Log every Nth frame")
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--grid", action="store_true", help="Log contact-sheet grids instead of a frame timeline")
    parser.add_argument("--play-grid", action="store_true", help="Log playable contact-sheet grids over relative episode time")
    parser.add_argument("--grid-cols", type=int, default=5, help="Episode cells per grid row")
    parser.add_argument("--samples-per-episode", type=int, default=5, help="Frames sampled across each episode cell")
    parser.add_argument("--thumb-width", type=int, default=128)
    parser.add_argument("--export-mp4", action="store_true", help="Write playable camera-grid MP4 files locally instead of opening Rerun.")
    parser.add_argument("--export-raw-mp4", action="store_true", help="Write raw per-camera episode MP4 files locally instead of opening Rerun.")
    parser.add_argument("--export-stills", action="store_true", help="Write camera stills locally instead of opening Rerun.")
    parser.add_argument("--still-frame", type=int, default=0, help="Relative episode frame for --export-stills.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dataset_videos"), help="Directory for --export-mp4 outputs.")
    parser.add_argument("--output-fps", type=float, default=10.0, help="FPS metadata for --export-mp4 videos.")
    args = parser.parse_args()

    episodes = parse_episodes(args.episodes, args.episode)
    label = "all" if episodes is None else ",".join(map(str, episodes[:5])) + ("..." if len(episodes) > 5 else "")
    print(f"Loading {args.repo_id} episodes {label}...")
    ds_kwargs = {"episodes": episodes, "video_backend": args.video_backend}
    if args.root is not None:
        ds_kwargs["root"] = args.root
    ds = LeRobotDataset(args.repo_id, **ds_kwargs)
    cameras = detect_cameras(ds, args.cameras)
    print(f"Cameras: {cameras}")

    if args.export_stills:
        selected = episodes if episodes is not None else [args.episode]
        for episode in selected:
            export_episode_stills(ds, episode, cameras, args.output_dir, args.still_frame)
        return

    if args.export_raw_mp4:
        selected = episodes if episodes is not None else [args.episode]
        for episode in selected:
            export_episode_raw_mp4(ds, episode, cameras, args.output_dir, args.stride, args.output_fps)
        return

    if args.export_mp4:
        export_playable_grid_mp4(
            ds,
            episodes,
            cameras,
            args.grid_cols,
            args.thumb_width,
            args.stride,
            args.output_dir,
            args.output_fps,
        )
        return

    rr.init(f"dataset_{args.repo_id.split('/')[-1]}_eps_{label}", spawn=True)

    if args.grid:
        log_episode_grid(ds, episodes, cameras, args.samples_per_episode, args.grid_cols, args.thumb_width)
        print("Rerun grid viewer should be open.")
        return
    if args.play_grid:
        log_playable_grid(ds, episodes, cameras, args.grid_cols, args.thumb_width, args.stride)
        print("Rerun playable grid viewer should be open.")
        return

    for i in range(len(ds)):
        if i % args.stride != 0:
            continue
        sample = ds[i]
        episode = int(scalar(sample["episode_index"]))
        frame = int(scalar(sample["frame_index"]))
        rr.set_time("episode", sequence=episode)
        rr.set_time("frame", sequence=frame)
        rr.set_time("time", duration=float(scalar(sample["timestamp"])))

        if not args.visual_only:
            state = sample["observation.state"].numpy()
            action = sample["action"].numpy()
            for joint_idx, name in enumerate(JOINT_NAMES):
                rr.log(f"state/{name}", rr.Scalars(float(state[joint_idx])))
                rr.log(f"action/{name}", rr.Scalars(float(action[joint_idx])))

        for camera in cameras:
            key = f"observation.images.{camera}"
            if key in sample:
                rr.log(f"camera/{camera}", rr.Image(image_to_uint8_hwc(sample[key])))

    print(f"Logged up to {len(ds)} frames. Rerun viewer should be open.")


if __name__ == "__main__":
    main()
