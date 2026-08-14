"""Read episodes out of a LeRobot dataset for the RECAP dataset builder.

RECAP's value function trains on "all of the data collected for the task so
far" (arXiv:2511.14759 Sec. V-D), which for us spans two storage formats: the
async evaluation rollouts land as trial directories of JSONL plus video, while
demonstrations and DAgger corrections are LeRobot datasets. This module exposes
the second of those in the same shape the builder already consumes.

Frames are decoded with OpenCV rather than through ``LeRobotDataset.__getitem__``
for two reasons: the item path pulls in torchcodec, whose shared libraries do not
load on every machine here, and it decodes one frame per call, while a v3
dataset packs many episodes into one video file and an episode's frames are a
contiguous slice of it.

Episodes are described before any pixels are read. The builder needs the longest
episode to choose ``C_fail`` before it writes anything, and decoding every
episode up front would cost tens of gigabytes for a dataset this size.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

OUTCOMES_DIR = "outcomes"
VALID_OUTCOMES = frozenset({"success", "failure"})


class MissingOutcomeError(ValueError):
    """Raised when an episode carries no success/failure label."""


@dataclass(frozen=True)
class LeRobotEpisodeRef:
    """Everything needed to load one episode, without loading it."""

    root: Path
    repo_id: str
    episode_index: int
    length: int
    task: str
    success: bool
    joint_names: tuple[str, ...]
    camera_names: tuple[str, ...]
    frame_shape: tuple[int, int, int]
    fps: int
    data_slice: tuple[int, int]
    video_refs: dict[str, tuple[Path, int]]
    has_intervention: bool

    @property
    def source_id(self) -> str:
        return f"{self.root}#episode-{self.episode_index:06d}"


def _read_meta(root: Path) -> dict:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Not a LeRobot dataset: {info_path} is missing")
    return json.loads(info_path.read_text())


def _episode_table(root: Path) -> pd.DataFrame:
    paths = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(
            f"{root} has no meta/episodes/: it was never finalized, so its "
            "episodes cannot be read"
        )
    frames = [pd.read_parquet(path) for path in paths]
    table = pd.concat(frames, ignore_index=True)
    return table.sort_values("episode_index").reset_index(drop=True)


def _outcome_for(root: Path, episode_index: int, default_success: bool | None) -> bool:
    path = root / OUTCOMES_DIR / f"episode-{episode_index:06d}.json"
    if not path.is_file():
        if default_success is None:
            raise MissingOutcomeError(
                f"Episode {episode_index} of {root} has no outcome sidecar at "
                f"{path}, and no default outcome was given for this source"
            )
        return default_success
    payload = json.loads(path.read_text())
    label = payload.get("outcome_label")
    if label not in VALID_OUTCOMES:
        raise ValueError(f"Invalid outcome label in {path}: {label!r}")
    return label == "success"


def _video_path(root: Path, key: str, chunk_index: int, file_index: int) -> Path:
    path = (
        root
        / "videos"
        / key
        / f"chunk-{int(chunk_index):03d}"
        / f"file-{int(file_index):03d}.mp4"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Episode video is missing: {path}")
    return path


def _probe_frame_shape(path: Path) -> tuple[int, int, int]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"Video contains no frames: {path}")
        return tuple(frame.shape)
    finally:
        capture.release()


def scan_lerobot_dataset(
    root: Path,
    *,
    repo_id: str | None = None,
    default_success: bool | None = None,
) -> list[LeRobotEpisodeRef]:
    """Describe every episode in a LeRobot dataset without decoding frames.

    ``default_success`` supplies the outcome for sources that carry no sidecar.
    Demonstrations are the case that needs it: they are expert successes by
    construction, so requiring a label for each would be busywork. Leave it
    ``None`` for rollouts, where a missing label is a real error.
    """
    root = Path(root).expanduser().resolve()
    info = _read_meta(root)
    repo_id = repo_id or info.get("repo_id") or root.name
    fps = int(info["fps"])
    features = info["features"]
    joint_names = tuple(features["observation.state"]["names"])
    action_names = tuple(features["action"]["names"])
    if action_names != joint_names:
        raise ValueError(
            f"{root} orders action and state differently: {action_names} vs "
            f"{joint_names}"
        )
    camera_keys = sorted(
        key for key in features if key.startswith("observation.images.")
    )
    if not camera_keys:
        raise ValueError(f"{root} has no camera features")
    has_intervention = "intervention" in features

    table = _episode_table(root)
    refs: list[LeRobotEpisodeRef] = []
    frame_shape: tuple[int, int, int] | None = None
    # Column names contain slashes, which itertuples renames, so read records.
    for row in table.to_dict("records"):
        episode_index = int(row["episode_index"])
        video_refs = {}
        for key in camera_keys:
            camera = key.removeprefix("observation.images.")
            path = _video_path(
                root,
                key,
                row[f"videos/{key}/chunk_index"],
                row[f"videos/{key}/file_index"],
            )
            start = int(round(float(row[f"videos/{key}/from_timestamp"]) * fps))
            video_refs[camera] = (path, start)
        if frame_shape is None:
            frame_shape = _probe_frame_shape(next(iter(video_refs.values()))[0])
        tasks = row["tasks"]
        task = str(tasks[0]) if len(tasks) else ""
        refs.append(
            LeRobotEpisodeRef(
                root=root,
                repo_id=repo_id,
                episode_index=episode_index,
                length=int(row["length"]),
                task=task,
                success=_outcome_for(root, episode_index, default_success),
                joint_names=joint_names,
                camera_names=tuple(
                    key.removeprefix("observation.images.") for key in camera_keys
                ),
                frame_shape=frame_shape,
                fps=fps,
                data_slice=(
                    int(row["dataset_from_index"]),
                    int(row["dataset_to_index"]),
                ),
                video_refs=video_refs,
                has_intervention=has_intervention,
            )
        )
    if not refs:
        raise ValueError(f"{root} contains no episodes")
    return refs


def _read_video_slice(path: Path, start: int, count: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        if start:
            capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        for _ in range(count):
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if len(frames) != count:
        raise ValueError(
            f"{path} yielded {len(frames)} frames from index {start}, expected "
            f"{count}. The episode video range and its recorded length disagree."
        )
    return np.stack(frames)


@dataclass
class LoadedEpisode:
    """One episode's arrays, ready to write as RECAP transitions."""

    states: np.ndarray
    actions: np.ndarray
    images: dict[str, np.ndarray]
    intervention: np.ndarray
    success: bool
    task: str
    source_id: str


def load_lerobot_episode(ref: LeRobotEpisodeRef) -> LoadedEpisode:
    """Decode one episode. Call this inside the write loop, not up front."""
    paths = sorted((ref.root / "data").rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"{ref.root} has no data/ parquet files")
    table = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    rows = table[table["episode_index"] == ref.episode_index]
    rows = rows.sort_values("frame_index")
    if len(rows) != ref.length:
        raise ValueError(
            f"Episode {ref.episode_index} of {ref.root} declares {ref.length} "
            f"frames but its data has {len(rows)}"
        )

    states = np.stack(
        [np.asarray(value, dtype=np.float32) for value in rows["observation.state"]]
    )
    actions = np.stack(
        [np.asarray(value, dtype=np.float32) for value in rows["action"]]
    )
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError(f"Non-finite state/action in {ref.source_id}")

    if ref.has_intervention:
        intervention = np.asarray(
            [bool(np.asarray(value).ravel()[0]) for value in rows["intervention"]],
            dtype=np.bool_,
        )
    else:
        # False, not True. This column exists to carry Sec. IV-B, which forces
        # the indicator positive only for "human corrections during autonomous
        # rollouts". Demonstrations are not that: the paper computes their
        # indicator from the advantage like any other data, thresholded so only
        # ~30% comes out positive. Marking them here would force all of it.
        intervention = np.zeros(ref.length, dtype=np.bool_)

    images = {
        camera: _read_video_slice(path, start, ref.length)
        for camera, (path, start) in ref.video_refs.items()
    }
    return LoadedEpisode(
        states=states,
        actions=actions,
        images=images,
        intervention=intervention,
        success=ref.success,
        task=ref.task,
        source_id=ref.source_id,
    )
