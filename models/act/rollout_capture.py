"""Helpers for recording and labeling live ACT evaluation rollouts."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TextIO

import cv2
import numpy as np

VALID_OUTCOMES = frozenset({"success", "failure"})


class RolloutVideoRecorder:
    """Write synchronized RGB camera observations to one MP4 per camera."""

    def __init__(
        self,
        output_dir: Path,
        camera_names: Sequence[str],
        fps: float,
        *,
        writer_factory=None,
    ) -> None:
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be a positive finite value")
        if not camera_names or len(set(camera_names)) != len(camera_names):
            raise ValueError("camera_names must contain unique camera names")
        for name in camera_names:
            if Path(name).name != name:
                raise ValueError(f"Invalid camera name: {name!r}")

        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.camera_names = tuple(camera_names)
        self.fps = float(fps)
        self._writer_factory = writer_factory or cv2.VideoWriter
        self._writers = {}
        self._frame_shapes: dict[str, tuple[int, int, int]] = {}
        self.frame_count = 0

    @property
    def video_paths(self) -> dict[str, Path]:
        return {
            camera: self.output_dir / f"{camera}.mp4"
            for camera in self.camera_names
        }

    def write(self, frames: Mapping[str, np.ndarray]) -> int:
        missing = set(self.camera_names) - set(frames)
        if missing:
            raise KeyError(f"Missing rollout camera frames: {sorted(missing)}")

        prepared = {}
        try:
            for camera in self.camera_names:
                frame = np.asarray(frames[camera])
                if frame.ndim != 3 or frame.shape[2] != 3:
                    raise ValueError(
                        f"Camera {camera!r} must have HWC RGB shape, got {frame.shape}"
                    )
                if frame.dtype != np.uint8:
                    raise ValueError(
                        f"Camera {camera!r} must be uint8, got {frame.dtype}"
                    )
                frame = np.ascontiguousarray(frame)
                shape = tuple(int(value) for value in frame.shape)
                if camera in self._frame_shapes and self._frame_shapes[camera] != shape:
                    raise ValueError(
                        f"Camera {camera!r} changed shape from "
                        f"{self._frame_shapes[camera]} to {shape}"
                    )
                prepared[camera] = frame

                if camera not in self._writers:
                    height, width = frame.shape[:2]
                    writer = self._writer_factory(
                        str(self.video_paths[camera]),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        self.fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        writer.release()
                        raise RuntimeError(
                            f"Failed to open rollout video writer for {camera!r}"
                        )
                    self._writers[camera] = writer
                    self._frame_shapes[camera] = shape

            frame_index = self.frame_count
            for camera, frame in prepared.items():
                self._writers[camera].write(
                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                )
            self.frame_count += 1
            return frame_index
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        for writer in self._writers.values():
            writer.release()
        self._writers.clear()


class RolloutArtifactWriter:
    """Persist aligned policy observations, actions, and camera videos."""

    def __init__(
        self,
        output_dir: Path,
        camera_names: Sequence[str],
        fps: float,
        *,
        writer_factory=None,
    ) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.video_recorder = RolloutVideoRecorder(
            self.output_dir / "videos",
            camera_names,
            fps,
            writer_factory=writer_factory,
        )
        self.observations_path = self.output_dir / "observations.jsonl"
        self.actions_path = self.output_dir / "actions.jsonl"
        self._observations_file = self.observations_path.open(
            "x",
            encoding="utf-8",
        )
        self._actions_file = self.actions_path.open("x", encoding="utf-8")
        self.observation_count = 0
        self.action_count = 0
        self.last_observation_frame_index: int | None = None
        self._closed = False

    @property
    def video_paths(self) -> dict[str, Path]:
        return self.video_recorder.video_paths

    def record_observation(
        self,
        frames: Mapping[str, np.ndarray],
        payload: Mapping,
    ) -> int:
        if self._closed:
            raise RuntimeError("Rollout artifact writer is closed")
        frame_index = self.video_recorder.write(frames)
        entry = {
            **dict(payload),
            "observation_index": self.observation_count,
            "frame_index": frame_index,
        }
        self._observations_file.write(json.dumps(entry, sort_keys=True) + "\n")
        self._observations_file.flush()
        self.last_observation_frame_index = frame_index
        self.observation_count += 1
        return frame_index

    def record_action(self, payload: Mapping) -> None:
        if self._closed:
            raise RuntimeError("Rollout artifact writer is closed")
        entry = {
            **dict(payload),
            "action_index": self.action_count,
            "observation_frame_index": self.last_observation_frame_index,
        }
        self._actions_file.write(json.dumps(entry, sort_keys=True) + "\n")
        self._actions_file.flush()
        self.action_count += 1

    def close(self) -> None:
        if self._closed:
            return
        self.video_recorder.close()
        self._observations_file.close()
        self._actions_file.close()
        self._closed = True


def prompt_for_outcome(
    *,
    stream: TextIO = sys.stdin,
    output: TextIO = sys.stdout,
) -> str | None:
    """Prompt an interactive operator for a binary rollout outcome."""
    if not stream.isatty():
        print(
            "[eval_async] stdin is non-interactive; outcome left unlabeled.",
            file=output,
            flush=True,
        )
        return None

    while True:
        print(
            "[eval_async] Label rollout: [s]uccess or [f]ailure: ",
            file=output,
            end="",
            flush=True,
        )
        response = stream.readline()
        if response == "":
            return None
        normalized = response.strip().lower()
        if normalized in {"s", "success"}:
            return "success"
        if normalized in {"f", "failure", "fail"}:
            return "failure"
        print(
            "[eval_async] Enter S for success or F for failure.",
            file=output,
            flush=True,
        )


def write_metadata_atomic(path: Path, payload: Mapping) -> None:
    """Atomically persist rollout metadata without leaving partial JSON."""
    outcome = payload.get("outcome_label")
    if outcome is not None and outcome not in VALID_OUTCOMES:
        raise ValueError(f"Invalid rollout outcome: {outcome!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)
