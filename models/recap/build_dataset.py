"""Convert labeled ACT evaluation rollouts into a RECAP LeRobot dataset.

Each executed action becomes one dataset transition. It is paired with the most
recent observation that the async policy client sent to the policy server, so
camera frames and observation state are repeated until a newer policy
observation is available.

RECAP describes failed terminal reward as a large negative constant, not a
literal IEEE negative infinity. This converter chooses a finite ``C_fail`` that
is greater than the longest rollout, ensuring every failed return is lower than
every successful return while keeping critic targets finite.

Example:
    uv run python -m models.recap.build_dataset \
      --rollout-root outputs/eval_rollouts \
      --target-repo RevanthGundala/rollout_sponge_recap \
      --output-root outputs/datasets/rollout_sponge_recap
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil

import cv2
from huggingface_hub import get_token
from lerobot.datasets import LeRobotDataset
import numpy as np

from models.act.eval.rollout_capture import VALID_OUTCOMES, write_metadata_atomic


class UnlabeledRolloutError(ValueError):
    """Raised when a recorded trial lacks a binary outcome label."""


@dataclass(frozen=True)
class CorrectionEpisode:
    """One human-teleoperated episode from a LeRobot correction dataset.

    RECAP forces these frames to positive advantage (Sec. IV-B), so they are
    kept distinct from autonomous rollout steps via the ``intervention`` flag.
    """

    repo_id: str
    root: Path
    episode_index: int
    start: int
    stop: int
    task: str

    @property
    def steps(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class RolloutTrial:
    root: Path
    metadata: dict
    observations: tuple[dict, ...]
    actions: tuple[dict, ...]
    frames: dict[str, tuple[np.ndarray, ...]] | None = None

    @property
    def success(self) -> bool:
        return self.metadata["outcome_label"] == "success"

    @property
    def camera_names(self) -> tuple[str, ...]:
        return tuple(self.metadata["camera_names"])

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(self.metadata["joint_names"])


def read_jsonl(path: Path) -> tuple[dict, ...]:
    rows = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object in {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"No records found in {path}")
    return tuple(rows)


def discover_trial_dirs(roots: list[Path]) -> list[Path]:
    discovered = set()
    for root in roots:
        root = root.expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"Rollout root does not exist: {root}")
        candidates = (
            [root / "metadata.json"]
            if (root / "metadata.json").is_file()
            else root.rglob("metadata.json")
        )
        for metadata_path in candidates:
            trial_dir = metadata_path.parent.resolve()
            if (
                (trial_dir / "observations.jsonl").is_file()
                and (trial_dir / "actions.jsonl").is_file()
            ):
                discovered.add(trial_dir)
    if not discovered:
        raise ValueError("No recorded ACT rollout trials were found")
    return sorted(discovered)


def _resolve_trial_file(trial_dir: Path, relative_path: str) -> Path:
    path = (trial_dir / relative_path).resolve()
    if not path.is_relative_to(trial_dir.resolve()):
        raise ValueError(
            f"Rollout metadata path escapes trial directory: {relative_path!r}"
        )
    if not path.is_file():
        raise FileNotFoundError(f"Rollout artifact is missing: {path}")
    return path


def decode_video(path: Path) -> tuple[np.ndarray, ...]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open rollout video: {path}")
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"Rollout video contains no frames: {path}")
    return tuple(frames)


def load_correction_episodes(
    repo_id: str,
    root: Path | None,
) -> tuple[list[CorrectionEpisode], LeRobotDataset]:
    """Read episode boundaries from a LeRobot human-correction dataset."""
    dataset = LeRobotDataset(repo_id, root=root)
    episodes = []
    for row in dataset.meta.episodes:
        tasks = row.get("tasks") or []
        if not tasks:
            raise ValueError(
                f"{repo_id} episode {row['episode_index']} has no task string"
            )
        episodes.append(
            CorrectionEpisode(
                repo_id=repo_id,
                root=Path(dataset.root),
                episode_index=int(row["episode_index"]),
                start=int(row["dataset_from_index"]),
                stop=int(row["dataset_to_index"]),
                task=str(tasks[0]),
            )
        )
    if not episodes:
        raise ValueError(f"{repo_id} contains no episodes")
    return episodes, dataset


def _image_to_hwc_uint8(image) -> np.ndarray:
    """Match the rollout path's HWC uint8 frames.

    LeRobotDataset returns CHW float tensors in [0, 1]; the video decoder used
    for autonomous trials returns HWC uint8, and both must agree before they are
    written into one dataset.
    """
    array = image.numpy() if hasattr(image, "numpy") else np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3-dimensional image, got {array.shape}")
    if array.shape[0] == 3 and array.shape[-1] != 3:
        array = np.transpose(array, (1, 2, 0))
    if array.dtype != np.uint8:
        if array.max() <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0.0, 255.0).round().astype(np.uint8)
    return np.ascontiguousarray(array)


def read_correction_step(
    dataset: LeRobotDataset,
    index: int,
    camera_names: tuple[str, ...],
    joint_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Return (qpos, action, frames) for one correction timestep."""
    sample = dataset[index]
    qpos = np.asarray(sample["observation.state"], dtype=np.float32).reshape(-1)
    action = np.asarray(sample["action"], dtype=np.float32).reshape(-1)
    if qpos.shape[0] != len(joint_names) or action.shape[0] != len(joint_names):
        raise ValueError(
            f"Correction frame {index} has {qpos.shape[0]} state and "
            f"{action.shape[0]} action dimensions; expected {len(joint_names)}"
        )
    frames = {
        camera: _image_to_hwc_uint8(sample[f"observation.images.{camera}"])
        for camera in camera_names
    }
    return qpos, action, frames


def validate_correction_schema(
    dataset: LeRobotDataset,
    *,
    camera_names: tuple[str, ...],
    joint_names: tuple[str, ...],
    fps: int,
    frame_shape: tuple[int, int, int],
) -> None:
    """Reject correction data that cannot share a schema with the rollouts."""
    if int(dataset.meta.fps) != int(fps):
        raise ValueError(
            f"Correction dataset runs at {dataset.meta.fps} Hz; rollouts use {fps} Hz"
        )
    features = dataset.meta.features
    expected_state = list(joint_names)
    actual_state = list(features["observation.state"].get("names") or [])
    if actual_state != expected_state:
        raise ValueError(
            f"Correction joint names {actual_state} do not match rollouts {expected_state}"
        )
    for camera in camera_names:
        key = f"observation.images.{camera}"
        if key not in features:
            raise ValueError(f"Correction dataset is missing camera {camera!r}")
    _, _, frames = read_correction_step(dataset, 0, camera_names, joint_names)
    for camera, frame in frames.items():
        if tuple(frame.shape) != tuple(frame_shape):
            raise ValueError(
                f"Correction camera {camera!r} frame shape {frame.shape} does not "
                f"match rollout shape {frame_shape}"
            )


def _require_outcome_label(metadata: dict, trial_dir: Path) -> str:
    """The operator's success/failure keypress for one rollout.

    This is the only ground truth in the pipeline: it sets the terminal reward,
    and through it every return, critic target, and advantage. There is no
    sensor for "the sponge is on the hot pad". An unlabeled trial is therefore
    skipped or escalated, never defaulted -- guessing "success" would feed a
    fabricated terminal reward of 0 into the critic.

    Raises UnlabeledRolloutError specifically, so callers can skip *this* case
    without also swallowing corrupt videos or malformed indices.
    """
    outcome = metadata.get("outcome_label")
    if outcome not in VALID_OUTCOMES:
        raise UnlabeledRolloutError(
            f"{trial_dir} is not labeled success/failure: {outcome!r}"
        )
    return outcome


def _validate_trial_observations(
    trial_dir: Path,
    observations: tuple[dict, ...],
    joint_names: tuple[str, ...],
    metadata: dict,
) -> dict[int, dict]:
    """Index observations by video frame, rejecting gaps and duplicates.

    Actions reference observations by frame index, so a gap here would later
    surface as an action pointing at nothing.
    """
    observations_by_frame: dict[int, dict] = {}
    for expected_index, observation in enumerate(observations):
        if observation.get("observation_index") != expected_index:
            raise ValueError(f"{trial_dir} observation indices are not contiguous")
        frame_index = observation.get("frame_index")
        if not isinstance(frame_index, int) or frame_index < 0:
            raise ValueError(f"{trial_dir} has an invalid observation frame index")
        if frame_index in observations_by_frame:
            raise ValueError(
                f"{trial_dir} has duplicate observation frame {frame_index}"
            )
        qpos = observation.get("qpos")
        if not isinstance(qpos, dict) or set(qpos) != set(joint_names):
            raise ValueError(
                f"{trial_dir} observation {expected_index} has incompatible qpos"
            )
        observations_by_frame[frame_index] = observation

    if set(observations_by_frame) != set(range(len(observations))):
        raise ValueError(
            f"{trial_dir} observation frame indices must be contiguous "
            f"0-{len(observations) - 1}"
        )
    if metadata.get("camera_frame_count") != len(observations):
        raise ValueError(f"{trial_dir} camera frame count disagrees with observations")
    return observations_by_frame


def _validate_trial_actions(
    trial_dir: Path,
    actions: tuple[dict, ...],
    observations_by_frame: dict[int, dict],
    joint_names: tuple[str, ...],
) -> None:
    """Every action must name the observation the policy actually saw."""
    for expected_index, action in enumerate(actions):
        if action.get("action_index") != expected_index:
            raise ValueError(f"{trial_dir} action indices are not contiguous")
        frame_index = action.get("observation_frame_index")
        if frame_index not in observations_by_frame:
            raise ValueError(
                f"{trial_dir} action {expected_index} references unknown "
                f"observation frame {frame_index!r}"
            )
        sent_action = action.get("sent_action")
        if not isinstance(sent_action, dict) or set(sent_action) != set(joint_names):
            raise ValueError(
                f"{trial_dir} action {expected_index} has incompatible sent_action"
            )


def _validate_trial_videos(
    trial_dir: Path, metadata: dict, camera_names: tuple[str, ...]
) -> None:
    videos = metadata.get("videos")
    if not isinstance(videos, dict) or set(videos) != set(camera_names):
        raise ValueError(f"{trial_dir} video metadata does not match camera_names")
    for camera in camera_names:
        _resolve_trial_file(trial_dir, videos[camera])


def load_trial(trial_dir: Path, *, decode_frames: bool = False) -> RolloutTrial:
    """Read one recorded rollout directory.

    The trial was written by a separate process, on separate hardware, in the
    omx-training repo, so everything it claims is checked before use. Frames
    are left on disk unless ``decode_frames`` is set: decoding every trial up
    front once needed 27.8 GB.
    """
    trial_dir = trial_dir.resolve()
    metadata = json.loads((trial_dir / "metadata.json").read_text())
    _require_outcome_label(metadata, trial_dir)

    camera_names = tuple(metadata.get("camera_names") or ())
    joint_names = tuple(metadata.get("joint_names") or ())
    if not camera_names or len(set(camera_names)) != len(camera_names):
        raise ValueError(f"{trial_dir} has invalid camera_names")
    if not joint_names or len(set(joint_names)) != len(joint_names):
        raise ValueError(f"{trial_dir} has invalid joint_names")

    observations = read_jsonl(trial_dir / "observations.jsonl")
    actions = read_jsonl(trial_dir / "actions.jsonl")
    if len(observations) != metadata.get("observation_count"):
        raise ValueError(f"{trial_dir} observation count disagrees with metadata")
    if len(actions) != metadata.get("action_count"):
        raise ValueError(f"{trial_dir} action count disagrees with metadata")

    observations_by_frame = _validate_trial_observations(
        trial_dir, observations, joint_names, metadata
    )
    _validate_trial_actions(trial_dir, actions, observations_by_frame, joint_names)
    _validate_trial_videos(trial_dir, metadata, camera_names)

    return RolloutTrial(
        root=trial_dir,
        metadata=metadata,
        observations=observations,
        actions=actions,
        frames=decode_trial_frames_from_parts(
            trial_dir,
            metadata,
            camera_names,
            len(observations),
        )
        if decode_frames
        else None,
    )


def decode_trial_frames_from_parts(
    trial_dir: Path,
    metadata: dict,
    camera_names: tuple[str, ...],
    expected_frames: int,
) -> dict[str, tuple[np.ndarray, ...]]:
    """Decode every camera video, enforcing the observation frame count."""
    videos = metadata["videos"]
    frames = {}
    for camera in camera_names:
        camera_frames = decode_video(_resolve_trial_file(trial_dir, videos[camera]))
        if len(camera_frames) != expected_frames:
            raise ValueError(
                f"{trial_dir} camera {camera!r} has {len(camera_frames)} frames; "
                f"expected {expected_frames}"
            )
        frames[camera] = camera_frames
    return frames


def decode_trial_frames(trial: RolloutTrial) -> dict[str, tuple[np.ndarray, ...]]:
    """Return cached frames when present, otherwise decode them on demand."""
    if trial.frames is not None:
        return trial.frames
    return decode_trial_frames_from_parts(
        trial.root,
        trial.metadata,
        trial.camera_names,
        len(trial.observations),
    )


def probe_camera_frame_shapes(trial: RolloutTrial) -> dict[str, tuple[int, ...]]:
    """Read one frame per camera so schema checks stay memory-cheap."""
    if trial.frames is not None:
        return {
            camera: tuple(frames[0].shape)
            for camera, frames in trial.frames.items()
        }
    shapes = {}
    for camera in trial.camera_names:
        path = _resolve_trial_file(trial.root, trial.metadata["videos"][camera])
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Could not open rollout video: {path}")
        try:
            ok, frame = capture.read()
        finally:
            capture.release()
        if not ok:
            raise RuntimeError(f"Rollout video has no frames: {path}")
        shapes[camera] = tuple(frame.shape)
    return shapes


def load_trials(
    trial_dirs: list[Path],
    *,
    skip_unlabeled: bool,
) -> list[RolloutTrial]:
    trials = []
    for trial_dir in trial_dirs:
        try:
            trials.append(load_trial(trial_dir))
        except UnlabeledRolloutError as exc:
            if not skip_unlabeled:
                raise
            print(f"Skipping unlabeled rollout {trial_dir}: {exc}")
    if not trials:
        raise ValueError("No labeled rollout trials remain")
    return trials


def resolve_failure_penalty(
    max_episode_steps: int,
    requested: float | None,
) -> float:
    if max_episode_steps <= 0:
        raise ValueError("max_episode_steps must be positive")
    failure_penalty = (
        float(max_episode_steps + 1)
        if requested is None
        else float(requested)
    )
    if not math.isfinite(failure_penalty):
        raise ValueError("failure penalty must be finite")
    if failure_penalty <= max_episode_steps:
        raise ValueError(
            "failure penalty must be greater than the longest rollout "
            f"({max_episode_steps})"
        )
    return failure_penalty


def recap_rewards(
    num_steps: int,
    *,
    success: bool,
    failure_penalty: float,
) -> tuple[np.ndarray, np.ndarray]:
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if not math.isfinite(failure_penalty) or failure_penalty <= 0:
        raise ValueError("failure_penalty must be positive and finite")
    rewards = np.full(num_steps, -1.0, dtype=np.float32)
    rewards[-1] = 0.0 if success else -failure_penalty
    returns_to_go = np.cumsum(rewards[::-1], dtype=np.float64)[::-1].astype(
        np.float32
    )
    if not np.isfinite(returns_to_go).all():
        raise ValueError("RECAP returns must be finite")
    return rewards, returns_to_go


def validate_compatible_trials(trials: list[RolloutTrial]) -> tuple:
    first = trials[0]
    first_shapes = probe_camera_frame_shapes(first)
    first_frame_shape = first_shapes[first.camera_names[0]]
    signature = (
        first.camera_names,
        first.joint_names,
        int(first.metadata["control_hz"]),
        first_frame_shape,
    )
    for trial in trials[1:]:
        shapes = probe_camera_frame_shapes(trial)
        candidate = (
            trial.camera_names,
            trial.joint_names,
            int(trial.metadata["control_hz"]),
            shapes[trial.camera_names[0]],
        )
        if candidate != signature:
            raise ValueError(
                f"Incompatible rollout schemas: {first.root} and {trial.root}"
            )
        for camera in trial.camera_names:
            if shapes[camera] != first_frame_shape:
                raise ValueError(
                    f"Camera frame shapes differ in {trial.root}"
                )
    return signature


def build_features(
    *,
    camera_names: tuple[str, ...],
    joint_names: tuple[str, ...],
    frame_shape: tuple[int, int, int],
    use_videos: bool,
) -> dict:
    height, width, channels = frame_shape
    if channels != 3:
        raise ValueError(f"Expected RGB camera frames, got {frame_shape}")
    image_dtype = "video" if use_videos else "image"
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(joint_names),),
            "names": list(joint_names),
        },
        "action": {
            "dtype": "float32",
            "shape": (len(joint_names),),
            "names": list(joint_names),
        },
        "reward": {"dtype": "float32", "shape": (1,), "names": None},
        "return_to_go": {
            "dtype": "float32",
            "shape": (1,),
            "names": None,
        },
        "terminal": {"dtype": "bool", "shape": (1,), "names": None},
        "success": {"dtype": "bool", "shape": (1,), "names": None},
        "intervention": {"dtype": "bool", "shape": (1,), "names": None},
        "source.observation_index": {
            "dtype": "int64",
            "shape": (1,),
            "names": None,
        },
        "source.observation_frame_index": {
            "dtype": "int64",
            "shape": (1,),
            "names": None,
        },
        "source.action_index": {
            "dtype": "int64",
            "shape": (1,),
            "names": None,
        },
    }
    for camera in camera_names:
        features[f"observation.images.{camera}"] = {
            "dtype": image_dtype,
            "shape": (3, height, width),
            "names": ["channels", "height", "width"],
        }
    return features


def _recap_frame(
    *,
    qpos: np.ndarray,
    action: np.ndarray,
    reward: float,
    return_to_go: float,
    terminal: bool,
    success: bool,
    intervention: bool,
    observation_index: int,
    observation_frame_index: int,
    action_index: int,
    task: str,
    images: dict[str, np.ndarray],
) -> dict:
    """One LeRobot frame in the RECAP schema.

    Rollout and correction episodes carry the same keys so a single dataset can
    hold both; they differ only in ``intervention`` and in whether the source
    indices point back into a rollout trial or a correction episode.
    """
    if not np.isfinite(qpos).all() or not np.isfinite(action).all():
        raise ValueError("Non-finite state/action")
    frame = {
        "observation.state": qpos,
        "action": action,
        "reward": np.asarray([reward], dtype=np.float32),
        "return_to_go": np.asarray([return_to_go], dtype=np.float32),
        "terminal": np.asarray([terminal], dtype=np.bool_),
        "success": np.asarray([success], dtype=np.bool_),
        "intervention": np.asarray([intervention], dtype=np.bool_),
        "source.observation_index": np.asarray([observation_index], dtype=np.int64),
        "source.observation_frame_index": np.asarray(
            [observation_frame_index], dtype=np.int64
        ),
        "source.action_index": np.asarray([action_index], dtype=np.int64),
        "task": task,
    }
    for camera, image in images.items():
        frame[f"observation.images.{camera}"] = image
    return frame


def _write_rollout_episode(
    target: LeRobotDataset,
    trial: RolloutTrial,
    *,
    episode_index: int,
    camera_names: list[str],
    joint_names: list[str],
    failure_penalty: float,
) -> dict:
    """Append one autonomous rollout to ``target`` and return its summary."""
    trial_frames = decode_trial_frames(trial)
    rewards, returns_to_go = recap_rewards(
        len(trial.actions),
        success=trial.success,
        failure_penalty=failure_penalty,
    )
    observations_by_frame = {
        observation["frame_index"]: observation for observation in trial.observations
    }
    for step, action in enumerate(trial.actions):
        source_frame = action["observation_frame_index"]
        observation = observations_by_frame[source_frame]
        qpos = np.asarray(
            [observation["qpos"][name] for name in joint_names], dtype=np.float32
        )
        executed_action = np.asarray(
            [action["sent_action"][name] for name in joint_names], dtype=np.float32
        )
        try:
            frame = _recap_frame(
                qpos=qpos,
                action=executed_action,
                reward=rewards[step],
                return_to_go=returns_to_go[step],
                terminal=step == len(trial.actions) - 1,
                success=trial.success,
                # An autonomous evaluation rollout has no human actions in it.
                intervention=False,
                observation_index=observation["observation_index"],
                observation_frame_index=source_frame,
                action_index=action["action_index"],
                task=trial.metadata["task"],
                images={
                    camera: trial_frames[camera][source_frame]
                    for camera in camera_names
                },
            )
        except ValueError as error:
            raise ValueError(
                f"Non-finite state/action in {trial.root} at action {step}"
            ) from error
        target.add_frame(frame)
    target.save_episode()
    del trial_frames
    return {
        "episode_index": episode_index,
        "source_trial": str(trial.root),
        "outcome_label": trial.metadata["outcome_label"],
        "steps": len(trial.actions),
        "initial_return": float(returns_to_go[0]),
        "terminal_reward": float(rewards[-1]),
        "intervention": False,
    }


def _write_correction_episode(
    target: LeRobotDataset,
    episode: CorrectionEpisode,
    correction_dataset: LeRobotDataset,
    *,
    episode_index: int,
    camera_names: list[str],
    joint_names: list[str],
    failure_penalty: float,
    succeeded: bool,
) -> dict:
    """Append one teleop correction to ``target`` and return its summary."""
    rewards, returns_to_go = recap_rewards(
        episode.steps,
        success=succeeded,
        failure_penalty=failure_penalty,
    )
    for step, index in enumerate(range(episode.start, episode.stop)):
        qpos, executed_action, images = read_correction_step(
            correction_dataset, index, camera_names, joint_names
        )
        try:
            frame = _recap_frame(
                qpos=qpos,
                action=executed_action,
                reward=rewards[step],
                return_to_go=returns_to_go[step],
                terminal=step == episode.steps - 1,
                success=succeeded,
                # RECAP Sec. IV-B forces human corrections to positive
                # advantage, so the labeler must be able to find them.
                intervention=True,
                observation_index=step,
                observation_frame_index=step,
                action_index=step,
                task=episode.task,
                images={camera: images[camera] for camera in camera_names},
            )
        except ValueError as error:
            raise ValueError(
                f"Non-finite state/action in correction episode "
                f"{episode.episode_index} at step {step}"
            ) from error
        target.add_frame(frame)
    target.save_episode()
    return {
        "episode_index": episode_index,
        "source_trial": f"{episode.repo_id}#episode_{episode.episode_index}",
        "outcome_label": "success" if succeeded else "failure",
        "steps": episode.steps,
        "initial_return": float(returns_to_go[0]),
        "terminal_reward": float(rewards[-1]),
        "intervention": True,
    }


def _recap_manifest(
    *,
    target_repo: str,
    output_root: Path,
    failure_penalty: float,
    corrections: list[CorrectionEpisode],
    rollout_episodes: int,
    max_episode_steps: int,
    episode_summaries: list[dict],
) -> dict:
    return {
        "schema_version": 1,
        "target_repo": target_repo,
        "output_root": str(output_root),
        "transition_granularity": "executed_action",
        "observation_alignment": (
            "latest async policy observation; image/state repeated until the "
            "next policy observation"
        ),
        "step_reward": -1.0,
        "success_terminal_reward": 0.0,
        "failure_terminal_reward": -failure_penalty,
        "failure_penalty": failure_penalty,
        "failure_penalty_scope": (
            "derived for this build; pass --failure-penalty explicitly when "
            "pooling independently-built datasets"
        ),
        "failure_value_semantics": (
            "finite surrogate for conceptual negative infinity; chosen above "
            "the longest rollout to keep all critic targets finite"
        ),
        "return_target": "undiscounted Monte Carlo return_to_go",
        "correction_repo": corrections[0].repo_id if corrections else None,
        "correction_episodes": len(corrections),
        "rollout_episodes": rollout_episodes,
        "correction_advantage": (
            "intervention=True frames are forced to positive advantage per "
            "RECAP Sec. IV-B; the critic is not consulted for them"
        ),
        "max_episode_steps": max_episode_steps,
        "episodes": episode_summaries,
    }


def build_dataset(
    *,
    trials: list[RolloutTrial],
    target_repo: str,
    output_root: Path,
    failure_penalty: float | None,
    use_videos: bool,
    corrections: list[CorrectionEpisode] | None = None,
    correction_dataset: LeRobotDataset | None = None,
    corrections_succeeded: bool = True,
) -> tuple[LeRobotDataset, dict]:
    if not trials:
        raise ValueError("At least one rollout trial is required")
    corrections = list(corrections or [])
    if corrections and correction_dataset is None:
        raise ValueError("correction_dataset is required when corrections are given")
    camera_names, joint_names, fps, frame_shape = validate_compatible_trials(trials)
    if corrections:
        validate_correction_schema(
            correction_dataset,
            camera_names=camera_names,
            joint_names=joint_names,
            fps=fps,
            frame_shape=frame_shape,
        )
    # C_fail must clear the longest episode from either source, or a long
    # correction could outrank a short failure.
    max_episode_steps = max(
        [len(trial.actions) for trial in trials]
        + [episode.steps for episode in corrections]
    )
    failure_penalty = resolve_failure_penalty(max_episode_steps, failure_penalty)
    target = LeRobotDataset.create(
        repo_id=target_repo,
        fps=fps,
        features=build_features(
            camera_names=camera_names,
            joint_names=joint_names,
            frame_shape=frame_shape,
            use_videos=use_videos,
        ),
        root=output_root,
        robot_type="omx_follower",
        use_videos=use_videos,
        image_writer_threads=len(camera_names),
    )

    episode_summaries = [
        _write_rollout_episode(
            target,
            trial,
            episode_index=episode_index,
            camera_names=camera_names,
            joint_names=joint_names,
            failure_penalty=failure_penalty,
        )
        for episode_index, trial in enumerate(trials)
    ]
    episode_summaries += [
        _write_correction_episode(
            target,
            episode,
            correction_dataset,
            episode_index=len(trials) + offset,
            camera_names=camera_names,
            joint_names=joint_names,
            failure_penalty=failure_penalty,
            succeeded=corrections_succeeded,
        )
        for offset, episode in enumerate(corrections)
    ]
    target.finalize()

    manifest = _recap_manifest(
        target_repo=target_repo,
        output_root=output_root,
        failure_penalty=failure_penalty,
        corrections=corrections,
        rollout_episodes=len(trials),
        max_episode_steps=max_episode_steps,
        episode_summaries=episode_summaries,
    )
    write_metadata_atomic(output_root / "recap_manifest.json", manifest)
    return target, manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rollout-root",
        type=Path,
        action="append",
        required=True,
        help=(
            "Recorded trial/session/root directory. Repeat to combine multiple "
            "rollout roots."
        ),
    )
    parser.add_argument("--target-repo", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--failure-penalty",
        type=float,
        default=None,
        help=(
            "Positive C_fail magnitude. Defaults to longest rollout + 1 and "
            "must remain finite and greater than the longest rollout."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-unlabeled",
        action="store_true",
        help=(
            "Skip trials without success/failure metadata. By default any "
            "unlabeled trial aborts the build."
        ),
    )
    parser.add_argument("--no-videos", action="store_true")
    parser.add_argument(
        "--correction-repo",
        default=None,
        help=(
            "LeRobot repo id of human-correction episodes to merge in. Their "
            "frames are tagged intervention=True and forced to positive "
            "advantage during RECAP policy training."
        ),
    )
    parser.add_argument(
        "--correction-root",
        type=Path,
        default=None,
        help="Local root for --correction-repo.",
    )
    parser.add_argument(
        "--corrections-failed",
        action="store_true",
        help=(
            "Label correction episodes as failures. Only use this when the "
            "operator did not finish the task; RECAP assumes corrections "
            "complete it."
        ),
    )
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--private", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output root already exists: {output_root}; pass --overwrite"
            )
        shutil.rmtree(output_root)

    trial_dirs = discover_trial_dirs(args.rollout_root)
    trials = load_trials(
        trial_dirs,
        skip_unlabeled=args.skip_unlabeled,
    )
    corrections: list[CorrectionEpisode] = []
    correction_dataset = None
    if args.correction_repo:
        corrections, correction_dataset = load_correction_episodes(
            args.correction_repo,
            args.correction_root,
        )
        print(
            f"Merging {len(corrections)} correction episodes from "
            f"{args.correction_repo}"
        )
    elif args.correction_root:
        raise ValueError("--correction-root requires --correction-repo")

    target, manifest = build_dataset(
        trials=trials,
        target_repo=args.target_repo,
        output_root=output_root,
        failure_penalty=args.failure_penalty,
        use_videos=not args.no_videos,
        corrections=corrections,
        correction_dataset=correction_dataset,
        corrections_succeeded=not args.corrections_failed,
    )
    print(
        f"Built {target.num_episodes} RECAP episodes at {target.root}; "
        f"C_fail={manifest['failure_penalty']:g}"
    )

    reread = LeRobotDataset(
        args.target_repo,
        root=output_root,
        download_videos=False,
    )
    expected_episodes = len(trials) + len(corrections)
    if reread.num_episodes != expected_episodes:
        raise RuntimeError(
            f"Read-back episode mismatch: {reread.num_episodes} vs {expected_episodes}"
        )
    print(f"Read-back ok: {reread.num_episodes} episodes, {len(reread)} frames")

    if args.push:
        if not get_token():
            raise RuntimeError(
                "No Hugging Face token found. Run huggingface-cli login or set HF_TOKEN."
            )
        target.push_to_hub(private=args.private)
        print(
            f"Pushed dataset to https://huggingface.co/datasets/{args.target_repo}"
        )


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Mixed-source builds
# ---------------------------------------------------------------------------
#
# RECAP finetunes the value function on "all of the data collected for the task
# so far" (arXiv:2511.14759 Sec. V-D). Here that spans two storage formats:
# autonomous evaluation rollouts are trial directories, while demonstrations and
# DAgger corrections are LeRobot datasets. Episodes are described first and
# decoded one at a time, because the longest episode sets C_fail before anything
# is written and holding every episode's frames would cost tens of gigabytes.


@dataclass
class EpisodeSpec:
    """One episode, described cheaply and loaded on demand."""

    source_id: str
    source_kind: str
    length: int
    success: bool
    task: str
    joint_names: tuple[str, ...]
    camera_names: tuple[str, ...]
    frame_shape: tuple[int, int, int]
    fps: int
    load: object


def _trial_spec(trial_dir: Path) -> EpisodeSpec:
    metadata = json.loads((trial_dir / "metadata.json").read_text())
    outcome = _require_outcome_label(metadata, trial_dir)

    def load():
        trial = load_trial(trial_dir)
        observations_by_frame = {
            observation["frame_index"]: observation
            for observation in trial.observations
        }
        joint_names = trial.joint_names
        states, actions, images = [], [], {c: [] for c in trial.camera_names}
        for action in trial.actions:
            source_frame = action["observation_frame_index"]
            observation = observations_by_frame[source_frame]
            states.append(
                [observation["qpos"][name] for name in joint_names]
            )
            actions.append(
                [action["sent_action"][name] for name in joint_names]
            )
            for camera in trial.camera_names:
                images[camera].append(trial.frames[camera][source_frame])
        return (
            np.asarray(states, dtype=np.float32),
            np.asarray(actions, dtype=np.float32),
            {c: np.stack(v) for c, v in images.items()},
            np.zeros(len(trial.actions), dtype=np.bool_),
        )

    return EpisodeSpec(
        source_id=str(trial_dir),
        source_kind="autonomous_rollout",
        length=int(metadata["action_count"]),
        success=outcome == "success",
        task=metadata["task"],
        joint_names=tuple(metadata["joint_names"]),
        camera_names=tuple(metadata["camera_names"]),
        frame_shape=(0, 0, 3),
        fps=int(metadata["control_hz"]),
        load=load,
    )


def _lerobot_specs(root: Path, *, kind: str, default_success: bool | None):
    from models.recap.lerobot_source import (
        load_lerobot_episode,
        scan_lerobot_dataset,
    )

    specs = []
    for ref in scan_lerobot_dataset(root, default_success=default_success):
        def load(ref=ref):
            episode = load_lerobot_episode(ref)
            return (
                episode.states,
                episode.actions,
                episode.images,
                episode.intervention,
            )

        specs.append(
            EpisodeSpec(
                source_id=ref.source_id,
                source_kind=kind,
                length=ref.length,
                success=ref.success,
                task=ref.task,
                joint_names=ref.joint_names,
                camera_names=ref.camera_names,
                frame_shape=ref.frame_shape,
                fps=ref.fps,
                load=load,
            )
        )
    return specs


def validate_compatible_specs(specs: list[EpisodeSpec]) -> tuple:
    first = specs[0]
    for spec in specs[1:]:
        if spec.joint_names != first.joint_names:
            raise ValueError(
                f"Joint names differ: {first.source_id} has {first.joint_names}, "
                f"{spec.source_id} has {spec.joint_names}"
            )
        if set(spec.camera_names) != set(first.camera_names):
            raise ValueError(
                f"Camera names differ: {first.source_id} has "
                f"{first.camera_names}, {spec.source_id} has {spec.camera_names}"
            )
        if spec.fps != first.fps:
            raise ValueError(
                f"Control rates differ: {first.fps} vs {spec.fps} "
                f"({spec.source_id})"
            )
    return first.camera_names, first.joint_names, first.fps


def build_mixed_dataset(
    *,
    specs: list[EpisodeSpec],
    target_repo: str,
    output_root: Path,
    failure_penalty: float | None,
    use_videos: bool,
    frame_shape: tuple[int, int, int] | None = None,
) -> tuple[LeRobotDataset, dict]:
    if not specs:
        raise ValueError("At least one episode is required")
    camera_names, joint_names, fps = validate_compatible_specs(specs)
    max_episode_steps = max(spec.length for spec in specs)
    failure_penalty = resolve_failure_penalty(max_episode_steps, failure_penalty)

    if frame_shape is None:
        frame_shape = next(
            (s.frame_shape for s in specs if s.frame_shape[0]),
            None,
        )
        if frame_shape is None:
            # Only trial sources are present, whose shape is known once the
            # first episode is decoded.
            _, _, images, _ = specs[0].load()
            frame_shape = images[camera_names[0]].shape[1:]

    features = build_features(
        camera_names=camera_names,
        joint_names=joint_names,
        frame_shape=tuple(frame_shape),
        use_videos=use_videos,
    )
    target = LeRobotDataset.create(
        repo_id=target_repo,
        fps=fps,
        features=features,
        root=output_root,
        robot_type="omx_follower",
        use_videos=use_videos,
        image_writer_threads=len(camera_names),
    )

    episode_summaries = []
    for episode_index, spec in enumerate(specs):
        states, actions, images, intervention = spec.load()
        if len(states) != spec.length:
            raise ValueError(
                f"{spec.source_id} declared {spec.length} steps but loaded "
                f"{len(states)}"
            )
        rewards, returns_to_go = recap_rewards(
            spec.length,
            success=spec.success,
            failure_penalty=failure_penalty,
        )
        for step in range(spec.length):
            frame = {
                "observation.state": states[step],
                "action": actions[step],
                "reward": np.asarray([rewards[step]], dtype=np.float32),
                "return_to_go": np.asarray(
                    [returns_to_go[step]], dtype=np.float32
                ),
                "terminal": np.asarray([step == spec.length - 1], dtype=np.bool_),
                "success": np.asarray([spec.success], dtype=np.bool_),
                "intervention": np.asarray(
                    [bool(intervention[step])], dtype=np.bool_
                ),
                "source.observation_index": np.asarray([step], dtype=np.int64),
                "source.observation_frame_index": np.asarray(
                    [step], dtype=np.int64
                ),
                "source.action_index": np.asarray([step], dtype=np.int64),
                "task": spec.task,
            }
            for camera in camera_names:
                frame[f"observation.images.{camera}"] = images[camera][step]
            target.add_frame(frame)
        target.save_episode()
        episode_summaries.append(
            {
                "episode_index": episode_index,
                "source_trial": spec.source_id,
                "source_kind": spec.source_kind,
                "outcome_label": "success" if spec.success else "failure",
                "steps": spec.length,
                "intervention_steps": int(np.count_nonzero(intervention)),
                "initial_return": float(returns_to_go[0]),
                "terminal_reward": float(rewards[-1]),
            }
        )
        # Release the episode's pixels before decoding the next one.
        del states, actions, images, intervention
        print(
            f"  [{episode_index + 1}/{len(specs)}] {spec.source_kind}: "
            f"{spec.length} steps, "
            f"{'success' if spec.success else 'failure'}"
        )
    target.finalize()

    successes = sum(1 for spec in specs if spec.success)
    manifest = {
        "schema_version": 2,
        "target_repo": target_repo,
        "output_root": str(output_root),
        "transition_granularity": "executed_action",
        "observation_alignment": (
            "latest async policy observation for rollout trials; one stored "
            "frame per transition for LeRobot sources"
        ),
        "step_reward": -1.0,
        "success_terminal_reward": 0.0,
        "failure_terminal_reward": -failure_penalty,
        "failure_penalty": failure_penalty,
        "failure_penalty_scope": (
            "derived for this build; pass --failure-penalty explicitly when "
            "pooling independently-built datasets"
        ),
        "failure_value_semantics": (
            "finite surrogate for conceptual negative infinity; chosen above "
            "the longest rollout to keep all critic targets finite"
        ),
        "return_target": "undiscounted Monte Carlo return_to_go",
        "max_episode_steps": max_episode_steps,
        "episode_count": len(specs),
        "success_count": successes,
        "failure_count": len(specs) - successes,
        "source_kind_counts": {
            kind: sum(1 for spec in specs if spec.source_kind == kind)
            for kind in sorted({spec.source_kind for spec in specs})
        },
        "episodes": episode_summaries,
    }
    write_metadata_atomic(output_root / "recap_manifest.json", manifest)
    return target, manifest
