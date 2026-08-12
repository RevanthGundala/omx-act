import json
from pathlib import Path

from lerobot.datasets import LeRobotDataset
import numpy as np
import pytest

from models.act.rollout_capture import RolloutVideoRecorder
import models.recap.build_dataset as build_dataset_module
from models.recap.build_dataset import (
    build_dataset,
    discover_trial_dirs,
    load_trial,
    load_trials,
    recap_rewards,
    resolve_failure_penalty,
)


JOINT_NAMES = ("joint_a.pos", "joint_b.pos")
CAMERA_NAMES = ("top", "wrist")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _make_trial(
    root: Path,
    *,
    outcome: str,
    num_actions: int = 3,
) -> Path:
    trial = root / "trial-001"
    trial.mkdir(parents=True)
    recorder = RolloutVideoRecorder(
        trial / "videos",
        CAMERA_NAMES,
        30,
    )
    frames = [
        {
            "top": np.full((24, 32, 3), index * 20, dtype=np.uint8),
            "wrist": np.full((24, 32, 3), 255 - index * 20, dtype=np.uint8),
        }
        for index in range(2)
    ]
    for frame in frames:
        recorder.write(frame)
    recorder.close()

    observations = [
        {
            "observation_index": 0,
            "frame_index": 0,
            "qpos": {"joint_a.pos": 1.0, "joint_b.pos": 2.0},
        },
        {
            "observation_index": 1,
            "frame_index": 1,
            "qpos": {"joint_a.pos": 3.0, "joint_b.pos": 4.0},
        },
    ]
    actions = []
    for index in range(num_actions):
        source_frame = 0 if index < 2 else 1
        actions.append(
            {
                "action_index": index,
                "observation_frame_index": source_frame,
                "sent_action": {
                    "joint_a.pos": 10.0 + index,
                    "joint_b.pos": 20.0 + index,
                },
            }
        )
    _write_jsonl(trial / "observations.jsonl", observations)
    _write_jsonl(trial / "actions.jsonl", actions)
    (trial / "metadata.json").write_text(
        json.dumps(
            {
                "outcome_label": outcome,
                "success": outcome == "success",
                "observation_count": len(observations),
                "action_count": len(actions),
                "camera_names": list(CAMERA_NAMES),
                "joint_names": list(JOINT_NAMES),
                "control_hz": 30,
                "camera_frame_count": len(observations),
                "videos": {
                    camera: f"videos/{camera}.mp4"
                    for camera in CAMERA_NAMES
                },
                "task": "Place the sponge in the bowl",
            }
        ),
        encoding="utf-8",
    )
    return trial


def test_recap_rewards_match_step_and_terminal_definition():
    success_rewards, success_returns = recap_rewards(
        3,
        success=True,
        failure_penalty=4,
    )
    failure_rewards, failure_returns = recap_rewards(
        3,
        success=False,
        failure_penalty=4,
    )

    assert success_rewards.tolist() == [-1.0, -1.0, 0.0]
    assert success_returns.tolist() == [-2.0, -1.0, 0.0]
    assert failure_rewards.tolist() == [-1.0, -1.0, -4.0]
    assert failure_returns.tolist() == [-6.0, -5.0, -4.0]


def test_failure_penalty_rejects_infinity_and_non_dominating_values():
    assert resolve_failure_penalty(8, None) == 9
    with pytest.raises(ValueError, match="finite"):
        resolve_failure_penalty(8, float("inf"))
    with pytest.raises(ValueError, match="greater than"):
        resolve_failure_penalty(8, 8)


def test_unlabeled_trial_is_rejected(tmp_path):
    trial = _make_trial(tmp_path, outcome="success")
    metadata_path = trial / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["outcome_label"] = None
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="not labeled"):
        load_trial(trial)
    with pytest.raises(ValueError, match="No labeled rollout"):
        load_trials([trial], skip_unlabeled=True)


@pytest.mark.parametrize("use_videos", [False, True])
def test_builds_action_granularity_lerobot_dataset(tmp_path, use_videos):
    success_dir = _make_trial(tmp_path / "success", outcome="success")
    failure_dir = _make_trial(tmp_path / "failure", outcome="failure")
    discovered = discover_trial_dirs([tmp_path])
    assert discovered == [failure_dir.resolve(), success_dir.resolve()]
    trials = [load_trial(path) for path in discovered]
    output_root = tmp_path / "dataset"

    target, manifest = build_dataset(
        trials=trials,
        target_repo=f"owner/rollout_recap_test_{int(use_videos)}",
        output_root=output_root,
        failure_penalty=None,
        use_videos=use_videos,
    )

    assert target.num_episodes == 2
    assert manifest["failure_penalty"] == 4
    assert manifest["failure_terminal_reward"] == -4
    reread = LeRobotDataset(
        f"owner/rollout_recap_test_{int(use_videos)}",
        root=output_root,
        return_uint8=True,
    )
    assert len(reread) == 6
    first_episode_rewards = [
        float(reread[index]["reward"].item())
        for index in range(3)
    ]
    first_episode_returns = [
        float(reread[index]["return_to_go"].item())
        for index in range(3)
    ]
    assert first_episode_rewards == [-1.0, -1.0, -4.0]
    assert first_episode_returns == [-6.0, -5.0, -4.0]
    assert reread[0]["observation.state"].tolist() == [1.0, 2.0]
    assert reread[1]["observation.state"].tolist() == [1.0, 2.0]
    assert reread[2]["observation.state"].tolist() == [3.0, 4.0]
    assert reread[2]["terminal"].item() is True
    assert reread[2]["success"].item() is False
    assert (output_root / "recap_manifest.json").is_file()


def test_load_trial_defers_video_decode_by_default(tmp_path):
    trial = _make_trial(tmp_path, outcome="success")

    lazy = load_trial(trial)
    assert lazy.frames is None

    eager = load_trial(trial, decode_frames=True)
    assert eager.frames is not None
    assert set(eager.frames) == set(CAMERA_NAMES)


def test_build_decodes_one_trial_at_a_time(tmp_path, monkeypatch):
    _make_trial(tmp_path / "a", outcome="success")
    _make_trial(tmp_path / "b", outcome="failure")
    trials = [load_trial(path) for path in discover_trial_dirs([tmp_path])]
    assert all(trial.frames is None for trial in trials)

    live_decodes = []
    peak = 0

    real_decode = build_dataset_module.decode_trial_frames

    def counting_decode(trial):
        nonlocal peak
        decoded = real_decode(trial)
        live_decodes.append(trial.root)
        peak = max(peak, len(live_decodes))
        return decoded

    def counting_save(self):
        live_decodes.clear()
        return original_save(self)

    original_save = LeRobotDataset.save_episode
    monkeypatch.setattr(build_dataset_module, "decode_trial_frames", counting_decode)
    monkeypatch.setattr(LeRobotDataset, "save_episode", counting_save)

    build_dataset(
        trials=trials,
        target_repo="owner/rollout_recap_stream_test",
        output_root=tmp_path / "dataset",
        failure_penalty=None,
        use_videos=True,
    )

    assert peak == 1
