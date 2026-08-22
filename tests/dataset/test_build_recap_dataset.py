import json
from pathlib import Path
from types import SimpleNamespace

from lerobot.datasets import LeRobotDataset
import numpy as np
import pytest

from models.act.eval.rollout_capture import RolloutVideoRecorder
import scripts.recap.build_dataset as build_dataset_module
from scripts.recap.build_dataset import (
    build_dataset,
    discover_trial_dirs,
    load_trial,
    load_trials,
    recap_rewards,
    resolve_failure_penalty,
)


JOINT_NAMES = ("joint_a.pos", "joint_b.pos")
CAMERA_NAMES = ("top", "wrist")


class _FakeCorrectionDataset:
    """Minimal stand-in for a LeRobot human-correction dataset."""

    meta = SimpleNamespace(
        fps=30,
        features={
            "observation.state": {"names": list(JOINT_NAMES)},
            "observation.images.top": {},
            "observation.images.wrist": {},
        },
    )

    def __getitem__(self, index):
        return {
            "observation.state": np.array([index, index + 1], dtype=np.float32),
            "action": np.array([index + 2, index + 3], dtype=np.float32),
            "observation.images.top": np.full((24, 32, 3), 5, dtype=np.uint8),
            "observation.images.wrist": np.full((24, 32, 3), 6, dtype=np.uint8),
        }


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


def test_image_to_hwc_uint8_matches_video_decoder_layout():
    import torch

    from scripts.recap.build_dataset import _image_to_hwc_uint8

    chw_float = torch.zeros(3, 4, 5)
    chw_float[0] = 1.0
    converted = _image_to_hwc_uint8(chw_float)
    assert converted.shape == (4, 5, 3)
    assert converted.dtype == np.uint8
    assert converted[0, 0, 0] == 255
    assert converted[0, 0, 1] == 0

    already_hwc = np.full((4, 5, 3), 7, dtype=np.uint8)
    assert np.array_equal(_image_to_hwc_uint8(already_hwc), already_hwc)


def test_corrections_are_tagged_and_scored_as_successes(tmp_path):
    _make_trial(tmp_path / "roll", outcome="failure")
    trials = [load_trial(path) for path in discover_trial_dirs([tmp_path / "roll"])]

    # Stand in for a LeRobot correction dataset with two 2-step episodes.
    correction_root = tmp_path / "corrections"
    corrections = [
        build_dataset_module.CorrectionEpisode(
            repo_id="owner/rollout_corr",
            root=correction_root,
            episode_index=index,
            start=index * 2,
            stop=index * 2 + 2,
            task="Place the sponge on the hot pad",
        )
        for index in range(2)
    ]

    output_root = tmp_path / "dataset"
    _, manifest = build_dataset(
        trials=trials,
        target_repo="owner/rollout_recap_merge_test",
        output_root=output_root,
        failure_penalty=None,
        use_videos=False,
        corrections=corrections,
        correction_dataset=_FakeCorrectionDataset(),
    )

    assert manifest["rollout_episodes"] == 1
    assert manifest["correction_episodes"] == 2

    reread = LeRobotDataset("owner/rollout_recap_merge_test", root=output_root)
    assert reread.num_episodes == 3

    rollout_steps = len(trials[0].actions)
    for index in range(rollout_steps):
        assert reread[index]["intervention"].item() is False
    for index in range(rollout_steps, len(reread)):
        assert reread[index]["intervention"].item() is True
        assert reread[index]["success"].item() is True

    # A 2-step success ends at 0 and starts one step below it.
    first_correction = rollout_steps
    assert reread[first_correction]["return_to_go"].item() == -1.0
    assert reread[first_correction + 1]["return_to_go"].item() == 0.0


def test_failure_penalty_clears_the_longest_correction(tmp_path):
    _make_trial(tmp_path / "roll", outcome="success", num_actions=3)
    trials = [load_trial(path) for path in discover_trial_dirs([tmp_path / "roll"])]
    long_correction = [
        build_dataset_module.CorrectionEpisode(
            repo_id="owner/rollout_corr",
            root=tmp_path,
            episode_index=0,
            start=0,
            stop=9,
            task="Place the sponge on the hot pad",
        )
    ]

    captured = {}

    def capture(max_episode_steps, requested):
        captured["max_episode_steps"] = max_episode_steps
        raise RuntimeError("stop after the C_fail decision")

    original = build_dataset_module.resolve_failure_penalty
    build_dataset_module.resolve_failure_penalty = capture
    try:
        with pytest.raises(RuntimeError, match="stop after"):
            build_dataset(
                trials=trials,
                target_repo="owner/rollout_recap_cfail_test",
                output_root=tmp_path / "unused",
                failure_penalty=None,
                use_videos=False,
                corrections=long_correction,
                correction_dataset=_FakeCorrectionDataset(),
            )
    finally:
        build_dataset_module.resolve_failure_penalty = original

    # The 9-step correction, not the 3-step rollout, must set the bound.
    assert captured["max_episode_steps"] == 9


def _frame_kwargs(**overrides):
    defaults = dict(
        qpos=np.asarray([1.0, 2.0], dtype=np.float32),
        action=np.asarray([3.0, 4.0], dtype=np.float32),
        reward=-1.0,
        return_to_go=-2.0,
        terminal=False,
        success=True,
        intervention=False,
        observation_index=7,
        observation_frame_index=9,
        action_index=11,
        task="Place the sponge on the hot pad",
        images={"top": np.zeros((2, 2, 3), dtype=np.uint8)},
    )
    defaults.update(overrides)
    return defaults


def test_recap_frame_uses_one_schema_for_rollouts_and_corrections():
    # Both episode kinds land in one dataset, so a key present on only one of
    # them would surface as a null column rather than an error.
    rollout = build_dataset_module._recap_frame(**_frame_kwargs(intervention=False))
    correction = build_dataset_module._recap_frame(**_frame_kwargs(intervention=True))

    assert set(rollout) == set(correction)
    assert rollout["intervention"].item() is False
    assert correction["intervention"].item() is True


def test_recap_frame_dtypes_are_pinned():
    # LeRobot infers column types from the first frame, so a float64 or a
    # Python bool here silently changes the schema of the whole dataset.
    frame = build_dataset_module._recap_frame(**_frame_kwargs())

    assert frame["reward"].dtype == np.float32
    assert frame["return_to_go"].dtype == np.float32
    assert frame["terminal"].dtype == np.bool_
    assert frame["success"].dtype == np.bool_
    assert frame["intervention"].dtype == np.bool_
    assert frame["source.observation_index"].dtype == np.int64
    assert frame["source.observation_frame_index"].dtype == np.int64
    assert frame["source.action_index"].dtype == np.int64
    assert frame["observation.images.top"].shape == (2, 2, 3)


@pytest.mark.parametrize("field", ["qpos", "action"])
def test_recap_frame_rejects_non_finite_state_and_action(field):
    bad = np.asarray([1.0, float("nan")], dtype=np.float32)
    with pytest.raises(ValueError, match="Non-finite"):
        build_dataset_module._recap_frame(**_frame_kwargs(**{field: bad}))


def test_non_finite_rollout_frame_names_the_trial(tmp_path):
    # The frame builder cannot know which trial it came from; the writer has to
    # add that, or a bad value in one of 60 trials is untraceable.
    trial_dir = _make_trial(tmp_path / "roll", outcome="success")
    trial = load_trial(trial_dir)
    trial.observations[0]["qpos"]["joint_a.pos"] = float("inf")

    with pytest.raises(ValueError, match=str(trial.root)):
        build_dataset_module._write_rollout_episode(
            SimpleNamespace(add_frame=lambda frame: None, save_episode=lambda: None),
            trial,
            episode_index=0,
            camera_names=["top"],
            joint_names=list(JOINT_NAMES),
            failure_penalty=4.0,
        )


def test_both_trial_readers_reject_an_unlabeled_trial_the_same_way(tmp_path):
    # load_trial and _trial_spec each open metadata.json for a different
    # purpose. When the label check was written out twice, the two could drift,
    # and only one of them raising the catchable type would turn
    # --skip-unlabeled into an unhandled crash on that path.
    trial_dir = _make_trial(tmp_path / "roll", outcome="success")
    metadata = json.loads((trial_dir / "metadata.json").read_text())
    del metadata["outcome_label"]
    (trial_dir / "metadata.json").write_text(json.dumps(metadata))

    with pytest.raises(build_dataset_module.UnlabeledRolloutError):
        load_trial(trial_dir)
    with pytest.raises(build_dataset_module.UnlabeledRolloutError):
        build_dataset_module._trial_spec(trial_dir)


def test_unlabeled_is_distinguishable_from_a_corrupt_trial(tmp_path):
    # --skip-unlabeled must skip only missing labels. A trial whose action
    # count disagrees with its metadata is damaged, and silently dropping it
    # would build a dataset short an episode with a misleading reason logged.
    trial_dir = _make_trial(tmp_path / "roll", outcome="success")
    metadata = json.loads((trial_dir / "metadata.json").read_text())
    metadata["action_count"] = 999
    (trial_dir / "metadata.json").write_text(json.dumps(metadata))

    with pytest.raises(ValueError) as caught:
        load_trials([trial_dir], skip_unlabeled=True)

    assert not isinstance(caught.value, build_dataset_module.UnlabeledRolloutError)
    assert "action count disagrees" in str(caught.value)
