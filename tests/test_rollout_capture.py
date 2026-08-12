import io
import json

import numpy as np
import pytest

import models.act.rollout_capture as rollout_capture
from models.act.rollout_capture import (
    RolloutArtifactWriter,
    RolloutVideoRecorder,
    prompt_for_outcome,
    write_metadata_atomic,
)


class InteractiveStringIO(io.StringIO):
    def isatty(self):
        return True


class FakeVideoWriter:
    def __init__(self, opened=True):
        self.opened = opened
        self.frames = []
        self.released = False

    def isOpened(self):
        return self.opened

    def write(self, frame):
        self.frames.append(frame.copy())

    def release(self):
        self.released = True


def test_outcome_prompt_accepts_success_after_invalid_input():
    output = io.StringIO()

    outcome = prompt_for_outcome(
        stream=InteractiveStringIO("maybe\ns\n"),
        output=output,
    )

    assert outcome == "success"
    assert "Enter S for success or F for failure" in output.getvalue()


def test_outcome_prompt_leaves_noninteractive_run_unlabeled():
    assert prompt_for_outcome(
        stream=io.StringIO("s\n"),
        output=io.StringIO(),
    ) is None


def test_outcome_prompt_refreshes_idle_callback_while_waiting(monkeypatch):
    stream = InteractiveStringIO("s\n")
    waits = iter([([], [], []), ([stream], [], [])])
    monkeypatch.setattr(
        rollout_capture.select,
        "select",
        lambda *_args: next(waits),
    )
    updates = []

    outcome = prompt_for_outcome(
        stream=stream,
        output=io.StringIO(),
        idle_callback=lambda: updates.append("preview"),
    )

    assert outcome == "success"
    assert updates == ["preview", "preview"]


def test_rollout_video_recorder_keeps_camera_frames_synchronized(tmp_path):
    writers = {}

    def writer_factory(path, _fourcc, fps, size):
        writer = FakeVideoWriter()
        writers[path] = (writer, fps, size)
        return writer

    recorder = RolloutVideoRecorder(
        tmp_path / "videos",
        ("top", "wrist"),
        30,
        writer_factory=writer_factory,
    )
    frames = {
        "top": np.zeros((12, 16, 3), dtype=np.uint8),
        "wrist": np.ones((12, 16, 3), dtype=np.uint8),
    }

    assert recorder.write(frames) == 0
    assert recorder.write(frames) == 1
    recorder.close()

    assert recorder.frame_count == 2
    assert len(writers) == 2
    for writer, fps, size in writers.values():
        assert fps == 30
        assert size == (16, 12)
        assert len(writer.frames) == 2
        assert writer.released


def test_rollout_video_recorder_rejects_missing_camera(tmp_path):
    recorder = RolloutVideoRecorder(
        tmp_path / "videos",
        ("top", "wrist"),
        30,
        writer_factory=lambda *_args: FakeVideoWriter(),
    )

    with pytest.raises(KeyError, match="wrist"):
        recorder.write(
            {"top": np.zeros((12, 16, 3), dtype=np.uint8)}
        )


def test_rollout_artifact_writer_aligns_actions_to_latest_observation(tmp_path):
    writers = {}

    def writer_factory(path, _fourcc, _fps, _size):
        writer = FakeVideoWriter()
        writers[path] = writer
        return writer

    artifact = RolloutArtifactWriter(
        tmp_path / "trial-001",
        ("top",),
        30,
        writer_factory=writer_factory,
    )
    frames = {"top": np.zeros((12, 16, 3), dtype=np.uint8)}

    artifact.record_observation(frames, {"qpos": {"joint.pos": 1.0}})
    artifact.record_action({"sent_action": {"joint.pos": 2.0}})
    artifact.record_observation(frames, {"qpos": {"joint.pos": 2.0}})
    artifact.record_action({"sent_action": {"joint.pos": 3.0}})
    artifact.close()
    artifact.close()

    observations = [
        json.loads(line)
        for line in artifact.observations_path.read_text().splitlines()
    ]
    actions = [
        json.loads(line)
        for line in artifact.actions_path.read_text().splitlines()
    ]
    assert [entry["frame_index"] for entry in observations] == [0, 1]
    assert [
        entry["observation_frame_index"]
        for entry in actions
    ] == [0, 1]
    assert artifact.observation_count == 2
    assert artifact.action_count == 2
    assert all(writer.released for writer in writers.values())


def test_rollout_metadata_is_written_atomically(tmp_path):
    path = tmp_path / "metadata.json"

    write_metadata_atomic(
        path,
        {
            "outcome_label": "failure",
            "success": False,
            "steps": 42,
        },
    )

    assert json.loads(path.read_text()) == {
        "outcome_label": "failure",
        "success": False,
        "steps": 42,
    }
    assert not path.with_suffix(".json.tmp").exists()


def test_rollout_metadata_rejects_invalid_outcome(tmp_path):
    with pytest.raises(ValueError, match="Invalid rollout outcome"):
        write_metadata_atomic(
            tmp_path / "metadata.json",
            {"outcome_label": "unknown"},
        )
