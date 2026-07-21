"""Integration tests for data/trim_dataset_noops.py: build a small trimmed dataset from the real
RevanthGundala/011-place-sponge-in-bowl source and verify it against the source dataset directly.

These require network access (HF Hub) and take a couple of minutes (video encoding). Run with:
    uv run python -m pytest tests/test_trim_dataset_noops.py -v -s

Not run as part of a fast/offline unit test suite -- see tests/test_noop_trim.py for the pure,
fast, offline-friendly logic tests.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SOURCE_REPO_ID = "RevanthGundala/011-place-sponge-in-bowl"
OUTPUT_REPO_ID = "RevanthGundala/012-place-sponge-in-bowl-pytest-smoketest"
NOOP_THRESHOLD_DEG = 0.1
NOOP_MARGIN_FRAMES = 4
MAX_EPISODES = 2


@pytest.fixture(scope="module")
def trimmed_dataset_dir(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("trim_dataset_noops_pytest") / "output"
    result = subprocess.run(
        [
            sys.executable, "-m", "data.trim_dataset_noops",
            "--repo-id", SOURCE_REPO_ID,
            "--output-repo-id", OUTPUT_REPO_ID,
            "--output-dir", str(output_dir),
            "--max-episodes", str(MAX_EPISODES),
            "--noop-threshold-deg", str(NOOP_THRESHOLD_DEG),
            "--noop-margin-frames", str(NOOP_MARGIN_FRAMES),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"trim_dataset_noops.py failed (exit {result.returncode})\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    yield output_dir
    shutil.rmtree(output_dir, ignore_errors=True)


@pytest.fixture(scope="module")
def source_ds():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset(SOURCE_REPO_ID, episodes=list(range(MAX_EPISODES)))


@pytest.fixture(scope="module")
def trimmed_ds(trimmed_dataset_dir):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset(OUTPUT_REPO_ID, root=trimmed_dataset_dir)


def _episode_frame_indices(ds, episode_index: int) -> np.ndarray:
    hf = ds.hf_dataset
    ep_idx = np.array(hf["episode_index"])
    frame_idx = np.array(hf["frame_index"])
    idx = np.where(ep_idx == episode_index)[0]
    return idx[np.argsort(frame_idx[idx])]


def test_episode_count_preserved(trimmed_ds):
    assert trimmed_ds.meta.total_episodes == MAX_EPISODES


def test_frame_counts_match_trim_bounds(source_ds, trimmed_ds):
    from data.noop_trim import find_noop_trim_bounds

    for ep in range(MAX_EPISODES):
        source_idx = _episode_frame_indices(source_ds, ep)
        source_actions = np.stack(
            [source_ds[int(i)]["action"].numpy() for i in source_idx]
        ).astype(np.float32)
        start, stop = find_noop_trim_bounds(source_actions, NOOP_THRESHOLD_DEG, NOOP_MARGIN_FRAMES)
        expected_kept = stop - start

        trimmed_idx = _episode_frame_indices(trimmed_ds, ep)
        assert len(trimmed_idx) == expected_kept, (
            f"episode {ep}: expected {expected_kept} kept frames (trim bounds {start}:{stop} "
            f"of {len(source_idx)}), got {len(trimmed_idx)}"
        )


def test_kept_frames_match_source_at_correct_offset(source_ds, trimmed_ds):
    """The trimmed dataset's frame 0 of an episode must equal the SOURCE's frame `start`, not
    some other offset -- this catches any off-by-one/misalignment bug in the conversion."""
    from data.noop_trim import find_noop_trim_bounds

    for ep in range(MAX_EPISODES):
        source_idx = _episode_frame_indices(source_ds, ep)
        source_actions = np.stack(
            [source_ds[int(i)]["action"].numpy() for i in source_idx]
        ).astype(np.float32)
        start, stop = find_noop_trim_bounds(source_actions, NOOP_THRESHOLD_DEG, NOOP_MARGIN_FRAMES)

        trimmed_idx = _episode_frame_indices(trimmed_ds, ep)
        # Spot-check first, middle, and last kept frame of this episode.
        n_kept = stop - start
        for offset in sorted({0, n_kept // 2, n_kept - 1}):
            source_frame = source_ds[int(source_idx[start + offset])]
            trimmed_frame = trimmed_ds[int(trimmed_idx[offset])]
            np.testing.assert_allclose(
                trimmed_frame["observation.state"].numpy(),
                source_frame["observation.state"].numpy(),
                err_msg=f"episode {ep} offset {offset}: state mismatch",
            )
            np.testing.assert_allclose(
                trimmed_frame["action"].numpy(),
                source_frame["action"].numpy(),
                err_msg=f"episode {ep} offset {offset}: action mismatch",
            )
            # Images are re-encoded through lossy AV1 video compression, so allow a small
            # mean-abs-diff tolerance rather than requiring bit-exact equality.
            img_diff = (trimmed_frame["observation.images.top"] - source_frame["observation.images.top"]).abs().mean().item()
            assert img_diff < 0.05, f"episode {ep} offset {offset}: image mean abs diff too large: {img_diff}"


def test_middle_of_episode_never_trimmed(source_ds, trimmed_ds):
    """The trimming logic only removes leading/trailing no-op runs -- verify no frames are
    missing from the interior of any episode (total kept == stop-start exactly, no gaps)."""
    for ep in range(MAX_EPISODES):
        trimmed_idx = _episode_frame_indices(trimmed_ds, ep)
        frame_indices = np.array(trimmed_ds.hf_dataset["frame_index"])[
            np.where(np.array(trimmed_ds.hf_dataset["episode_index"]) == ep)[0]
        ]
        # frame_index within the trimmed episode must be a contiguous 0..N-1 range (no gaps).
        assert sorted(frame_indices.tolist()) == list(range(len(frame_indices)))


def test_task_string_preserved(source_ds, trimmed_ds):
    assert trimmed_ds[0]["task"] == source_ds[0]["task"]


def test_metadata_preserved(source_ds, trimmed_ds):
    assert trimmed_ds.fps == source_ds.fps
    assert trimmed_ds.meta.robot_type == source_ds.meta.robot_type
    assert trimmed_ds.meta.features["observation.state"]["names"] == source_ds.meta.features["observation.state"]["names"]
    assert trimmed_ds.meta.features["action"]["names"] == source_ds.meta.features["action"]["names"]
    source_cameras = sorted(k for k in source_ds.meta.features if k.startswith("observation.images."))
    trimmed_cameras = sorted(k for k in trimmed_ds.meta.features if k.startswith("observation.images."))
    assert trimmed_cameras == source_cameras


def test_no_more_frames_than_source(source_ds, trimmed_ds):
    """Sanity bound: trimming can only remove frames, never add them."""
    assert trimmed_ds.meta.total_frames <= source_ds.meta.total_frames
    assert trimmed_ds.meta.total_frames > 0
