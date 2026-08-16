"""Unit tests for dataset_tools.dagger.validators (pure numpy/list helpers, no dataset I/O)."""
from __future__ import annotations

import numpy as np
import pytest

from dataset_tools.dagger.validators import (
    EXCLUDED_CORRECTION_EPISODES,
    select_correction_episodes,
    validate_action_maxima_within_seed,
    validate_contiguous_episode_indices,
    validate_exact_fps,
    validate_finite,
    validate_frame_correspondence,
    validate_no_unexpected_idle_edges,
)


# --- select_correction_episodes -------------------------------------------------


def test_select_correction_episodes_default_excludes_nothing():
    # The 016 corrections source has no known-bad episode (independently
    # audited frame-by-frame), so the default exclusion set is empty and all
    # 18 candidates are kept.
    episodes = select_correction_episodes(range(18))
    assert episodes == list(range(18))
    assert EXCLUDED_CORRECTION_EPISODES == frozenset()


def test_select_correction_episodes_custom_exclusion():
    episodes = select_correction_episodes([1, 2, 3, 4], excluded=frozenset({2, 4}))
    assert episodes == [1, 3]


def test_select_correction_episodes_empty_when_all_excluded():
    assert select_correction_episodes([5], excluded=frozenset({5})) == []


# --- validate_contiguous_episode_indices ----------------------------------------


def test_contiguous_episode_indices_passes():
    episode_indices = np.array([0, 0, 0, 1, 1, 2, 2, 2])
    validate_contiguous_episode_indices(episode_indices, expected_total=3)


def test_contiguous_episode_indices_detects_gap():
    episode_indices = np.array([0, 0, 2, 2])
    with pytest.raises(AssertionError, match="Non-contiguous"):
        validate_contiguous_episode_indices(episode_indices, expected_total=3)


def test_contiguous_episode_indices_detects_wrong_total():
    episode_indices = np.array([0, 0, 1, 1])
    with pytest.raises(AssertionError, match="Expected contiguous"):
        validate_contiguous_episode_indices(episode_indices, expected_total=3)


def test_contiguous_episode_indices_detects_interleaving():
    # Episode 0 frames are split by an episode 1 frame -- not contiguous.
    episode_indices = np.array([0, 1, 0])
    with pytest.raises(AssertionError, match="not contiguous"):
        validate_contiguous_episode_indices(episode_indices, expected_total=2)


# --- validate_finite -------------------------------------------------------------


def test_validate_finite_passes_on_clean_array():
    validate_finite(np.array([1.0, 2.0, -3.5]), name="action")


def test_validate_finite_detects_nan():
    with pytest.raises(AssertionError, match="action"):
        validate_finite(np.array([1.0, float("nan")]), name="action")


def test_validate_finite_detects_inf():
    with pytest.raises(AssertionError):
        validate_finite(np.array([1.0, float("inf")]), name="state")


# --- validate_exact_fps ----------------------------------------------------------


def test_validate_exact_fps_passes():
    timestamps = np.arange(0, 10) / 30.0
    validate_exact_fps(timestamps, [(0, 10)], fps=30)


def test_validate_exact_fps_detects_dropped_frame():
    timestamps = np.array([0 / 30, 1 / 30, 3 / 30, 4 / 30])  # frame 2 missing
    with pytest.raises(AssertionError, match="timestamp delta"):
        validate_exact_fps(timestamps, [(0, 4)], fps=30)


def test_validate_exact_fps_respects_episode_boundaries():
    # Two independent episodes concatenated; each internally exact 30Hz, and
    # the (legitimate) jump between episode 0's last ts and episode 1's first
    # ts must NOT be checked since it crosses a boundary.
    ep0 = np.arange(0, 5) / 30.0
    ep1 = np.arange(100, 105) / 30.0  # unrelated large jump vs ep0's tail
    timestamps = np.concatenate([ep0, ep1])
    validate_exact_fps(timestamps, [(0, 5), (5, 10)], fps=30)


# --- validate_action_maxima_within_seed -----------------------------------------


def test_action_maxima_within_seed_passes():
    seed = np.array([[10.0, -20.0], [5.0, 15.0]])
    correction = np.array([[8.0, -18.0], [4.0, 12.0]])
    validate_action_maxima_within_seed(seed, correction)


def test_action_maxima_within_seed_detects_violation():
    seed = np.array([[10.0, -20.0], [5.0, 15.0]])
    correction = np.array([[8.0, -25.0], [4.0, 12.0]])  # dim 1 exceeds seed's 20 by 5, past the 1.0 default tolerance
    with pytest.raises(AssertionError, match="dims"):
        validate_action_maxima_within_seed(seed, correction)


def test_action_maxima_within_seed_allows_small_tolerance():
    # A genuine corrective hold just past the seed envelope (e.g. 0.22 degrees,
    # as observed on the real 016 corrections source) must pass under the
    # default 1.0 degree tolerance.
    seed = np.array([[10.0, -20.0]])
    correction = np.array([[10.2, -20.15]])
    validate_action_maxima_within_seed(seed, correction)


def test_action_maxima_within_seed_detects_violation_beyond_tolerance():
    seed = np.array([[10.0, -20.0]])
    correction = np.array([[11.5, -20.0]])  # dim 0 exceeds seed's 10 by 1.5, past the 1.0 default tolerance
    with pytest.raises(AssertionError, match="dims"):
        validate_action_maxima_within_seed(seed, correction)


def test_action_maxima_within_seed_custom_tolerance():
    seed = np.array([[10.0]])
    correction = np.array([[10.6]])
    validate_action_maxima_within_seed(seed, correction, tolerance_deg=1.0)
    with pytest.raises(AssertionError, match="dims"):
        validate_action_maxima_within_seed(seed, correction, tolerance_deg=0.5)


# --- validate_no_unexpected_idle_edges -------------------------------------------


def test_no_unexpected_idle_edges_passes_within_margin():
    # After trimming with margin_frames=2, at most 2 residual no-op frames per edge.
    deltas = [0.01, 0.01, 5.0, 5.0, 5.0, 0.01, 0.01]
    action = np.cumsum([0.0, *deltas]).reshape(-1, 1)
    validate_no_unexpected_idle_edges(action, threshold_deg=0.1, margin_frames=2)


def test_no_unexpected_idle_edges_detects_excess_residual():
    # 5 leading no-op deltas but we only claim margin_frames=2 was used to trim.
    deltas = [0.01, 0.01, 0.01, 0.01, 0.01, 5.0, 5.0]
    action = np.cumsum([0.0, *deltas]).reshape(-1, 1)
    with pytest.raises(AssertionError, match="Unexpected idle edge"):
        validate_no_unexpected_idle_edges(action, threshold_deg=0.1, margin_frames=2)


# --- validate_frame_correspondence -----------------------------------------------


def test_frame_correspondence_passes_on_exact_slice():
    source = np.arange(20).reshape(10, 2)
    output = source[3:8]
    validate_frame_correspondence(source, output, start=3)


def test_frame_correspondence_detects_offset_mismatch():
    source = np.arange(20).reshape(10, 2)
    output = source[3:8]
    with pytest.raises(AssertionError, match="mismatch"):
        validate_frame_correspondence(source, output, start=4)  # wrong start


def test_frame_correspondence_detects_value_drift():
    source = np.arange(20).reshape(10, 2)
    output = source[3:8].copy()
    output[2, 0] += 1  # simulate corrupted/drifted frame
    with pytest.raises(AssertionError, match="mismatch"):
        validate_frame_correspondence(source, output, start=3)
