"""Unit tests for scripts.dataset.trimming.noop_trim.find_noop_trim_bounds -- pure logic, no dataset I/O.

Ported/adapted from FastWAM's _find_noop_trim_bounds (see scripts/dataset/trimming/noop_trim.py docstring).
"""
import numpy as np
import pytest

from scripts.dataset.trimming.noop_trim import find_noop_trim_bounds


def make_action(deltas: list[float]) -> np.ndarray:
    """Build a (len(deltas)+1, 1) action array whose consecutive frame-to-frame |delta| values
    are exactly `deltas` (single joint, for simplicity -- the function takes max across joints,
    tested separately below)."""
    action = np.zeros((len(deltas) + 1, 1), dtype=np.float32)
    for i, d in enumerate(deltas):
        action[i + 1, 0] = action[i, 0] + d
    return action


def test_no_trim_when_threshold_is_zero():
    action = make_action([0.0, 0.0, 5.0, 0.0, 0.0])
    start, stop = find_noop_trim_bounds(action, threshold_deg=0.0, margin_frames=4)
    assert (start, stop) == (0, 6)


def test_single_frame_episode_returns_full_range():
    action = np.zeros((1, 6), dtype=np.float32)
    start, stop = find_noop_trim_bounds(action, threshold_deg=0.1, margin_frames=4)
    assert (start, stop) == (0, 1)


def test_trims_leading_noop_run_beyond_margin():
    # 10 leading near-zero deltas (settle), then a real move, then more near-zero (trailing).
    deltas = [0.01] * 10 + [5.0] + [0.02] * 3
    action = make_action(deltas)  # 15 frames total
    start, stop = find_noop_trim_bounds(action, threshold_deg=0.1, margin_frames=4)
    # lead_noop=10 -> lead_trim=max(0,10-4)=6; trail_noop=3 -> trail_trim=max(0,3-4)=0
    assert start == 6
    assert stop == 15  # no trailing trim since trail_noop(3) <= margin(4)


def test_trims_trailing_noop_run_beyond_margin():
    deltas = [0.02] * 2 + [5.0] + [0.01] * 10
    action = make_action(deltas)  # 14 frames total
    start, stop = find_noop_trim_bounds(action, threshold_deg=0.1, margin_frames=4)
    assert start == 0  # lead_noop(2) <= margin(4)
    # trail_noop=10 -> trail_trim=max(0,10-4)=6 -> stop = 14 - 6 = 8
    assert stop == 8


def test_margin_frames_kept_at_each_edge_not_trimmed_away():
    # exactly margin_frames of no-op at each edge should NOT be trimmed at all.
    deltas = [0.01] * 4 + [5.0] + [0.01] * 4
    action = make_action(deltas)  # 10 frames
    start, stop = find_noop_trim_bounds(action, threshold_deg=0.1, margin_frames=4)
    assert (start, stop) == (0, 10)


def test_never_trims_a_pause_in_the_middle_of_the_episode():
    # A real move, then a long near-zero pause in the MIDDLE, then another real move.
    # Only leading/trailing runs may be trimmed; the middle pause must survive untouched.
    deltas = [5.0] + [0.01] * 15 + [5.0]
    action = make_action(deltas)  # 18 frames
    start, stop = find_noop_trim_bounds(action, threshold_deg=0.1, margin_frames=4)
    assert (start, stop) == (0, 18)


def test_all_frames_noop_keeps_at_least_two_frames():
    deltas = [0.01] * 20
    action = make_action(deltas)  # 21 frames, all below threshold
    start, stop = find_noop_trim_bounds(action, threshold_deg=0.1, margin_frames=4)
    assert stop - start >= 2
    assert (start, stop) == (0, 2)  # degenerate-episode fallback


def test_uses_max_across_joints_not_a_single_joint():
    # joint 0 has a big early move; joint 1 has a big late move. Neither the leading nor
    # trailing run should be trimmable once max-across-joints correctly picks up both.
    n = 12
    action = np.zeros((n, 2), dtype=np.float32)
    action[2, 0] = 10.0  # big jump on joint 0 near the start (frame 1->2)
    action[9, 1] = 10.0  # big jump on joint 1 near the end (frame 8->9)
    start, stop = find_noop_trim_bounds(action, threshold_deg=0.1, margin_frames=4)
    # lead_noop counts frames before index 1 (delta at index1->2 is the first big one) = 1 -> trim 0
    # trail_noop counts frames after the joint-1 jump similarly small -> trim 0
    assert start == 0
    assert stop == n


def test_threshold_boundary_is_strict_less_than():
    # A delta exactly AT threshold should count as a real move (not < threshold), matching the
    # ported reference implementation's `if d < threshold_deg`. Uses whole-number deltas (exactly
    # representable in float32) to avoid float rounding noise around the boundary comparison.
    deltas = [1.0, 1.0, 5.0, 1.0, 1.0]
    action = make_action(deltas)
    start, stop = find_noop_trim_bounds(action, threshold_deg=1.0, margin_frames=0)
    # 1.0 is NOT < 1.0, so none of the leading/trailing deltas count as no-ops at all.
    assert (start, stop) == (0, 6)


@pytest.mark.parametrize("margin", [0, 2, 4, 8])
def test_kept_range_never_exceeds_original_length(margin):
    deltas = [0.01] * 20 + [5.0] + [0.01] * 20
    action = make_action(deltas)
    start, stop = find_noop_trim_bounds(action, threshold_deg=0.1, margin_frames=margin)
    assert 0 <= start <= stop <= action.shape[0]
