"""Pure, dataset-I/O-free helpers for building the seed+DAgger-correction aggregate.

`dataset_tools/trimming/trim_dataset.py` is Hub-only, single-repo, and always drops any
`intervention` column, so it can't be reused unchanged for this job: we need to
combine TWO source datasets (an already-trimmed seed and a raw corrections
cache), trim only the corrections, and attach per-frame provenance. This module
holds the selection/validation logic that makes that combination auditable and
unit-testable without touching a real dataset. The actual dataset I/O
(`LeRobotDataset` loading / `add_frame` / `save_episode`) lives in
`dataset_tools/dagger/build_aggregate.py`, which composes these helpers.

Everything here operates on plain numpy arrays / ints so it can be tested with
small synthetic inputs (see tests/dataset_tools/test_validators.py).

Note: an earlier aggregate (015, built from the now-abandoned
013-place-sponge-in-bowl-diverse-50-dagger corrections) hardcoded an
excluded-episode set here for one abnormal hardware-artifact episode in that
source dataset. The 016-place-sponge-in-bowl-custom-act-dagger-corrections
source used by the current aggregate has been independently audited
frame-by-frame (max per-joint step and frame0 action/state gap on every
episode) and has no analogous corrupt episode, so the default excluded set
below is empty. `select_correction_episodes` still accepts an explicit
`excluded` override so a future bad episode can be dropped without touching
dataset I/O code.
"""
from __future__ import annotations

import numpy as np

from dataset_tools.trimming.noop_trim import find_noop_trim_bounds

EXCLUDED_CORRECTION_EPISODES: frozenset[int] = frozenset()


def select_correction_episodes(
    available_episodes: list[int] | range,
    excluded: frozenset[int] = EXCLUDED_CORRECTION_EPISODES,
) -> list[int]:
    """Return the sorted correction episode indices to include in the aggregate.

    `available_episodes` is the full set of correction-candidate episode indices
    present in the source (dagger) dataset. Anything in `excluded` is dropped.
    """
    return sorted(int(e) for e in available_episodes if int(e) not in excluded)


def validate_contiguous_episode_indices(episode_indices: np.ndarray, expected_total: int) -> None:
    """Assert `episode_indices` (per-frame episode_index column of the built
    aggregate) covers exactly `0..expected_total-1`, contiguously, with no gaps
    or repeats-out-of-order (frames of a given episode may repeat that episode's
    index consecutively, but the set of distinct episode indices seen, in the
    order first encountered, must be `0, 1, ..., expected_total-1`).
    """
    distinct_in_order: list[int] = []
    seen: set[int] = set()
    for value in episode_indices:
        value = int(value)
        if value not in seen:
            if distinct_in_order and value != distinct_in_order[-1] + 1:
                raise AssertionError(
                    f"Non-contiguous episode_index sequence: after {distinct_in_order[-1]} saw {value}"
                )
            distinct_in_order.append(value)
            seen.add(value)
        elif distinct_in_order[-1] != value:
            raise AssertionError(f"Episode {value} frames are not contiguous in the stream")
    if distinct_in_order != list(range(expected_total)):
        raise AssertionError(
            f"Expected contiguous episode indices 0..{expected_total - 1}, "
            f"got {len(distinct_in_order)} episodes: {distinct_in_order[:5]}...{distinct_in_order[-5:]}"
        )


def validate_finite(array: np.ndarray, name: str) -> None:
    """Assert every element of `array` is finite (no NaN/Inf)."""
    if not np.all(np.isfinite(array)):
        bad = int(np.count_nonzero(~np.isfinite(array)))
        raise AssertionError(f"{name} contains {bad} non-finite value(s)")


def validate_exact_fps(
    timestamps: np.ndarray,
    episode_boundaries: list[tuple[int, int]],
    fps: int,
    tolerance_s: float = 1e-4,
) -> None:
    """Assert per-episode consecutive-frame timestamp deltas equal `1/fps`
    within `tolerance_s`. `episode_boundaries` is a list of (start, stop)
    index ranges (Python slice semantics) into `timestamps`, one per episode.
    """
    expected_delta = 1.0 / fps
    for start, stop in episode_boundaries:
        episode_ts = timestamps[start:stop]
        if len(episode_ts) < 2:
            continue
        deltas = np.diff(episode_ts)
        bad = np.abs(deltas - expected_delta) > tolerance_s
        if np.any(bad):
            bad_index = int(np.argmax(bad))
            raise AssertionError(
                f"Episode frames [{start}:{stop}) has a timestamp delta "
                f"{deltas[bad_index]:.6f}s at local frame {bad_index}, expected {expected_delta:.6f}s "
                f"(fps={fps})"
            )


def validate_action_maxima_within_seed(
    seed_action: np.ndarray,
    correction_action: np.ndarray,
    tolerance_deg: float = 1.0,
) -> None:
    """Assert each correction action dimension's max absolute value does not
    exceed the corresponding seed dimension's max absolute value by more than
    `tolerance_deg`. Guards against corrections introducing out-of-distribution
    action steps (the original incident this defends against was a 6.5-7.4
    degree uncommanded jump).

    `tolerance_deg` defaults to 1.0: measured on the real 016 corrections
    source, genuine (non-corrupt, independently delta-audited) corrections
    held elbow_flex/wrist_flex/gripper steady at most 0.10-0.22 degrees beyond
    the 014 seed's 50-episode envelope for many consecutive frames each --
    consistent with a deliberate corrective hold reaching a hair past any
    single pre-collected demo, not a jerk/spike. 1.0 degree keeps an order of
    magnitude of margin below that observed benign case while remaining two
    orders of magnitude tighter than the corruption signature it targets.
    """
    seed_max = np.abs(seed_action).max(axis=0)
    correction_max = np.abs(correction_action).max(axis=0)
    violations = correction_max > seed_max + tolerance_deg
    if np.any(violations):
        bad_dims = np.nonzero(violations)[0].tolist()
        raise AssertionError(
            f"Correction action maxima exceed seed maxima (+{tolerance_deg} deg tolerance) on dims "
            f"{bad_dims}: correction={correction_max[violations]}, seed={seed_max[violations]}"
        )


def validate_no_unexpected_idle_edges(
    action: np.ndarray,
    threshold_deg: float,
    margin_frames: int,
) -> None:
    """Assert an already-trimmed episode's `action` array has no more than
    `margin_frames` of residual near-zero-motion frames at either edge.

    We intentionally retain `margin_frames` of the detected no-op run at each
    edge when trimming (see `find_noop_trim_bounds`), so re-detecting a no-op
    run of length <= margin_frames at an edge is expected. A longer run means
    idle padding leaked through and the episode has an "unexpected idle edge".
    """
    lead, trail = find_noop_trim_bounds(action, threshold_deg=threshold_deg, margin_frames=0)
    residual_lead = lead
    residual_trail = action.shape[0] - trail
    if residual_lead > margin_frames or residual_trail > margin_frames:
        raise AssertionError(
            f"Unexpected idle edge: residual_lead={residual_lead} residual_trail={residual_trail} "
            f"exceed margin_frames={margin_frames}"
        )


def validate_frame_correspondence(
    source_array: np.ndarray,
    output_array: np.ndarray,
    start: int,
) -> None:
    """Assert `output_array` equals `source_array[start:start + len(output_array)]`
    exactly, i.e. every kept output frame is byte-for-byte the same source frame
    at the expected offset (no accidental reordering/duplication/drift during
    trimming).
    """
    stop = start + output_array.shape[0]
    expected = source_array[start:stop]
    if expected.shape != output_array.shape or not np.array_equal(expected, output_array):
        raise AssertionError(
            f"Frame correspondence mismatch: expected source[{start}:{stop}] "
            f"(shape {expected.shape}) to equal output (shape {output_array.shape})"
        )
