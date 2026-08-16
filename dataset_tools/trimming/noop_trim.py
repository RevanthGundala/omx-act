"""Core no-op trimming logic, factored out so it can be unit-tested without any dataset I/O.

Ported from ~/projects/FastWAM/scripts/convert_omx_lerobot_to_fastwam.py's
`_find_noop_trim_bounds`, which trims the near-static "settle" segment teleop demos have at the
start (and sometimes end) of an episode -- frame-to-frame joint motion below a small threshold,
before the operator actually starts the real reach.

Why this matters for act-sim specifically: the settle segment is exactly what causes ACT's
predicted action chunks to be "back-loaded" (the real reach only shows up late in the 30-step
chunk, per outputs/chunk_aim_probe.py), which in turn is the root cause of both the sync+TE
closed-loop freeze and the intermittent async-eval freeze diagnosed this session (see
docs/act_debugging_learnings.md). Trimming the settle segment out of the training data should
make the reach start much closer to frame 0 of each episode, so learned chunks front-load the
reach instead.
"""
from __future__ import annotations

import numpy as np


def find_noop_trim_bounds(
    action: np.ndarray,
    threshold_deg: float = 0.1,
    margin_frames: int = 4,
) -> tuple[int, int]:
    """Return (start, stop) frame indices to KEEP (Python slice semantics: action[start:stop]),
    trimming leading/trailing runs of near-zero frame-to-frame joint motion.

    Args:
        action: (T, action_dim) array of raw (unnormalized) per-frame actions for one episode,
            in degrees for this robot.
        threshold_deg: max per-joint |delta| below which a frame-to-frame transition counts as
            a no-op. Set to 0 to disable trimming entirely.
        margin_frames: keep this many extra frames at each edge of the detected no-op run, so
            the cut isn't abrupt. Actual trim = max(0, detected_run_length - margin_frames).

    Only the leading/trailing runs are considered -- a pause in the MIDDLE of an episode (e.g.
    a brief hold mid-reach) is never trimmed, matching the source-of-truth FastWAM
    implementation this was ported from.
    """
    num_frames = action.shape[0]
    if threshold_deg <= 0 or num_frames <= 1:
        return 0, num_frames

    deltas = np.abs(np.diff(action, axis=0)).max(axis=1)  # (T-1,)

    lead_noop = 0
    for d in deltas:
        if d < threshold_deg:
            lead_noop += 1
        else:
            break

    trail_noop = 0
    for d in deltas[::-1]:
        if d < threshold_deg:
            trail_noop += 1
        else:
            break

    lead_trim = max(0, lead_noop - margin_frames)
    trail_trim = max(0, trail_noop - margin_frames)

    start = lead_trim
    stop = num_frames - trail_trim
    if stop <= start:
        # Degenerate episode (e.g. almost entirely static); keep at least 2 frames.
        start, stop = 0, min(2, num_frames)
    return start, stop
