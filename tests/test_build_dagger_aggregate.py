"""Unit tests for data.build_dagger_aggregate's pure planning helpers
(_intervention_correction_candidates, build_plans). No dataset I/O: uses small
synthetic pandas DataFrames shaped like the real parquet frame tables.

Regression coverage for a real bug caught while building the original 015
aggregate (from the now-abandoned 013 dagger corrections): that dagger cache
embedded its OWN raw copy of the seed episodes at the SAME episode-index
values as the real seed dataset (`intervention=False`). Selecting correction
candidates by episode-index range/uniqueness alone (as opposed to the
`intervention` flag) silently pulled those embedded seed copies in as
"corrections" too, nearly doubling the intended episode/frame counts. The 016
corrections source used by the current aggregate does not have this problem
(every one of its 18 episodes is already correction-only), but the dedup
logic is kept as a generic safety net and is still exercised here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.build_dagger_aggregate import _intervention_correction_candidates, build_plans


def _make_frame(episode_index: int, frame_index: int, global_index: int, intervention: bool, action_val: float = 1.0) -> dict:
    return {
        "episode_index": episode_index,
        "frame_index": frame_index,
        "index": global_index,
        "action": np.array([action_val] * 6, dtype=np.float32),
        "observation.state": np.array([action_val] * 6, dtype=np.float32),
        "intervention": np.array([intervention]),
    }


def test_intervention_correction_candidates_excludes_embedded_seed_copies():
    rows = []
    g = 0
    # Episodes 0, 1: embedded raw seed copies (intervention=False) -- must NOT
    # be treated as correction candidates.
    for ep in (0, 1):
        for f in range(5):
            rows.append(_make_frame(ep, f, g, False))
            g += 1
    # Episodes 50, 51: genuine corrections (intervention=True).
    for ep in (50, 51):
        for f in range(5):
            rows.append(_make_frame(ep, f, g, True))
            g += 1
    df = pd.DataFrame(rows)

    candidates = _intervention_correction_candidates(df)

    assert candidates == [50, 51]


def test_intervention_correction_candidates_raises_on_mixed_episode():
    rows = []
    g = 0
    for f, flag in enumerate([True, True, False]):
        rows.append(_make_frame(7, f, g, flag))
        g += 1
    df = pd.DataFrame(rows)

    with pytest.raises(AssertionError):
        _intervention_correction_candidates(df)


def test_build_plans_excludes_embedded_seed_copies_and_keeps_all_genuine_corrections():
    seed_rows = []
    g = 0
    for f in range(3):
        seed_rows.append(_make_frame(0, f, g, False))
        g += 1
    seed_df = pd.DataFrame(seed_rows)

    corr_rows = []
    g = 100
    # Embedded raw seed copy at episode 0 -- shares an episode index with the
    # real seed dataset but must not become a "correction" plan.
    for f in range(3):
        corr_rows.append(_make_frame(0, f, g, False))
        g += 1
    # Real correction episodes: all kept by default (016's corrections source
    # has no known-bad episode, unlike the old 013-sourced aggregate).
    for ep in (50, 51, 63):
        for f in range(6):
            corr_rows.append(_make_frame(ep, f, g, True))
            g += 1
    corrections_df = pd.DataFrame(corr_rows)

    plans, _trim_report = build_plans(seed_df, corrections_df, noop_threshold_deg=0.1, noop_margin_frames=4)

    labels = [(p.source_label, p.source_episode_index) for p in plans]
    assert ("seed", 0) in labels
    assert ("correction", 0) not in labels
    assert ("correction", 50) in labels
    assert ("correction", 51) in labels
    assert ("correction", 63) in labels
    assert sum(1 for label, _ in labels if label == "seed") == 1
    assert sum(1 for label, _ in labels if label == "correction") == 3
