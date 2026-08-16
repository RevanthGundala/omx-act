import numpy as np
import pandas as pd
import pytest

from scripts.dataset.dagger.build_correction_split import (
    build_plans,
    parse_holdout,
    parse_source,
)


def frame(episode, frame_index, intervention=True, value=0.0):
    return {
        "episode_index": episode,
        "frame_index": frame_index,
        "index": episode * 100 + frame_index,
        "intervention": np.array([intervention]),
        "action": np.full(6, value, dtype=np.float32),
    }


def test_parse_source_and_holdout(tmp_path):
    source = parse_source(f"old|owner/repo|{tmp_path}")
    assert source.label == "old"
    assert source.repo_id == "owner/repo"
    assert source.root == tmp_path
    assert parse_holdout("old:7") == ("old", 7)


def test_plans_are_disjoint_and_trimmed():
    tables = {
        "old": pd.DataFrame(
            [
                frame(0, 0, value=0),
                frame(0, 1, value=0),
                frame(0, 2, value=1),
                frame(0, 3, value=2),
                frame(1, 0, value=0),
                frame(1, 1, value=1),
            ]
        ),
        "new": pd.DataFrame(
            [
                frame(0, 0, value=0),
                frame(0, 1, value=1),
            ]
        ),
    }

    plans = build_plans(
        tables,
        {("old", 1)},
        threshold=0.1,
        margin_frames=0,
    )

    assert [(plan.source_label, plan.source_episode, plan.split) for plan in plans] == [
        ("old", 0, "train"),
        ("old", 1, "holdout"),
        ("new", 0, "train"),
    ]
    assert plans[0].start == 1


def test_unknown_holdout_is_rejected():
    table = pd.DataFrame([frame(0, 0), frame(0, 1, value=1)])
    with pytest.raises(ValueError, match="Unknown holdout"):
        build_plans(
            {"old": table},
            {("new", 0)},
            threshold=0.1,
            margin_frames=0,
        )


def test_plans_allow_all_corrections_in_training():
    table = pd.DataFrame([frame(0, 0), frame(0, 1, value=1)])

    plans = build_plans(
        {"old": table},
        set(),
        threshold=0.1,
        margin_frames=0,
    )

    assert len(plans) == 1
    assert plans[0].split == "train"
