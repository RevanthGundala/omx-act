"""Matched-neighbour I(a ; I | o) check: planted signal vs circular-shift null."""

import numpy as np
import pytest

from models.recap.act_recap import AdvantageCondition
from models.recap.check_action_label_mi import (
    POSITIVE,
    circular_shift_labels,
    neighbor_indices,
    smoke_report,
    summarize_matcher,
)


def test_planted_action_shift_is_detected_and_shift_null_is_not():
    report = smoke_report()
    planted = report["planted"]
    shifted = report["shifted_labels"]
    assert planted["p"] < 0.05
    assert planted["observed_lift"] > 0.2
    assert shifted["p"] > 0.05


def test_circular_shift_preserves_rate_and_run_count():
    labels = np.array([0, 0, 0, 1, 1, 0, 1, 1, 1, 0])
    episodes = np.array([0] * 5 + [1] * 5)
    rng = np.random.default_rng(0)
    shifted = circular_shift_labels(labels, episodes, rng)
    assert shifted.mean() == pytest.approx(labels.mean())
    assert not np.array_equal(shifted, labels)
    # within-episode rate preserved
    assert shifted[:5].sum() == labels[:5].sum()
    assert shifted[5:].sum() == labels[5:].sum()


def test_neighbors_never_come_from_the_same_episode():
    features = np.eye(6, dtype=np.float64)
    episodes = np.array([0, 0, 1, 1, 2, 2])
    nbr = neighbor_indices(features, episodes, k=2)
    for i, row in enumerate(nbr):
        assert episodes[i] not in episodes[row]


def test_single_class_raises():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="both POS and NEG"):
        summarize_matcher(
            "x",
            rng.normal(size=(20, 4)),
            rng.normal(size=(20, 6)),
            np.full(20, POSITIVE),
            np.repeat(np.arange(4), 5),
            k=3,
            n_perm=5,
        )


def test_positive_constant_matches_enum():
    assert POSITIVE == int(AdvantageCondition.POSITIVE)
