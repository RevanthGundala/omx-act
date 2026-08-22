"""The check that a RECAP policy actually uses its advantage token."""

import torch

from models.recap.act_recap import AdvantageCondition
from models.recap.analysis.check_token_sensitivity import (
    observation_scale,
    summarize,
    verdict,
)


class _FakePolicy:
    """Policy whose response to the token is controlled by ``token_gain``.

    ``token_gain=0`` reproduces the failure this check exists to catch: a
    policy that ignores its conditioning entirely.
    """

    def __init__(self, token_gain: float, *, correct_direction: bool = True):
        self.token_gain = token_gain
        self.correct_direction = correct_direction

    def __call__(self, images, qpos, advantage=None):
        # Predictions track the observation, so scene changes always move them.
        base = qpos.unsqueeze(1).repeat(1, 2, 1)
        token = advantage.float().view(-1, 1, 1)
        sign = 1.0 if self.correct_direction else -1.0
        offset = sign * self.token_gain * (token - float(AdvantageCondition.NEGATIVE))
        return base + offset, None, None


def _loader(batches: int = 3, batch: int = 4, dim: int = 3, labels=None):
    items = []
    for index in range(batches):
        qpos = torch.arange(batch * dim, dtype=torch.float32).reshape(batch, dim) + index
        entry = {
            "images": torch.zeros(batch, 1, 3, 4, 4),
            "qpos": qpos,
            # The demonstrated action is what a POSITIVE token should track.
            "actions": qpos.unsqueeze(1).repeat(1, 2, 1) + 1.0,
        }
        if labels is not None:
            entry["advantage_condition"] = labels
        items.append(entry)
    return items


def test_a_policy_that_ignores_the_token_fails():
    """The whole point: no gradient signal ever reports this failure."""
    report = summarize(_FakePolicy(token_gain=0.0), _loader(), torch.device("cpu"), 3)

    assert report["token_effect_sd"] == 0.0
    ok, message = verdict(report, min_relative=0.05)
    assert not ok
    assert "ignoring its conditioning" in message


def test_a_policy_that_uses_the_token_passes():
    labels = torch.full((4,), int(AdvantageCondition.POSITIVE))
    report = summarize(
        _FakePolicy(token_gain=1.0), _loader(labels=labels), torch.device("cpu"), 3
    )

    assert report["token_effect_sd"] > 0
    ok, _ = verdict(report, min_relative=0.05)
    assert ok


def test_a_token_effect_in_the_wrong_direction_fails():
    """Moving the actions is not enough; it has to move them the right way."""
    labels = torch.full((4,), int(AdvantageCondition.POSITIVE))
    report = summarize(
        _FakePolicy(token_gain=1.0, correct_direction=False),
        _loader(labels=labels),
        torch.device("cpu"),
        3,
    )

    assert report["token_effect_sd"] > 0
    assert report["positive_label_advantage_sd"] < 0
    ok, message = verdict(report, min_relative=0.05)
    assert not ok
    assert "wrong direction" in message


def test_sensitivity_is_reported_against_the_effect_of_the_observation():
    """A raw delta means nothing without the scale the policy already moves at."""
    report = summarize(_FakePolicy(token_gain=0.5), _loader(), torch.device("cpu"), 3)

    expected = report["token_effect_sd"] / report["observation_effect_sd"]
    assert abs(report["relative_sensitivity"] - expected) < 1e-9
    assert report["observation_effect_sd"] > 0


def test_observation_scale_needs_more_than_one_observation():
    import pytest

    with pytest.raises(ValueError, match="at least two"):
        observation_scale(torch.zeros(1, 2, 3))
