"""CPU smoke for the held-out L1 token ablation (CHECK 4)."""

import json

import pytest
import torch

from models.recap.act_recap import AdvantageCondition
from models.recap.check_token_loss import (
    accumulate_action_l1,
    flip_conditions,
    format_report,
    report_from_means,
    require_held_out_split,
    run_smoke,
    smoke_batch,
    tokens_for,
)


class _FakePolicy:
    """Predicts the target only when the token is POSITIVE; else target + 1."""

    training = False

    def eval(self):
        return self

    def __call__(self, images, qpos, future_actions=None, action_mask=None, advantage=None):
        actions = future_actions
        pred = actions.clone()
        wrong = advantage != int(AdvantageCondition.POSITIVE)
        pred[wrong] = actions[wrong] + 1.0
        return pred, None, None


def _labeled_batch(labels):
    batch = smoke_batch(batch_size=len(labels))
    batch["advantage_condition"] = torch.as_tensor(labels, dtype=torch.long)
    batch["actions"] = torch.zeros_like(batch["actions"])
    return batch


def test_flip_swaps_pos_and_neg_only():
    tokens = torch.tensor(
        [
            int(AdvantageCondition.NEGATIVE),
            int(AdvantageCondition.POSITIVE),
            int(AdvantageCondition.UNCONDITIONED),
        ]
    )
    flipped = flip_conditions(tokens)
    assert flipped.tolist() == [
        int(AdvantageCondition.POSITIVE),
        int(AdvantageCondition.NEGATIVE),
        int(AdvantageCondition.UNCONDITIONED),
    ]


def test_tokens_for_constant_overrides():
    true = torch.tensor([0, 1, 2])
    assert tokens_for("all_pos", true).tolist() == [1, 1, 1]
    assert tokens_for("all_neg", true).tolist() == [0, 0, 0]
    assert tokens_for("unconditioned", true).tolist() == [2, 2, 2]
    assert tokens_for("true", true).tolist() == [0, 1, 2]


def test_require_held_out_split_refuses_train_data():
    with pytest.raises(ValueError, match="held-out"):
        require_held_out_split(1.0)
    with pytest.raises(ValueError, match="train_split"):
        require_held_out_split(0.0)
    require_held_out_split(0.8)


def test_flip_raises_l1_when_the_true_token_is_the_right_one():
    """True POS labels match the fake policy; flipping them must increase L1."""
    batch = _labeled_batch([int(AdvantageCondition.POSITIVE), int(AdvantageCondition.POSITIVE)])
    scored = accumulate_action_l1(_FakePolicy(), [batch], torch.device("cpu"))
    means = scored["means"]
    assert means["true"] == pytest.approx(0.0)
    assert means["flipped"] == pytest.approx(1.0)
    assert means["all_pos"] == pytest.approx(0.0)
    assert means["all_neg"] == pytest.approx(1.0)
    assert means["unconditioned"] == pytest.approx(1.0)


def test_report_deltas_are_relative_to_true_l1():
    report = report_from_means(
        {"true": 0.50, "flipped": 0.55, "all_pos": 0.50, "all_neg": 0.60, "unconditioned": 0.50},
        n_frames=8,
        n_batches=1,
    )
    assert report["conditions"]["flipped"]["delta"] == pytest.approx(0.05)
    assert report["conditions"]["flipped"]["delta_pct"] == pytest.approx(10.0)
    assert report["conditions"]["all_neg"]["delta_pct"] == pytest.approx(20.0)
    assert report["labels_informative"] is True
    text = format_report(report)
    assert "CHECK 4" in text
    assert "flipped" in text


def test_inert_token_is_reported_as_uninformative():
    report = report_from_means(
        {"true": 0.50, "flipped": 0.501, "all_pos": 0.50, "all_neg": 0.50, "unconditioned": 0.50},
        n_frames=4,
        n_batches=1,
    )
    assert report["labels_informative"] is False
    assert "not informative" in format_report(report)


def test_cpu_smoke_runs_a_tiny_real_act_policy(tmp_path):
    report = run_smoke()
    assert report["smoke"] is True
    assert report["n_frames"] == 2
    assert set(report["conditions"]) == set(
        ("true", "flipped", "all_pos", "all_neg", "unconditioned")
    )
    for row in report["conditions"].values():
        assert row["action_loss"] == row["action_loss"]  # not NaN
        assert row["action_loss"] >= 0
    out = tmp_path / "smoke.json"
    out.write_text(json.dumps(report))
    assert json.loads(out.read_text())["n_batches"] == 1
