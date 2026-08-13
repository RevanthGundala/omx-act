from types import SimpleNamespace
from dataclasses import replace
import json

import pytest
import torch

from configs.act.base import ACTConfig
from models.act.act import ACTPolicy as BaseACTPolicy
from models.act.attention import Attention as BaseAttention
from models.recap import train_critic
from models.recap.act_critic import ACTCritic
from models.recap.act_recap import ACTPolicy as RecapACTPolicy, AdvantageCondition
from models.recap.attention import Attention as RecapAttention


NUM_BINS_FOR_TEST = 11


def _config() -> ACTConfig:
    return ACTConfig(
        dataset_repo_id="owner/dataset",
        camera_names=("top",),
        chunk_size=4,
        d_model=32,
        joint_dim=6,
        d_z=8,
        num_vae_encoder_layers=1,
        num_encoder_layers=1,
        num_decoder_layers=1,
        num_heads=4,
        mlp_dim=64,
        dropout=0.0,
    )


def test_recap_attention_matches_base_attention():
    torch.manual_seed(0)
    base = BaseAttention(32, num_heads=4, dropout=0.0).eval()
    recap = RecapAttention(32, num_heads=4, dropout=0.0).eval()
    recap.load_state_dict(base.state_dict())
    x = torch.randn(2, 5, 32)
    pos = torch.randn(1, 5, 32)
    mask = torch.tensor(
        [[[[True, True, True, True, True]]], [[[True, True, True, False, False]]]]
    )

    assert torch.allclose(
        recap(x, key_padding_mask=mask, pos=pos),
        base(x, key_padding_mask=mask, pos=pos),
    )


def test_recap_cross_attention_matches_base_attention():
    torch.manual_seed(0)
    base = BaseAttention(
        32,
        num_heads=4,
        dropout=0.0,
        use_cross_attention=True,
    ).eval()
    recap = RecapAttention(
        32,
        num_heads=4,
        dropout=0.0,
        use_cross_attention=True,
    ).eval()
    recap.load_state_dict(base.state_dict())
    x = torch.randn(2, 4, 32)
    memory = torch.randn(2, 6, 32)
    pos = torch.randn(1, 6, 32)
    query_pos = torch.randn(1, 4, 32)

    assert torch.allclose(
        recap(x, encoder_out=memory, pos=pos, query_pos=query_pos),
        base(x, encoder_out=memory, pos=pos, query_pos=query_pos),
    )


def test_recap_policy_loads_base_act_weights():
    torch.manual_seed(0)
    base = BaseACTPolicy(_config(), pretrained_backbone=False).eval()
    recap = RecapACTPolicy(_config(), pretrained_backbone=False).eval()
    incompatible = recap.load_state_dict(base.state_dict(), strict=False)

    assert incompatible.unexpected_keys == []
    assert set(incompatible.missing_keys) == {
        "decoder.adv_proj.weight",
        "decoder.adv_proj.bias",
        "decoder.adv_pos_emb.weight",
    }


def test_recap_policy_defaults_to_unconditioned_token():
    model = RecapACTPolicy(_config(), pretrained_backbone=False).eval()
    images = torch.randn(2, 1, 3, 64, 64)
    qpos = torch.randn(2, 6)

    with torch.inference_mode():
        default_actions = model(images, qpos)[0]
        explicit_actions = model(
            images,
            qpos,
            advantage=AdvantageCondition.UNCONDITIONED,
        )[0]

    assert torch.equal(default_actions, explicit_actions)


def test_recap_policy_accepts_condition_and_preserves_training_call_order():
    model = RecapACTPolicy(_config(), pretrained_backbone=False).train()
    images = torch.randn(2, 1, 3, 64, 64)
    qpos = torch.randn(2, 6)
    actions = torch.randn(2, 4, 6)
    action_mask = torch.ones(2, 4, dtype=torch.bool)
    advantage = torch.tensor(
        [
            AdvantageCondition.POSITIVE,
            AdvantageCondition.UNCONDITIONED,
        ]
    )

    predicted, mu, log_var = model(
        images,
        qpos,
        actions,
        action_mask,
        advantage=advantage,
    )

    assert predicted.shape == (2, 4, 6)
    assert mu.shape == (2, 8)
    assert log_var.shape == (2, 8)
    predicted.mean().backward()
    assert model.decoder.adv_proj.weight.grad is not None


def test_recap_policy_rejects_invalid_advantage_shape():
    model = RecapACTPolicy(_config(), pretrained_backbone=False).eval()
    images = torch.randn(2, 1, 3, 64, 64)
    qpos = torch.randn(2, 6)

    with pytest.raises(ValueError, match="advantage must"):
        model(images, qpos, advantage=torch.ones(2, 2))


def test_recap_policy_rejects_invalid_advantage_class():
    model = RecapACTPolicy(_config(), pretrained_backbone=False).eval()
    images = torch.randn(2, 1, 3, 64, 64)
    qpos = torch.randn(2, 6)

    with pytest.raises(ValueError, match=r"must be in \[0, 2\]"):
        model(images, qpos, advantage=torch.tensor([1, 3]))


def test_recap_critic_is_observation_only_distributional_value_model():
    model = ACTCritic(
        replace(_config(), chunk_size=3),
        pretrained_backbone=False,
        num_bins=11,
    ).train()
    images = torch.randn(2, 1, 3, 64, 64)
    qpos = torch.randn(2, 6)

    distribution = model(images, qpos)
    targets = torch.tensor([2, 7])
    loss = -distribution.log_prob(targets).mean()
    loss.backward()

    assert distribution.logits.shape == (2, 11)
    assert torch.isfinite(loss)
    assert model.decoder.value_head.weight.grad is not None


def test_recap_critic_can_initialize_from_base_act_decoder():
    config = _config()
    actor = BaseACTPolicy(config, pretrained_backbone=False)
    critic = ACTCritic(config, pretrained_backbone=False, num_bins=11)

    incompatible = critic.decoder.load_state_dict(
        actor.decoder.state_dict(),
        strict=False,
    )

    assert incompatible.missing_keys == [
        "value_head.weight",
        "value_head.bias",
    ]
    assert incompatible.unexpected_keys == [
        "act_proj.weight",
        "act_proj.bias",
    ]


def test_critic_return_range_is_derived_from_recap_manifest(tmp_path):
    (tmp_path / "recap_manifest.json").write_text(
        json.dumps({"failure_penalty": 201.0, "max_episode_steps": 200})
    )

    value_min, value_max = train_critic.load_return_range(tmp_path)

    # Worst case is the longest episode failing on its final step.
    assert value_min == pytest.approx(-(200 - 1) - 201.0)
    assert value_max == 0.0


def test_critic_return_normalization_maps_into_unit_range_and_round_trips():
    value_min, value_max = -400.0, 0.0
    raw = torch.tensor([-400.0, -250.5, -1.0, 0.0])

    normalized = train_critic.normalize_return(raw, value_min, value_max)

    assert float(normalized.min()) >= -1.0
    assert float(normalized.max()) <= 0.0
    assert normalized[0].item() == pytest.approx(-1.0)
    assert normalized[-1].item() == pytest.approx(0.0)
    assert torch.allclose(
        train_critic.denormalize_return(normalized, value_min, value_max), raw
    )


def test_critic_return_bins_stay_in_range_for_extreme_and_typical_returns():
    value_min, value_max = -400.0, 0.0
    raw = torch.tensor([-1e6, -400.0, -399.0, -200.0, -1.0, 0.0, 1e6])

    bins = train_critic.return_to_bin(raw, value_min, value_max)

    assert int(bins.min()) >= 0
    assert int(bins.max()) <= train_critic.NUM_BINS - 1
    assert bins[1].item() == 0
    assert bins[-2].item() == train_critic.NUM_BINS - 1
    # Out-of-range returns clamp onto the extreme bins rather than escaping them.
    assert bins[0].item() == 0
    assert bins[-1].item() == train_critic.NUM_BINS - 1
    assert torch.all(bins.diff() >= 0)


def test_critic_train_step_reduces_loss_on_cpu():
    torch.manual_seed(0)
    model = ACTCritic(
        replace(_config(), chunk_size=2),
        pretrained_backbone=False,
        num_bins=train_critic.NUM_BINS,
    ).train()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    device = torch.device("cpu")
    value_min, value_max = -400.0, 0.0
    batch = {
        "images": torch.randn(2, 1, 3, 64, 64),
        "qpos": torch.randn(2, 6),
        "return_to_go": torch.tensor([-400.0, 0.0]),
    }

    losses = []
    for _ in range(4):
        optim.zero_grad(set_to_none=True)
        loss, distribution = train_critic.compute_loss(
            model, batch, device, value_min, value_max
        )
        loss.backward()
        optim.step()
        losses.append(loss.item())

    assert distribution.logits.shape == (2, train_critic.NUM_BINS)
    assert losses[-1] < losses[0]

    with torch.no_grad():
        predicted = train_critic.expected_return(distribution, value_min, value_max)
    assert predicted.shape == (2,)
    assert torch.all(predicted >= value_min)
    assert torch.all(predicted <= value_max)


def test_split_episodes_holds_out_whole_episodes_of_both_outcomes():
    outcomes = {i: (i < 8) for i in range(18)}  # 8 success, 10 failure

    train, val = train_critic.split_episodes(outcomes, num_val_episodes=4, seed=0)

    assert len(val) == 4
    assert len(train) == 14
    assert not set(train) & set(val)
    assert sorted(train + val) == sorted(outcomes)
    # Both outcomes must appear or success/failure AUC is undefined.
    assert any(outcomes[i] for i in val)
    assert any(not outcomes[i] for i in val)


def test_split_episodes_is_deterministic_and_seed_sensitive():
    outcomes = {i: (i < 8) for i in range(18)}

    assert train_critic.split_episodes(outcomes, 4, seed=3) == train_critic.split_episodes(
        outcomes, 4, seed=3
    )
    assert train_critic.split_episodes(outcomes, 4, seed=3) != train_critic.split_episodes(
        outcomes, 4, seed=4
    )


def test_split_episodes_disabled_returns_every_episode_for_training():
    outcomes = {i: (i % 2 == 0) for i in range(6)}

    train, val = train_critic.split_episodes(outcomes, num_val_episodes=0, seed=0)

    assert val == []
    assert train == sorted(outcomes)


def test_split_episodes_rejects_impossible_holdout():
    outcomes = {0: True, 1: False, 2: False}

    with pytest.raises(ValueError, match="Cannot hold out"):
        train_critic.split_episodes(outcomes, num_val_episodes=3, seed=0)


def test_warm_start_transfers_every_shared_act_decoder_weight(tmp_path):
    config = _config()
    actor = BaseACTPolicy(config, pretrained_backbone=False)
    critic = ACTCritic(config, pretrained_backbone=False, num_bins=NUM_BINS_FOR_TEST)
    checkpoint = tmp_path / "act.pt"
    torch.save({"model_state_dict": actor.state_dict()}, checkpoint)

    before = critic.decoder.backbone[0].weight.clone()
    report = train_critic.load_actor_decoder_weights(critic, checkpoint)

    assert report["transferred_params"] > 0
    assert not torch.equal(before, critic.decoder.backbone[0].weight)
    torch.testing.assert_close(
        critic.decoder.backbone[0].weight,
        actor.decoder.backbone[0].weight,
    )


def test_warm_start_rejects_a_checkpoint_without_decoder_weights(tmp_path):
    critic = ACTCritic(_config(), pretrained_backbone=False, num_bins=NUM_BINS_FOR_TEST)
    checkpoint = tmp_path / "bad.pt"
    torch.save({"model_state_dict": {"something.else": torch.zeros(1)}}, checkpoint)

    with pytest.raises(ValueError, match="no decoder"):
        train_critic.load_actor_decoder_weights(critic, checkpoint)


def test_validate_critic_run_rejects_unsafe_configurations():
    from deploy.train_recap_critic_modal import validate_critic_run

    ok = dict(
        dataset_repo_id="owner/dataset",
        steps=1000,
        save_freq=100,
        val_episodes=4,
        job_name="recap_critic",
    )
    validate_critic_run(**ok)

    with pytest.raises(ValueError, match="owner/name"):
        validate_critic_run(**{**ok, "dataset_repo_id": "no-slash"})
    # The final step must never be the first save.
    with pytest.raises(ValueError, match="save_freq"):
        validate_critic_run(**{**ok, "save_freq": 1000})
    with pytest.raises(ValueError, match="positive"):
        validate_critic_run(**{**ok, "steps": 0})
    with pytest.raises(ValueError, match="non-negative"):
        validate_critic_run(**{**ok, "val_episodes": -1})


def test_derive_return_metadata_matches_the_manifest_scheme():
    import numpy as np

    class FakeHFDataset:
        def select_columns(self, columns):
            return self

        def to_dict(self):
            # Two 3-step episodes: one success (terminal 0), one failure (-10).
            return {
                "episode_index": np.array([0, 0, 0, 1, 1, 1]),
                "reward": np.array([-1.0, -1.0, 0.0, -1.0, -1.0, -10.0]),
                "success": np.array([True, True, True, False, False, False]),
            }

    dataset = SimpleNamespace(hf_dataset=FakeHFDataset())
    max_steps, penalty, outcomes = train_critic.derive_return_metadata(dataset)

    assert max_steps == 3
    assert penalty == 10.0
    assert outcomes == {0: True, 1: False}


@pytest.mark.parametrize(
    "level,expects_backbone_trainable,expects_head_trainable",
    [("none", True, True), ("backbone", False, True), ("encoder", False, True), ("all", False, True)],
)
def test_freeze_levels_control_which_parameters_train(
    level, expects_backbone_trainable, expects_head_trainable
):
    critic = ACTCritic(_config(), pretrained_backbone=False, num_bins=NUM_BINS_FOR_TEST)

    report = train_critic.freeze_critic(critic, level)

    assert critic.decoder.backbone[0].weight.requires_grad is expects_backbone_trainable
    assert critic.decoder.value_head.weight.requires_grad is expects_head_trainable
    assert report["trainable_params"] > 0
    if level == "all":
        # Only the value head may train.
        assert report["trainable_params"] == sum(
            p.numel() for p in critic.decoder.value_head.parameters()
        )


def test_freeze_levels_are_monotonic_and_validated():
    counts = []
    for level in train_critic.FREEZE_LEVELS:
        critic = ACTCritic(_config(), pretrained_backbone=False, num_bins=NUM_BINS_FOR_TEST)
        counts.append(train_critic.freeze_critic(critic, level)["trainable_params"])
    assert counts == sorted(counts, reverse=True)

    with pytest.raises(ValueError, match="freeze level must be one of"):
        train_critic.freeze_critic(
            ACTCritic(_config(), pretrained_backbone=False, num_bins=NUM_BINS_FOR_TEST),
            "bogus",
        )


def test_warm_start_reports_chunk_size_skew_clearly(tmp_path):
    from dataclasses import replace

    actor = BaseACTPolicy(replace(_config(), chunk_size=4), pretrained_backbone=False)
    critic = ACTCritic(
        replace(_config(), chunk_size=9), pretrained_backbone=False, num_bins=NUM_BINS_FOR_TEST
    )
    checkpoint = tmp_path / "act.pt"
    torch.save({"model_state_dict": actor.state_dict()}, checkpoint)

    with pytest.raises(ValueError, match="RECAP_CRITIC_CHUNK_SIZE"):
        train_critic.load_actor_decoder_weights(critic, checkpoint)


def test_val_baseline_is_constant_across_model_states():
    """The baseline describes the data, so training must not move it."""

    class StubDistribution:
        def __init__(self, logits):
            self.logits = logits
            self.probs = torch.softmax(logits, dim=-1)

    returns = torch.tensor([-100.0, -50.0, -400.0, -700.0])
    successes = torch.tensor([True, True, False, False])
    batches = [
        {"return_to_go": returns, "success": successes, "images": None, "qpos": None}
    ]

    def make_model(bias):
        class StubModel:
            training = False

            def eval(self):
                return self

            def train(self, mode=True):
                return self

            def __call__(self, images, qpos):
                logits = torch.zeros(len(returns), NUM_BINS_FOR_TEST)
                logits[:, bias] = 10.0
                return StubDistribution(logits)

        return StubModel()

    original = train_critic.compute_loss
    train_critic.compute_loss = lambda model, batch, device, lo, hi: (
        torch.tensor(0.0),
        model(None, None),
    )
    try:
        early = train_critic.evaluate(make_model(0), batches, "cpu", -1000.0, 0.0)
        late = train_critic.evaluate(make_model(9), batches, "cpu", -1000.0, 0.0)
    finally:
        train_critic.compute_loss = original

    assert early["val_return_mae_baseline"] == pytest.approx(
        late["val_return_mae_baseline"]
    )
    expected = float((returns.mean() - returns).abs().mean())
    assert early["val_return_mae_baseline"] == pytest.approx(expected)
    # The two models predict differently, so MAE must still respond to the model.
    assert early["val_return_mae"] != pytest.approx(late["val_return_mae"])
