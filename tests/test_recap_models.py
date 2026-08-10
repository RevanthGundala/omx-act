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


def test_critic_return_bins_stay_in_range_for_extreme_and_typical_returns(monkeypatch):
    value_min, value_max = -400.0, 0.0
    raw = torch.tensor([-1e6, -400.0, -399.0, -200.0, -1.0, 0.0, 1e6])
    model = ACTCritic(
        replace(_config(), chunk_size=2),
        pretrained_backbone=False,
        num_bins=train_critic.NUM_BINS,
    ).eval()
    batch = {
        "images": torch.randn(len(raw), 1, 3, 64, 64),
        "qpos": torch.randn(len(raw), 6),
        "return_to_go": raw,
    }

    # The discretization is inlined into compute_loss, so capture the bin
    # indices it hands to cross-entropy.
    captured = {}
    real_cross_entropy = train_critic.F.cross_entropy

    def spy(logits, targets, *args, **kwargs):
        captured["targets"] = targets
        return real_cross_entropy(logits, targets, *args, **kwargs)

    monkeypatch.setattr(train_critic.F, "cross_entropy", spy)
    with torch.no_grad():
        train_critic.compute_loss(model, batch, torch.device("cpu"), value_min, value_max)

    bins = captured["targets"]
    assert bins.dtype == torch.long
    assert bins.shape == (len(raw),)
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
