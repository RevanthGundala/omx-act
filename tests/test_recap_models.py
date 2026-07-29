from dataclasses import replace

import pytest
import torch

from configs.act.base import ACTConfig
from models.act.act import ACTPolicy as BaseACTPolicy
from models.act.attention import Attention as BaseAttention
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
