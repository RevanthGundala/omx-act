from dataclasses import asdict

import pytest
import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.modeling_act import ACTTemporalEnsembler

from configs.act.base import ACTConfig
from models.act.lerobot_policy import (
    CustomACTConfig,
    CustomACTPolicy,
    _adapter_config,
)
from models.recap.act_recap import (
    ACTPolicy as RecapACTPolicy,
    AdvantageCondition,
)


def _source_config() -> ACTConfig:
    return ACTConfig(
        dataset_repo_id="owner/dataset",
        camera_names=("top",),
        chunk_size=3,
        d_model=32,
        d_z=8,
        num_vae_encoder_layers=1,
        num_encoder_layers=1,
        num_decoder_layers=1,
        num_heads=4,
        mlp_dim=64,
        joint_dim=2,
        exp_weight=0.01,
    )


def _policy_config(
    source: ACTConfig,
    *,
    recap: bool = False,
) -> CustomACTConfig:
    return CustomACTConfig(
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (2,)),
            "observation.images.top": PolicyFeature(FeatureType.VISUAL, (3, 64, 64)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (2,))},
        device="cpu",
        chunk_size=source.chunk_size,
        n_action_steps=1,
        temporal_ensemble_coeff=source.exp_weight,
        dim_model=source.d_model,
        n_heads=source.num_heads,
        dim_feedforward=source.mlp_dim,
        n_encoder_layers=source.num_encoder_layers,
        n_decoder_layers=source.num_decoder_layers,
        latent_dim=source.d_z,
        n_vae_encoder_layers=source.num_vae_encoder_layers,
        source_model_config=asdict(source),
        dataset_repo_id=source.dataset_repo_id,
        action_space="absolute",
        recap=recap,
    )


def test_adapter_select_action_honors_temporal_ensemble_config(monkeypatch):
    source = _source_config()
    policy = CustomACTPolicy(_policy_config(source))
    chunks = [
        torch.tensor([[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]]),
        torch.tensor([[[5.0, 50.0], [6.0, 60.0], [7.0, 70.0]]]),
    ]
    monkeypatch.setattr(policy, "predict_action_chunk", lambda _batch: chunks.pop(0))
    reference = ACTTemporalEnsembler(source.exp_weight, source.chunk_size)

    expected_first = reference.update(
        torch.tensor([[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]])
    )
    expected_second = reference.update(
        torch.tensor([[[5.0, 50.0], [6.0, 60.0], [7.0, 70.0]]])
    )

    assert torch.equal(policy.select_action({}), expected_first)
    assert torch.allclose(policy.select_action({}), expected_second)


def test_exported_adapter_preserves_native_temporal_ensemble_weight():
    source = _source_config()

    config = _adapter_config(source, asdict(source), "cpu", "absolute")

    assert config.temporal_ensemble_coeff == source.exp_weight


def test_recap_adapter_defaults_to_positive_and_can_select_another_token(
    monkeypatch,
):
    from models.act.lerobot_policy import ADVANTAGE_ENV_VAR

    monkeypatch.delenv(ADVANTAGE_ENV_VAR, raising=False)
    source = _source_config()
    policy = CustomACTPolicy(_policy_config(source, recap=True))
    assert isinstance(policy.model, RecapACTPolicy)
    # Deployment conditions on POSITIVE, so that is the default a RECAP
    # checkpoint evaluates with unless an ablation overrides it.
    assert policy.advantage_condition is AdvantageCondition.POSITIVE
    policy._action_queue.append(torch.ones(1))
    captured = []

    def forward(images, qpos, *, advantage):
        captured.append(advantage)
        return (
            torch.zeros(images.shape[0], source.chunk_size, source.joint_dim),
            None,
            None,
        )

    monkeypatch.setattr(policy.model, "forward", forward)
    policy.set_advantage_condition("negative")
    assert not policy._action_queue
    batch = {
        "observation.state": torch.zeros(1, source.joint_dim),
        "observation.images.top": torch.zeros(1, 3, 64, 64),
    }

    policy.predict_action_chunk(batch)

    assert captured == [AdvantageCondition.NEGATIVE]
    assert policy.advantage_condition is AdvantageCondition.NEGATIVE


def test_recap_adapter_honors_the_advantage_environment_override(monkeypatch):
    from models.act.lerobot_policy import ADVANTAGE_ENV_VAR

    source = _source_config()
    # eval_async passes the token to the server process this way.
    monkeypatch.setenv(ADVANTAGE_ENV_VAR, "unconditioned")
    policy = CustomACTPolicy(_policy_config(source, recap=True))
    assert policy.advantage_condition is AdvantageCondition.UNCONDITIONED

    # A non-RECAP checkpoint has no token to condition on and ignores the override.
    baseline = CustomACTPolicy(_policy_config(source, recap=False))
    assert baseline.advantage_condition is AdvantageCondition.UNCONDITIONED


def test_base_adapter_rejects_advantage_conditioning():
    policy = CustomACTPolicy(_policy_config(_source_config()))

    with pytest.raises(RuntimeError, match="RECAP-exported"):
        policy.set_advantage_condition(AdvantageCondition.POSITIVE)


def test_adapter_config_marks_recap_exports():
    source = _source_config()

    config = _adapter_config(
        source,
        asdict(source),
        "cpu",
        "absolute",
        recap=True,
    )

    assert config.recap is True
