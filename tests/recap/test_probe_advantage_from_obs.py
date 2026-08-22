"""Correlation probe: can I be read from o without an advantage token?"""

from dataclasses import replace

import pytest
import torch

from configs.act.base import ACTConfig
from models.recap.critic import ACTCritic
from models.recap.policy import AdvantageCondition
from models.recap.analysis.probe_advantage import (
    POSITIVE,
    TinyVisionQposProbe,
    binary_auc,
    compute_metrics,
    eval_mask,
    interpret_auc,
    smoke_test,
    train_head_on_features,
)
from train.modal.common import validate_probe_run


def _tiny_critic_config() -> ACTConfig:
    return ACTConfig(
        dataset_repo_id="owner/dataset",
        camera_names=("wrist", "top"),
        chunk_size=2,
        d_model=32,
        joint_dim=6,
        d_z=8,
        num_encoder_layers=1,
        num_decoder_layers=1,
        num_heads=4,
        mlp_dim=64,
        dropout=0.0,
    )


def test_critic_encode_is_the_value_head_input():
    torch.manual_seed(0)
    model = ACTCritic(_tiny_critic_config(), pretrained_backbone=False, num_bins=11).eval()
    images = torch.randn(2, 2, 3, 32, 32)
    qpos = torch.randn(2, 6)

    features = model.encode(images, qpos)
    distribution = model(images, qpos)

    assert features.shape == (2, 32)
    # Categorical stores normalized logits; compare the induced distribution.
    from torch.distributions import Categorical

    assert torch.allclose(
        Categorical(logits=model.decoder.value_head(features)).logits,
        distribution.logits,
    )


def test_encode_does_not_use_an_advantage_token(repo_root):
    source = (repo_root / "models" / "recap" / "critic" / "model.py").read_text()
    assert "adv_proj" not in source
    assert "AdvantageCondition" not in source
    model = ACTCritic(_tiny_critic_config(), pretrained_backbone=False, num_bins=11)
    assert not hasattr(model, "adv_proj")


def test_binary_auc_is_one_for_perfect_ranking_and_half_for_ties():
    labels = torch.tensor([0, 0, 1, 1])
    assert binary_auc(torch.tensor([0.0, 0.1, 0.9, 1.0]), labels) == pytest.approx(1.0)
    assert binary_auc(torch.zeros(4), labels) == pytest.approx(0.5)
    assert torch.isnan(torch.tensor(binary_auc(torch.ones(2), torch.ones(2))))


def test_metrics_exclude_intervention_frames():
    """Corrections were forced POSITIVE; scoring them is the intervention shortcut."""
    labels = torch.tensor([1, 1, 0, 1, 0, 1])
    intervention = torch.tensor([1, 1, 0, 0, 0, 0], dtype=torch.bool)
    # Perfect on the two intervention frames, constant (chance) on the rest.
    scores = torch.tensor([10.0, 10.0, 0.0, 0.0, 0.0, 0.0])

    with_all = compute_metrics(scores, labels, intervention, exclude_intervention=False)
    held = compute_metrics(scores, labels, intervention, exclude_intervention=True)

    assert with_all["n"] == 6
    assert held["n"] == 4
    assert held["n_excluded"] == 2
    assert held["auc"] == pytest.approx(0.5)
    assert eval_mask(labels, intervention).sum() == 4


def test_linear_head_recovers_a_planted_feature():
    torch.manual_seed(0)
    n, dim = 80, 8
    labels = torch.randint(0, 2, (n,))
    features = torch.randn(n, dim)
    features = features.clone()
    features[:, 0] = (labels.float() * 4.0) - 2.0
    train, val = slice(0, 50), slice(50, None)
    head, losses = train_head_on_features(
        features[train],
        labels[train],
        kind="linear",
        steps=80,
        batch_size=32,
        seed=0,
        device=torch.device("cpu"),
    )
    scores = head(features[val]).detach()
    metrics = compute_metrics(scores, labels[val], torch.zeros(30, dtype=torch.bool))
    assert losses[-1] < losses[0]
    assert metrics["auc"] > 0.85
    assert metrics["accuracy"] > metrics["majority_accuracy"] - 1e-6


def test_tiny_vision_probe_accepts_top_and_wrist_and_rejects_language_shaped_input():
    model = TinyVisionQposProbe(joint_dim=6, num_cameras=2)
    images = torch.randn(3, 2, 3, 48, 64)
    qpos = torch.randn(3, 6)
    logits = model(images, qpos)
    assert logits.shape == (3,)
    with pytest.raises(ValueError, match="Expected 2 cameras"):
        model(torch.randn(3, 1, 3, 48, 64), qpos)


def test_interpret_auc_distinguishes_redundant_from_unreadable():
    assert "redundant" in interpret_auc(0.92)
    assert "redundant" in interpret_auc(0.05)
    assert "NOT evidence that I is action-conditional" in interpret_auc(0.51)
    assert "undefined" in interpret_auc(float("nan"))


def test_validate_probe_run_refuses_full_dataset_fit():
    with pytest.raises(ValueError, match="never train_split=1.0"):
        validate_probe_run(
            dataset_repo_id="RevanthGundala/034-recap-balanced",
            steps=100,
            val_episodes=0,
            job_name="x",
        )


def test_validate_probe_run_requires_critic_when_backend_is_critic():
    with pytest.raises(ValueError, match="critic_checkpoint"):
        validate_probe_run(
            dataset_repo_id="RevanthGundala/034-recap-balanced",
            steps=100,
            val_episodes=8,
            job_name="x",
            backend="critic",
        )


def test_cpu_smoke_on_fake_batches():
    report = smoke_test(device="cpu", steps=40)
    assert report["qpos_linear"]["auc"] >= 0.7
    assert report["encode_dim"] == [8, 32]
    assert POSITIVE == int(AdvantageCondition.POSITIVE)


def test_config_from_critic_checkpoint_ignores_unknown_keys(tmp_path):
    from models.recap.analysis.probe_advantage import config_from_critic_checkpoint

    cfg = replace(_tiny_critic_config(), dataset_repo_id="RevanthGundala/034-recap-balanced")
    model = ACTCritic(cfg, pretrained_backbone=False, num_bins=11)
    path = tmp_path / "critic.pt"
    torch.save(
        {
            "config": {**cfg.__dict__, "dataset_format": "lerobot", "benchmark_dataset_dir": None},
            "model_state_dict": model.state_dict(),
            "num_bins": 11,
            "value_min": -950.0,
            "value_max": 0.0,
        },
        path,
    )
    loaded = config_from_critic_checkpoint(path)
    assert loaded.dataset_repo_id == "RevanthGundala/034-recap-balanced"
    assert loaded.camera_names == ("wrist", "top")
