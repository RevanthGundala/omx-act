import torch

from configs.act.base import ACTConfig
from models.act.act import ACTPolicy


def _small_policy() -> ACTPolicy:
    config = ACTConfig(
        dataset_repo_id="owner/dataset",
        camera_names=("top",),
        chunk_size=4,
        d_model=32,
        joint_dim=2,
        d_z=8,
        num_vae_encoder_layers=1,
        num_encoder_layers=1,
        num_decoder_layers=1,
        num_heads=4,
        mlp_dim=64,
        dropout=0.0,
    )
    return ACTPolicy(config, pretrained_backbone=False)


def test_eval_mode_ignores_future_actions_like_lerobot_act():
    torch.manual_seed(0)
    policy = _small_policy().eval()
    images = torch.randn(2, 1, 3, 64, 64)
    state = torch.randn(2, 2)
    future_actions = torch.randn(2, 4, 2)
    action_mask = torch.ones(2, 4, dtype=torch.bool)

    with torch.inference_mode():
        expected, _, _ = policy(images, state)
        actual, mu, log_var = policy(
            images,
            state,
            future_actions,
            action_mask,
        )

    assert torch.equal(actual, expected)
    assert mu is None
    assert log_var is None


def test_train_mode_still_uses_future_actions_for_vae_posterior():
    torch.manual_seed(0)
    policy = _small_policy().train()
    images = torch.randn(2, 1, 3, 64, 64)
    state = torch.randn(2, 2)
    future_actions = torch.randn(2, 4, 2)
    action_mask = torch.ones(2, 4, dtype=torch.bool)

    _, mu, log_var = policy(images, state, future_actions, action_mask)

    assert mu is not None
    assert log_var is not None
