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
from dataclasses import asdict, replace
from lerobot.datasets import LeRobotDataset
import numpy as np
from models.recap import label_advantage, train_critic, train_policy
from models.recap.build_dataset import build_features, recap_rewards

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
    from train.modal.recap_critic import validate_critic_run

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



# --------------------------------------------------------------------------
# Advantage labeling, LeRobot sources, and label binding
# --------------------------------------------------------------------------

def _labeling_config() -> ACTConfig:
    return replace(_config(), joint_dim=2, chunk_size=2, camera_names=("top",))


def _build_recap_dataset(root, repo_id, episodes):
    """Two-episode RECAP dataset written straight through build_dataset's schema."""
    features = build_features(
        camera_names=("top",),
        joint_names=("a", "b"),
        frame_shape=(64, 64, 3),
        use_videos=False,
    )
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=10,
        features=features,
        root=root,
        use_videos=False,
    )
    max_episode_steps = max(steps for steps, _ in episodes)
    failure_penalty = float(max_episode_steps + 1)
    for episode_index, (steps, success) in enumerate(episodes):
        rewards, returns_to_go = recap_rewards(
            steps,
            success=success,
            failure_penalty=failure_penalty,
        )
        for step in range(steps):
            dataset.add_frame(
                {
                    "observation.state": np.array(
                        [episode_index, step],
                        dtype=np.float32,
                    ),
                    "action": np.array([0.1 * step, 0.2 * step], dtype=np.float32),
                    "reward": np.array([rewards[step]], dtype=np.float32),
                    "return_to_go": np.array(
                        [returns_to_go[step]],
                        dtype=np.float32,
                    ),
                    "terminal": np.array([step == steps - 1], dtype=np.bool_),
                    "success": np.array([success], dtype=np.bool_),
                    "intervention": np.array([False], dtype=np.bool_),
                    "source.observation_index": np.array([step], dtype=np.int64),
                    "source.observation_frame_index": np.array([step], dtype=np.int64),
                    "source.action_index": np.array([step], dtype=np.int64),
                    "observation.images.top": np.full(
                        (64, 64, 3),
                        (episode_index * 40 + step * 10) % 256,
                        dtype=np.uint8,
                    ),
                    "task": "Place the sponge in the bowl",
                }
            )
        dataset.save_episode()
    dataset.finalize()
    (root / "recap_manifest.json").write_text(
        json.dumps(
            {
                "failure_penalty": failure_penalty,
                "max_episode_steps": max_episode_steps,
            }
        )
    )
    return dataset


# Episodes of length 3 (success) and 5 (failure) concatenated into one flat index.
_EPISODE_END = torch.tensor([3, 3, 3, 8, 8, 8, 8, 8])
_RETURNS_TO_GO = torch.tensor(
    [-2.0, -1.0, 0.0, -14.0, -13.0, -12.0, -11.0, -10.0]
)
_VALUES = torch.tensor([-5.0, -4.0, -3.0, -2.0, -1.0, -6.0, -7.0, -8.0])


def test_n_step_window_never_crosses_an_episode_boundary():
    advantages = label_advantage.n_step_advantages(
        _RETURNS_TO_GO,
        _VALUES,
        _EPISODE_END,
        n_step=2,
    )

    # Hand-computed with gamma = 1; frames 1 and 2 sit within two steps of the
    # next episode's frames, so a flat idx + n would read across the boundary.
    assert advantages.tolist() == pytest.approx(
        [0.0, 3.0, 3.0, -6.0, -8.0, -4.0, -4.0, -2.0]
    )

    # Rewriting only the second episode's values must not move the first
    # episode's advantages.
    perturbed = _VALUES.clone()
    perturbed[3:] += 100.0
    shifted = label_advantage.n_step_advantages(
        _RETURNS_TO_GO,
        perturbed,
        _EPISODE_END,
        n_step=2,
    )
    assert torch.allclose(shifted[:3], advantages[:3])
    assert not torch.allclose(shifted[3:], advantages[3:])


def test_n_step_advantage_truncates_to_remaining_return_without_bootstrap():
    advantages = label_advantage.n_step_advantages(
        _RETURNS_TO_GO,
        _VALUES,
        _EPISODE_END,
        n_step=2,
    )
    monte_carlo = _RETURNS_TO_GO - _VALUES

    # Frames within n steps of their terminal step use the exact remaining
    # Monte Carlo return and drop the bootstrap term entirely.
    truncated = [1, 2, 6, 7]
    assert torch.allclose(advantages[truncated], monte_carlo[truncated])
    bootstrapped = [0, 3, 4, 5]
    assert not torch.allclose(advantages[bootstrapped], monte_carlo[bootstrapped])

    # n at least as long as the longest episode recovers the paper's own
    # Monte Carlo estimator A = R_t - V(o_t) everywhere.
    assert torch.allclose(
        label_advantage.n_step_advantages(
            _RETURNS_TO_GO,
            _VALUES,
            _EPISODE_END,
            n_step=5,
        ),
        monte_carlo,
    )


def test_n_step_advantage_rejects_non_positive_window():
    with pytest.raises(ValueError, match="at least 1"):
        label_advantage.n_step_advantages(
            _RETURNS_TO_GO,
            _VALUES,
            _EPISODE_END,
            n_step=0,
        )


def test_advantage_conditions_are_binary_and_never_emit_unconditioned():
    advantages = torch.tensor([2.0, -2.0, 0.0, 0.05, -0.05])

    conditions = label_advantage.advantage_conditions(advantages)

    # The paper's indicator is binary, 1(A > eps_l), so labeling never produces
    # the unconditioned token. That token marks the policy trainer's random
    # conditioning dropout (Sec. V-B), which has nothing to do with advantages.
    assert conditions.tolist() == [
        AdvantageCondition.POSITIVE,
        AdvantageCondition.NEGATIVE,
        AdvantageCondition.NEGATIVE,
        AdvantageCondition.POSITIVE,
        AdvantageCondition.NEGATIVE,
    ]
    assert int((conditions == AdvantageCondition.UNCONDITIONED).sum()) == 0

    # A shifted threshold moves the split without inventing a third state.
    shifted = label_advantage.advantage_conditions(advantages, threshold=0.1)
    assert shifted.tolist() == [
        AdvantageCondition.POSITIVE,
        AdvantageCondition.NEGATIVE,
        AdvantageCondition.NEGATIVE,
        AdvantageCondition.NEGATIVE,
        AdvantageCondition.NEGATIVE,
    ]
    assert int((shifted == AdvantageCondition.UNCONDITIONED).sum()) == 0


def test_episode_layout_reads_boundaries_and_returns_from_the_dataset(tmp_path):
    root = tmp_path / "dataset"
    _build_recap_dataset(root, "owner/recap_layout", [(3, True), (5, False)])
    dataset = LeRobotDataset("owner/recap_layout", root=root)

    layout = label_advantage.episode_layout(dataset)

    assert layout["episode_end"].tolist() == _EPISODE_END.tolist()
    assert layout["frame_index"].tolist() == [0, 1, 2, 0, 1, 2, 3, 4]
    assert layout["episode_index"].tolist() == [0, 0, 0, 1, 1, 1, 1, 1]
    assert layout["rewards"].tolist() == [
        -1.0,
        -1.0,
        0.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -6.0,
    ]
    # Returns restart at every episode boundary rather than accumulating across
    # the flat index.
    assert layout["returns_to_go"].tolist() == [
        -2.0,
        -1.0,
        0.0,
        -10.0,
        -9.0,
        -8.0,
        -7.0,
        -6.0,
    ]


def test_advantage_labeling_end_to_end_on_cpu(tmp_path, monkeypatch):
    torch.manual_seed(0)
    root = tmp_path / "dataset"
    _build_recap_dataset(root, "owner/recap_labels", [(3, True), (5, False)])
    cfg = replace(_labeling_config(), dataset_repo_id="owner/recap_labels")
    device = torch.device("cpu")
    value_min, value_max = train_critic.load_return_range(root)

    critic = ACTCritic(cfg, pretrained_backbone=False, num_bins=ACTConfig.recap_num_bins)
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    train_critic.save_checkpoint(
        checkpoint_dir,
        1,
        critic,
        torch.optim.AdamW(critic.parameters(), lr=1e-4),
        cfg,
        None,
        value_min,
        value_max,
    )

    loaded, loaded_min, loaded_max = label_advantage.load_critic(
        checkpoint_dir / "last.pt", device
    )
    assert (loaded_min, loaded_max) == (value_min, value_max)
    assert not loaded.training

    # CriticDataset resolves a non-cache dataset root from the environment,
    # since build_dataset.py writes wherever --output-root points.
    monkeypatch.setenv("RECAP_CRITIC_DATASET_ROOT", str(root))
    dataset = train_critic.CriticDataset(cfg)
    labels = label_advantage.label_dataset(
        loaded,
        dataset,
        device,
        loaded_min,
        loaded_max,
        n_step=2,
        batch_size=3,
    )

    assert labels["values"].shape == (8,)
    assert torch.all(labels["values"] >= value_min)
    assert torch.all(labels["values"] <= value_max)
    assert labels["conditions"].dtype == torch.long
    assert set(labels["conditions"].tolist()) <= {
        int(AdvantageCondition.POSITIVE),
        int(AdvantageCondition.NEGATIVE),
    }
    # The default path resolves eps_l from the target positive fraction rather
    # than leaving it at zero, and both labels are represented.
    assert labels["thresholds"].shape == (8,)
    assert int((labels["conditions"] == AdvantageCondition.POSITIVE).sum()) > 0
    assert int((labels["conditions"] == AdvantageCondition.NEGATIVE).sum()) > 0
    assert torch.allclose(
        labels["thresholds"],
        label_advantage.task_percentile_thresholds(
            labels["advantages"],
            labels["task_index"],
            ACTConfig.recap_positive_fraction,
        ),
    )
    # The critic stays in eval mode under no_grad, so scoring is deterministic
    # and the advantages are exactly the boundary-aware arithmetic.
    layout = label_advantage.episode_layout(dataset.dataset)
    assert torch.allclose(
        labels["advantages"],
        label_advantage.n_step_advantages(
            layout["returns_to_go"],
            labels["values"],
            layout["episode_end"],
            n_step=2,
        ),
    )
    assert labels["advantages"].isfinite().all()

    # Labels drive the advantage-conditioned policy without further conversion.
    policy = RecapACTPolicy(cfg, pretrained_backbone=False).eval()
    with torch.inference_mode():
        predicted = policy(
            torch.randn(8, 1, 3, 64, 64),
            torch.randn(8, 2),
            advantage=labels["conditions"],
        )[0]
    assert predicted.shape == (8, 2, 2)


def test_n_step_advantage_rejects_misaligned_inputs():
    with pytest.raises(ValueError, match="aligned per frame"):
        label_advantage.n_step_advantages(
            _RETURNS_TO_GO,
            _VALUES[:-1],
            _EPISODE_END,
            n_step=2,
        )


def test_percentile_threshold_hits_the_requested_positive_fraction():
    advantages = torch.arange(100, dtype=torch.float32)

    for positive_fraction in (0.1, 0.3, 0.4, 0.7):
        thresholds = label_advantage.task_percentile_thresholds(
            advantages,
            positive_fraction=positive_fraction,
        )
        conditions = label_advantage.advantage_conditions(advantages, thresholds)
        positive = int((conditions == AdvantageCondition.POSITIVE).sum())
        assert positive == round(positive_fraction * 100)

    # eps_l for a 30% positive target sits at the 70th percentile, not the 30th.
    assert float(
        label_advantage.task_percentile_thresholds(
            advantages,
            positive_fraction=0.3,
        )[0]
    ) == pytest.approx(float(torch.quantile(advantages.double(), 0.7)))


def test_percentile_threshold_still_splits_a_miscalibrated_all_positive_critic():
    # A critic biased so badly that every advantage is positive: a sign split
    # collapses to a single label and the conditioning token carries nothing.
    advantages = torch.arange(1, 101, dtype=torch.float32)
    assert torch.all(advantages > 0)

    sign_split = label_advantage.advantage_conditions(advantages, 0.0)
    assert set(sign_split.tolist()) == {int(AdvantageCondition.POSITIVE)}

    thresholds = label_advantage.task_percentile_thresholds(
        advantages,
        positive_fraction=0.3,
    )
    conditions = label_advantage.advantage_conditions(advantages, thresholds)
    assert int((conditions == AdvantageCondition.POSITIVE).sum()) == 30
    assert int((conditions == AdvantageCondition.NEGATIVE).sum()) == 70
    assert float(thresholds[0]) > 0


def test_percentile_threshold_is_a_global_statistic_not_a_per_batch_one():
    advantages = torch.arange(100, dtype=torch.float32)

    whole = label_advantage.task_percentile_thresholds(advantages, positive_fraction=0.3)
    halves = torch.cat(
        [
            label_advantage.task_percentile_thresholds(
                advantages[:50],
                positive_fraction=0.3,
            ),
            label_advantage.task_percentile_thresholds(
                advantages[50:],
                positive_fraction=0.3,
            ),
        ]
    )

    # Thresholding batch-by-batch would force 30% positive inside every batch,
    # which is a different labeling than 30% positive over the whole set.
    assert not torch.allclose(whole, halves)
    assert torch.all(whole == whole[0])
    batched = label_advantage.advantage_conditions(advantages, halves)
    assert int((batched == AdvantageCondition.POSITIVE).sum()) == 30
    assert not torch.equal(
        label_advantage.advantage_conditions(advantages, whole),
        batched,
    )


def test_percentile_thresholds_are_computed_per_task():
    # Two tasks on wildly different advantage scales, as different-length or
    # different-difficulty tasks would be.
    slow = torch.linspace(-10.0, -1.0, 50)
    fast = torch.linspace(100.0, 200.0, 50)
    advantages = torch.cat([slow, fast])
    task_index = torch.cat([torch.zeros(50), torch.ones(50)]).long()

    per_task = label_advantage.task_percentile_thresholds(
        advantages,
        task_index,
        positive_fraction=0.3,
    )
    per_task_conditions = label_advantage.advantage_conditions(advantages, per_task)

    assert float(per_task[0]) < 0 < float(per_task[-1])
    for task in (0, 1):
        selected = task_index == task
        positive = int(
            (per_task_conditions[selected] == AdvantageCondition.POSITIVE).sum()
        )
        assert positive == 15

    # One shared threshold would hand the whole low-scale task a single label.
    pooled = label_advantage.task_percentile_thresholds(advantages, positive_fraction=0.3)
    pooled_conditions = label_advantage.advantage_conditions(advantages, pooled)
    assert int((pooled_conditions[task_index == 0] == AdvantageCondition.POSITIVE).sum()) == 0


def test_percentile_threshold_rejects_invalid_inputs():
    advantages = torch.arange(10, dtype=torch.float32)

    # Range and emptiness are delegated to torch.quantile rather than restated.
    for invalid in (-0.1, 1.5, float("nan")):
        with pytest.raises(RuntimeError, match="q must be in the range"):
            label_advantage.task_percentile_thresholds(
                advantages,
                positive_fraction=invalid,
            )
    with pytest.raises(RuntimeError, match="non-empty"):
        label_advantage.task_percentile_thresholds(torch.empty(0))
    with pytest.raises(ValueError, match="aligned per frame"):
        label_advantage.task_percentile_thresholds(
            advantages,
            torch.zeros(3, dtype=torch.long),
        )


def test_absolute_threshold_mode_remains_available_for_debugging():
    advantages = torch.tensor([-1.0, 0.0, 1.0, 5.0])

    pinned = label_advantage.advantage_conditions(advantages, threshold=1.0)

    assert pinned.tolist() == [
        AdvantageCondition.NEGATIVE,
        AdvantageCondition.NEGATIVE,
        AdvantageCondition.NEGATIVE,
        AdvantageCondition.POSITIVE,
    ]
    with pytest.raises(ValueError, match="aligned per frame"):
        label_advantage.advantage_conditions(advantages, torch.zeros(2))


class _StubACTDataset:
    """Minimal stand-in for models.act.helper.ACTDataset (which needs data/)."""

    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "qpos": torch.zeros(2),
            "images": torch.zeros(1, 3, 8, 8),
            "actions": torch.zeros(2, 2),
            "action_mask": torch.ones(2, dtype=torch.bool),
        }


def test_advantage_dataset_keys_labels_by_global_frame_index():
    # An episode-filtered ACT dataset yields positions 0..2 for global frames
    # 5, 6, 7, so positional lookup would silently read the wrong labels.
    conditions = torch.tensor(
        [
            AdvantageCondition.NEGATIVE,
            AdvantageCondition.NEGATIVE,
            AdvantageCondition.NEGATIVE,
            AdvantageCondition.NEGATIVE,
            AdvantageCondition.NEGATIVE,
            AdvantageCondition.POSITIVE,
            AdvantageCondition.NEGATIVE,
            AdvantageCondition.POSITIVE,
        ]
    )
    dataset = train_policy.AdvantageDataset(
        _StubACTDataset(3),
        conditions,
        frame_ids=torch.tensor([5, 6, 7]),
    )

    labels = [int(dataset[index]["advantage_condition"]) for index in range(3)]

    assert labels == [
        AdvantageCondition.POSITIVE,
        AdvantageCondition.NEGATIVE,
        AdvantageCondition.POSITIVE,
    ]
    # Positional lookup would have produced the first three labels instead.
    assert labels != [int(conditions[index]) for index in range(3)]


def test_advantage_dataset_on_an_episode_subset_keeps_whole_dataset_labels():
    # Training on a subset must not require relabeling: each frame has to keep
    # the exact token it would have received as part of the whole dataset. The
    # total_frames guard therefore counts the full dataset, not the subset.
    conditions = torch.tensor(
        [AdvantageCondition.NEGATIVE] * 5
        + [AdvantageCondition.POSITIVE, AdvantageCondition.NEGATIVE, AdvantageCondition.POSITIVE]
    )
    subset = train_policy.AdvantageDataset(
        _StubACTDataset(3),
        conditions,
        frame_ids=torch.tensor([5, 6, 7]),
        total_frames=len(conditions),
    )

    assert [int(subset[index]["advantage_condition"]) for index in range(3)] == [
        AdvantageCondition.POSITIVE,
        AdvantageCondition.NEGATIVE,
        AdvantageCondition.POSITIVE,
    ]


def test_advantage_dataset_rejects_total_frames_measured_on_the_subset():
    # Regression guard: reading total_frames off the filtered loader would make
    # a legitimate subset run look like a dataset mismatch and refuse to start.
    with pytest.raises(ValueError, match="built for a different"):
        train_policy.AdvantageDataset(
            _StubACTDataset(3),
            conditions=torch.zeros(8, dtype=torch.long),
            frame_ids=torch.tensor([5, 6, 7]),
            total_frames=3,
        )


@pytest.mark.parametrize(
    "spec,expected",
    [
        (None, None),
        ("", None),
        ("   ", None),
        ("48-58", list(range(48, 59))),
        ("59,60,61", [59, 60, 61]),
        ("48-50,65", [48, 49, 50, 65]),
        (" 48 - 50 , 65 ", [48, 49, 50, 65]),
        ("7-7", [7]),
        ("50,48,49,48", [48, 49, 50]),
    ],
)
def test_parse_episode_subset(spec, expected):
    assert train_policy.parse_episode_subset(spec) == expected


@pytest.mark.parametrize("spec", ["58-48", "-3", "48-", "abc"])
def test_parse_episode_subset_rejects_bad_specs(spec):
    with pytest.raises(ValueError):
        train_policy.parse_episode_subset(spec)


@pytest.fixture
def act_helper(monkeypatch):
    """models.act.helper with the episode split stubbed to a no-op."""
    from models.act import helper

    monkeypatch.setattr(
        helper, "resolve_train_holdout_episodes", lambda total, split, seed: (list(range(total)), [])
    )
    return helper


def test_make_dataloaders_episode_subset_intersects_the_seeded_split(act_helper, monkeypatch):
    captured = {}

    class _Meta:
        total_episodes = 66

        def __init__(self, **kwargs):
            pass

    class _Dataset:
        def __init__(self, cfg, episodes=None, image_transforms=None):
            captured.setdefault("episodes", []).append(episodes)

        def __len__(self):
            return 10

    monkeypatch.setattr(act_helper, "LeRobotDatasetMetadata", _Meta)
    monkeypatch.setattr(act_helper, "ACTDataset", _Dataset)

    cfg = ACTConfig(dataset_repo_id="x", train_split=1.0, num_workers=0, batch_size=2)
    act_helper.make_dataloaders(cfg, episodes=[48, 49, 65])

    assert captured["episodes"][0] == [48, 49, 65]


def test_make_dataloaders_without_a_subset_trains_on_everything(act_helper, monkeypatch):
    captured = {}

    class _Meta:
        total_episodes = 66

        def __init__(self, **kwargs):
            pass

    class _Dataset:
        def __init__(self, cfg, episodes=None, image_transforms=None):
            captured.setdefault("episodes", []).append(episodes)

        def __len__(self):
            return 10

    monkeypatch.setattr(act_helper, "LeRobotDatasetMetadata", _Meta)
    monkeypatch.setattr(act_helper, "ACTDataset", _Dataset)

    cfg = ACTConfig(dataset_repo_id="x", train_split=1.0, num_workers=0, batch_size=2)
    act_helper.make_dataloaders(cfg)

    assert captured["episodes"][0] == list(range(66))


def test_make_dataloaders_rejects_episode_indices_outside_the_dataset(act_helper, monkeypatch):
    class _Meta:
        total_episodes = 66

        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr(act_helper, "LeRobotDatasetMetadata", _Meta)

    cfg = ACTConfig(dataset_repo_id="x", train_split=1.0)
    with pytest.raises(ValueError, match="has 66 episodes"):
        act_helper.make_dataloaders(cfg, episodes=[48, 999])


def test_advantage_dataset_pins_positive_for_the_sft_stage():
    dataset = train_policy.AdvantageDataset(
        _StubACTDataset(4),
        conditions=None,
        frame_ids=torch.arange(4),
    )

    labels = [int(dataset[index]["advantage_condition"]) for index in range(4)]

    assert labels == [int(AdvantageCondition.POSITIVE)] * 4


def test_advantage_dataset_rejects_labels_that_do_not_cover_the_dataset():
    with pytest.raises(ValueError, match="relabel"):
        train_policy.AdvantageDataset(
            _StubACTDataset(2),
            conditions=torch.zeros(4, dtype=torch.long),
            frame_ids=torch.tensor([3, 9]),
        )


def test_conditioning_dropout_replaces_the_requested_share():
    conditions = torch.full((2000,), int(AdvantageCondition.POSITIVE), dtype=torch.long)
    generator = torch.Generator().manual_seed(0)

    dropped = train_policy.apply_conditioning_dropout(conditions, 0.30, generator)

    share = float((dropped == AdvantageCondition.UNCONDITIONED).float().mean())
    assert share == pytest.approx(0.30, abs=0.03)
    # Whatever is not dropped keeps its original label.
    kept = dropped != AdvantageCondition.UNCONDITIONED
    assert torch.all(dropped[kept] == AdvantageCondition.POSITIVE)

    # SFT runs with dropout disabled, which must be an exact no-op.
    assert torch.equal(
        train_policy.apply_conditioning_dropout(conditions, 0.0, generator),
        conditions,
    )
    with pytest.raises(ValueError, match="dropout must be"):
        train_policy.apply_conditioning_dropout(conditions, 1.5, generator)


def test_policy_warm_start_accepts_base_act_weights_and_trains_the_token(tmp_path):
    torch.manual_seed(0)
    cfg = _labeling_config()
    base = BaseACTPolicy(cfg, pretrained_backbone=False)
    checkpoint_path = tmp_path / "bc.pt"
    torch.save({"model_state_dict": base.state_dict(), "config": {}}, checkpoint_path)

    model = RecapACTPolicy(cfg, pretrained_backbone=False)
    train_policy.warm_start(model, checkpoint_path, torch.device("cpu"))

    # Everything except the advantage token carries over from behavior cloning.
    assert torch.equal(
        model.decoder.joint_proj.weight,
        base.decoder.joint_proj.weight,
    )

    model.train()
    batch = {
        "qpos": torch.randn(2, 2),
        "images": torch.randn(2, 1, 3, 64, 64),
        "actions": torch.randn(2, 2, 2),
        "action_mask": torch.ones(2, 2, dtype=torch.bool),
        "advantage_condition": torch.tensor(
            [AdvantageCondition.POSITIVE, AdvantageCondition.UNCONDITIONED]
        ),
    }
    loss, action_loss, kl_loss = train_policy.compute_loss(
        model, batch, torch.device("cpu"), cfg.kl_weight
    )
    loss.backward()

    assert torch.isfinite(loss) and torch.isfinite(action_loss) and torch.isfinite(kl_loss)
    assert model.decoder.adv_proj.weight.grad is not None
    assert torch.any(model.decoder.adv_proj.weight.grad != 0)


def test_policy_warm_start_rejects_a_checkpoint_missing_real_parameters(tmp_path):
    cfg = _labeling_config()
    state = BaseACTPolicy(cfg, pretrained_backbone=False).state_dict()
    del state["decoder.joint_proj.weight"]
    checkpoint_path = tmp_path / "broken.pt"
    torch.save({"model_state_dict": state, "config": {}}, checkpoint_path)

    with pytest.raises(ValueError, match="missing non-advantage parameters"):
        train_policy.warm_start(
            RecapACTPolicy(cfg, pretrained_backbone=False),
            checkpoint_path,
            torch.device("cpu"),
        )


def test_load_conditions_reads_the_labeler_sidecar(tmp_path):
    path = tmp_path / "advantage_labels.json"
    path.write_text(json.dumps({"conditions": [0, 1, 1, 0]}))

    assert train_policy.load_conditions(path).tolist() == [0, 1, 1, 0]

    path.write_text(json.dumps({"conditions": [0, 5]}))
    with pytest.raises(ValueError, match="out-of-range"):
        train_policy.load_conditions(path)


def test_recap_policies_default_to_positive_conditioning_at_inference(monkeypatch):
    from models.act.lerobot_policy import (
        ADVANTAGE_ENV_VAR,
        default_advantage_condition,
        resolve_advantage_condition,
    )

    monkeypatch.delenv(ADVANTAGE_ENV_VAR, raising=False)
    # Deployment asks for POSITIVE; defaulting to UNCONDITIONED would evaluate
    # the marginal policy the conditioning is meant to steer away from.
    assert default_advantage_condition(True) is AdvantageCondition.POSITIVE
    # A non-RECAP checkpoint has no advantage token to condition on.
    assert default_advantage_condition(False) is AdvantageCondition.UNCONDITIONED

    monkeypatch.setenv(ADVANTAGE_ENV_VAR, "negative")
    assert default_advantage_condition(True) is AdvantageCondition.NEGATIVE
    assert default_advantage_condition(False) is AdvantageCondition.UNCONDITIONED

    monkeypatch.setenv(ADVANTAGE_ENV_VAR, "bogus")
    with pytest.raises(ValueError, match="Unknown advantage condition"):
        default_advantage_condition(True)

    assert resolve_advantage_condition("POSITIVE") is AdvantageCondition.POSITIVE
    assert resolve_advantage_condition(int(AdvantageCondition.NEGATIVE)) is AdvantageCondition.NEGATIVE
    with pytest.raises(ValueError, match="Invalid advantage condition"):
        resolve_advantage_condition(7)


def test_policy_config_is_inherited_from_the_warm_start_checkpoint(tmp_path, monkeypatch):
    # The real baseline used chunk_size=30 and train_split=1.0, neither of which
    # is the ACTConfig default, and recorded a container-absolute output_root.
    source = replace(
        _labeling_config(),
        chunk_size=30,
        train_split=1.0,
        dataset_repo_id="owner/baseline",
        save_freq=2000,
        num_train_steps=20000,
        job_name="act_custom_baseline",
        output_root="/outputs/act_experiments",
    )
    checkpoint_path = tmp_path / "baseline.pt"
    torch.save(
        {
            "model_state_dict": BaseACTPolicy(source, pretrained_backbone=False).state_dict(),
            "config": {**asdict(source), "an_unknown_future_field": 1},
        },
        checkpoint_path,
    )
    monkeypatch.setenv("RECAP_POLICY_PRETRAINED_CHECKPOINT", str(checkpoint_path))
    monkeypatch.delenv("RECAP_POLICY_STEPS", raising=False)
    monkeypatch.delenv("RECAP_POLICY_DATASET_REPO_ID", raising=False)
    monkeypatch.delenv("RECAP_POLICY_OUTPUT_ROOT", raising=False)

    cfg = train_policy.config_from_env()

    # Architecture and data handling must match, or warm_start cannot load the
    # weights and the comparison against the baseline is not controlled.
    assert cfg.chunk_size == 30
    assert cfg.train_split == 1.0
    assert cfg.dataset_repo_id == "owner/baseline"
    assert cfg.learning_rate == source.learning_rate
    # Run identity is this run's, not the baseline's.
    assert cfg.job_name == "recap_policy"
    # Inheriting a container-absolute output_root would be unwritable elsewhere.
    assert cfg.output_root == ACTConfig.output_root

    # The inherited config is what makes the warm start loadable.
    train_policy.warm_start(
        RecapACTPolicy(cfg, pretrained_backbone=False),
        checkpoint_path,
        torch.device("cpu"),
    )

    monkeypatch.setenv("RECAP_POLICY_STEPS", "10000")
    assert train_policy.config_from_env().num_train_steps == 10000


def test_policy_config_rejects_a_checkpoint_without_a_config(tmp_path):
    path = tmp_path / "bare.pt"
    torch.save({"model_state_dict": {}}, path)

    with pytest.raises(ValueError, match="no config dictionary"):
        train_policy.config_from_checkpoint(path)


def test_corrections_are_forced_positive_regardless_of_advantage():
    """Sec. IV-B: force I_t = True for human corrections during rollouts.

    A rescue that arrives late trails a long negative run-up, so the critic can
    score it below eps_l. Conditioning on that would teach the policy to
    imitate expert recoveries under the negative token.
    """
    from models.recap.act_recap import AdvantageCondition
    from models.recap.label_advantage import force_corrections_positive

    conditions = torch.tensor(
        [
            int(AdvantageCondition.NEGATIVE),
            int(AdvantageCondition.NEGATIVE),
            int(AdvantageCondition.POSITIVE),
            int(AdvantageCondition.NEGATIVE),
        ]
    )
    intervention = torch.tensor([False, True, False, True])

    forced = force_corrections_positive(conditions, intervention)

    assert forced.tolist() == [
        int(AdvantageCondition.NEGATIVE),
        int(AdvantageCondition.POSITIVE),
        int(AdvantageCondition.POSITIVE),
        int(AdvantageCondition.POSITIVE),
    ]


def test_forcing_never_downgrades_an_autonomous_positive():
    from models.recap.act_recap import AdvantageCondition
    from models.recap.label_advantage import force_corrections_positive

    conditions = torch.full((5,), int(AdvantageCondition.POSITIVE))
    forced = force_corrections_positive(conditions, torch.zeros(5, dtype=torch.bool))
    assert torch.equal(forced, conditions)


def test_forcing_rejects_a_misaligned_intervention_vector():
    from models.recap.label_advantage import force_corrections_positive

    with pytest.raises(ValueError, match="does not match"):
        force_corrections_positive(torch.zeros(4, dtype=torch.long), torch.zeros(3, dtype=torch.bool))


def test_the_threshold_can_ignore_frames_that_are_forced_positive():
    """Corrections would otherwise move eps_l for data whose label is decided."""
    from models.recap.label_advantage import task_percentile_thresholds

    # Corrections carry extreme advantages; excluding them must change eps_l.
    advantages = torch.tensor([0.0, 1.0, 2.0, 3.0, 100.0, 200.0])
    mask = torch.tensor([True, True, True, True, False, False])

    with_all = task_percentile_thresholds(advantages, None, 0.5)
    without = task_percentile_thresholds(advantages, None, 0.5, mask=mask)

    assert float(without[0]) < float(with_all[0])
    # The resolved threshold still applies to every frame, masked or not.
    assert without.shape == advantages.shape


def test_a_mask_that_excludes_everything_is_rejected():
    from models.recap.label_advantage import task_percentile_thresholds

    with pytest.raises(ValueError, match="excludes every frame"):
        task_percentile_thresholds(
            torch.zeros(4), None, 0.4, mask=torch.zeros(4, dtype=torch.bool)
        )


def test_the_masked_threshold_hits_the_requested_fraction_on_unmasked_data():
    from models.recap.label_advantage import task_percentile_thresholds

    advantages = torch.arange(100.0)
    mask = torch.ones(100, dtype=torch.bool)
    thresholds = task_percentile_thresholds(advantages, None, 0.40, mask=mask)
    positive = (advantages > thresholds).float().mean()
    assert abs(float(positive) - 0.40) < 0.02


class _FrameIdACTDataset:
    """Stands in for models.act.helper.ACTDataset over a fixed frame count."""

    def __init__(self, frame_ids):
        self._frame_ids = list(frame_ids)
        self.dataset = SimpleNamespace(
            hf_dataset={"index": self._frame_ids}
        )

    def __len__(self):
        return len(self._frame_ids)

    def __getitem__(self, idx):
        return {"qpos": torch.zeros(6)}


def test_labels_from_a_different_dataset_are_rejected():
    """Frame ids staying in range is not evidence the labels belong here.

    A warm-start checkpoint names the demonstration set it was trained on, so
    inheriting it while using labels built for the larger RECAP set leaves every
    id in range and pairs transitions with other transitions' tokens.
    """
    from models.recap.train_policy import AdvantageDataset

    dataset = _FrameIdACTDataset(range(48))
    conditions = torch.zeros(96, dtype=torch.long)

    with pytest.raises(ValueError, match="built for a different dataset"):
        AdvantageDataset(dataset, conditions, total_frames=48)


def test_matching_label_and_dataset_lengths_are_accepted():
    from models.recap.train_policy import AdvantageDataset

    dataset = _FrameIdACTDataset(range(48))
    wrapped = AdvantageDataset(dataset, torch.zeros(48, dtype=torch.long), total_frames=48)
    assert len(wrapped) == 48


def test_a_held_out_split_does_not_trip_the_binding_check():
    """make_dataloaders holds episodes out, so the split is smaller than the set."""
    from models.recap.train_policy import AdvantageDataset

    # 30 training frames drawn from a 48-frame dataset, labels cover all 48.
    dataset = _FrameIdACTDataset(range(30))
    wrapped = AdvantageDataset(
        dataset, torch.zeros(48, dtype=torch.long), total_frames=48
    )
    assert len(wrapped) == 30


def test_out_of_range_frame_ids_are_still_caught():
    from models.recap.train_policy import AdvantageDataset

    dataset = _FrameIdACTDataset(range(60))
    with pytest.raises(ValueError, match="relabel"):
        AdvantageDataset(dataset, torch.zeros(48, dtype=torch.long), total_frames=48)
