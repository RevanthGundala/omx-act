from dataclasses import replace
import json

from lerobot.datasets import LeRobotDataset
import numpy as np
import pytest
import torch

from configs.act.base import ACTConfig
from models.act.act import ACTPolicy as BaseACTPolicy
from models.act.attention import Attention as BaseAttention
from models.recap import label_advantage, train_critic, train_policy
from models.recap.act_critic import ACTCritic
from models.recap.act_recap import ACTPolicy as RecapACTPolicy, AdvantageCondition
from models.recap.attention import Attention as RecapAttention
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


def test_advantage_labeling_end_to_end_on_cpu(tmp_path):
    torch.manual_seed(0)
    root = tmp_path / "dataset"
    _build_recap_dataset(root, "owner/recap_labels", [(3, True), (5, False)])
    cfg = replace(_labeling_config(), dataset_repo_id="owner/recap_labels")
    device = torch.device("cpu")
    value_min, value_max = train_critic.load_return_range(root)

    critic = ACTCritic(cfg, pretrained_backbone=False, num_bins=train_critic.NUM_BINS)
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

    dataset = train_critic.CriticDataset(cfg, root=root)
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
            label_advantage.DEFAULT_POSITIVE_FRACTION,
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
