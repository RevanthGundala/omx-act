"""Label RECAP transitions with an advantage-conditioning token (arXiv:2511.14759).

The critic from models/recap/train_critic.py is observation-only, so the
advantage of an executed action is recovered from the reward sequence that
followed it rather than from a Q(o, a). Sec. III gives the n-step form

    A_n(o_t, a_t) = sum_{k=0}^{n-1} r_{t+k} + V(o_{t+n}) - V(o_t)

and states the setting is undiscounted, so gamma = 1 throughout. When the window
runs past the end of an episode the sum is truncated at the terminal step and the
bootstrap term is dropped, which leaves the exact remaining Monte Carlo return;
setting ``n_step >= max_episode_steps`` therefore reproduces the paper's own
estimator A = R_t - V(o_t) (Sec. IV-A) exactly.

Labels are produced in a single pass and held in memory rather than written back
as a dataset column. The critic is retrained on every RECAP iteration, so a
materialized column would force a full dataset rebuild each time, while this pass
is one no-grad forward over the frames plus vectorized arithmetic.

``UNCONDITIONED`` is deliberately not emitted at the default settings. In the
paper the unconditioned state is a classifier-free-guidance dropout token: the
indicator is binary ("Advantage: positive" / "Advantage: negative", Sec. V-B) and
is *randomly omitted* during training so that both a conditional and an
unconditional model are available at inference. Assigning it to near-zero
advantages would double-book the token, so the default deadband of 0 gives a
strict sign split and leaves conditioning dropout to the policy trainer. The
paper's own threshold is the 30th percentile of the value function's predictions
for the task (Sec. IV-B, Sec. V-D) rather than a symmetric band around zero.

Example:
    RECAP_CRITIC_CHECKPOINT=outputs/act_experiments/recap_critic/<run>/checkpoints/last.pt \
      uv run python -m models.recap.label_advantage
"""

from __future__ import annotations

from dataclasses import replace
import json
import math
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from configs.act.base import ACTConfig
from lerobot.datasets import LeRobotDataset
from models.recap.act_critic import ACTCritic
from models.recap.act_recap import AdvantageCondition
from models.recap.train_critic import NUM_BINS, CriticDataset, expected_return
from utils.config import RECORD_DATASET_REPO_ID

# The paper defines the n-step estimator but never fixes a finite n, using the
# full Monte Carlo return in practice. Rewards are a constant -1 per step, so for
# an untruncated window A_n = -n + V(o_{t+n}) - V(o_t): the label asks whether the
# critic's value improved faster than the clock over n steps. A short window keeps
# credit assignment local, a long one trades critic bias for variance.
DEFAULT_N_STEP = 5
# 0 gives an exhaustive sign split, matching the paper's binary indicator.
DEFAULT_DEADBAND = 0.0


def episode_layout(dataset: LeRobotDataset) -> dict[str, torch.Tensor]:
    """Recover per-frame episode boundaries and rewards from a RECAP dataset.

    LeRobot indexes frames with a flat global counter across concatenated
    episodes, so ``episode_end`` (the exclusive global index of each frame's own
    episode) is what keeps an n-step window from walking into the next episode.
    """
    columns = dataset.hf_dataset
    required = ("episode_index", "frame_index", "reward", "return_to_go", "terminal")
    missing = [name for name in required if name not in columns.column_names]
    if missing:
        raise ValueError(f"Dataset is missing RECAP columns {missing}; rebuild with build_dataset.py")

    episode_index = torch.as_tensor(columns["episode_index"]).long()
    frame_index = torch.as_tensor(columns["frame_index"]).long()
    rewards = torch.as_tensor(columns["reward"]).float()
    stored_returns = torch.as_tensor(columns["return_to_go"]).float()
    terminal = torch.as_tensor(columns["terminal"]).bool()
    num_frames = len(dataset)
    if episode_index.shape != (num_frames,):
        raise ValueError(f"Expected {num_frames} flat frames, got {tuple(episode_index.shape)}")

    starts = torch.nonzero(frame_index == 0).flatten()
    if starts.numel() == 0 or starts[0] != 0:
        raise ValueError("RECAP dataset does not start on the first frame of an episode")
    ends = torch.cat([starts[1:], torch.tensor([num_frames])])
    lengths = ends - starts
    episode_start = starts.repeat_interleave(lengths)
    episode_end = ends.repeat_interleave(lengths)

    if not torch.equal(frame_index, torch.arange(num_frames) - episode_start):
        raise ValueError("Frame indices are not contiguous within every episode")
    if not torch.equal(episode_index, episode_index[starts].repeat_interleave(lengths)):
        raise ValueError("Episode indices are not contiguous within every episode")
    if not torch.equal(terminal, torch.zeros(num_frames, dtype=torch.bool).index_fill_(0, ends - 1, True)):
        raise ValueError("Terminal flags do not mark exactly the last frame of every episode")

    # Undiscounted suffix sums restarted at each episode boundary; disagreeing
    # with the stored column would mean the boundaries were misread.
    suffix = torch.cat([rewards.flip(0).cumsum(0).flip(0), torch.zeros(1)])
    returns_to_go = suffix[:-1] - suffix[episode_end]
    if not torch.allclose(returns_to_go, stored_returns, atol=1e-3):
        raise ValueError("Recomputed return-to-go disagrees with the dataset column")

    return {
        "episode_index": episode_index,
        "frame_index": frame_index,
        "episode_end": episode_end,
        "rewards": rewards,
        "returns_to_go": returns_to_go,
    }


def n_step_advantages(
    returns_to_go: torch.Tensor,
    values: torch.Tensor,
    episode_end: torch.Tensor,
    n_step: int = DEFAULT_N_STEP,
) -> torch.Tensor:
    """Undiscounted n-step advantage, truncated at the terminal step.

    ``returns_to_go[t] - returns_to_go[t + n]`` is the reward sum over the window
    because both are suffix sums of the same episode. Once ``t + n`` reaches the
    episode end there is no next observation to bootstrap from, so the estimate
    falls back to the exact remaining return ``returns_to_go[t] - values[t]``.
    """
    if n_step < 1:
        raise ValueError(f"n_step must be at least 1, got {n_step}")
    if values.shape != returns_to_go.shape or episode_end.shape != returns_to_go.shape:
        raise ValueError(
            "returns_to_go, values and episode_end must be aligned per frame, got "
            f"{tuple(returns_to_go.shape)}, {tuple(values.shape)}, {tuple(episode_end.shape)}"
        )
    num_frames = returns_to_go.shape[0]
    index = torch.arange(num_frames)
    bootstrap = (index + n_step) < episode_end
    # Clamped so the gather can never read the next episode; masked lanes are
    # discarded by torch.where below.
    target = (index + n_step).clamp(max=num_frames - 1)
    zero = torch.zeros((), dtype=returns_to_go.dtype)
    future_return = torch.where(bootstrap, returns_to_go[target], zero)
    future_value = torch.where(bootstrap, values[target], zero)
    return returns_to_go - future_return + future_value - values


def advantage_conditions(
    advantages: torch.Tensor,
    deadband: float = DEFAULT_DEADBAND,
) -> torch.Tensor:
    """Map scalar advantages onto AdvantageCondition class IDs.

    A deadband of 0 reproduces the paper's indicator 1(A > epsilon) with
    epsilon = 0 and never emits UNCONDITIONED. A positive deadband is a
    deviation from the paper that parks ambiguous transitions on the
    unconditioned token.
    """
    if not math.isfinite(deadband) or deadband < 0:
        raise ValueError(f"deadband must be finite and non-negative, got {deadband}")
    conditions = torch.full(advantages.shape, int(AdvantageCondition.UNCONDITIONED), dtype=torch.long)
    conditions[advantages > deadband] = int(AdvantageCondition.POSITIVE)
    conditions[advantages <= -deadband] = int(AdvantageCondition.NEGATIVE)
    return conditions


def load_critic(checkpoint_path: str | Path, device: torch.device) -> tuple[ACTCritic, float, float]:
    """Rebuild a trained critic along with the value range it was trained on."""
    checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=True)
    cfg = ACTConfig(**checkpoint["config"])
    model = ACTCritic(cfg, pretrained_backbone=False, num_bins=checkpoint.get("num_bins", NUM_BINS))
    model.load_state_dict(checkpoint["model_state_dict"])
    # Decode with the checkpoint's own range so labels match how it was trained.
    return model.to(device).eval(), float(checkpoint["value_min"]), float(checkpoint["value_max"])


@torch.no_grad()
def critic_values(
    model: ACTCritic,
    dataset: CriticDataset,
    device: torch.device,
    value_min: float,
    value_max: float,
    batch_size: int = 8,
    num_workers: int = 0,
) -> torch.Tensor:
    """Score every observation in dataset order, in raw return units."""
    was_training = model.training
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    values = [
        expected_return(
            model(batch["images"].to(device), batch["qpos"].to(device)), value_min, value_max
        ).cpu()
        for batch in loader
    ]
    model.train(was_training)
    return torch.cat(values).float()


def label_dataset(
    model: ACTCritic,
    dataset: CriticDataset,
    device: torch.device,
    value_min: float,
    value_max: float,
    n_step: int = DEFAULT_N_STEP,
    deadband: float = DEFAULT_DEADBAND,
    batch_size: int = 8,
    num_workers: int = 0,
) -> dict[str, torch.Tensor]:
    layout = episode_layout(dataset.dataset)
    values = critic_values(
        model, dataset, device, value_min, value_max, batch_size=batch_size, num_workers=num_workers
    )
    advantages = n_step_advantages(layout["returns_to_go"], values, layout["episode_end"], n_step)
    return {
        "values": values,
        "advantages": advantages,
        "conditions": advantage_conditions(advantages, deadband),
    }


def config_from_env() -> ACTConfig:
    base_cfg = ACTConfig(
        dataset_repo_id=os.environ.get("RECAP_ADVANTAGE_DATASET_REPO_ID")
        or os.environ.get("RECAP_CRITIC_DATASET_REPO_ID")
        or RECORD_DATASET_REPO_ID
    )
    return replace(base_cfg, job_name=os.environ.get("RECAP_ADVANTAGE_JOB_NAME", "recap_advantage"))


def main():
    cfg = config_from_env()
    checkpoint_path = os.environ.get("RECAP_CRITIC_CHECKPOINT")
    if not checkpoint_path:
        raise ValueError("Set RECAP_CRITIC_CHECKPOINT to a checkpoint from models/recap/train_critic.py")
    n_step = int(os.environ.get("RECAP_ADVANTAGE_N_STEP", DEFAULT_N_STEP))
    deadband = float(os.environ.get("RECAP_ADVANTAGE_DEADBAND", DEFAULT_DEADBAND))
    device_name = os.environ.get("RECAP_DEVICE")
    if device_name is None:
        device_name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_name)

    dataset = CriticDataset(cfg)
    model, value_min, value_max = load_critic(checkpoint_path, device)
    labels = label_dataset(
        model,
        dataset,
        device,
        value_min,
        value_max,
        n_step=n_step,
        deadband=deadband,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    conditions = labels["conditions"]
    counts = {
        condition.name: int((conditions == int(condition)).sum()) for condition in AdvantageCondition
    }
    output_path = Path(
        os.environ.get("RECAP_ADVANTAGE_OUTPUT", dataset.dataset.root / "advantage_labels.json")
    )
    output_path.write_text(
        json.dumps(
            {
                "dataset_repo_id": cfg.dataset_repo_id,
                "critic_checkpoint": str(checkpoint_path),
                "n_step": n_step,
                "deadband": deadband,
                "value_min": value_min,
                "value_max": value_max,
                "counts": counts,
                "conditions": conditions.tolist(),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Dataset: {cfg.dataset_repo_id}@{cfg.dataset_revision} ({len(conditions)} transitions)")
    print(f"n_step={n_step} deadband={deadband:g} advantage_mean={float(labels['advantages'].mean()):.3f}")
    print(f"Labels: {counts}")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
