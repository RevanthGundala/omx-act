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

``UNCONDITIONED`` is never emitted on the default path. In the paper the
unconditioned state is a classifier-free-guidance dropout token: the indicator is
binary ("Advantage: positive" / "Advantage: negative", Sec. V-B) and is *randomly
omitted* during training so both a conditional and an unconditional model are
available at inference. Assigning it to near-zero advantages would double-book the
token, so conditioning dropout is left to the policy trainer.

The indicator threshold is a **percentile of the advantage distribution**, taken
per task, not a fixed epsilon. Sec. IV-B thresholds the advantage directly,
``delta(A(o, a, l) > eps_l)``, so ``eps_l`` carries advantage units. Sec. V-D
describes it as "the 30% percentile of values predicted by the value function",
but Appendix A gives the operational definition: the threshold is chosen "such
that approximately 30% of the demonstration data has positive advantage". Only
the advantage reading is dimensionally coherent - values are normalized to
(-1, 0) while advantages concentrate near 0, so an absolute value-percentile
around -0.7 would mark essentially every transition positive.

Direction: the paper's 30% is the fraction labeled POSITIVE, so eps_l sits at the
*70th* percentile of advantages. Appendix A confirms the sign of the relation on
its own terms - it "increase[s] the threshold such that only approximately 10% of
the data has positive advantage", and a higher threshold can only reduce the
positive fraction. This module is therefore parameterized by the target positive
fraction rather than by a raw percentile, so the direction cannot be misread.

A fixed absolute threshold, optionally with a symmetric deadband, is kept for
debugging and ablations via the ``threshold`` and ``deadband`` arguments, but it
is not the default.

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
# The paper sets the indicator threshold eps_l from a percentile rather than
# fixing it at zero (Sec. IV-B, Appendix A). A learned critic is rarely calibrated
# well enough for A > 0 to be meaningful: a systematically biased critic pushes
# nearly every transition to one label, which makes the conditioning token carry
# no information. A percentile keeps the positive/negative split usable whatever
# the bias. Parameterized as the target POSITIVE fraction, which is what Appendix
# A actually pins down (~30% positive in pre-training, ~40% in fine-tuning).
DEFAULT_POSITIVE_FRACTION = 0.30


def task_percentile_thresholds(
    advantages: torch.Tensor,
    task_index: torch.Tensor | None = None,
    positive_fraction: float = DEFAULT_POSITIVE_FRACTION,
) -> torch.Tensor:
    """Resolve eps_l per task so that ``positive_fraction`` of it scores POSITIVE.

    eps_l is the ``1 - positive_fraction`` quantile of the task's advantages, so
    a target of 0.30 puts eps_l at the 70th percentile and leaves ~30% above it.

    A quantile is a global statistic, so the whole advantage vector has to exist
    before any label can be assigned; it cannot be derived per sample or per
    batch. The paper takes the threshold per task ("for the task l", Sec. V-D),
    so the quantile is computed within each ``task_index`` group and broadcast
    back over that group's frames.

    Units: Sec. IV-B applies the threshold to the advantage itself,
    ``delta(A(o, a, l) > eps_l)``, and Appendix A fixes it by the fraction of
    data with positive *advantage*, so the quantile is taken over advantages
    rather than over the critic's value predictions.
    """
    if not math.isfinite(positive_fraction) or not 0.0 <= positive_fraction <= 1.0:
        raise ValueError(f"positive_fraction must be in [0, 1], got {positive_fraction}")
    if advantages.numel() == 0:
        raise ValueError("Cannot take a percentile of an empty advantage tensor")
    quantile = 1.0 - positive_fraction
    if task_index is None:
        return torch.full_like(advantages, float(torch.quantile(advantages.double(), quantile)))
    if task_index.shape != advantages.shape:
        raise ValueError(
            f"task_index must be aligned per frame, got {tuple(task_index.shape)} "
            f"for {tuple(advantages.shape)} advantages"
        )
    thresholds = torch.empty_like(advantages)
    for task in torch.unique(task_index):
        selected = task_index == task
        thresholds[selected] = float(torch.quantile(advantages[selected].double(), quantile))
    return thresholds

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
    # The paper thresholds per task, so carry the task grouping alongside.
    task_index = (
        torch.as_tensor(columns["task_index"]).long()
        if "task_index" in columns.column_names
        else torch.zeros(len(dataset), dtype=torch.long)
    )
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
        "task_index": task_index,
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
    threshold: torch.Tensor | float = 0.0,
    deadband: float = DEFAULT_DEADBAND,
) -> torch.Tensor:
    """Map scalar advantages onto AdvantageCondition class IDs.

    Implements the paper's indicator 1(A > eps_l), where ``threshold`` is eps_l
    and may be a scalar or one value per frame (per-task thresholds). A deadband
    of 0 is exhaustive and never emits UNCONDITIONED. A positive deadband is a
    deviation from the paper that parks ambiguous transitions on the
    unconditioned token, which otherwise belongs to conditioning dropout.
    """
    if not math.isfinite(deadband) or deadband < 0:
        raise ValueError(f"deadband must be finite and non-negative, got {deadband}")
    threshold = torch.as_tensor(threshold, dtype=advantages.dtype)
    if threshold.ndim and threshold.shape != advantages.shape:
        raise ValueError(
            f"threshold must be scalar or aligned per frame, got {tuple(threshold.shape)} "
            f"for {tuple(advantages.shape)} advantages"
        )
    if not torch.isfinite(threshold).all():
        raise ValueError("threshold must be finite")
    conditions = torch.full(advantages.shape, int(AdvantageCondition.UNCONDITIONED), dtype=torch.long)
    conditions[advantages > threshold + deadband] = int(AdvantageCondition.POSITIVE)
    conditions[advantages <= threshold - deadband] = int(AdvantageCondition.NEGATIVE)
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
    positive_fraction: float | None = DEFAULT_POSITIVE_FRACTION,
    threshold: float | None = None,
    batch_size: int = 8,
    num_workers: int = 0,
) -> dict[str, torch.Tensor]:
    """Label every transition, resolving eps_l from a per-task positive fraction.

    The advantage vector is built in full before any threshold is taken, because
    a quantile is a global statistic over the labeled set. Pass ``threshold`` to
    pin eps_l to an absolute value instead (useful for debugging); it takes
    precedence over ``positive_fraction``.
    """
    layout = episode_layout(dataset.dataset)
    values = critic_values(
        model, dataset, device, value_min, value_max, batch_size=batch_size, num_workers=num_workers
    )
    # First pass: every advantage. Second pass: the threshold that needs them all.
    advantages = n_step_advantages(layout["returns_to_go"], values, layout["episode_end"], n_step)
    if threshold is None:
        if positive_fraction is None:
            raise ValueError("Provide either a positive_fraction or an absolute threshold")
        thresholds = task_percentile_thresholds(
            advantages, layout["task_index"], positive_fraction
        )
    else:
        thresholds = torch.full_like(advantages, float(threshold))
    return {
        "values": values,
        "advantages": advantages,
        "conditions": advantage_conditions(advantages, thresholds, deadband),
        "thresholds": thresholds,
        "task_index": layout["task_index"],
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
    positive_fraction = float(
        os.environ.get("RECAP_ADVANTAGE_POSITIVE_FRACTION", DEFAULT_POSITIVE_FRACTION)
    )
    absolute = os.environ.get("RECAP_ADVANTAGE_THRESHOLD")
    threshold = float(absolute) if absolute is not None else None
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
        positive_fraction=positive_fraction,
        threshold=threshold,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    conditions = labels["conditions"]
    counts = {
        condition.name: int((conditions == int(condition)).sum()) for condition in AdvantageCondition
    }
    task_index = labels["task_index"]
    task_thresholds = {
        str(int(task)): float(labels["thresholds"][task_index == task][0])
        for task in torch.unique(task_index)
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
                "positive_fraction": None if threshold is not None else positive_fraction,
                "thresholds_by_task": task_thresholds,
                "value_min": value_min,
                "value_max": value_max,
                "counts": counts,
                "conditions": conditions.tolist(),
            },
            indent=2,
        )
        + "\n"
    )
    positive = counts[AdvantageCondition.POSITIVE.name]
    print(f"Dataset: {cfg.dataset_repo_id}@{cfg.dataset_revision} ({len(conditions)} transitions)")
    source = "absolute" if threshold is not None else f"positive_fraction={positive_fraction:g}"
    print(
        f"n_step={n_step} eps_l per task={task_thresholds} ({source}) "
        f"deadband={deadband:g} advantage_mean={float(labels['advantages'].mean()):.3f}"
    )
    print(f"Labels: {counts} (positive share {positive / max(1, len(conditions)):.1%})")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
