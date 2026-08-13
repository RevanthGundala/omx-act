"""Sanity-check the RECAP dataset with a tiny state-only MLP.

The full critic is a 35M-parameter vision model, so a poor result there is
ambiguous: bad data, bad normalization, or an under-trained backbone. This
strips the problem to its smallest form -- joint state in, return out -- so a
failure points squarely at the dataset or the value normalization.

The MLP cannot see the sponge, so it should NOT match a vision critic. What it
must do is beat the mean-prediction baseline, which is only possible if the
returns carry learnable structure.

Usage:
    PYTHONPATH=. uv run python -m models.recap.sanity_mlp \
        --dataset-root outputs/datasets/031-recap-gen1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from models.recap.train_critic import (
    NUM_BINS,
    bin_centers,
    normalize_return,
    return_to_bin,
)


def load_columns(dataset_root: Path) -> dict[str, np.ndarray]:
    """Read scalar columns straight from parquet; no video decode."""
    import pandas as pd

    files = sorted((dataset_root / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet data under {dataset_root}")
    frame = pd.concat([pd.read_parquet(path) for path in files]).reset_index(drop=True)
    return {
        "state": np.stack(frame["observation.state"].apply(np.asarray).values).astype(
            np.float32
        ),
        "returns": np.asarray(frame["return_to_go"], dtype=np.float32).ravel(),
        "episode": np.asarray(frame["episode_index"]).ravel(),
        "success": np.asarray(frame["success"]).ravel().astype(bool),
        "progress": np.asarray(frame["frame_index"], dtype=np.float32).ravel(),
    }


def value_range(dataset_root: Path) -> tuple[float, float]:
    manifest = json.loads((dataset_root / "recap_manifest.json").read_text())
    max_steps = int(manifest["max_episode_steps"])
    penalty = float(manifest["failure_penalty"])
    return -(max_steps - 1) - penalty, 0.0


class SanityMLP(nn.Module):
    """Joint state -> distribution over the same bins the real critic uses."""

    def __init__(self, input_dim: int, num_bins: int = NUM_BINS, width: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, num_bins),
        )

    def forward(self, x):
        return self.net(x)


def evaluate(model, features, returns, centers, value_min, value_max):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        probabilities = torch.softmax(logits, dim=-1)
        predicted = (probabilities * centers).sum(dim=-1)
        targets = return_to_bin(returns, value_min, value_max)
        loss = nn.functional.cross_entropy(logits, targets)
    model.train()
    return float(loss), predicted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--val-episodes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--with-progress",
        action="store_true",
        help=(
            "Append normalized frame index to the input. Returns are largely a "
            "countdown, so this checks how much is explained by time alone."
        ),
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    root = args.dataset_root.expanduser().resolve()
    columns = load_columns(root)
    value_min, value_max = value_range(root)

    outcomes = {
        int(e): bool(columns["success"][columns["episode"] == e][0])
        for e in np.unique(columns["episode"])
    }
    from models.recap.train_critic import split_episodes

    train_eps, val_eps = split_episodes(outcomes, args.val_episodes, args.seed)
    train_mask = np.isin(columns["episode"], train_eps)
    val_mask = np.isin(columns["episode"], val_eps)

    state = columns["state"]
    if args.with_progress:
        span = np.maximum(columns["progress"].max(), 1.0)
        state = np.concatenate(
            [state, (columns["progress"] / span)[:, None].astype(np.float32)], axis=1
        )

    mean = state[train_mask].mean(axis=0)
    std = state[train_mask].std(axis=0).clip(min=1e-6)
    normalized = (state - mean) / std

    x_train = torch.from_numpy(normalized[train_mask])
    y_train = torch.from_numpy(columns["returns"][train_mask])
    x_val = torch.from_numpy(normalized[val_mask])
    y_val = torch.from_numpy(columns["returns"][val_mask])

    print(f"train frames={len(x_train)} val frames={len(x_val)}")
    print(f"value range=[{value_min:g}, {value_max:g}] bins={NUM_BINS}")
    print(f"held-out episodes={val_eps} outcomes={[outcomes[e] for e in val_eps]}")

    model = SanityMLP(x_train.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    centers = bin_centers(value_min, value_max)
    targets = return_to_bin(y_train, value_min, value_max)

    generator = torch.Generator().manual_seed(args.seed)
    for step in range(args.steps):
        index = torch.randint(0, len(x_train), (256,), generator=generator)
        logits = model(x_train[index])
        loss = nn.functional.cross_entropy(logits, targets[index])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 500 == 0:
            print(f"  step={step} train_loss={float(loss):.4f}")

    val_loss, predicted = evaluate(model, x_val, y_val, centers, value_min, value_max)
    mae = float((predicted - y_val).abs().mean())
    baseline = float((y_train.mean() - y_val).abs().mean())

    print()
    print(f"val cross-entropy : {val_loss:.4f}   (chance = {np.log(NUM_BINS):.4f})")
    print(f"val return MAE    : {mae:.1f}")
    print(f"mean baseline MAE : {baseline:.1f}")
    print(f"improvement       : {100 * (1 - mae / baseline):+.1f}%")

    val_success = torch.from_numpy(columns["success"][val_mask])
    if val_success.any() and (~val_success).any():
        positive = predicted[val_success]
        negative = predicted[~val_success]
        wins = (positive[:, None] > negative[None, :]).float().mean()
        ties = (positive[:, None] == negative[None, :]).float().mean()
        print(f"success AUC       : {float(wins + 0.5 * ties):.3f}   (chance = 0.500)")

    normalized_returns = normalize_return(y_train, value_min, value_max)
    print()
    print("normalization check (train returns -> (-1, 0)):")
    print(f"  min={float(normalized_returns.min()):.4f} max={float(normalized_returns.max()):.4f}")
    bins = return_to_bin(y_train, value_min, value_max)
    print(f"  bin index range=[{int(bins.min())}, {int(bins.max())}] of {NUM_BINS}")
    print(f"  distinct bins used={len(torch.unique(bins))}")


if __name__ == "__main__":
    main()
