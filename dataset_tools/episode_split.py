"""Deterministic episode-level train/holdout split, shared between models/act/helper.py (the
custom trainer's dataloaders) and train/vastai/lerobot_official.py (the stock lerobot-train
launcher), so both trainers hold out the SAME fraction of data using the SAME algorithm for a fair
comparison.

Episode-level, not frame-level: holds out whole episodes so no validation frame is a
near-duplicate of a training frame from the same episode. See docs/act_debugging_learnings.md /
session history for why the previous frame-level random_split in models/act/helper.py was a
methodological weakness (an optimistic val_loss, though it never affected actual training content
or any of the real decisions made this session -- see chat history around 2026-07-10).
"""
from __future__ import annotations

import random


def resolve_train_holdout_episodes(
    n_episodes: int,
    train_split: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Return (train_episodes, holdout_episodes), both sorted lists of episode indices summing to
    n_episodes. If train_split >= 1.0, holdout_episodes is empty (train on everything).

    Deterministic given the same (n_episodes, train_split, seed) -- uses Python's `random` module
    seeded fresh each call, not global RNG state, so results don't depend on call order/history.
    """
    if train_split >= 1.0:
        return list(range(n_episodes)), []
    n_holdout = max(1, round(n_episodes * (1.0 - train_split)))
    n_holdout = min(n_holdout, n_episodes - 1)  # always keep at least one training episode
    rng = random.Random(seed)
    holdout = sorted(rng.sample(range(n_episodes), k=n_holdout))
    train_episodes = sorted(set(range(n_episodes)) - set(holdout))
    return train_episodes, holdout
