"""Unit tests for data.episode_split.resolve_train_holdout_episodes."""
import pytest

from dataset_tools.episode_split import resolve_train_holdout_episodes


def test_no_holdout_when_train_split_is_one():
    train, holdout = resolve_train_holdout_episodes(50, 1.0, seed=1000)
    assert train == list(range(50))
    assert holdout == []


def test_holdout_and_train_partition_all_episodes():
    train, holdout = resolve_train_holdout_episodes(50, 0.95, seed=1000)
    assert sorted(train + holdout) == list(range(50))
    assert set(train).isdisjoint(holdout)


def test_holdout_fraction_is_approximately_correct():
    train, holdout = resolve_train_holdout_episodes(50, 0.95, seed=1000)
    # ~5% of 50 -> 2 or 3 depending on floating point rounding at the exact boundary; either is a
    # correct interpretation of "~5%".
    assert len(holdout) in (2, 3)
    assert len(train) == 50 - len(holdout)


def test_deterministic_given_same_seed():
    train1, holdout1 = resolve_train_holdout_episodes(50, 0.95, seed=1000)
    train2, holdout2 = resolve_train_holdout_episodes(50, 0.95, seed=1000)
    assert train1 == train2
    assert holdout1 == holdout2


def test_different_seed_gives_different_holdout():
    _, holdout_a = resolve_train_holdout_episodes(50, 0.95, seed=1000)
    _, holdout_b = resolve_train_holdout_episodes(50, 0.95, seed=2000)
    assert holdout_a != holdout_b


def test_always_keeps_at_least_one_training_episode():
    train, holdout = resolve_train_holdout_episodes(2, 0.0, seed=1000)
    assert len(train) >= 1


def test_at_least_one_holdout_when_split_less_than_one():
    train, holdout = resolve_train_holdout_episodes(50, 0.999, seed=1000)
    assert len(holdout) >= 1


@pytest.mark.parametrize("n_episodes,train_split", [(50, 0.95), (10, 0.9), (100, 0.8), (3, 0.5)])
def test_partition_covers_all_episodes_various_sizes(n_episodes, train_split):
    train, holdout = resolve_train_holdout_episodes(n_episodes, train_split, seed=1000)
    assert sorted(train + holdout) == list(range(n_episodes))
