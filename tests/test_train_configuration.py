import pytest

from models.act.train import config_from_env


def test_config_from_env_can_train_on_all_episodes(monkeypatch):
    monkeypatch.setenv("ACT_TRAIN_DATASET_REPO_ID", "owner/dataset")
    monkeypatch.setenv("ACT_TRAIN_SPLIT", "1.0")

    config = config_from_env()

    assert config.dataset_repo_id == "owner/dataset"
    assert config.train_split == 1.0


def test_config_from_env_rejects_invalid_split(monkeypatch):
    monkeypatch.setenv("ACT_TRAIN_DATASET_REPO_ID", "owner/dataset")
    monkeypatch.setenv("ACT_TRAIN_SPLIT", "0")

    with pytest.raises(ValueError, match="ACT_TRAIN_SPLIT"):
        config_from_env()


def test_config_from_env_enables_stock_parity(monkeypatch):
    monkeypatch.setenv("ACT_TRAIN_DATASET_REPO_ID", "owner/dataset")
    monkeypatch.setenv("ACT_TRAIN_STOCK_PARITY", "1")

    assert config_from_env().stock_parity
