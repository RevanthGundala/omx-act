from dataclasses import replace

import numpy as np
import pytest
import torch

import models.act.train as train_module
from configs.act.base import ACTConfig
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


def test_seed_training_controls_numpy_and_torch():
    train_module.seed_training(123)
    first_numpy = np.random.random(4)
    first_torch = torch.rand(4)

    train_module.seed_training(123)

    assert np.array_equal(np.random.random(4), first_numpy)
    assert torch.equal(torch.rand(4), first_torch)


def test_main_seeds_before_model_initialization(monkeypatch, tmp_path):
    events = []
    config = replace(
        ACTConfig(dataset_repo_id="owner/dataset"),
        output_root=str(tmp_path),
        job_name="seed-order",
        num_train_steps=0,
        num_workers=0,
        seed=123,
    )

    class FakeModel:
        def to(self, _device):
            return self

        def parameters(self):
            return []

        def train(self):
            return self

    monkeypatch.setattr(train_module, "config_from_env", lambda: config)
    monkeypatch.setattr(
        train_module,
        "seed_training",
        lambda seed: events.append(("seed", seed)),
        raising=False,
    )
    monkeypatch.setattr(
        train_module,
        "ACTPolicy",
        lambda *_args, **_kwargs: events.append(("model", None)) or FakeModel(),
    )
    monkeypatch.setattr(train_module.torch.optim, "AdamW", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(train_module, "make_dataloaders", lambda _config: ([], None))
    monkeypatch.setenv("ACT_DISABLE_MLFLOW", "1")
    monkeypatch.setenv("ACT_DEVICE", "cpu")

    train_module.main()

    assert events[:2] == [("seed", 123), ("model", None)]
