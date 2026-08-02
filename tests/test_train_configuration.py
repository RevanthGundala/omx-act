from dataclasses import replace
import json

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


def test_config_from_env_overrides_output_root(monkeypatch):
    monkeypatch.setenv("ACT_TRAIN_DATASET_REPO_ID", "owner/dataset")
    monkeypatch.setenv("ACT_TRAIN_OUTPUT_ROOT", "/outputs/act_experiments")

    assert config_from_env().output_root == "/outputs/act_experiments"


def test_config_from_env_enables_default_image_transforms(monkeypatch):
    monkeypatch.setenv("ACT_TRAIN_IMAGE_TRANSFORMS", "true")

    assert config_from_env().image_transforms is True


def test_config_from_env_rejects_invalid_image_transforms(monkeypatch):
    monkeypatch.setenv("ACT_TRAIN_IMAGE_TRANSFORMS", "sometimes")

    with pytest.raises(ValueError, match="ACT_TRAIN_IMAGE_TRANSFORMS"):
        config_from_env()


def test_seed_training_controls_numpy_and_torch():
    train_module.seed_training(123)
    first_numpy = np.random.random(4)
    first_torch = torch.rand(4)

    train_module.seed_training(123)

    assert np.array_equal(np.random.random(4), first_numpy)
    assert torch.equal(torch.rand(4), first_torch)


def test_write_training_status_records_progress(tmp_path):
    status_path = tmp_path / "monitor" / "status.json"

    train_module.write_training_status(
        str(status_path),
        {"status": "running", "step": 100, "metrics": {"loss": 0.5}},
    )

    payload = json.loads(status_path.read_text())
    assert payload["status"] == "running"
    assert payload["step"] == 100
    assert payload["metrics"]["loss"] == 0.5
    assert payload["timestamp"]


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


def test_resume_rejects_legacy_custom_architecture(monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "resume.pt"
    torch.save(
        {
            "config": {},
            "model_state_dict": {"decoder.camera_emb": torch.zeros(1)},
            "optimizer_state_dict": {},
            "step": 0,
        },
        checkpoint_path,
    )
    config = replace(
        ACTConfig(dataset_repo_id="owner/dataset"),
        output_root=str(tmp_path),
        job_name="resume-parity",
        num_train_steps=0,
        num_workers=0,
    )
    monkeypatch.setattr(train_module, "config_from_env", lambda: config)
    monkeypatch.setenv("ACT_DISABLE_MLFLOW", "1")
    monkeypatch.setenv("ACT_DEVICE", "cpu")
    monkeypatch.setenv("ACT_TRAIN_RESUME_CHECKPOINT", str(checkpoint_path))

    with pytest.raises(ValueError, match="legacy custom ACT architecture"):
        train_module.main()


def test_pretrained_checkpoint_loads_weights_with_fresh_optimizer(monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "pretrained.pt"
    torch.save(
        {
            "config": {"dataset_repo_id": "owner/seed"},
            "model_state_dict": {"decoder.final_norm.weight": torch.ones(1)},
            "optimizer_state_dict": {"must_not_load": True},
            "step": 20_000,
        },
        checkpoint_path,
    )
    config = replace(
        ACTConfig(dataset_repo_id="owner/dagger"),
        output_root=str(tmp_path),
        job_name="dagger-finetune",
        num_train_steps=0,
        num_workers=0,
    )
    events = []

    class FakeModel:
        def to(self, _device):
            return self

        def parameters(self):
            return []

        def load_state_dict(self, state_dict):
            events.append(("model_load", state_dict))

        def train(self):
            return self

    class FakeOptimizer:
        def load_state_dict(self, state_dict):
            events.append(("optimizer_load", state_dict))

    monkeypatch.setattr(train_module, "config_from_env", lambda: config)
    monkeypatch.setattr(train_module, "ACTPolicy", lambda *_args, **_kwargs: FakeModel())
    monkeypatch.setattr(
        train_module.torch.optim,
        "AdamW",
        lambda *_args, **_kwargs: FakeOptimizer(),
    )
    monkeypatch.setattr(train_module, "make_dataloaders", lambda _config: ([], None))
    monkeypatch.setenv("ACT_DISABLE_MLFLOW", "1")
    monkeypatch.setenv("ACT_DEVICE", "cpu")
    monkeypatch.setenv("ACT_TRAIN_PRETRAINED_CHECKPOINT", str(checkpoint_path))

    train_module.main()

    assert [name for name, _value in events] == ["model_load"]
