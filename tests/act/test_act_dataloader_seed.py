from types import SimpleNamespace

import torch
from torch.utils.data import Dataset

import models.act.helper as helper
from configs.act.base import ACTConfig


class _IndexDataset(Dataset):
    def __init__(self, _config, episodes, image_transforms):
        assert episodes == [0]
        assert image_transforms is False
        self.values = torch.arange(12)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]


def test_training_shuffle_is_independent_of_prior_global_rng_use(monkeypatch):
    monkeypatch.setattr(
        helper,
        "LeRobotDatasetMetadata",
        lambda **_kwargs: SimpleNamespace(total_episodes=1),
    )
    monkeypatch.setattr(
        helper,
        "resolve_train_holdout_episodes",
        lambda *_args: ([0], []),
    )
    monkeypatch.setattr(helper, "ACTDataset", _IndexDataset)
    config = ACTConfig(
        dataset_repo_id="owner/dataset",
        batch_size=3,
        num_workers=0,
        train_split=1.0,
        seed=123,
    )

    def shuffled_order(random_draws):
        torch.manual_seed(999)
        train_loader, _ = helper.make_dataloaders(config)
        torch.rand(random_draws)
        return torch.cat(list(train_loader)).tolist()

    assert shuffled_order(1) == shuffled_order(100)
