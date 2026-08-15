from types import SimpleNamespace

import models.act.helper as helper
from configs.act.base import ACTConfig


class _FakeLeRobotDataset:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __len__(self):
        return 0


def _metadata(**_kwargs):
    stats = {
        "observation.state": {"mean": [0.0] * 6, "std": [1.0] * 6},
        "action": {"mean": [0.0] * 6, "std": [1.0] * 6},
    }
    return SimpleNamespace(stats=stats, fps=30)


def test_custom_dataset_uses_lerobot_default_transforms(monkeypatch):
    monkeypatch.setattr(helper, "LeRobotDatasetMetadata", _metadata)
    monkeypatch.setattr(helper, "LeRobotDataset", _FakeLeRobotDataset)
    config = ACTConfig(dataset_repo_id="owner/dataset", image_transforms=True)

    dataset = helper.ACTDataset(config)

    transforms = dataset.dataset.kwargs["image_transforms"]
    assert transforms._cfg.enable is True
    assert set(transforms._cfg.tfs) == {
        "brightness",
        "contrast",
        "saturation",
        "hue",
        "sharpness",
        "affine",
    }


def test_custom_dataset_can_disable_transforms_for_validation(monkeypatch):
    monkeypatch.setattr(helper, "LeRobotDatasetMetadata", _metadata)
    monkeypatch.setattr(helper, "LeRobotDataset", _FakeLeRobotDataset)
    config = ACTConfig(dataset_repo_id="owner/dataset", image_transforms=True)

    dataset = helper.ACTDataset(config, image_transforms=False)

    assert dataset.dataset.kwargs["image_transforms"]._cfg.enable is False
