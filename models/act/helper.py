import torch
from torch.utils.data import DataLoader, Dataset

from configs.act.base import ACTConfig
from data.episode_split import resolve_train_holdout_episodes
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.transforms.transforms import ImageTransforms, ImageTransformsConfig


def make_dataloaders(cfg: ACTConfig, episodes: list[int] | None = None):
    metadata = LeRobotDatasetMetadata(repo_id=cfg.dataset_repo_id, revision=cfg.dataset_revision)
    train_episodes, holdout_episodes = resolve_train_holdout_episodes(
        metadata.total_episodes, cfg.train_split, cfg.seed
    )
    if episodes is not None:
        # Intersect rather than replace, so the seeded train/holdout split stays
        # the one every other run used and the subset is the only difference.
        allowed = set(episodes)
        unknown = allowed - set(range(metadata.total_episodes))
        if unknown:
            raise ValueError(
                f"Episode subset references {sorted(unknown)} but "
                f"{cfg.dataset_repo_id} has {metadata.total_episodes} episodes"
            )
        train_episodes = [episode for episode in train_episodes if episode in allowed]
        holdout_episodes = [episode for episode in holdout_episodes if episode in allowed]
        if not train_episodes:
            raise ValueError("Episode subset left no training episodes")
        print(f"[make_dataloaders] restricted to {len(allowed)} requested episodes", flush=True)
    print(
        f"[make_dataloaders] {metadata.total_episodes} episodes total, holding out "
        f"{holdout_episodes} ({len(holdout_episodes)}/{metadata.total_episodes} = "
        f"{100 * len(holdout_episodes) / metadata.total_episodes:.1f}%), "
        f"training on {len(train_episodes)} episodes.",
        flush=True,
    )

    train_dataset = ACTDataset(
        cfg,
        episodes=train_episodes,
        image_transforms=cfg.image_transforms,
    )
    train_generator = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=train_generator,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = None
    if holdout_episodes:
        val_dataset = ACTDataset(cfg, episodes=holdout_episodes, image_transforms=False)
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    return train_loader, val_loader


class ACTDataset(Dataset):
    def __init__(
        self,
        cfg: ACTConfig,
        episodes: list[int] | None = None,
        image_transforms: bool | None = None,
    ):
        self.camera_names = cfg.camera_names
        self.chunk_size = cfg.chunk_size
        self.image_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self.image_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

        metadata = LeRobotDatasetMetadata(repo_id=cfg.dataset_repo_id, revision=cfg.dataset_revision)
        self.qpos_mean = torch.as_tensor(metadata.stats["observation.state"]["mean"], dtype=torch.float32)
        self.qpos_std = torch.as_tensor(metadata.stats["observation.state"]["std"], dtype=torch.float32).clamp_min(1e-6)
        self.action_mean = torch.as_tensor(metadata.stats["action"]["mean"], dtype=torch.float32)
        self.action_std = torch.as_tensor(metadata.stats["action"]["std"], dtype=torch.float32).clamp_min(1e-6)

        fps = metadata.fps

        transforms_enabled = cfg.image_transforms if image_transforms is None else image_transforms
        self.dataset = LeRobotDataset(
            repo_id=cfg.dataset_repo_id,
            revision=cfg.dataset_revision,
            episodes=episodes,
            image_transforms=ImageTransforms(
                ImageTransformsConfig(enable=transforms_enabled)
            ),
            delta_timestamps={
                "observation.state": [0.0],
                "action": [i / fps for i in range(cfg.chunk_size)],
                **{f"observation.images.{camera}": [0.0] for camera in cfg.camera_names},
            },
        )


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        qpos = sample["observation.state"].float()
        if qpos.ndim == 2 and qpos.shape[0] == 1:
            qpos = qpos[0]
        qpos = (qpos - self.qpos_mean) / self.qpos_std
        actions = (sample["action"].float() - self.action_mean) / self.action_std
        if actions.ndim != 2:
            raise ValueError(f"Expected chunked actions with shape (chunk_size, action_dim), got {tuple(actions.shape)}")
        if actions.shape[0] != self.chunk_size:
            raise ValueError(f"Expected chunk_size={self.chunk_size}, got {actions.shape[0]}")

        images = []
        for camera in self.camera_names:
            image = sample[f"observation.images.{camera}"]
            if image.ndim != 3:
                raise ValueError(f"Expected {camera} image with 3 dims, got {tuple(image.shape)}")
            if image.shape[0] not in (1, 3) and image.shape[-1] in (1, 3):
                image = image.permute(2, 0, 1).contiguous()
            if image.shape[0] != 3:
                raise ValueError(f"Expected {camera} RGB image with shape (3,H,W) or (H,W,3), got {tuple(image.shape)}")
            image = image.float()
            if image.max() > 1.0:
                image = image / 255.0
            image = (image - self.image_mean.to(image.device)) / self.image_std.to(image.device)
            images.append(image)

        is_pad = sample.get("action_is_pad")
        action_mask = torch.ones(actions.shape[0], dtype=torch.bool) if is_pad is None else ~is_pad.bool()

        return {
            "qpos": qpos,
            "images": torch.stack(images, dim=0),
            "actions": actions,
            "action_mask": action_mask,
        }

    def post_process(self, qpos, images, actions):
        return {
            "qpos": qpos * self.qpos_std + self.qpos_mean,
            "images": images * self.image_std + self.image_mean,
            "actions": actions * self.action_std + self.action_mean 
        }
