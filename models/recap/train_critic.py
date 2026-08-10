"""Train the RECAP distributional critic (arXiv:2511.14759) on Monte Carlo returns.

The critic is an observation-only V(o) trained by cross-entropy against the
undiscounted empirical return-to-go that models/recap/build_dataset.py already
stores in the dataset. There is no discount factor and no Bellman bootstrapping
(Sec. III), returns are discretized into 201 bins (Sec. IV-A), and values are
normalized into (-1, 0) per task using the maximum episode length (Sec. V-C).
"""

from dataclasses import asdict, replace
from datetime import datetime
import json
import os
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from configs.act.base import ACTConfig
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.transforms.transforms import ImageTransforms, ImageTransformsConfig
from models.recap.act_critic import ACTCritic
from utils.config import RECORD_DATASET_REPO_ID

# Sec. IV-A: the empirical return is discretized into B = 201 bins.
NUM_BINS = 201
IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGE_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def load_return_range(dataset_root: Path) -> tuple[float, float]:
    """Derive the raw return-to-go range that maps onto (-1, 0).

    build_dataset.py emits -1 per step with a terminal reward of 0 on success
    and -failure_penalty on failure, undiscounted. The best achievable return is
    therefore 0 (success at the terminal step) and the worst is the longest
    episode failing on its final step.
    """
    manifest_path = Path(dataset_root) / "recap_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {manifest_path}; build the dataset with models/recap/build_dataset.py."
        )
    manifest = json.loads(manifest_path.read_text())
    max_episode_steps = int(manifest["max_episode_steps"])
    failure_penalty = float(manifest["failure_penalty"])
    return -(max_episode_steps - 1) - failure_penalty, 0.0


def normalize_return(returns, value_min: float, value_max: float):
    """Map raw return-to-go into the paper's (-1, 0) value range (Sec. V-C)."""
    return ((returns - value_max) / (value_max - value_min)).clamp(-1.0, 0.0)


def denormalize_return(values, value_min: float, value_max: float):
    return values * (value_max - value_min) + value_max


def bin_centers(value_min: float, value_max: float, num_bins: int = NUM_BINS, device=None):
    index = torch.arange(num_bins, dtype=torch.float32, device=device)
    return denormalize_return(-1.0 + (index + 0.5) / num_bins, value_min, value_max)


def expected_return(distribution, value_min: float, value_max: float):
    """Decode a predicted bin distribution back into raw return units."""
    logits = distribution.logits
    centers = bin_centers(value_min, value_max, logits.shape[-1], logits.device)
    return (distribution.probs * centers).sum(dim=-1)


def compute_loss(model, batch, device, value_min: float, value_max: float):
    images = batch["images"].to(device)
    qpos = batch["qpos"].to(device)
    returns = batch["return_to_go"].to(device)
    distribution = model(images, qpos)
    num_bins = distribution.logits.shape[-1]
    normalized = normalize_return(returns, value_min, value_max)
    # The paper fixes B = 201 but is silent on bin spacing (uniform vs. HL-Gauss),
    # so we use uniform bins over the normalized (-1, 0) range.
    targets = ((normalized + 1.0) * num_bins).long().clamp(0, num_bins - 1)
    # Sec. IV-A: cross-entropy against the discretized empirical MC return.
    loss = F.cross_entropy(distribution.logits, targets)
    return loss, distribution


class CriticDataset(Dataset):
    """Observation plus Monte Carlo return-to-go; the critic never sees actions."""

    def __init__(self, cfg: ACTConfig, root: str | Path | None = None):
        metadata = LeRobotDatasetMetadata(
            repo_id=cfg.dataset_repo_id, revision=cfg.dataset_revision, root=root
        )
        self.camera_names = cfg.camera_names
        self.qpos_mean = torch.as_tensor(metadata.stats["observation.state"]["mean"], dtype=torch.float32)
        self.qpos_std = torch.as_tensor(
            metadata.stats["observation.state"]["std"], dtype=torch.float32
        ).clamp_min(1e-6)
        self.dataset = LeRobotDataset(
            repo_id=cfg.dataset_repo_id,
            revision=cfg.dataset_revision,
            root=root,
            image_transforms=ImageTransforms(ImageTransformsConfig(enable=cfg.image_transforms)),
        )
        self.value_min, self.value_max = load_return_range(self.dataset.root)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        # Without delta_timestamps LeRobot returns a single unbatched frame:
        # state (joint_dim,), images (3,H,W) already float32 in [0,1], and a
        # scalar return_to_go. Anything else means the dataset is malformed.
        qpos = sample["observation.state"].float()

        images = []
        for camera in self.camera_names:
            image = sample[f"observation.images.{camera}"].float()
            if image.ndim != 3 or image.shape[0] != 3:
                raise ValueError(
                    f"Expected {camera} image with shape (3,H,W), got {tuple(image.shape)}"
                )
            images.append((image - IMAGE_MEAN) / IMAGE_STD)

        return {
            "qpos": (qpos - self.qpos_mean) / self.qpos_std,
            "images": torch.stack(images, dim=0),
            "return_to_go": sample["return_to_go"].float(),
        }


def config_from_env() -> ACTConfig:
    base_cfg = ACTConfig(
        dataset_repo_id=os.environ.get("RECAP_CRITIC_DATASET_REPO_ID") or RECORD_DATASET_REPO_ID
    )
    return replace(
        base_cfg,
        job_name=os.environ.get("RECAP_CRITIC_JOB_NAME", "recap_critic"),
        output_root=os.environ.get("RECAP_CRITIC_OUTPUT_ROOT", base_cfg.output_root),
        num_train_steps=int(os.environ.get("RECAP_CRITIC_STEPS", base_cfg.num_train_steps)),
        save_freq=int(os.environ.get("RECAP_CRITIC_SAVE_FREQ", base_cfg.save_freq)),
    )


def save_checkpoint(checkpoint_dir, step, model, optim, cfg, metrics, value_min, value_max):
    checkpoint = {
        "step": step,
        "config": asdict(cfg),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optim.state_dict(),
        "metrics": metrics,
        "num_bins": NUM_BINS,
        "value_min": value_min,
        "value_max": value_max,
    }
    step_path = checkpoint_dir / f"step_{step:06d}.pt"
    torch.save(checkpoint, step_path)
    torch.save(checkpoint, checkpoint_dir / "last.pt")
    print(f"Saved checkpoint: {step_path}", flush=True)
    return step_path


def main():
    cfg = config_from_env()
    device_name = os.environ.get("RECAP_DEVICE")
    if device_name is None:
        device_name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_name)
    torch.manual_seed(cfg.seed)

    dataset = CriticDataset(cfg)
    value_min, value_max = dataset.value_min, dataset.value_max
    loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(cfg.seed),
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    model = ACTCritic(cfg, pretrained_backbone=True, num_bins=NUM_BINS).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    run_dir = Path(cfg.output_root) / cfg.job_name / datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")
    # Keep at least one mid-run checkpoint so the final step is never the first save.
    save_freq = max(1, min(cfg.save_freq, cfg.num_train_steps // 2))

    print(f"Dataset: {cfg.dataset_repo_id}@{cfg.dataset_revision}")
    print(f"Return range: [{value_min:g}, {value_max:g}] -> (-1, 0) over {NUM_BINS} bins")
    print(f"Output: {run_dir}")

    model.train()
    step = 0
    metrics = None
    while step < cfg.num_train_steps:
        for batch in loader:
            step_start = time.perf_counter()
            optim.zero_grad(set_to_none=True)
            loss, distribution = compute_loss(model, batch, device, value_min, value_max)
            with torch.no_grad():
                value_mean = float(expected_return(distribution, value_min, value_max).mean())
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()

            metrics = {
                "step": step,
                "split": "train",
                "loss": loss.item(),
                "value_mean": value_mean,
                "grad_norm": float(grad_norm),
                "step_seconds": time.perf_counter() - step_start,
            }
            if step % cfg.log_freq == 0:
                with (run_dir / "metrics.jsonl").open("a") as file:
                    file.write(json.dumps(metrics, sort_keys=True) + "\n")
                print(
                    f"step={step} loss={loss.item():.4f} value_mean={value_mean:.2f}",
                    flush=True,
                )
            if (step + 1) % save_freq == 0 or (step + 1) >= cfg.num_train_steps:
                save_checkpoint(
                    checkpoint_dir, step + 1, model, optim, cfg, metrics, value_min, value_max
                )

            step += 1
            if step >= cfg.num_train_steps:
                break


if __name__ == "__main__":
    main()
