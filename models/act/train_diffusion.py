from dataclasses import asdict, dataclass, replace
from datetime import datetime
import json
import os
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, random_split

from configs.act.base import ACTConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors


@dataclass(frozen=True)
class DiffusionTrainState:
    run_config: dict
    policy_config: dict


def make_diffusion_policy_config(cfg: ACTConfig, device_name: str) -> DiffusionConfig:
    input_features = {"observation.state": PolicyFeature(FeatureType.STATE, (cfg.joint_dim,))}
    for camera in cfg.camera_names:
        input_features[f"observation.images.{camera}"] = PolicyFeature(FeatureType.VISUAL, (3, 480, 640))

    return DiffusionConfig(
        input_features=input_features,
        output_features={"action": PolicyFeature(FeatureType.ACTION, (cfg.joint_dim,))},
        device=device_name,
        n_obs_steps=int(os.environ.get("DIFFUSION_N_OBS_STEPS", "2")),
        horizon=int(os.environ.get("DIFFUSION_HORIZON", "64")),
        n_action_steps=int(os.environ.get("DIFFUSION_N_ACTION_STEPS", "32")),
        resize_shape=(240, 320),
        crop_ratio=1.0,
        optimizer_lr=float(os.environ.get("DIFFUSION_LR", "1e-4")),
    )


def make_dataloaders(cfg: ACTConfig, policy_cfg: DiffusionConfig, device: torch.device):
    metadata = LeRobotDatasetMetadata(repo_id=cfg.dataset_repo_id, revision=cfg.dataset_revision)
    dataset = LeRobotDataset(
        repo_id=cfg.dataset_repo_id,
        revision=cfg.dataset_revision,
        delta_timestamps=resolve_delta_timestamps(policy_cfg, metadata),
    )
    train_size = int(len(dataset) * cfg.train_split)
    train_size = min(max(train_size, 1), len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(cfg.seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = None if val_size == 0 else DataLoader(
        dataset=val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )
    return metadata, train_loader, val_loader


def prepare_batch(batch, preprocessor, cfg: ACTConfig):
    for camera in cfg.camera_names:
        key = f"observation.images.{camera}"
        if batch[key].dtype == torch.uint8:
            batch[key] = batch[key].float() / 255.0
    return preprocessor(batch)


def main():
    base_cfg = ACTConfig(dataset_repo_id=_required_dataset_repo_id("ACT_TRAIN_DATASET_REPO_ID"))
    cfg = replace(
        base_cfg,
        job_name=os.environ.get("ACT_TRAIN_JOB_NAME", f"{base_cfg.job_name}_diffusion_train"),
        num_train_steps=int(os.environ.get("ACT_TRAIN_STEPS", base_cfg.num_train_steps)),
        chunk_size=int(os.environ.get("ACT_TRAIN_CHUNK_SIZE") or base_cfg.chunk_size),
        save_freq=int(os.environ.get("ACT_TRAIN_SAVE_FREQ", base_cfg.save_freq)),
    )
    device_name = os.environ.get("ACT_DEVICE")
    if device_name is None:
        device_name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_name)

    policy_cfg = make_diffusion_policy_config(cfg, device_name)
    metadata, train_loader, val_loader = make_dataloaders(cfg, policy_cfg, device)
    model = make_policy(policy_cfg, ds_meta=metadata)
    preprocessor, _ = make_pre_post_processors(policy_cfg=policy_cfg, dataset_stats=metadata.stats)
    optim = torch.optim.AdamW(
        model.get_optim_params(),
        lr=policy_cfg.optimizer_lr,
        betas=policy_cfg.optimizer_betas,
        eps=policy_cfg.optimizer_eps,
        weight_decay=policy_cfg.optimizer_weight_decay,
    )

    run_dir = Path(cfg.output_root) / cfg.job_name / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    train_state = DiffusionTrainState(
        run_config=asdict(cfg),
        policy_config=policy_cfg.to_dict() if hasattr(policy_cfg, "to_dict") else asdict(policy_cfg),
    )
    (run_dir / "config.json").write_text(json.dumps(asdict(train_state), indent=2) + "\n")

    print(f"Dataset: {cfg.dataset_repo_id}@{cfg.dataset_revision}")
    print("Model kind: diffusion")
    print(f"Steps: {cfg.num_train_steps}")
    print(f"Horizon: {policy_cfg.horizon}")
    print(f"Obs steps: {policy_cfg.n_obs_steps}")
    print(f"Action steps: {policy_cfg.n_action_steps}")
    print(f"Save freq: {cfg.save_freq}")
    print(f"Output: {run_dir}")

    model.train()
    step = 0
    while step < cfg.num_train_steps:
        for batch in train_loader:
            step_start = time.perf_counter()
            optim.zero_grad(set_to_none=True)
            batch = prepare_batch(batch, preprocessor, cfg)
            loss, _ = model.forward(batch)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()

            train_metrics = {
                "step": step,
                "split": "train",
                "loss": loss.item(),
                "grad_norm": float(grad_norm),
                "step_seconds": time.perf_counter() - step_start,
            }
            if step % cfg.log_freq == 0:
                with (run_dir / "metrics.jsonl").open("a") as file:
                    file.write(json.dumps(train_metrics, sort_keys=True) + "\n")
                print(f"step={step} loss={loss.item():.4f}", flush=True)

            if val_loader is not None and step % cfg.eval_freq == 0:
                val_loss = 0.0
                val_batches = 0
                model.eval()
                with torch.no_grad():
                    for val_batch in val_loader:
                        val_batch = prepare_batch(val_batch, preprocessor, cfg)
                        loss, _ = model.forward(val_batch)
                        val_loss += loss.item()
                        val_batches += 1
                model.train()
                val_metrics = {"step": step, "split": "val", "loss": val_loss / max(val_batches, 1)}
                with (run_dir / "metrics.jsonl").open("a") as file:
                    file.write(json.dumps(val_metrics, sort_keys=True) + "\n")
                print(f"step={step} val_loss={val_metrics['loss']:.4f}", flush=True)

            if (step + 1) % cfg.save_freq == 0 or (step + 1) >= cfg.num_train_steps:
                save_checkpoint(checkpoint_dir, step + 1, model, optim, train_state, train_metrics)

            step += 1
            if step >= cfg.num_train_steps:
                break


def save_checkpoint(checkpoint_dir, step, model, optim, train_state, metrics):
    checkpoint = {
        "step": step,
        "config": asdict(train_state),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optim.state_dict(),
        "metrics": metrics,
    }
    step_path = checkpoint_dir / f"step_{step:06d}.pt"
    torch.save(checkpoint, step_path)
    torch.save(checkpoint, checkpoint_dir / "last.pt")
    print(f"Saved checkpoint: {step_path}", flush=True)


if __name__ == "__main__":
    main()
