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
from lerobot.policies.act.configuration_act import ACTConfig as LeRobotACTConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from utils.config import RECORD_DATASET_REPO_ID


@dataclass(frozen=True)
class LeRobotTrainState:
    run_config: dict
    policy_config: dict


def compute_loss(model, preprocessor, batch, cfg):
    for camera in cfg.camera_names:
        key = f"observation.images.{camera}"
        if batch[key].dtype == torch.uint8:
            batch[key] = batch[key].float() / 255.0
    batch = preprocessor(batch)
    loss, loss_dict = model.forward(batch)
    action_loss = torch.as_tensor(loss_dict.get("l1_loss", loss.item()), device=loss.device)
    kl_loss = torch.as_tensor(loss_dict.get("kld_loss", 0.0), device=loss.device)
    return loss, action_loss, kl_loss


def make_lerobot_policy_config(cfg: ACTConfig, device_name: str) -> LeRobotACTConfig:
    input_features = {"observation.state": PolicyFeature(FeatureType.STATE, (cfg.joint_dim,))}
    for camera in cfg.camera_names:
        input_features[f"observation.images.{camera}"] = PolicyFeature(FeatureType.VISUAL, (3, 480, 640))

    return LeRobotACTConfig(
        input_features=input_features,
        output_features={"action": PolicyFeature(FeatureType.ACTION, (cfg.joint_dim,))},
        device=device_name,
        chunk_size=cfg.chunk_size,
        n_action_steps=1,
        temporal_ensemble_coeff=cfg.exp_weight,
        dim_model=cfg.d_model,
        n_heads=cfg.num_heads,
        dim_feedforward=cfg.mlp_dim,
        n_encoder_layers=cfg.num_encoder_layers,
        n_decoder_layers=cfg.num_decoder_layers,
        latent_dim=cfg.d_z,
        n_vae_encoder_layers=cfg.num_vae_encoder_layers,
        dropout=cfg.dropout,
        kl_weight=cfg.kl_weight,
        optimizer_lr=cfg.learning_rate,
        optimizer_lr_backbone=cfg.learning_rate,
        optimizer_weight_decay=cfg.weight_decay,
    )


def make_dataloaders(cfg: ACTConfig, policy_cfg: LeRobotACTConfig, device: torch.device):
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


def main():
    base_cfg = ACTConfig(dataset_repo_id=os.environ.get("ACT_TRAIN_DATASET_REPO_ID") or RECORD_DATASET_REPO_ID)
    cfg = replace(
        base_cfg,
        job_name=os.environ.get("ACT_TRAIN_JOB_NAME", f"{base_cfg.job_name}_lerobot_train"),
        num_train_steps=int(os.environ.get("ACT_TRAIN_STEPS", base_cfg.num_train_steps)),
        chunk_size=int(os.environ.get("ACT_TRAIN_CHUNK_SIZE") or base_cfg.chunk_size),
        save_freq=int(os.environ.get("ACT_TRAIN_SAVE_FREQ", base_cfg.save_freq)),
    )
    device_name = os.environ.get("ACT_DEVICE")
    if device_name is None:
        device_name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_name)

    policy_cfg = make_lerobot_policy_config(cfg, device_name)
    metadata, train_loader, val_loader = make_dataloaders(cfg, policy_cfg, device)
    model = make_policy(policy_cfg, ds_meta=metadata)
    preprocessor, _ = make_pre_post_processors(policy_cfg=policy_cfg, dataset_stats=metadata.stats)
    optim = torch.optim.AdamW(model.get_optim_params(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    resume_checkpoint = os.environ.get("ACT_TRAIN_RESUME_CHECKPOINT")
    start_step = 0
    if resume_checkpoint:
        checkpoint = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optim.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = int(checkpoint["step"])
        print(f"Resumed checkpoint {resume_checkpoint} at step {start_step}", flush=True)

    run_dir = Path(cfg.output_root) / cfg.job_name / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    train_state = LeRobotTrainState(
        run_config=asdict(cfg),
        policy_config=policy_cfg.to_dict() if hasattr(policy_cfg, "to_dict") else asdict(policy_cfg),
    )
    (run_dir / "config.json").write_text(json.dumps(asdict(train_state), indent=2) + "\n")

    mlflow_module = None
    if os.environ.get("ACT_DISABLE_MLFLOW") == "1":
        print("MLflow disabled by ACT_DISABLE_MLFLOW=1")
    else:
        import mlflow

        output_root = Path(cfg.output_root)
        mlflow_base = output_root.parent if output_root.name == "act_experiments" else output_root
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{(mlflow_base / 'mlflow.db').resolve()}")
        artifact_uri = os.environ.get("MLFLOW_ARTIFACT_URI", (mlflow_base / "mlflow_artifacts").resolve().as_uri())
        mlflow.set_tracking_uri(tracking_uri)
        if mlflow.get_experiment_by_name(cfg.job_name) is None:
            mlflow.create_experiment(cfg.job_name, artifact_location=artifact_uri)
        mlflow.set_experiment(cfg.job_name)
        mlflow.start_run(run_name=run_dir.name)
        mlflow.log_params({
            key: value if isinstance(value, str | int | float | bool) else json.dumps(value)
            for key, value in asdict(cfg).items()
        })
        mlflow.set_tags({
            "run_dir": str(run_dir),
            "dataset": f"{cfg.dataset_repo_id}@{cfg.dataset_revision}",
            "model_kind": "lerobot_act",
        })
        mlflow_module = mlflow
        print(f"MLflow tracking URI: {tracking_uri}")
        print(f"MLflow run ID: {mlflow.active_run().info.run_id}")

    print(f"Dataset: {cfg.dataset_repo_id}@{cfg.dataset_revision}")
    print("Model kind: lerobot_act")
    print(f"Steps: {cfg.num_train_steps}")
    print(f"Chunk size: {cfg.chunk_size}")
    print(f"Save freq: {cfg.save_freq}")
    print(f"Output: {run_dir}")

    model.train()
    step = start_step
    try:
        if mlflow_module is not None:
            mlflow_module.log_artifact(str(run_dir / "config.json"), artifact_path="config")
        while step < cfg.num_train_steps:
            for batch in train_loader:
                step_start = time.perf_counter()
                optim.zero_grad(set_to_none=True)
                loss, action_loss, kl_loss = compute_loss(model, preprocessor, batch, cfg)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optim.step()

                train_metrics = {
                    "step": step,
                    "split": "train",
                    "loss": loss.item(),
                    "action_l1": action_loss.item(),
                    "kl": kl_loss.item(),
                    "weighted_kl": cfg.kl_weight * kl_loss.item(),
                    "grad_norm": float(grad_norm),
                    "step_seconds": time.perf_counter() - step_start,
                }
                if step % cfg.log_freq == 0:
                    with (run_dir / "metrics.jsonl").open("a") as file:
                        file.write(json.dumps(train_metrics, sort_keys=True) + "\n")
                    if mlflow_module is not None:
                        mlflow_module.log_metrics(
                            {f"train/{key}": float(value) for key, value in train_metrics.items() if isinstance(value, int | float)},
                            step=step,
                        )
                    print(
                        f"step={step} loss={loss.item():.4f} "
                        f"action_l1={action_loss.item():.4f} kl={kl_loss.item():.4f}",
                        flush=True,
                    )
                if val_loader is not None and step % cfg.eval_freq == 0:
                    model.eval()
                    val_loss = 0.0
                    val_action_loss = 0.0
                    val_kl_loss = 0.0
                    val_batches = 0
                    try:
                        with torch.no_grad():
                            for val_batch in val_loader:
                                loss, action_loss, kl_loss = compute_loss(model, preprocessor, val_batch, cfg)
                                val_loss += loss.item()
                                val_action_loss += action_loss.item()
                                val_kl_loss += kl_loss.item()
                                val_batches += 1
                    finally:
                        model.train()
                    val_batches = max(val_batches, 1)
                    val_metrics = {
                        "step": step,
                        "split": "val",
                        "loss": val_loss / val_batches,
                        "action_l1": val_action_loss / val_batches,
                        "kl": val_kl_loss / val_batches,
                        "weighted_kl": cfg.kl_weight * val_kl_loss / val_batches,
                    }
                    with (run_dir / "metrics.jsonl").open("a") as file:
                        file.write(json.dumps(val_metrics, sort_keys=True) + "\n")
                    if mlflow_module is not None:
                        mlflow_module.log_metrics(
                            {f"val/{key}": float(value) for key, value in val_metrics.items() if isinstance(value, int | float)},
                            step=step,
                        )
                    print(f"step={step} val_loss={val_metrics['loss']:.4f}", flush=True)

                if (step + 1) % cfg.save_freq == 0 or (step + 1) >= cfg.num_train_steps:
                    save_checkpoint(checkpoint_dir, step + 1, model, optim, train_state, train_metrics)

                step += 1
                if step >= cfg.num_train_steps:
                    break
    finally:
        if mlflow_module is not None:
            if (run_dir / "metrics.jsonl").exists():
                mlflow_module.log_artifact(str(run_dir / "metrics.jsonl"), artifact_path="metrics")
            mlflow_module.end_run()


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
