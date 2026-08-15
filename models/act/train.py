from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from configs.act.base import ACTConfig
from models.act.act import ACTPolicy
from models.act.helper import make_dataloaders


def seed_training(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_training_status(path: str | None, payload: dict) -> None:
    if not path:
        return
    status_path = Path(path)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(
        json.dumps(
            {"timestamp": datetime.now(timezone.utc).isoformat(), **payload},
            sort_keys=True,
        )
        + "\n"
    )


def compute_loss(model, batch, device, kl_weight):
    images = batch["images"].to(device)
    qpos = batch["qpos"].to(device)
    actions = batch["actions"].to(device)
    action_mask = batch["action_mask"].to(device)
    act_pred, mu, log_var = model(images, qpos, actions, action_mask)
    action_error = F.l1_loss(act_pred, actions, reduction="none")
    valid_values = (action_mask.sum() * actions.shape[-1]).clamp_min(1.0)
    action_loss = (action_error * action_mask.unsqueeze(-1)).sum() / valid_values
    kl_loss = (-0.5 * (1 + log_var - mu.pow(2) - log_var.exp())).sum(dim=1).mean()
    loss = action_loss + kl_weight * kl_loss
    return loss, action_loss, kl_loss


def _required_dataset_repo_id(variable: str) -> str:
    """Read the dataset from the environment, refusing to guess.

    This used to fall back to a module-level constant, which for a long time
    named an abandoned task. A run with the variable unset then trained
    successfully on the wrong dataset and said nothing.
    """
    dataset_repo_id = os.environ.get(variable)
    if not dataset_repo_id:
        raise ValueError(
            f"{variable} is required; set it to the dataset this run should use"
        )
    return dataset_repo_id


def config_from_env() -> ACTConfig:
    base_cfg = ACTConfig(dataset_repo_id=_required_dataset_repo_id("ACT_TRAIN_DATASET_REPO_ID"))
    train_split = os.environ.get("ACT_TRAIN_SPLIT")
    image_transforms = os.environ.get("ACT_TRAIN_IMAGE_TRANSFORMS")
    if image_transforms is not None and image_transforms.lower() not in {"true", "false"}:
        raise ValueError(
            "ACT_TRAIN_IMAGE_TRANSFORMS must be 'true' or 'false', "
            f"got {image_transforms!r}."
        )
    cfg = replace(
        base_cfg,
        job_name=os.environ.get("ACT_TRAIN_JOB_NAME", base_cfg.job_name),
        output_root=os.environ.get("ACT_TRAIN_OUTPUT_ROOT", base_cfg.output_root),
        num_train_steps=int(os.environ.get("ACT_TRAIN_STEPS", base_cfg.num_train_steps)),
        chunk_size=int(os.environ.get("ACT_TRAIN_CHUNK_SIZE") or base_cfg.chunk_size),
        save_freq=int(os.environ.get("ACT_TRAIN_SAVE_FREQ", base_cfg.save_freq)),
        train_split=float(train_split) if train_split else base_cfg.train_split,
        image_transforms=(
            image_transforms.lower() == "true"
            if image_transforms is not None
            else base_cfg.image_transforms
        ),
    )
    if not 0 < cfg.train_split <= 1:
        raise ValueError(f"ACT_TRAIN_SPLIT must be in (0, 1], got {cfg.train_split}.")
    return cfg


def main():
    cfg = config_from_env()
    device_name = os.environ.get("ACT_DEVICE")
    if device_name is None:
        device_name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_name)
    resume_checkpoint = os.environ.get("ACT_TRAIN_RESUME_CHECKPOINT")
    pretrained_checkpoint = os.environ.get("ACT_TRAIN_PRETRAINED_CHECKPOINT")
    if resume_checkpoint and pretrained_checkpoint:
        raise ValueError(
            "ACT_TRAIN_RESUME_CHECKPOINT and ACT_TRAIN_PRETRAINED_CHECKPOINT are mutually exclusive."
        )
    checkpoint = None
    checkpoint_path = resume_checkpoint or pretrained_checkpoint
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("config"), dict):
            raise ValueError(f"Checkpoint {checkpoint_path} is missing a config dictionary.")
        model_state = checkpoint.get("model_state_dict")
        if (
            not isinstance(model_state, dict)
            or "decoder.camera_emb" in model_state
            or "decoder.final_norm.weight" not in model_state
        ):
            raise ValueError(
                "Resume checkpoint uses the legacy custom ACT architecture. "
                "Start a fresh run with the current LeRobot-aligned architecture."
            )

    seed_training(cfg.seed)
    model = ACTPolicy(cfg, pretrained_backbone=True).to(device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    train_loader, val_loader = make_dataloaders(cfg)
    start_step = 0
    if resume_checkpoint:
        checkpoint_transforms = bool(checkpoint["config"].get("image_transforms", False))
        if checkpoint_transforms != cfg.image_transforms:
            raise ValueError(
                "Resume checkpoint image_transforms setting does not match this run: "
                f"checkpoint={checkpoint_transforms}, run={cfg.image_transforms}."
            )
        optim.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = int(checkpoint["step"])
        print(f"Resumed checkpoint {resume_checkpoint} at step {start_step}", flush=True)
    elif pretrained_checkpoint:
        print(
            f"Warm-started weights from {pretrained_checkpoint}; optimizer and step counter are fresh.",
            flush=True,
        )
    run_dir = Path(cfg.output_root) / cfg.job_name / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")

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
        mlflow.set_tags({"run_dir": str(run_dir), "dataset": f"{cfg.dataset_repo_id}@{cfg.dataset_revision}"})
        mlflow_module = mlflow
        print(f"MLflow tracking URI: {tracking_uri}")
        print(f"MLflow run ID: {mlflow.active_run().info.run_id}")

    print(f"Dataset: {cfg.dataset_repo_id}@{cfg.dataset_revision}")
    print(f"Output: {run_dir}")

    model.train()

    step = start_step
    status_path = os.environ.get("ACT_TRAIN_STATUS_PATH")
    latest_metrics = None
    latest_checkpoint = None
    write_training_status(
        status_path,
        {
            "status": "running",
            "step": step,
            "device": str(device),
            "run_dir": str(run_dir),
            "latest_checkpoint": latest_checkpoint,
        },
    )
    try:
        if mlflow_module is not None:
            mlflow_module.log_artifact(str(run_dir / "config.json"), artifact_path="config")
        while step < cfg.num_train_steps:
            for batch in train_loader:
                step_start = time.perf_counter()
                optim.zero_grad(set_to_none=True)
                loss, action_loss, kl_loss = compute_loss(model, batch, device, cfg.kl_weight)
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
                latest_metrics = train_metrics
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
                    write_training_status(
                        status_path,
                        {
                            "status": "running",
                            "step": step,
                            "device": str(device),
                            "run_dir": str(run_dir),
                            "latest_checkpoint": latest_checkpoint,
                            "metrics": train_metrics,
                        },
                    )
                if val_loader is not None and step % cfg.eval_freq == 0:
                    # Match stock LeRobot ACT: validation computes the VAE
                    # objective in train mode, but without gradients.
                    model.train()
                    val_loss = 0.0
                    val_action_loss = 0.0
                    val_kl_loss = 0.0
                    val_batches = 0
                    with torch.no_grad():
                        for val_batch in val_loader:
                            loss, action_loss, kl_loss = compute_loss(model, val_batch, device, cfg.kl_weight)
                            val_loss += loss.item()
                            val_action_loss += action_loss.item()
                            val_kl_loss += kl_loss.item()
                            val_batches += 1
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
                    model.train()

                if (step + 1) % cfg.save_freq == 0 or (step + 1) >= cfg.num_train_steps:
                    latest_checkpoint = str(
                        save_checkpoint(checkpoint_dir, step + 1, model, optim, cfg, train_metrics)
                    )

                step += 1
                if step >= cfg.num_train_steps:
                    break
        write_training_status(
            status_path,
            {
                "status": "complete",
                "step": step,
                "device": str(device),
                "run_dir": str(run_dir),
                "latest_checkpoint": latest_checkpoint,
                "metrics": latest_metrics,
            },
        )
    except Exception as exc:
        write_training_status(
            status_path,
            {
                "status": "failed",
                "step": step,
                "device": str(device),
                "run_dir": str(run_dir),
                "latest_checkpoint": latest_checkpoint,
                "metrics": latest_metrics,
                "error": repr(exc),
            },
        )
        raise
    finally:
        if mlflow_module is not None:
            if (run_dir / "metrics.jsonl").exists():
                mlflow_module.log_artifact(str(run_dir / "metrics.jsonl"), artifact_path="metrics")
            mlflow_module.end_run()


def save_checkpoint(checkpoint_dir, step, model, optim, cfg, metrics):
    checkpoint = {
        "step": step,
        "config": asdict(cfg),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optim.state_dict(),
        "metrics": metrics,
    }
    step_path = checkpoint_dir / f"step_{step:06d}.pt"
    torch.save(checkpoint, step_path)
    torch.save(checkpoint, checkpoint_dir / "last.pt")
    print(f"Saved checkpoint: {step_path}", flush=True)
    return step_path


if __name__ == "__main__":
    main()
