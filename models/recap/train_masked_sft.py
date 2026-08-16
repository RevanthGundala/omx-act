"""Train all-positive RECAP SFT with true per-example token omission.

This is intentionally separate from ``train_policy.py``. The legacy trainer
uses a learned UNCONDITIONED token for conditioning dropout; this experiment
matches PI's stated omission behavior by replacing the condition id with
``MASKED_ADVANTAGE`` on a random share of examples, which
``act_recap_masked.ACTPolicy`` omits from encoder and decoder attention.

All examples still target the same expert actions. Masked dropout therefore
learns conditional and unconditional branches; it does not teach POSITIVE
versus NEGATIVE semantics.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import time

import torch

from models.recap.act_recap_masked import ACTPolicy, MASKED_ADVANTAGE
from models.recap.train_policy import (
    AdvantageDataset,
    compute_loss,
    config_from_env,
    parse_episode_subset,
    warm_start,
)


def apply_masked_conditioning_dropout(
    conditions: torch.Tensor,
    dropout: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace a random share of condition ids with ``MASKED_ADVANTAGE``.

    Returns ``(conditions, dropped)``. The sentinel is what
    ``act_recap_masked.ACTPolicy`` omits from encoder and decoder attention, so
    a dropped example trains the unconditional branch rather than a third
    learned token value.
    """
    if conditions.ndim != 1:
        raise ValueError(
            f"Expected one condition ID per sample, got {tuple(conditions.shape)}"
        )
    if not 0.0 <= dropout <= 1.0:
        raise ValueError(f"dropout must be in [0, 1], got {dropout}")

    conditions = conditions.long().clone()
    dropped = torch.rand(conditions.shape, generator=generator) < dropout
    conditions[dropped] = MASKED_ADVANTAGE
    return conditions, dropped


def save_checkpoint(
    checkpoint_dir: Path,
    step: int,
    model: ACTPolicy,
    optimizer: torch.optim.Optimizer,
    cfg,
    metrics: dict,
    dropout: float,
) -> Path:
    checkpoint = {
        "step": step,
        "config": asdict(cfg),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "recap": True,
        "masked_advantage": True,
        "conditioning_dropout": dropout,
    }
    path = checkpoint_dir / f"step_{step:06d}.pt"
    torch.save(checkpoint, path)
    torch.save(checkpoint, checkpoint_dir / "last.pt")
    print(f"Saved checkpoint: {path}", flush=True)
    return path


def main() -> None:
    from models.act.helper import make_dataloaders

    if os.environ.get("RECAP_POLICY_LABELS"):
        raise ValueError(
            "train_masked_sft is all-positive SFT; unset RECAP_POLICY_LABELS"
        )
    if os.environ.get("RECAP_POLICY_SYNTHETIC_SHIFT"):
        raise ValueError(
            "train_masked_sft does not support synthetic target shifts"
        )

    cfg = config_from_env()
    device_name = os.environ.get("RECAP_DEVICE")
    if device_name is None:
        device_name = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    device = torch.device(device_name)
    torch.manual_seed(cfg.seed)

    dropout = float(
        os.environ.get(
            "RECAP_POLICY_DROPOUT",
            cfg.recap_conditioning_dropout,
        )
    )
    if not 0.0 <= dropout <= 1.0:
        raise ValueError(f"dropout must be in [0, 1], got {dropout}")

    episodes = parse_episode_subset(os.environ.get("RECAP_POLICY_EPISODES"))
    base_loader, _ = make_dataloaders(cfg, episodes=episodes)
    dataset = AdvantageDataset(base_loader.dataset, conditions=None)
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(cfg.seed),
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = ACTPolicy(cfg, pretrained_backbone=True).to(device)
    pretrained = os.environ.get("RECAP_POLICY_PRETRAINED_CHECKPOINT")
    if not pretrained:
        raise ValueError("RECAP_POLICY_PRETRAINED_CHECKPOINT is required")
    warm_start(model, pretrained, device)
    print(
        f"Warm-started from {pretrained}; advantage token is freshly initialized",
        flush=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    dropout_generator = torch.Generator().manual_seed(cfg.seed)

    run_dir = (
        Path(cfg.output_root)
        / cfg.job_name
        / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")
    (run_dir / "masking.json").write_text(
        json.dumps(
            {
                "mode": "attention_mask",
                "dropout": dropout,
                "condition": AdvantageCondition.POSITIVE.name,
            },
            indent=2,
        )
        + "\n"
    )
    if episodes is not None:
        (run_dir / "episodes.json").write_text(json.dumps(episodes) + "\n")
    save_freq = max(1, min(cfg.save_freq, cfg.num_train_steps // 2))

    print(f"Dataset: {cfg.dataset_repo_id}@{cfg.dataset_revision}")
    print(
        "Stage: masked SFT (indicator fixed POSITIVE when present); "
        f"mask dropout={dropout:g}"
    )
    print(
        f"Inherited architecture: chunk_size={cfg.chunk_size} "
        f"cameras={cfg.camera_names} train_split={cfg.train_split} "
        f"lr={cfg.learning_rate:g} steps={cfg.num_train_steps}"
    )
    print(f"Output: {run_dir}")

    model.train()
    step = 0
    while step < cfg.num_train_steps:
        for batch in train_loader:
            step_start = time.perf_counter()
            conditions, dropped = apply_masked_conditioning_dropout(
                batch["advantage_condition"],
                dropout,
                dropout_generator,
            )
            batch["advantage_condition"] = conditions

            optimizer.zero_grad(set_to_none=True)
            loss, action_loss, kl_loss = compute_loss(
                model,
                batch,
                device,
                cfg.kl_weight,
            )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                cfg.grad_clip,
            )
            optimizer.step()

            metrics = {
                "step": step,
                "split": "train",
                "loss": loss.item(),
                "action_loss": action_loss.item(),
                "kl_loss": kl_loss.item(),
                "grad_norm": float(grad_norm),
                "masked_fraction": float(dropped.float().mean()),
                "step_seconds": time.perf_counter() - step_start,
            }
            if step % cfg.log_freq == 0:
                with (run_dir / "metrics.jsonl").open("a") as file:
                    file.write(json.dumps(metrics, sort_keys=True) + "\n")
                print(
                    f"step={step} loss={loss.item():.4f} "
                    f"action={action_loss.item():.4f} "
                    f"masked={metrics['masked_fraction']:.1%}",
                    flush=True,
                )
            if (step + 1) % save_freq == 0 or (
                step + 1
            ) >= cfg.num_train_steps:
                save_checkpoint(
                    checkpoint_dir,
                    step + 1,
                    model,
                    optimizer,
                    cfg,
                    metrics,
                    dropout,
                )

            step += 1
            if step >= cfg.num_train_steps:
                break


if __name__ == "__main__":
    main()
