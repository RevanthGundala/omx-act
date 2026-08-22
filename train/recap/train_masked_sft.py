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

from dataclasses import asdict, replace
from datetime import datetime
import json
import os
from pathlib import Path
import time

import torch

from models.recap.act_recap_masked import (
    ACTPolicy,
    MASKED_ADVANTAGE,
    AdvantageCondition,
)

# What this architecture adds on top of a behaviour-cloned ACT checkpoint.
ADVANTAGE_PARAMETERS = (
    "decoder.advantage_embedding.weight",
    "decoder.adv_pos_emb.weight",
)
from train.recap.train_policy import (
    AdvantageDataset,
    compute_loss,
    config_from_env,
    parse_episode_subset,
    warm_start,
)


def advantage_attention(model, dropped: torch.Tensor) -> dict[str, float]:
    """Softmax weight placed on the advantage token during the last forward.

    Reads the weights the attention modules already cached, so this measures
    the training step that just ran rather than costing an extra forward. The
    cache is populated before attention dropout, so these are true softmax
    probabilities.

    Reported separately for examples whose token was kept and examples whose
    token was dropped, which is what distinguishes "the model attends to the
    token" from "the mask actually removes it".
    """
    decoder = model.decoder
    batch_size = dropped.shape[0]

    def slot_zero(blocks, attribute):
        weights = [
            getattr(block, attribute).last_attn_weights
            for block in blocks
            if getattr(block, attribute) is not None
            and getattr(block, attribute).last_attn_weights is not None
        ]
        if not weights:
            return None
        return torch.cat([w[..., 0].reshape(batch_size, -1) for w in weights], dim=1)

    metrics: dict[str, float] = {}
    for name, weights in (
        ("encoder", slot_zero(decoder.transformer_encoder, "self_attn")),
        ("decoder", slot_zero(decoder.transformer_decoder, "cross_attn")),
    ):
        if weights is None:
            continue
        kept = ~dropped
        if kept.any():
            metrics[f"{name}_attention_visible"] = float(weights[kept].mean())
            metrics[f"{name}_attention_visible_max"] = float(weights[kept].max())
        if dropped.any():
            metrics[f"{name}_attention_omitted_max"] = float(weights[dropped].max())
    return metrics


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
    pretrained = os.environ.get("RECAP_POLICY_PRETRAINED_CHECKPOINT")

    # Without a checkpoint there is no architecture to inherit, and the
    # ACTConfig defaults (chunk_size=100, train_split=0.95) are not what any
    # other run on this task used. Refuse to guess rather than silently train
    # something incomparable.
    chunk_size = os.environ.get("RECAP_POLICY_CHUNK_SIZE")
    train_split = os.environ.get("RECAP_POLICY_TRAIN_SPLIT")
    if not pretrained and not (chunk_size and train_split):
        raise ValueError(
            "Training from scratch requires RECAP_POLICY_CHUNK_SIZE and "
            "RECAP_POLICY_TRAIN_SPLIT: with no checkpoint to inherit from, the "
            "ACTConfig defaults would quietly differ from every other run"
        )
    if chunk_size:
        cfg = replace(cfg, chunk_size=int(chunk_size))
    if train_split:
        cfg = replace(cfg, train_split=float(train_split))

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
    if pretrained:
        warm_start(model, pretrained, device, allowed_missing=ADVANTAGE_PARAMETERS)
        print(
            f"Warm-started from {pretrained}; advantage token is freshly initialized",
            flush=True,
        )
    else:
        # The token is then present from the first step, rather than bolted onto
        # a network that has already converged without it.
        print("Training from scratch; no warm start", flush=True)

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
                **advantage_attention(model, dropped.to(device)),
            }
            if step % cfg.log_freq == 0:
                with (run_dir / "metrics.jsonl").open("a") as file:
                    file.write(json.dumps(metrics, sort_keys=True) + "\n")
                print(
                    f"step={step} loss={loss.item():.4f} "
                    f"action={action_loss.item():.4f} "
                    f"masked={metrics['masked_fraction']:.1%} "
                    f"attn_vis={metrics.get('decoder_attention_visible', float('nan')):.2e} "
                    f"attn_omit={metrics.get('decoder_attention_omitted_max', 0.0):.2e}",
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
