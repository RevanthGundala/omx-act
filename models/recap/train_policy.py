"""Train the advantage-conditioned RECAP policy (arXiv:2511.14759).

Covers both stages of the paper's improvement loop, which differ only in where
the conditioning token comes from:

* SFT. Finetune on demonstrations with the indicator pinned to POSITIVE. The
  paper fixes "I_t to True in this stage, which we found to lead to slightly
  better results". Leave RECAP_POLICY_LABELS unset for this.
* Improvement loop. Train on all collected data with per-transition labels from
  models/recap/label_advantage.py, plus conditioning dropout so the policy also
  learns the unconditioned branch used for classifier-free guidance.

Labels are keyed by the dataset's own global frame index rather than by position
in the dataloader, so shuffling and the train/holdout episode split cannot
misalign a label with its transition.

Example:
    RECAP_POLICY_PRETRAINED_CHECKPOINT=outputs/act_experiments/<run>/checkpoints/last.pt \
      uv run python -m models.recap.train_policy
"""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime
import json
import os
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from configs.act.base import ACTConfig
from models.recap.act_recap import ACTPolicy, AdvantageCondition
from utils.config import RECORD_DATASET_REPO_ID

# Appendix F: "we randomly drop out the conditioning on the advantage indicator
# 30% of the time", so both the conditional and unconditional policy are learned.
DEFAULT_DROPOUT = 0.30
# The advantage token is the only thing a base ACT checkpoint cannot supply.
ADVANTAGE_PARAMETERS = ("decoder.adv_proj.weight", "decoder.adv_proj.bias", "decoder.adv_pos_emb.weight")


def load_conditions(labels_path: str | Path) -> torch.Tensor:
    """Read advantage_labels.json into a tensor keyed by global frame index."""
    payload = json.loads(Path(labels_path).read_text())
    conditions = torch.as_tensor(payload["conditions"], dtype=torch.long)
    if conditions.ndim != 1:
        raise ValueError(f"Expected a flat condition list, got shape {tuple(conditions.shape)}")
    invalid = (conditions < 0) | (conditions >= len(AdvantageCondition))
    if invalid.any():
        raise ValueError("advantage_labels.json contains out-of-range condition IDs")
    return conditions


class AdvantageDataset(Dataset):
    """Attach a conditioning token to each ACT training sample.

    ``dataset`` is a models.act.helper.ACTDataset. ``conditions`` is indexed by
    global frame index; passing None pins every sample to POSITIVE for the SFT
    stage.
    """

    def __init__(self, dataset, conditions: torch.Tensor | None = None, frame_ids: torch.Tensor | None = None):
        self.dataset = dataset
        self.conditions = conditions
        if frame_ids is None:
            # The 'index' column is the absolute frame index, so it stays correct
            # when the underlying dataset is filtered to a subset of episodes.
            frame_ids = torch.as_tensor(dataset.dataset.hf_dataset["index"])
        self.frame_ids = frame_ids.long()
        if len(self.frame_ids) != len(dataset):
            raise ValueError(
                f"Got {len(self.frame_ids)} frame ids for {len(dataset)} samples"
            )
        if conditions is not None:
            out_of_range = self.frame_ids >= len(conditions)
            if out_of_range.any():
                raise ValueError(
                    f"Labels cover {len(conditions)} frames but the dataset references index "
                    f"{int(self.frame_ids.max())}; relabel with models/recap/label_advantage.py"
                )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if self.conditions is None:
            condition = torch.tensor(int(AdvantageCondition.POSITIVE), dtype=torch.long)
        else:
            condition = self.conditions[self.frame_ids[idx]]
        item["advantage_condition"] = condition
        return item


def apply_conditioning_dropout(conditions: torch.Tensor, dropout: float, generator=None) -> torch.Tensor:
    """Replace a random share of tokens with UNCONDITIONED (Appendix F)."""
    if not 0.0 <= dropout <= 1.0:
        raise ValueError(f"dropout must be in [0, 1], got {dropout}")
    if dropout == 0.0:
        return conditions
    dropped = torch.rand(conditions.shape, generator=generator) < dropout
    return torch.where(
        dropped.to(conditions.device),
        torch.full_like(conditions, int(AdvantageCondition.UNCONDITIONED)),
        conditions,
    )


def compute_loss(model, batch, device, kl_weight):
    """L1 + KL, matching models/act/train.py, with the advantage token added."""
    images = batch["images"].to(device)
    qpos = batch["qpos"].to(device)
    actions = batch["actions"].to(device)
    action_mask = batch["action_mask"].to(device)
    advantage = batch["advantage_condition"].to(device)
    act_pred, mu, log_var = model(images, qpos, actions, action_mask, advantage=advantage)
    action_error = F.l1_loss(act_pred, actions, reduction="none")
    valid_values = (action_mask.sum() * actions.shape[-1]).clamp_min(1.0)
    action_loss = (action_error * action_mask.unsqueeze(-1)).sum() / valid_values
    kl_loss = (-0.5 * (1 + log_var - mu.pow(2) - log_var.exp())).sum(dim=1).mean()
    return action_loss + kl_weight * kl_loss, action_loss, kl_loss


def warm_start(model: ACTPolicy, checkpoint_path: str | Path, device: torch.device) -> ACTPolicy:
    """Graft a behavior-cloned ACT checkpoint onto the advantage-conditioned policy."""
    checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint {checkpoint_path} has no model_state_dict")
    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(
            f"Checkpoint {checkpoint_path} has unexpected parameters: {incompatible.unexpected_keys}"
        )
    unexpected_missing = set(incompatible.missing_keys) - set(ADVANTAGE_PARAMETERS)
    if unexpected_missing:
        raise ValueError(
            f"Checkpoint {checkpoint_path} is missing non-advantage parameters: {sorted(unexpected_missing)}"
        )
    return model


def config_from_env() -> ACTConfig:
    base_cfg = ACTConfig(
        dataset_repo_id=os.environ.get("RECAP_POLICY_DATASET_REPO_ID") or RECORD_DATASET_REPO_ID
    )
    return replace(
        base_cfg,
        job_name=os.environ.get("RECAP_POLICY_JOB_NAME", "recap_policy"),
        output_root=os.environ.get("RECAP_POLICY_OUTPUT_ROOT", base_cfg.output_root),
        num_train_steps=int(os.environ.get("RECAP_POLICY_STEPS", base_cfg.num_train_steps)),
        save_freq=int(os.environ.get("RECAP_POLICY_SAVE_FREQ", base_cfg.save_freq)),
    )


def save_checkpoint(checkpoint_dir, step, model, optim, cfg, metrics):
    checkpoint = {
        "step": step,
        "config": asdict(cfg),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optim.state_dict(),
        "metrics": metrics,
        "recap": True,
    }
    step_path = checkpoint_dir / f"step_{step:06d}.pt"
    torch.save(checkpoint, step_path)
    torch.save(checkpoint, checkpoint_dir / "last.pt")
    print(f"Saved checkpoint: {step_path}", flush=True)
    return step_path


def main():
    # Imported lazily: models.act.helper pulls in data.episode_split, which is
    # not present in every checkout.
    from models.act.helper import make_dataloaders

    cfg = config_from_env()
    device_name = os.environ.get("RECAP_DEVICE")
    if device_name is None:
        device_name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_name)
    torch.manual_seed(cfg.seed)

    labels_path = os.environ.get("RECAP_POLICY_LABELS")
    conditions = load_conditions(labels_path) if labels_path else None
    # SFT pins the indicator to True, so dropout only applies once real labels do.
    default_dropout = DEFAULT_DROPOUT if conditions is not None else 0.0
    dropout = float(os.environ.get("RECAP_POLICY_DROPOUT", default_dropout))

    train_loader, _ = make_dataloaders(cfg)
    train_loader = torch.utils.data.DataLoader(
        dataset=AdvantageDataset(train_loader.dataset, conditions),
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(cfg.seed),
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = ACTPolicy(cfg, pretrained_backbone=True).to(device)
    pretrained = os.environ.get("RECAP_POLICY_PRETRAINED_CHECKPOINT")
    if pretrained:
        warm_start(model, pretrained, device)
        print(f"Warm-started from {pretrained}; advantage token is freshly initialized")
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    dropout_generator = torch.Generator().manual_seed(cfg.seed)

    run_dir = Path(cfg.output_root) / cfg.job_name / datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")
    save_freq = max(1, min(cfg.save_freq, cfg.num_train_steps // 2))

    stage = "improvement loop" if conditions is not None else "SFT (indicator fixed to POSITIVE)"
    print(f"Dataset: {cfg.dataset_repo_id}@{cfg.dataset_revision}")
    print(f"Stage: {stage}; conditioning dropout={dropout:g}")
    print(f"Output: {run_dir}")

    model.train()
    step = 0
    metrics = None
    while step < cfg.num_train_steps:
        for batch in train_loader:
            step_start = time.perf_counter()
            batch["advantage_condition"] = apply_conditioning_dropout(
                batch["advantage_condition"], dropout, dropout_generator
            )
            optim.zero_grad(set_to_none=True)
            loss, action_loss, kl_loss = compute_loss(model, batch, device, cfg.kl_weight)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()

            metrics = {
                "step": step,
                "split": "train",
                "loss": loss.item(),
                "action_loss": action_loss.item(),
                "kl_loss": kl_loss.item(),
                "grad_norm": float(grad_norm),
                "step_seconds": time.perf_counter() - step_start,
            }
            if step % cfg.log_freq == 0:
                with (run_dir / "metrics.jsonl").open("a") as file:
                    file.write(json.dumps(metrics, sort_keys=True) + "\n")
                print(
                    f"step={step} loss={loss.item():.4f} action={action_loss.item():.4f}",
                    flush=True,
                )
            if (step + 1) % save_freq == 0 or (step + 1) >= cfg.num_train_steps:
                save_checkpoint(checkpoint_dir, step + 1, model, optim, cfg, metrics)

            step += 1
            if step >= cfg.num_train_steps:
                break


if __name__ == "__main__":
    main()
