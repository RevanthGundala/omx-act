"""Ablate the real advantage token and measure POLICY LOSS (L1).

``check_token_sensitivity.py`` reports action displacement as a fraction of
observation-driven spread. This check asks a different question: do the real
labels change the training objective? If flipping or zeroing the token barely
moves L1 on held-out episodes, the labels are not informative for this policy.

Compared on IDENTICAL held-out batches:

* ``true`` — critic label
* ``flipped`` — POS <-> NEG (UNCONDITIONED unchanged)
* ``all_pos`` / ``all_neg`` — constant tokens
* ``unconditioned`` — the legacy 3-way embedding's UNCONDITIONED class

Must use a held-out episode split. Every prior token number was measured on
training data (``train_split=1.0``). Default here is ``0.8``.

At eval the CVAE latent is the zero vector (``model.eval``), matching
inference. Reported loss is ``action_loss`` from ``compute_loss`` (masked L1).

Example (full 034 image eval belongs on Modal):

    PYTHONPATH=. .venv/bin/python -m utils.analysis.token_loss \\
      --checkpoint /path/to/step_020000.pt \\
      --dataset-repo-id RevanthGundala/034-recap-balanced \\
      --labels /path/to/034-recap-balanced.json \\
      --train-split 0.8

CPU plumbing smoke (no checkpoint, no images):

    PYTHONPATH=. .venv/bin/python -m utils.analysis.token_loss --smoke
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import torch

from configs.act.base import ACTConfig
from models.act.episode_split import resolve_train_holdout_episodes
from models.recap.policy import ACTPolicy, AdvantageCondition
from train.recap.policy import (
    AdvantageDataset,
    compute_loss,
    config_from_checkpoint,
    load_conditions,
    parse_episode_subset,
)

CONDITION_ORDER = ("true", "flipped", "all_pos", "all_neg", "unconditioned")


def flip_conditions(conditions: torch.Tensor) -> torch.Tensor:
    """Swap POS and NEG. UNCONDITIONED (and any other id) is left alone."""
    flipped = conditions.clone()
    positive = conditions == int(AdvantageCondition.POSITIVE)
    negative = conditions == int(AdvantageCondition.NEGATIVE)
    flipped[positive] = int(AdvantageCondition.NEGATIVE)
    flipped[negative] = int(AdvantageCondition.POSITIVE)
    return flipped


def tokens_for(name: str, true_tokens: torch.Tensor) -> torch.Tensor:
    if name == "true":
        return true_tokens
    if name == "flipped":
        return flip_conditions(true_tokens)
    if name == "all_pos":
        return torch.full_like(true_tokens, int(AdvantageCondition.POSITIVE))
    if name == "all_neg":
        return torch.full_like(true_tokens, int(AdvantageCondition.NEGATIVE))
    if name == "unconditioned":
        return torch.full_like(true_tokens, int(AdvantageCondition.UNCONDITIONED))
    raise ValueError(f"unknown condition {name!r}")


def _valid_action_count(batch: dict) -> float:
    mask = batch["action_mask"]
    joints = batch["actions"].shape[-1]
    return float((mask.sum() * joints).clamp_min(1).item())


@torch.no_grad()
def accumulate_action_l1(
    model: ACTPolicy,
    batches,
    device: torch.device,
    *,
    max_frames: int | None = None,
) -> dict:
    """Mean masked L1 per condition, weighted by valid action values.

    Batches are consumed once. Every condition is scored on that same batch
    before the next one is loaded, so the comparison is paired.
    """
    model.eval()
    weighted = {name: 0.0 for name in CONDITION_ORDER}
    weight = 0.0
    n_frames = 0
    n_batches = 0

    for batch in batches:
        if max_frames is not None and n_frames >= max_frames:
            break
        true_tokens = batch["advantage_condition"].to(device)
        valid = _valid_action_count(batch)
        for name in CONDITION_ORDER:
            override = dict(batch)
            override["advantage_condition"] = tokens_for(name, true_tokens)
            _, action_loss, _ = compute_loss(model, override, device, kl_weight=0.0)
            weighted[name] += float(action_loss.item()) * valid
        weight += valid
        n_frames += int(batch["qpos"].shape[0])
        n_batches += 1

    if n_batches == 0 or weight <= 0:
        raise ValueError("no held-out batches to score")

    means = {name: weighted[name] / weight for name in CONDITION_ORDER}
    return {
        "means": means,
        "n_frames": n_frames,
        "n_batches": n_batches,
        "valid_action_values": weight,
    }


def report_from_means(
    means: dict[str, float],
    *,
    n_frames: int,
    n_batches: int,
    meta: dict | None = None,
) -> dict:
    true_l1 = means["true"]
    conditions = {}
    max_abs_pct = 0.0
    for name in CONDITION_ORDER:
        loss = means[name]
        delta = loss - true_l1
        pct = 0.0 if true_l1 == 0 else 100.0 * delta / true_l1
        conditions[name] = {
            "action_loss": loss,
            "delta": delta,
            "delta_pct": pct,
        }
        max_abs_pct = max(max_abs_pct, abs(pct))
    payload = {
        "n_frames": n_frames,
        "n_batches": n_batches,
        "conditions": conditions,
        "max_abs_delta_pct": max_abs_pct,
        "labels_informative": max_abs_pct >= 1.0,
        "loss": "action_loss (masked L1); CVAE z=0 because model.eval()",
    }
    if meta:
        payload.update(meta)
    return payload


def format_report(report: dict) -> str:
    lines = [
        "CHECK 4: advantage-token L1 ablation (held-out episodes)",
        f"checkpoint:   {report.get('checkpoint', '(none)')}",
        f"dataset:      {report.get('dataset_repo_id', '(none)')}",
        f"train_split:  {report.get('train_split', 'n/a')}  "
        f"holdout_episodes: {report.get('n_holdout_episodes', 'n/a')}  "
        f"frames: {report['n_frames']}  batches: {report['n_batches']}",
        f"device:       {report.get('device', 'n/a')}  "
        f"{report.get('loss', 'action_loss')}",
    ]
    if report.get("smoke"):
        lines.append("NOTE: CPU plumbing smoke with a tiny fake batch — not a policy result.")
    if report.get("subset"):
        lines.append(
            f"NOTE: {report['n_frames']}-frame subset smoke, not the full holdout. "
            "Full 034 image eval belongs on Modal."
        )
    lines.append("")
    lines.append(f"{'condition':<16}{'L1':>12}{'Δ vs true':>14}{'Δ%':>10}")
    for name in CONDITION_ORDER:
        row = report["conditions"][name]
        lines.append(
            f"{name:<16}{row['action_loss']:12.6f}{row['delta']:+14.6f}{row['delta_pct']:+10.2f}%"
        )
    lines.append("")
    if report["labels_informative"]:
        lines.append(
            f"Largest |Δ%| = {report['max_abs_delta_pct']:.2f}%. "
            "Labels move L1; they are not inert for this policy."
        )
    else:
        lines.append(
            f"Largest |Δ%| = {report['max_abs_delta_pct']:.2f}%. "
            "Loss barely moves — labels are not informative for this policy."
        )
    return "\n".join(lines)


def smoke_config() -> ACTConfig:
    return ACTConfig(
        dataset_repo_id="smoke/token-loss",
        camera_names=("top",),
        chunk_size=4,
        d_model=32,
        joint_dim=6,
        d_z=8,
        num_vae_encoder_layers=1,
        num_encoder_layers=1,
        num_decoder_layers=1,
        num_heads=4,
        mlp_dim=64,
        dropout=0.0,
        batch_size=2,
        train_split=0.8,
    )


def smoke_batch(batch_size: int = 2, chunk_size: int = 4, joint_dim: int = 6) -> dict:
    return {
        "images": torch.randn(batch_size, 1, 3, 32, 32),
        "qpos": torch.randn(batch_size, joint_dim),
        "actions": torch.randn(batch_size, chunk_size, joint_dim),
        "action_mask": torch.ones(batch_size, chunk_size),
        "advantage_condition": torch.tensor(
            [int(AdvantageCondition.NEGATIVE), int(AdvantageCondition.POSITIVE)][:batch_size],
            dtype=torch.long,
        ),
    }


def run_smoke(device: torch.device | None = None) -> dict:
    device = device or torch.device("cpu")
    torch.manual_seed(0)
    cfg = smoke_config()
    model = ACTPolicy(cfg, pretrained_backbone=False).to(device).eval()
    scored = accumulate_action_l1(model, [smoke_batch()], device)
    return report_from_means(
        scored["means"],
        n_frames=scored["n_frames"],
        n_batches=scored["n_batches"],
        meta={
            "smoke": True,
            "device": str(device),
            "checkpoint": "(random tiny ACTPolicy)",
            "dataset_repo_id": cfg.dataset_repo_id,
            "train_split": cfg.train_split,
            "n_holdout_episodes": 0,
        },
    )


def require_held_out_split(train_split: float) -> None:
    if train_split >= 1.0:
        raise ValueError(
            "This check must use a held-out episode split. train_split=1.0 "
            "would repeat the training-data token numbers."
        )
    if not 0.0 < train_split < 1.0:
        raise ValueError(f"train_split must be in (0, 1), got {train_split}")


def load_policy(checkpoint_path: Path, device: torch.device) -> tuple[ACTPolicy, ACTConfig]:
    cfg = config_from_checkpoint(checkpoint_path)
    model = ACTPolicy(cfg, pretrained_backbone=False).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, cfg


def held_out_loader(
    cfg: ACTConfig,
    labels_path: Path,
    *,
    episodes: list[int] | None = None,
    num_workers: int = 0,
):
    from lerobot.datasets import LeRobotDatasetMetadata
    from models.act.helper import make_dataloaders

    require_held_out_split(cfg.train_split)
    conditions = load_conditions(labels_path)
    metadata = LeRobotDatasetMetadata(
        repo_id=cfg.dataset_repo_id, revision=cfg.dataset_revision
    )
    train_episodes, holdout_episodes = resolve_train_holdout_episodes(
        metadata.total_episodes, cfg.train_split, cfg.seed
    )
    if episodes is not None:
        allowed = set(episodes)
        holdout_episodes = [episode for episode in holdout_episodes if episode in allowed]
    if not holdout_episodes:
        raise ValueError("held-out split is empty; lower train_split or widen --episodes")

    _, val_loader = make_dataloaders(cfg, episodes=episodes)
    if val_loader is None:
        raise ValueError("make_dataloaders returned no holdout loader")
    dataset = AdvantageDataset(
        val_loader.dataset,
        conditions,
        total_frames=metadata.total_frames,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return loader, holdout_episodes, train_episodes


def run_ablation(
    checkpoint: Path,
    *,
    dataset_repo_id: str,
    labels: Path,
    train_split: float = 0.8,
    device: torch.device | None = None,
    max_frames: int | None = None,
    batch_size: int | None = None,
    episodes: str | None = None,
    num_workers: int | None = None,
) -> dict:
    require_held_out_split(train_split)
    if device is None:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    model, cfg = load_policy(checkpoint, device)
    cfg = replace(cfg, dataset_repo_id=dataset_repo_id, train_split=train_split)
    if batch_size is not None:
        cfg = replace(cfg, batch_size=batch_size)
    workers = 0 if device.type != "cuda" else cfg.num_workers
    if num_workers is not None:
        workers = num_workers
    loader, holdout_episodes, _train_episodes = held_out_loader(
        cfg,
        labels,
        episodes=parse_episode_subset(episodes),
        num_workers=workers,
    )
    scored = accumulate_action_l1(model, loader, device, max_frames=max_frames)
    return report_from_means(
        scored["means"],
        n_frames=scored["n_frames"],
        n_batches=scored["n_batches"],
        meta={
            "smoke": False,
            "subset": max_frames is not None,
            "device": str(device),
            "checkpoint": str(checkpoint),
            "dataset_repo_id": cfg.dataset_repo_id,
            "labels": str(labels),
            "train_split": cfg.train_split,
            "holdout_episodes": holdout_episodes,
            "n_holdout_episodes": len(holdout_episodes),
            "max_frames": max_frames,
        },
    )


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dataset-repo-id", default=None)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Episode holdout fraction. Default 0.8; 1.0 is refused.",
    )
    parser.add_argument("--episodes", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a tiny random-policy fake batch on CPU. No dataset or checkpoint.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.smoke:
        report = run_smoke(torch.device(args.device) if args.device else torch.device("cpu"))
    else:
        missing = [
            name
            for name, value in (
                ("--checkpoint", args.checkpoint),
                ("--dataset-repo-id", args.dataset_repo_id),
                ("--labels", args.labels),
            )
            if not value
        ]
        if missing:
            raise ValueError(f"missing required arguments: {', '.join(missing)}")
        device = torch.device(
            args.device
            or (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        )
        report = run_ablation(
            args.checkpoint,
            dataset_repo_id=args.dataset_repo_id,
            labels=args.labels,
            train_split=args.train_split,
            device=device,
            max_frames=args.max_frames,
            batch_size=args.batch_size,
            episodes=args.episodes,
            num_workers=args.num_workers,
        )
    text = format_report(report)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
