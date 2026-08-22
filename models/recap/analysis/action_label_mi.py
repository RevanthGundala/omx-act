"""Does I change the action at a fixed observation?

The advantage token is learnable iff I(a ; I | o) > 0: the same (or very
similar) observation must appear with different actions that receive different
labels. Matching on joints alone is not that test — sponge pose lives in the
images — so the default matcher is the frozen critic encoder.

Compared quantities, on cross-episode k-nearest neighbours:

* action distance to same-I vs opposite-I neighbours
* a circular-within-episode label shift as the null (rate and run structure
  preserved, alignment destroyed)
* residual action spread among neighbours (is a|o unimodal?)

Intervention frames are excluded by default: their POSITIVE label was forced.

Example (full 034 image encode belongs on Modal):

    PYTHONPATH=. python -m models.recap.analysis.action_label_mi \\
      --dataset-repo-id RevanthGundala/034-recap-balanced \\
      --labels .../034-recap-balanced.json \\
      --critic-checkpoint .../best_critic.pt
"""

from __future__ import annotations

from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from configs.act.base import ACTConfig
from models.recap.policy import AdvantageCondition
from train.recap.label_advantage import episode_layout, n_step_advantages
from train.recap.policy import load_conditions


POSITIVE = int(AdvantageCondition.POSITIVE)
NEGATIVE = int(AdvantageCondition.NEGATIVE)


def l2_normalize(features: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.clip(denom, 1e-8, None)


def zscore(values: np.ndarray) -> np.ndarray:
    return (values - values.mean(0)) / (values.std(0) + 1e-8)


def neighbor_indices(
    features: np.ndarray,
    episodes: np.ndarray,
    k: int = 40,
) -> np.ndarray:
    """Cross-episode kNN in cosine space. ``features`` should already be normalized."""
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    n = len(features)
    if n <= k:
        raise ValueError(f"need more than k={k} frames, got {n}")
    sim = features @ features.T
    same = episodes[:, None] == episodes[None, :]
    sim = np.where(same, -np.inf, sim)
    # argpartition then sort the shortlist so neighbour 0 is the nearest
    part = np.argpartition(-sim, kth=k, axis=1)[:, :k]
    row = np.arange(n)[:, None]
    order = np.argsort(-sim[row, part], axis=1)
    return part[row, order]


def pair_action_distances(actions: np.ndarray, neighbors: np.ndarray) -> np.ndarray:
    """RMS action distance from each frame to each of its neighbours."""
    delta = actions[:, None, :] - actions[neighbors]
    return np.linalg.norm(delta, axis=2) / np.sqrt(actions.shape[1])


def per_frame_lift(
    neighbor_action_dist: np.ndarray,
    neighbor_labels: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """opp-mean minus same-mean, nan when a frame has only one class of neighbour."""
    same = neighbor_labels == labels[:, None]
    n_same = same.sum(axis=1)
    n_opp = (~same).sum(axis=1)
    usable = (n_same > 0) & (n_opp > 0)
    lift = np.full(len(labels), np.nan, dtype=np.float64)
    if not usable.any():
        return lift
    same_mean = np.divide(
        (neighbor_action_dist * same).sum(axis=1),
        np.clip(n_same, 1, None),
    )
    opp_mean = np.divide(
        (neighbor_action_dist * ~same).sum(axis=1),
        np.clip(n_opp, 1, None),
    )
    lift[usable] = opp_mean[usable] - same_mean[usable]
    return lift


def circular_shift_labels(
    labels: np.ndarray,
    episodes: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    out = labels.copy()
    for episode in np.unique(episodes):
        index = np.flatnonzero(episodes == episode)
        if len(index) < 2:
            continue
        out[index] = np.roll(labels[index], int(rng.integers(1, len(index))))
    return out


def permutation_pvalue(observed: float, null: np.ndarray) -> float:
    return float((np.sum(np.abs(null) >= abs(observed)) + 1) / (len(null) + 1))


def summarize_matcher(
    name: str,
    features: np.ndarray,
    actions: np.ndarray,
    labels: np.ndarray,
    episodes: np.ndarray,
    *,
    k: int = 40,
    n_perm: int = 200,
    seed: int = 0,
) -> dict:
    """One matcher (critic features, or qpos) → lift vs circular-shift null."""
    if len(features) != len(actions) or len(features) != len(labels):
        raise ValueError("features, actions, and labels must be aligned")
    if len(np.unique(labels)) < 2:
        raise ValueError(f"{name}: need both POS and NEG in the scored set")

    feat = l2_normalize(np.asarray(features, dtype=np.float64))
    act = zscore(np.asarray(actions, dtype=np.float64))
    lab = np.asarray(labels)
    ep = np.asarray(episodes)
    neighbors = neighbor_indices(feat, ep, k=k)
    dists = pair_action_distances(act, neighbors)
    lift = per_frame_lift(dists, lab[neighbors], lab)
    usable = np.isfinite(lift)
    observed = float(np.nanmean(lift))
    scale = float(dists[usable].mean()) if usable.any() else float("nan")

    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        shifted = circular_shift_labels(lab, ep, rng)
        null[i] = float(np.nanmean(per_frame_lift(dists, shifted[neighbors], shifted)))

    null_mean = float(null.mean())
    null_std = float(null.std() + 1e-12)
    z = (observed - null_mean) / null_std
    residual_std = float(act[neighbors].std(axis=1).mean()) if len(act) else float("nan")
    return {
        "matcher": name,
        "n_frames": int(len(lab)),
        "n_paired": int(usable.sum()),
        "k": k,
        "n_perm": n_perm,
        "observed_lift": observed,
        "relative_lift_pct": float(100.0 * observed / scale) if scale else float("nan"),
        "null_mean": null_mean,
        "null_std": null_std,
        "z": float(z),
        "p": permutation_pvalue(observed - null_mean, null - null_mean),
        "mean_nn_action_dist": float(dists.min(axis=1).mean()),
        "mean_k_action_dist": float(dists.mean()),
        "residual_action_std": residual_std,
        "pos_rate": float((lab == POSITIVE).mean()),
    }


def encode_critic_features(
    dataset,
    model,
    device: torch.device,
    *,
    batch_size: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Pooled encoder features and scalar V(o) for every dataset frame."""
    from torch.utils.data import DataLoader

    from train.recap.critic import expected_return

    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    features, values = [], []
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            qpos = batch["qpos"].to(device)
            feat = model.encode(images, qpos)
            dist = model.decoder.value_head(feat)
            import torch.distributions as D

            value = expected_return(
                D.Categorical(logits=dist),
                dataset.value_min,
                dataset.value_max,
            )
            features.append(feat.float().cpu())
            values.append(value.float().cpu())
    return torch.cat(features).numpy(), torch.cat(values).numpy()


def load_transition_tables(dataset) -> dict[str, np.ndarray]:
    """Scalar columns from the HF table — no video decode."""
    columns = dataset.hf_dataset
    action = np.stack([np.asarray(row, dtype=np.float32) for row in columns["action"]])
    qpos = np.stack([np.asarray(row, dtype=np.float32) for row in columns["observation.state"]])
    return {
        "action": action,
        "qpos": qpos,
        "episode": np.asarray(columns["episode_index"], dtype=np.int64),
        "index": np.asarray(columns["index"], dtype=np.int64),
        "intervention": np.asarray(columns["intervention"], dtype=bool).reshape(-1),
        "return_to_go": np.asarray(columns["return_to_go"], dtype=np.float32).reshape(-1),
    }


def score_split(
    tables: dict[str, np.ndarray],
    labels: np.ndarray,
    *,
    critic_features: np.ndarray | None,
    mask: np.ndarray,
    k: int,
    n_perm: int,
    seed: int,
) -> dict:
    """Run qpos (control) and critic (real) matchers on one mask."""
    if int(mask.sum()) < k + 2:
        return {"skipped": True, "n": int(mask.sum())}
    lab = labels[mask]
    if len(np.unique(lab)) < 2:
        return {"skipped": True, "n": int(mask.sum()), "reason": "single_class"}
    payload = {"n": int(mask.sum()), "skipped": False}
    payload["qpos"] = summarize_matcher(
        "qpos",
        tables["qpos"][mask],
        tables["action"][mask],
        lab,
        tables["episode"][mask],
        k=k,
        n_perm=n_perm,
        seed=seed,
    )
    if critic_features is not None:
        payload["critic"] = summarize_matcher(
            "critic",
            critic_features[mask],
            tables["action"][mask],
            lab,
            tables["episode"][mask],
            k=k,
            n_perm=n_perm,
            seed=seed,
        )
    return payload


def run_check(
    *,
    dataset_repo_id: str,
    labels_path: str | Path,
    critic_checkpoint: str | Path | None,
    device: torch.device,
    k: int = 40,
    n_perm: int = 200,
    batch_size: int = 8,
    seed: int = 0,
) -> dict:
    from models.recap.analysis.probe_advantage import load_frozen_critic
    from train.recap.critic import CriticDataset

    cfg = ACTConfig(dataset_repo_id=dataset_repo_id)
    dataset = CriticDataset(cfg)
    tables = load_transition_tables(dataset.dataset)
    labels = load_conditions(labels_path).numpy()
    if len(labels) != len(tables["action"]):
        raise ValueError(
            f"labels cover {len(labels)} frames, dataset has {len(tables['action'])}"
        )

    critic_features = None
    values = None
    advantages = None
    if critic_checkpoint is not None:
        # Same loader as the I-from-o probe: drops checkpoint-only keys
        # (e.g. dataset_format) that current ACTConfig rejects.
        model, _cfg, value_min, value_max = load_frozen_critic(critic_checkpoint, device)
        dataset.value_min, dataset.value_max = value_min, value_max
        critic_features, values = encode_critic_features(
            dataset, model, device, batch_size=batch_size
        )
        layout = episode_layout(dataset.dataset)
        advantages = n_step_advantages(
            layout["returns_to_go"],
            torch.as_tensor(values),
            layout["episode_end"],
        ).numpy()

    intervention = tables["intervention"]
    fair = ~intervention
    report = {
        "dataset_repo_id": dataset_repo_id,
        "labels_path": str(labels_path),
        "critic_checkpoint": str(critic_checkpoint) if critic_checkpoint else None,
        "device": str(device),
        "k": k,
        "n_perm": n_perm,
        "n_frames": int(len(labels)),
        "n_intervention": int(intervention.sum()),
        "note": (
            "critic matcher is I(a;I|o). qpos matcher ignores cameras and is "
            "the known-bad control. Intervention frames excluded (forced POS)."
        ),
        "all_non_intervention": score_split(
            tables, labels, critic_features=critic_features, mask=fair,
            k=k, n_perm=n_perm, seed=seed,
        ),
    }
    if values is not None:
        report["value_std"] = float(np.std(values))
        report["advantage_std"] = float(np.std(advantages))
        report["advantage_mean"] = float(np.mean(advantages))
        report["advantage_quantiles"] = {
            str(q): float(np.quantile(advantages, q))
            for q in (0.1, 0.25, 0.5, 0.75, 0.9)
        }
    return report


def format_report(report: dict) -> str:
    lines = [
        "I(a ; I | o) matched-neighbour check",
        f"dataset: {report.get('dataset_repo_id')}  n={report.get('n_frames')}  "
        f"k={report.get('k')}  perms={report.get('n_perm')}",
        report.get("note", ""),
        "",
    ]
    block = report.get("all_non_intervention", {})
    if block.get("skipped"):
        lines.append(f"skipped: {block}")
        return "\n".join(lines)
    for key in ("critic", "qpos"):
        if key not in block:
            continue
        row = block[key]
        lines.append(
            f"{key:<8} lift={row['observed_lift']:+.5f}  "
            f"rel={row['relative_lift_pct']:+.2f}%  "
            f"z={row['z']:+.2f}  p={row['p']:.3f}  "
            f"paired={row['n_paired']}/{row['n_frames']}  "
            f"resid_std={row['residual_action_std']:.3f}"
        )
    if "advantage_std" in report:
        lines.append(
            f"continuous A: mean={report['advantage_mean']:.2f} "
            f"std={report['advantage_std']:.2f}  "
            f"V std={report['value_std']:.2f}"
        )
    return "\n".join(lines)


def smoke_report() -> dict:
    """CPU plumbing: planted action shift at fixed features must produce lift."""
    rng = np.random.default_rng(0)
    n, d, a_dim = 240, 8, 6
    episodes = np.repeat(np.arange(8), 30)
    features = rng.normal(size=(n, d))
    # Shared observation clusters: 8 poses, 30 visits each from different eps
    # already handled by episode ids. Plant: POS moves action +1 on dim 0.
    labels = np.where(rng.random(n) > 0.5, POSITIVE, NEGATIVE)
    actions = rng.normal(size=(n, a_dim)) * 0.05
    actions[labels == POSITIVE, 0] += 1.0
    planted = summarize_matcher(
        "planted", features, actions, labels, episodes, k=8, n_perm=40, seed=0
    )
    shuffled = circular_shift_labels(labels, episodes, rng)
    nullish = summarize_matcher(
        "shifted", features, actions, shuffled, episodes, k=8, n_perm=40, seed=1
    )
    return {"planted": planted, "shifted_labels": nullish}


def run_qpos_only_from_parquet(parquet_root: str | Path, labels_path: str | Path, *, k: int, n_perm: int, seed: int = 0) -> dict:
    """Control matcher that does not decode video. Not I(a;I|o)."""
    import pandas as pd

    root = Path(parquet_root)
    files = sorted(root.glob("data/**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet under {root}/data")
    df = pd.concat([pd.read_parquet(path) for path in files]).sort_values("index").reset_index(drop=True)
    tables = {
        "action": np.stack(df["action"].map(np.asarray)),
        "qpos": np.stack(df["observation.state"].map(np.asarray)),
        "episode": df["episode_index"].to_numpy(),
        "index": df["index"].to_numpy(),
        "intervention": df["intervention"].to_numpy().astype(bool),
        "return_to_go": df["return_to_go"].to_numpy().astype(np.float32),
    }
    labels = load_conditions(labels_path).numpy()
    report = {
        "dataset_repo_id": str(root),
        "labels_path": str(labels_path),
        "critic_checkpoint": None,
        "device": "cpu",
        "k": k,
        "n_perm": n_perm,
        "n_frames": int(len(labels)),
        "n_intervention": int(tables["intervention"].sum()),
        "note": "qpos-only control from parquet; cameras ignored.",
        "all_non_intervention": score_split(
            tables,
            labels,
            critic_features=None,
            mask=~tables["intervention"],
            k=k,
            n_perm=n_perm,
            seed=seed,
        ),
    }
    return report


def main() -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-repo-id", default=os.environ.get("RECAP_MI_DATASET_REPO_ID"))
    parser.add_argument("--labels", default=os.environ.get("RECAP_MI_LABELS"))
    parser.add_argument("--critic-checkpoint", default=os.environ.get("RECAP_MI_CRITIC_CHECKPOINT"))
    parser.add_argument("--parquet-root", default=os.environ.get("RECAP_MI_PARQUET_ROOT"))
    parser.add_argument("--k", type=int, default=int(os.environ.get("RECAP_MI_K", "40")))
    parser.add_argument("--n-perm", type=int, default=int(os.environ.get("RECAP_MI_N_PERM", "200")))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", default=os.environ.get("RECAP_MI_OUTPUT"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        report = smoke_report()
        print(json.dumps(report, indent=2))
        return report
    if args.parquet_root and args.labels and not args.critic_checkpoint:
        report = run_qpos_only_from_parquet(
            args.parquet_root, args.labels, k=args.k, n_perm=args.n_perm
        )
        print(format_report(report))
        if args.output:
            Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
        return report
    if not args.dataset_repo_id or not args.labels:
        raise SystemExit("Need --dataset-repo-id and --labels (or --smoke)")
    device_name = os.environ.get("RECAP_DEVICE")
    if device_name is None:
        device_name = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    report = run_check(
        dataset_repo_id=args.dataset_repo_id,
        labels_path=args.labels,
        critic_checkpoint=args.critic_checkpoint,
        device=torch.device(device_name),
        k=args.k,
        n_perm=args.n_perm,
        batch_size=args.batch_size,
    )
    report["started_at"] = datetime.now(timezone.utc).isoformat()
    print(format_report(report))
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {path}")
    return report


if __name__ == "__main__":
    main()
