"""CHECK 2: can a tiny model predict the binary advantage label I from o?

The RECAP policy is conditioned on I, the binarized n-step advantage. If I is
already a (shallow) function of the observation o = (images, qpos), then
I(a; I | o) cannot be large and the token is redundant with the encoder.

This is *not* a 20k ACT retrain. Preferred path: freeze a trained critic
backbone (same ACT encoder family, no advantage token) and fit a linear or
2-layer head on cached features. Fallback: a tiny CNN on downsampled images
plus a qpos MLP.

Metrics are ROC-AUC and accuracy on held-out *episodes*, excluding human
correction frames. Corrections were forced POSITIVE at labeling time, so
scoring them would just measure "does this look like an intervention".

Interpretation:
    high AUC  => I is readable from o, so I is largely redundant with o.
    AUC ~ 0.5 => I is not linearly / shallowly readable from these features
                 of o. That is NOT the same as I being action-conditional;
                 a deeper function of o could still predict I.

Example (local smoke, no dataset):
    PYTHONPATH=. .venv/bin/python -m models.recap.analysis.probe_advantage_from_obs --smoke

Example (full run; prefer Modal):
    RECAP_PROBE_DATASET_REPO_ID=RevanthGundala/034-recap-balanced \\
      RECAP_PROBE_LABELS=.../034-recap-balanced.json \\
      RECAP_PROBE_CRITIC_CHECKPOINT=.../best_critic.pt \\
      PYTHONPATH=. python -m models.recap.analysis.probe_advantage_from_obs
"""

from __future__ import annotations

from dataclasses import asdict, fields, replace
from datetime import datetime
import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

from configs.act.base import ACTConfig
from models.recap.act_critic import ACTCritic
from models.recap.act_recap import AdvantageCondition
from models.recap.train_critic import expected_return, load_episode_outcomes, split_episodes
from models.recap.train_policy import load_conditions


POSITIVE = int(AdvantageCondition.POSITIVE)
NEGATIVE = int(AdvantageCondition.NEGATIVE)
UNCONDITIONED = int(AdvantageCondition.UNCONDITIONED)


class LinearHead(nn.Module):
    """One linear layer from a frozen embedding to a binary logit."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Linear(in_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class MLPHead(nn.Module):
    """Two-layer probe; still tiny relative to the frozen encoder."""

    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        if hidden < 1:
            raise ValueError(f"hidden must be positive, got {hidden}")
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class TinyVisionQposProbe(nn.Module):
    """Fallback when no critic checkpoint is available.

    Downsamples both cameras, runs a two-stride conv stack, and concatenates
    qpos. No language tower: this task has no language input.
    """

    def __init__(self, joint_dim: int = 6, num_cameras: int = 2, hidden: int = 128):
        super().__init__()
        if num_cameras < 1:
            raise ValueError(f"num_cameras must be positive, got {num_cameras}")
        self.num_cameras = num_cameras
        self.conv = nn.Sequential(
            nn.Conv2d(3 * num_cameras, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(32 + joint_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, images: torch.Tensor, qpos: torch.Tensor) -> torch.Tensor:
        images = _prepare_probe_images(images)
        if images.shape[1] != self.num_cameras:
            raise ValueError(f"Expected {self.num_cameras} cameras, got {images.shape[1]}")
        down = F.interpolate(images.flatten(0, 1), size=(64, 64), mode="bilinear", align_corners=False)
        down = down.unflatten(0, (images.shape[0], self.num_cameras)).flatten(1, 2)
        vision = self.conv(down).flatten(1)
        return self.head(torch.cat([vision, qpos], dim=-1)).squeeze(-1)


def _prepare_probe_images(images: torch.Tensor) -> torch.Tensor:
    if images.dim() == 4:
        images = images.unsqueeze(1)
    if images.dim() != 5:
        raise ValueError(f"Expected images with 4 or 5 dims, got {tuple(images.shape)}")
    if images.shape[2] not in (1, 3) and images.shape[-1] in (1, 3):
        images = images.permute(0, 1, 4, 2, 3).contiguous()
    if not images.is_floating_point():
        images = images.float() / 255.0
    return images


def binary_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Mann-Whitney AUC; 0.5 on ties. Undefined if either class is missing."""
    scores = scores.detach().float().reshape(-1)
    labels = labels.detach().bool().reshape(-1)
    if scores.numel() != labels.numel():
        raise ValueError("scores and labels must be aligned")
    positive = scores[labels]
    negative = scores[~labels]
    if positive.numel() == 0 or negative.numel() == 0:
        return float("nan")
    wins = (positive[:, None] > negative[None, :]).float().mean()
    ties = (positive[:, None] == negative[None, :]).float().mean()
    return float(wins + 0.5 * ties)


def majority_accuracy(labels: torch.Tensor) -> float:
    if labels.numel() == 0:
        return float("nan")
    positive = float(labels.float().mean())
    return max(positive, 1.0 - positive)


def eval_mask(
    labels: torch.Tensor,
    intervention: torch.Tensor,
    *,
    exclude_intervention: bool = True,
) -> torch.Tensor:
    """Frames that may be scored: binary I, and not a forced-positive correction."""
    labels = labels.long().reshape(-1)
    intervention = intervention.bool().reshape(-1)
    if labels.numel() != intervention.numel():
        raise ValueError("labels and intervention must be aligned")
    binary = (labels == POSITIVE) | (labels == NEGATIVE)
    if exclude_intervention:
        return binary & ~intervention
    return binary


def compute_metrics(
    scores: torch.Tensor,
    labels: torch.Tensor,
    intervention: torch.Tensor | None = None,
    *,
    threshold: float = 0.0,
    exclude_intervention: bool = True,
) -> dict:
    """ROC-AUC and thresholded accuracy on the non-correction binary subset.

    ``scores`` are logits (or any ranking score). Accuracy uses ``threshold``
    on the raw score, so 0.0 is the natural cut for an uncalibrated logit.
    """
    if intervention is None:
        intervention = torch.zeros_like(labels, dtype=torch.bool)
    mask = eval_mask(labels, intervention, exclude_intervention=exclude_intervention)
    kept_scores = scores.reshape(-1)[mask]
    kept_labels = labels.reshape(-1)[mask] == POSITIVE
    excluded = int((~mask).sum())
    auc = binary_auc(kept_scores, kept_labels)
    oriented = oriented_auc(auc)
    if kept_labels.numel() == 0:
        accuracy = float("nan")
    else:
        accuracy = float(((kept_scores > threshold) == kept_labels).float().mean())
    return {
        "auc": auc,
        "oriented_auc": oriented,
        "accuracy": accuracy,
        "n": int(kept_labels.numel()),
        "n_excluded": excluded,
        "positive_rate": float(kept_labels.float().mean()) if kept_labels.numel() else float("nan"),
        "majority_accuracy": majority_accuracy(kept_labels),
        "interpretation": interpret_auc(auc),
    }


def oriented_auc(auc: float) -> float:
    """Readability of I from the score, ignoring sign.

    AUC=0 is a perfect anti-ranking, which still means I is a function of the
    features. Chance is 0.5; report max(auc, 1-auc).
    """
    if auc != auc:
        return auc
    return max(float(auc), 1.0 - float(auc))


def interpret_auc(auc: float) -> str:
    if auc != auc:  # NaN
        return "undefined (one class missing after excluding interventions)"
    strength = oriented_auc(auc)
    if strength >= 0.8:
        return (
            "HIGH: I is readable from o, so I is largely redundant with the "
            "observation. I(a; I | o) cannot be large."
        )
    if strength <= 0.6:
        return (
            "NEAR CHANCE: I is not linearly/shallowly readable from these "
            "features of o. This is NOT evidence that I is action-conditional; "
            "a deeper function of o could still predict I."
        )
    return (
        "MODERATE: o carries some information about I, but I is not a "
        "near-deterministic function of these features."
    )


def make_head(kind: str, in_dim: int, hidden: int = 128) -> nn.Module:
    if kind == "linear":
        return LinearHead(in_dim)
    if kind == "mlp":
        return MLPHead(in_dim, hidden=hidden)
    raise ValueError(f"head must be linear or mlp, got {kind!r}")


def train_head_on_features(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    kind: str = "mlp",
    hidden: int = 128,
    steps: int = 500,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 0,
    device: torch.device | None = None,
) -> tuple[nn.Module, list[float]]:
    """Fit a tiny classifier on *cached* frozen features. Labels are 0/1."""
    if features.ndim != 2:
        raise ValueError(f"features must be (N, D), got {tuple(features.shape)}")
    if labels.shape[0] != features.shape[0]:
        raise ValueError("features and labels must be aligned")
    if steps <= 0:
        raise ValueError("steps must be positive")
    device = device or features.device
    targets = (labels.long().reshape(-1) == POSITIVE).float()
    head = make_head(kind, features.shape[1], hidden=hidden).to(device)
    optim = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    dataset = TensorDataset(features.detach().float(), targets)
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        drop_last=False,
    )
    head.train()
    losses: list[float] = []
    step = 0
    while step < steps:
        for batch_x, batch_y in loader:
            optim.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(
                head(batch_x.to(device)),
                batch_y.to(device),
            )
            loss.backward()
            optim.step()
            losses.append(float(loss.detach()))
            step += 1
            if step >= steps:
                break
    head.eval()
    return head, losses


@torch.no_grad()
def score_head(head: nn.Module, features: torch.Tensor, device: torch.device) -> torch.Tensor:
    head.eval()
    return head(features.to(device).float()).cpu()


def config_from_critic_checkpoint(checkpoint_path: str | Path) -> ACTConfig:
    """Rebuild the critic architecture, ignoring checkpoint-only extra keys."""
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    raw = checkpoint.get("config")
    if not isinstance(raw, dict):
        raise ValueError(f"Checkpoint {checkpoint_path} has no config dictionary")
    allowed = {item.name for item in fields(ACTConfig)}
    source = {key: value for key, value in raw.items() if key in allowed}
    if "camera_names" in source:
        source["camera_names"] = tuple(source["camera_names"])
    if "dataset_repo_id" not in source:
        raise ValueError(f"Checkpoint {checkpoint_path} config is missing dataset_repo_id")
    return ACTConfig(**source)


def load_frozen_critic(
    checkpoint_path: str | Path, device: torch.device
) -> tuple[ACTCritic, ACTConfig, float, float]:
    cfg = config_from_critic_checkpoint(checkpoint_path)
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    model = ACTCritic(
        cfg,
        pretrained_backbone=False,
        num_bins=int(checkpoint.get("num_bins", cfg.recap_num_bins)),
    )
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state, strict=True)
    for parameter in model.parameters():
        parameter.requires_grad = False
    value_min = float(checkpoint.get("value_min", -1.0))
    value_max = float(checkpoint.get("value_max", 0.0))
    return model.to(device).eval(), cfg, value_min, value_max


class ProbeDataset(Dataset):
    """Observation + binary I, keyed by the dataset's global frame index.

    Wraps CriticDataset so image/qpos normalization matches the frozen critic.
    Intervention flags travel with each sample so metrics can drop corrections.
    """

    def __init__(self, inner: Dataset, conditions: torch.Tensor):
        self.inner = inner
        self.conditions = conditions.long()
        hf = inner.dataset.hf_dataset
        self.frame_ids = torch.as_tensor(hf["index"]).long()
        if "intervention" in hf.column_names:
            self.interventions = torch.as_tensor(hf["intervention"]).bool().flatten()
        else:
            self.interventions = torch.zeros(len(inner), dtype=torch.bool)
        self.episode_ids = torch.as_tensor(hf["episode_index"]).long().flatten()
        if len(self.frame_ids) != len(inner):
            raise ValueError(
                f"Got {len(self.frame_ids)} frame ids for {len(inner)} samples"
            )
        out_of_range = self.frame_ids >= len(self.conditions)
        if out_of_range.any():
            raise ValueError(
                f"Labels cover {len(self.conditions)} frames but the dataset "
                f"references index {int(self.frame_ids.max())}"
            )

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        item = self.inner[idx]
        frame_id = int(self.frame_ids[idx])
        item["label"] = self.conditions[frame_id]
        item["intervention"] = self.interventions[idx]
        item["frame_id"] = frame_id
        item["episode_id"] = int(self.episode_ids[idx])
        return item


@torch.no_grad()
def cache_critic_features(
    critic: ACTCritic,
    loader: DataLoader,
    device: torch.device,
    value_min: float = -1.0,
    value_max: float = 0.0,
) -> dict[str, torch.Tensor]:
    """One frozen forward over the split. Images are not kept."""
    critic.eval()
    blobs = {key: [] for key in ("features", "qpos", "values", "labels", "intervention", "episode_id")}
    for batch in loader:
        images = batch["images"].to(device)
        qpos = batch["qpos"].to(device)
        features = critic.encode(images, qpos)
        values = expected_return(
            torch.distributions.Categorical(logits=critic.decoder.value_head(features)),
            value_min,
            value_max,
        )
        blobs["features"].append(features.float().cpu())
        blobs["qpos"].append(qpos.float().cpu())
        blobs["values"].append(values.float().cpu())
        blobs["labels"].append(batch["label"].long().cpu().reshape(-1))
        blobs["intervention"].append(batch["intervention"].bool().cpu().reshape(-1))
        blobs["episode_id"].append(batch["episode_id"].long().cpu().reshape(-1))
    return {key: torch.cat(parts) for key, parts in blobs.items()}


def _trainable_mask(labels: torch.Tensor, intervention: torch.Tensor) -> torch.Tensor:
    """Train only on autonomous binary frames so the head cannot learn intervention=>POS."""
    return eval_mask(labels, intervention, exclude_intervention=True)


def fit_feature_probes(
    train: dict[str, torch.Tensor],
    val: dict[str, torch.Tensor],
    *,
    steps: int,
    seed: int,
    device: torch.device,
    hidden: int = 128,
    heads: tuple[str, ...] = ("linear", "mlp"),
) -> dict:
    train_mask = _trainable_mask(train["labels"], train["intervention"])
    reports = {}
    feature_sets = {
        "vision_qpos": train["features"],
        "qpos": train["qpos"],
        "value": train["values"].reshape(-1, 1),
    }
    val_sets = {
        "vision_qpos": val["features"],
        "qpos": val["qpos"],
        "value": val["values"].reshape(-1, 1),
    }
    for feature_name, train_x in feature_sets.items():
        for kind in heads:
            head, losses = train_head_on_features(
                train_x[train_mask],
                train["labels"][train_mask],
                kind=kind,
                hidden=hidden,
                steps=steps,
                seed=seed,
                device=device,
            )
            scores = score_head(head, val_sets[feature_name], device)
            metrics = compute_metrics(scores, val["labels"], val["intervention"])
            metrics["train_loss_last"] = losses[-1] if losses else float("nan")
            metrics["train_frames"] = int(train_mask.sum())
            reports[f"{feature_name}_{kind}"] = metrics
    return reports


def train_tiny_vision_probe(
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    joint_dim: int,
    num_cameras: int,
    steps: int,
    seed: int,
    device: torch.device,
    hidden: int = 128,
    lr: float = 1e-3,
) -> dict:
    torch.manual_seed(seed)
    model = TinyVisionQposProbe(joint_dim=joint_dim, num_cameras=num_cameras, hidden=hidden).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    step = 0
    last_loss = float("nan")
    while step < steps:
        for batch in train_loader:
            mask = _trainable_mask(batch["label"], batch["intervention"])
            if not bool(mask.any()):
                step += 1
                if step >= steps:
                    break
                continue
            images = batch["images"].to(device)[mask]
            qpos = batch["qpos"].to(device)[mask]
            targets = (batch["label"].to(device)[mask] == POSITIVE).float()
            optim.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(images, qpos), targets)
            loss.backward()
            optim.step()
            last_loss = float(loss.detach())
            step += 1
            if step >= steps:
                break
    scores, labels, interventions = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            scores.append(model(batch["images"].to(device), batch["qpos"].to(device)).cpu())
            labels.append(batch["label"].long().cpu().reshape(-1))
            interventions.append(batch["intervention"].bool().cpu().reshape(-1))
    metrics = compute_metrics(torch.cat(scores), torch.cat(labels), torch.cat(interventions))
    metrics["train_loss_last"] = last_loss
    return {"tiny_vision_qpos": metrics}


def smoke_test(device: str = "cpu", steps: int = 40) -> dict:
    """CPU-only sanity check on a few fake batches. Does not touch the dataset."""
    torch.manual_seed(0)
    device_t = torch.device(device)
    n, dim, cameras = 64, 6, 2
    qpos = torch.randn(n, dim)
    images = torch.randn(n, cameras, 3, 32, 32)
    # Plant a qpos-readable label so the qpos head must beat chance.
    labels = (qpos[:, 0] > 0).long()
    intervention = torch.zeros(n, dtype=torch.bool)
    intervention[:4] = True
    labels[:4] = POSITIVE
    features = torch.cat([qpos, torch.randn(n, 8)], dim=1)
    features = features.clone()
    features[:, 0] = qpos[:, 0]
    split = n // 2
    train = {
        "features": features[:split],
        "qpos": qpos[:split],
        "values": qpos[:split, 0],
        "labels": labels[:split],
        "intervention": intervention[:split],
        "episode_id": torch.zeros(split, dtype=torch.long),
    }
    val = {
        "features": features[split:],
        "qpos": qpos[split:],
        "values": qpos[split:, 0],
        "labels": labels[split:],
        "intervention": intervention[split:],
        "episode_id": torch.ones(n - split, dtype=torch.long),
    }
    reports = fit_feature_probes(train, val, steps=steps, seed=0, device=device_t, heads=("linear",))

    cfg = ACTConfig(
        dataset_repo_id="smoke/unused",
        camera_names=("wrist", "top"),
        chunk_size=2,
        d_model=32,
        joint_dim=dim,
        d_z=8,
        num_encoder_layers=1,
        num_decoder_layers=1,
        num_heads=4,
        mlp_dim=64,
        dropout=0.0,
    )
    critic = ACTCritic(cfg, pretrained_backbone=False, num_bins=11).to(device_t).eval()
    encoded = critic.encode(images[:8].to(device_t), qpos[:8].to(device_t))
    dist = critic(images[:8].to(device_t), qpos[:8].to(device_t))
    if encoded.shape != (8, cfg.d_model):
        raise RuntimeError(f"encode returned {tuple(encoded.shape)}")
    reconstructed = torch.distributions.Categorical(
        logits=critic.decoder.value_head(encoded)
    ).logits
    if not torch.allclose(reconstructed, dist.logits, atol=1e-5):
        raise RuntimeError("encode is not the value-head input")

    tiny = TinyVisionQposProbe(joint_dim=dim, num_cameras=cameras).to(device_t)
    tiny_logits = tiny(images[:8].to(device_t), qpos[:8].to(device_t))
    if tiny_logits.shape != (8,):
        raise RuntimeError(f"tiny probe returned {tuple(tiny_logits.shape)}")

    reports["encode_dim"] = list(encoded.shape)
    reports["tiny_batch_logits"] = list(tiny_logits.shape)
    reports["qpos_linear_auc"] = reports["qpos_linear"]["auc"]
    if reports["qpos_linear"]["auc"] < 0.7:
        raise RuntimeError(
            f"smoke qpos probe AUC {reports['qpos_linear']['auc']:.3f} "
            "failed to recover a planted qpos-threshold label"
        )
    return reports


def _required_env(*variables: str) -> str:
    for variable in variables:
        value = os.environ.get(variable)
        if value:
            return value
    raise ValueError(f"Set one of {', '.join(variables)}")


def config_from_env() -> ACTConfig:
    checkpoint = os.environ.get("RECAP_PROBE_CRITIC_CHECKPOINT")
    if checkpoint:
        base = config_from_critic_checkpoint(checkpoint)
    else:
        base = ACTConfig(dataset_repo_id=_required_env("RECAP_PROBE_DATASET_REPO_ID"))
    dataset_repo_id = os.environ.get("RECAP_PROBE_DATASET_REPO_ID")
    if dataset_repo_id:
        base = replace(base, dataset_repo_id=dataset_repo_id)
    return replace(
        base,
        job_name=os.environ.get("RECAP_PROBE_JOB_NAME", "recap_probe_i_from_o"),
        output_root=os.environ.get("RECAP_PROBE_OUTPUT_ROOT", base.output_root),
        num_train_steps=int(os.environ.get("RECAP_PROBE_STEPS", 2000)),
        image_transforms=False,
    )


def main() -> dict:
    from models.recap.train_critic import CriticDataset

    cfg = config_from_env()
    labels_path = os.environ.get("RECAP_PROBE_LABELS")
    if not labels_path:
        raise ValueError("Set RECAP_PROBE_LABELS to the advantage_labels JSON")
    critic_path = os.environ.get("RECAP_PROBE_CRITIC_CHECKPOINT")
    backend = os.environ.get("RECAP_PROBE_BACKEND", "auto")
    val_episodes = int(os.environ.get("RECAP_PROBE_VAL_EPISODES", 8))
    if val_episodes <= 0:
        raise ValueError("Held-out episodes are required; never train_split=1.0")
    if backend not in ("auto", "critic", "tiny"):
        raise ValueError(f"Unknown RECAP_PROBE_BACKEND={backend!r}")
    if backend == "critic" and not critic_path:
        raise ValueError("RECAP_PROBE_BACKEND=critic requires RECAP_PROBE_CRITIC_CHECKPOINT")
    use_critic = backend != "tiny" and bool(critic_path)
    if backend == "auto" and not critic_path:
        use_critic = False

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

    conditions = load_conditions(labels_path)
    probe_meta = CriticDataset(cfg)
    outcomes = load_episode_outcomes(probe_meta.dataset.root, probe_meta.dataset)
    train_episodes, val_episodes_ids = split_episodes(outcomes, val_episodes, cfg.seed)
    del probe_meta
    if not val_episodes_ids:
        raise ValueError("Held-out episodes are required; never train_split=1.0")

    train_data = ProbeDataset(CriticDataset(cfg, episodes=train_episodes), conditions)
    val_data = ProbeDataset(CriticDataset(cfg, episodes=val_episodes_ids), conditions)
    loader_kwargs = dict(
        batch_size=cfg.batch_size,
        num_workers=int(os.environ.get("RECAP_PROBE_NUM_WORKERS", cfg.num_workers)),
        pin_memory=device.type == "cuda",
    )
    train_loader = DataLoader(train_data, shuffle=False, **loader_kwargs)
    val_loader = DataLoader(val_data, shuffle=False, **loader_kwargs)

    run_dir = Path(cfg.output_root) / cfg.job_name / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)

    payload = {
        "dataset_repo_id": cfg.dataset_repo_id,
        "labels_path": labels_path,
        "critic_checkpoint": critic_path,
        "backend": "critic" if use_critic else "tiny",
        "train_episodes": train_episodes,
        "val_episodes": val_episodes_ids,
        "steps": cfg.num_train_steps,
        "seed": cfg.seed,
        "device": device_name,
        "note": (
            "Intervention frames are excluded from training and metrics because "
            "labeling forced them POSITIVE."
        ),
    }

    if use_critic:
        critic, critic_cfg, value_min, value_max = load_frozen_critic(critic_path, device)
        del critic_cfg
        print(f"Encoding train split ({len(train_data)} frames) with frozen critic...", flush=True)
        train_cache = cache_critic_features(
            critic, train_loader, device, value_min=value_min, value_max=value_max
        )
        print(f"Encoding val split ({len(val_data)} frames)...", flush=True)
        val_cache = cache_critic_features(
            critic, val_loader, device, value_min=value_min, value_max=value_max
        )
        reports = fit_feature_probes(
            train_cache,
            val_cache,
            steps=cfg.num_train_steps,
            seed=cfg.seed,
            device=device,
        )
        payload["n_train_frames"] = int(train_cache["labels"].numel())
        payload["n_val_frames"] = int(val_cache["labels"].numel())
    else:
        print("No critic checkpoint; training tiny CNN + qpos fallback...", flush=True)
        reports = train_tiny_vision_probe(
            train_loader,
            val_loader,
            joint_dim=cfg.joint_dim,
            num_cameras=len(cfg.camera_names),
            steps=cfg.num_train_steps,
            seed=cfg.seed,
            device=device,
        )
        # qpos-only still runs from a cheap cache of just qpos/labels.
        train_qpos, train_y, train_i = [], [], []
        val_qpos, val_y, val_i = [], [], []
        for batch in train_loader:
            train_qpos.append(batch["qpos"])
            train_y.append(batch["label"].long().reshape(-1))
            train_i.append(batch["intervention"].bool().reshape(-1))
        for batch in val_loader:
            val_qpos.append(batch["qpos"])
            val_y.append(batch["label"].long().reshape(-1))
            val_i.append(batch["intervention"].bool().reshape(-1))
        train_cache = {
            "features": torch.cat(train_qpos),
            "qpos": torch.cat(train_qpos),
            "values": torch.cat(train_qpos)[:, :1].squeeze(-1),
            "labels": torch.cat(train_y),
            "intervention": torch.cat(train_i),
        }
        val_cache = {
            "features": torch.cat(val_qpos),
            "qpos": torch.cat(val_qpos),
            "values": torch.cat(val_qpos)[:, :1].squeeze(-1),
            "labels": torch.cat(val_y),
            "intervention": torch.cat(val_i),
        }
        reports.update(
            fit_feature_probes(
                train_cache,
                val_cache,
                steps=cfg.num_train_steps,
                seed=cfg.seed,
                device=device,
                heads=("linear", "mlp"),
            )
        )
        payload["n_train_frames"] = int(train_cache["labels"].numel())
        payload["n_val_frames"] = int(val_cache["labels"].numel())

    payload["probes"] = reports
    (run_dir / "probe_report.json").write_text(json.dumps(payload, indent=2) + "\n")
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str) + "\n")
    print(json.dumps({"output": str(run_dir), "probes": reports}, indent=2, default=str), flush=True)
    return payload


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help="CPU fake-batch check; no dataset, no GPU")
    parser.add_argument("--dataset-repo-id", default=None)
    parser.add_argument("--labels", dest="labels_path", default=None)
    parser.add_argument("--critic-checkpoint", default=None)
    parser.add_argument("--job-name", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--val-episodes", type=int, default=None)
    parser.add_argument("--backend", choices=("auto", "critic", "tiny"), default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    if args.smoke:
        report = smoke_test()
        print(json.dumps(report, indent=2))
    else:
        if args.dataset_repo_id:
            os.environ["RECAP_PROBE_DATASET_REPO_ID"] = args.dataset_repo_id
        if args.labels_path:
            os.environ["RECAP_PROBE_LABELS"] = args.labels_path
        if args.critic_checkpoint:
            os.environ["RECAP_PROBE_CRITIC_CHECKPOINT"] = args.critic_checkpoint
        if args.job_name:
            os.environ["RECAP_PROBE_JOB_NAME"] = args.job_name
        if args.output_root:
            os.environ["RECAP_PROBE_OUTPUT_ROOT"] = args.output_root
        if args.steps is not None:
            os.environ["RECAP_PROBE_STEPS"] = str(args.steps)
        if args.val_episodes is not None:
            os.environ["RECAP_PROBE_VAL_EPISODES"] = str(args.val_episodes)
        if args.backend:
            os.environ["RECAP_PROBE_BACKEND"] = args.backend
        if args.device:
            os.environ["RECAP_DEVICE"] = args.device
        main()
