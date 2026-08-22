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
import random
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from configs.act.base import ACTConfig
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.transforms.transforms import ImageTransforms, ImageTransformsConfig
from models.recap.critic import ACTCritic

# Sec. IV-A: the empirical return is discretized into B = 201 bins.
NUM_BINS = 201
IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGE_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def derive_return_metadata(dataset) -> tuple[int, float, dict[int, bool]]:
    """Recover the reward bounds and per-episode outcomes from the dataset.

    LeRobot's download filter may not fetch recap_manifest.json alongside a Hub
    copy, so the trainer must be able to reconstruct these from the stored
    columns. Reads only scalar columns, so no video is decoded.
    """
    import numpy as np

    columns = dataset.hf_dataset.select_columns(
        ["episode_index", "reward", "success"]
    ).to_dict()
    episodes = np.asarray(columns["episode_index"])
    rewards = np.asarray(columns["reward"], dtype=np.float64).ravel()
    successes = np.asarray(columns["success"]).ravel()

    unique = np.unique(episodes)
    max_episode_steps = max(int((episodes == index).sum()) for index in unique)
    failure_penalty = float(-rewards.min())
    if failure_penalty <= 0:
        raise ValueError(
            "Cannot derive failure_penalty: dataset contains no failure episodes"
        )
    outcomes = {
        int(index): bool(successes[episodes == index][0]) for index in unique
    }
    return max_episode_steps, failure_penalty, outcomes


def load_return_range(dataset_root: Path, dataset=None) -> tuple[float, float]:
    """Derive the raw return-to-go range that maps onto (-1, 0).

    build_dataset.py emits -1 per step with a terminal reward of 0 on success
    and -failure_penalty on failure, undiscounted. The best achievable return is
    therefore 0 (success at the terminal step) and the worst is the longest
    episode failing on its final step.
    """
    manifest_path = Path(dataset_root) / "recap_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        max_episode_steps = int(manifest["max_episode_steps"])
        failure_penalty = float(manifest["failure_penalty"])
    elif dataset is not None:
        max_episode_steps, failure_penalty, _ = derive_return_metadata(dataset)
    else:
        raise FileNotFoundError(
            f"Missing {manifest_path}; build the dataset with scripts/recap/build_dataset.py."
        )
    return -(max_episode_steps - 1) - failure_penalty, 0.0


def normalize_return(returns, value_min: float, value_max: float):
    """Map raw return-to-go into the paper's (-1, 0) value range (Sec. V-C)."""
    return ((returns - value_max) / (value_max - value_min)).clamp(-1.0, 0.0)


def denormalize_return(values, value_min: float, value_max: float):
    return values * (value_max - value_min) + value_max


def return_to_bin(returns, value_min: float, value_max: float, num_bins: int = NUM_BINS):
    normalized = normalize_return(returns, value_min, value_max)
    # The paper fixes B = 201 but is silent on bin spacing (uniform vs. HL-Gauss),
    # so we use uniform bins over the normalized (-1, 0) range.
    return ((normalized + 1.0) * num_bins).long().clamp(0, num_bins - 1)


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
    targets = return_to_bin(returns, value_min, value_max, distribution.logits.shape[-1])
    # Sec. IV-A: cross-entropy against the discretized empirical MC return.
    loss = F.cross_entropy(distribution.logits, targets)
    return loss, distribution


def load_episode_outcomes(dataset_root: Path, dataset=None) -> dict[int, bool]:
    """Map episode index to success, from the manifest build_dataset.py wrote."""
    manifest_path = Path(dataset_root) / "recap_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        return {
            int(entry["episode_index"]): entry["outcome_label"] == "success"
            for entry in manifest["episodes"]
        }
    if dataset is None:
        raise FileNotFoundError(f"Missing {manifest_path} and no dataset to derive from")
    return derive_return_metadata(dataset)[2]


def split_episodes(
    outcomes: dict[int, bool],
    num_val_episodes: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Hold out whole episodes, stratified by outcome.

    The split must be episode-level: consecutive frames within an episode are
    nearly identical, so a frame-level split would leak the answer and make
    validation loss meaningless. Stratifying keeps both outcomes in the
    validation set, without which success/failure separation cannot be scored.
    """
    if num_val_episodes <= 0:
        return sorted(outcomes), []
    successes = sorted(index for index, ok in outcomes.items() if ok)
    failures = sorted(index for index, ok in outcomes.items() if not ok)
    if min(len(successes), len(failures)) < 1:
        raise ValueError("Dataset needs both success and failure episodes")

    generator = random.Random(seed)
    generator.shuffle(successes)
    generator.shuffle(failures)
    num_val_successes = max(1, round(num_val_episodes * len(successes) / len(outcomes)))
    num_val_failures = max(1, num_val_episodes - num_val_successes)
    if num_val_successes >= len(successes) or num_val_failures >= len(failures):
        raise ValueError(
            f"Cannot hold out {num_val_episodes} episodes from "
            f"{len(successes)} successes and {len(failures)} failures"
        )
    val = successes[:num_val_successes] + failures[:num_val_failures]
    train = [index for index in outcomes if index not in set(val)]
    return sorted(train), sorted(val)


def load_actor_decoder_weights(critic: ACTCritic, checkpoint_path: Path) -> dict:
    """Warm-start the critic's encoder from a trained ACT policy.

    Everything except the value head is shape-compatible with ACT's decoder, so
    the critic can inherit the representation instead of starting from ImageNet.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    decoder_state = {
        key[len("decoder.") :]: value
        for key, value in state.items()
        if key.startswith("decoder.")
    }
    if not decoder_state:
        raise ValueError(f"{checkpoint_path} contains no decoder.* weights")
    critic_state = critic.decoder.state_dict()
    mismatched = {
        key: (tuple(value.shape), tuple(critic_state[key].shape))
        for key, value in decoder_state.items()
        if key in critic_state and value.shape != critic_state[key].shape
    }
    if mismatched:
        # q_emb is sized by chunk_size, so this is nearly always a config skew
        # between the actor checkpoint and the critic being initialized.
        details = ", ".join(
            f"{key}: checkpoint {actor} vs critic {mine}"
            for key, (actor, mine) in sorted(mismatched.items())
        )
        raise ValueError(
            f"Shape mismatch loading {checkpoint_path}: {details}. "
            "Set RECAP_CRITIC_CHUNK_SIZE to the actor's chunk_size."
        )
    incompatible = critic.decoder.load_state_dict(decoder_state, strict=False)
    unexpected = [key for key in incompatible.unexpected_keys if not key.startswith("act_proj")]
    missing = [key for key in incompatible.missing_keys if not key.startswith("value_head")]
    if unexpected or missing:
        raise ValueError(
            f"Unexpected ACT/critic mismatch; missing={missing} unexpected={unexpected}"
        )
    transferred = sum(
        value.numel()
        for key, value in decoder_state.items()
        if not key.startswith("act_proj")
    )
    return {"checkpoint": str(checkpoint_path), "transferred_params": transferred}


FREEZE_LEVELS = ("none", "backbone", "encoder", "all")


def freeze_critic(critic: ACTCritic, level: str) -> dict:
    """Freeze a prefix of the critic so a warm-started representation survives.

    With only a few thousand training frames against 35M parameters, full
    fine-tuning can destroy transferred weights before the value head converges.
    Levels are cumulative: 'backbone' freezes vision, 'encoder' also freezes
    scene fusion, and 'all' trains the value head alone.
    """
    if level not in FREEZE_LEVELS:
        raise ValueError(f"freeze level must be one of {FREEZE_LEVELS}, got {level!r}")
    frozen_prefixes: tuple[str, ...] = ()
    if level == "backbone":
        frozen_prefixes = ("backbone",)
    elif level == "encoder":
        frozen_prefixes = ("backbone", "image_proj", "transformer_encoder")
    elif level == "all":
        frozen_prefixes = tuple(
            name for name, _ in critic.decoder.named_children() if name != "value_head"
        ) + ("q_emb",)

    for name, parameter in critic.decoder.named_parameters():
        parameter.requires_grad = not name.startswith(frozen_prefixes) if frozen_prefixes else True

    trainable = sum(p.numel() for p in critic.parameters() if p.requires_grad)
    total = sum(p.numel() for p in critic.parameters())
    if trainable == 0:
        raise ValueError(f"freeze level {level!r} left no trainable parameters")
    return {"level": level, "trainable_params": trainable, "total_params": total}


@torch.no_grad()
def evaluate(model, loader, device, value_min: float, value_max: float) -> dict:
    """Held-out return accuracy plus success/failure separation.

    Return MAE answers "did it learn the value function", and AUC answers the
    question that actually matters for RECAP: does V(o) rank eventual successes
    above eventual failures on episodes it never saw?
    """
    if loader is None:
        return {}
    was_training = model.training
    model.eval()
    losses, errors, values, labels, targets = [], [], [], [], []
    for batch in loader:
        returns = batch["return_to_go"].to(device)
        loss, distribution = compute_loss(model, batch, device, value_min, value_max)
        predicted = expected_return(distribution, value_min, value_max)
        losses.append(float(loss) * returns.shape[0])
        errors.append(float((predicted - returns).abs().sum()))
        values.append(predicted.cpu())
        targets.append(returns.cpu())
        labels.append(batch["success"].cpu())
    model.train(was_training)

    total = sum(value.numel() for value in values)
    values = torch.cat(values)
    targets = torch.cat(targets)
    labels = torch.cat(labels).bool()
    metrics = {
        "val_loss": sum(losses) / max(total, 1),
        "val_return_mae": sum(errors) / max(total, 1),
        "val_frames": total,
    }
    # The best constant predictor on the held-out returns. This is a property of
    # the data, so it must not change as the model trains -- if it drifts, the
    # baseline is being computed from predictions rather than ground truth.
    metrics["val_return_mae_baseline"] = float(
        (targets.mean() - targets).abs().mean()
    )
    if labels.any() and (~labels).any():
        positive = values[labels]
        negative = values[~labels]
        wins = (positive[:, None] > negative[None, :]).float().mean()
        ties = (positive[:, None] == negative[None, :]).float().mean()
        metrics["val_success_auc"] = float(wins + 0.5 * ties)
    return metrics


class CriticDataset(Dataset):
    """Observation plus Monte Carlo return-to-go; the critic never sees actions."""

    def __init__(self, cfg: ACTConfig, episodes: list[int] | None = None):
        # build_dataset.py writes to an arbitrary --output-root, which need not
        # match the HF cache layout LeRobot resolves repo ids against.
        root = os.environ.get("RECAP_CRITIC_DATASET_ROOT")
        root = Path(root).expanduser() if root else None
        metadata = LeRobotDatasetMetadata(
            repo_id=cfg.dataset_repo_id,
            revision=cfg.dataset_revision,
            root=root,
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
            episodes=episodes,
        )
        self.value_min, self.value_max = load_return_range(self.dataset.root, self.dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        qpos = sample["observation.state"].float()
        if qpos.ndim == 2 and qpos.shape[0] == 1:
            qpos = qpos[0]

        images = []
        for camera in self.camera_names:
            image = sample[f"observation.images.{camera}"].float()
            if image.ndim == 4 and image.shape[0] == 1:
                image = image[0]
            if image.shape[0] != 3 and image.shape[-1] == 3:
                image = image.permute(2, 0, 1).contiguous()
            if image.shape[0] != 3:
                raise ValueError(f"Expected {camera} RGB image, got {tuple(image.shape)}")
            if image.max() > 1.0:
                image = image / 255.0
            images.append((image - IMAGE_MEAN) / IMAGE_STD)

        return {
            "qpos": (qpos - self.qpos_mean) / self.qpos_std,
            "images": torch.stack(images, dim=0),
            "return_to_go": sample["return_to_go"].float().flatten()[0],
            "success": sample["success"].bool().flatten()[0],
        }


def _required_dataset_repo_id(*variables: str) -> str:
    """Read the dataset from the environment, refusing to guess.

    This used to fall back to a module-level constant, which for a long time
    named an abandoned task. A run with the variable unset then trained
    successfully on the wrong dataset and said nothing.
    """
    for variable in variables:
        value = os.environ.get(variable)
        if value:
            return value
    raise ValueError(
        f"Set one of {', '.join(variables)} to the dataset this run should use"
    )


def config_from_env() -> ACTConfig:
    base_cfg = ACTConfig(
        dataset_repo_id=_required_dataset_repo_id("RECAP_CRITIC_DATASET_REPO_ID")
    )
    return replace(
        base_cfg,
        job_name=os.environ.get("RECAP_CRITIC_JOB_NAME", "recap_critic"),
        output_root=os.environ.get("RECAP_CRITIC_OUTPUT_ROOT", base_cfg.output_root),
        num_train_steps=int(os.environ.get("RECAP_CRITIC_STEPS", base_cfg.num_train_steps)),
        save_freq=int(os.environ.get("RECAP_CRITIC_SAVE_FREQ", base_cfg.save_freq)),
        # q_emb is sized by chunk_size, so warm-starting from an actor requires
        # this to match that actor's chunk_size.
        chunk_size=int(os.environ.get("RECAP_CRITIC_CHUNK_SIZE") or base_cfg.chunk_size),
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

    probe = CriticDataset(cfg)
    value_min, value_max = probe.value_min, probe.value_max
    outcomes = load_episode_outcomes(probe.dataset.root, probe.dataset)
    num_val_episodes = int(os.environ.get("RECAP_CRITIC_VAL_EPISODES", 4))
    train_episodes, val_episodes = split_episodes(outcomes, num_val_episodes, cfg.seed)
    del probe

    dataset = CriticDataset(cfg, episodes=train_episodes)
    loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(cfg.seed),
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = None
    if val_episodes:
        val_loader = DataLoader(
            dataset=CriticDataset(cfg, episodes=val_episodes),
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    model = ACTCritic(cfg, pretrained_backbone=True, num_bins=NUM_BINS).to(device)
    warm_start = None
    init_from = os.environ.get("RECAP_CRITIC_INIT_FROM")
    if init_from:
        warm_start = load_actor_decoder_weights(model, Path(init_from).expanduser())
        model.to(device)
        print(
            f"Warm-started from {warm_start['checkpoint']} "
            f"({warm_start['transferred_params']:,} params)"
        )
    freeze = freeze_critic(model, os.environ.get("RECAP_CRITIC_FREEZE", "none"))
    print(
        f"Freeze level={freeze['level']}: "
        f"{freeze['trainable_params']:,} / {freeze['total_params']:,} params trainable "
        f"({100 * freeze['trainable_params'] / freeze['total_params']:.1f}%)"
    )
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    run_dir = Path(cfg.output_root) / cfg.job_name / datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                **asdict(cfg),
                "train_episodes": train_episodes,
                "val_episodes": val_episodes,
                "warm_start": warm_start,
                "freeze": freeze,
                "num_bins": NUM_BINS,
            },
            indent=2,
        )
        + "\n"
    )
    # Keep at least one mid-run checkpoint so the final step is never the first save.
    save_freq = max(1, min(cfg.save_freq, cfg.num_train_steps // 2))

    print(f"Dataset: {cfg.dataset_repo_id}@{cfg.dataset_revision}")
    print(f"Return range: [{value_min:g}, {value_max:g}] -> (-1, 0) over {NUM_BINS} bins")
    print(
        f"Episodes: {len(train_episodes)} train / {len(val_episodes)} val "
        f"(held out {val_episodes})"
    )
    print(f"Frames: {len(dataset)} train")
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
                val_metrics = evaluate(model, val_loader, device, value_min, value_max)
                if val_metrics:
                    metrics = {**metrics, **val_metrics}
                    with (run_dir / "metrics.jsonl").open("a") as file:
                        file.write(json.dumps(metrics, sort_keys=True) + "\n")
                    print(
                        f"  [val] step={step + 1} loss={val_metrics['val_loss']:.4f} "
                        f"return_mae={val_metrics['val_return_mae']:.1f} "
                        f"(mean-baseline {val_metrics['val_return_mae_baseline']:.1f}) "
                        f"success_auc={val_metrics.get('val_success_auc', float('nan')):.3f}",
                        flush=True,
                    )
                save_checkpoint(
                    checkpoint_dir, step + 1, model, optim, cfg, metrics, value_min, value_max
                )

            step += 1
            if step >= cfg.num_train_steps:
                break


if __name__ == "__main__":
    main()
