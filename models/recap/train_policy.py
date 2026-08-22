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

from dataclasses import asdict, fields, replace
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


def synthetic_conditions(frame_ids: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Deterministic coin-flip tokens, one bit per absolute frame id.

    Used by the capability test: the bit is independent of the critic, so a
    policy that learns it cannot be riding our real advantage labels.
    """
    mixed = frame_ids.to(torch.int64) * 2654435761 + int(seed) * 1_000_003
    return (mixed % 2).long()


def shift_actions_for_synthetic_token(
    actions: torch.Tensor,
    conditions: torch.Tensor,
    shift: float,
    first_action_only: bool = False,
) -> torch.Tensor:
    """Add ``shift`` (in z-units) to every joint when the token is POSITIVE.

    ``first_action_only`` restricts the shift to chunk offset 0. One token has to
    move all 30 predicted timesteps, so a signal that is real at the first action
    is diluted across the chunk; comparing the two says whether that dilution is
    what hides a small effect.
    """
    if shift <= 0:
        raise ValueError(f"synthetic shift must be positive, got {shift}")
    positive = conditions == int(AdvantageCondition.POSITIVE)
    while positive.ndim < actions.ndim:
        positive = positive.unsqueeze(-1)
    delta = shift * positive.to(dtype=actions.dtype).expand_as(actions).clone()
    if first_action_only and actions.ndim >= 2:
        # Chunk axis is second from last: (..., chunk, joints).
        delta[..., 1:, :] = 0.0
    return actions + delta


def shuffle_labels_within_episode(
    conditions: torch.Tensor, episode_ids: torch.Tensor, seed: int = 0
) -> torch.Tensor:
    """Circularly shift each episode's labels, destroying only their alignment.

    The positive rate and the run structure are preserved exactly, so a policy
    trained on these cannot beat one trained on the real labels unless the real
    labels are aligned with something. This is the null the real-label run has
    never been compared against.
    """
    generator = torch.Generator().manual_seed(seed)
    shuffled = conditions.clone()
    for episode in torch.unique(episode_ids):
        index = torch.nonzero(episode_ids == episode, as_tuple=True)[0]
        if len(index) < 2:
            continue
        offset = int(torch.randint(1, len(index), (1,), generator=generator))
        shuffled[index] = conditions[index].roll(offset)
    return shuffled


class AdvantageDataset(Dataset):
    """Attach a conditioning token to each ACT training sample.

    ``dataset`` is a models.act.helper.ACTDataset. ``conditions`` is indexed by
    global frame index; passing None pins every sample to POSITIVE for the SFT
    stage.

    ``synthetic_shift`` replaces the labels with a coin flip and adds that
    many z-units to the action target on POSITIVE frames. The token is then
    the only way to reduce L1, which is the capability test for whether this
    trainer can learn to use a token at all.
    """

    def __init__(
        self,
        dataset,
        conditions: torch.Tensor | None = None,
        frame_ids: torch.Tensor | None = None,
        total_frames: int | None = None,
        synthetic_shift: float | None = None,
        synthetic_seed: int = 0,
        synthetic_first_action_only: bool = False,
    ):
        self.dataset = dataset
        self.synthetic_shift = synthetic_shift
        self.synthetic_seed = synthetic_seed
        self.synthetic_first_action_only = synthetic_first_action_only
        if synthetic_shift is not None and synthetic_shift <= 0:
            raise ValueError(f"synthetic shift must be positive, got {synthetic_shift}")
        if synthetic_shift is not None and conditions is not None:
            raise ValueError(
                "synthetic_shift replaces the advantage labels; do not also pass conditions"
            )
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
            if total_frames is not None and total_frames != len(conditions):
                # Equal length is what ties the labels to this dataset. A
                # shorter dataset keeps every frame id in range, so the check
                # above passes while each transition trains against another
                # transition's token -- silently, and only for some frames.
                raise ValueError(
                    f"Labels cover {len(conditions)} frames but the dataset has "
                    f"{total_frames}; these labels were built for a different "
                    "dataset. Relabel with models/recap/label_advantage.py, or "
                    "point RECAP_POLICY_DATASET_REPO_ID at the labeled dataset."
                )

    def __len__(self):
        return len(self.dataset)

    def _condition_for(self, idx: int) -> torch.Tensor:
        if self.synthetic_shift is not None:
            return synthetic_conditions(self.frame_ids[idx : idx + 1], self.synthetic_seed)[0]
        if self.conditions is None:
            return torch.tensor(int(AdvantageCondition.POSITIVE), dtype=torch.long)
        return self.conditions[self.frame_ids[idx]]

    def __getitem__(self, idx):
        item = self.dataset[idx]
        condition = self._condition_for(idx)
        if self.synthetic_shift is not None:
            item["actions"] = shift_actions_for_synthetic_token(
                item["actions"],
                condition,
                self.synthetic_shift,
                self.synthetic_first_action_only,
            )
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


def compute_loss(model, batch, device, kl_weight, zero_latent: bool = False):
    """L1 + KL, matching models/act/train.py, with the advantage token added.

    ``zero_latent`` withholds the target chunk from the CVAE encoder, so z is the
    zero vector it already is at inference. During normal training z is inferred
    from the very actions being predicted, which gives the decoder a channel that
    carries the answer -- and a token competing with that channel can be ignored
    for reasons that have nothing to do with the token. Running with it off is
    how to tell those two explanations apart.
    """
    images = batch["images"].to(device)
    qpos = batch["qpos"].to(device)
    actions = batch["actions"].to(device)
    action_mask = batch["action_mask"].to(device)
    advantage = batch["advantage_condition"].to(device)
    act_pred, mu, log_var = model(
        images,
        qpos,
        None if zero_latent else actions,
        action_mask,
        advantage=advantage,
    )
    action_error = F.l1_loss(act_pred, actions, reduction="none")
    valid_values = (action_mask.sum() * actions.shape[-1]).clamp_min(1.0)
    action_loss = (action_error * action_mask.unsqueeze(-1)).sum() / valid_values
    if mu is None or log_var is None:
        kl_loss = torch.zeros((), device=action_loss.device, dtype=action_loss.dtype)
    else:
        kl_loss = (-0.5 * (1 + log_var - mu.pow(2) - log_var.exp())).sum(dim=1).mean()
    return action_loss + kl_weight * kl_loss, action_loss, kl_loss


def warm_start(
    model: ACTPolicy,
    checkpoint_path: str | Path,
    device: torch.device,
    *,
    allowed_missing: tuple[str, ...] = ADVANTAGE_PARAMETERS,
) -> ACTPolicy:
    """Graft a behavior-cloned ACT checkpoint onto the advantage-conditioned policy.

    ``allowed_missing`` names the parameters the checkpoint is expected not to
    have, because they are what this architecture adds. It is an argument rather
    than a constant: act_recap_masked.py adds a different set.
    """
    checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint {checkpoint_path} has no model_state_dict")
    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(
            f"Checkpoint {checkpoint_path} has unexpected parameters: {incompatible.unexpected_keys}"
        )
    unexpected_missing = set(incompatible.missing_keys) - set(allowed_missing)
    if unexpected_missing:
        raise ValueError(
            f"Checkpoint {checkpoint_path} is missing non-advantage parameters: {sorted(unexpected_missing)}"
        )
    return model


def config_from_checkpoint(checkpoint_path: str | Path) -> ACTConfig:
    """Rebuild the architecture and hyperparameters a checkpoint was trained with.

    The advantage-conditioned policy is a warm start of an existing ACT run, so
    inheriting that run's config is what keeps the comparison controlled and what
    makes the weights loadable at all. The baseline used chunk_size=30 and
    train_split=1.0, neither of which is the ACTConfig default.
    """
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    raw = checkpoint.get("config")
    if not isinstance(raw, dict):
        raise ValueError(f"Checkpoint {checkpoint_path} has no config dictionary")
    allowed = {item.name for item in fields(ACTConfig)}
    source = {key: value for key, value in raw.items() if key in allowed}
    if "camera_names" in source:
        source["camera_names"] = tuple(source["camera_names"])
    return ACTConfig(**source)


def parse_episode_subset(spec: str | None) -> list[int] | None:
    """Parse RECAP_POLICY_EPISODES, e.g. "48-58,60" -> [48..58, 60].

    Restricting the training set does not require relabeling: labels are indexed
    by absolute frame id, so a subset trains against exactly the tokens it would
    have received as part of the whole dataset.
    """
    if spec is None or not spec.strip():
        return None
    episodes: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part.lstrip("-"):
            start, _, end = part.partition("-")
            start_index, end_index = int(start), int(end)
            if end_index < start_index:
                raise ValueError(f"Empty episode range {part!r}")
            episodes.update(range(start_index, end_index + 1))
        else:
            episodes.add(int(part))
    if any(episode < 0 for episode in episodes):
        raise ValueError(f"Negative episode index in {spec!r}")
    if not episodes:
        raise ValueError(f"No episodes parsed from {spec!r}")
    return sorted(episodes)


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
    pretrained = os.environ.get("RECAP_POLICY_PRETRAINED_CHECKPOINT")
    if pretrained:
        base_cfg = config_from_checkpoint(pretrained)
    else:
        base_cfg = ACTConfig(
            dataset_repo_id=_required_dataset_repo_id("RECAP_POLICY_DATASET_REPO_ID")
        )
    dataset_repo_id = os.environ.get("RECAP_POLICY_DATASET_REPO_ID")
    if dataset_repo_id:
        base_cfg = replace(base_cfg, dataset_repo_id=dataset_repo_id)
    return replace(
        base_cfg,
        job_name=os.environ.get("RECAP_POLICY_JOB_NAME", "recap_policy"),
        # Never inherit the checkpoint's output_root: a run made on Modal records
        # a container-absolute path that would be unwritable elsewhere.
        output_root=os.environ.get("RECAP_POLICY_OUTPUT_ROOT", ACTConfig.output_root),
        num_train_steps=int(os.environ.get("RECAP_POLICY_STEPS", base_cfg.num_train_steps)),
        save_freq=int(os.environ.get("RECAP_POLICY_SAVE_FREQ", base_cfg.save_freq)),
        log_freq=int(os.environ.get("RECAP_POLICY_LOG_FREQ", base_cfg.log_freq)),
        train_split=float(
            os.environ.get("RECAP_POLICY_TRAIN_SPLIT", base_cfg.train_split)
        ),
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
    # Imported inside main so that importing this module for its dataset and
    # labelling helpers does not pull in LeRobot.
    from models.act.helper import make_dataloaders

    cfg = config_from_env()
    device_name = os.environ.get("RECAP_DEVICE")
    if device_name is None:
        device_name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_name)
    torch.manual_seed(cfg.seed)

    labels_path = os.environ.get("RECAP_POLICY_LABELS")
    conditions = load_conditions(labels_path) if labels_path else None
    zero_latent = os.environ.get("RECAP_POLICY_ZERO_LATENT") == "1"
    shuffle_labels = os.environ.get("RECAP_POLICY_SHUFFLE_LABELS") == "1"
    first_action_only = os.environ.get("RECAP_POLICY_SYNTHETIC_FIRST_ACTION") == "1"
    synthetic_raw = os.environ.get("RECAP_POLICY_SYNTHETIC_SHIFT")
    if synthetic_raw and labels_path:
        raise ValueError(
            "RECAP_POLICY_SYNTHETIC_SHIFT replaces the critic labels; "
            "unset RECAP_POLICY_LABELS"
        )
    synthetic_shift = float(synthetic_raw) if synthetic_raw else None
    # SFT pins the indicator to True, so dropout only applies once real labels do.
    # The synthetic capability test also defaults to 0: dropout would hide the
    # planted bit on 30% of updates and muddy the only question we are asking.
    default_dropout = (
        0.0
        if synthetic_shift is not None or conditions is None
        else cfg.recap_conditioning_dropout
    )
    dropout = float(os.environ.get("RECAP_POLICY_DROPOUT", default_dropout))

    episodes = parse_episode_subset(os.environ.get("RECAP_POLICY_EPISODES"))
    train_loader, val_loader = make_dataloaders(cfg, episodes=episodes)

    if shuffle_labels:
        if conditions is None:
            raise ValueError("RECAP_POLICY_SHUFFLE_LABELS needs real labels to shuffle")
        episode_ids = torch.as_tensor(
            train_loader.dataset.dataset.hf_dataset["episode_index"]
        )
        frame_ids = torch.as_tensor(train_loader.dataset.dataset.hf_dataset["index"])
        conditions = conditions.clone()
        conditions[frame_ids] = shuffle_labels_within_episode(
            conditions[frame_ids], episode_ids, cfg.seed
        )
        print(
            "SHUFFLED labels: rate and run structure preserved, alignment destroyed",
            flush=True,
        )
    # The label file is per dataset, so compare against the full dataset rather
    # than this split: make_dataloaders holds episodes out and may be restricted
    # to a subset, and a smaller training set is not evidence that the labels
    # belong to another dataset. Read the count from metadata, because the
    # loader's hf_dataset only covers the episodes actually being trained on.
    from lerobot.datasets import LeRobotDatasetMetadata
    from models.act.helper import local_dataset_root

    total_frames = LeRobotDatasetMetadata(
        repo_id=cfg.dataset_repo_id,
        revision=cfg.dataset_revision,
        root=local_dataset_root(),
    ).total_frames
    train_loader = torch.utils.data.DataLoader(
        dataset=AdvantageDataset(
            train_loader.dataset,
            conditions,
            total_frames=total_frames,
            synthetic_shift=synthetic_shift,
            synthetic_seed=cfg.seed,
            synthetic_first_action_only=first_action_only,
        ),
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

    if synthetic_shift is not None:
        stage = f"synthetic capability test (shift={synthetic_shift:g} z-units)"
    elif conditions is not None:
        stage = "improvement loop"
    else:
        stage = "SFT (indicator fixed to POSITIVE)"
    if synthetic_shift is not None:
        (run_dir / "synthetic.json").write_text(
            json.dumps({"shift": synthetic_shift, "seed": cfg.seed}) + "\n"
        )
    print(f"Dataset: {cfg.dataset_repo_id}@{cfg.dataset_revision}")
    if episodes is not None:
        (run_dir / "episodes.json").write_text(json.dumps(episodes) + "\n")
        print(f"Episode subset: {len(episodes)} episodes -> {episodes}")
    print(f"Stage: {stage}; conditioning dropout={dropout:g}")
    print(
        f"Inherited architecture: chunk_size={cfg.chunk_size} cameras={cfg.camera_names} "
        f"train_split={cfg.train_split} lr={cfg.learning_rate:g} steps={cfg.num_train_steps}"
    )
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
            loss, action_loss, kl_loss = compute_loss(
                model, batch, device, cfg.kl_weight, zero_latent=zero_latent
            )
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
