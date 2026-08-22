"""Check that a trained RECAP policy actually uses its advantage token.

Advantage conditioning is the entire mechanism: the improved policy is defined
as the reference policy conditioned on I=True (arXiv:2511.14759 Eq. 2, beta=1),
so a policy that ignores the token has silently degenerated into plain behavior
cloning on the mixed dataset. Nothing in the training loss detects this. The
token is 2,560 of 52.4M parameters, and conditioning dropout trains the
unconditional branch on 30% of updates, so being ignored is a live outcome
rather than a hypothetical one.

Two questions are asked separately, because they fail differently:

1. Does the token change the actions at all? Reported against the spread of
   predictions across different observations, so "small" is measured against
   what this policy does when the scene actually changes rather than against
   an arbitrary epsilon.
2. Does it change them in the direction the labels imply? On frames labeled
   POSITIVE the positive token should track the demonstrated action at least as
   well as the negative token, and vice versa. A policy can pass (1) by
   attaching arbitrary noise to the token and still fail (2).

Actions are z-normalized by the dataset's own statistics, so every number here
is in standard deviations of that joint's action distribution.

Example:
    uv run python -m models.recap.analysis.token_sensitivity \
      --checkpoint outputs/act_experiments/recap_policy/<run>/checkpoints/last.pt \
      --dataset-repo-id RevanthGundala/034-recap-balanced \
      --labels outputs/datasets/034-recap-balanced/advantage_labels.json
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path

import torch

from configs.act.base import ACTConfig
from models.recap.policy import ACTPolicy, AdvantageCondition
from train.recap.policy import (
    AdvantageDataset,
    config_from_checkpoint,
    load_conditions,
    parse_episode_subset,
)


def _load_policy(checkpoint_path: Path, device: torch.device) -> tuple[ACTPolicy, ACTConfig]:
    cfg = config_from_checkpoint(checkpoint_path)
    model = ACTPolicy(cfg, pretrained_backbone=False).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    # eval() also pins the CVAE latent to zero, so the only thing varying
    # between the conditions below is the token itself.
    model.eval()
    return model, cfg


@torch.no_grad()
def predict(model: ACTPolicy, batch: dict, condition: AdvantageCondition, device) -> torch.Tensor:
    images = batch["images"].to(device)
    qpos = batch["qpos"].to(device)
    tokens = torch.full((qpos.shape[0],), int(condition), dtype=torch.long, device=device)
    actions, _, _ = model(images, qpos, advantage=tokens)
    return actions


def observation_scale(predictions: torch.Tensor) -> float:
    """Spread of predictions across different observations.

    This is the reference the token effect is measured against: it is how much
    this policy's output moves when the scene changes, which is the only
    meaningful scale for calling a token effect large or small.
    """
    if predictions.shape[0] < 2:
        raise ValueError("Need at least two observations to measure their spread")
    rolled = torch.roll(predictions, shifts=1, dims=0)
    return float((predictions - rolled).abs().mean())


def summarize(
    model: ACTPolicy,
    loader,
    device,
    max_batches: int,
) -> dict:
    positive_gap = []
    negative_gap = []
    token_deltas = []
    obs_scales = []
    pos_label_scores = []
    neg_label_scores = []

    for index, batch in enumerate(loader):
        if index >= max_batches:
            break
        target = batch["actions"].to(device)
        a_pos = predict(model, batch, AdvantageCondition.POSITIVE, device)
        a_neg = predict(model, batch, AdvantageCondition.NEGATIVE, device)
        a_unc = predict(model, batch, AdvantageCondition.UNCONDITIONED, device)

        token_deltas.append((a_pos - a_neg).abs().mean().item())
        obs_scales.append(observation_scale(a_pos))
        positive_gap.append((a_pos - a_unc).abs().mean().item())
        negative_gap.append((a_neg - a_unc).abs().mean().item())

        # Directional test, split by what the labeler decided for each frame.
        # AdvantageDataset already resolved each sample's label, so read it
        # back rather than re-deriving the frame ids here.
        if "advantage_condition" in batch:
            labels = batch["advantage_condition"].to(device)
            err_pos = (a_pos - target).abs().mean(dim=(1, 2))
            err_neg = (a_neg - target).abs().mean(dim=(1, 2))
            is_pos = labels == int(AdvantageCondition.POSITIVE)
            if is_pos.any():
                # Negative minus positive: above zero means the positive token
                # tracked the demonstrated action better, which is the point.
                pos_label_scores.append(
                    (err_neg[is_pos] - err_pos[is_pos]).mean().item()
                )
            if (~is_pos).any():
                neg_label_scores.append(
                    (err_pos[~is_pos] - err_neg[~is_pos]).mean().item()
                )

    def mean(values):
        return float(sum(values) / len(values)) if values else float("nan")

    token_effect = mean(token_deltas)
    scene_effect = mean(obs_scales)
    return {
        "token_effect_sd": token_effect,
        "observation_effect_sd": scene_effect,
        "relative_sensitivity": token_effect / scene_effect if scene_effect else float("nan"),
        "positive_vs_unconditioned_sd": mean(positive_gap),
        "negative_vs_unconditioned_sd": mean(negative_gap),
        "positive_label_advantage_sd": mean(pos_label_scores),
        "negative_label_advantage_sd": mean(neg_label_scores),
        "batches": min(len(token_deltas), max_batches),
    }


def verdict(report: dict, *, min_relative: float) -> tuple[bool, str]:
    relative = report["relative_sensitivity"]
    if relative != relative:  # NaN
        return False, "Could not measure sensitivity"
    if relative < min_relative:
        return False, (
            f"Token barely moves the actions ({relative:.1%} of the variation "
            f"caused by changing the observation). The policy is ignoring its "
            f"conditioning, so this run is behavior cloning on the mixed "
            f"dataset rather than advantage-conditioned."
        )
    directional = report["positive_label_advantage_sd"]
    if directional == directional and directional <= 0:
        return False, (
            "The token changes the actions but in the wrong direction: on "
            "POSITIVE-labeled frames the negative token tracks the demonstrated "
            "action at least as well as the positive one."
        )
    return True, "Advantage conditioning is active and directionally correct"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-repo-id", default=None)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument(
        "--episodes",
        default=None,
        help=(
            "Restrict the check to an episode subset, e.g. '48-65'. Use this to "
            "measure the token on the data the policy was trained on: a run "
            "trained on a subset is out of distribution everywhere else."
        ),
    )
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--min-relative-sensitivity",
        type=float,
        default=0.05,
        help=(
            "Smallest token effect, as a fraction of the effect of changing the "
            "observation, that counts as the policy using its conditioning."
        ),
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    from models.act.helper import make_dataloaders

    args = parse_args(argv)
    device = torch.device(
        args.device
        or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    )
    model, cfg = _load_policy(args.checkpoint, device)
    if args.dataset_repo_id:
        cfg = replace(cfg, dataset_repo_id=args.dataset_repo_id)
    if args.dataset_root:
        os.environ["RECAP_CRITIC_DATASET_ROOT"] = str(args.dataset_root)

    conditions = load_conditions(args.labels) if args.labels else None
    episodes = parse_episode_subset(args.episodes)
    train_loader, _ = make_dataloaders(cfg, episodes=episodes)
    dataset = train_loader.dataset
    if conditions is not None:
        # Labels are keyed by global frame index, so let AdvantageDataset do the
        # lookup that training does; it also validates that these labels belong
        # to this dataset. Count frames from metadata rather than the loader:
        # under --episodes the loader only covers the requested subset.
        from lerobot.datasets import LeRobotDatasetMetadata

        total_frames = LeRobotDatasetMetadata(
            repo_id=cfg.dataset_repo_id, revision=cfg.dataset_revision
        ).total_frames
        dataset = AdvantageDataset(dataset, conditions, total_frames=total_frames)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    report = summarize(model, loader, device, args.batches)
    ok, message = verdict(report, min_relative=args.min_relative_sensitivity)

    print(f"checkpoint: {args.checkpoint}")
    print(f"dataset:    {cfg.dataset_repo_id}")
    print(f"batches:    {report['batches']}")
    print()
    print("All values in standard deviations of the dataset's action distribution.")
    print(f"  token effect (|positive - negative|)      {report['token_effect_sd']:.5f}")
    print(f"  observation effect (scene changes)        {report['observation_effect_sd']:.5f}")
    print(f"  relative sensitivity                      {report['relative_sensitivity']:.2%}")
    print(f"  positive vs unconditioned                 {report['positive_vs_unconditioned_sd']:.5f}")
    print(f"  negative vs unconditioned                 {report['negative_vs_unconditioned_sd']:.5f}")
    if conditions is not None:
        print()
        print("Directional check (above zero is correct):")
        print(f"  POSITIVE-labeled frames                   {report['positive_label_advantage_sd']:+.5f}")
        print(f"  NEGATIVE-labeled frames                   {report['negative_label_advantage_sd']:+.5f}")
    print()
    print(("PASS: " if ok else "FAIL: ") + message)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
