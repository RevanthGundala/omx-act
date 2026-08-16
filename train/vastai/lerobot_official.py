"""Launch official `lerobot-train` ACT training on Vast.ai.

This intentionally avoids the custom training loops in `models/act/*`.
It uploads this repo for dependency/config context, installs LeRobot, then runs
the upstream `lerobot-train` CLI with explicit ACT arguments.

Example:
    uv run python train/vastai/lerobot_official.py \
      --dataset-repo-id RevanthGundala/009-place-towel-in-bowl-10fps \
      --steps 10000 \
      --chunk-size 30
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.vastai import act as vast


PHOTOMETRIC_TRANSFORMS = {
    "brightness": {
        "weight": 1.0,
        "type": "ColorJitter",
        "kwargs": {"brightness": [0.8, 1.2]},
    },
    "contrast": {
        "weight": 1.0,
        "type": "ColorJitter",
        "kwargs": {"contrast": [0.8, 1.2]},
    },
    "saturation": {
        "weight": 1.0,
        "type": "ColorJitter",
        "kwargs": {"saturation": [0.5, 1.5]},
    },
    "hue": {
        "weight": 1.0,
        "type": "ColorJitter",
        "kwargs": {"hue": [-0.05, 0.05]},
    },
    "sharpness": {
        "weight": 1.0,
        "type": "SharpnessJitter",
        "kwargs": {"sharpness": [0.5, 1.5]},
    },
}


def resolve_train_episodes(dataset_repo_id: str, train_split: float, seed: int) -> list[int] | None:
    """Return the sorted list of episode indices to train on, or None to train on all episodes
    (i.e. omit --dataset.episodes entirely, preserving the original 100%-of-data behavior).

    Delegates the actual split algorithm to data.episode_split.resolve_train_holdout_episodes,
    the SAME function models/act/helper.py's make_dataloaders uses, so both trainers hold out
    identical episodes given the same (train_split, seed) -- true apples-to-apples comparison.
    """
    if train_split >= 1.0:
        return None
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    from models.act.episode_split import resolve_train_holdout_episodes

    n_episodes = LeRobotDatasetMetadata(dataset_repo_id).total_episodes
    train_episodes, holdout = resolve_train_holdout_episodes(n_episodes, train_split, seed)
    print(f"[train_lerobot_official] {n_episodes} episodes total, holding out {holdout} "
          f"({len(holdout)}/{n_episodes} = {100 * len(holdout) / n_episodes:.1f}%), "
          f"training on {len(train_episodes)} episodes.", flush=True)
    return train_episodes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch official LeRobot ACT training on Vast.ai.")
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--chunk-size", type=int, default=30)
    parser.add_argument("--n-action-steps", type=int, default=1)
    parser.add_argument("--temporal-ensemble-coeff", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--save-freq", type=int, default=None)
    parser.add_argument(
        "--image-transforms",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable LeRobot's default dataset image transforms (default: enabled). "
        "Use --no-image-transforms for an unaugmented run.",
    )
    parser.add_argument(
        "--photometric-only-image-transforms",
        action="store_true",
        help="Enable brightness, contrast, saturation, hue, and sharpness transforms "
        "while disabling geometric affine rotation/translation.",
    )
    parser.add_argument(
        "--train-split", type=float, default=1.0,
        help="Fraction of episodes to train on; the remainder is held out (never trained on) via "
        "--dataset.episodes, matching models/act/train.py's cfg.train_split. Default 1.0 preserves "
        "the original behavior (train on all episodes, no holdout). Use e.g. 0.95 to hold out ~5%% "
        "of episodes for a fair comparison against the custom trainer's 95/5 split. NOTE: this is "
        "an EPISODE-level holdout (whole episodes excluded), not a frame-level random_split like "
        "models/act/train.py currently uses -- episode-level avoids any risk of validation-frame "
        "leakage from the same episode appearing in training.",
    )
    parser.add_argument(
        "--split-seed", type=int, default=1000,
        help="Seed for the deterministic random episode holdout (only used when --train-split < 1.0). "
        "Matches configs/act/base.py's ACTConfig.seed default so the same seed value is used "
        "throughout the project, though episode-level and frame-level sampling will still pick a "
        "different actual holdout set.",
    )
    parser.add_argument("--resume-checkpoint", default=None, help="Unused; present for Vast launcher compatibility.")
    parser.add_argument("--output-dir", default="outputs/vastai")
    parser.add_argument("--remote-workspace", default=vast.REMOTE_WORKSPACE)
    parser.add_argument("--gpu-name", default=os.environ.get("OMX_GPU_NAME", "RTX_4090"))
    parser.add_argument("--min-gpu-ram-mb", type=int, default=20_000)
    parser.add_argument("--disk-gb", type=int, default=80)
    parser.add_argument("--boot-timeout", type=int, default=vast.BOOT_TIMEOUT)
    parser.add_argument("--image", default=os.environ.get("OMX_VAST_IMAGE", vast.DEFAULT_IMAGE))
    parser.add_argument("--instance-label", default="act-sim-lerobot-official-train")
    parser.add_argument("--exclude-host-id", action="append", default=[])
    parser.add_argument("--exclude-machine-id", action="append", default=[])
    parser.add_argument("--max-runtime-minutes", type=int, default=vast.DEFAULT_MAX_RUNTIME_MINUTES)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print the remote lerobot-train command and exit.")
    return parser.parse_args()


def remote_training_command(args: argparse.Namespace) -> str:
    from huggingface_hub import get_token

    if args.photometric_only_image_transforms and not args.image_transforms:
        raise ValueError(
            "--photometric-only-image-transforms cannot be combined with --no-image-transforms."
        )

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or get_token() or ""
    transform_suffix = "_photometric" if args.photometric_only_image_transforms else ""
    job_name = f"lerobot_official_act_chunk{args.chunk_size}_steps{args.steps}{transform_suffix}"
    output_dir = f"outputs/lerobot_official/{job_name}"
    train_episodes = resolve_train_episodes(args.dataset_repo_id, args.train_split, args.split_seed)
    # NOTE on quoting: --dataset.episodes=[...] contains spaces/commas/brackets, so shlex.quote()
    # wraps it in single quotes -- but the WHOLE remote command below is itself embedded inside an
    # outer single-quoted `bash -lc '...'` string. A single-quoted substring nested inside another
    # single-quoted string breaks: POSIX single quotes have no escaping, so the shell sees the
    # inner opening quote as closing the OUTER one early, silently truncating everything after it
    # (this exact bug produced a YAML parse error citing just "[0," -- the argument got cut off
    # mid-list). Fix: hand-wrap this one argument in DOUBLE quotes instead of shlex.quote(); double
    # quotes are safe to nest inside an outer single-quoted string (single quotes treat every
    # character, including embedded double quotes, completely literally). Tracked via a parallel
    # `already_quoted` set (by list position) rather than shlex.quote()-ing everything uniformly.
    command = [
        "lerobot-train",
        f"--dataset.repo_id={args.dataset_repo_id}",
        "--dataset.revision=main",
        f"--dataset.image_transforms.enable={str(args.image_transforms).lower()}",
        "--policy.type=act",
        "--policy.device=cuda",
        "--policy.push_to_hub=false",
        f"--policy.chunk_size={args.chunk_size}",
        f"--policy.n_action_steps={args.n_action_steps}",
        f"--policy.temporal_ensemble_coeff={args.temporal_ensemble_coeff}",
        "--policy.dim_model=512",
        "--policy.n_heads=8",
        "--policy.dim_feedforward=3200",
        "--policy.n_encoder_layers=4",
        "--policy.n_decoder_layers=1",
        "--policy.latent_dim=32",
        "--policy.n_vae_encoder_layers=4",
        "--policy.dropout=0.1",
        "--policy.kl_weight=10.0",
        "--optimizer.lr=1e-5",
        "--optimizer.weight_decay=1e-4",
        "--optimizer.grad_clip_norm=10.0",
        f"--steps={args.steps}",
        f"--batch_size={args.batch_size}",
        f"--save_freq={args.save_freq or args.steps}",
        "--save_checkpoint=true",
        "--eval_freq=1000",
        "--log_freq=100",
        "--wandb.enable=false",
        f"--output_dir={output_dir}",
        f"--job_name={job_name}",
    ]
    if args.photometric_only_image_transforms:
        transforms_json = json.dumps(PHOTOMETRIC_TRANSFORMS, separators=(",", ":"))
        command.insert(4, f"--dataset.image_transforms.tfs={transforms_json}")
    if train_episodes is not None:
        command.insert(3, f"--dataset.episodes={train_episodes}")
    already_quoted = {
        i
        for i, part in enumerate(command)
        if part.startswith("--dataset.episodes=")
        or part.startswith("--dataset.image_transforms.tfs=")
    }

    def render_part(i: int, part: str) -> str:
        if i not in already_quoted:
            return shlex.quote(part)
        escaped = (
            part.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("$", "\\$")
            .replace("`", "\\`")
        )
        return f'"{escaped}"'

    rendered = " \\\n+  ".join(render_part(i, part) for i, part in enumerate(command)).replace("\\\n+  ", "\\\n  ")
    return f"""bash -lc 'set -euo pipefail
cd {shlex.quote(args.remote_workspace)}
. /opt/venv/bin/activate
export PYTHONUNBUFFERED=1
export HF_TOKEN={json.dumps(hf_token)}
export HUGGINGFACE_HUB_TOKEN={json.dumps(hf_token)}
python -m pip install --no-cache-dir "lerobot[diffusion,dynamixel,training] @ git+https://github.com/huggingface/lerobot.git@{vast.LEROBOT_GIT_REF}"
python -m pip install --no-cache-dir --force-reinstall \
  torch==2.7.1 torchvision==0.22.1 torchcodec==0.5 \
  --index-url https://download.pytorch.org/whl/cu128
python - <<PY
import lerobot
import torch
import torchcodec
from importlib.metadata import version
print("LeRobot:", getattr(lerobot, "__version__", "unknown"))
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda, "available:", torch.cuda.is_available())
print("TorchCodec:", version("torchcodec"))
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable after installing the cu128 PyTorch build")
PY
{rendered} 2>&1 | tee {shlex.quote(vast.REMOTE_LOG)}
'"""


def main() -> None:
    args = parse_args()
    if args.dry_run:
        print(remote_training_command(args))
        return

    api_key = os.environ.get("VASTAI_API_KEY") or os.environ.get("VAST_API_KEY") or vast.read_local_vast_key()
    if not api_key:
        print("ERROR: VASTAI_API_KEY is not set.")
        sys.exit(1)

    # Monkey-patch only the command builder; reuse the proven Vast lifecycle.
    original_parse_args = vast.parse_args
    original_remote_training_command = vast.remote_training_command
    try:
        vast.parse_args = lambda: args
        vast.remote_training_command = remote_training_command
        vast.main()
    finally:
        vast.parse_args = original_parse_args
        vast.remote_training_command = original_remote_training_command


if __name__ == "__main__":
    main()
