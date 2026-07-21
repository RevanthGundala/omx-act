"""Resume official `lerobot-train` ACT training on Vast.ai from a local checkpoint.

The stock `train_lerobot_official_vastai.py` launcher always trains from step 0
(its `--resume-checkpoint` flag is a no-op). This launcher does a *true* resume:
it uploads a local lerobot-official run directory (containing
`checkpoints/last/{pretrained_model,training_state}`) to a fresh Vast instance
and runs `lerobot-train --resume=true`, continuing the optimizer/step state until
the new (larger) `--steps` target.

Example (continue the 10k sponge run to 20k):
    uv run python deploy/resume_lerobot_official_vastai.py \
      --local-run-dir outputs/vastai/instance-43132629/outputs/lerobot_official/lerobot_official_act_chunk30_steps10000 \
      --steps 20000
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deploy import train_act_vastai as vast


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume official LeRobot ACT training on Vast.ai.")
    parser.add_argument(
        "--local-run-dir",
        required=True,
        help="Local lerobot-official run dir containing checkpoints/last/{pretrained_model,training_state}.",
    )
    parser.add_argument("--steps", type=int, required=True, help="New TOTAL step target (must exceed checkpoint step).")
    parser.add_argument("--dataset-repo-id", default="", help="Informational only; dataset comes from checkpoint config.")
    parser.add_argument("--chunk-size", type=int, default=30)
    parser.add_argument("--save-freq", type=int, default=None)
    parser.add_argument("--output-dir", default="outputs/vastai")
    parser.add_argument("--remote-workspace", default=vast.REMOTE_WORKSPACE)
    parser.add_argument("--gpu-name", default=os.environ.get("OMX_GPU_NAME", "RTX_4090"))
    parser.add_argument("--min-gpu-ram-mb", type=int, default=20_000)
    parser.add_argument("--disk-gb", type=int, default=80)
    parser.add_argument("--boot-timeout", type=int, default=vast.BOOT_TIMEOUT)
    parser.add_argument("--image", default=os.environ.get("OMX_VAST_IMAGE", vast.DEFAULT_IMAGE))
    parser.add_argument("--instance-label", default="act-sim-lerobot-official-resume")
    parser.add_argument("--exclude-host-id", action="append", default=[])
    parser.add_argument("--exclude-machine-id", action="append", default=[])
    parser.add_argument("--max-runtime-minutes", type=int, default=vast.DEFAULT_MAX_RUNTIME_MINUTES)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print the remote resume command and exit.")
    args = parser.parse_args()
    # Fields the vast lifecycle reads that aren't exposed as flags here.
    args.runner = "lerobot"
    args.resume_checkpoint = None
    return args


def _remote_output_rel(args: argparse.Namespace) -> str:
    """Relative (to remote workspace) output dir, matching the local run dir name."""
    return f"outputs/lerobot_official/{Path(args.local_run_dir).name}"


def upload_resume_checkpoint(instance: dict, args: argparse.Namespace) -> str:
    """Rsync the local run dir up to the remote so lerobot-train can resume from it."""
    local = Path(args.local_run_dir).expanduser().resolve()
    last = local / "checkpoints" / "last" / "pretrained_model" / "train_config.json"
    if not last.exists():
        raise FileNotFoundError(f"Expected resumable checkpoint at {last}")
    remote_rel = _remote_output_rel(args)
    remote_abs = f"{args.remote_workspace}/{remote_rel}"
    remote_parent = str(Path(remote_abs).parent)
    vast.run_ssh(instance, f"mkdir -p {shlex.quote(remote_parent)}")
    print(f"Uploading resume run dir {local} -> {instance['ssh_host']}:{remote_abs}")
    subprocess.run(
        [
            "rsync",
            "-az",
            "--delete",
            "-e",
            vast.rsync_ssh(instance),
            f"{local}/",
            f"root@{instance['ssh_host']}:{remote_abs}/",
        ],
        check=True,
    )
    return remote_abs


def remote_training_command(args: argparse.Namespace) -> str:
    from huggingface_hub import get_token

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or get_token() or ""
    remote_rel = _remote_output_rel(args)
    config_path = f"{remote_rel}/checkpoints/last/pretrained_model/train_config.json"
    command = [
        "lerobot-train",
        f"--config_path={config_path}",
        "--resume=true",
        f"--steps={args.steps}",
    ]
    if args.save_freq is not None:
        command.append(f"--save_freq={args.save_freq}")
    rendered = " \\\n  ".join(shlex.quote(part) for part in command)
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


def download_outputs(instance: dict, args: argparse.Namespace, local_output_dir: Path) -> None:
    """Download only the target checkpoint, retrying before the instance can be destroyed."""
    if args.no_download:
        return

    remote_rel = _remote_output_rel(args)
    checkpoint_name = f"{args.steps:06d}"
    local_run = local_output_dir / remote_rel
    local_checkpoint = local_run / "checkpoints" / checkpoint_name
    local_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    local_output_dir.mkdir(parents=True, exist_ok=True)

    transfers = [
        (
            f"root@{instance['ssh_host']}:{args.remote_workspace}/{remote_rel}/"
            f"checkpoints/{checkpoint_name}/",
            f"{local_checkpoint}/",
        ),
        (
            f"root@{instance['ssh_host']}:{vast.REMOTE_LOG}",
            str(local_output_dir / "train.log"),
        ),
    ]
    for source, destination in transfers:
        for attempt in range(1, 4):
            result = subprocess.run(
                [
                    "rsync",
                    "-az",
                    "--partial",
                    "-e",
                    vast.rsync_ssh(instance),
                    source,
                    destination,
                ],
                check=False,
            )
            if result.returncode == 0:
                break
            if attempt == 3:
                raise RuntimeError(
                    f"Failed to download {source} after {attempt} attempts; "
                    f"instance {instance['id']} was retained."
                )
            print(f"Download attempt {attempt} failed; retrying in 15 seconds.", flush=True)
            time.sleep(15)


def main() -> None:
    args = parse_args()
    if args.dry_run:
        print(remote_training_command(args))
        return

    api_key = os.environ.get("VASTAI_API_KEY") or os.environ.get("VAST_API_KEY") or vast.read_local_vast_key()
    if not api_key:
        print("ERROR: VASTAI_API_KEY is not set.")
        sys.exit(1)

    original_parse_args = vast.parse_args
    original_remote_training_command = vast.remote_training_command
    original_upload_resume_checkpoint = vast.upload_resume_checkpoint
    original_download_outputs = vast.download_outputs
    try:
        vast.parse_args = lambda: args
        vast.remote_training_command = remote_training_command
        vast.upload_resume_checkpoint = upload_resume_checkpoint
        vast.download_outputs = download_outputs
        vast.main()
    finally:
        vast.parse_args = original_parse_args
        vast.remote_training_command = original_remote_training_command
        vast.upload_resume_checkpoint = original_upload_resume_checkpoint
        vast.download_outputs = original_download_outputs


if __name__ == "__main__":
    main()
