"""
Launch the local act-sim one-batch ACT debug run on a Vast.ai GPU.

Usage:
    export VASTAI_API_KEY="..."
    export HF_TOKEN="..."  # optional for private datasets
    uv run python deploy/train_act_vastai.py --steps 300

For faster, more reliable startup, build/push deploy/vastai.Dockerfile and run
with OMX_VAST_IMAGE set to that image. The launcher still bootstraps missing
pieces over SSH so the generic CUDA image remains usable as a fallback.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import sys
import time

from vastai_sdk import VastAI


POLL_INTERVAL = 15
BOOT_TIMEOUT = 480
SSH_TIMEOUT = 900
SEARCH_LIMIT = 20
SEARCH_ORDER = "dph+"
MAX_LAUNCH_ATTEMPTS = 8
MIN_RELIABILITY = 0.98
DEFAULT_MAX_RUNTIME_MINUTES = 180
REMOTE_WORKSPACE = "/workspace/act-sim"
REMOTE_LOG = "/workspace/act-sim/train.log"
DEFAULT_IMAGE = "nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04"
LEROBOT_GIT_REF = "87242cfced5228b6181fb0047021d684d424df38"
BAD_STATUSES = {"offline", "exited", "error", "dead"}
BAD_STATUS_MESSAGES = (
    "OCI runtime create failed",
    "failed to create task for container",
    "failed to inject CDI devices",
    "pull access denied",
    "manifest unknown",
    "unauthorized",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch act-sim ACT training on Vast.ai.")
    parser.add_argument("--steps", type=int, default=5_000, help="Number of normal dataset training steps.")
    parser.add_argument("--chunk-size", type=int, default=100, help="Training action chunk size.")
    parser.add_argument(
        "--save-freq",
        type=int,
        default=None,
        help="Checkpoint save frequency. Defaults to --steps.",
    )
    parser.add_argument(
        "--dataset-repo-id",
        default=os.environ.get("ACT_TRAIN_DATASET_REPO_ID"),
        help="Override the LeRobot dataset repo used for training.",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=None,
        help="Episode fraction used for training. Set to 1.0 to train on every episode.",
    )
    parser.add_argument(
        "--runner",
        choices=["custom", "lerobot", "diffusion"],
        default="custom",
        help="Which normal dataset trainer to execute.",
    )
    parser.add_argument(
        "--stock-parity",
        action="store_true",
        help="Train the custom model with stock LeRobot transformer semantics.",
    )
    parser.add_argument("--resume-checkpoint", default=None, help="Local checkpoint .pt to upload and resume from.")
    parser.add_argument("--gpu-name", default=os.environ.get("OMX_GPU_NAME", "RTX_4090"))
    parser.add_argument("--min-gpu-ram-mb", type=int, default=20_000)
    parser.add_argument(
        "--min-cuda-version",
        default=None,
        help="Require offer cuda_max_good to be at least this version (for example, 12.8).",
    )
    parser.add_argument("--disk-gb", type=int, default=80)
    parser.add_argument("--boot-timeout", type=int, default=BOOT_TIMEOUT)
    parser.add_argument("--image", default=os.environ.get("OMX_VAST_IMAGE", DEFAULT_IMAGE))
    parser.add_argument("--instance-label", default="act-sim-train")
    parser.add_argument("--job-name", default=None, help="Override the remote training job/output name.")
    parser.add_argument("--remote-workspace", default=REMOTE_WORKSPACE)
    parser.add_argument("--output-dir", default="outputs/vastai")
    parser.add_argument("--exclude-host-id", action="append", default=[], help="Vast host_id to skip. Repeatable.")
    parser.add_argument("--exclude-machine-id", action="append", default=[], help="Vast machine_id to skip. Repeatable.")
    parser.add_argument(
        "--allow-existing-instance-id",
        action="append",
        type=int,
        default=[],
        help="Explicitly acknowledged unrelated Vast instance ID. Repeatable.",
    )
    parser.add_argument(
        "--max-runtime-minutes",
        type=int,
        default=DEFAULT_MAX_RUNTIME_MINUTES,
        help="Hard billing-safety limit for an instance, including setup and downloads (default: 180).",
    )
    parser.add_argument("--no-download", action="store_true", help="Do not copy remote outputs back after completion.")
    parser.add_argument(
        "--final-checkpoint-only",
        action="store_true",
        help="Skip intermediate snapshots and download only the final checkpoint plus run metadata.",
    )
    return parser.parse_args()


def coerce_result_data(result):
    if isinstance(result, (dict, list)):
        return result
    if not isinstance(result, str):
        return str(result)
    try:
        return json.loads(result)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(result)
        except (ValueError, SyntaxError):
            return result


def parse_instance_id(result) -> int | None:
    result_data = coerce_result_data(result)
    if isinstance(result_data, dict):
        for key in ("new_contract", "instance_id", "contract_id", "id"):
            value = result_data.get(key)
            if isinstance(value, int):
                return value
    return None


def sanitize_error(error: object) -> str:
    api_key = os.environ.get("VASTAI_API_KEY") or os.environ.get("VAST_API_KEY") or ""
    message = re.sub(r"api_key=[^&\s]+", "api_key=***", str(error))
    return message.replace(api_key, "***") if api_key else message


def read_local_vast_key() -> str | None:
    key_path = repo_root() / ".secrets" / "vast_key"
    if not key_path.exists():
        return None
    key = key_path.read_text().strip()
    return key or None


def format_offer(offer: dict) -> str:
    price = offer.get("dph_total", offer.get("dph_base", "?"))
    reliability = offer.get("reliability2", offer.get("reliability", "?"))
    return (
        f"offer={offer.get('id')} host={offer.get('host_id')} machine={offer.get('machine_id')} "
        f"gpu={offer.get('gpu_name')} price=${price}/hr reliability={reliability} "
        f"location={offer.get('geolocation', 'unknown')}"
    )


def _version_tuple(value: object) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in str(value).split("."))
    except ValueError:
        return ()


def search_candidate_offers(vast: VastAI, args: argparse.Namespace) -> list[dict]:
    query = f"gpu_name={args.gpu_name} rentable=True rented=False"
    offers = coerce_result_data(vast.search_offers(query=query, limit=SEARCH_LIMIT, order=SEARCH_ORDER))
    if not isinstance(offers, list):
        raise RuntimeError(f"Unexpected search_offers result: {offers}")

    filtered = []
    excluded_hosts = {str(value) for value in args.exclude_host_id}
    excluded_machines = {str(value) for value in args.exclude_machine_id}
    for offer in offers:
        if not isinstance(offer, dict):
            continue
        if str(offer.get("host_id")) in excluded_hosts:
            continue
        if str(offer.get("machine_id")) in excluded_machines:
            continue
        if int(offer.get("num_gpus", 0)) != 1:
            continue
        if float(offer.get("disk_space", 0)) < args.disk_gb:
            continue
        if float(offer.get("gpu_ram", 0)) < args.min_gpu_ram_mb:
            continue
        if offer.get("verification") != "verified":
            continue
        if offer.get("is_vm_deverified"):
            continue
        reliability = float(offer.get("reliability2", offer.get("reliability", 0.0)) or 0.0)
        if reliability < MIN_RELIABILITY:
            continue
        min_cuda_version = getattr(args, "min_cuda_version", None)
        if min_cuda_version and _version_tuple(offer.get("cuda_max_good")) < _version_tuple(min_cuda_version):
            continue
        filtered.append(offer)
    return filtered


def setup_onstart_script() -> str:
    return """#!/bin/bash
set -euo pipefail
mkdir -p /workspace
echo "=== ACT-SIM CONTAINER READY ==="
tail -f /dev/null
"""


def launch_offer(vast: VastAI, offer: dict, args: argparse.Namespace):
    print(f"Trying {format_offer(offer)}")
    return vast.create_instance(
        id=offer["id"],
        image=args.image,
        disk=args.disk_gb,
        onstart_cmd=setup_onstart_script(),
        runtype="ssh_proxy",
        label=args.instance_label,
        cancel_unavail=True,
    )


def attach_ssh_key(vast: VastAI, instance_id: int) -> None:
    ssh_key_path = Path.home() / ".ssh" / "id_ed25519.pub"
    if not ssh_key_path.exists():
        ssh_key_path = Path.home() / ".ssh" / "id_rsa.pub"
    if not ssh_key_path.exists():
        raise FileNotFoundError("No SSH public key found at ~/.ssh/id_ed25519.pub or ~/.ssh/id_rsa.pub")
    vast.attach_ssh(instance_id=instance_id, ssh_key=ssh_key_path.read_text().strip())
    print("SSH key attached")


def wait_for_instance_running(vast: VastAI, instance_id: int, timeout: int) -> dict:
    print("Waiting for instance to boot...")
    start = time.time()
    while time.time() - start < timeout:
        data = coerce_result_data(vast.show_instance(id=instance_id))
        if not isinstance(data, dict):
            raise RuntimeError(f"Unexpected show_instance result: {data}")
        status = data.get("actual_status", data.get("status_msg", "unknown"))
        status_msg = str(data.get("status_msg", ""))
        print(f"  status={status} elapsed={int(time.time() - start)}s")
        if any(message in status_msg for message in BAD_STATUS_MESSAGES):
            raise RuntimeError(f"Instance startup failed: {status_msg}")
        if status == "running":
            if not data.get("ssh_host") or not data.get("ssh_port"):
                raise RuntimeError("Instance is running but does not report ssh_host/ssh_port.")
            return data
        if status in BAD_STATUSES:
            raise RuntimeError(f"Instance entered bad status: {status}")
        time.sleep(POLL_INTERVAL)
    raise TimeoutError(f"Instance not running after {timeout}s")


def ssh_base(instance: dict) -> list[str]:
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=20",
        "-p",
        str(instance["ssh_port"]),
        f"root@{instance['ssh_host']}",
    ]


def rsync_ssh(instance: dict) -> str:
    return f"ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p {int(instance['ssh_port'])}"


def run_ssh(instance: dict, command: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*ssh_base(instance), command],
        cwd=repo_root(),
        text=True,
        check=check,
    )


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def wait_for_ssh(instance: dict) -> None:
    print("Waiting for SSH...")
    start = time.time()
    last_error = None
    while time.time() - start < SSH_TIMEOUT:
        try:
            run_ssh(instance, "echo SSH_OK")
            return
        except subprocess.CalledProcessError as exc:
            last_error = exc
            time.sleep(POLL_INTERVAL)
    raise TimeoutError(f"SSH not ready after {SSH_TIMEOUT}s: {sanitize_error(last_error)}")


def bootstrap_remote(instance: dict) -> None:
    print("Checking remote Python/CUDA readiness...")
    readiness_command = r"""bash -lc 'set -euo pipefail
test -x /opt/venv/bin/python
/opt/venv/bin/python - <<PY
import torch
print("Torch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available.")
print("Detected GPU:", torch.cuda.get_device_name(0))
PY
'"""
    try:
        run_ssh(instance, readiness_command)
        return
    except subprocess.CalledProcessError:
        print("Prebuilt image readiness check failed; falling back to remote bootstrap install.")

    command = r"""bash -lc 'set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
echo "Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq python3 python3-dev python3-pip python3-venv git ffmpeg linux-libc-dev build-essential clang rsync
if [[ ! -d /opt/venv ]]; then
  echo "Creating /opt/venv..."
  python3 -m venv /opt/venv
fi
. /opt/venv/bin/activate
echo "Checking CUDA Torch..."
python -m pip install --no-cache-dir --upgrade pip
if ! python - <<PY
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
then
  echo "Installing CUDA Torch..."
  python -m pip install --no-cache-dir --force-reinstall \
    "torch==2.7.1" "torchvision==0.22.1" \
    --index-url https://download.pytorch.org/whl/cu128
fi
python - <<PY
import torch
print("Torch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available.")
print("Detected GPU:", torch.cuda.get_device_name(0))
PY
'"""
    run_ssh(instance, command)


def upload_repo(instance: dict, remote_workspace: str) -> None:
    root = repo_root()
    print(f"Uploading local repo {root} -> {instance['ssh_host']}:{remote_workspace}")
    run_ssh(instance, f"rm -rf {shlex.quote(remote_workspace)} && mkdir -p {shlex.quote(remote_workspace)}")
    subprocess.run(
        [
            "rsync",
            "-az",
            "--delete",
            "--exclude",
            ".git/",
            "--exclude",
            ".venv/",
            "--exclude",
            "__pycache__/",
            "--exclude",
            ".ruff_cache/",
            "--exclude",
            ".pytest_cache/",
            "--exclude",
            "outputs/",
            "-e",
            rsync_ssh(instance),
            f"{root}/",
            f"root@{instance['ssh_host']}:{remote_workspace}/",
        ],
        check=True,
    )


def upload_resume_checkpoint(instance: dict, args: argparse.Namespace) -> str:
    if not args.resume_checkpoint:
        return ""
    checkpoint = Path(args.resume_checkpoint).expanduser()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint}")
    remote_path = f"{args.remote_workspace}/.resume/{checkpoint.name}"
    run_ssh(instance, f"mkdir -p {shlex.quote(args.remote_workspace + '/.resume')}")
    print(f"Uploading resume checkpoint {checkpoint} -> {remote_path}")
    subprocess.run(
        [
            "rsync",
            "-az",
            "-e",
            rsync_ssh(instance),
            str(checkpoint),
            f"root@{instance['ssh_host']}:{remote_path}",
        ],
        check=True,
    )
    return remote_path


def remote_training_command(args: argparse.Namespace) -> str:
    from huggingface_hub import get_token

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or get_token() or ""
    module = {
        "custom": "models.act.train",
        "lerobot": "models.act.train_lerobot",
        "diffusion": "models.act.train_diffusion",
    }[args.runner]
    parity_suffix = "_stock_parity" if args.runner == "custom" and getattr(args, "stock_parity", False) else ""
    job_name = args.job_name or (
        f"act_{args.runner}_train_chunk{args.chunk_size}_steps{args.steps}{parity_suffix}"
    )
    return f"""bash -lc 'set -euo pipefail
cd {shlex.quote(args.remote_workspace)}
. /opt/venv/bin/activate
export PYTHONUNBUFFERED=1
export HF_TOKEN={json.dumps(hf_token)}
export HUGGINGFACE_HUB_TOKEN={json.dumps(hf_token)}
export ACT_DEVICE=cuda
export ACT_DISABLE_MLFLOW=1
export ACT_TRAIN_STEPS={int(args.steps)}
export ACT_TRAIN_CHUNK_SIZE={json.dumps(str(args.chunk_size))}
export ACT_TRAIN_SAVE_FREQ={int(args.save_freq or args.steps)}
export ACT_TRAIN_JOB_NAME={json.dumps(job_name)}
export ACT_TRAIN_DATASET_REPO_ID={json.dumps(args.dataset_repo_id or "")}
export ACT_TRAIN_SPLIT={json.dumps("" if args.train_split is None else str(args.train_split))}
export ACT_TRAIN_STOCK_PARITY={"1" if getattr(args, "stock_parity", False) else "0"}
export ACT_TRAIN_RESUME_CHECKPOINT={json.dumps(getattr(args, "remote_resume_checkpoint", "") or "")}
python -m pip install --no-cache-dir -e .
python -m pip install --no-cache-dir --no-deps --force-reinstall git+https://github.com/huggingface/lerobot.git@{LEROBOT_GIT_REF}
python -m pip install --no-cache-dir "termcolor>=2.4.0,<4.0.0"
python - <<PY
import lerobot
import torch
import torchvision
print("LeRobot:", getattr(lerobot, "__version__", "unknown"))
print("Torch:", torch.__version__, "Torchvision:", torchvision.__version__)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable after final dependency installation.")
PY
python -m {module} 2>&1 | tee {shlex.quote(REMOTE_LOG)}
'"""


def run_remote_training(instance: dict, args: argparse.Namespace, local_output_dir: Path) -> None:
    print("Starting remote training...")
    live_log = local_output_dir / "live.log"
    with live_log.open("w") as log_file:
        process = subprocess.Popen(
            [*ssh_base(instance), remote_training_command(args)],
            cwd=repo_root(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(f"[remote] {line}", end="")
            log_file.write(line)
            log_file.flush()
            if "Saved checkpoint:" in line:
                args.remote_latest_checkpoint = line.split("Saved checkpoint:", 1)[1].strip()
                if not args.final_checkpoint_only:
                    print("Checkpoint saved remotely; syncing outputs snapshot...", flush=True)
                    download_outputs(instance, args, local_output_dir)
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Remote training failed with exit code {return_code}; see {live_log}")


def _validated_checkpoint_path(value: str) -> PurePosixPath:
    checkpoint = PurePosixPath(value)
    if (
        not value
        or checkpoint.is_absolute()
        or not checkpoint.parts
        or checkpoint.parts[0] != "outputs"
        or ".." in checkpoint.parts
        or checkpoint.suffix != ".pt"
    ):
        raise ValueError(f"Unsafe or unexpected remote checkpoint path: {value!r}")
    return checkpoint


def _rsync_with_retries(command: list[str], *, attempts: int = 3) -> None:
    last_return_code = None
    for attempt in range(1, attempts + 1):
        result = subprocess.run(command, check=False)
        if result.returncode == 0:
            return
        last_return_code = result.returncode
        print(f"rsync attempt {attempt}/{attempts} failed with exit code {result.returncode}.", flush=True)
        if attempt < attempts:
            time.sleep(5 * attempt)
    raise RuntimeError(f"rsync failed after {attempts} attempts (exit code {last_return_code}).")


def download_final_checkpoint(instance: dict, args: argparse.Namespace, local_output_dir: Path) -> None:
    checkpoint = _validated_checkpoint_path(getattr(args, "remote_latest_checkpoint", ""))
    run_dir = checkpoint.parent.parent
    local_run_dir = local_output_dir.joinpath(*run_dir.parts)
    local_checkpoint_dir = local_run_dir / "checkpoints"
    local_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for remote_path, local_path in (
        (run_dir / "config.json", local_run_dir / "config.json"),
        (run_dir / "metrics.jsonl", local_run_dir / "metrics.jsonl"),
        (checkpoint, local_checkpoint_dir / checkpoint.name),
    ):
        local_path.parent.mkdir(parents=True, exist_ok=True)
        _rsync_with_retries(
            [
                "rsync",
                "-az",
                "--partial",
                "-e",
                rsync_ssh(instance),
                f"root@{instance['ssh_host']}:{args.remote_workspace}/{remote_path}",
                str(local_path),
            ]
        )

    remote_checkpoint = f"{args.remote_workspace}/{checkpoint}"
    remote_hash = subprocess.run(
        [*ssh_base(instance), f"sha256sum {shlex.quote(remote_checkpoint)}"],
        cwd=repo_root(),
        text=True,
        capture_output=True,
        check=True,
    ).stdout.split()[0]
    local_hash = subprocess.run(
        ["shasum", "-a", "256", str(local_checkpoint_dir / checkpoint.name)],
        cwd=repo_root(),
        text=True,
        capture_output=True,
        check=True,
    ).stdout.split()[0]
    if remote_hash != local_hash:
        raise RuntimeError(
            f"Final checkpoint hash mismatch: remote={remote_hash}, local={local_hash}."
        )
    print(f"Verified final checkpoint SHA-256: {local_hash}", flush=True)


def download_outputs(instance: dict, args: argparse.Namespace, local_output_dir: Path) -> None:
    if args.no_download:
        return
    local_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading remote logs and outputs to {local_output_dir}")
    _rsync_with_retries(
        [
            "rsync",
            "-az",
            "-e",
            rsync_ssh(instance),
            f"root@{instance['ssh_host']}:{REMOTE_LOG}",
            str(local_output_dir / "train.log"),
        ]
    )
    if args.final_checkpoint_only:
        download_final_checkpoint(instance, args, local_output_dir)
        return
    subprocess.run(
        [
            "rsync",
            "-az",
            "-e",
            rsync_ssh(instance),
            f"root@{instance['ssh_host']}:{args.remote_workspace}/outputs/",
            str(local_output_dir / "outputs/"),
        ],
        check=False,
    )


def destroy_instance(vast: VastAI, instance_id: int) -> None:
    print(f"Destroying instance {instance_id}...")
    last_error = None
    for attempt in range(1, 4):
        try:
            vast.destroy_instance(id=instance_id)
            print(f"Destroyed instance {instance_id}")
            return
        except Exception as exc:
            last_error = exc
            print(f"Destroy attempt {attempt}/3 failed: {sanitize_error(exc)}")
            time.sleep(5)
    raise RuntimeError(
        f"CRITICAL: failed to destroy billable instance {instance_id}: {sanitize_error(last_error)}"
    )


def assert_no_managed_instances(vast: VastAI, allowed_instance_ids: list[int] | None = None) -> None:
    instances = coerce_result_data(vast.show_instances())
    if not isinstance(instances, list):
        raise RuntimeError(f"Unable to audit existing Vast instances: {instances}")
    allowed_ids = set(allowed_instance_ids or [])
    managed = [
        instance
        for instance in instances
        if str(instance.get("label", "")).startswith("act-sim")
        and int(instance.get("id", -1)) not in allowed_ids
    ]
    if managed:
        details = ", ".join(
            f"{instance.get('id')} ({instance.get('label')}, {instance.get('actual_status')})"
            for instance in managed
        )
        raise RuntimeError(
            f"Refusing to launch while existing act-sim Vast instances are active: {details}"
        )


def start_instance_watchdog(instance_id: int, max_runtime_minutes: int) -> subprocess.Popen:
    if max_runtime_minutes <= 0:
        raise ValueError("--max-runtime-minutes must be positive.")
    watchdog = repo_root() / "deploy" / "vastai_instance_watchdog.py"
    log_dir = repo_root() / "outputs" / "vastai"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = (log_dir / f"watchdog-{instance_id}.log").open("a")
    return subprocess.Popen(
        [
            sys.executable,
            str(watchdog),
            "--instance-id",
            str(instance_id),
            "--max-runtime-minutes",
            str(max_runtime_minutes),
        ],
        cwd=repo_root(),
        stdin=subprocess.DEVNULL,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def main() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    args = parse_args()
    if args.train_split is not None and not 0 < args.train_split <= 1:
        raise ValueError("--train-split must be in (0, 1].")
    api_key = os.environ.get("VASTAI_API_KEY") or os.environ.get("VAST_API_KEY") or read_local_vast_key()
    if not api_key:
        print("ERROR: VASTAI_API_KEY is not set.")
        sys.exit(1)

    vast = VastAI(api_key=api_key)
    assert_no_managed_instances(vast, args.allow_existing_instance_id)
    print(f"Searching for {args.gpu_name} offers...")
    candidates = search_candidate_offers(vast, args)
    if not candidates:
        raise RuntimeError(f"No viable {args.gpu_name} offers found.")

    instance_id = None
    instance = None
    watchdog = None
    local_output_dir = None
    try:
        last_error = None
        for attempt, offer in enumerate(candidates[:MAX_LAUNCH_ATTEMPTS], start=1):
            try:
                print(f"Launch attempt {attempt}/{min(len(candidates), MAX_LAUNCH_ATTEMPTS)}")
                result = launch_offer(vast, offer, args)
                instance_id = parse_instance_id(result)
                if not instance_id:
                    raise RuntimeError(f"Failed to parse instance ID from: {result}")
                print(f"Instance {instance_id} launched")
                watchdog = start_instance_watchdog(instance_id, args.max_runtime_minutes)
                print(
                    f"Billing watchdog armed for {args.max_runtime_minutes} minutes "
                    f"(PID {watchdog.pid})."
                )
                attach_ssh_key(vast, instance_id)
                instance = wait_for_instance_running(vast, instance_id, args.boot_timeout)
                break
            except Exception as exc:
                last_error = exc
                print(f"Launch failed for {format_offer(offer)}: {sanitize_error(exc)}")
                if instance_id is not None:
                    destroy_instance(vast, instance_id)
                    if watchdog is not None:
                        watchdog.terminate()
                        watchdog.wait(timeout=10)
                        watchdog = None
                    instance_id = None
                    instance = None
        else:
            raise RuntimeError(f"Unable to launch a healthy instance: {sanitize_error(last_error)}")

        assert instance_id is not None and instance is not None
        local_output_dir = repo_root() / args.output_dir / f"instance-{instance_id}"
        local_output_dir.mkdir(parents=True, exist_ok=True)
        wait_for_ssh(instance)
        bootstrap_remote(instance)
        upload_repo(instance, args.remote_workspace)
        args.remote_resume_checkpoint = upload_resume_checkpoint(instance, args)
        args.training_started = True
        run_remote_training(instance, args, local_output_dir)
    finally:
        active_error = sys.exc_info()[1]
        download_error = None
        if instance is not None and local_output_dir is not None and getattr(args, "training_started", False):
            try:
                download_outputs(instance, args, local_output_dir)
            except Exception as exc:
                download_error = exc
                print(f"Final output download failed: {sanitize_error(exc)}", flush=True)
        destroyed = False
        if instance_id is not None:
            destroy_instance(vast, instance_id)
            destroyed = True
        if watchdog is not None and destroyed:
            watchdog.terminate()
            watchdog.wait(timeout=10)
        if download_error is not None and active_error is None:
            raise RuntimeError("Training completed but final output download failed.") from download_error


if __name__ == "__main__":
    main()
