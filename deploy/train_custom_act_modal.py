"""
Train the custom models/act ACT implementation on Modal.

Usage:
    modal run --detach deploy/train_custom_act_modal.py::train_remote --profile pour_water
    modal run deploy/train_custom_act_modal.py --profile pour_water --preflight-only
    modal run deploy/train_custom_act_modal.py --profile sim_transfer_cube_reference \
      --benchmark-dataset-dir data/benchmarks/act_sim_transfer_cube_scripted --upload-benchmark-data
    modal run deploy/train_custom_act_modal.py --profile sim_transfer_cube_reference --run-name seed0

Download outputs:
    modal volume get omx-custom-act-training-logs / outputs/modal_custom_act
"""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
import importlib
from pathlib import Path
import uuid

import modal

try:
    from modal_training_common import (
        validate_dataset_run,
        validate_pretrained_checkpoint_path,
        validate_volume_relative_path,
    )
except ModuleNotFoundError:
    from deploy.modal_training_common import (
        validate_dataset_run,
        validate_pretrained_checkpoint_path,
        validate_volume_relative_path,
    )


REMOTE_WORKSPACE = Path("/workspace")
REMOTE_OUTPUTS = Path("/outputs")
REMOTE_BENCHMARK_DATA = Path("/benchmark-data")
MODAL_GPU = "A10G"
MODAL_MAX_RUNTIME_SECONDS = 3 * 60 * 60
VOLUME_COMMIT_INTERVAL_SECONDS = 300
LEROBOT_GIT_REF = "87242cfced5228b6181fb0047021d684d424df38"

hf_secret = modal.Secret.from_name("huggingface")

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04", add_python="3.12")
    .apt_install("git", "ffmpeg", "linux-libc-dev", "clang")
    .pip_install(
        "torch==2.7.1",
        "torchvision==0.22.1",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "huggingface-hub>=1.0,<2.0",
        f"lerobot[training] @ git+https://github.com/huggingface/lerobot.git@{LEROBOT_GIT_REF}",
        "av>=15,<16",
        "h5py>=3.15",
        "mlflow>=2.17",
        "pandas",
        "pyarrow",
        "safetensors",
        "torchcodec==0.5.0",
    )
    .add_local_dir("models", remote_path=str(REMOTE_WORKSPACE / "models"))
    .add_local_dir("configs", remote_path=str(REMOTE_WORKSPACE / "configs"))
    .add_local_dir("utils", remote_path=str(REMOTE_WORKSPACE / "utils"))
    .add_local_dir("data", remote_path=str(REMOTE_WORKSPACE / "data"))
    .add_local_file("deploy/modal_training_common.py", remote_path="/root/modal_training_common.py")
)
if Path("benchmarks").is_dir():
    image = image.add_local_dir("benchmarks", remote_path=str(REMOTE_WORKSPACE / "benchmarks"))

app = modal.App("omx-custom-act-training", image=image)
vol = modal.Volume.from_name("omx-custom-act-training-logs", create_if_missing=True)
benchmark_vol = modal.Volume.from_name("omx-custom-act-benchmark-data", create_if_missing=True)


@app.function(timeout=300)
def runtime_preflight() -> str:
    import subprocess

    import lerobot
    import torch
    import torchcodec
    import torchvision
    from torchcodec.decoders import VideoDecoder

    video_path = "/tmp/torchcodec_preflight.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x64:duration=1:rate=2",
            "-y",
            video_path,
        ],
        check=True,
    )
    frame = VideoDecoder(video_path)[0]
    return (
        f"torch={torch.__version__} torchvision={torchvision.__version__} "
        f"torchcodec={torchcodec.__version__} lerobot={lerobot.__version__} "
        f"decoded_frame_shape={tuple(frame.shape)}"
    )


def load_profile(profile: str):
    module = importlib.import_module(f"configs.act.{profile}")
    return module.config


def preflight_profile(profile: str) -> None:
    config = load_profile(profile)
    print(f"PROFILE {profile}")
    print(f"  modal_gpu={MODAL_GPU}")
    print(f"  dataset_format={config.dataset_format}")
    if config.dataset_format == "act_hdf5":
        print(f"  benchmark_task={config.benchmark_task_name}")
        print(f"  benchmark_dataset_dir={config.benchmark_dataset_dir}")
        print(f"  benchmark_num_episodes={config.benchmark_num_episodes}")
        print(f"  benchmark_episode_len={config.benchmark_episode_len}")
    else:
        print(f"  dataset={config.dataset_repo_id}@{config.dataset_revision}")
    print(f"  cameras={config.camera_names}")
    print(f"  batch_size={config.batch_size}")
    print(f"  chunk_size={config.chunk_size}")
    print(f"  d_model={config.d_model}")
    print(f"  encoder_layers={config.num_encoder_layers}")
    print(f"  decoder_layers={config.num_decoder_layers}")
    print(f"  lr={config.learning_rate}")
    print(f"  kl_weight={config.kl_weight}")
    print(f"  steps={config.num_train_steps}")

    if config.dataset_format == "act_hdf5":
        print("  dataset_check=skipped; benchmark HDF5 data is staged in Modal volume")
    else:
        try:
            from huggingface_hub import HfApi
        except ModuleNotFoundError:
            print("  dataset_check=skipped; install/use uv env for local huggingface_hub preflight")
        else:
            HfApi().dataset_info(config.dataset_repo_id, revision=config.dataset_revision)
            print("  dataset_check=ok")


def benchmark_remote_dir(profile: str) -> Path:
    return REMOTE_BENCHMARK_DATA / profile


def upload_benchmark_dataset(profile: str, local_dataset_dir: str) -> None:
    local_path = Path(local_dataset_dir).expanduser()
    if not local_path.is_dir():
        raise FileNotFoundError(f"Benchmark dataset dir not found: {local_path}")
    episode_files = sorted(local_path.glob("episode_*.hdf5"))
    if not episode_files:
        raise FileNotFoundError(f"No episode_*.hdf5 files found in {local_path}")

    remote_dir = f"/{profile}"
    print(f"Uploading {len(episode_files)} benchmark episodes to Modal volume {remote_dir}...")
    with benchmark_vol.batch_upload(force=True) as batch:
        batch.put_directory(local_path, remote_dir)
    print("Benchmark upload complete.")


@app.function(volumes={REMOTE_BENCHMARK_DATA: benchmark_vol})
def list_benchmark_data(profile: str) -> list[str]:
    import os

    benchmark_vol.reload()
    root_dir = benchmark_remote_dir(profile)
    results = []
    if not root_dir.exists():
        return results
    for root, _dirs, files in os.walk(root_dir):
        for filename in files:
            full = os.path.join(root, filename)
            results.append(os.path.relpath(full, root_dir))
    return sorted(results)


def configure_remote_profile(config, profile: str):
    remote_output_root = REMOTE_OUTPUTS / "act_experiments"
    updated = replace(config, output_root=str(remote_output_root))
    if updated.dataset_format == "act_hdf5":
        updated = replace(updated, benchmark_dataset_dir=str(benchmark_remote_dir(profile)))
    return updated


@app.function(
    gpu=MODAL_GPU,
    cpu=4.0,
    timeout=MODAL_MAX_RUNTIME_SECONDS,
    memory=32_768,
    volumes={REMOTE_OUTPUTS: vol},
    secrets=[hf_secret],
)
def train_dataset_remote(
    dataset_repo_id: str,
    steps: int,
    chunk_size: int,
    save_freq: int,
    train_split: float,
    job_name: str,
    run_token: str,
    pretrained_checkpoint: str | None,
    resume_checkpoint: str | None,
    image_transforms: bool,
) -> str:
    import json
    import os
    import subprocess
    import sys
    import threading
    import time

    validate_dataset_run(
        dataset_repo_id=dataset_repo_id,
        steps=steps,
        chunk_size=chunk_size,
        save_freq=save_freq,
        train_split=train_split,
        job_name=job_name,
    )
    pretrained_checkpoint = validate_pretrained_checkpoint_path(pretrained_checkpoint)
    resume_checkpoint = validate_pretrained_checkpoint_path(resume_checkpoint)
    if pretrained_checkpoint and resume_checkpoint:
        raise ValueError(
            "pretrained_checkpoint and resume_checkpoint are mutually exclusive."
        )
    vol.reload()
    pretrained_path = (
        REMOTE_OUTPUTS / pretrained_checkpoint
        if pretrained_checkpoint is not None
        else None
    )
    resume_path = (
        REMOTE_OUTPUTS / resume_checkpoint
        if resume_checkpoint is not None
        else None
    )
    if pretrained_path is not None and not pretrained_path.is_file():
        raise FileNotFoundError(
            f"Warm-start checkpoint is missing from Modal volume: {pretrained_path}"
        )
    if resume_path is not None and not resume_path.is_file():
        raise FileNotFoundError(
            f"Resume checkpoint is missing from Modal volume: {resume_path}"
        )
    os.environ.update(
        {
            "PYTHONUNBUFFERED": "1",
            "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
            "HUGGINGFACE_HUB_TOKEN": os.environ.get("HF_TOKEN", ""),
            "ACT_DEVICE": "cuda",
            "ACT_DISABLE_MLFLOW": "1",
            "ACT_TRAIN_DATASET_REPO_ID": dataset_repo_id,
            "ACT_TRAIN_STEPS": str(steps),
            "ACT_TRAIN_CHUNK_SIZE": str(chunk_size),
            "ACT_TRAIN_SAVE_FREQ": str(save_freq),
            "ACT_TRAIN_SPLIT": str(train_split),
            "ACT_TRAIN_IMAGE_TRANSFORMS": str(image_transforms).lower(),
            "ACT_TRAIN_JOB_NAME": job_name,
            "ACT_TRAIN_OUTPUT_ROOT": str(REMOTE_OUTPUTS / "act_experiments"),
            "ACT_TRAIN_STATUS_PATH": str(
                REMOTE_OUTPUTS / "modal_monitors" / f"{run_token}_training.json"
            ),
            "ACT_TRAIN_RESUME_CHECKPOINT": (
                str(resume_path) if resume_path is not None else ""
            ),
            "ACT_TRAIN_PRETRAINED_CHECKPOINT": (
                str(pretrained_path) if pretrained_path is not None else ""
            ),
        }
    )
    sys.path.insert(0, str(REMOTE_WORKSPACE))
    os.chdir(REMOTE_WORKSPACE)

    import torch

    print("=== CUDA PREFLIGHT ===", flush=True)
    print(f"run_token={run_token}", flush=True)
    print(f"torch={torch.__version__} cuda_runtime={torch.version.cuda}", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; refusing CPU fallback on a paid GPU job.")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"gpu={gpu_name} vram_gb={gpu_memory_gb:.1f}", flush=True)
    print(
        "training_config "
        f"dataset={dataset_repo_id} steps={steps} chunk_size={chunk_size} "
        f"save_freq={save_freq} train_split={train_split} job_name={job_name} "
        f"image_transforms={image_transforms} "
        f"pretrained_checkpoint={pretrained_path} resume_checkpoint={resume_path}",
        flush=True,
    )

    monitor_dir = REMOTE_OUTPUTS / "modal_monitors"
    monitor_dir.mkdir(parents=True, exist_ok=True)
    telemetry_path = monitor_dir / f"{run_token}.jsonl"
    stop_monitor = threading.Event()

    def sample_gpu() -> None:
        while not stop_monitor.is_set():
            timestamp = datetime.now(timezone.utc).isoformat()
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                fields = [value.strip() for value in result.stdout.strip().split(",")]
                entry = {
                    "timestamp": timestamp,
                    "gpu_name": fields[0],
                    "utilization_percent": float(fields[1]),
                    "memory_used_mib": float(fields[2]),
                    "memory_total_mib": float(fields[3]),
                    "temperature_c": float(fields[4]),
                }
            except Exception as exc:
                entry = {"timestamp": timestamp, "telemetry_error": repr(exc)}
            with telemetry_path.open("a") as file:
                file.write(json.dumps(entry, sort_keys=True) + "\n")
            print(f"GPU_TELEMETRY {json.dumps(entry, sort_keys=True)}", flush=True)
            vol.commit()
            stop_monitor.wait(VOLUME_COMMIT_INTERVAL_SECONDS)

    monitor_thread = threading.Thread(target=sample_gpu, name="gpu-telemetry", daemon=True)
    monitor_thread.start()
    try:
        from models.act.train import main as train_main

        train_main()
    finally:
        stop_monitor.set()
        monitor_thread.join(timeout=10)
        vol.commit()
    return json.dumps(
        {
            "status": "complete",
            "run_token": run_token,
            "job_name": job_name,
            "output_root": str(REMOTE_OUTPUTS / "act_experiments" / job_name),
            "pretrained_checkpoint": str(pretrained_path) if pretrained_path is not None else None,
            "resume_checkpoint": str(resume_path) if resume_path is not None else None,
        },
        sort_keys=True,
    )


@app.function(
    gpu=MODAL_GPU,
    # Bounded by the shared hard runtime rather than a day, so an unattended
    # RECAP run cannot bill indefinitely if the local client disconnects.
    timeout=MODAL_MAX_RUNTIME_SECONDS,
    memory=65_536,
    volumes={REMOTE_OUTPUTS: vol},
    secrets=[hf_secret],
)
def train_recap_policy_remote(
    pretrained_checkpoint: str,
    steps: int = 10_000,
    save_freq: int = 2_000,
    job_name: str = "recap_policy",
    labels_path: str | None = None,
    dropout: str | None = None,
    dataset_repo_id: str | None = None,
) -> str:
    """Train the advantage-conditioned RECAP policy.

    Architecture and optimization hyperparameters are inherited from
    ``pretrained_checkpoint`` by models.recap.train_policy, so this entrypoint
    only carries run identity and budget. Omitting ``labels_path`` runs the SFT
    stage with the indicator fixed to POSITIVE.

    ``dataset_repo_id`` must name the dataset the labels were built from.
    Without it the trainer inherits the warm-start checkpoint's dataset, which
    is the demonstration set the baseline was trained on, while the labels
    index the larger RECAP set. Nothing raises in that case -- the frame ids
    stay in range -- so every transition would silently train against another
    transition's advantage token.
    """
    import json
    import os
    import subprocess
    import sys
    import threading

    pretrained_checkpoint = validate_pretrained_checkpoint_path(pretrained_checkpoint)
    if pretrained_checkpoint is None:
        raise ValueError("pretrained_checkpoint is required for RECAP policy training.")
    pretrained_path = REMOTE_OUTPUTS / pretrained_checkpoint
    if not pretrained_path.is_file():
        raise FileNotFoundError(
            f"Warm-start checkpoint is missing from Modal volume: {pretrained_path}"
        )
    labels_full_path = ""
    if labels_path:
        if not dataset_repo_id:
            raise ValueError(
                "labels_path requires dataset_repo_id: advantage labels are keyed "
                "by frame index into the dataset they were built from, and "
                "inheriting the warm-start checkpoint's dataset instead would "
                "pair every transition with another transition's token without "
                "raising."
            )
        labels_full_path = str(REMOTE_OUTPUTS / validate_volume_relative_path(labels_path))
        if not Path(labels_full_path).is_file():
            raise FileNotFoundError(f"Advantage labels are missing: {labels_full_path}")

    run_token = uuid.uuid4().hex
    environment = {
        "PYTHONUNBUFFERED": "1",
        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
        "HUGGINGFACE_HUB_TOKEN": os.environ.get("HF_TOKEN", ""),
        "RECAP_DEVICE": "cuda",
        "RECAP_POLICY_PRETRAINED_CHECKPOINT": str(pretrained_path),
        "RECAP_POLICY_STEPS": str(steps),
        "RECAP_POLICY_SAVE_FREQ": str(save_freq),
        "RECAP_POLICY_JOB_NAME": job_name,
        "RECAP_POLICY_OUTPUT_ROOT": str(REMOTE_OUTPUTS / "act_experiments"),
    }
    if labels_full_path:
        environment["RECAP_POLICY_LABELS"] = labels_full_path
    if dataset_repo_id:
        environment["RECAP_POLICY_DATASET_REPO_ID"] = dataset_repo_id
    if dropout is not None:
        environment["RECAP_POLICY_DROPOUT"] = str(dropout)
    os.environ.update(environment)

    sys.path.insert(0, str(REMOTE_WORKSPACE))
    os.chdir(REMOTE_WORKSPACE)

    import torch

    print("=== CUDA PREFLIGHT ===", flush=True)
    print(f"run_token={run_token}", flush=True)
    print(f"torch={torch.__version__} cuda_runtime={torch.version.cuda}", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; refusing CPU fallback on a paid GPU job.")
    print(
        f"gpu={torch.cuda.get_device_name(0)} "
        f"vram_gb={torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}",
        flush=True,
    )
    stage = "improvement_loop" if labels_full_path else "sft_positive"
    print(
        f"recap_config stage={stage} pretrained={pretrained_path} steps={steps} "
        f"save_freq={save_freq} job_name={job_name} labels={labels_full_path or None}",
        flush=True,
    )

    monitor_dir = REMOTE_OUTPUTS / "modal_monitors"
    monitor_dir.mkdir(parents=True, exist_ok=True)
    telemetry_path = monitor_dir / f"{run_token}_recap.jsonl"
    stop_monitor = threading.Event()

    def sample_gpu() -> None:
        while not stop_monitor.is_set():
            timestamp = datetime.now(timezone.utc).isoformat()
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                values = [value.strip() for value in result.stdout.strip().split(",")]
                entry = {
                    "timestamp": timestamp,
                    "gpu_name": values[0],
                    "utilization_percent": float(values[1]),
                    "memory_used_mib": float(values[2]),
                    "memory_total_mib": float(values[3]),
                    "temperature_c": float(values[4]),
                }
            except Exception as exc:
                entry = {"timestamp": timestamp, "telemetry_error": repr(exc)}
            with telemetry_path.open("a") as file:
                file.write(json.dumps(entry, sort_keys=True) + "\n")
            print(f"GPU_TELEMETRY {json.dumps(entry, sort_keys=True)}", flush=True)
            # Commit so checkpoints survive if the container is torn down.
            vol.commit()
            stop_monitor.wait(VOLUME_COMMIT_INTERVAL_SECONDS)

    monitor_thread = threading.Thread(target=sample_gpu, name="gpu-telemetry", daemon=True)
    monitor_thread.start()
    try:
        from models.recap.train_policy import main as train_main

        train_main()
    finally:
        stop_monitor.set()
        monitor_thread.join(timeout=10)
        vol.commit()
    return json.dumps(
        {
            "status": "complete",
            "run_token": run_token,
            "stage": stage,
            "job_name": job_name,
            "output_root": str(REMOTE_OUTPUTS / "act_experiments" / job_name),
            "pretrained_checkpoint": str(pretrained_path),
        },
        sort_keys=True,
    )


@app.function(
    gpu=MODAL_GPU,
    timeout=86_400,
    memory=65_536,
    volumes={REMOTE_OUTPUTS: vol, REMOTE_BENCHMARK_DATA: benchmark_vol},
    secrets=[hf_secret],
)
def train_remote(profile: str, run_name: str | None = None) -> str:
    import os
    import sys
    import threading

    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
    os.environ["HUGGINGFACE_HUB_TOKEN"] = os.environ.get("HF_TOKEN", "")

    sys.path.insert(0, str(REMOTE_WORKSPACE))
    os.chdir(REMOTE_WORKSPACE)

    import torch

    print("=== CUDA PREFLIGHT ===")
    print("Torch:", torch.__version__)
    print("CUDA runtime:", torch.version.cuda)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this Modal container.")
    print("Detected GPU:", torch.cuda.get_device_name(0))
    print("VRAM (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))

    config_module = importlib.import_module(f"configs.act.{profile}")
    config_module.config = configure_remote_profile(config_module.config, profile)
    if config_module.config.dataset_format == "act_hdf5":
        benchmark_vol.reload()
        dataset_dir = Path(config_module.config.benchmark_dataset_dir)
        episode_files = sorted(dataset_dir.glob("episode_*.hdf5"))
        expected = config_module.config.benchmark_num_episodes
        print(f"benchmark_dataset_dir={dataset_dir}")
        print(f"benchmark_episode_files={len(episode_files)}")
        if expected is not None and len(episode_files) != expected:
            raise RuntimeError(f"Expected {expected} benchmark episodes in {dataset_dir}, found {len(episode_files)}")
        if not episode_files:
            raise RuntimeError(
                f"No benchmark episodes found in {dataset_dir}. "
                "Run the local entrypoint with --upload-benchmark-data first."
            )
    remote_output_root = Path(config_module.config.output_root)

    from models.act.train import main as train_main

    print("=== STARTING CUSTOM ACT TRAINING ===")
    print(f"profile={profile}")
    print(f"run_name={run_name}")
    print(f"output_root={remote_output_root}")
    print(f"config={asdict(config_module.config)}")

    stop_commits = threading.Event()

    def commit_loop() -> None:
        while not stop_commits.wait(VOLUME_COMMIT_INTERVAL_SECONDS):
            print("Committing outputs volume for periodic checkpoint download...", flush=True)
            vol.commit()

    commit_thread = threading.Thread(target=commit_loop, daemon=True)
    commit_thread.start()
    try:
        train_main(profile=profile, run_name=run_name, dry_run=False)
    finally:
        stop_commits.set()
        commit_thread.join(timeout=5)
        vol.commit()
    return f"Custom ACT training complete for profile={profile}; outputs saved in volume omx-custom-act-training-logs."


@app.function(volumes={REMOTE_OUTPUTS: vol})
def list_outputs() -> list[str]:
    import os

    vol.reload()
    results = []
    for root, _dirs, files in os.walk(REMOTE_OUTPUTS):
        for filename in files:
            full = os.path.join(root, filename)
            results.append(os.path.relpath(full, REMOTE_OUTPUTS))
    return sorted(results)


@app.local_entrypoint()
def main(
    profile: str = "pour_water",
    run_name: str | None = None,
    preflight_only: bool = False,
    benchmark_dataset_dir: str | None = None,
    upload_benchmark_data: bool = False,
    list_benchmark_only: bool = False,
    dataset_repo_id: str | None = None,
    steps: int = 20_000,
    chunk_size: int = 30,
    save_freq: int = 2_000,
    train_split: float = 1.0,
    job_name: str = "act_custom_lerobot_arch_chunk30_steps20000_all50",
    pretrained_checkpoint: str | None = None,
    resume_checkpoint: str | None = None,
    image_transforms: bool = False,
):
    if dataset_repo_id is not None:
        validate_dataset_run(
            dataset_repo_id=dataset_repo_id,
            steps=steps,
            chunk_size=chunk_size,
            save_freq=save_freq,
            train_split=train_split,
            job_name=job_name,
        )
        pretrained_checkpoint = validate_pretrained_checkpoint_path(pretrained_checkpoint)
        resume_checkpoint = validate_pretrained_checkpoint_path(resume_checkpoint)
        if pretrained_checkpoint and resume_checkpoint:
            raise ValueError(
                "--pretrained-checkpoint and --resume-checkpoint are mutually exclusive."
            )
        print(
            "DATASET_RUN "
            f"gpu={MODAL_GPU} hard_timeout_seconds={MODAL_MAX_RUNTIME_SECONDS} "
            f"dataset={dataset_repo_id} steps={steps} chunk_size={chunk_size} "
            f"save_freq={save_freq} train_split={train_split} job_name={job_name} "
            f"image_transforms={image_transforms} "
            f"pretrained_checkpoint={pretrained_checkpoint} "
            f"resume_checkpoint={resume_checkpoint}"
        )
        if preflight_only:
            print("Preflight only; not launching Modal GPU job.")
            return
        run_token = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-"
            + uuid.uuid4().hex[:8]
        )
        function_call = train_dataset_remote.spawn(
            dataset_repo_id=dataset_repo_id,
            steps=steps,
            chunk_size=chunk_size,
            save_freq=save_freq,
            train_split=train_split,
            job_name=job_name,
            run_token=run_token,
            pretrained_checkpoint=pretrained_checkpoint,
            resume_checkpoint=resume_checkpoint,
            image_transforms=image_transforms,
        )
        print(f"MODAL_FUNCTION_CALL_ID={function_call.object_id}")
        print(f"MODAL_RUN_TOKEN={run_token}")
        print(f"MODAL_JOB_NAME={job_name}")
        print(f"MODAL_HARD_TIMEOUT_SECONDS={MODAL_MAX_RUNTIME_SECONDS}")
        print("MODAL_OUTPUT_VOLUME=omx-custom-act-training-logs")
        print(f"MODAL_OUTPUT_PATH=act_experiments/{job_name}")
        return

    preflight_profile(profile)
    config = load_profile(profile)
    if upload_benchmark_data:
        if config.dataset_format != "act_hdf5":
            raise ValueError("--upload-benchmark-data is only valid for dataset_format='act_hdf5' profiles")
        upload_benchmark_dataset(profile, benchmark_dataset_dir or config.benchmark_dataset_dir or "")
    if list_benchmark_only:
        files = list_benchmark_data.remote(profile)
        print(f"Benchmark files in Modal volume for {profile}:")
        for filename in files[:20]:
            print(f"  {filename}")
        if len(files) > 20:
            print(f"  ... {len(files) - 20} more")
        return
    if preflight_only:
        print("Preflight only; not launching Modal GPU job.")
        return
    if upload_benchmark_data:
        print("Benchmark upload complete; not launching Modal GPU job.")
        print("Start training with:")
        print(f"  modal run deploy/train_custom_act_modal.py --profile {profile} --run-name {run_name or 'seed0'}")
        return

    function_call = train_remote.spawn(profile, run_name)
    print(f"Spawned Modal custom ACT training call: {function_call.object_id}")
    print("Outputs volume: omx-custom-act-training-logs")
    print("Download later with:")
    print("  modal volume get omx-custom-act-training-logs / outputs/modal_custom_act")


@app.local_entrypoint()
def recap_policy(
    pretrained_checkpoint: str,
    dataset_repo_id: str,
    labels_path: str | None = None,
    steps: int = 20_000,
    save_freq: int = 2_000,
    job_name: str = "recap_policy",
    dropout: str | None = None,
):
    """Launch advantage-conditioned RECAP policy training.

    ``labels_path`` selects the stage: omitted runs SFT with the indicator
    pinned to POSITIVE, supplied runs the improvement loop. It is refused
    without ``dataset_repo_id``, because labels are keyed by frame index into
    the dataset they were built from.

    Both paths are volume-relative, since the container reads them from the
    training volume rather than from the image.
    """
    stage = "improvement loop" if labels_path else "SFT (indicator pinned POSITIVE)"
    print("=== RECAP POLICY LAUNCH ===")
    print(f"stage={stage}")
    print(f"dataset={dataset_repo_id}")
    print(f"warm_start={pretrained_checkpoint}")
    print(f"labels={labels_path or '(none)'}")
    print(f"steps={steps} save_freq={save_freq} dropout={dropout or '(config default)'}")
    print(f"MODAL_GPU={MODAL_GPU}")
    print(f"MODAL_HARD_TIMEOUT_SECONDS={MODAL_MAX_RUNTIME_SECONDS}")

    result = train_recap_policy_remote.remote(
        pretrained_checkpoint=pretrained_checkpoint,
        steps=steps,
        save_freq=save_freq,
        job_name=job_name,
        labels_path=labels_path,
        dropout=dropout,
        dataset_repo_id=dataset_repo_id,
    )
    print(result)
    print("Outputs volume: omx-custom-act-training-logs")
    print(f"  modal volume get omx-custom-act-training-logs act_experiments/{job_name} .")
