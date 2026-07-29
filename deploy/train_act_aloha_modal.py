"""
Stock LeRobot ACT on Modal.

By default, trains a fresh official LeRobot ACT policy on the current OMX
dataset. The old ALOHA sim-transfer-cube dataset/eval path remains available
with dataset="aloha".

Usage:
    # Towel dataset run using official lerobot-train.
    modal run --detach deploy/train_act_aloha_modal.py::train_remote --steps 10000

    # Old ALOHA smoke (train + save), then eval the saved checkpoint.
    modal run --detach deploy/train_act_aloha_modal.py::train_aloha_remote --steps 2000
    modal run deploy/train_act_aloha_modal.py::eval_remote --steps 2000 --episodes 50

    # Old ALOHA full run.
    modal run --detach deploy/train_act_aloha_modal.py::train_aloha_remote --steps 80000

Download checkpoints:
    modal volume get omx-act-aloha-logs /act_lerobot_009-place-towel-in-bowl-10fps/<steps>/checkpoints/last/pretrained_model ./pretrained_model
"""

from __future__ import annotations

from datetime import datetime, timezone
import shlex
import subprocess
from pathlib import Path
import uuid

import modal

try:
    from modal_training_common import validate_volume_relative_path
except ModuleNotFoundError:
    from deploy.modal_training_common import validate_volume_relative_path


REMOTE_OUTPUTS = Path("/outputs")
MODAL_GPU = "A10G"
MODAL_MAX_RUNTIME_SECONDS = 3 * 60 * 60
VOLUME_COMMIT_INTERVAL_SECONDS = 300
LEROBOT_GIT_REF = "87242cfced5228b6181fb0047021d684d424df38"
ALOHA_DATASET_REPO_ID = "lerobot/aloha_sim_transfer_cube_human"
DEFAULT_OMX_DATASET_REPO_ID = "RevanthGundala/007-place-penne-pasta-in-bowl"
TOWEL_DATASET_REPO_ID = "RevanthGundala/009-place-towel-in-bowl-10fps"

hf_secret = modal.Secret.from_name("huggingface")

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04", add_python="3.12")
    .apt_install(
        "git", "ffmpeg", "linux-libc-dev", "clang",
        # Headless OpenGL/EGL for MuJoCo rendering during gym-aloha eval.
        "libegl1", "libgl1", "libgles2", "libglib2.0-0", "libosmesa6",
    )
    .pip_install(
        "torch==2.7.1",
        "torchvision==0.22.1",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        f"lerobot[training] @ git+https://github.com/huggingface/lerobot.git@{LEROBOT_GIT_REF}",
        "gym-aloha>=0.1.4",
        "accelerate",
        "av>=15,<16",
        "huggingface-hub>=1.0,<2.0",
        "safetensors",
        "torchcodec==0.5.0",
    )
    # EGL = headless GPU rendering for MuJoCo (no display/X server in container).
    .env({"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"})
    .add_local_file("deploy/modal_training_common.py", remote_path="/root/modal_training_common.py")
)

app = modal.App("omx-act-aloha-training", image=image)
vol = modal.Volume.from_name("omx-act-aloha-logs", create_if_missing=True)


def _run(command: list[str]) -> None:
    print("RUN:", " \\\n    ".join(shlex.quote(p) for p in command), flush=True)
    subprocess.run(command, check=True)


def _repo_slug(repo_id: str) -> str:
    return repo_id.split("/")[-1].replace("_", "-")


def _run_dir(
    dataset: str,
    steps: int,
    dataset_repo_id: str | None = None,
    run_name_suffix: str = "",
) -> Path:
    name = "act_aloha" if dataset == "aloha" else f"act_lerobot_{_repo_slug(dataset_repo_id or DEFAULT_OMX_DATASET_REPO_ID)}"
    if run_name_suffix:
        name = f"{name}_{run_name_suffix}"
    return REMOTE_OUTPUTS / name / str(steps)


def _dataset_repo_id(dataset: str) -> str:
    if dataset == "aloha":
        return ALOHA_DATASET_REPO_ID
    if dataset == "omx":
        return DEFAULT_OMX_DATASET_REPO_ID
    raise ValueError(f"Unsupported dataset={dataset!r}. Use 'omx' or 'aloha'.")


@app.function(
    gpu=MODAL_GPU,
    timeout=86_400,
    memory=65_536,
    volumes={REMOTE_OUTPUTS: vol},
    secrets=[hf_secret],
)
def train_remote(
    steps: int = 10_000,
    batch_size: int = 8,
    save_freq: int | None = 2_500,
    eval_freq: int = 1_000,
    log_freq: int = 100,
    resume_from_steps: int | None = None,
    dataset_repo_id: str = TOWEL_DATASET_REPO_ID,
    chunk_size: int = 30,
    n_action_steps: int = 1,
    temporal_ensemble_coeff: float | None = 0.01,
    use_temporal_ensemble: bool = True,
    use_vae: bool = True,
    kl_weight: float = 10.0,
    lr: float = 1e-5,
    lr_backbone: float = 1e-5,
    run_name_suffix: str = "",
    dataset_episodes: str = "",
) -> str:
    import torch

    print("Torch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available in Modal container.")

    if not use_temporal_ensemble:
        temporal_ensemble_coeff = None

    output_dir = _run_dir(
        "omx",
        steps,
        dataset_repo_id=dataset_repo_id,
        run_name_suffix=run_name_suffix,
    )
    effective_save_freq = save_freq or (steps - resume_from_steps if resume_from_steps is not None else steps)
    command = [
        "lerobot-train",
        "--policy.type=act",
        "--policy.push_to_hub=false",
        f"--dataset.repo_id={dataset_repo_id}",
        "--dataset.revision=main",
        "--policy.device=cuda",
        f"--policy.chunk_size={chunk_size}",
        f"--policy.n_action_steps={n_action_steps}",
        "--policy.dim_model=512",
        "--policy.n_heads=8",
        "--policy.dim_feedforward=3200",
        "--policy.n_encoder_layers=4",
        "--policy.n_decoder_layers=1",
        "--policy.latent_dim=32",
        "--policy.n_vae_encoder_layers=4",
        f"--policy.use_vae={str(use_vae).lower()}",
        "--policy.dropout=0.1",
        f"--policy.kl_weight={kl_weight}",
        f"--optimizer.lr={lr}",
        "--optimizer.weight_decay=1e-4",
        "--optimizer.grad_clip_norm=1.0",
        f"--policy.optimizer_lr_backbone={lr_backbone}",
        f"--steps={steps}",
        f"--batch_size={batch_size}",
        f"--save_freq={effective_save_freq}",
        "--save_checkpoint=true",
        f"--eval_freq={eval_freq}",
        f"--log_freq={log_freq}",
        "--wandb.enable=false",
        f"--output_dir={output_dir}",
        f"--job_name=act_lerobot_{_repo_slug(dataset_repo_id)}{('_' + run_name_suffix) if run_name_suffix else ''}",
    ]
    if temporal_ensemble_coeff is not None:
        command.append(f"--policy.temporal_ensemble_coeff={temporal_ensemble_coeff}")
    if dataset_episodes:
        command.append(f"--dataset.episodes=[{dataset_episodes}]")
    if resume_from_steps is not None:
        checkpoint = (
            _run_dir(
                "omx",
                steps,
                dataset_repo_id=dataset_repo_id,
                run_name_suffix=run_name_suffix,
            )
            / "checkpoints"
            / f"{resume_from_steps:06d}"
        )
        config_path = checkpoint / "pretrained_model" / "train_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"No train_config.json at {config_path}")
        command.extend([
            "--resume=true",
            f"--config_path={config_path}",
        ])
    _run(command)
    vol.commit()
    return f"Trained LeRobot ACT on {dataset_repo_id} for {steps} steps; checkpoint under {output_dir}/checkpoints."


@app.function(
    gpu=MODAL_GPU,
    cpu=4.0,
    timeout=MODAL_MAX_RUNTIME_SECONDS,
    memory=32_768,
    volumes={REMOTE_OUTPUTS: vol},
    secrets=[hf_secret],
)
def resume_dataset_remote(
    checkpoint_path: str,
    output_path: str,
    steps: int,
    save_freq: int,
    log_freq: int,
    run_token: str,
) -> str:
    import json
    import os
    import re
    import subprocess
    import threading
    import time

    import torch

    checkpoint_path = validate_volume_relative_path(
        checkpoint_path,
        expected_prefix="resume_sources",
    )
    output_path = validate_volume_relative_path(
        output_path,
        expected_prefix="act_lerobot_014-place-sponge-in-bowl-diverse-50",
    )
    if steps <= 20_000:
        raise ValueError("steps must be greater than the 20k resume checkpoint.")
    if not 0 < save_freq < steps:
        raise ValueError("save_freq must be positive and smaller than steps.")

    vol.reload()
    checkpoint_dir = REMOTE_OUTPUTS / checkpoint_path
    config_path = checkpoint_dir / "pretrained_model" / "train_config.json"
    training_step_path = checkpoint_dir / "training_state" / "training_step.json"
    if not config_path.is_file() or not training_step_path.is_file():
        raise FileNotFoundError(
            f"Resume checkpoint is incomplete: {checkpoint_dir}"
        )
    resume_step = int(json.loads(training_step_path.read_text())["step"])
    if resume_step >= steps:
        raise ValueError(
            f"Resume step {resume_step} must be smaller than target step {steps}."
        )
    output_dir = REMOTE_OUTPUTS / output_path
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")

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
    print(
        f"resume_checkpoint={checkpoint_dir} resume_step={resume_step} "
        f"target_step={steps} save_freq={save_freq} output_dir={output_dir}",
        flush=True,
    )

    monitor_dir = REMOTE_OUTPUTS / "modal_monitors"
    monitor_dir.mkdir(parents=True, exist_ok=True)
    status_path = monitor_dir / f"{run_token}_training.json"
    telemetry_path = monitor_dir / f"{run_token}.jsonl"
    state_lock = threading.Lock()
    state = {
        "status": "running",
        "step": resume_step,
        "loss": None,
        "resume_step": resume_step,
        "target_step": steps,
        "output_dir": str(output_dir),
    }

    def write_status(**updates) -> None:
        with state_lock:
            state.update(updates)
            payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **state,
            }
        status_path.write_text(json.dumps(payload, sort_keys=True) + "\n")

    stop_monitor = threading.Event()

    def sample_gpu() -> None:
        while not stop_monitor.is_set():
            timestamp = datetime.now(timezone.utc).isoformat()
            telemetry = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
            )
            entry = {
                "timestamp": timestamp,
                "exit_code": telemetry.returncode,
                "telemetry": telemetry.stdout.strip(),
                "error": telemetry.stderr.strip(),
            }
            with telemetry_path.open("a") as file:
                file.write(json.dumps(entry, sort_keys=True) + "\n")
            print(f"GPU_TELEMETRY {json.dumps(entry, sort_keys=True)}", flush=True)
            write_status()
            vol.commit()
            stop_monitor.wait(VOLUME_COMMIT_INTERVAL_SECONDS)

    write_status()
    monitor_thread = threading.Thread(target=sample_gpu, name="gpu-telemetry", daemon=True)
    monitor_thread.start()
    command = [
        "lerobot-train",
        "--resume=true",
        f"--config_path={config_path}",
        f"--steps={steps}",
        f"--save_freq={save_freq}",
        "--save_checkpoint=true",
        "--eval_freq=0",
        f"--log_freq={log_freq}",
        "--wandb.enable=false",
        f"--output_dir={output_dir}",
        "--job_name=lerobot_014_unaugmented_resume20k_to50k",
    ]
    print("RUN:", " \\\n    ".join(shlex.quote(part) for part in command), flush=True)
    progress_pattern = re.compile(r"step:(\d+).*?loss:([0-9.eE+-]+)")
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            matches = progress_pattern.findall(line)
            if matches:
                step_value, loss_value = matches[-1]
                write_status(step=int(step_value), loss=float(loss_value))
        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)
        write_status(status="complete", step=steps)
    except Exception as exc:
        write_status(status="failed", error=repr(exc))
        raise
    finally:
        stop_monitor.set()
        monitor_thread.join(timeout=10)
        vol.commit()
    return json.dumps(
        {
            "status": "complete",
            "run_token": run_token,
            "resume_step": resume_step,
            "target_step": steps,
            "output_dir": str(output_dir),
        },
        sort_keys=True,
    )


@app.function(
    timeout=7_200,
    memory=16_384,
    secrets=[hf_secret],
)
def clone_single_episode_dataset_remote(
    source_repo_id: str = TOWEL_DATASET_REPO_ID,
    target_repo_id: str = "RevanthGundala/009-place-towel-in-bowl-10fps-ep1-only",
    episode: int = 1,
    target_fps: int | None = None,
    private: bool = False,
) -> str:
    import numpy as np
    import torch
    from lerobot.datasets import LeRobotDataset

    def tensor_to_frame_value(value):
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                return value.item()
            return value.detach().cpu().numpy()
        return value

    def frame_for_clone(sample: dict, feature_keys: list[str]) -> dict:
        frame = {}
        for key in feature_keys:
            value = tensor_to_frame_value(sample[key])
            if key.startswith("observation.images.") and isinstance(value, np.ndarray):
                if value.ndim == 3 and value.shape[0] in (1, 3):
                    value = np.transpose(value, (1, 2, 0))
                if value.dtype != np.uint8 and value.max() <= 1.0:
                    value = (value * 255).clip(0, 255).astype(np.uint8)
            frame[key] = value
        frame["task"] = sample["task"]
        return frame

    source = LeRobotDataset(source_repo_id, episodes=[episode])
    features = {
        key: value
        for key, value in source.features.items()
        if key not in {"timestamp", "frame_index", "episode_index", "index", "task_index"}
    }
    target = LeRobotDataset.create(
        repo_id=target_repo_id,
        fps=target_fps or getattr(source, "fps", source.meta.info["fps"]),
        features=features,
        robot_type=source.meta.info.get("robot_type"),
        use_videos=True,
    )
    feature_keys = list(features)
    for idx in range(len(source)):
        target.add_frame(frame_for_clone(source[idx], feature_keys))
    target.save_episode()
    target.finalize()
    target.push_to_hub(private=private)
    return f"Pushed {len(source)} frames from {source_repo_id} episode {episode} to {target_repo_id}"


@app.function(
    gpu=MODAL_GPU,
    timeout=86_400,
    memory=65_536,
    volumes={REMOTE_OUTPUTS: vol},
    secrets=[hf_secret],
)
def train_aloha_remote(steps: int = 2000, batch_size: int = 8, save_freq: int | None = None) -> str:
    import torch

    print("Torch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available in Modal container.")

    output_dir = _run_dir("aloha", steps)
    command = [
        "lerobot-train",
        "--policy.type=act",
        "--policy.push_to_hub=false",
        f"--dataset.repo_id={ALOHA_DATASET_REPO_ID}",
        "--policy.device=cuda",
        f"--steps={steps}",
        f"--batch_size={batch_size}",
        f"--save_freq={save_freq or steps}",
        "--save_checkpoint=true",
        "--eval_freq=0",
        "--log_freq=200",
        "--wandb.enable=false",
        f"--output_dir={output_dir}",
    ]
    _run(command)
    vol.commit()
    return f"Trained ALOHA ACT for {steps} steps; checkpoint under {output_dir}/checkpoints."


@app.function(
    gpu=MODAL_GPU,
    timeout=86_400,
    memory=65_536,
    volumes={REMOTE_OUTPUTS: vol},
    secrets=[hf_secret],
)
def eval_remote(steps: int = 2000, episodes: int = 50, checkpoint: str = "last") -> str:
    vol.reload()
    pretrained = _run_dir("aloha", steps) / "checkpoints" / checkpoint / "pretrained_model"
    if not pretrained.exists():
        raise FileNotFoundError(f"No checkpoint at {pretrained}. Run train_remote --steps {steps} first.")

    output_dir = _run_dir("aloha", steps) / "eval" / checkpoint
    command = [
        "lerobot-eval",
        f"--policy.path={pretrained}",
        "--policy.device=cuda",
        "--env.type=aloha",
        "--env.task=AlohaTransferCube-v0",
        f"--eval.n_episodes={episodes}",
        "--eval.batch_size=10",
        f"--output_dir={output_dir}",
    ]
    _run(command)
    vol.commit()
    return f"Evaluated {pretrained} over {episodes} episodes; results under {output_dir}."


@app.local_entrypoint()
def main(
    dataset: str = "omx",
    steps: int = 10_000,
    episodes: int = 50,
    do_eval: bool = False,
    dataset_repo_id: str = TOWEL_DATASET_REPO_ID,
    chunk_size: int = 30,
    n_action_steps: int = 1,
    temporal_ensemble_coeff: float | None = 0.01,
    use_temporal_ensemble: bool = True,
    use_vae: bool = True,
    kl_weight: float = 10.0,
    lr: float = 1e-5,
    lr_backbone: float = 1e-5,
    run_name_suffix: str = "",
    dataset_episodes: str = "",
    batch_size: int = 8,
    save_freq: int = 2_500,
    eval_freq: int = 1_000,
    log_freq: int = 100,
):
    if dataset not in {"omx", "aloha"}:
        raise ValueError("dataset must be 'omx' or 'aloha'")
    print(
        "Spawning LeRobot ACT training: "
        f"dataset={dataset} repo={dataset_repo_id} steps={steps} chunk_size={chunk_size} "
        f"n_action_steps={n_action_steps} temporal_ensemble_coeff={temporal_ensemble_coeff} "
        f"use_vae={use_vae} kl_weight={kl_weight} lr={lr} lr_backbone={lr_backbone} "
        f"use_temporal_ensemble={use_temporal_ensemble} run_name_suffix={run_name_suffix!r} "
        f"dataset_episodes={dataset_episodes!r} "
        f"batch_size={batch_size} save_freq={save_freq} eval_freq={eval_freq} log_freq={log_freq}"
    )
    handle = (
        train_aloha_remote.spawn(steps)
        if dataset == "aloha"
        else train_remote.spawn(
            steps=steps,
            batch_size=batch_size,
            save_freq=save_freq,
            eval_freq=eval_freq,
            log_freq=log_freq,
            dataset_repo_id=dataset_repo_id,
            chunk_size=chunk_size,
            n_action_steps=n_action_steps,
            temporal_ensemble_coeff=temporal_ensemble_coeff,
            use_temporal_ensemble=use_temporal_ensemble,
            use_vae=use_vae,
            kl_weight=kl_weight,
            lr=lr,
            lr_backbone=lr_backbone,
            run_name_suffix=run_name_suffix,
            dataset_episodes=dataset_episodes,
        )
    )
    print(f"  train call: {handle.object_id}")
    if do_eval and dataset == "aloha":
        print("After training completes, eval with:")
        print(f"  modal run deploy/train_act_aloha_modal.py::eval_remote --steps {steps} --episodes {episodes}")
    elif do_eval:
        print("No Modal eval is configured for OMX datasets.")
    print("Outputs volume: omx-act-aloha-logs")


@app.local_entrypoint()
def resume_main(
    checkpoint_path: str,
    output_path: str,
    steps: int = 50_000,
    save_freq: int = 2_000,
    log_freq: int = 100,
    preflight_only: bool = False,
):
    checkpoint_path = validate_volume_relative_path(
        checkpoint_path,
        expected_prefix="resume_sources",
    )
    output_path = validate_volume_relative_path(
        output_path,
        expected_prefix="act_lerobot_014-place-sponge-in-bowl-diverse-50",
    )
    print(
        "LEROBOT_RESUME "
        f"gpu={MODAL_GPU} hard_timeout_seconds={MODAL_MAX_RUNTIME_SECONDS} "
        f"checkpoint_path={checkpoint_path} output_path={output_path} "
        f"steps={steps} save_freq={save_freq} log_freq={log_freq}"
    )
    if preflight_only:
        print("Preflight only; not launching Modal GPU job.")
        return
    run_token = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + uuid.uuid4().hex[:8]
    )
    function_call = resume_dataset_remote.spawn(
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        steps=steps,
        save_freq=save_freq,
        log_freq=log_freq,
        run_token=run_token,
    )
    print(f"MODAL_FUNCTION_CALL_ID={function_call.object_id}")
    print(f"MODAL_RUN_TOKEN={run_token}")
    print(f"MODAL_HARD_TIMEOUT_SECONDS={MODAL_MAX_RUNTIME_SECONDS}")
    print("MODAL_OUTPUT_VOLUME=omx-act-aloha-logs")
    print(f"MODAL_OUTPUT_PATH={output_path}")
