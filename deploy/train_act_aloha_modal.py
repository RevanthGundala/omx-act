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

import shlex
import subprocess
from pathlib import Path

import modal


REMOTE_OUTPUTS = Path("/outputs")
MODAL_GPU = "A10G"
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
        "torch",
        "torchvision",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "lerobot[training]>=0.5.1",
        "gym-aloha>=0.1.4",
        "accelerate",
        "huggingface-hub>=1.0,<2.0",
    )
    # EGL = headless GPU rendering for MuJoCo (no display/X server in container).
    .env({"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"})
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
