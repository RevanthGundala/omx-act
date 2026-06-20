"""
Stock LeRobot ACT on Modal.

By default, trains a fresh LeRobot ACT policy on our pour-water dataset using the
same ACT hyperparameters as configs.act.base. The old ALOHA sim-transfer-cube
dataset/eval path remains available with dataset="aloha".

Usage:
    # Pour-water comparison run matching our ACT hyperparams.
    modal run --detach deploy/train_act_aloha_modal.py::train_remote --steps 5000

    # Old ALOHA smoke (train + save), then eval the saved checkpoint.
    modal run --detach deploy/train_act_aloha_modal.py::train_aloha_remote --steps 2000
    modal run deploy/train_act_aloha_modal.py::eval_remote --steps 2000 --episodes 50

    # Old ALOHA full run.
    modal run --detach deploy/train_act_aloha_modal.py::train_aloha_remote --steps 80000

Download checkpoints:
    modal volume get omx-act-aloha-logs /act_lerobot_pour_water/<steps>/checkpoints/last/pretrained_model ./pretrained_model
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

import modal


REMOTE_OUTPUTS = Path("/outputs")
MODAL_GPU = "A10G"
ALOHA_DATASET_REPO_ID = "lerobot/aloha_sim_transfer_cube_human"
POUR_WATER_DATASET_REPO_ID = "RevanthGundala/007-place-penne-pasta-in-bowl"

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


def _run_dir(dataset: str, steps: int) -> Path:
    name = "act_aloha" if dataset == "aloha" else "act_lerobot_pour_water"
    return REMOTE_OUTPUTS / name / str(steps)


def _dataset_repo_id(dataset: str) -> str:
    if dataset == "aloha":
        return ALOHA_DATASET_REPO_ID
    if dataset == "pour_water":
        return POUR_WATER_DATASET_REPO_ID
    raise ValueError(f"Unsupported dataset={dataset!r}. Use 'pour_water' or 'aloha'.")


@app.function(
    gpu=MODAL_GPU,
    timeout=86_400,
    memory=65_536,
    volumes={REMOTE_OUTPUTS: vol},
    secrets=[hf_secret],
)
def train_remote(
    steps: int = 2000,
    batch_size: int = 8,
    save_freq: int | None = None,
    resume_from_steps: int | None = None,
) -> str:
    import torch

    print("Torch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available in Modal container.")

    output_dir = _run_dir("pour_water", resume_from_steps or steps)
    effective_save_freq = save_freq or (steps - resume_from_steps if resume_from_steps is not None else steps)
    command = [
        "lerobot-train",
        "--policy.type=act",
        "--policy.push_to_hub=false",
        f"--dataset.repo_id={POUR_WATER_DATASET_REPO_ID}",
        "--dataset.revision=main",
        "--policy.device=cuda",
        "--policy.chunk_size=100",
        "--policy.n_action_steps=100",
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
        "--optimizer.grad_clip_norm=1.0",
        f"--steps={steps}",
        f"--batch_size={batch_size}",
        f"--save_freq={effective_save_freq}",
        "--save_checkpoint=true",
        "--eval_freq=0",
        "--log_freq=100",
        "--wandb.enable=false",
        f"--output_dir={output_dir}",
        "--job_name=act_lerobot_pour_water",
    ]
    if resume_from_steps is not None:
        checkpoint = _run_dir("pour_water", resume_from_steps) / "checkpoints" / f"{resume_from_steps:06d}"
        config_path = checkpoint / "pretrained_model" / "train_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"No train_config.json at {config_path}")
        command.extend([
            "--resume=true",
            f"--config_path={config_path}",
        ])
    _run(command)
    vol.commit()
    return f"Trained LeRobot ACT on pour-water for {steps} steps; checkpoint under {output_dir}/checkpoints."


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
def main(dataset: str = "pour_water", steps: int = 5000, episodes: int = 50, do_eval: bool = False):
    if dataset not in {"pour_water", "aloha"}:
        raise ValueError("dataset must be 'pour_water' or 'aloha'")
    print(f"Spawning LeRobot ACT training: dataset={dataset} steps={steps}")
    handle = train_aloha_remote.spawn(steps) if dataset == "aloha" else train_remote.spawn(steps)
    print(f"  train call: {handle.object_id}")
    if do_eval and dataset == "aloha":
        print("After training completes, eval with:")
        print(f"  modal run deploy/train_act_aloha_modal.py::eval_remote --steps {steps} --episodes {episodes}")
    elif do_eval:
        print("No Modal eval is configured for the pour-water dataset.")
    print("Outputs volume: omx-act-aloha-logs")
