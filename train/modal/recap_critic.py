"""Train the RECAP distributional critic (arXiv:2511.14759) on a Modal GPU.

The critic is far too slow to train locally: a ResNet18 over 480x640 frames plus
a 602-token transformer does not complete a single step in minutes on MPS.

Usage:
    modal run train/modal/recap_critic.py \
        --dataset-repo-id RevanthGundala/031-recap-gen1 \
        --steps 20000

Download results with:
    modal volume get omx-recap-critic-logs / outputs/modal_recap_critic
"""

from __future__ import annotations

from pathlib import Path

import modal

try:
    from modal_common import validate_volume_relative_path
except ModuleNotFoundError:
    from train.modal.common import validate_volume_relative_path


REMOTE_WORKSPACE = Path("/workspace")
REMOTE_OUTPUTS = Path("/outputs")
MODAL_GPU = "A10G"
# Hard ceiling so an unattended run cannot bill indefinitely.
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
        "pandas",
        "pyarrow",
        "safetensors",
        "torchcodec==0.5.0",
    )
    .add_local_dir("models", remote_path=str(REMOTE_WORKSPACE / "models"))
    .add_local_dir("configs", remote_path=str(REMOTE_WORKSPACE / "configs"))
    .add_local_dir("utils", remote_path=str(REMOTE_WORKSPACE / "utils"))
    .add_local_dir("train", remote_path=str(REMOTE_WORKSPACE / "train"))
    .add_local_file("train/modal/common.py", remote_path="/root/modal_common.py")
)

# Warm-start checkpoints are baked into the image rather than uploaded to a
# volume, so a run is reproducible from the local checkout alone.
WARM_START_SOURCES = {
    "warm_start/custom_act_clean48_20k.pt": (
        "outputs/modal_custom_act_clean48_chunk30_20k/step_020000.pt"
    ),
}
for remote_name, local_path in WARM_START_SOURCES.items():
    if Path(local_path).is_file():
        image = image.add_local_file(
            local_path, remote_path=str(REMOTE_WORKSPACE / remote_name)
        )

app = modal.App("omx-recap-critic-training", image=image)
vol = modal.Volume.from_name("omx-recap-critic-logs", create_if_missing=True)


def validate_critic_run(
    *,
    dataset_repo_id: str,
    steps: int,
    save_freq: int,
    val_episodes: int,
    job_name: str,
    freeze: str = "none",
) -> None:
    if freeze not in ("none", "backbone", "encoder", "all"):
        raise ValueError(f"freeze must be one of none/backbone/encoder/all, got {freeze!r}")
    if not dataset_repo_id or "/" not in dataset_repo_id:
        raise ValueError("dataset_repo_id must be a Hugging Face owner/name identifier.")
    if steps <= 0 or save_freq <= 0:
        raise ValueError("steps and save_freq must be positive.")
    if save_freq >= steps:
        raise ValueError(
            "save_freq must be smaller than steps so the final checkpoint is not the first save."
        )
    if val_episodes < 0:
        raise ValueError("val_episodes must be non-negative.")
    if not job_name:
        raise ValueError("job_name must be non-empty.")


@app.function(
    gpu=MODAL_GPU,
    timeout=MODAL_MAX_RUNTIME_SECONDS,
    memory=32_768,
    volumes={REMOTE_OUTPUTS: vol},
    secrets=[hf_secret],
)
def train_critic_remote(
    dataset_repo_id: str,
    steps: int,
    save_freq: int,
    val_episodes: int,
    job_name: str,
    chunk_size: int,
    freeze: str,
    init_from: str | None = None,
    dataset_root: str | None = None,
) -> str:
    import os
    import sys
    import threading

    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["HUGGINGFACE_HUB_TOKEN"] = os.environ.get("HF_TOKEN", "")

    sys.path.insert(0, str(REMOTE_WORKSPACE))
    os.chdir(REMOTE_WORKSPACE)

    import torch

    print("=== CUDA PREFLIGHT ===")
    print("Torch:", torch.__version__)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this Modal container.")
    print("Detected GPU:", torch.cuda.get_device_name(0))
    print("VRAM (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))

    os.environ["RECAP_CRITIC_DATASET_REPO_ID"] = dataset_repo_id
    os.environ["RECAP_CRITIC_STEPS"] = str(steps)
    os.environ["RECAP_CRITIC_SAVE_FREQ"] = str(save_freq)
    os.environ["RECAP_CRITIC_VAL_EPISODES"] = str(val_episodes)
    os.environ["RECAP_CRITIC_JOB_NAME"] = job_name
    os.environ["RECAP_CRITIC_OUTPUT_ROOT"] = str(REMOTE_OUTPUTS)
    os.environ["RECAP_CRITIC_CHUNK_SIZE"] = str(chunk_size)
    os.environ["RECAP_CRITIC_FREEZE"] = freeze
    os.environ["RECAP_DEVICE"] = "cuda"
    vol.reload()
    if dataset_root:
        data_path = REMOTE_OUTPUTS / validate_volume_relative_path(dataset_root)
        if not data_path.exists():
            raise FileNotFoundError(f"dataset_root missing on critic volume: {data_path}")
        os.environ["RECAP_CRITIC_DATASET_ROOT"] = str(data_path)
    if init_from:
        relative = validate_volume_relative_path(init_from)
        volume_ckpt = REMOTE_OUTPUTS / relative
        image_ckpt = REMOTE_WORKSPACE / relative
        if volume_ckpt.is_file():
            os.environ["RECAP_CRITIC_INIT_FROM"] = str(volume_ckpt)
        elif image_ckpt.is_file():
            os.environ["RECAP_CRITIC_INIT_FROM"] = str(image_ckpt)
        else:
            raise FileNotFoundError(
                f"init_from not found on volume or image: {relative}"
            )

    from train.recap.critic import main as train_main

    print("=== STARTING RECAP CRITIC TRAINING ===")
    print(f"dataset={dataset_repo_id} steps={steps} val_episodes={val_episodes}")

    stop_commits = threading.Event()

    def commit_loop() -> None:
        while not stop_commits.wait(VOLUME_COMMIT_INTERVAL_SECONDS):
            print("Committing outputs volume...", flush=True)
            vol.commit()

    commit_thread = threading.Thread(target=commit_loop, daemon=True)
    commit_thread.start()
    try:
        train_main()
    finally:
        stop_commits.set()
        commit_thread.join(timeout=5)
        vol.commit()
    return f"RECAP critic training complete; outputs in volume omx-recap-critic-logs/{job_name}."


@app.function(
    gpu=MODAL_GPU,
    timeout=MODAL_MAX_RUNTIME_SECONDS,
    memory=32_768,
    volumes={REMOTE_OUTPUTS: vol},
    secrets=[hf_secret],
)
def label_remote(
    dataset_repo_id: str,
    dataset_root: str,
    critic_checkpoint: str,
    output_path: str,
) -> str:
    """Score every frame with a trained critic and write advantage_labels.json."""
    import json
    import os
    import sys

    os.environ["PYTHONUNBUFFERED"] = "1"
    sys.path.insert(0, str(REMOTE_WORKSPACE))
    os.chdir(REMOTE_WORKSPACE)

    import torch

    print("=== CUDA PREFLIGHT ===", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; refusing CPU fallback on a paid GPU job.")
    print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)

    vol.reload()
    data_path = REMOTE_OUTPUTS / validate_volume_relative_path(dataset_root)
    ckpt_path = REMOTE_OUTPUTS / validate_volume_relative_path(critic_checkpoint)
    out_path = REMOTE_OUTPUTS / validate_volume_relative_path(output_path)
    if not data_path.exists():
        raise FileNotFoundError(f"dataset_root missing: {data_path}")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"critic checkpoint missing: {ckpt_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        {
            "RECAP_DEVICE": "cuda",
            "RECAP_CRITIC_CHECKPOINT": str(ckpt_path),
            "RECAP_ADVANTAGE_DATASET_REPO_ID": dataset_repo_id,
            "RECAP_CRITIC_DATASET_ROOT": str(data_path),
            "RECAP_ADVANTAGE_OUTPUT": str(out_path),
        }
    )
    print(
        f"label_config dataset={dataset_repo_id} root={data_path} "
        f"critic={ckpt_path} out={out_path}",
        flush=True,
    )
    from train.recap.label_advantage import main as label_main

    label_main()
    vol.commit()
    payload = json.loads(out_path.read_text())
    return json.dumps(
        {
            "status": "complete",
            "output": str(out_path),
            "counts": payload.get("counts"),
            "positive_fraction_non_correction": payload.get(
                "positive_fraction_non_correction"
            ),
            "thresholds_by_task": payload.get("thresholds_by_task"),
        },
        sort_keys=True,
    )


@app.local_entrypoint()
def label(
    dataset_repo_id: str,
    dataset_root: str,
    critic_checkpoint: str,
    output_path: str,
) -> None:
    """Write advantage labels onto the critic volume. Does not train."""
    print("=== RECAP ADVANTAGE LABEL ===")
    print(f"dataset={dataset_repo_id}")
    print(f"dataset_root={dataset_root}")
    print(f"critic={critic_checkpoint}")
    print(f"output={output_path}")
    print(f"MODAL_GPU={MODAL_GPU}")
    result = label_remote.remote(
        dataset_repo_id=dataset_repo_id,
        dataset_root=dataset_root,
        critic_checkpoint=critic_checkpoint,
        output_path=output_path,
    )
    print(result)


@app.function(volumes={REMOTE_OUTPUTS: vol})
def list_outputs() -> list[str]:
    vol.reload()
    return sorted(str(path.relative_to(REMOTE_OUTPUTS)) for path in REMOTE_OUTPUTS.rglob("*.pt"))


@app.local_entrypoint()
def main(
    dataset_repo_id: str = "RevanthGundala/031-recap-gen1",
    steps: int = 20_000,
    save_freq: int = 2_000,
    val_episodes: int = 4,
    job_name: str = "recap_critic",
    chunk_size: int = 30,
    freeze: str = "none",
    init_from: str | None = None,
    dataset_root: str | None = None,
    list_only: bool = False,
) -> None:
    if list_only:
        for path in list_outputs.remote():
            print(path)
        return

    validate_critic_run(
        dataset_repo_id=dataset_repo_id,
        steps=steps,
        save_freq=save_freq,
        val_episodes=val_episodes,
        job_name=job_name,
        freeze=freeze,
    )
    if init_from is not None:
        init_from = validate_volume_relative_path(init_from)
    if dataset_root is not None:
        dataset_root = validate_volume_relative_path(dataset_root)

    print("=== RECAP CRITIC LAUNCH ===")
    print(f"dataset={dataset_repo_id}")
    print(f"dataset_root={dataset_root}")
    print(f"steps={steps} save_freq={save_freq} val_episodes={val_episodes}")
    print(f"init_from={init_from} chunk_size={chunk_size} freeze={freeze}")
    print(f"MODAL_GPU={MODAL_GPU}")
    print(f"MODAL_HARD_TIMEOUT_SECONDS={MODAL_MAX_RUNTIME_SECONDS}")

    result = train_critic_remote.remote(
        dataset_repo_id=dataset_repo_id,
        steps=steps,
        save_freq=save_freq,
        val_episodes=val_episodes,
        job_name=job_name,
        chunk_size=chunk_size,
        freeze=freeze,
        init_from=init_from,
        dataset_root=dataset_root,
    )
    print(result)
    print("Outputs volume: omx-recap-critic-logs")
    print("  modal volume get omx-recap-critic-logs / outputs/modal_recap_critic")
