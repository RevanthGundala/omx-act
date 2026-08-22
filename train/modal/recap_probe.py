"""CHECK 2: probe whether advantage label I is readable from o on Modal.

Does not retrain ACT. Freezes the trained critic encoder and fits a tiny
head, plus a qpos-only baseline. Intervention frames are dropped from
metrics because labeling forced them POSITIVE.

Usage (parent launches under cloud-training-safety-monitor; do not run unpaid
from an agent session):

    modal run --detach train/modal/recap_probe.py \\
      --dataset-repo-id RevanthGundala/034-recap-balanced \\
      --labels-path act_experiments/recap_labels/034-recap-balanced.json \\
      --critic-checkpoint recap_critic_balanced66_20k/best_critic.pt \\
      --job-name recap_probe_i_from_o \\
      --val-episodes 8 \\
      --steps 2000

App: omx-recap-advantage-probe
GPU: A10G
Hard timeout: 3h
Volumes: omx-custom-act-training-logs (/outputs), omx-recap-critic-logs (/critic)
"""

from __future__ import annotations

from pathlib import Path
import uuid

import modal

try:
    from modal_common import validate_probe_run, validate_volume_relative_path
except ModuleNotFoundError:
    from train.modal.common import validate_probe_run, validate_volume_relative_path


REMOTE_WORKSPACE = Path("/workspace")
REMOTE_OUTPUTS = Path("/outputs")
REMOTE_CRITIC = Path("/critic")
MODAL_GPU = "A10G"
MODAL_MAX_RUNTIME_SECONDS = 3 * 60 * 60
VOLUME_COMMIT_INTERVAL_SECONDS = 300
LEROBOT_GIT_REF = "87242cfced5228b6181fb0047021d684d424df38"
APP_NAME = "omx-recap-advantage-probe"

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

app = modal.App(APP_NAME, image=image)
train_vol = modal.Volume.from_name("omx-custom-act-training-logs", create_if_missing=True)
critic_vol = modal.Volume.from_name("omx-recap-critic-logs", create_if_missing=True)


def resolve_existing_file(*candidates: Path) -> Path:
    for path in candidates:
        if path.is_file():
            return path
    nearby: list[str] = []
    seen: set[Path] = set()
    for path in candidates:
        root = path if path.is_dir() else path.parent
        if root in seen or not root.exists():
            continue
        seen.add(root)
        nearby.extend(str(hit) for hit in sorted(root.rglob("*.pt"))[:12])
        nearby.extend(str(hit) for hit in sorted(root.rglob("*.json"))[:8])
    tried = ", ".join(str(path) for path in candidates)
    hint = f" Nearby files: {nearby}" if nearby else ""
    raise FileNotFoundError(f"None of the candidate paths exist: {tried}.{hint}")


@app.function(
    gpu=MODAL_GPU,
    timeout=MODAL_MAX_RUNTIME_SECONDS,
    memory=32_768,
    volumes={REMOTE_OUTPUTS: train_vol, REMOTE_CRITIC: critic_vol},
    secrets=[hf_secret],
)
def probe_remote(
    dataset_repo_id: str,
    labels_path: str,
    job_name: str,
    steps: int,
    val_episodes: int,
    critic_checkpoint: str | None = None,
    backend: str = "auto",
) -> str:
    import json
    import os
    import sys
    import threading

    validate_probe_run(
        dataset_repo_id=dataset_repo_id,
        steps=steps,
        val_episodes=val_episodes,
        job_name=job_name,
        labels_path=labels_path,
        critic_checkpoint=critic_checkpoint,
        backend=backend,
    )

    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["HUGGINGFACE_HUB_TOKEN"] = os.environ.get("HF_TOKEN", "")
    sys.path.insert(0, str(REMOTE_WORKSPACE))
    os.chdir(REMOTE_WORKSPACE)

    import torch

    print("=== CUDA PREFLIGHT ===", flush=True)
    print(f"Torch: {torch.__version__}", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; refusing CPU fallback on a paid GPU job.")
    print(f"Detected GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(
        f"VRAM (GB): {round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)}",
        flush=True,
    )

    train_vol.reload()
    critic_vol.reload()
    labels_full = resolve_existing_file(REMOTE_OUTPUTS / validate_volume_relative_path(labels_path))
    critic_full = ""
    if critic_checkpoint:
        relative = validate_volume_relative_path(critic_checkpoint)
        critic_full = str(
            resolve_existing_file(REMOTE_CRITIC / relative, REMOTE_OUTPUTS / relative)
        )

    os.environ.update(
        {
            "RECAP_DEVICE": "cuda",
            "RECAP_PROBE_DATASET_REPO_ID": dataset_repo_id,
            "RECAP_PROBE_LABELS": str(labels_full),
            "RECAP_PROBE_JOB_NAME": job_name,
            "RECAP_PROBE_OUTPUT_ROOT": str(REMOTE_OUTPUTS / "act_experiments"),
            "RECAP_PROBE_STEPS": str(steps),
            "RECAP_PROBE_VAL_EPISODES": str(val_episodes),
            "RECAP_PROBE_BACKEND": backend,
        }
    )
    if critic_full:
        os.environ["RECAP_PROBE_CRITIC_CHECKPOINT"] = critic_full

    print(
        "probe_config "
        f"dataset={dataset_repo_id} labels={labels_full} critic={critic_full or None} "
        f"backend={backend} steps={steps} val_episodes={val_episodes} job_name={job_name}",
        flush=True,
    )

    stop_commits = threading.Event()

    def commit_loop() -> None:
        while not stop_commits.wait(VOLUME_COMMIT_INTERVAL_SECONDS):
            print("Committing outputs volume...", flush=True)
            train_vol.commit()

    commit_thread = threading.Thread(target=commit_loop, daemon=True)
    commit_thread.start()
    try:
        from utils.analysis.probe_advantage import main as probe_main

        result = probe_main()
    finally:
        stop_commits.set()
        commit_thread.join(timeout=5)
        train_vol.commit()
    return json.dumps(
        {
            "status": "complete",
            "app": APP_NAME,
            "job_name": job_name,
            "output_root": str(REMOTE_OUTPUTS / "act_experiments" / job_name),
            "probes": result.get("probes"),
        },
        sort_keys=True,
        default=str,
    )


@app.local_entrypoint()
def main(
    dataset_repo_id: str = "RevanthGundala/034-recap-balanced",
    labels_path: str = "act_experiments/recap_labels/034-recap-balanced.json",
    critic_checkpoint: str | None = "recap_critic_balanced66_20k/best_critic.pt",
    job_name: str = "recap_probe_i_from_o",
    steps: int = 2000,
    val_episodes: int = 8,
    backend: str = "auto",
    preflight_only: bool = False,
) -> None:
    validate_probe_run(
        dataset_repo_id=dataset_repo_id,
        steps=steps,
        val_episodes=val_episodes,
        job_name=job_name,
        labels_path=labels_path,
        critic_checkpoint=critic_checkpoint,
        backend=backend,
    )
    print("=== RECAP ADVANTAGE PROBE LAUNCH ===")
    print(f"MODAL_APP_NAME={APP_NAME}")
    print(f"dataset={dataset_repo_id}")
    print(f"labels={labels_path}")
    print(f"critic_checkpoint={critic_checkpoint}")
    print(f"backend={backend} steps={steps} val_episodes={val_episodes}")
    print(f"job_name={job_name}")
    print(f"MODAL_GPU={MODAL_GPU}")
    print(f"MODAL_HARD_TIMEOUT_SECONDS={MODAL_MAX_RUNTIME_SECONDS}")
    print("MODAL_OUTPUT_VOLUME=omx-custom-act-training-logs")
    print("MODAL_CRITIC_VOLUME=omx-recap-critic-logs")
    if preflight_only:
        print("Preflight only; not launching Modal GPU job.")
        return

    run_token = uuid.uuid4().hex[:8]
    function_call = probe_remote.spawn(
        dataset_repo_id=dataset_repo_id,
        labels_path=labels_path,
        job_name=job_name,
        steps=steps,
        val_episodes=val_episodes,
        critic_checkpoint=critic_checkpoint,
        backend=backend,
    )
    print(f"MODAL_FUNCTION_CALL_ID={function_call.object_id}")
    print(f"MODAL_RUN_TOKEN={run_token}")
    print(f"MODAL_JOB_NAME={job_name}")
    print("MODAL_OUTPUT_PATH=act_experiments/" + job_name)


@app.function(
    gpu=MODAL_GPU,
    timeout=MODAL_MAX_RUNTIME_SECONDS,
    memory=32_768,
    volumes={REMOTE_OUTPUTS: train_vol, REMOTE_CRITIC: critic_vol},
    secrets=[hf_secret],
)
def action_mi_remote(
    dataset_repo_id: str,
    labels_path: str,
    critic_checkpoint: str,
    job_name: str,
    k: int = 40,
    n_perm: int = 200,
) -> str:
    """Encode o with the frozen critic and test I(a ; I | o). Does not train."""
    import json
    import os
    import sys

    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["HUGGINGFACE_HUB_TOKEN"] = os.environ.get("HF_TOKEN", "")
    sys.path.insert(0, str(REMOTE_WORKSPACE))
    os.chdir(REMOTE_WORKSPACE)

    import torch

    print("=== CUDA PREFLIGHT ===", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; refusing CPU fallback on a paid GPU job.")
    print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)
    print(
        f"vram_gb={torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}",
        flush=True,
    )

    train_vol.reload()
    critic_vol.reload()
    labels_full = resolve_existing_file(REMOTE_OUTPUTS / validate_volume_relative_path(labels_path))
    critic_full = resolve_existing_file(
        REMOTE_CRITIC / validate_volume_relative_path(critic_checkpoint),
        REMOTE_OUTPUTS / validate_volume_relative_path(critic_checkpoint),
    )
    os.environ["RECAP_DEVICE"] = "cuda"
    from datetime import datetime, timezone

    from utils.analysis.action_label_mi import format_report, run_check

    print(
        "action_mi_config "
        f"dataset={dataset_repo_id} labels={labels_full} critic={critic_full} "
        f"k={k} n_perm={n_perm}",
        flush=True,
    )
    report = run_check(
        dataset_repo_id=dataset_repo_id,
        labels_path=labels_full,
        critic_checkpoint=critic_full,
        device=torch.device("cuda"),
        k=k,
        n_perm=n_perm,
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_dir = REMOTE_OUTPUTS / "act_experiments" / job_name / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(format_report(report), flush=True)
    print(f"wrote {report_path}", flush=True)
    train_vol.commit()
    critic_row = report.get("all_non_intervention", {}).get("critic", {})
    return json.dumps(
        {
            "status": "complete",
            "report_path": str(report_path),
            "critic_lift": critic_row.get("observed_lift"),
            "critic_p": critic_row.get("p"),
            "critic_z": critic_row.get("z"),
        },
        sort_keys=True,
    )


@app.local_entrypoint()
def action_mi(
    dataset_repo_id: str = "RevanthGundala/034-recap-balanced",
    labels_path: str = "act_experiments/recap_labels/034-recap-balanced.json",
    critic_checkpoint: str = "recap_critic_balanced66_20k/20260814-011745/checkpoints/step_020000.pt",
    job_name: str = "recap_action_label_mi",
    k: int = 40,
    n_perm: int = 200,
) -> None:
    """Launch the I(a ; I | o) matched-neighbour check. Does not train."""
    validate_volume_relative_path(labels_path)
    validate_volume_relative_path(critic_checkpoint)
    if not dataset_repo_id or "/" not in dataset_repo_id:
        raise ValueError("dataset_repo_id must be a Hugging Face owner/name identifier.")
    print("=== RECAP ACTION-LABEL MI ===")
    print(f"dataset={dataset_repo_id}")
    print(f"labels={labels_path}")
    print(f"critic={critic_checkpoint}")
    print(f"k={k} n_perm={n_perm} job_name={job_name}")
    print(f"MODAL_GPU={MODAL_GPU}")
    print(f"MODAL_HARD_TIMEOUT_SECONDS={MODAL_MAX_RUNTIME_SECONDS}")
    function_call = action_mi_remote.spawn(
        dataset_repo_id=dataset_repo_id,
        labels_path=labels_path,
        critic_checkpoint=critic_checkpoint,
        job_name=job_name,
        k=k,
        n_perm=n_perm,
    )
    print(f"MODAL_FUNCTION_CALL_ID={function_call.object_id}")
    print("MODAL_OUTPUT_PATH=act_experiments/" + job_name)

