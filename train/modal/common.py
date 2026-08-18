"""Pure validation helpers shared by Modal training launchers and tests."""

from __future__ import annotations

from pathlib import PurePosixPath


def validate_dataset_run(
    *,
    dataset_repo_id: str,
    steps: int,
    chunk_size: int,
    save_freq: int,
    train_split: float,
    job_name: str,
) -> None:
    if not dataset_repo_id or "/" not in dataset_repo_id:
        raise ValueError("dataset_repo_id must be a Hugging Face owner/name identifier.")
    if steps <= 0 or chunk_size <= 0 or save_freq <= 0:
        raise ValueError("steps, chunk_size, and save_freq must be positive.")
    if save_freq >= steps:
        raise ValueError("save_freq must be smaller than steps so the final checkpoint is not the first save.")
    if not 0 < train_split <= 1:
        raise ValueError("train_split must be in (0, 1].")
    if not job_name:
        raise ValueError("job_name must be non-empty.")


def validate_pretrained_checkpoint_path(value: str | None) -> str | None:
    if value is None:
        return None
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or path.parts[0] != "act_experiments"
        or ".." in path.parts
        or path.suffix != ".pt"
    ):
        raise ValueError(
            "pretrained_checkpoint must be a safe .pt path under act_experiments/ in the Modal volume."
        )
    return str(path)


def validate_probe_run(
    *,
    dataset_repo_id: str,
    steps: int,
    val_episodes: int,
    job_name: str,
    labels_path: str | None = None,
    critic_checkpoint: str | None = None,
    backend: str = "auto",
) -> None:
    """Refuse a probe that cannot answer the redundancy question.

    A full-dataset fit (val_episodes=0 / train_split=1.0) would leak I through
    near-duplicate neighbouring frames and make a high AUC meaningless.
    """
    if backend not in ("auto", "critic", "tiny"):
        raise ValueError(f"backend must be one of auto/critic/tiny, got {backend!r}")
    if not dataset_repo_id or "/" not in dataset_repo_id:
        raise ValueError("dataset_repo_id must be a Hugging Face owner/name identifier.")
    if steps <= 0:
        raise ValueError("steps must be positive.")
    if val_episodes <= 0:
        raise ValueError(
            "Held-out episodes are required; never train_split=1.0 for this probe."
        )
    if not job_name:
        raise ValueError("job_name must be non-empty.")
    if backend == "critic" and not critic_checkpoint:
        raise ValueError("backend='critic' requires critic_checkpoint.")
    if labels_path is not None:
        validate_volume_relative_path(labels_path)
    if critic_checkpoint is not None:
        validate_volume_relative_path(critic_checkpoint)


def validate_volume_relative_path(
    value: str,
    *,
    expected_prefix: str | None = None,
) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError("Modal volume paths must be non-empty, relative, and traversal-free.")
    if expected_prefix is not None and path.parts[0] != expected_prefix:
        raise ValueError(f"Modal volume path must be under {expected_prefix}/.")
    return str(path)
