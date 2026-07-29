import pytest

from deploy.modal_training_common import (
    validate_dataset_run,
    validate_pretrained_checkpoint_path,
    validate_volume_relative_path,
)


def test_validate_dataset_run_accepts_periodic_checkpoints():
    validate_dataset_run(
        dataset_repo_id="owner/dataset",
        steps=20_000,
        chunk_size=30,
        save_freq=2_000,
        train_split=1.0,
        job_name="parity-run",
    )


def test_validate_dataset_run_rejects_final_only_checkpoint():
    with pytest.raises(ValueError, match="final checkpoint"):
        validate_dataset_run(
            dataset_repo_id="owner/dataset",
            steps=20_000,
            chunk_size=30,
            save_freq=20_000,
            train_split=1.0,
            job_name="parity-run",
        )


def test_validate_pretrained_checkpoint_path_accepts_volume_checkpoint():
    assert validate_pretrained_checkpoint_path(
        "act_experiments/base/run/checkpoints/last.pt"
    ) == "act_experiments/base/run/checkpoints/last.pt"


def test_validate_pretrained_checkpoint_path_accepts_resume_checkpoint():
    assert validate_pretrained_checkpoint_path(
        "act_experiments/base/run/checkpoints/step_020000.pt"
    ) == "act_experiments/base/run/checkpoints/step_020000.pt"


def test_validate_volume_relative_path_accepts_scoped_path():
    assert validate_volume_relative_path(
        "resume_sources/lerobot/checkpoint",
        expected_prefix="resume_sources",
    ) == "resume_sources/lerobot/checkpoint"


@pytest.mark.parametrize(
    "path",
    ["/resume_sources/checkpoint", "../resume_sources/checkpoint", ""],
)
def test_validate_volume_relative_path_rejects_unsafe_path(path):
    with pytest.raises(ValueError, match="relative"):
        validate_volume_relative_path(path)


@pytest.mark.parametrize(
    "path",
    [
        "/outputs/act_experiments/base/last.pt",
        "../act_experiments/base/last.pt",
        "other/base/last.pt",
        "act_experiments/base/model.safetensors",
    ],
)
def test_validate_pretrained_checkpoint_path_rejects_unsafe_paths(path):
    with pytest.raises(ValueError, match="safe .pt path"):
        validate_pretrained_checkpoint_path(path)
