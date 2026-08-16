from types import SimpleNamespace

import pytest

from train.vastai.act import (
    _validated_checkpoint_path,
    _version_tuple,
    assert_no_managed_instances,
    remote_training_command,
)


def test_remote_training_command_exports_train_split(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "fake-token")
    args = SimpleNamespace(
        runner="custom",
        chunk_size=30,
        steps=20_000,
        save_freq=2_000,
        job_name="custom-all50",
        dataset_repo_id="owner/dataset",
        train_split=1.0,
        remote_resume_checkpoint="",
        remote_workspace="/workspace/act-sim",
    )

    command = remote_training_command(args)

    assert "export ACT_TRAIN_SPLIT=\"1.0\"" in command
    assert 'export ACT_TRAIN_JOB_NAME="custom-all50"' in command


@pytest.mark.parametrize(
    "value",
    [
        "",
        "/workspace/outputs/checkpoint.pt",
        "../outputs/checkpoint.pt",
        "outputs/../checkpoint.pt",
        "other/checkpoint.pt",
        "outputs/checkpoint.bin",
    ],
)
def test_validated_checkpoint_path_rejects_unexpected_paths(value):
    with pytest.raises(ValueError):
        _validated_checkpoint_path(value)


def test_validated_checkpoint_path_accepts_training_output():
    value = "outputs/act_experiments/job/run/checkpoints/step_020000.pt"

    assert str(_validated_checkpoint_path(value)) == value


def test_version_tuple_orders_cuda_versions():
    assert _version_tuple("12.8") > _version_tuple("12.4")
    assert _version_tuple("13.0") > _version_tuple("12.8")
    assert _version_tuple(None) == ()


class _FakeVast:
    def show_instances(self):
        return [
            {
                "id": 123,
                "label": "act-sim-unrelated",
                "actual_status": "running",
            }
        ]


def test_account_audit_allows_only_explicit_instance_ids():
    assert_no_managed_instances(_FakeVast(), [123])

    with pytest.raises(RuntimeError, match="existing act-sim Vast instances"):
        assert_no_managed_instances(_FakeVast(), [])
