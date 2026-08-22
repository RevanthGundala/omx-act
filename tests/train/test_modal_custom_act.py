import ast

import pytest

from train.modal.common import (
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


def _recap_entrypoint_source(repo_root) -> str:
    """Source of the RECAP entrypoint.

    Read as text rather than imported because the launcher needs the `modal`
    package, which is not a dependency of the test environment.
    """
    import ast

    path = repo_root / "train" / "modal" / "custom_act.py"
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "train_recap_policy_remote":
            return ast.get_source_segment(path.read_text(), node) or ""
    raise AssertionError("train_recap_policy_remote is missing from the Modal launcher")


def test_token_loss_entrypoint_is_bounded_eval_only(repo_root):
    """CHECK 4 must not be able to bill for a day, and must not train."""
    import ast

    path = repo_root / "train" / "modal" / "custom_act.py"
    source = path.read_text()
    tree = ast.parse(source)
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == "check_token_loss_remote"
    )
    timeouts = [
        keyword.value
        for decorator in node.decorator_list
        if isinstance(decorator, ast.Call)
        for keyword in decorator.keywords
        if keyword.arg == "timeout"
    ]
    assert len(timeouts) == 1
    assert isinstance(timeouts[0], ast.Name)
    assert timeouts[0].id == "MODAL_MAX_RUNTIME_SECONDS"
    assert "refusing CPU fallback" in ast.get_source_segment(source, node)
    assert "run_ablation" in ast.get_source_segment(source, node)
    assert "train_main" not in ast.get_source_segment(source, node)


def test_recap_entrypoint_is_bounded_by_the_hard_runtime(repo_root):
    """An unattended GPU run must not be able to bill for a full day if the
    local client disconnects, so the RECAP function uses the shared hard
    runtime rather than the 24h timeout the older entrypoints carry.
    """
    import ast

    path = repo_root / "train" / "modal" / "custom_act.py"
    source = path.read_text()
    tree = ast.parse(source)
    decorators = [
        decorator
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "train_recap_policy_remote"
        for decorator in node.decorator_list
    ]
    assert decorators, "RECAP entrypoint has no @app.function decorator"
    timeouts = [
        keyword.value
        for decorator in decorators
        if isinstance(decorator, ast.Call)
        for keyword in decorator.keywords
        if keyword.arg == "timeout"
    ]
    assert len(timeouts) == 1
    assert isinstance(timeouts[0], ast.Name)
    assert timeouts[0].id == "MODAL_MAX_RUNTIME_SECONDS"


def test_recap_entrypoint_refuses_cpu_fallback_and_pins_the_gpu_device(repo_root):
    source = _recap_entrypoint_source(repo_root)

    # Paying for a GPU while silently training on CPU is the expensive failure.
    assert "torch.cuda.is_available()" in source
    assert "refusing CPU fallback" in source
    assert '"RECAP_DEVICE": "cuda"' in source
    # Checkpoints must survive teardown, so the volume is committed periodically.
    assert "vol.commit()" in source


def test_recap_entrypoint_defaults_to_the_sft_stage(repo_root):
    source = _recap_entrypoint_source(repo_root)

    # Omitting labels must run SFT with the indicator fixed to POSITIVE rather
    # than silently training against stale advantage labels.
    assert "labels_path: str | None = None" in source
    assert 'RECAP_POLICY_LABELS' in source
    assert 'stage = "improvement_loop" if labels_full_path else "sft_positive"' in source


def test_masked_sft_uses_a_separate_trainer_path(repo_root):
    source = _recap_entrypoint_source(repo_root)

    # The masked experiment must not quietly run through the legacy trainer;
    # they differ in architecture, not just hyperparameters.
    assert "masked_sft: bool = False" in source
    assert "from train.recap.train_masked_sft import main as train_main" in source
    assert 'stage = "masked_sft_positive"' in source


def test_masked_sft_refuses_labels_and_synthetic_shift(repo_root):
    source = _recap_entrypoint_source(repo_root)

    # All-positive SFT with omission is a different question from either the
    # improvement loop or the planted-token capability test.
    assert "masked_sft is all-positive SFT" in source
    assert "synthetic_shift replaces critic labels" in source


@pytest.mark.parametrize(
    "launcher",
    ["train/modal/custom_act.py", "train/modal/recap_critic.py"],
)
def test_every_mounted_path_exists(repo_root, launcher):
    """Regression: the image mounted ``data/`` after that directory was renamed.

    Modal resolves mounts locally before provisioning, so this failed the launch
    outright -- caught only by actually launching, which is the one way of
    finding it that costs money. A stale mount is invisible to import-time
    checks because nothing in the repo references the path.
    """
    source = (repo_root / launcher).read_text()
    tree = ast.parse(source)

    mounted = [
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"add_local_dir", "add_local_file"}
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    ]
    assert mounted, f"no mounts found in {launcher}; the parser is wrong"

    missing = [path for path in mounted if not (repo_root / path).exists()]
    assert missing == [], f"{launcher} mounts paths that do not exist: {missing}"
