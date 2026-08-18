import ast

from train.modal.common import validate_probe_run


def test_validate_probe_run_refuses_no_holdout():
    import pytest

    with pytest.raises(ValueError, match="never train_split=1.0"):
        validate_probe_run(
            dataset_repo_id="RevanthGundala/034-recap-balanced",
            steps=2000,
            val_episodes=0,
            job_name="recap_probe_i_from_o",
        )


def test_probe_launcher_mounts_exist(repo_root):
    source = (repo_root / "train" / "modal" / "recap_probe.py").read_text()
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
    assert mounted
    missing = [path for path in mounted if not (repo_root / path).exists()]
    assert missing == []


def test_probe_remote_is_bounded_gpu_only_and_commits(repo_root):
    path = repo_root / "train" / "modal" / "recap_probe.py"
    source = path.read_text()
    tree = ast.parse(source)
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == "probe_remote"
    )
    body = ast.get_source_segment(source, node) or ""
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
    assert "omx-recap-advantage-probe" in source
    assert "refusing CPU fallback" in body
    assert '"RECAP_DEVICE": "cuda"' in body
    assert "train_vol.commit()" in body
    assert "val_episodes" in body
