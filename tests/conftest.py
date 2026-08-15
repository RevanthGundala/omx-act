"""Shared pytest fixtures.

Test files live one directory deeper than they used to, so anything that needs
the repository root must not count parent directories. Ask for ``repo_root``
instead.
"""

from pathlib import Path

import pytest


def _find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("Could not locate the repository root from tests/conftest.py")


REPO_ROOT = _find_repo_root()


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path to the repository root, independent of test file depth."""
    return REPO_ROOT
