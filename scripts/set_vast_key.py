"""Prompt for a Vast.ai API key and save it to a local ignored env file.

This avoids putting keys in shell history, command arguments, or chat logs.

Usage:
    uv run python scripts/set_vast_key.py
    source .secrets/vast.env
"""

from __future__ import annotations

from pathlib import Path
import sys


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def main() -> None:
    print("Vast.ai API key: ", end="", flush=True)
    key = sys.stdin.readline().strip()
    if not key:
        raise SystemExit("No key provided.")

    secrets_dir = Path(".secrets")
    secrets_dir.mkdir(exist_ok=True)
    env_path = secrets_dir / "vast.env"
    key_path = secrets_dir / "vast_key"
    env_path.write_text(
        "export VASTAI_API_KEY=" + shell_quote(key) + "\n"
        "export VAST_API_KEY=\"$VASTAI_API_KEY\"\n"
    )
    key_path.write_text(key + "\n")
    env_path.chmod(0o600)
    key_path.chmod(0o600)
    print(f"Saved Vast.ai key to {env_path} and {key_path}.")


if __name__ == "__main__":
    main()
