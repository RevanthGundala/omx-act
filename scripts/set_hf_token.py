"""Prompt for a Hugging Face token and save it to the local HF cache.

This avoids putting tokens in shell history, command arguments, or chat logs.

Usage:
    uv run python scripts/set_hf_token.py
"""

from __future__ import annotations

from getpass import getpass

from huggingface_hub import HfApi, login


def main() -> None:
    token = getpass("Hugging Face token: ").strip()
    if not token:
        raise SystemExit("No token provided.")

    api = HfApi(token=token)
    user = api.whoami()
    login(token=token, add_to_git_credential=False)
    print(f"Saved Hugging Face token for {user.get('name') or user.get('fullname') or 'authenticated user'}.")


if __name__ == "__main__":
    main()
