"""Destroy one Vast.ai instance after a hard billing-safety timeout."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from vastai_sdk import VastAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deploy.train_act_vastai import read_local_vast_key


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance-id", type=int, required=True)
    parser.add_argument("--max-runtime-minutes", type=int, required=True)
    args = parser.parse_args()

    time.sleep(args.max_runtime_minutes * 60)
    api_key = (
        os.environ.get("VASTAI_API_KEY")
        or os.environ.get("VAST_API_KEY")
        or read_local_vast_key()
    )
    if not api_key:
        raise RuntimeError("Cannot enforce Vast timeout: API key is unavailable.")

    print(
        f"Hard runtime limit reached; destroying Vast instance {args.instance_id}.",
        flush=True,
    )
    vast = VastAI(api_key=api_key)
    last_error = None
    for attempt in range(1, 6):
        try:
            vast.destroy_instance(id=args.instance_id)
            print(f"Destroyed Vast instance {args.instance_id}.", flush=True)
            return
        except Exception as exc:
            last_error = exc
            print(f"Destroy attempt {attempt}/5 failed: {exc}", flush=True)
            time.sleep(30)
    raise RuntimeError(
        f"Failed to destroy Vast instance {args.instance_id}: {last_error}"
    )


if __name__ == "__main__":
    main()
