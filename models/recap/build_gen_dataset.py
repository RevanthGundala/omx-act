"""Assemble a RECAP training set from demonstrations, rollouts, and corrections.

RECAP finetunes the value function on "all of the data collected for the task so
far" (arXiv:2511.14759 Sec. V-D). Those three kinds of data live in two storage
formats here, so this composes them into one dataset with the reward, return,
and intervention columns the critic and the advantage labeler need.

Episodes are decoded one at a time. Loading every episode first would cost tens
of gigabytes at this size, which is the failure the rollout builder already hit.

Example:
    uv run python -m models.recap.build_gen_dataset \
      --demo-dataset ~/.cache/huggingface/lerobot/026-place-sponge-on-hot-pad-clean \
      --rollout-root outputs/eval_rollouts_recap_positive/act_eval_async_20260813-104729 \
      --correction-dataset ~/.cache/huggingface/lerobot/rollout_032-recap-gen2-dagger \
      --target-repo RevanthGundala/033-recap-gen2 \
      --output-root outputs/datasets/033-recap-gen2
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil

from models.recap.build_dataset import (
    _lerobot_specs,
    _trial_spec,
    build_mixed_dataset,
    discover_trial_dirs,
)


def parse_episode_list(spec: str | None) -> set[int] | None:
    """Parse ``0,3,6`` into a set of episode indices."""
    if spec is None or spec.strip() == "":
        return None
    values = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        values.add(int(part))
    if not values:
        raise ValueError(f"empty episode list: {spec!r}")
    return values


def recap_dataset_specs(root: Path) -> list:
    """Load an existing RECAP mix, keeping source_kind and success from the manifest.

    Passing 034 as ``--demo-dataset`` would mark every episode a demonstration
    success, including the 6 failures. The manifest is the source of truth.
    """
    root = Path(root).expanduser().resolve()
    manifest_path = root / "recap_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"{root} has no recap_manifest.json; pass a RECAP mix, not a raw LeRobot set"
        )
    manifest = json.loads(manifest_path.read_text())
    by_episode = {int(episode["episode_index"]): episode for episode in manifest["episodes"]}
    specs = _lerobot_specs(root, kind="recap", default_success=True)
    out = []
    for spec in specs:
        episode_index = int(spec.source_id.rsplit("episode-", 1)[1])
        if episode_index not in by_episode:
            raise ValueError(
                f"{spec.source_id} is not in {manifest_path}"
            )
        info = by_episode[episode_index]
        kind = info["source_kind"]
        success = info["outcome_label"] == "success"
        out.append(replace(spec, source_kind=kind, success=success))
    return out


def collect_specs(args: argparse.Namespace) -> list:
    excluded = {
        Path(path).expanduser().resolve() for path in (args.exclude_trial or [])
    }
    specs = []
    for root in args.recap_dataset or []:
        specs.extend(recap_dataset_specs(root))
    for root in args.demo_dataset or []:
        # Demonstrations carry no outcome sidecar: they are expert successes by
        # construction, which is what RECAP pre-trains the value function on.
        specs.extend(
            _lerobot_specs(root, kind="demonstration", default_success=True)
        )
    for root in args.rollout_root or []:
        for trial_dir in discover_trial_dirs([root]):
            if trial_dir.resolve() in excluded:
                continue
            specs.append(_trial_spec(trial_dir))
    for trial_dir in args.rollout_trial or []:
        trial_dir = Path(trial_dir).expanduser().resolve()
        if not (trial_dir / "metadata.json").is_file():
            raise FileNotFoundError(f"Not a recorded trial directory: {trial_dir}")
        specs.append(_trial_spec(trial_dir))
    allowed = parse_episode_list(getattr(args, "correction_episodes", None))
    for root in args.correction_dataset or []:
        # A correction episode is labeled by the operator, so a missing sidecar
        # is an error rather than an assumed success.
        group = _lerobot_specs(root, kind="correction", default_success=None)
        if allowed is not None:
            keep = []
            for spec in group:
                episode_index = int(spec.source_id.rsplit("episode-", 1)[1])
                if episode_index in allowed:
                    keep.append(spec)
            missing = allowed - {
                int(spec.source_id.rsplit("episode-", 1)[1]) for spec in keep
            }
            if missing:
                raise ValueError(
                    f"{root} is missing correction episodes {sorted(missing)}"
                )
            group = keep
        specs.extend(group)
    if not specs:
        raise ValueError("No episodes were found in any source")
    seen = set()
    for spec in specs:
        if spec.source_id in seen:
            raise ValueError(f"Episode included twice: {spec.source_id}")
        seen.add(spec.source_id)
    return specs


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demo-dataset",
        type=Path,
        action="append",
        help="LeRobot dataset of demonstrations. Every episode counts as a success.",
    )
    parser.add_argument(
        "--rollout-root",
        type=Path,
        action="append",
        help="Directory of recorded autonomous evaluation trials.",
    )
    parser.add_argument(
        "--rollout-trial",
        type=Path,
        action="append",
        help=(
            "One recorded trial directory to include. Use this to add specific "
            "episodes, such as failures from an earlier evaluation, without "
            "pulling in the whole session."
        ),
    )
    parser.add_argument(
        "--exclude-trial",
        type=Path,
        action="append",
        help=(
            "Trial directory to drop from a --rollout-root. Lets a set stay at "
            "a fixed episode budget while its composition changes."
        ),
    )
    parser.add_argument(
        "--recap-dataset",
        type=Path,
        action="append",
        help=(
            "Existing RECAP mix (has recap_manifest.json). Preserves source_kind "
            "and success; do not pass 034 as --demo-dataset."
        ),
    )
    parser.add_argument(
        "--correction-dataset",
        type=Path,
        action="append",
        help=(
            "LeRobot dataset of DAgger rollouts with outcome sidecars and "
            "per-frame intervention flags."
        ),
    )
    parser.add_argument(
        "--correction-episodes",
        default=None,
        help="Comma-separated episode indices to keep from --correction-dataset (e.g. 0,3,6).",
    )
    parser.add_argument("--target-repo", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--failure-penalty", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-videos", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the composition without writing a dataset.",
    )
    return parser.parse_args(argv)


def summarize(specs: list) -> None:
    print(f"\n{'kind':<20} {'episodes':>9} {'frames':>9} {'success':>8} {'failure':>8}")
    for kind in sorted({spec.source_kind for spec in specs}):
        group = [spec for spec in specs if spec.source_kind == kind]
        successes = sum(1 for spec in group if spec.success)
        print(
            f"{kind:<20} {len(group):>9} {sum(s.length for s in group):>9} "
            f"{successes:>8} {len(group) - successes:>8}"
        )
    successes = sum(1 for spec in specs if spec.success)
    print(
        f"{'TOTAL':<20} {len(specs):>9} {sum(s.length for s in specs):>9} "
        f"{successes:>8} {len(specs) - successes:>8}"
    )
    print(f"\nlongest episode: {max(spec.length for spec in specs)} steps")


def main(argv=None) -> int:
    args = parse_args(argv)
    specs = collect_specs(args)
    summarize(specs)
    if args.dry_run:
        print("\nDry run: nothing written.")
        return 0

    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        if not args.overwrite:
            raise SystemExit(
                f"Refusing to overwrite {output_root}; pass --overwrite to replace it"
            )
        shutil.rmtree(output_root)

    print(f"\nWriting {len(specs)} episodes to {output_root}")
    _, manifest = build_mixed_dataset(
        specs=specs,
        target_repo=args.target_repo,
        output_root=output_root,
        failure_penalty=args.failure_penalty,
        use_videos=not args.no_videos,
    )
    print(
        f"\nDone: {manifest['episode_count']} episodes, "
        f"{manifest['success_count']} success / {manifest['failure_count']} failure, "
        f"C_fail={manifest['failure_penalty']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
