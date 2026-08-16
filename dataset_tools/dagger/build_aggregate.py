"""Merge a seed demonstration set with DAgger corrections into one dataset.

Produces a fresh local LeRobot v3.0 dataset. Every seed frame is tagged
``intervention=False`` and every correction frame ``intervention=True``, so
provenance survives into the aggregate -- RECAP advantage labelling depends on
that column to force corrections positive (Sec. IV-B).

Source datasets are required arguments. They previously defaulted to the
014 -> 017 "place sponge in bowl" pair; that task was abandoned for the hot-pad
task, and 017 is the dataset whose late motion onset caused the original ACT
failure, so a no-argument run rebuilt precisely the wrong thing. This builder
most recently produced ``028-custom48-dagger18``.

Why this does not reuse ``dataset_tools.trimming.trim_dataset``: that helper is
Hub-only, single-source, and always drops the ``intervention`` column, none of
which fit combining two datasets with per-frame provenance. It also cannot read
a corrections cache that was never pushed to the Hub.

Trim bounds are computed from the parquet state/action columns with pandas, so
video is decoded only for frames that are actually kept, exactly once each.
``dataset_tools.trimming.trim_dataset`` decodes every frame to find the bounds and then decodes
the kept frames again.

Usage (local build; never pushes by default):
    uv run python -m dataset_tools.dagger.build_aggregate \\
        --seed-repo-id RevanthGundala/026-place-sponge-on-hot-pad-clean \\
        --corrections-repo-id rollout_027-custom48-dagger \\
        --corrections-root ~/.cache/huggingface/lerobot/rollout_027-custom48-dagger \\
        --output-repo-id RevanthGundala/028-custom48-dagger18 \\
        --output-dir outputs/datasets/028-custom48-dagger18

Add ``--push`` only once a Hub push has been explicitly approved.
"""
from __future__ import annotations

import argparse
import glob
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from dataset_tools.dagger.validators import (
    select_correction_episodes,
    validate_action_maxima_within_seed,
    validate_contiguous_episode_indices,
    validate_exact_fps,
    validate_finite,
    validate_frame_correspondence,
    validate_no_unexpected_idle_edges,
)
from dataset_tools.trimming.noop_trim import find_noop_trim_bounds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # Datasets are required rather than defaulted. These once defaulted to the
    # 014 -> 017 "place sponge in bowl" pair; that task was abandoned, and 017
    # is the dataset whose late motion onset caused the original ACT failure.
    # A no-argument run would have quietly rebuilt it.
    parser.add_argument("--seed-repo-id", required=True)
    parser.add_argument(
        "--seed-episodes",
        default=None,
        help=(
            "Optional comma-separated source seed episode indices. "
            "By default all seed episodes are included."
        ),
    )
    parser.add_argument("--corrections-repo-id", required=True)
    parser.add_argument("--corrections-root", type=Path, required=True)
    parser.add_argument("--output-dir", required=True, type=Path, help="Fresh local target directory.")
    parser.add_argument("--output-repo-id", required=True)
    parser.add_argument("--noop-threshold-deg", type=float, default=0.1)
    parser.add_argument("--noop-margin-frames", type=int, default=4)
    seed_edges = parser.add_mutually_exclusive_group()
    seed_edges.add_argument(
        "--preserve-seed-idle-edges",
        action="store_true",
        help=(
            "Keep seed episodes byte-for-byte even when they retain more idle "
            "edge frames than --noop-margin-frames. Corrections are still trimmed "
            "and validated."
        ),
    )
    seed_edges.add_argument(
        "--trim-seed-idle-edges",
        action="store_true",
        help=(
            "Trim leading and trailing no-op runs from seed episodes using the "
            "same threshold and margin as corrections."
        ),
    )
    parser.add_argument(
        "--action-range-tolerance",
        type=float,
        default=1.0,
        help=(
            "How far correction actions may exceed the seed action envelope, in "
            "normalized joint units. The default 1.0 targets a data-corruption "
            "signature. Genuine DAgger corrections recover from states the demos "
            "never visited, so a deliberate widening (with the overshoot audited "
            "first) is expected; it is recorded in the provenance report."
        ),
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push the built dataset to the Hub. OFF by default -- this builder is local-only "
        "unless explicitly asked to push.",
    )
    return parser.parse_args()


def _load_frame_table(root: Path) -> pd.DataFrame:
    """Load the full per-frame parquet table (all chunks/files) for a local
    LeRobot v3.0 dataset root, sorted by the global `index` column (which
    equals row position for these datasets)."""
    files = sorted(glob.glob(str(root / "data" / "chunk-*" / "file-*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet data files found under {root}/data/")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.sort_values("index").reset_index(drop=True)


@dataclass
class EpisodePlan:
    source_label: str  # "seed" or "correction"
    source_episode_index: int  # episode_index within the SOURCE dataset
    start: int  # kept-frame start, Python slice semantics, into the source episode's own frames
    stop: int  # kept-frame stop
    global_indices: np.ndarray  # source dataset global `index` values for kept frames, in order
    intervention: bool
    trim_lead: int = field(init=False)
    trim_trail: int = field(init=False)

    def __post_init__(self) -> None:
        self.trim_lead = self.start
        self.trim_trail = -1  # filled in by caller once total frame count is known


def _to_bool(value) -> bool:
    return bool(value[0]) if hasattr(value, "__len__") else bool(value)


def _intervention_correction_candidates(corrections_df: pd.DataFrame) -> list[int]:
    """Return the episode indices in `corrections_df` that are genuine DAgger
    corrections (`intervention=True` for every frame).

    Some dagger caches (e.g. the older teleop-collected 013 dagger source)
    embed their own copy of the seed episodes (`intervention=False`) at the
    same episode-index range as the real seed data -- those must NOT be
    treated as correction candidates just because they share a dataset with
    them. Using the `intervention` flag (rather than a hardcoded episode-number
    range) to pick candidates keeps this correct regardless of the source
    cache's episode numbering, and is a no-op safety check for sources (like
    016) where every episode is already correction-only.
    """
    flagged = corrections_df.groupby("episode_index")["intervention"].apply(
        lambda col: col.apply(_to_bool)
    )
    per_episode_all_true = flagged.groupby(level=0).all()
    per_episode_any_true = flagged.groupby(level=0).any()
    mixed = per_episode_all_true.index[per_episode_all_true != per_episode_any_true]
    if len(mixed) > 0:
        raise AssertionError(
            f"Episode(s) {mixed.tolist()} have a mix of intervention=True/False frames; "
            "expected every episode to be entirely seed (False) or entirely correction (True)."
        )
    return sorted(int(e) for e in per_episode_all_true.index[per_episode_all_true])


def build_plans(
    seed_df: pd.DataFrame,
    corrections_df: pd.DataFrame,
    noop_threshold_deg: float,
    noop_margin_frames: int,
    *,
    trim_seed_idle_edges: bool = False,
    seed_episodes: set[int] | None = None,
) -> tuple[list[EpisodePlan], list[dict]]:
    """Pure planning step: decide which frames go into the aggregate and in
    what order, without touching video or writing anything. Returns
    (plans, trim_report)."""
    plans: list[EpisodePlan] = []
    trim_report: list[dict] = []

    for episode in sorted(seed_df["episode_index"].unique().tolist()):
        if seed_episodes is not None and int(episode) not in seed_episodes:
            continue
        sub = seed_df[seed_df["episode_index"] == episode].sort_values("frame_index")
        start, stop = 0, len(sub)
        if trim_seed_idle_edges:
            action = np.stack(sub["action"].to_numpy()).astype(np.float32)
            start, stop = find_noop_trim_bounds(
                action,
                noop_threshold_deg,
                noop_margin_frames,
            )
        plan = EpisodePlan(
            source_label="seed",
            source_episode_index=int(episode),
            start=start,
            stop=stop,
            global_indices=sub["index"].to_numpy()[start:stop],
            intervention=False,
        )
        plan.trim_trail = len(sub) - stop
        plans.append(plan)
        if trim_seed_idle_edges:
            trim_report.append(
                {
                    "source": "seed",
                    "episode": int(episode),
                    "frames": int(len(sub)),
                    "trim_lead": int(start),
                    "trim_trail": int(len(sub) - stop),
                    "kept": int(stop - start),
                }
            )

    correction_candidates = _intervention_correction_candidates(corrections_df)
    correction_episodes = select_correction_episodes(correction_candidates)
    excluded = sorted(set(correction_candidates) - set(correction_episodes))
    print(
        f"Correction episodes selected: {len(correction_episodes)} of {len(correction_candidates)} "
        f"candidates (excluded {excluded})"
    )

    for episode in correction_episodes:
        sub = corrections_df[corrections_df["episode_index"] == episode].sort_values("frame_index")
        action = np.stack(sub["action"].to_numpy()).astype(np.float32)
        start, stop = find_noop_trim_bounds(action, noop_threshold_deg, noop_margin_frames)
        plan = EpisodePlan(
            source_label="correction",
            source_episode_index=int(episode),
            start=start,
            stop=stop,
            global_indices=sub["index"].to_numpy()[start:stop],
            intervention=True,
        )
        plan.trim_trail = len(sub) - stop
        plans.append(plan)
        trim_report.append(
            {
                "source": "correction",
                "episode": int(episode),
                "frames": int(len(sub)),
                "trim_lead": int(start),
                "trim_trail": int(len(sub) - stop),
                "kept": int(stop - start),
            }
        )

    return plans, trim_report


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"--output-dir already exists: {args.output_dir}. Choose a fresh directory.")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    print(f"Loading seed dataset metadata: {args.seed_repo_id}")
    seed_ds = LeRobotDataset(args.seed_repo_id)
    print(f"Loading corrections dataset metadata: {args.corrections_repo_id} (root={args.corrections_root})")
    corrections_ds = LeRobotDataset(args.corrections_repo_id, root=args.corrections_root)

    fps = int(round(seed_ds.fps))
    if int(round(corrections_ds.fps)) != fps:
        raise AssertionError(f"fps mismatch: seed={seed_ds.fps} corrections={corrections_ds.fps}")

    camera_keys = sorted(key for key in seed_ds.meta.features if key.startswith("observation.images."))
    corrections_camera_keys = sorted(
        key for key in corrections_ds.meta.features if key.startswith("observation.images.")
    )
    if camera_keys != corrections_camera_keys:
        raise AssertionError(f"Camera keys differ: seed={camera_keys} corrections={corrections_camera_keys}")

    state_names = seed_ds.meta.features["observation.state"].get("names")
    action_names = seed_ds.meta.features["action"].get("names")
    if state_names != corrections_ds.meta.features["observation.state"].get("names"):
        raise AssertionError("observation.state joint names differ between seed and corrections.")
    if action_names != corrections_ds.meta.features["action"].get("names"):
        raise AssertionError("action joint names differ between seed and corrections.")

    state_dim = int(seed_ds.meta.features["observation.state"]["shape"][0])
    action_dim = int(seed_ds.meta.features["action"]["shape"][0])
    sample_frame = seed_ds[0]
    height, width = sample_frame[camera_keys[0]].shape[-2:]
    robot_type = seed_ds.meta.robot_type

    print("Loading per-frame tables (parquet only, no video decode)...")
    seed_df = _load_frame_table(Path(seed_ds.root))
    corrections_df = _load_frame_table(Path(corrections_ds.root))
    selected_seed_episodes = None
    if args.seed_episodes:
        selected_seed_episodes = {
            int(value.strip())
            for value in args.seed_episodes.split(",")
            if value.strip()
        }
        available = {
            int(value) for value in seed_df["episode_index"].unique().tolist()
        }
        missing = selected_seed_episodes - available
        if missing:
            raise ValueError(f"Unknown seed episodes: {sorted(missing)}")
        if not selected_seed_episodes:
            raise ValueError("--seed-episodes selected no episodes")

    plans, trim_report = build_plans(
        seed_df,
        corrections_df,
        args.noop_threshold_deg,
        args.noop_margin_frames,
        trim_seed_idle_edges=args.trim_seed_idle_edges,
        seed_episodes=selected_seed_episodes,
    )
    num_seed_episodes = sum(1 for p in plans if p.source_label == "seed")
    num_correction_episodes = sum(1 for p in plans if p.source_label == "correction")
    total_kept_frames = sum(len(p.global_indices) for p in plans)
    print(
        f"Aggregate plan: {num_seed_episodes} seed + {num_correction_episodes} correction "
        f"= {len(plans)} episodes, {total_kept_frames} frames"
    )

    # --- pre-write validation on the PLAN (fail fast, before any video I/O) -----------
    seed_action_all = np.stack(seed_df["action"].to_numpy()).astype(np.float32)
    seed_state_all = np.stack(seed_df["observation.state"].to_numpy()).astype(np.float32)
    validate_finite(seed_action_all, name="seed.action")
    validate_finite(seed_state_all, name="seed.state")

    correction_action_kept = []
    for plan in plans:
        if plan.source_label != "correction":
            continue
        sub = corrections_df[corrections_df["episode_index"] == plan.source_episode_index].sort_values(
            "frame_index"
        )
        action = np.stack(sub["action"].to_numpy()).astype(np.float32)[plan.start : plan.stop]
        state = np.stack(sub["observation.state"].to_numpy()).astype(np.float32)[plan.start : plan.stop]
        validate_finite(action, name=f"correction[{plan.source_episode_index}].action")
        validate_finite(state, name=f"correction[{plan.source_episode_index}].state")
        validate_no_unexpected_idle_edges(action, args.noop_threshold_deg, args.noop_margin_frames)
        correction_action_kept.append(action)
    correction_action_all = np.concatenate(correction_action_kept, axis=0)
    validate_action_maxima_within_seed(
        seed_action_all,
        correction_action_all,
        tolerance_deg=args.action_range_tolerance,
    )
    print("Pre-write plan validation passed (finite, idle edges, action maxima within seed).")

    if args.preserve_seed_idle_edges:
        print("Preserving seed idle edges exactly; validating trimmed corrections only.")
    elif args.trim_seed_idle_edges:
        for plan in plans:
            if plan.source_label != "seed":
                continue
            sub = seed_df[
                seed_df["episode_index"] == plan.source_episode_index
            ].sort_values("frame_index")
            action = np.stack(sub["action"].to_numpy()).astype(np.float32)[
                plan.start : plan.stop
            ]
            validate_no_unexpected_idle_edges(
                action,
                args.noop_threshold_deg,
                args.noop_margin_frames,
            )
        print("Trimmed and validated no-op edges for seed and correction episodes.")
    else:
        for episode in sorted(seed_df["episode_index"].unique().tolist()):
            sub = seed_df[seed_df["episode_index"] == episode].sort_values("frame_index")
            action = np.stack(sub["action"].to_numpy()).astype(np.float32)
            validate_no_unexpected_idle_edges(
                action,
                args.noop_threshold_deg,
                args.noop_margin_frames,
            )

    # --- build the target dataset ------------------------------------------------------
    features = {
        "observation.state": {"dtype": "float32", "shape": (state_dim,), "names": state_names},
        "action": {"dtype": "float32", "shape": (action_dim,), "names": action_names},
        "intervention": {"dtype": "bool", "shape": (1,), "names": None},
    }
    for camera_key in camera_keys:
        features[camera_key] = {
            "dtype": "video",
            "shape": (3, height, width),
            "names": ["channels", "height", "width"],
        }

    target_ds = LeRobotDataset.create(
        repo_id=args.output_repo_id,
        fps=fps,
        features=features,
        root=args.output_dir,
        robot_type=robot_type,
        use_videos=True,
    )

    correspondence_checks = 0
    for plan_index, plan in enumerate(plans):
        source_ds = seed_ds if plan.source_label == "seed" else corrections_ds
        source_df = seed_df if plan.source_label == "seed" else corrections_df
        sub = source_df[source_df["episode_index"] == plan.source_episode_index].sort_values("frame_index")
        full_action = np.stack(sub["action"].to_numpy()).astype(np.float32)
        full_state = np.stack(sub["observation.state"].to_numpy()).astype(np.float32)

        fetched_action = []
        fetched_state = []
        for global_index in plan.global_indices:
            item = source_ds[int(global_index)]
            action = item["action"].numpy().astype(np.float32)
            state = item["observation.state"].numpy().astype(np.float32)
            fetched_action.append(action)
            fetched_state.append(state)
            frame = {
                "observation.state": state,
                "action": action,
                "task": item["task"],
                "intervention": np.array([plan.intervention], dtype=bool),
            }
            for camera_key in camera_keys:
                frame[camera_key] = item[camera_key].numpy()
            target_ds.add_frame(frame)

        # Exact source-frame correspondence: what we just wrote for this episode must equal
        # source[start:stop] precisely (catches decode/index-mapping/trim-boundary bugs).
        validate_frame_correspondence(full_action, np.stack(fetched_action), start=plan.start)
        validate_frame_correspondence(full_state, np.stack(fetched_state), start=plan.start)
        correspondence_checks += 1

        target_ds.save_episode()
        print(
            f"  [{plan_index + 1}/{len(plans)}] {plan.source_label} episode "
            f"{plan.source_episode_index}: kept {len(plan.global_indices)} frames "
            f"(intervention={plan.intervention})",
            flush=True,
        )

    target_ds.finalize()
    print(f"\nWrote {len(plans)} episodes / {total_kept_frames} frames to {args.output_dir}")
    print(f"Frame-correspondence checks passed for {correspondence_checks}/{len(plans)} episodes.")

    # --- post-build validation against the actual persisted artifact ------------------
    built_df = _load_frame_table(args.output_dir)
    validate_contiguous_episode_indices(built_df["episode_index"].to_numpy(), expected_total=len(plans))
    validate_finite(np.stack(built_df["action"].to_numpy()), name="aggregate.action")
    validate_finite(np.stack(built_df["observation.state"].to_numpy()), name="aggregate.state")

    episode_boundaries = []
    for episode in sorted(built_df["episode_index"].unique().tolist()):
        idx = built_df.index[built_df["episode_index"] == episode]
        episode_boundaries.append((int(idx.min()), int(idx.max()) + 1))
    validate_exact_fps(built_df["timestamp"].to_numpy(), episode_boundaries, fps=fps)

    built_seed_action = np.stack(
        built_df.loc[~built_df["intervention"].apply(lambda v: bool(v[0]) if hasattr(v, "__len__") else bool(v)), "action"].to_numpy()
    ).astype(np.float32)
    built_correction_action = np.stack(
        built_df.loc[built_df["intervention"].apply(lambda v: bool(v[0]) if hasattr(v, "__len__") else bool(v)), "action"].to_numpy()
    ).astype(np.float32)
    validate_action_maxima_within_seed(
        built_seed_action,
        built_correction_action,
        tolerance_deg=args.action_range_tolerance,
    )

    for start, stop in episode_boundaries:
        intervention = built_df["intervention"].iloc[start]
        intervention = (
            bool(intervention[0])
            if hasattr(intervention, "__len__")
            else bool(intervention)
        )
        if args.preserve_seed_idle_edges and not intervention:
            continue
        action = np.stack(built_df["action"].to_numpy()[start:stop]).astype(np.float32)
        validate_no_unexpected_idle_edges(action, args.noop_threshold_deg, args.noop_margin_frames)

    print(
        "Post-build validation passed: contiguous episodes, finite, exact fps, "
        "action maxima, correction idle edges."
    )

    # Videos readable: decode first/middle/last frame of every episode via LeRobotDataset.
    reload_ds = LeRobotDataset(args.output_repo_id, root=args.output_dir)
    for start, stop in episode_boundaries:
        for offset in (start, (start + stop - 1) // 2, stop - 1):
            item = reload_ds[offset]
            for camera_key in camera_keys:
                image = item[camera_key]
                if image.shape[-3:] != (3, height, width):
                    raise AssertionError(f"Unreadable/malformed video frame at global index {offset}: {camera_key} shape {image.shape}")
    print(f"Video readability check passed for {len(episode_boundaries)} episodes (first/middle/last frame each).")

    report = {
        "output_dir": str(args.output_dir),
        "output_repo_id": args.output_repo_id,
        "seed_repo_id": args.seed_repo_id,
        "seed_episode_indices": sorted(
            plan.source_episode_index
            for plan in plans
            if plan.source_label == "seed"
        ),
        "corrections_repo_id": args.corrections_repo_id,
        "num_seed_episodes": num_seed_episodes,
        "num_correction_episodes": num_correction_episodes,
        "total_episodes": len(plans),
        "total_frames": total_kept_frames,
        "noop_threshold_deg": args.noop_threshold_deg,
        "noop_margin_frames": args.noop_margin_frames,
        "action_range_tolerance": args.action_range_tolerance,
        "preserve_seed_idle_edges": args.preserve_seed_idle_edges,
        "trim_seed_idle_edges": args.trim_seed_idle_edges,
        "seed_trim_report": [
            entry for entry in trim_report if entry["source"] == "seed"
        ],
        "correction_trim_report": [
            entry for entry in trim_report if entry["source"] == "correction"
        ],
        "validations_passed": [
            "contiguous_episode_indices",
            "finite_state_action",
            "exact_fps",
            "action_maxima_within_seed",
            (
                "no_unexpected_correction_idle_edges"
                if args.preserve_seed_idle_edges
                else "no_unexpected_idle_edges"
            ),
            "exact_frame_correspondence",
            "video_readability",
        ],
    }
    report_path = args.output_dir / "build_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote build report: {report_path}")

    if args.push:
        print(f"\nPushing to hub: {args.output_repo_id}")
        target_ds.push_to_hub()
        print("Push complete.")
    else:
        print("\n--push not set; dataset left local only.")


if __name__ == "__main__":
    main()
