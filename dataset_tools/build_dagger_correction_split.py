"""Build disjoint train and holdout datasets from multiple DAgger correction sets."""

from __future__ import annotations

import argparse
import glob
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from dataset_tools.noop_trim import find_noop_trim_bounds


@dataclass(frozen=True)
class CorrectionSource:
    label: str
    repo_id: str
    root: Path


@dataclass(frozen=True)
class CorrectionPlan:
    source_label: str
    source_episode: int
    start: int
    stop: int
    split: str


def parse_source(value: str) -> CorrectionSource:
    parts = value.split("|", 2)
    if len(parts) != 3 or not all(parts):
        raise argparse.ArgumentTypeError(
            "--source must be LABEL|REPO_ID|ROOT"
        )
    return CorrectionSource(parts[0], parts[1], Path(parts[2]).expanduser())


def parse_holdout(value: str) -> tuple[str, int]:
    try:
        label, episode = value.rsplit(":", 1)
        return label, int(episode)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError(
            "--holdout must be LABEL:EPISODE"
        ) from exc


def load_frame_table(root: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(root / "data" / "chunk-*" / "file-*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files under {root}/data")
    return (
        pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
        .sort_values("index")
        .reset_index(drop=True)
    )


def build_plans(
    source_tables: dict[str, pd.DataFrame],
    holdouts: set[tuple[str, int]],
    *,
    threshold: float,
    margin_frames: int,
) -> list[CorrectionPlan]:
    plans = []
    available = set()
    for label, table in source_tables.items():
        for episode in sorted(int(value) for value in table["episode_index"].unique()):
            available.add((label, episode))
            episode_rows = table[table["episode_index"] == episode].sort_values(
                "frame_index"
            )
            intervention = episode_rows["intervention"].apply(
                lambda value: bool(value[0])
                if hasattr(value, "__len__")
                else bool(value)
            )
            if not intervention.all():
                raise AssertionError(
                    f"{label}:{episode} contains non-intervention frames"
                )
            action = np.stack(episode_rows["action"].to_numpy()).astype(np.float32)
            start, stop = find_noop_trim_bounds(
                action,
                threshold_deg=threshold,
                margin_frames=margin_frames,
            )
            plans.append(
                CorrectionPlan(
                    source_label=label,
                    source_episode=episode,
                    start=start,
                    stop=stop,
                    split="holdout" if (label, episode) in holdouts else "train",
                )
            )
    missing = holdouts - available
    if missing:
        raise ValueError(f"Unknown holdout corrections: {sorted(missing)}")
    return plans


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=parse_source, required=True)
    parser.add_argument("--holdout", action="append", type=parse_holdout, default=[])
    parser.add_argument("--train-output-dir", type=Path, required=True)
    parser.add_argument("--train-repo-id", required=True)
    parser.add_argument("--holdout-output-dir", type=Path)
    parser.add_argument("--holdout-repo-id")
    parser.add_argument("--noop-threshold", type=float, default=0.1)
    parser.add_argument("--noop-margin-frames", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if bool(args.holdout_output_dir) != bool(args.holdout_repo_id):
        raise ValueError(
            "--holdout-output-dir and --holdout-repo-id must be provided together"
        )
    if args.holdout and args.holdout_output_dir is None:
        raise ValueError("Holdout episodes require holdout output arguments")
    output_dirs = [args.train_output_dir]
    if args.holdout_output_dir is not None:
        output_dirs.append(args.holdout_output_dir)
    for output_dir in output_dirs:
        if output_dir.exists():
            raise FileExistsError(f"Output already exists: {output_dir}")

    from lerobot.datasets import LeRobotDataset

    sources = {}
    tables = {}
    labels = set()
    for source in args.source:
        if source.label in labels:
            raise ValueError(f"Duplicate source label: {source.label}")
        labels.add(source.label)
        sources[source.label] = LeRobotDataset(source.repo_id, root=source.root)
        tables[source.label] = load_frame_table(source.root)

    reference = next(iter(sources.values()))
    fps = int(round(reference.fps))
    camera_keys = sorted(
        key for key in reference.meta.features if key.startswith("observation.images.")
    )
    state_feature = reference.meta.features["observation.state"]
    action_feature = reference.meta.features["action"]
    for label, dataset in sources.items():
        if int(round(dataset.fps)) != fps:
            raise AssertionError(f"FPS mismatch for {label}")
        if sorted(
            key for key in dataset.meta.features if key.startswith("observation.images.")
        ) != camera_keys:
            raise AssertionError(f"Camera mismatch for {label}")
        if dataset.meta.features["observation.state"].get("names") != state_feature.get(
            "names"
        ):
            raise AssertionError(f"State order mismatch for {label}")
        if dataset.meta.features["action"].get("names") != action_feature.get("names"):
            raise AssertionError(f"Action order mismatch for {label}")

    plans = build_plans(
        tables,
        set(args.holdout),
        threshold=args.noop_threshold,
        margin_frames=args.noop_margin_frames,
    )
    split_plans = {
        split: [plan for plan in plans if plan.split == split]
        for split in ("train", "holdout")
    }
    if not split_plans["train"]:
        raise ValueError("Training split must be non-empty")
    if args.holdout and not split_plans["holdout"]:
        raise ValueError("Holdout split must be non-empty")

    sample = reference[0]
    height, width = sample[camera_keys[0]].shape[-2:]
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": tuple(state_feature["shape"]),
            "names": state_feature.get("names"),
        },
        "action": {
            "dtype": "float32",
            "shape": tuple(action_feature["shape"]),
            "names": action_feature.get("names"),
        },
        "intervention": {"dtype": "bool", "shape": (1,), "names": None},
    }
    for key in camera_keys:
        features[key] = {
            "dtype": "video",
            "shape": (3, height, width),
            "names": ["channels", "height", "width"],
        }

    outputs = {
        "train": LeRobotDataset.create(
            args.train_repo_id,
            fps,
            root=args.train_output_dir,
            robot_type=reference.meta.robot_type,
            features=features,
            use_videos=True,
        ),
    }
    if args.holdout_output_dir is not None:
        outputs["holdout"] = LeRobotDataset.create(
            args.holdout_repo_id,
            fps,
            root=args.holdout_output_dir,
            robot_type=reference.meta.robot_type,
            features=features,
            use_videos=True,
        )

    report = {"train": [], "holdout": []}
    try:
        for split, output in outputs.items():
            selected = split_plans[split]
            for output_episode, plan in enumerate(selected):
                table = tables[plan.source_label]
                episode_rows = table[
                    table["episode_index"] == plan.source_episode
                ].sort_values("frame_index")
                indices = episode_rows["index"].to_numpy()[plan.start : plan.stop]
                source = sources[plan.source_label]
                for index in indices:
                    item = source[int(index)]
                    frame = {
                        "observation.state": item["observation.state"].numpy(),
                        "action": item["action"].numpy(),
                        "task": item["task"],
                        "intervention": np.array([True], dtype=bool),
                    }
                    for key in camera_keys:
                        frame[key] = item[key].numpy()
                    output.add_frame(frame)
                output.save_episode()
                report[split].append(
                    {
                        "output_episode": output_episode,
                        "source": plan.source_label,
                        "source_episode": plan.source_episode,
                        "source_frames": len(episode_rows),
                        "trim_lead": plan.start,
                        "trim_trail": len(episode_rows) - plan.stop,
                        "kept_frames": plan.stop - plan.start,
                    }
                )
                print(
                    f"{split} [{output_episode + 1}/{len(selected)}] "
                    f"{plan.source_label}:{plan.source_episode} "
                    f"kept={plan.stop - plan.start}",
                    flush=True,
                )
    finally:
        for output in outputs.values():
            output.finalize()

    provenance = {
        "sources": [
            {
                "label": source.label,
                "repo_id": source.repo_id,
                "root": str(source.root),
            }
            for source in args.source
        ],
        "holdouts": [
            {"source": label, "episode": episode}
            for label, episode in sorted(set(args.holdout))
        ],
        "noop_threshold": args.noop_threshold,
        "noop_margin_frames": args.noop_margin_frames,
        **report,
    }
    split_output_dirs = {"train": args.train_output_dir}
    if args.holdout_output_dir is not None:
        split_output_dirs["holdout"] = args.holdout_output_dir
    for split, output_dir in split_output_dirs.items():
        split_report = {
            **provenance,
            "split": split,
            "episodes": len(report[split]),
            "frames": sum(item["kept_frames"] for item in report[split]),
        }
        (output_dir / "build_report.json").write_text(
            json.dumps(split_report, indent=2)
        )


if __name__ == "__main__":
    main()
