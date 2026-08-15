"""Audit whether an ACT dataset actually requires object-conditioned actions.

This script estimates object-position diversity from the top camera, qpos
diversity, and the key shortcut question:

    For similar qpos states, do different object positions imply different actions?

If actions are predictable from qpos alone, behavior cloning can learn a
trajectory prior without grounding to objects even when the visual dataset looks
diverse.

Example:
    uv run python evaluation/dataset_shortcut_audit.py \
      --dataset RevanthGundala/007-place-penne-pasta-in-bowl \
      --output-json outputs/dataset_audits/shortcut_audit.json \
      --output-image outputs/dataset_audits/shortcut_audit.jpg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))



JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def to_hwc_uint8(image: torch.Tensor) -> np.ndarray:
    array = image.detach().cpu().numpy()
    if array.ndim == 3 and array.shape[0] in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    if array.max() <= 1.0:
        array = (array * 255).clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def detect_penne_top(image: np.ndarray) -> tuple[float, float, float] | None:
    """Crude yellow/orange object detector for the penne/pasta in top camera."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array([15, 45, 80]), np.array([45, 255, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if 10 < cv2.contourArea(contour) < 3000]
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None
    x = moments["m10"] / moments["m00"]
    y = moments["m01"] / moments["m00"]
    return float(x), float(y), float(cv2.contourArea(contour))


def episode_indices(dataset: LeRobotDataset) -> dict[int, list[int]]:
    by_episode: dict[int, list[int]] = {}
    for idx in range(len(dataset)):
        episode = int(dataset.hf_dataset[idx]["episode_index"])
        by_episode.setdefault(episode, []).append(idx)
    return by_episode


def ridge_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, lam: float = 1e-3) -> np.ndarray:
    x_train = np.c_[x_train, np.ones(len(x_train))]
    x_test = np.c_[x_test, np.ones(len(x_test))]
    weights = np.linalg.solve(x_train.T @ x_train + lam * np.eye(x_train.shape[1]), x_train.T @ y_train)
    return x_test @ weights


def l1(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))


def l2(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(pred - target, axis=1)))


def nearest_neighbor_shortcut_metrics(
    qpos: np.ndarray,
    actions: np.ndarray,
    objects: np.ndarray,
    episodes: np.ndarray,
    *,
    sample_count: int,
    neighbor_count: int,
) -> dict:
    rng = np.random.default_rng(0)
    indices = np.arange(len(qpos))
    if len(indices) > sample_count:
        indices = rng.choice(indices, size=sample_count, replace=False)

    rows = []
    for idx in indices:
        same = np.where(episodes != episodes[idx])[0]
        if len(same) == 0:
            continue
        distances = np.linalg.norm(qpos[same] - qpos[idx], axis=1)
        nearest_order = np.argsort(distances)[:neighbor_count]
        nearest = same[nearest_order]
        if len(nearest) == 0:
            continue

        action_d = np.linalg.norm(actions[nearest] - actions[idx], axis=1)
        object_d = np.linalg.norm(objects[nearest] - objects[idx], axis=1)
        qpos_d = distances[nearest_order]
        rows.append(
            {
                "qpos_distance": float(np.mean(qpos_d)),
                "action_distance": float(np.mean(action_d)),
                "object_distance_px": float(np.mean(object_d)),
                "object_action_ratio": float(np.mean(action_d) / max(float(np.mean(object_d)), 1e-6)),
            }
        )

    if not rows:
        return {"count": 0}

    def summary(key: str) -> dict[str, float]:
        values = np.array([row[key] for row in rows], dtype=np.float32)
        return {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p90": float(np.quantile(values, 0.9)),
            "max": float(values.max()),
        }

    return {
        "count": len(rows),
        "qpos_distance": summary("qpos_distance"),
        "action_distance": summary("action_distance"),
        "object_distance_px": summary("object_distance_px"),
        "object_action_ratio": summary("object_action_ratio"),
    }


def make_start_scatter(path: Path, starts: list[dict]) -> None:
    width, height = 640, 480
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    for start in starts:
        x, y = int(round(start["object_x"])), int(round(start["object_y"]))
        episode = int(start["episode"])
        color = (
            int(50 + 180 * ((episode * 37) % 100) / 100),
            int(50 + 180 * ((episode * 53) % 100) / 100),
            int(50 + 180 * ((episode * 71) % 100) / 100),
        )
        cv2.circle(canvas, (x, y), 5, color, -1)
        cv2.putText(canvas, str(episode), (x + 6, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    cv2.putText(canvas, "Detected start object positions (top camera)", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)


def normalize_columns(features: np.ndarray) -> np.ndarray:
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    return (features - mean) / np.clip(std, 1e-6, None)


def episode_action_signature(dataset: LeRobotDataset, indices: list[int], a_mean: np.ndarray, a_std: np.ndarray, samples: int = 5) -> np.ndarray:
    positions = np.linspace(0, len(indices) - 1, samples).astype(int)
    actions = []
    for position in positions:
        action = dataset[indices[position]]["action"].numpy().astype(np.float32).reshape(-1)
        actions.append((action - a_mean) / a_std)
    return np.concatenate(actions)


def select_diverse_episodes(features_by_episode: dict[int, np.ndarray], keep_fraction: float) -> tuple[list[int], list[int]]:
    episodes = sorted(features_by_episode)
    keep_count = int(round(len(episodes) * keep_fraction))
    keep_count = min(max(1, keep_count), len(episodes))
    feature_matrix = normalize_columns(np.stack([features_by_episode[episode] for episode in episodes]))

    centroid = feature_matrix.mean(axis=0)
    first = int(np.argmin(np.linalg.norm(feature_matrix - centroid, axis=1)))
    selected = [first]
    min_dist = np.linalg.norm(feature_matrix - feature_matrix[first], axis=1)

    while len(selected) < keep_count:
        next_idx = int(np.argmax(min_dist))
        selected.append(next_idx)
        min_dist = np.minimum(min_dist, np.linalg.norm(feature_matrix - feature_matrix[next_idx], axis=1))

    keep = sorted(episodes[idx] for idx in selected)
    drop = sorted(set(episodes) - set(keep))
    return keep, drop


def make_episode_strip(
    dataset: LeRobotDataset,
    indices: list[int],
    *,
    width: int,
    height: int,
    samples_per_episode: int,
) -> np.ndarray:
    positions = np.linspace(0, len(indices) - 1, samples_per_episode).astype(int)
    strip = np.zeros((height, width * samples_per_episode, 3), dtype=np.uint8)
    for col, position in enumerate(positions):
        image = to_hwc_uint8(dataset[indices[position]]["observation.images.top"])
        thumb = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        strip[:, col * width : (col + 1) * width] = thumb
    return strip


def make_split_visualization(
    path: Path,
    dataset: LeRobotDataset,
    by_episode: dict[int, list[int]],
    keep_episodes: list[int],
    drop_episodes: list[int],
    *,
    samples_per_episode: int = 3,
    thumb_width: int = 96,
    thumb_height: int = 72,
    cols: int = 5,
) -> None:
    keep_set = set(keep_episodes)
    drop_set = set(drop_episodes)
    episodes = sorted(by_episode)

    def panel(panel_episodes: list[int], title: str, mark_all: bool) -> np.ndarray:
        cell_w = thumb_width * samples_per_episode
        cell_h = thumb_height + 22
        rows = int(np.ceil(max(len(panel_episodes), 1) / cols))
        header_h = 34
        image = np.full((header_h + rows * cell_h, cols * cell_w, 3), 245, dtype=np.uint8)
        cv2.putText(image, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        for idx, episode in enumerate(panel_episodes):
            row = idx // cols
            col = idx % cols
            x0 = col * cell_w
            y0 = header_h + row * cell_h
            strip = make_episode_strip(
                dataset,
                by_episode[episode],
                width=thumb_width,
                height=thumb_height,
                samples_per_episode=samples_per_episode,
            )
            image[y0 + 22 : y0 + 22 + thumb_height, x0 : x0 + cell_w] = strip
            is_drop = episode in drop_set
            color = (0, 0, 255) if is_drop else (0, 170, 0)
            label = f"DROP ep{episode}" if is_drop else f"KEEP ep{episode}"
            if mark_all or is_drop:
                cv2.rectangle(image, (x0, y0 + 22), (x0 + cell_w - 1, y0 + 22 + thumb_height - 1), color, 4)
            cv2.putText(image, label, (x0 + 3, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        return image

    all_panel = panel(episodes, "All episodes: green=keep, red=drop", mark_all=True)
    drop_panel = panel(drop_episodes, "Dropped episodes only", mark_all=True)
    height = max(all_panel.shape[0], drop_panel.shape[0])
    combined = np.full((height, all_panel.shape[1] + drop_panel.shape[1] + 16, 3), 235, dtype=np.uint8)
    combined[: all_panel.shape[0], : all_panel.shape[1]] = all_panel
    combined[: drop_panel.shape[0], all_panel.shape[1] + 16 :] = drop_panel
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), combined)


def make_feature_explanation_visualization(
    path: Path,
    dataset: LeRobotDataset,
    by_episode: dict[int, list[int]],
    starts: list[dict],
    *,
    samples_per_episode: int = 5,
    thumb_width: int = 128,
    thumb_height: int = 96,
    cols: int = 2,
) -> None:
    starts_by_episode = {int(start["episode"]): start for start in starts}
    episodes = sorted(starts_by_episode)
    cell_w = thumb_width * samples_per_episode
    cell_h = thumb_height + 34
    rows = int(np.ceil(len(episodes) / cols))
    header_h = 38
    canvas = np.full((header_h + rows * cell_h, cols * cell_w, 3), 245, dtype=np.uint8)
    cv2.putText(
        canvas,
        "Feature construction: red dot = detected object_xy in first frame; columns = sampled action-trajectory frames",
        (8, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 0, 0),
        2,
    )

    for idx, episode in enumerate(episodes):
        indices = by_episode[episode]
        positions = np.linspace(0, len(indices) - 1, samples_per_episode).astype(int)
        row = idx // cols
        col = idx % cols
        x0 = col * cell_w
        y0 = header_h + row * cell_h
        cv2.putText(
            canvas,
            f"ep{episode} sampled action frames",
            (x0 + 4, y0 + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
        )
        start = starts_by_episode[episode]
        object_x = float(start["object_x"])
        object_y = float(start["object_y"])
        for sample_col, position in enumerate(positions):
            image = to_hwc_uint8(dataset[indices[position]]["observation.images.top"])
            thumb = cv2.resize(image, (thumb_width, thumb_height), interpolation=cv2.INTER_AREA)
            if sample_col == 0:
                px = int(round(object_x / 640.0 * thumb_width))
                py = int(round(object_y / 480.0 * thumb_height))
                cv2.circle(thumb, (px, py), 6, (255, 0, 0), -1)
                cv2.circle(thumb, (px, py), 8, (255, 255, 255), 1)
                cv2.putText(thumb, "object_xy", (max(0, px - 35), max(12, py - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            cv2.putText(
                thumb,
                f"s{sample_col}",
                (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )
            tx = x0 + sample_col * thumb_width
            ty = y0 + 24
            canvas[ty : ty + thumb_height, tx : tx + thumb_width] = cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit qpos shortcut vs object-conditioned diversity.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--stride", type=int, default=5, help="Use every Nth frame per episode for full-dataset metrics.")
    parser.add_argument("--neighbor-samples", type=int, default=1000)
    parser.add_argument("--neighbor-count", type=int, default=10)
    parser.add_argument("--keep-fraction", type=float, default=0.75, help="Fraction of detected episodes to keep in the recommended diverse split.")
    parser.add_argument("--split-output-json", default="outputs/dataset_audits/recommended_split.json")
    parser.add_argument("--split-output-image", default="outputs/dataset_audits/recommended_split.jpg")
    parser.add_argument("--feature-output-image", default="outputs/dataset_audits/feature_construction.jpg")
    parser.add_argument("--output-json", default="outputs/dataset_audits/shortcut_audit.json")
    parser.add_argument("--output-image", default="outputs/dataset_audits/shortcut_audit.jpg")
    args = parser.parse_args()

    dataset = LeRobotDataset(args.dataset, revision=args.revision)
    metadata = LeRobotDatasetMetadata(args.dataset, revision=args.revision)
    q_mean = np.asarray(metadata.stats["observation.state"]["mean"], dtype=np.float32).reshape(-1)
    q_std = np.asarray(metadata.stats["observation.state"]["std"], dtype=np.float32).reshape(-1).clip(1e-6)
    a_mean = np.asarray(metadata.stats["action"]["mean"], dtype=np.float32).reshape(-1)
    a_std = np.asarray(metadata.stats["action"]["std"], dtype=np.float32).reshape(-1).clip(1e-6)

    by_episode = episode_indices(dataset)
    episodes_sorted = sorted(by_episode)

    starts = []
    rows = []
    for episode in episodes_sorted:
        indices = by_episode[episode]
        first_sample = dataset[indices[0]]
        detection = detect_penne_top(to_hwc_uint8(first_sample["observation.images.top"]))
        if detection is None:
            continue
        object_x, object_y, area = detection
        starts.append(
            {
                "episode": episode,
                "object_x": object_x,
                "object_y": object_y,
                "area": area,
                "length": len(indices),
            }
        )

        sampled_indices = indices[:: max(args.stride, 1)]
        max_position = max(len(sampled_indices) - 1, 1)
        for pos, idx in enumerate(sampled_indices):
            sample = dataset[idx]
            qpos = (sample["observation.state"].numpy().astype(np.float32).reshape(-1) - q_mean) / q_std
            action = (sample["action"].numpy().astype(np.float32).reshape(-1) - a_mean) / a_std
            phase = np.array([pos / max_position], dtype=np.float32)
            rows.append(
                {
                    "episode": episode,
                    "qpos": qpos,
                    "action": action,
                    "phase": phase,
                    "object": np.array([object_x, object_y], dtype=np.float32),
                }
            )

    if len(starts) < 2 or len(rows) < 10:
        raise RuntimeError("Not enough detected starts/rows to audit dataset.")

    object_points = np.array([[start["object_x"], start["object_y"]] for start in starts], dtype=np.float32)
    qpos = np.stack([row["qpos"] for row in rows])
    actions = np.stack([row["action"] for row in rows])
    phases = np.stack([row["phase"] for row in rows])
    objects = np.stack([row["object"] for row in rows])
    episodes = np.array([row["episode"] for row in rows], dtype=np.int64)

    train_episodes = set(episodes_sorted[: int(0.8 * len(episodes_sorted))])
    train_mask = np.array([episode in train_episodes for episode in episodes])
    test_mask = ~train_mask

    feature_sets = {
        "mean_action": None,
        "phase_only": phases,
        "object_only": objects / np.array([640.0, 480.0], dtype=np.float32),
        "qpos_only": qpos,
        "qpos_plus_phase": np.concatenate([qpos, phases], axis=1),
        "qpos_plus_object": np.concatenate([qpos, objects / np.array([640.0, 480.0], dtype=np.float32)], axis=1),
        "qpos_phase_object": np.concatenate([qpos, phases, objects / np.array([640.0, 480.0], dtype=np.float32)], axis=1),
    }

    regression = {}
    for name, features in feature_sets.items():
        if features is None:
            pred = np.repeat(actions[train_mask].mean(axis=0, keepdims=True), test_mask.sum(), axis=0)
        else:
            pred = ridge_predict(features[train_mask], actions[train_mask], features[test_mask])
        regression[name] = {
            "mae": l1(pred, actions[test_mask]),
            "l2": l2(pred, actions[test_mask]),
        }

    first_rows = []
    for start in starts:
        sample = dataset[by_episode[start["episode"]][0]]
        q = (sample["observation.state"].numpy().astype(np.float32).reshape(-1) - q_mean) / q_std
        a = (sample["action"].numpy().astype(np.float32).reshape(-1) - a_mean) / a_std
        o = np.array([start["object_x"] / 640.0, start["object_y"] / 480.0], dtype=np.float32)
        first_rows.append((start["episode"], q, o, a))

    first_episodes = np.array([row[0] for row in first_rows], dtype=np.int64)
    first_train_mask = np.array([episode in train_episodes for episode in first_episodes])
    first_q = np.stack([row[1] for row in first_rows])
    first_o = np.stack([row[2] for row in first_rows])
    first_a = np.stack([row[3] for row in first_rows])
    first_regression = {}
    for name, features in {
        "mean_action": None,
        "object_only": first_o,
        "qpos_only": first_q,
        "qpos_plus_object": np.concatenate([first_q, first_o], axis=1),
    }.items():
        if features is None:
            pred = np.repeat(first_a[first_train_mask].mean(axis=0, keepdims=True), (~first_train_mask).sum(), axis=0)
        else:
            pred = ridge_predict(features[first_train_mask], first_a[first_train_mask], features[~first_train_mask], lam=1e-2)
        first_regression[name] = {
            "mae": l1(pred, first_a[~first_train_mask]),
            "l2": l2(pred, first_a[~first_train_mask]),
        }

    neighbor_metrics = nearest_neighbor_shortcut_metrics(
        qpos,
        actions,
        objects,
        episodes,
        sample_count=args.neighbor_samples,
        neighbor_count=args.neighbor_count,
    )

    object_distances = []
    for i in range(len(object_points)):
        for j in range(i + 1, len(object_points)):
            object_distances.append(float(np.linalg.norm(object_points[i] - object_points[j])))

    starts_by_episode = {int(start["episode"]): start for start in starts}
    features_by_episode = {}
    for episode in sorted(starts_by_episode):
        first_sample = dataset[by_episode[episode][0]]
        start_qpos = (first_sample["observation.state"].numpy().astype(np.float32).reshape(-1) - q_mean) / q_std
        object_feature = np.array(
            [starts_by_episode[episode]["object_x"] / 640.0, starts_by_episode[episode]["object_y"] / 480.0],
            dtype=np.float32,
        )
        action_feature = episode_action_signature(dataset, by_episode[episode], a_mean, a_std)
        features_by_episode[episode] = np.concatenate(
            [
                start_qpos,
                object_feature * 2.0,
                action_feature,
            ]
        )
    keep_episodes, drop_episodes = select_diverse_episodes(features_by_episode, args.keep_fraction)
    split = {
        "dataset": args.dataset,
        "revision": args.revision,
        "keep_fraction": args.keep_fraction,
        "keep_count": len(keep_episodes),
        "drop_count": len(drop_episodes),
        "keep_episodes": keep_episodes,
        "drop_episodes": drop_episodes,
        "method": "greedy farthest-point diverse subset over normalized [start_qpos, 2*object_xy, sampled_action_trajectory]",
        "note": "This split removes near-duplicate episodes; it does not guarantee object-conditioned supervision unless qpos-only metrics degrade on the subset.",
    }

    output = {
        "dataset": args.dataset,
        "revision": args.revision,
        "episodes": len(episodes_sorted),
        "frames": len(dataset),
        "detected_starts": len(starts),
        "sampled_rows": len(rows),
        "object_start_distribution_px": {
            "x_mean": float(object_points[:, 0].mean()),
            "x_std": float(object_points[:, 0].std()),
            "x_min": float(object_points[:, 0].min()),
            "x_max": float(object_points[:, 0].max()),
            "y_mean": float(object_points[:, 1].mean()),
            "y_std": float(object_points[:, 1].std()),
            "y_min": float(object_points[:, 1].min()),
            "y_max": float(object_points[:, 1].max()),
            "pairwise_distance_mean": float(np.mean(object_distances)),
            "pairwise_distance_median": float(np.median(object_distances)),
        },
        "heldout_regression": regression,
        "first_frame_regression": first_regression,
        "nearest_qpos_neighbors": neighbor_metrics,
        "recommended_split": split,
        "starts": starts,
        "interpretation": {
            "qpos_shortcut_present": regression["qpos_only"]["mae"] <= regression["qpos_plus_object"]["mae"] * 1.05,
            "object_adds_predictive_value": regression["qpos_plus_object"]["mae"] < regression["qpos_only"]["mae"] * 0.95,
        },
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, indent=2) + "\n")
    make_start_scatter(Path(args.output_image), starts)
    split_json = Path(args.split_output_json)
    split_json.parent.mkdir(parents=True, exist_ok=True)
    split_json.write_text(json.dumps(split, indent=2) + "\n")
    make_split_visualization(
        Path(args.split_output_image),
        dataset,
        by_episode,
        keep_episodes,
        drop_episodes,
    )
    make_feature_explanation_visualization(
        Path(args.feature_output_image),
        dataset,
        by_episode,
        starts,
    )

    print(f"Wrote {output_json}")
    print(f"Wrote {args.output_image}")
    print(f"Wrote {split_json}")
    print(f"Wrote {args.split_output_image}")
    print(f"Wrote {args.feature_output_image}")
    print("Object start spread px:")
    dist = output["object_start_distribution_px"]
    print(
        f"  x mean/std/min/max = {dist['x_mean']:.1f}/{dist['x_std']:.1f}/{dist['x_min']:.1f}/{dist['x_max']:.1f}"
    )
    print(
        f"  y mean/std/min/max = {dist['y_mean']:.1f}/{dist['y_std']:.1f}/{dist['y_min']:.1f}/{dist['y_max']:.1f}"
    )
    print("Held-out regression MAE/L2:")
    for name, metrics in regression.items():
        print(f"  {name:18s} MAE={metrics['mae']:.4f} L2={metrics['l2']:.4f}")
    print("First-frame regression MAE/L2:")
    for name, metrics in first_regression.items():
        print(f"  {name:18s} MAE={metrics['mae']:.4f} L2={metrics['l2']:.4f}")
    print("Interpretation:", output["interpretation"])
    print(f"Recommended split: keep {len(keep_episodes)} episodes, drop {len(drop_episodes)} episodes")
    print("Dropped episodes:", drop_episodes)


if __name__ == "__main__":
    main()
