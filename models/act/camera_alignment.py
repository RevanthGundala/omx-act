from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
import json
import math
from pathlib import Path
import sys
from typing import TextIO

import cv2
import numpy as np

from models.act.scene_check import (
    BLUE_RANGE,
    TABLE_FRACTION,
    WHITE_RANGE,
    coverage_from_frames,
)
from models.act.terminal_colors import colorize_status


@dataclass(frozen=True)
class CameraTransform:
    center_dx_px: float
    center_dy_px: float
    rotation_deg: float
    scale: float
    inliers: int
    inlier_ratio: float


def diagnostic_dataset_reference(
    policy_path: Path,
    repo_id_override: str | None,
    revision_override: str | None,
) -> tuple[str, str]:
    train_config_path = policy_path / "train_config.json"
    if not train_config_path.exists():
        raise FileNotFoundError(f"no train_config.json in {policy_path}")
    dataset = json.loads(train_config_path.read_text())["dataset"]
    policy_repo_id = dataset["repo_id"]
    policy_revision = dataset.get("revision") or "main"
    repo_id = repo_id_override or policy_repo_id
    revision = revision_override or (
        policy_revision if repo_id_override is None else "main"
    )
    return repo_id, revision


def prepare_diagnostic_references(
    repo_id: str,
    *,
    revision: str,
    camera_name: str,
    max_alignment_references: int,
    enable_alignment: bool,
):
    coverage_frames = load_training_start_frames(
        repo_id,
        revision=revision,
        camera_name=camera_name,
        max_references=None,
    )
    coverage = coverage_from_frames(coverage_frames)
    checker = None
    if enable_alignment:
        checker = CameraAlignmentChecker.from_frames(
            _select_evenly(coverage_frames, max_alignment_references)
        )
        checker.coverage_frames = tuple(coverage_frames)
    return checker, coverage


def load_session_diagnostics(
    policy_path: Path,
    *,
    repo_id_override: str | None,
    revision_override: str | None,
    camera_view_check: bool,
    camera_reference_count: int,
    prepare=None,
):
    prepare = prepare or prepare_diagnostic_references
    repo_id, revision = diagnostic_dataset_reference(
        policy_path,
        repo_id_override,
        revision_override,
    )
    print(
        "[eval_async] loading top-camera diagnostic references from "
        f"{repo_id}@{revision}...",
        flush=True,
    )
    return prepare(
        repo_id,
        revision=revision,
        camera_name="top",
        max_alignment_references=camera_reference_count,
        enable_alignment=camera_view_check,
    )


def resolve_dataset_start_pose(
    robot,
    policy_path: Path,
    *,
    repo_id_override: str | None,
    revision_override: str | None,
    statistics_loader,
):
    repo_id, revision = diagnostic_dataset_reference(
        policy_path,
        repo_id_override,
        revision_override,
    )
    statistics = statistics_loader(repo_id, revision)
    keys = list(robot.action_features)
    mean = np.asarray(statistics["mean"], dtype=np.float32).reshape(-1)
    if len(mean) != len(keys):
        raise ValueError(
            f"Dataset start mean has {len(mean)} joints but robot has {len(keys)}."
        )
    pose = {key: float(mean[index]) for index, key in enumerate(keys)}
    return pose, statistics, repo_id, revision


def _background_feature_mask(image_rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    height, width = image_rgb.shape[:2]
    table_height = int(height * TABLE_FRACTION)
    feature_mask = np.full((height, width), 255, dtype=np.uint8)
    kernel = np.ones((31, 31), dtype=np.uint8)

    for lower, upper, minimum_area in (
        (BLUE_RANGE[0], BLUE_RANGE[1], 150),
        (WHITE_RANGE[0], WHITE_RANGE[1], 200),
    ):
        color_mask = cv2.inRange(hsv, lower, upper)
        color_mask[table_height:, :] = 0
        contours, _ = cv2.findContours(
            color_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        contours = [
            contour
            for contour in contours
            if minimum_area < cv2.contourArea(contour) < height * width * 0.15
        ]
        if not contours:
            continue
        object_mask = np.zeros_like(feature_mask)
        cv2.drawContours(
            object_mask,
            [max(contours, key=cv2.contourArea)],
            -1,
            255,
            thickness=-1,
        )
        object_mask = cv2.dilate(object_mask, kernel)
        feature_mask[object_mask > 0] = 0

    return feature_mask


def _feature_inputs(image_rgb: np.ndarray):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    mask = _background_feature_mask(image_rgb)
    orb = cv2.ORB_create(nfeatures=1800, fastThreshold=10)
    return orb.detectAndCompute(gray, mask)


def estimate_camera_transform(
    reference_rgb: np.ndarray,
    live_rgb: np.ndarray,
) -> CameraTransform | None:
    if reference_rgb.shape != live_rgb.shape or reference_rgb.ndim != 3:
        raise ValueError(
            f"Camera frames must have the same HxWxC shape, got "
            f"{reference_rgb.shape} and {live_rgb.shape}."
        )

    reference_points, reference_descriptors = _feature_inputs(reference_rgb)
    live_points, live_descriptors = _feature_inputs(live_rgb)
    if reference_descriptors is None or live_descriptors is None:
        return None

    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(
        reference_descriptors,
        live_descriptors,
        k=2,
    )
    good_matches = [
        first
        for pair in matches
        if len(pair) == 2
        for first, second in [pair]
        if first.distance < 0.75 * second.distance
    ]
    if len(good_matches) < 8:
        return None

    reference_xy = np.float32(
        [reference_points[match.queryIdx].pt for match in good_matches]
    )
    live_xy = np.float32(
        [live_points[match.trainIdx].pt for match in good_matches]
    )
    matrix, inlier_mask = cv2.estimateAffinePartial2D(
        reference_xy,
        live_xy,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=3000,
        confidence=0.995,
    )
    if matrix is None or inlier_mask is None:
        return None

    inliers = int(inlier_mask.sum())
    if inliers < 6:
        return None
    a, b = float(matrix[0, 0]), float(matrix[1, 0])
    scale = math.sqrt(a * a + b * b)
    rotation_deg = math.degrees(math.atan2(b, a))
    height, width = reference_rgb.shape[:2]
    center = np.array([width / 2.0, height / 2.0, 1.0])
    mapped_center = matrix @ center
    return CameraTransform(
        center_dx_px=float(mapped_center[0] - center[0]),
        center_dy_px=float(mapped_center[1] - center[1]),
        rotation_deg=rotation_deg,
        scale=scale,
        inliers=inliers,
        inlier_ratio=inliers / len(good_matches),
    )


def _percentile_limit(values, floor: float) -> float:
    if not values:
        return floor
    return max(floor, float(np.percentile(values, 95)) * 1.5)


class CameraAlignmentChecker:
    def __init__(
        self,
        reference_frames,
        thresholds,
        *,
        minimum_references: int,
        coverage_frames=None,
    ):
        self.reference_frames = tuple(reference_frames)
        self.coverage_frames = tuple(
            self.reference_frames if coverage_frames is None else coverage_frames
        )
        self.thresholds = thresholds
        self.minimum_references = minimum_references

    @classmethod
    def from_frames(cls, reference_frames):
        frames = tuple(np.ascontiguousarray(frame) for frame in reference_frames)
        baseline = [
            estimate_camera_transform(first, second)
            for first, second in combinations(frames, 2)
        ]
        baseline = [estimate for estimate in baseline if estimate is not None]
        thresholds = {
            "center_dx_px": _percentile_limit(
                [abs(estimate.center_dx_px) for estimate in baseline],
                4.0,
            ),
            "center_dy_px": _percentile_limit(
                [abs(estimate.center_dy_px) for estimate in baseline],
                4.0,
            ),
            "rotation_deg": _percentile_limit(
                [abs(estimate.rotation_deg) for estimate in baseline],
                0.5,
            ),
            "scale_delta": _percentile_limit(
                [abs(estimate.scale - 1.0) for estimate in baseline],
                0.01,
            ),
        }
        return cls(
            frames,
            thresholds,
            minimum_references=min(3, len(frames)),
        )

    @classmethod
    def from_dataset(
        cls,
        repo_id: str,
        *,
        revision: str = "main",
        camera_name: str = "top",
        max_references: int = 12,
    ):
        coverage_frames = load_training_start_frames(
            repo_id,
            revision=revision,
            camera_name=camera_name,
            max_references=None,
        )
        reference_frames = _select_evenly(coverage_frames, max_references)
        checker = cls.from_frames(reference_frames)
        checker.coverage_frames = tuple(coverage_frames)
        return checker

    def check(self, live_rgb: np.ndarray) -> dict:
        estimates = [
            estimate_camera_transform(reference, live_rgb)
            for reference in self.reference_frames
        ]
        estimates = [estimate for estimate in estimates if estimate is not None]
        if len(estimates) < self.minimum_references:
            return {
                "status": "low_confidence",
                "matched_references": len(estimates),
                "total_references": len(self.reference_frames),
                "verdict": (
                    "CAMERA VIEW LOW CONFIDENCE: too few background-feature "
                    "matches; check occlusion, lighting, or camera position."
                ),
            }

        estimates.sort(key=lambda estimate: estimate.inliers, reverse=True)
        selected = estimates[: min(5, len(estimates))]
        result = {
            "center_dx_px": float(
                np.median([estimate.center_dx_px for estimate in selected])
            ),
            "center_dy_px": float(
                np.median([estimate.center_dy_px for estimate in selected])
            ),
            "rotation_deg": float(
                np.median([estimate.rotation_deg for estimate in selected])
            ),
            "scale": float(np.median([estimate.scale for estimate in selected])),
            "inliers": int(np.median([estimate.inliers for estimate in selected])),
            "matched_references": len(estimates),
            "total_references": len(self.reference_frames),
        }
        within_training_view = (
            abs(result["center_dx_px"]) <= self.thresholds["center_dx_px"]
            and abs(result["center_dy_px"]) <= self.thresholds["center_dy_px"]
            and abs(result["rotation_deg"]) <= self.thresholds["rotation_deg"]
            and abs(result["scale"] - 1.0) <= self.thresholds["scale_delta"]
        )
        result["status"] = "pass" if within_training_view else "warn"
        result["verdict"] = (
            "CAMERA VIEW PASS (screening only, not calibration)"
            if within_training_view
            else "CAMERA VIEW WARN: background alignment differs from training starts"
        )
        return result


def _image_to_rgb_uint8(image) -> np.ndarray:
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    if image.dtype != np.uint8:
        if image.max(initial=0) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def load_training_start_frames(
    repo_id: str,
    *,
    revision: str,
    camera_name: str,
    max_references: int | None,
) -> list[np.ndarray]:
    from lerobot.datasets import LeRobotDataset

    dataset = LeRobotDataset(repo_id, revision=revision)
    start_indices = [
        index
        for index in range(len(dataset))
        if int(dataset.hf_dataset[index]["frame_index"]) == 0
    ]
    if not start_indices:
        raise ValueError(f"Dataset {repo_id}@{revision} has no episode-start frames.")
    if max_references is not None and len(start_indices) > max_references:
        selected = np.linspace(
            0,
            len(start_indices) - 1,
            max_references,
        ).round().astype(int)
        start_indices = [start_indices[index] for index in selected]
    key = f"observation.images.{camera_name}"
    return [_image_to_rgb_uint8(dataset[index][key]) for index in start_indices]


def _select_evenly(frames, maximum: int):
    frames = tuple(frames)
    if len(frames) <= maximum:
        return frames
    selected = np.linspace(0, len(frames) - 1, maximum).round().astype(int)
    return tuple(frames[index] for index in selected)


def _camera_view_diagnosis(result: dict, thresholds: dict[str, float]) -> str:
    mismatches = []
    center_dx = result["center_dx_px"]
    center_dy = result["center_dy_px"]
    rotation = result["rotation_deg"]
    scale = result["scale"]

    if abs(center_dx) > thresholds["center_dx_px"]:
        direction = "right" if center_dx > 0 else "left"
        mismatches.append(
            f"background center is shifted {direction} {abs(center_dx):.1f}px "
            f"(limit {thresholds['center_dx_px']:.1f}px)"
        )
    if abs(center_dy) > thresholds["center_dy_px"]:
        direction = "down" if center_dy > 0 else "up"
        mismatches.append(
            f"background center is shifted {direction} {abs(center_dy):.1f}px "
            f"(limit {thresholds['center_dy_px']:.1f}px)"
        )
    if abs(rotation) > thresholds["rotation_deg"]:
        mismatches.append(
            f"camera roll is {rotation:+.2f}deg relative to training "
            f"(limit +/-{thresholds['rotation_deg']:.2f}deg)"
        )
    if abs(scale - 1.0) > thresholds["scale_delta"]:
        mismatches.append(
            f"view scale is {scale:.3f} "
            f"(allowed {1.0 - thresholds['scale_delta']:.3f}.."
            f"{1.0 + thresholds['scale_delta']:.3f})"
        )

    if not mismatches:
        return "View center, camera roll, and scale match the training starts."
    return "Likely mismatch: " + "; ".join(mismatches) + "."


def print_camera_view_check(
    checker: CameraAlignmentChecker,
    image_rgb: np.ndarray,
    *,
    label: str,
    output: TextIO | None = None,
) -> dict:
    output = sys.stdout if output is None else output
    result = checker.check(image_rgb)
    if result["status"] == "low_confidence":
        details = (
            f"refs={result['matched_references']}/{result['total_references']}"
        )
    else:
        details = (
            f"refs={result['matched_references']}/{result['total_references']} "
            f"inliers_med={result['inliers']} "
            f"center_shift=({result['center_dx_px']:+.1f},"
            f"{result['center_dy_px']:+.1f})px "
            f"roll={result['rotation_deg']:+.2f}deg scale={result['scale']:.3f}"
        )
        diagnosis = _camera_view_diagnosis(result, checker.thresholds)
    banner = "=" * 78
    color = "green" if result["status"] == "pass" else "yellow"
    verdict = colorize_status(result["verdict"], color, output)
    print(banner, file=output, flush=True)
    print(f"[{label}] {verdict}", file=output, flush=True)
    if result["status"] != "low_confidence":
        print(f"[{label}] {diagnosis}", file=output, flush=True)
    print(f"[{label}] {details}", file=output, flush=True)
    print(banner, file=output, flush=True)
    return result
