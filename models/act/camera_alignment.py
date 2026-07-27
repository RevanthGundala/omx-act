from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
import math

import cv2
import numpy as np

from models.act.scene_check import BLUE_RANGE, TABLE_FRACTION, WHITE_RANGE


@dataclass(frozen=True)
class CameraTransform:
    dx_px: float
    dy_px: float
    rotation_deg: float
    scale: float
    inliers: int
    inlier_ratio: float


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
    return CameraTransform(
        dx_px=float(matrix[0, 2]),
        dy_px=float(matrix[1, 2]),
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
    def __init__(self, reference_frames, thresholds, *, minimum_references: int):
        self.reference_frames = tuple(reference_frames)
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
            "dx_px": _percentile_limit(
                [abs(estimate.dx_px) for estimate in baseline],
                4.0,
            ),
            "dy_px": _percentile_limit(
                [abs(estimate.dy_px) for estimate in baseline],
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
        return cls.from_frames(
            load_training_start_frames(
                repo_id,
                revision=revision,
                camera_name=camera_name,
                max_references=max_references,
            )
        )

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
            "dx_px": float(np.median([estimate.dx_px for estimate in selected])),
            "dy_px": float(np.median([estimate.dy_px for estimate in selected])),
            "rotation_deg": float(
                np.median([estimate.rotation_deg for estimate in selected])
            ),
            "scale": float(np.median([estimate.scale for estimate in selected])),
            "inliers": int(np.median([estimate.inliers for estimate in selected])),
            "matched_references": len(estimates),
            "total_references": len(self.reference_frames),
        }
        within_training_view = (
            abs(result["dx_px"]) <= self.thresholds["dx_px"]
            and abs(result["dy_px"]) <= self.thresholds["dy_px"]
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
    max_references: int,
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
    if len(start_indices) > max_references:
        selected = np.linspace(
            0,
            len(start_indices) - 1,
            max_references,
        ).round().astype(int)
        start_indices = [start_indices[index] for index in selected]
    key = f"observation.images.{camera_name}"
    return [_image_to_rgb_uint8(dataset[index][key]) for index in start_indices]


def print_camera_view_check(
    checker: CameraAlignmentChecker,
    image_rgb: np.ndarray,
    *,
    label: str,
) -> dict:
    result = checker.check(image_rgb)
    if result["status"] == "low_confidence":
        details = (
            f"refs={result['matched_references']}/{result['total_references']}"
        )
    else:
        details = (
            f"refs={result['matched_references']}/{result['total_references']} "
            f"inliers_med={result['inliers']} dx={result['dx_px']:+.1f}px "
            f"dy={result['dy_px']:+.1f}px rot={result['rotation_deg']:+.2f}deg "
            f"scale={result['scale']:.3f}"
        )
    banner = "=" * 78
    print(banner, flush=True)
    print(f"[{label}] {result['verdict']}", flush=True)
    print(f"[{label}] {details}", flush=True)
    print(banner, flush=True)
    return result
