"""One-shot scene coverage check: given a live top-camera frame, detect the sponge/bowl
positions and report whether the current physical setup matches an in-distribution layout
from the training dataset (RevanthGundala/011-place-sponge-in-bowl), or an under/uncovered one.

Detection method: simple HSV color-blob centroid (sponge = blue cloth, bowl = white dish),
restricted to the top ~55% of the frame to exclude the robot arm base / floor trim, which
otherwise gets mis-detected as a giant "blue" or "white" blob (see
outputs/dataset_audits/011_position_detections.json and the debug images that led to these
exact thresholds -- do not loosen them without re-verifying against
outputs/dataset_audits/011_start_frames/_montage.jpg).

This is NOT a general-purpose object detector (no YOLO/model involved) -- it is tuned
specifically to this rig's lighting/table/objects. If the sponge or bowl color/lighting setup
changes, these thresholds will need re-tuning (see outputs/dataset_audits/011_position_detections_debug.jpg
for the sanity-check methodology).
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
COVERAGE_JSON = REPO_ROOT / "outputs/dataset_audits/011_position_detections.json"

# Tuned against outputs/dataset_audits/011_start_frames (50/50 detected correctly).
BLUE_RANGE = ((85, 95, 150), (140, 255, 255))    # sponge
WHITE_RANGE = ((0, 0, 200), (180, 40, 255))      # bowl
TABLE_FRACTION = 0.55  # exclude the bottom ~45% (arm base, floor trim) from detection


def _centroid(mask: np.ndarray, min_area: float) -> tuple[float, float] | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > min_area]
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    m = cv2.moments(c)
    if m["m00"] == 0:
        return None
    return (m["m10"] / m["m00"], m["m01"] / m["m00"])


def detect_sponge_bowl(image_rgb: np.ndarray) -> dict[str, tuple[float, float] | None]:
    """image_rgb: HxWx3 uint8, RGB channel order (matches OpenCVCamera's default output).
    Returns normalized (x, y) in [0,1] for each object, or None if not detected."""
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    table_h = int(h * TABLE_FRACTION)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    blue_mask = cv2.inRange(hsv, BLUE_RANGE[0], BLUE_RANGE[1])
    white_mask = cv2.inRange(hsv, WHITE_RANGE[0], WHITE_RANGE[1])
    blue_mask[table_h:, :] = 0
    white_mask[table_h:, :] = 0

    sponge = _centroid(blue_mask, min_area=150)
    bowl = _centroid(white_mask, min_area=200)
    return {
        "sponge_xy_norm": (sponge[0] / w, sponge[1] / h) if sponge else None,
        "bowl_xy_norm": (bowl[0] / w, bowl[1] / h) if bowl else None,
    }


def xy_bucket(xy: tuple[float, float]) -> str:
    x, y = xy
    col = "left" if x < 1 / 3 else ("center" if x < 2 / 3 else "right")
    row = "near" if y < 1 / 3 else ("mid" if y < 2 / 3 else "far")
    return f"{col}-{row}"


def load_coverage(path: Path = COVERAGE_JSON) -> tuple[Counter, Counter, int]:
    """Returns (sponge_bucket_counts, bowl_bucket_counts, n_episodes) from the precomputed
    per-episode detections. Raises FileNotFoundError if the audit hasn't been run yet."""
    import json
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run the 011 position audit first "
            "(see outputs/dataset_audits/011_position_detections.json generation in session history)."
        )
    data = json.loads(path.read_text())
    sponge_counts = Counter(r["sponge_bucket"] for r in data if r.get("sponge_bucket"))
    bowl_counts = Counter(r["bowl_bucket"] for r in data if r.get("bowl_bucket"))
    return sponge_counts, bowl_counts, len(data)


def coverage_from_frames(
    frames: list[np.ndarray] | tuple[np.ndarray, ...],
) -> tuple[Counter, Counter, int]:
    sponge_counts = Counter()
    bowl_counts = Counter()
    for frame in frames:
        detection = detect_sponge_bowl(frame)
        if detection["sponge_xy_norm"] is not None:
            sponge_counts[xy_bucket(detection["sponge_xy_norm"])] += 1
        if detection["bowl_xy_norm"] is not None:
            bowl_counts[xy_bucket(detection["bowl_xy_norm"])] += 1
    return sponge_counts, bowl_counts, len(frames)


def check_scene(
    image_rgb: np.ndarray,
    coverage_path: Path = COVERAGE_JSON,
    *,
    coverage: tuple[Counter, Counter, int] | None = None,
) -> dict:
    """Full check: detect + bucket + compare against training coverage. Returns a dict with
    the raw detections, buckets, per-bucket episode counts, and a plain-language verdict."""
    det = detect_sponge_bowl(image_rgb)
    result = {**det, "sponge_bucket": None, "bowl_bucket": None,
              "sponge_bucket_count": 0, "bowl_bucket_count": 0, "n_episodes": 0,
              "verdict": "unknown"}

    if det["sponge_xy_norm"] is None or det["bowl_xy_norm"] is None:
        result["verdict"] = (
            "COULD NOT DETECT " +
            ("sponge" if det["sponge_xy_norm"] is None else "") +
            (" and " if det["sponge_xy_norm"] is None and det["bowl_xy_norm"] is None else "") +
            ("bowl" if det["bowl_xy_norm"] is None else "") +
            " -- check lighting/placement, or objects may be occluded by the arm."
        )
        return result

    if coverage is None:
        try:
            coverage = load_coverage(coverage_path)
        except FileNotFoundError as exc:
            result["verdict"] = f"detected positions but no coverage baseline available: {exc}"
            return result
    sponge_counts, bowl_counts, n_episodes = coverage

    sponge_bucket = xy_bucket(det["sponge_xy_norm"])
    bowl_bucket = xy_bucket(det["bowl_xy_norm"])
    sponge_count = sponge_counts.get(sponge_bucket, 0)
    bowl_count = bowl_counts.get(bowl_bucket, 0)

    result.update({
        "sponge_bucket": sponge_bucket, "bowl_bucket": bowl_bucket,
        "sponge_bucket_count": sponge_count, "bowl_bucket_count": bowl_count,
        "n_episodes": n_episodes,
    })

    min_count = min(sponge_count, bowl_count)
    if min_count == 0:
        result["verdict"] = (
            f"OUT OF DISTRIBUTION: sponge bucket '{sponge_bucket}' ({sponge_count}/{n_episodes} "
            f"episodes) x bowl bucket '{bowl_bucket}' ({bowl_count}/{n_episodes} episodes) -- "
            "at least one of these positions was NEVER seen during training."
        )
    elif min_count <= 2:
        result["verdict"] = (
            f"SPARSE/EDGE CASE: sponge '{sponge_bucket}' ({sponge_count}/{n_episodes}), "
            f"bowl '{bowl_bucket}' ({bowl_count}/{n_episodes}) -- rarely seen, treat as a stress test."
        )
    else:
        result["verdict"] = (
            f"IN DISTRIBUTION: sponge '{sponge_bucket}' ({sponge_count}/{n_episodes} episodes), "
            f"bowl '{bowl_bucket}' ({bowl_count}/{n_episodes} episodes) -- well covered."
        )
    return result


def print_scene_check(
    image_rgb: np.ndarray,
    label: str = "eval",
    *,
    coverage: tuple[Counter, Counter, int] | None = None,
) -> dict:
    """Convenience wrapper: run check_scene and print a one-line summary. Returns the full result."""
    result = check_scene(image_rgb, coverage=coverage)
    sx = result["sponge_xy_norm"]
    bx = result["bowl_xy_norm"]
    sx_s = f"({sx[0]:.2f},{sx[1]:.2f})" if sx else "?"
    bx_s = f"({bx[0]:.2f},{bx[1]:.2f})" if bx else "?"
    banner = "=" * 78
    # Printed as a boxed, all-caps banner (not just a plain log line) specifically because a
    # single easy-to-miss line got scrolled past by console noise in practice -- this check only
    # matters if it's actually seen before the rollout starts.
    print(banner, flush=True)
    print(f"[{label}] SCENE CHECK -- sponge_xy={sx_s} bowl_xy={bx_s}", flush=True)
    print(f"[{label}] {result['verdict'].upper()}", flush=True)
    print(banner, flush=True)
    return result
