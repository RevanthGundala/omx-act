import io

import cv2
import numpy as np

import models.act.camera_alignment as camera_alignment
from models.act.camera_alignment import (
    CameraAlignmentChecker,
    estimate_camera_transform,
    prepare_diagnostic_references,
    print_camera_view_check,
)


class _TTYStringIO(io.StringIO):
    def isatty(self):
        return True


def _background(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = np.full((360, 480, 3), 90, dtype=np.uint8)
    for _ in range(180):
        center = tuple(rng.integers([0, 0], [480, 360]).tolist())
        color = tuple(int(value) for value in rng.integers(30, 220, size=3))
        cv2.circle(image, center, int(rng.integers(2, 7)), color, -1)
    for _ in range(30):
        p1 = tuple(rng.integers([0, 0], [480, 360]).tolist())
        p2 = tuple(rng.integers([0, 0], [480, 360]).tolist())
        cv2.line(image, p1, p2, (160, 160, 160), 1)
    return image


def _with_objects(background: np.ndarray, offset: int) -> np.ndarray:
    image = background.copy()
    cv2.rectangle(image, (40 + offset, 40), (100 + offset, 90), (30, 80, 220), -1)
    cv2.circle(image, (320 - offset, 100), 35, (245, 245, 245), -1)
    return image


def test_camera_alignment_ignores_moved_objects_when_background_is_stable():
    background = _background()
    references = [_with_objects(background, offset) for offset in (0, 20, 40, 60)]
    checker = CameraAlignmentChecker.from_frames(references)

    result = checker.check(_with_objects(background, 90))

    assert result["status"] == "pass"
    assert abs(result["center_dx_px"]) < 2
    assert abs(result["center_dy_px"]) < 2
    assert abs(result["rotation_deg"]) < 0.5
    assert abs(result["scale"] - 1) < 0.01


def test_camera_alignment_recovers_known_background_transform():
    reference = _with_objects(_background(), 0)
    expected_angle = 2.0
    expected_scale = 1.02
    matrix = cv2.getRotationMatrix2D((240, 180), expected_angle, expected_scale)
    matrix[:, 2] += (12, -8)
    moved = cv2.warpAffine(reference, matrix, (480, 360))

    estimate = estimate_camera_transform(reference, moved)

    assert estimate is not None
    assert estimate.inliers >= 30
    assert abs(estimate.center_dx_px - 12) < 1
    assert abs(estimate.center_dy_px + 8) < 1
    assert abs(estimate.rotation_deg + expected_angle) < 0.75
    assert abs(estimate.scale - expected_scale) < 0.02


def test_camera_alignment_never_passes_low_feature_frames():
    blank = np.zeros((360, 480, 3), dtype=np.uint8)
    checker = CameraAlignmentChecker.from_frames([blank, blank.copy()])

    result = checker.check(blank.copy())

    assert result["status"] == "low_confidence"


def test_dataset_checker_keeps_all_starts_for_scene_coverage(monkeypatch):
    frames = [_with_objects(_background(), offset) for offset in range(0, 100, 20)]
    monkeypatch.setattr(
        camera_alignment,
        "load_training_start_frames",
        lambda *_args, **_kwargs: frames,
    )

    checker = CameraAlignmentChecker.from_dataset(
        "owner/dataset",
        max_references=2,
    )

    assert len(checker.reference_frames) == 2
    assert len(checker.coverage_frames) == 5


def test_scene_coverage_loads_when_camera_alignment_is_disabled(monkeypatch):
    frames = [_with_objects(_background(), offset) for offset in range(0, 100, 20)]
    monkeypatch.setattr(
        camera_alignment,
        "load_training_start_frames",
        lambda *_args, **_kwargs: frames,
    )

    checker, coverage = prepare_diagnostic_references(
        "owner/dataset",
        revision="main",
        camera_name="top",
        max_alignment_references=2,
        enable_alignment=False,
    )

    assert checker is None
    assert coverage[2] == 5


def test_camera_view_check_colors_warning_verdict_yellow():
    class WarningChecker:
        thresholds = {
            "center_dx_px": 10.0,
            "center_dy_px": 10.0,
            "rotation_deg": 1.5,
            "scale_delta": 0.03,
        }

        def check(self, _image):
            return {
                "status": "warn",
                "verdict": "CAMERA VIEW WARN: shifted",
                "matched_references": 12,
                "total_references": 12,
                "inliers": 100,
                "center_dx_px": -3.0,
                "center_dy_px": -5.0,
                "rotation_deg": 7.58,
                "scale": 0.998,
            }

    output = _TTYStringIO()

    print_camera_view_check(
        WarningChecker(),
        np.zeros((10, 10, 3), dtype=np.uint8),
        label="test",
        output=output,
    )

    assert "\033[1;33mCAMERA VIEW WARN" in output.getvalue()
    assert "camera roll is +7.58deg" in output.getvalue()
    assert "center_shift=(-3.0,-5.0)px" in output.getvalue()
