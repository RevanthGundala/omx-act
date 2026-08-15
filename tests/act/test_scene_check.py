import io

import cv2
import numpy as np

from models.act.scene_check import check_scene, coverage_from_frames, print_scene_check


class _TTYStringIO(io.StringIO):
    def isatty(self):
        return True


def _frame(sponge_x: int, bowl_x: int) -> np.ndarray:
    image = np.zeros((360, 480, 3), dtype=np.uint8)
    cv2.rectangle(image, (sponge_x, 50), (sponge_x + 50, 90), (20, 70, 230), -1)
    cv2.circle(image, (bowl_x, 110), 30, (245, 245, 245), -1)
    return image


def test_scene_check_uses_supplied_training_frame_coverage():
    references = [_frame(30, 330), _frame(45, 345), _frame(60, 360)]
    coverage = coverage_from_frames(references)

    result = check_scene(_frame(50, 350), coverage=coverage)

    assert result["n_episodes"] == 3
    assert result["sponge_bucket_count"] == 3
    assert result["bowl_bucket_count"] == 3
    assert result["verdict"].startswith("IN DISTRIBUTION")


def test_scene_check_colors_pass_and_failure_verdicts():
    frame = _frame(50, 350)
    passing_output = _TTYStringIO()
    failing_output = _TTYStringIO()

    print_scene_check(
        frame,
        coverage=coverage_from_frames([frame, frame, frame]),
        output=passing_output,
    )
    print_scene_check(
        frame,
        coverage=coverage_from_frames([_frame(350, 30)]),
        output=failing_output,
    )

    assert "\033[1;32mIN DISTRIBUTION" in passing_output.getvalue()
    assert "\033[1;31mOUT OF DISTRIBUTION" in failing_output.getvalue()
