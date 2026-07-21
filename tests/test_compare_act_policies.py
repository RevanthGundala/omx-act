import numpy as np
import pytest

from evaluation.compare_act_policies import (
    omx_end_effector_transforms,
    pose_errors,
)


def test_zero_pose_matches_open_manipulator_chain_dimensions():
    transform = omx_end_effector_transforms(np.zeros(6))

    assert transform[:3, 3] == pytest.approx([0.286, 0.0, 0.1875])


def test_shoulder_pan_rotates_arm_around_base():
    action = np.zeros(6)
    action[0] = 90

    transform = omx_end_effector_transforms(action)

    assert transform[:3, 3] == pytest.approx([0.012, 0.274, 0.1875])


def test_pose_errors_are_zero_for_identical_actions():
    actions = np.asarray(
        [
            [0, -60, 50, 40, 0, 55],
            [10, -20, 5, -15, 20, 30],
        ],
        dtype=np.float32,
    )

    position_error, orientation_error = pose_errors(actions, actions)

    assert np.array_equal(position_error, np.zeros(2))
    assert np.array_equal(orientation_error, np.zeros(2))
