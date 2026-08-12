import json

import numpy as np
import pytest

import evaluation.compare_act_policies as comparison
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


def test_pose_errors_convert_normalized_omx_units_to_degrees():
    zero = np.zeros((1, 6), dtype=np.float32)
    quarter_turn = zero.copy()
    quarter_turn[0, 0] = 50.0

    position_error, orientation_error = pose_errors(quarter_turn, zero)

    assert position_error[0] > 300
    assert orientation_error[0] == pytest.approx(90.0)


def test_episode_start_pairs_skip_missing_and_cross_episode_next_frames():
    class FakeDataset:
        hf_dataset = [
            {"episode_index": 0, "frame_index": 0},
            {"episode_index": 1, "frame_index": 0},
            {"episode_index": 2, "frame_index": 0},
            {"episode_index": 1, "frame_index": 1},
            {"episode_index": 2, "frame_index": 1},
        ]

        def __len__(self):
            return len(self.hf_dataset)

    assert comparison.episode_start_frame_pairs(FakeDataset()) == [(1, 3), (2, 4)]


def test_scene_tracking_serializes_undefined_correlations_as_null():
    constant_actions = np.ones((3, 4, 6), dtype=np.float32)

    tracking = comparison._scene_tracking(constant_actions, constant_actions)

    assert all(joint["correlation"] is None for joint in tracking.values())
    json.dumps(tracking, allow_nan=False)
