"""Tests for the merge-compatibility rules shared by the DAgger builders."""

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.dataset.lerobot_schema import assert_mergeable, merged_features


class _Dataset:
    """Minimal stand-in for a LeRobotDataset: fps, features, and one frame."""

    def __init__(self, *, fps=30, cameras=("top", "wrist"), state_names=("a", "b"),
                 action_names=("a", "b"), robot_type="omx_follower", shape=(3, 4, 8)):
        features = {
            "observation.state": {"shape": (len(state_names),), "names": list(state_names)},
            "action": {"shape": (len(action_names),), "names": list(action_names)},
        }
        for camera in cameras:
            features[f"observation.images.{camera}"] = {"shape": shape}
        self.fps = fps
        self.meta = SimpleNamespace(features=features, robot_type=robot_type)
        self._frame = {
            f"observation.images.{camera}": np.zeros(shape, np.uint8) for camera in cameras
        }

    def __getitem__(self, index):
        return self._frame


def test_schema_describes_the_reference_dataset():
    schema = assert_mergeable({"seed": _Dataset(), "corrections": _Dataset()})

    assert schema.fps == 30
    assert schema.camera_keys == [
        "observation.images.top",
        "observation.images.wrist",
    ]
    assert (schema.state_dim, schema.action_dim) == (2, 2)
    assert (schema.height, schema.width) == (4, 8)
    assert schema.robot_type == "omx_follower"


def test_reordered_joints_are_rejected():
    # The dangerous case: same joints, different order. Both datasets look
    # identical by name and dimension, and merging them would produce a dataset
    # whose actions permute partway through with nothing raising.
    with pytest.raises(ValueError, match="action joint names differ"):
        assert_mergeable(
            {
                "seed": _Dataset(action_names=("a", "b")),
                "corrections": _Dataset(action_names=("b", "a")),
            }
        )


def test_reordered_state_is_rejected():
    with pytest.raises(ValueError, match="observation.state joint names differ"):
        assert_mergeable(
            {
                "seed": _Dataset(state_names=("a", "b")),
                "corrections": _Dataset(state_names=("b", "a")),
            }
        )


def test_fps_mismatch_is_rejected():
    with pytest.raises(ValueError, match="fps mismatch"):
        assert_mergeable({"seed": _Dataset(fps=30), "corrections": _Dataset(fps=10)})


def test_camera_mismatch_is_rejected():
    with pytest.raises(ValueError, match="camera keys differ"):
        assert_mergeable(
            {
                "seed": _Dataset(cameras=("top", "wrist")),
                "corrections": _Dataset(cameras=("top",)),
            }
        )


def test_error_names_the_offending_source():
    with pytest.raises(ValueError, match="corrections"):
        assert_mergeable({"seed": _Dataset(fps=30), "corrections": _Dataset(fps=10)})


def test_a_source_without_cameras_is_rejected():
    with pytest.raises(ValueError, match="no observation.images"):
        assert_mergeable({"seed": _Dataset(cameras=())})


def test_no_sources_is_rejected():
    with pytest.raises(ValueError, match="At least one source"):
        assert_mergeable({})


def test_merged_features_carry_provenance_and_video():
    schema = assert_mergeable({"seed": _Dataset()})

    features = merged_features(schema)

    # intervention is what tells demonstrations from corrections downstream;
    # RECAP advantage labelling forces the correction frames positive.
    assert features["intervention"] == {"dtype": "bool", "shape": (1,), "names": None}
    assert features["observation.state"]["shape"] == (2,)
    assert features["observation.state"]["names"] == ["a", "b"]
    assert features["observation.images.top"] == {
        "dtype": "video",
        "shape": (3, 4, 8),
        "names": ["channels", "height", "width"],
    }
