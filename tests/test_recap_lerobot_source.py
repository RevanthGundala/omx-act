"""Reading demonstrations and DAgger corrections out of LeRobot datasets."""

import json
from pathlib import Path

import numpy as np
import pytest

from models.recap.lerobot_source import (
    MissingOutcomeError,
    _outcome_for,
)


def _write_outcome(root: Path, index: int, label: str) -> None:
    (root / "outcomes").mkdir(parents=True, exist_ok=True)
    (root / "outcomes" / f"episode-{index:06d}.json").write_text(
        json.dumps({"episode_index": index, "outcome_label": label})
    )


def test_a_rollout_without_an_outcome_is_an_error(tmp_path):
    """A correction episode is labeled by the operator; a gap is a real fault."""
    with pytest.raises(MissingOutcomeError, match="no outcome sidecar"):
        _outcome_for(tmp_path, 0, None)


def test_demonstrations_may_default_to_success(tmp_path):
    """Demonstrations carry no sidecar: they are expert successes by design."""
    assert _outcome_for(tmp_path, 0, True) is True


def test_a_present_sidecar_wins_over_the_default(tmp_path):
    _write_outcome(tmp_path, 3, "failure")
    assert _outcome_for(tmp_path, 3, True) is False


def test_an_invalid_label_is_rejected(tmp_path):
    (tmp_path / "outcomes").mkdir()
    (tmp_path / "outcomes" / "episode-000000.json").write_text(
        json.dumps({"episode_index": 0, "outcome_label": "maybe"})
    )
    with pytest.raises(ValueError, match="Invalid outcome label"):
        _outcome_for(tmp_path, 0, None)


def test_an_unfinalized_dataset_is_reported_clearly(tmp_path):
    from models.recap.lerobot_source import _episode_table

    (tmp_path / "meta" / "episodes").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="never finalized"):
        _episode_table(tmp_path)


def test_demonstrations_are_not_marked_as_interventions():
    """Sec. IV-B forces corrections positive, not demonstrations.

    Demonstrations are thresholded like any other data, so marking them would
    force the large majority of frames positive and flatten the signal.
    """
    import inspect

    from models.recap import lerobot_source

    source = inspect.getsource(lerobot_source.load_lerobot_episode)
    assert "np.zeros(ref.length, dtype=np.bool_)" in source
    assert "np.ones(ref.length" not in source


def test_autonomous_rollout_frames_are_not_interventions():
    """A rollout trial is entirely policy action."""
    import inspect

    from models.recap import build_dataset

    source = inspect.getsource(build_dataset.build_dataset)
    assert '"intervention": np.asarray([False], dtype=np.bool_)' in source


def test_the_builder_stores_an_intervention_column():
    from models.recap.build_dataset import build_features

    features = build_features(
        camera_names=("top",),
        joint_names=("a", "b"),
        frame_shape=(4, 4, 3),
        use_videos=False,
    )
    assert features["intervention"] == {
        "dtype": "bool",
        "shape": (1,),
        "names": None,
    }
