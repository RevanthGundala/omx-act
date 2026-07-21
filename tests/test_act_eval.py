from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import get_policy_class

from configs.act.base import ACTConfig
from models.act.eval_async import return_home
from models.act.eval_common import move_to_joint_state, start_state_violations
from models.act.lerobot_policy import CustomACTConfig, CustomACTPolicy, export_checkpoint
from utils.config import JOINT_NAMES


class _FakeBus:
    def __init__(self, state):
        self.state = np.asarray(state, dtype=np.float32)

    def sync_read(self, _register):
        return {name: float(self.state[i]) for i, name in enumerate(JOINT_NAMES)}


class _FakeRobot:
    action_features = {f"{name}.pos": float for name in JOINT_NAMES}

    def __init__(self, state=None):
        self.bus = _FakeBus(np.zeros(len(JOINT_NAMES)) if state is None else state)
        self.commands = []

    def send_action(self, action):
        self.commands.append(action)
        self.bus.state = np.array(
            [action[f"{name}.pos"] for name in JOINT_NAMES],
            dtype=np.float32,
        )
        return action


def _source_config() -> ACTConfig:
    return ACTConfig(
        dataset_repo_id="owner/dataset",
        camera_names=("wrist", "top"),
        chunk_size=4,
        d_model=32,
        d_z=8,
        num_vae_encoder_layers=1,
        num_encoder_layers=1,
        num_decoder_layers=1,
        num_heads=4,
        mlp_dim=64,
        joint_dim=6,
    )


def _adapter_config(source: ACTConfig) -> CustomACTConfig:
    return CustomACTConfig(
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (6,)),
            "observation.images.wrist": PolicyFeature(FeatureType.VISUAL, (3, 64, 64)),
            "observation.images.top": PolicyFeature(FeatureType.VISUAL, (3, 64, 64)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (6,))},
        device="cpu",
        chunk_size=4,
        n_action_steps=1,
        temporal_ensemble_coeff=None,
        dim_model=32,
        n_heads=4,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        latent_dim=8,
        n_vae_encoder_layers=1,
        source_model_config=asdict(source),
        dataset_repo_id=source.dataset_repo_id,
        action_space="absolute",
    )


def test_custom_act_is_registered_with_lerobot_factory():
    assert get_policy_class("custom_act") is CustomACTPolicy


def test_adapter_prediction_matches_wrapped_model():
    torch.manual_seed(0)
    policy = CustomACTPolicy(_adapter_config(_source_config()))
    policy.eval()
    batch = {
        "observation.state": torch.randn(1, 6),
        "observation.images.wrist": torch.randn(1, 3, 64, 64),
        "observation.images.top": torch.randn(1, 3, 64, 64),
    }
    with torch.inference_mode():
        expected = policy.model(
            torch.stack(
                [batch["observation.images.wrist"], batch["observation.images.top"]],
                dim=1,
            ),
            batch["observation.state"],
        )[0]
        actual = policy.predict_action_chunk(batch)
    assert torch.equal(actual, expected)


def test_export_rejects_delta_action_checkpoint(tmp_path):
    with pytest.raises(NotImplementedError, match="deltas as absolute targets"):
        export_checkpoint(
            Path("missing.pt"),
            tmp_path / "out",
            device="cpu",
            action_space="delta",
            overwrite=False,
        )


def test_move_to_joint_state_uses_motor_telemetry_only():
    robot = _FakeRobot()
    target = np.ones(len(JOINT_NAMES), dtype=np.float32) * 2
    result = move_to_joint_state(
        robot,
        target=target,
        timeout=0.1,
        ramp_seconds=0,
        hz=1000,
        max_delta=1,
        tolerance=0.01,
    )
    assert result["reached"]
    assert np.allclose(result["state"], target)
    assert len(robot.commands) == 2


def test_move_to_joint_state_requires_rate_limit_without_ramp():
    with pytest.raises(ValueError, match="requires max_delta"):
        move_to_joint_state(
            _FakeRobot(),
            target=np.ones(len(JOINT_NAMES), dtype=np.float32),
            timeout=1,
            ramp_seconds=0,
            hz=30,
            max_delta=0,
            tolerance=0.5,
        )


def test_move_to_joint_state_accepts_per_joint_support():
    robot = _FakeRobot(np.array([0, 0, 2, 0, 0, 5], dtype=np.float32))
    result = move_to_joint_state(
        robot,
        target=np.zeros(len(JOINT_NAMES), dtype=np.float32),
        timeout=1,
        ramp_seconds=3,
        hz=30,
        max_delta=2,
        tolerance=np.array([0.5, 0.5, 3, 0.5, 0.5, 6], dtype=np.float32),
    )
    assert result["reached"]
    assert result["max_support_ratio"] < 1
    assert robot.commands == []


def test_start_state_violations_use_per_joint_dataset_support():
    violations = start_state_violations(
        {"wide.pos": 12.0, "narrow.pos": 21.0},
        {"wide.pos": 10.0, "narrow.pos": 20.0},
        {"wide.pos": 2.0, "narrow.pos": 0.1},
        tolerance_floor=0.5,
        max_z=3.0,
    )
    assert "wide.pos" not in violations
    assert violations["narrow.pos"] == pytest.approx({"error": 1.0, "allowed": 0.5})


def test_return_home_limits_each_command_even_with_zero_duration():
    robot = _FakeRobot()
    return_home(
        robot,
        {f"{name}.pos": 5.0 for name in JOINT_NAMES},
        duration_s=0,
        fps=1000,
        label="test",
        max_step_delta=1.0,
    )
    commands = np.array(
        [[item[f"{name}.pos"] for name in JOINT_NAMES] for item in robot.commands]
    )
    assert len(commands) == 5
    assert np.max(np.diff(np.vstack([np.zeros(len(JOINT_NAMES)), commands]), axis=0)) <= 1.0
