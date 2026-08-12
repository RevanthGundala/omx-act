import socket
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from lerobot.async_inference.robot_client import RobotClient
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import get_policy_class

from configs.act.base import ACTConfig
from models.act.eval_async import (
    LockedRobotClient,
    apply_wrist_flex_start_override,
    install_action_rate_limiter,
    install_action_smoother,
    launch_server,
    return_home,
    run_guarded_cleanup,
    start_state_reference,
    trial_has_recorded_actions,
    visualize_action_queue_size_if_present,
)
from models.act.eval_common import move_to_joint_state, start_state_violations
from models.act.lerobot_policy import CustomACTConfig, CustomACTPolicy, export_checkpoint
from utils.config import JOINT_NAMES


class _FakeBus:
    def __init__(self, state):
        self.state = np.asarray(state, dtype=np.float32)

    def sync_read(self, _register):
        return {name: float(self.state[i]) for i, name in enumerate(JOINT_NAMES)}

    def read(self, _register, _motor, *, normalize=False):
        assert not normalize
        return 0


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


def test_move_to_joint_state_runs_idle_callback_while_moving():
    robot = _FakeRobot()
    updates = []

    move_to_joint_state(
        robot,
        target=np.ones(len(JOINT_NAMES), dtype=np.float32),
        timeout=0.1,
        ramp_seconds=0.1,
        hz=1000,
        max_delta=1.0,
        tolerance=0.01,
        loop_callback=lambda: updates.append("preview"),
    )

    assert updates


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


def test_move_to_joint_state_rejects_non_finite_tolerance():
    # A non-finite tolerance (e.g. from --start-state-tolerance inf) makes
    # `abs(error) <= tolerance` vacuously true, so the very first telemetry read
    # would be reported as "reached" with zero commands sent, completely
    # bypassing pose verification. This must be rejected up front instead.
    robot = _FakeRobot(np.zeros(len(JOINT_NAMES), dtype=np.float32))
    with pytest.raises(ValueError, match="finite"):
        move_to_joint_state(
            robot,
            target=np.ones(len(JOINT_NAMES), dtype=np.float32) * 100,
            timeout=1,
            ramp_seconds=1,
            hz=30,
            max_delta=1,
            tolerance=float("inf"),
        )
    assert robot.commands == []


def test_start_state_violations_rejects_non_finite_tolerance_floor():
    # tolerance_floor=inf makes `allowed = max(inf, ...)` infinite, so
    # `error > allowed` can never be true and every violation is silently
    # swallowed. This must be rejected instead of returning a clean result.
    with pytest.raises(ValueError, match="finite"):
        start_state_violations(
            {"wide.pos": 999.0},
            {"wide.pos": 10.0},
            {"wide.pos": 2.0},
            tolerance_floor=float("inf"),
            max_z=3.0,
        )


def test_start_state_reference_uses_seed_override(tmp_path):
    policy_path = tmp_path / "pretrained_model"
    policy_path.mkdir()
    (policy_path / "train_config.json").write_text(
        '{"dataset": {"repo_id": "owner/dagger", "revision": "dagger-rev"}}'
    )

    assert start_state_reference(policy_path, "owner/seed", None) == (
        "owner/seed",
        "main",
    )
    assert start_state_reference(policy_path, "owner/seed", "seed-rev") == (
        "owner/seed",
        "seed-rev",
    )


def test_wrist_flex_start_override_stays_within_dataset_support():
    pose = {f"{name}.pos": 0.0 for name in JOINT_NAMES}
    statistics = {
        "min": np.full(len(JOINT_NAMES), -10.0, dtype=np.float32),
        "max": np.full(len(JOINT_NAMES), 10.0, dtype=np.float32),
    }

    updated = apply_wrist_flex_start_override(pose, statistics, 6.0)

    assert updated["wrist_flex.pos"] == 6.0
    assert pose["wrist_flex.pos"] == 0.0


def test_wrist_flex_start_override_none_uses_dataset_pose():
    pose = {f"{name}.pos": float(index) for index, name in enumerate(JOINT_NAMES)}
    statistics = {
        "min": np.full(len(JOINT_NAMES), -10.0, dtype=np.float32),
        "max": np.full(len(JOINT_NAMES), 10.0, dtype=np.float32),
    }

    assert apply_wrist_flex_start_override(pose, statistics, None) == pose


def test_wrist_flex_start_offset_shifts_dataset_mean():
    pose = {f"{name}.pos": 0.0 for name in JOINT_NAMES}
    pose["wrist_flex.pos"] = 4.0
    statistics = {
        "min": np.full(len(JOINT_NAMES), -10.0, dtype=np.float32),
        "max": np.full(len(JOINT_NAMES), 10.0, dtype=np.float32),
    }

    updated = apply_wrist_flex_start_override(pose, statistics, None, -1.25)

    assert updated["wrist_flex.pos"] == 2.75
    assert pose["wrist_flex.pos"] == 4.0


def test_wrist_flex_start_override_rejects_out_of_distribution_target():
    pose = {f"{name}.pos": 0.0 for name in JOINT_NAMES}
    statistics = {
        "min": np.full(len(JOINT_NAMES), -10.0, dtype=np.float32),
        "max": np.full(len(JOINT_NAMES), 10.0, dtype=np.float32),
    }

    with pytest.raises(ValueError, match="outside dataset start support"):
        apply_wrist_flex_start_override(pose, statistics, 11.0)


def test_wrist_flex_start_offset_rejects_out_of_distribution_target():
    pose = {f"{name}.pos": 0.0 for name in JOINT_NAMES}
    pose["wrist_flex.pos"] = 9.0
    statistics = {
        "min": np.full(len(JOINT_NAMES), -10.0, dtype=np.float32),
        "max": np.full(len(JOINT_NAMES), 10.0, dtype=np.float32),
    }

    with pytest.raises(ValueError, match="outside dataset start support"):
        apply_wrist_flex_start_override(pose, statistics, None, 2.0)


def test_wrist_flex_start_adjustments_are_mutually_exclusive():
    pose = {f"{name}.pos": 0.0 for name in JOINT_NAMES}
    statistics = {
        "min": np.full(len(JOINT_NAMES), -10.0, dtype=np.float32),
        "max": np.full(len(JOINT_NAMES), 10.0, dtype=np.float32),
    }

    with pytest.raises(ValueError, match="mutually exclusive"):
        apply_wrist_flex_start_override(pose, statistics, 1.0, 1.0)


def test_return_home_limits_each_command_even_with_zero_duration():
    robot = _FakeRobot()
    result = return_home(
        robot,
        {f"{name}.pos": 5.0 for name in JOINT_NAMES},
        duration_s=0,
        fps=1000,
        label="test",
        max_step_delta=1.0,
        tolerance=0.01,
        settle_timeout_s=0,
    )
    commands = np.array(
        [[item[f"{name}.pos"] for name in JOINT_NAMES] for item in robot.commands]
    )
    assert len(commands) == 5
    assert np.max(np.diff(np.vstack([np.zeros(len(JOINT_NAMES)), commands]), axis=0)) <= 1.0
    assert result["reached"]


def test_return_home_bypasses_policy_smoothing_and_reaches_target():
    robot = _FakeRobot()
    install_action_smoother(robot, alpha=0.7)
    install_action_rate_limiter(robot, max_delta=2.0)
    target = {f"{name}.pos": 100.0 for name in JOINT_NAMES}

    result = return_home(
        robot,
        target,
        duration_s=0,
        fps=1000,
        label="test",
        max_step_delta=1.0,
        tolerance=0.01,
        settle_timeout_s=0,
    )

    commands = np.array(
        [[item[f"{name}.pos"] for name in JOINT_NAMES] for item in robot.commands]
    )
    assert result["reached"]
    assert np.allclose(robot.bus.state, 100.0)
    assert np.max(
        np.abs(np.diff(np.vstack([np.zeros(len(JOINT_NAMES)), commands]), axis=0))
    ) <= 1.0 + 1e-5


def test_return_home_bypasses_rate_limiter_and_reaches_target():
    """return_home explicitly bypasses the EMA action smoother while ramping (see
    test_return_home_bypasses_policy_smoothing_and_reaches_target) because its own
    max_step_delta/timeout math is meant to be authoritative for the park move.
    install_action_rate_limiter's clamp has no equivalent bypass, so when its
    max_delta is tighter than return_home's max_step_delta, every ramp command is
    silently reclamped smaller than return_home assumed -- the arm keeps
    approaching the target but never gets there within the timeout return_home
    computed for a larger step size, and a fully reachable target is incorrectly
    reported as unreached.
    """
    robot = _FakeRobot()
    install_action_rate_limiter(robot, max_delta=0.1)
    target = {f"{name}.pos": 100.0 for name in JOINT_NAMES}

    result = return_home(
        robot,
        target,
        duration_s=0,
        fps=1000,
        label="test",
        max_step_delta=1.0,
        tolerance=0.01,
        settle_timeout_s=0,
    )

    assert result["reached"]
    assert np.allclose(robot.bus.state, 100.0)


def test_return_home_rejects_false_success_from_stalled_telemetry():
    class StalledRobot(_FakeRobot):
        def send_action(self, action):
            self.commands.append(action)
            return action

    robot = StalledRobot()

    with pytest.raises(RuntimeError, match="pose was not reached"):
        return_home(
            robot,
            {f"{name}.pos": 2.0 for name in JOINT_NAMES},
            duration_s=0,
            fps=1000,
            label="test",
            max_step_delta=1.0,
            tolerance=0.1,
            settle_timeout_s=0,
        )


def test_return_home_reports_pose_not_reached_despite_telemetry_read_failure():
    """The diagnostic "pose was not reached" RuntimeError must win even when a
    follow-up motor telemetry read (Goal_Position/Present_Position/etc.) itself
    raises -- e.g. the same communication fault that caused the stall also
    breaks a later register read. Without per-register error handling, that
    read exception replaces the informative pose-not-reached diagnostic with a
    much less useful raw bus-read error.
    """

    class FlakyBus(_FakeBus):
        def read(self, register, motor, *, normalize=False):
            if register == "Hardware_Error_Status":
                raise ConnectionError("simulated bus read failure")
            return super().read(register, motor, normalize=normalize)

    class StalledRobot(_FakeRobot):
        def send_action(self, action):
            self.commands.append(action)
            return action

    robot = StalledRobot()
    robot.bus = FlakyBus(robot.bus.state)

    with pytest.raises(RuntimeError, match="pose was not reached"):
        return_home(
            robot,
            {f"{name}.pos": 2.0 for name in JOINT_NAMES},
            duration_s=0,
            fps=1000,
            label="test",
            max_step_delta=1.0,
            tolerance=0.1,
            settle_timeout_s=0,
        )


def test_async_action_rate_limiter_clamps_every_joint_from_live_start():
    robot = _FakeRobot()
    install_action_rate_limiter(robot, max_delta=2.0)

    target = {f"{name}.pos": 20.0 for name in JOINT_NAMES}
    first = robot.send_action(target)
    second = robot.send_action(target)

    assert all(value == pytest.approx(2.0) for value in first.values())
    assert all(value == pytest.approx(4.0) for value in second.values())


def test_async_action_rate_limiter_clamps_against_live_position_not_stale_command():
    """install_action_rate_limiter's docstring says it limits "every final joint
    target relative to the previously sent command" -- but it treats its own echoed
    request as ground truth for "previously sent", never checking whether the arm
    actually got there. If a joint stalls (cannot reach its commanded targets --
    exactly the overloaded/binding failure mode this codebase's wrist-flex hardware
    has shown), the internal clamp reference keeps ratcheting forward every tick
    while the arm doesn't move, so once free the very next command can jump the arm
    by far more than max_delta relative to where it physically is -- defeating the
    entire purpose of the hard safety limit.
    """

    class StalledRobot(_FakeRobot):
        def send_action(self, action):
            self.commands.append(action)
            return action  # echoes the request; bus.state (the real joint) never moves

    robot = StalledRobot()
    install_action_rate_limiter(robot, max_delta=2.0)

    target = {f"{name}.pos": 0.0 for name in JOINT_NAMES}
    for _ in range(20):
        target = {key: value + 2.0 for key, value in target.items()}
        robot.send_action(target)

    live_position = robot.bus.state[0]
    assert live_position == 0.0  # the arm truly never moved

    sent = robot.send_action({f"{name}.pos": 1000.0 for name in JOINT_NAMES})
    assert abs(sent[f"{JOINT_NAMES[0]}.pos"] - live_position) <= 2.0


def test_async_action_rate_limiter_bounds_final_smoothed_command():
    robot = _FakeRobot()
    install_action_smoother(robot, alpha=0.7)
    install_action_rate_limiter(robot, max_delta=2.0)

    robot.send_action({f"{name}.pos": 20.0 for name in JOINT_NAMES})
    robot.send_action({f"{name}.pos": -20.0 for name in JOINT_NAMES})

    commands = np.array(
        [[item[f"{name}.pos"] for name in JOINT_NAMES] for item in robot.commands]
    )
    assert np.max(np.abs(np.diff(np.vstack([np.zeros(len(JOINT_NAMES)), commands]), axis=0))) <= 2.0


def test_action_smoother_rejects_alpha_outside_unit_interval():
    """install_action_rate_limiter validates max_delta ("must be a positive finite
    value"); install_action_smoother has no equivalent check on alpha, even though
    its own docstring formula (sent = alpha*target + (1-alpha)*prev) is only a valid
    low-pass blend -- guaranteed to stay between target and prev -- for alpha in
    [0, 1]. Outside that range the EMA extrapolates instead of blends, which matters
    because main() installs the smoother *before* the rate limiter, so the limiter's
    clamped output becomes the smoother's input: an out-of-range alpha would let the
    smoother's extrapolated output blow past --max-joint-delta's "Hard safety limit
    ... between consecutive commands for every joint" (e.g. alpha=-10.0 turns a
    command already clamped to 2.0 units away into one 20 units away).
    """
    robot = _FakeRobot()
    with pytest.raises(ValueError, match="alpha"):
        install_action_smoother(robot, alpha=-10.0)


def test_async_action_rate_limiter_rejects_nonfinite_policy_action():
    robot = _FakeRobot()
    install_action_rate_limiter(robot, max_delta=2.0)
    target = {f"{name}.pos": 0.0 for name in JOINT_NAMES}
    target[f"{JOINT_NAMES[0]}.pos"] = float("nan")

    with pytest.raises(ValueError, match="non-finite action"):
        robot.send_action(target)


def test_async_action_rate_limiter_rejects_unknown_joint():
    robot = _FakeRobot()
    install_action_rate_limiter(robot, max_delta=2.0)

    with pytest.raises(KeyError, match="unknown joint"):
        robot.send_action({"not_a_joint.pos": 0.0})


def test_launch_server_detects_dead_process_despite_port_already_listening():
    """launch_server's readiness loop only checked whether *something* accepts a
    connection on host:port, never whether the process it just spawned (proc) was
    still alive. If an unrelated/stale process already occupies the port -- e.g. a
    leftover from a previous crashed run -- and the freshly spawned server process
    dies immediately (for instance because it could not bind that same port), the
    old code would still report success and hand back an already-dead proc.
    """
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        host, port = listener.getsockname()

        class DeadProc:
            returncode = 1

            def poll(self):
                return self.returncode

            def terminate(self):
                pass

        with pytest.raises(RuntimeError, match="exited"):
            launch_server(
                host,
                port,
                fps=30,
                log_path=None,
                timeout=1.0,
                policy_type="custom_act",
                popen=lambda *args, **kwargs: DeadProc(),
            )


def test_visualize_action_queue_size_if_present_skips_empty_queue():
    """The real visualize_action_queue_size calls max() on the list, which raises
    ValueError on an empty queue. This call sits in run_robot_client's cleanup
    finally block, so an unguarded raise there would mask whatever exception is
    already propagating out of cleanup (e.g. a watchdog safety-abort reason).
    """
    calls = []

    visualize_action_queue_size_if_present([], visualize=calls.append)

    assert calls == []


def test_visualize_action_queue_size_if_present_forwards_nonempty_queue():
    calls = []

    visualize_action_queue_size_if_present([1, 2, 3], visualize=calls.append)

    assert calls == [[1, 2, 3]]


def test_run_guarded_cleanup_swallows_exception_and_returns_default():
    """Cleanup steps that sit ahead of the robot shutdown/return-home/disconnect
    sequence in run_robot_client's finally block (trial-capture writer close,
    camera preview close) must never raise: doing so would skip parking,
    disconnect, and torque-disable entirely.
    """

    def boom():
        raise RuntimeError("writer flush failed")

    result = run_guarded_cleanup("trial capture cleanup", boom, default="sentinel")

    assert result == "sentinel"


def test_run_guarded_cleanup_returns_action_result_on_success():
    result = run_guarded_cleanup("trial capture cleanup", lambda: {"trial": 1})

    assert result == {"trial": 1}


def test_trial_has_recorded_actions_requires_at_least_one_action():
    """Outcome-label eligibility must key off action_count, not observation_count:
    observations are sent to the policy server on a fixed cadence to request
    actions regardless of whether any action has arrived and executed yet, so a
    trial that ends before the first action chunk arrives can have
    observation_count > 0 with action_count == 0 -- the robot never did
    anything, so it shouldn't be eligible for a success/failure label.
    """
    assert trial_has_recorded_actions(0) is False
    assert trial_has_recorded_actions(1) is True
    assert trial_has_recorded_actions(5) is True


def test_locked_robot_client_disconnects_robot_on_partial_construction_failure():
    """RobotClient.__init__ connects the robot (self.robot.connect()) well before
    it finishes -- setting up the gRPC channel, RemotePolicyConfig, etc. can still
    raise afterward. If construction fails past that point, the partially-built
    instance is discarded and nothing else holds a reference to the now-connected
    robot, leaking an open hardware connection. LockedRobotClient must disconnect
    the robot before letting the exception propagate.
    """
    disconnected = []

    class FakeRobot:
        def connect(self):
            pass

        def disconnect(self):
            disconnected.append(True)

    def fake_init(self, *args, **kwargs):
        self.robot = FakeRobot()
        self.robot.connect()
        raise RuntimeError("boom after robot.connect()")

    with patch.object(RobotClient, "__init__", fake_init):
        with pytest.raises(RuntimeError, match="boom after robot.connect"):
            LockedRobotClient()

    assert disconnected == [True]


def test_wrist_flex_seats_from_below_when_descending():
    """Regression: parking failed with wrist_flex 25 ticks short of goal.

    Wrist-flex sticks moving negative (~0.9 units short) but tracks positive to
    ~0.11. Approaching a lower target directly ends on the sticky direction, so
    the move must first go below the target and come back up.
    """
    from models.act.eval_common import move_to_joint_state
    from utils.config import JOINT_NAMES

    wrist = JOINT_NAMES.index("wrist_flex")
    target = np.zeros(len(JOINT_NAMES), dtype=np.float32)
    target[wrist] = 46.5
    commands = []

    class Robot:
        def __init__(self):
            self.state = {f"{n}.pos": 0.0 for n in JOINT_NAMES}
            # Start ABOVE the target so the move descends.
            self.state["wrist_flex.pos"] = 50.0

        def get_observation(self):
            return dict(self.state)

        def send_action(self, action):
            commands.append(action["wrist_flex.pos"])
            self.state.update(action)
            return dict(action)

    move_to_joint_state(
        Robot(),
        target=target,
        timeout=1.0,
        ramp_seconds=0.2,
        hz=30,
        max_delta=2.0,
        tolerance=0.5,
        wrist_flex_seat_offset=3.0,
    )

    # It must dip below the 46.5 target before settling up onto it.
    assert min(commands) < 46.5, "never seated below the target"
    assert commands[-1] >= min(commands), "final approach must move upward"


def test_wrist_flex_seating_skipped_when_ascending():
    """Rising into the target is already the reliable direction."""
    from models.act.eval_common import move_to_joint_state
    from utils.config import JOINT_NAMES

    wrist = JOINT_NAMES.index("wrist_flex")
    target = np.zeros(len(JOINT_NAMES), dtype=np.float32)
    target[wrist] = 46.5
    commands = []

    class Robot:
        def __init__(self):
            self.state = {f"{n}.pos": 0.0 for n in JOINT_NAMES}
            self.state["wrist_flex.pos"] = 44.0

        def get_observation(self):
            return dict(self.state)

        def send_action(self, action):
            commands.append(action["wrist_flex.pos"])
            self.state.update(action)
            return dict(action)

    move_to_joint_state(
        Robot(),
        target=target,
        timeout=1.0,
        ramp_seconds=0.2,
        hz=30,
        max_delta=2.0,
        tolerance=0.5,
        wrist_flex_seat_offset=3.0,
    )

    assert min(commands) >= 44.0 - 1e-6, "should not dip below when already rising"
