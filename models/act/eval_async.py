"""Run a LeRobot policy live on the OMX follower using *asynchronous inference*.

This fixes the "MPS runs ~10 Hz but we trained at 30 Hz" jerk. A ``PolicyServer``
(here on MPS, locally) returns a full action chunk via ``predict_action_chunk``
-- bypassing the checkpoint's ``n_action_steps`` / ``temporal_ensemble_coeff`` --
and a ``RobotClient`` replays that queue at ``--fps``, fetching the next chunk
before the current one drains so there are no idle frames. Overlapping chunks are
blended with ``--aggregate-fn`` (the async successor to temporal ensembling).

This script launches the server for you and runs the client in one command. Point
``--server-address`` at a remote GPU with ``--no-launch-server`` to offload inference.

Example:
    uv run python models/act/eval_async.py --task "Place the sponge in the bowl"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import select
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from lerobot.async_inference.configs import RobotClientConfig
from lerobot.async_inference.helpers import visualize_action_queue_size
from lerobot.async_inference.robot_client import RobotClient
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.omx_follower.config_omx_follower import OmxFollowerConfig
from lerobot.robots.omx_follower.omx_follower import OmxFollower
from lerobot.transport import services_pb2

from models.act.camera_alignment import (
    diagnostic_dataset_reference,
    load_session_diagnostics,
    resolve_dataset_start_pose,
)
from models.act.lerobot_policy import ADVANTAGE_ENV_VAR
from models.act.eval_common import (
    clear_stuck_port,
    dataset_start_statistics,
    joint_positions,
    move_to_joint_state,
    start_state_violations,
)
from models.act.rollout_capture import (
    RolloutArtifactWriter,
    prompt_for_outcome,
    write_metadata_atomic,
)
from utils.config import CAMERAS, CAMERA_HEIGHT, CAMERA_WIDTH, FOLLOWER_PORT, FPS, JOINT_NAMES

DEFAULT_CKPT = (
    REPO_ROOT
    / "outputs/vastai/instance-43164542/outputs/lerobot_official"
    / "lerobot_official_act_chunk30_steps20000/checkpoints/020000/pretrained_model"
)


def force_no_calibration() -> None:
    """OMX arms ship pre-calibrated, but the async client calls connect(calibrate=True).
    Patch connect() to skip calibration, matching the synchronous eval."""
    original = OmxFollower.connect
    OmxFollower.connect = lambda self, calibrate=False: original(self, calibrate=False)


def camera_config(name: str, fourcc: str | None, warmup_s: int) -> OpenCVCameraConfig:
    if name not in CAMERAS:
        raise KeyError(f"camera '{name}' is not configured in utils.config.CAMERAS")
    # MJPG keeps USB bandwidth low so both cams hold 640x480@30.
    return OpenCVCameraConfig(
        index_or_path=CAMERAS[name], fps=FPS, width=CAMERA_WIDTH, height=CAMERA_HEIGHT,
        warmup_s=warmup_s, fourcc=fourcc,
    )


class TopCameraPreview:
    WINDOW_NAME = "ACT eval - top camera"

    def __init__(self, camera, *, cv2_module=None) -> None:
        if cv2_module is None:
            import cv2 as cv2_module

        self.camera = camera
        self.cv2 = cv2_module
        self.closed = False
        self.shown = False

    def update(self) -> None:
        if self.closed:
            return
        frame = np.asarray(self.camera.read_latest())
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError(
                f"Top camera preview requires HWC uint8 RGB frames, got "
                f"shape={frame.shape} dtype={frame.dtype}"
            )
        display_frame = self.cv2.cvtColor(frame, self.cv2.COLOR_RGB2BGR)
        self.cv2.imshow(self.WINDOW_NAME, display_frame)
        self.shown = True
        key = self.cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            self.close()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.shown:
            self.cv2.destroyWindow(self.WINDOW_NAME)
            self.cv2.waitKey(1)


def preflight_cameras(names: list[str], fourcc: str | None, warmup_s: int, retries: int) -> None:
    """Open each camera, verify it delivers a real CAMERA_WIDTHxCAMERA_HEIGHT frame, then release.

    macOS/AVFoundation sometimes cold-starts a USB cam at 640x360 or leaves it in a bad state
    from a prior run; opening+closing here verifies the device and clears that state before the
    robot client grabs it. A benign 'failed to set fourcc' warning is expected on macOS.
    """
    want = (CAMERA_HEIGHT, CAMERA_WIDTH)
    for name in names:
        idx = CAMERAS[name]
        for attempt in range(1, retries + 1):
            cam = OpenCVCamera(camera_config(name, fourcc, warmup_s))
            try:
                cam.connect()
                shape = cam.async_read(timeout_ms=2000).shape
                if shape[:2] == want:
                    print(f"[preflight] {name} (index {idx}) OK -> {shape}", flush=True)
                    break
                print(f"[preflight] {name} (index {idx}) {attempt}/{retries}: got {shape[:2]}, want {want}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[preflight] {name} (index {idx}) {attempt}/{retries} failed: {exc}", flush=True)
            finally:
                try:
                    cam.disconnect()
                except Exception:  # noqa: BLE001
                    pass
            time.sleep(1.0)
        else:
            raise RuntimeError(
                f"camera '{name}' (index {idx}) could not deliver {CAMERA_WIDTH}x{CAMERA_HEIGHT}@{FPS} after "
                f"{retries} attempts. Unplug/replug it (or free it from another process) and retry."
            )
    time.sleep(0.5)  # let the robot client re-open the freed devices cleanly


def setup_file_logging(log_dir: Path, ts: str) -> tuple[Path, Path]:
    """Tee LeRobot client logs to a file; return (client_log, server_log) paths."""
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_dir / f"client_{ts}.log")
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    return log_dir / f"client_{ts}.log", log_dir / f"policy_server_{ts}.log"


def launch_server(
    host: str,
    port: int,
    fps: int,
    log_path: Path | None,
    timeout: float,
    policy_type: str,
    advantage: str | None = None,
    *,
    popen=subprocess.Popen,
) -> subprocess.Popen:
    """Launch the local policy server and block until its port is accepting connections."""
    if policy_type == "custom_act":
        cmd = [
            sys.executable,
            "-m",
            "models.act.lerobot_policy",
            "serve",
            f"--host={host}",
            f"--port={port}",
            f"--fps={fps}",
        ]
    else:
        cmd = [
            sys.executable,
            "-m",
            "lerobot.async_inference.policy_server",
            f"--host={host}",
            f"--port={port}",
            f"--fps={fps}",
        ]
    # The policy is constructed inside the server process, so the conditioning
    # token has to travel with the environment rather than as a local default.
    env = None
    if advantage:
        env = {**os.environ, ADVANTAGE_ENV_VAR: advantage}
    print(f"[eval_async] launching policy server: {' '.join(cmd)}", flush=True)
    stdout = open(log_path, "w") if log_path else None  # noqa: SIM115 (open for process lifetime)
    if log_path:
        print(f"[eval_async] policy server log -> {log_path}", flush=True)
    proc = popen(cmd, cwd=str(REPO_ROOT), stdout=stdout,
                            stderr=subprocess.STDOUT if stdout else None, env=env)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"policy server process exited early with code {proc.returncode} "
                f"before {host}:{port} became ready"
            )
        with socket.socket() as sock:
            sock.settimeout(1.0)
            if sock.connect_ex((host, port)) == 0:
                print(f"[eval_async] policy server ready on {host}:{port}", flush=True)
                return proc
        time.sleep(0.3)
    proc.terminate()
    raise RuntimeError(f"policy server did not open {host}:{port} within {timeout}s")


def resolve_home_pose(target: str, robot, policy_path: Path) -> dict[str, float]:
    """Where the arm parks on Ctrl-C, as {'<joint>.pos': deg} in action_features order.

        start         live pose at rollout start
        zeros         all joints 0 (upright, gripper closed)
        mean          mean over ALL dataset frames -- a mid-reach pose, can hit the table
        dataset_start mean of each episode's FIRST frame -- the safe ready pose demos begin
                      from, gripper open (recommended)
    """
    keys = list(robot.action_features)
    if target == "start":
        return joint_positions(robot)
    if target == "zeros":
        return dict.fromkeys(keys, 0.0)

    if target == "mean":
        from safetensors.torch import load_file

        stats_file = next(iter(sorted(policy_path.glob("*unnormalizer*.safetensors"))), None)
        if stats_file is None:
            raise FileNotFoundError(f"no unnormalizer safetensors in {policy_path}")
        stats = load_file(str(stats_file))
        vec = next((v for k, v in stats.items() if k.endswith("action.mean")), None)
        if vec is None:
            raise KeyError(f"'action.mean' not found in {stats_file.name}")
        vec = vec.flatten().tolist()
    else:  # dataset_start
        import json

        from lerobot.datasets import LeRobotDataset

        repo_id = json.loads((policy_path / "train_config.json").read_text())["dataset"]["repo_id"]
        ds = LeRobotDataset(repo_id)
        ep = np.asarray(ds.hf_dataset["episode_index"])
        fr = np.asarray(ds.hf_dataset["frame_index"])
        firsts = [int(np.where(ep == e)[0][np.argmin(fr[ep == e])]) for e in np.unique(ep)]
        vec = np.stack([ds.hf_dataset["observation.state"][i] for i in firsts]).astype("float32").mean(0).tolist()

    if len(vec) != len(keys):
        raise ValueError(f"home vector has {len(vec)} dims but robot has {len(keys)} joints")
    return {k: float(vec[i]) for i, k in enumerate(keys)}


def dataset_reference(policy_path: Path) -> tuple[str, str]:
    train_config_path = policy_path / "train_config.json"
    if not train_config_path.exists():
        raise FileNotFoundError(f"no train_config.json in {policy_path}")
    dataset = json.loads(train_config_path.read_text())["dataset"]
    return dataset["repo_id"], dataset.get("revision") or "main"


def start_state_reference(
    policy_path: Path,
    repo_id_override: str | None,
    revision_override: str | None,
) -> tuple[str, str]:
    policy_repo_id, policy_revision = dataset_reference(policy_path)
    repo_id = repo_id_override or policy_repo_id
    revision = revision_override or (
        policy_revision if repo_id_override is None else "main"
    )
    return repo_id, revision


def apply_wrist_flex_start_override(
    pose: dict[str, float],
    statistics: dict[str, np.ndarray],
    wrist_flex: float | None,
) -> dict[str, float]:
    updated = dict(pose)
    if wrist_flex is None:
        return updated
    if not np.isfinite(wrist_flex):
        raise ValueError("--start-state-wrist-flex must be finite.")

    index = JOINT_NAMES.index("wrist_flex")
    minimum = float(statistics["min"][index])
    maximum = float(statistics["max"][index])
    if not minimum <= wrist_flex <= maximum:
        raise ValueError(
            f"--start-state-wrist-flex={wrist_flex:g} is outside dataset start "
            f"support [{minimum:.2f}, {maximum:.2f}]."
        )
    updated["wrist_flex.pos"] = float(wrist_flex)
    print(
        f"[eval_async] overriding dataset-start wrist_flex to {wrist_flex:.2f} "
        f"joint units (dataset support {minimum:.2f}..{maximum:.2f}).",
        flush=True,
    )
    return updated


def install_action_smoother(robot, alpha: float) -> None:
    """Wrap ``robot.send_action`` with a per-joint EMA low-pass to damp residual jitter.

    sent[j] = alpha*target[j] + (1-alpha)*prev[j]. The gripper is passed through
    unfiltered so open/close stays crisp. The filter state seeds from the first
    target (no startup lurch). The return_home sweep also goes through this filter,
    but it already interpolates slowly so the added lag is negligible.
    """
    if not np.isfinite(alpha) or not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be within [0, 1].")

    raw_send = robot.send_action
    state: dict[str, float] = {}

    def smoothed_send(action: dict[str, float]):
        if getattr(robot, "_act_eval_bypass_action_smoothing", False):
            out = action
        else:
            out = {}
            for k, v in action.items():
                if "gripper" in k or k not in state:
                    out[k] = v
                else:
                    out[k] = alpha * v + (1.0 - alpha) * state[k]
        sent = raw_send(out)
        state.update({key: float(value) for key, value in sent.items()})
        return sent

    robot.send_action = smoothed_send


def install_action_rate_limiter(robot, max_delta: float) -> None:
    """Limit every final joint target relative to the robot's live position."""
    if not np.isfinite(max_delta) or max_delta <= 0:
        raise ValueError("max_delta must be a positive finite value.")

    raw_send = robot.send_action
    warning_state = {"last_print": 0.0}

    def limited_send(action: dict[str, float]):
        # return_home() sets this while it drives its own verified ramp (max_step_delta
        # already bounds each command); without this check its ramp/timeout math gets
        # silently re-clamped by a tighter max_delta here and can time out before the
        # target is reached, even though it is fully reachable given more time.
        bypass = getattr(robot, "_act_eval_bypass_action_smoothing", False)
        # Clamp against the robot's live position, not the last *requested* value: if a
        # joint stalls (cannot reach its commanded targets -- an overloaded/binding
        # joint), tracking only the echoed request would let the clamp reference
        # ratchet forward every tick while the arm doesn't move, so once free the arm
        # could be commanded to jump by far more than max_delta relative to reality.
        live = joint_positions(robot)
        limited = {}
        clipped = {}
        for key, value in action.items():
            if key not in live:
                raise KeyError(f"Policy produced an action for unknown joint {key!r}.")
            value = float(value)
            if not np.isfinite(value):
                raise ValueError(f"Policy produced a non-finite action for {key}: {value}")
            if bypass:
                bounded = value
            else:
                prior = live[key]
                bounded = float(np.clip(value, prior - max_delta, prior + max_delta))
            limited[key] = bounded
            if bounded != value:
                clipped[key] = (value, bounded)

        sent = raw_send(limited)

        now = time.monotonic()
        if clipped and now - warning_state["last_print"] >= 1.0:
            details = ", ".join(
                f"{key} {requested:.2f}->{bounded:.2f}"
                for key, (requested, bounded) in clipped.items()
            )
            print(
                f"[eval_async] SAFETY CLAMP active "
                f"(max {max_delta:g} joint units/command): {details}",
                flush=True,
            )
            warning_state["last_print"] = now
        return sent

    robot.send_action = limited_send


def return_home(
    robot,
    target: dict[str, float],
    duration_s: float,
    fps: int,
    label: str,
    *,
    max_step_delta: float | None = None,
    tolerance: float = 1.0,
    settle_timeout_s: float = 2.0,
) -> dict:
    """Move to ``target`` safely and verify the final pose from motor telemetry."""
    if fps <= 0:
        raise ValueError("fps must be positive.")
    if max_step_delta is not None and max_step_delta <= 0:
        raise ValueError("max_step_delta must be positive when provided.")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive.")
    if settle_timeout_s < 0:
        raise ValueError("settle_timeout_s must be non-negative.")
    try:
        current = joint_positions(robot)
    except ConnectionError as exc:
        if "Port is in use" not in str(exc) or not clear_stuck_port(robot):
            raise
        print("[eval_async] recovered from a stuck serial port (Ctrl-C landed mid-transaction); retrying.", flush=True)
        current = joint_positions(robot)
    max_distance = max(abs(target[key] - current[key]) for key in target)
    rate_limited_duration = (
        max_distance / (max_step_delta * fps)
        if max_step_delta is not None
        else 0.0
    )
    ramp_duration = max(duration_s, rate_limited_duration, 1 / fps)
    print(
        f"\n[eval_async] returning to {label} pose over at least "
        f"{ramp_duration:.1f}s (verify within {tolerance:g} joint units)...",
        flush=True,
    )
    for k in sorted(target):
        print(f"[eval_async]   {k:18s} {current.get(k, float('nan')):8.2f} -> {target[k]:8.2f} "
              f"(Δ{target[k] - current.get(k, target[k]):+.2f})", flush=True)
    target_vector = np.array([target[f"{name}.pos"] for name in JOINT_NAMES], dtype=np.float32)
    previous_bypass = getattr(robot, "_act_eval_bypass_action_smoothing", False)
    robot._act_eval_bypass_action_smoothing = True
    try:
        result = move_to_joint_state(
            robot,
            target=target_vector,
            timeout=ramp_duration + settle_timeout_s,
            ramp_seconds=ramp_duration,
            hz=fps,
            max_delta=max_step_delta or 0,
            tolerance=tolerance,
        )
    finally:
        robot._act_eval_bypass_action_smoothing = previous_bypass
    if not result["reached"]:
        motor = result["worst_joint"]
        registers = (
            "Goal_Position",
            "Present_Position",
            "Torque_Enable",
            "Hardware_Error_Status",
            "Operating_Mode",
        )
        motor_status = {}
        for register in registers:
            try:
                motor_status[register] = robot.bus.read(register, motor, normalize=False)
            except Exception as exc:  # noqa: BLE001 -- always raise the pose-not-reached error below
                motor_status[register] = f"<read failed: {exc}>"
        raise RuntimeError(
            f"{label} pose was not reached: worst joint={result['worst_joint']} "
            f"max error={result['max_abs_error']:.2f} joint units after "
            f"{ramp_duration + settle_timeout_s:.1f}s; raw motor status={motor_status}"
        )
    print(
        f"[eval_async] {label} pose reached "
        f"(max error {result['max_abs_error']:.2f} joint units).",
        flush=True,
    )
    return result


def heartbeat(client, interval: float = 2.0) -> None:
    """Periodic liveness line, since the control loop is otherwise silent."""
    start = time.time()
    while not client.shutdown_event.wait(interval):
        with client.latest_action_lock:
            la = client.latest_action
        print(f"[eval_async] alive t={time.time() - start:5.1f}s  latest_action={la}  "
              f"queue={client.action_queue.qsize()}", flush=True)


class ActionStreamWatchdog:
    def __init__(
        self,
        client,
        *,
        first_action_timeout: float,
        starvation_timeout: float,
        poll_interval: float = 0.1,
    ):
        self.client = client
        self.first_action_timeout = first_action_timeout
        self.starvation_timeout = starvation_timeout
        self.poll_interval = poll_interval
        self.reason: str | None = None

    def run(self) -> None:
        started = time.perf_counter()
        last_progress = started
        last_action = -1
        received_actions = False
        while not self.client.shutdown_event.wait(self.poll_interval):
            now = time.perf_counter()
            trial_active = getattr(self.client, "trial_active", None)
            if trial_active is not None and not trial_active.is_set():
                started = now
                last_progress = now
                last_action = -1
                received_actions = False
                continue
            with self.client.latest_action_lock:
                latest_action = self.client.latest_action
            with self.client.action_queue_lock:
                queue_size = self.client.action_queue.qsize()

            if latest_action > last_action:
                last_action = latest_action
                last_progress = now
                received_actions = True
            elif queue_size > 0 and not received_actions:
                last_progress = now
                received_actions = True

            if (
                not received_actions
                and self.first_action_timeout > 0
                and now - started >= self.first_action_timeout
            ):
                self.reason = (
                    "no first action chunk received within "
                    f"{self.first_action_timeout:g}s"
                )
            elif (
                received_actions
                and queue_size == 0
                and self.starvation_timeout > 0
                and now - last_progress >= self.starvation_timeout
            ):
                self.reason = (
                    "action queue remained empty for "
                    f"{self.starvation_timeout:g}s"
                )

            if self.reason is not None:
                print(f"[eval_async] watchdog abort: {self.reason}", flush=True)
                self.client.shutdown_event.set()
                return


class LockedRobotClient(RobotClient):
    def __init__(self, *args, **kwargs):
        try:
            super().__init__(*args, **kwargs)
        except BaseException:
            # RobotClient.__init__ connects the robot (self.robot.connect()) well
            # before it finishes (gRPC channel setup, RemotePolicyConfig, etc. can
            # still raise afterward). If construction fails past that point, the
            # partially-built instance is discarded and nothing else holds a
            # reference to the now-connected robot -- leaking an open hardware
            # connection. Disconnect it here before letting the exception
            # propagate.
            robot = getattr(self, "robot", None)
            if robot is not None:
                try:
                    robot.disconnect()
                except Exception:  # noqa: BLE001 -- already unwinding a failure
                    pass
            raise
        self.action_queue_lock = threading.RLock()
        self.trial_active = threading.Event()
        self._control_loop_started = False
        self.trial_started_at = 0.0
        self.observation_callback = None
        self.action_callback = None
        self.loop_callback = None

    def _aggregate_action_queues(self, *args, **kwargs):
        with self.action_queue_lock:
            trial_active = getattr(self, "trial_active", None)
            if trial_active is not None and not trial_active.is_set():
                return
            trial_started_at = getattr(self, "trial_started_at", 0.0)
            if args and trial_started_at:
                incoming_actions = args[0]
                if (
                    incoming_actions
                    and incoming_actions[0].get_timestamp() < trial_started_at
                ):
                    return
            # Keep the lock across upstream's deque snapshot and queue swap.
            return super()._aggregate_action_queues(*args, **kwargs)

    def pause_trial(self) -> None:
        self.trial_active.clear()
        with self.action_queue_lock:
            while not self.action_queue.empty():
                self.action_queue.get_nowait()

    def prepare_trial(self) -> None:
        self.pause_trial()
        self.stub.Ready(services_pb2.Empty())
        with self.action_queue_lock:
            with self.latest_action_lock:
                self.latest_action = -1
            self.action_chunk_size = -1
            self.must_go.set()
            self.trial_started_at = time.time()
            self.trial_active.set()

    def control_loop_observation(self, task: str, verbose: bool = False):
        observation = super().control_loop_observation(task, verbose)
        callback = self.observation_callback
        if observation is not None and callback is not None:
            callback(observation)
        return observation

    def control_trial(
        self,
        *,
        task: str,
        duration_s: float,
        stop_requested,
        verbose: bool = False,
        clock=time.perf_counter,
        sleep=time.sleep,
    ) -> str:
        if not self._control_loop_started:
            self.start_barrier.wait()
            self._control_loop_started = True

        started = clock()
        reason = "shutdown"
        while self.running and self.trial_active.is_set():
            loop_started = clock()
            if stop_requested():
                # self.running can flip to False (shutdown_event set by the watchdog
                # thread) between the while-condition check above and here; without
                # this re-check a watchdog abort racing an operator/duration stop in
                # the same iteration would be reported as "operator"/"duration"
                # instead of "shutdown", and after_trial() would park the arm despite
                # the abort.
                reason = "operator" if self.running else "shutdown"
                break
            if duration_s > 0 and loop_started - started >= duration_s:
                reason = "duration" if self.running else "shutdown"
                break

            if self.actions_available():
                self.control_loop_action(verbose)
            if self._ready_to_send_observation():
                self.control_loop_observation(task, verbose)
            loop_callback = getattr(self, "loop_callback", None)
            if loop_callback is not None:
                loop_callback()

            sleep(max(0.0, self.config.environment_dt - (clock() - loop_started)))

        self.pause_trial()
        return reason


def run_trial_sequence(
    client: LockedRobotClient,
    *,
    trials: int,
    task: str,
    duration_s: float,
    stop_requested_factory,
    before_trial,
    after_trial,
) -> list[str]:
    reasons = []
    for trial in range(1, trials + 1):
        before_trial(trial)
        client.prepare_trial()
        reason = client.control_trial(
            task=task,
            duration_s=duration_s,
            stop_requested=stop_requested_factory(trial),
        )
        reasons.append(reason)
        after_trial(trial, reason)
        if reason == "shutdown" or not getattr(client, "running", True):
            break
    return reasons


def stdin_enter_pressed(stream=sys.stdin) -> bool:
    if not stream.isatty():
        return False
    readable, _, _ = select.select([stream], [], [], 0)
    if not readable:
        return False
    stream.readline()
    return True


def prompt_for_trial_start(
    *,
    trial: int,
    total_trials: int,
    stream=sys.stdin,
) -> bool:
    if not stream.isatty():
        print(
            f"[eval_async] Trial {trial}/{total_trials}: stdin is non-interactive; "
            "starting without a repositioning prompt.",
            flush=True,
        )
        return False
    print(
        f"\n[eval_async] Trial {trial}/{total_trials}: position the objects, "
        "then press Enter to start...",
        flush=True,
    )
    return bool(stream.readline())


def trial_stop_instructions(
    *,
    duration_s: float,
    enter_enabled: bool,
    stdin_interactive: bool,
) -> str:
    controls = []
    if enter_enabled and stdin_interactive:
        controls.append("press Enter to end early")
    if duration_s > 0:
        controls.append(f"ends after {duration_s:g}s")
    controls.append("Ctrl-C stops the session")
    return "; ".join(controls)


def should_park_after_trial(reason: str) -> bool:
    return reason != "shutdown"


def trial_has_recorded_actions(action_count: int) -> bool:
    """Outcome-label eligibility must key off action_count, not observation_count.

    Observations are sent to the policy server on a fixed cadence to request
    actions, independent of whether any action has arrived and executed yet.
    A trial that ends (operator stop, duration timeout, or interruption)
    before the first action chunk ever arrives can have observation_count > 0
    with action_count == 0 -- the robot never actually did anything, so
    prompting for a success/failure label on it would be meaningless.
    """
    return action_count > 0


def install_trace_logger(robot, trace_file=None, entry_callback=None) -> None:
    """Wrap robot.send_action to log qpos + target + actually-sent action per control tick, for
    apples-to-apples comparison against models/act/eval_async_custom.py's trace.

    Install AFTER install_action_smoother (main() does this): by the time this wraps
    robot.send_action, `send_action` already points at smoothed_send, so `action` here is the
    PRE-smoothing target and the return value of raw_send(action) is the actual smoothed value
    written to the bus -- giving us both target_action and sent_action for free.
    """
    raw_send = robot.send_action
    state = {"step": 0, "start": time.perf_counter()}

    def traced_send(action: dict[str, float]):
        sent = raw_send(action)
        try:
            qpos = joint_positions(robot)
        except Exception:  # noqa: BLE001 -- trace logging must never crash the control loop
            qpos = None
        entry = {
            "step": state["step"],
            "elapsed_s": time.perf_counter() - state["start"],
            "qpos": qpos,
            "target_action": {k: float(v) for k, v in action.items()},
            "sent_action": {k: float(v) for k, v in sent.items()},
        }
        if trace_file is not None:
            trace_file.write(json.dumps(entry) + "\n")
            trace_file.flush()
        if entry_callback is not None:
            entry_callback(entry)
        state["step"] += 1
        return sent

    robot.send_action = traced_send


def record_rollout_action_transition(
    *,
    client,
    writer: RolloutArtifactWriter,
    entry: dict,
    camera_names: list[str],
) -> None:
    """Record an action against its preceding observation, then its next state."""
    writer.record_action(entry)
    qpos = entry.get("qpos")
    if not isinstance(qpos, dict):
        raise RuntimeError(
            "Could not capture post-action joint state for RECAP rollout"
        )
    frames = {
        camera: client.robot.cameras[camera].read_latest()
        for camera in camera_names
    }
    writer.record_observation(
        frames,
        {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "source": "post_action",
            "policy_timestep": entry["step"] + 1,
            "qpos": qpos,
        },
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy-path", default=str(DEFAULT_CKPT), help="pretrained_model checkpoint dir")
    p.add_argument("--policy-type", default="act")
    p.add_argument(
        "--advantage",
        default="positive",
        choices=["positive", "negative", "unconditioned"],
        help=(
            "Advantage token a RECAP policy is conditioned on. Deployment asks for "
            "positive; the others are for ablations. Ignored by non-RECAP checkpoints."
        ),
    )
    p.add_argument("--task", default="Place the sponge in the bowl")
    p.add_argument("--robot-port", default=FOLLOWER_PORT, help="OMX follower serial port")
    p.add_argument("--robot-id", default="omx_follower_arm")
    p.add_argument("--cameras", default="top,wrist", help="comma-separated names from utils.config.CAMERAS")
    p.add_argument(
        "--show-top-camera",
        action="store_true",
        help="Show the live top-camera feed during each trial; q or Esc closes only the preview.",
    )
    p.add_argument("--camera-fourcc", default="MJPG", help="FOURCC for all cams ('' to auto-detect; wrist needs MJPG)")
    p.add_argument("--camera-warmup-s", type=int, default=5)
    p.add_argument("--no-camera-preflight", action="store_true", help="skip the camera verify/clear-state step")
    p.add_argument("--camera-preflight-retries", type=int, default=3)
    p.add_argument(
        "--camera-view-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Compare the live top-camera background against training episode starts "
            "and print PASS/WARN/LOW CONFIDENCE (screening only, not calibration)."
        ),
    )
    p.add_argument(
        "--camera-reference-count",
        type=int,
        default=12,
        help="Maximum number of training start frames used by --camera-view-check.",
    )
    p.add_argument(
        "--diagnostic-dataset-repo-id",
        default=None,
        help=(
            "Dataset used for scene/camera baselines. Defaults to the policy "
            "training dataset; DAgger policies should pass the seed dataset."
        ),
    )
    p.add_argument(
        "--diagnostic-dataset-revision",
        default=None,
        help="Revision for --diagnostic-dataset-repo-id (default: main).",
    )

    p.add_argument("--fps", type=int, default=FPS, help="control-loop rate the robot replays actions at")
    p.add_argument("--policy-device", default="mps", choices=["mps", "cuda", "cpu", "xpu"])
    p.add_argument("--client-device", default="cpu")
    p.add_argument("--actions-per-chunk", type=int, default=30, help="actions per inference (<= policy chunk_size)")
    p.add_argument("--chunk-size-threshold", type=float, default=0.4,
                   help="re-request a chunk when the queue drops to this fraction (0=fully open-loop, 1=replan every step). "
                        "Both extremes are jerky: HIGH (>=0.7) replans faster than the queue drains (~85ms/~2.5-step "
                        "inference) and overwrites the imminent action with stale predictions -> oscillation; LOW (<=0.2) "
                        "gives big jumps at each rare seam and never re-aims. ~0.4 replans every ~18 steps (>> latency), "
                        "overwriting future not imminent actions, while draining far enough to start aiming (t~18).")
    p.add_argument("--aggregate-fn", default="conservative",
                   help="blend old vs incoming chunk per timestep. conservative=0.3*new (damps jitter, recommended), "
                        "average=0.5*new, weighted_average=0.7*new (reactive but jerky), latest_only=overwrite.")
    p.add_argument("--action-smoothing", type=float, default=0.7,
                   help="EMA low-pass on executed joint targets: sent = a*target + (1-a)*prev. "
                        "1.0 = off (raw targets), lower = smoother but laggier. ~0.6-0.8 removes residual jitter; "
                        "gripper is exempt so grasps stay crisp.")
    p.add_argument(
        "--max-joint-delta",
        type=float,
        default=2.0,
        help=(
            "Hard safety limit in normalized joint units between consecutive commands for every joint, "
            "including the gripper (default: 2.0; must be positive)."
        ),
    )
    p.add_argument("--visualize-queue", action="store_true", help="plot action-queue size on exit")
    p.add_argument("--trials", type=int, default=1, help="Number of trials to run without reconnecting hardware.")
    p.add_argument(
        "--trial-duration",
        type=float,
        default=None,
        help=(
            "Maximum seconds per trial. Defaults to 60 for multi-trial sessions; "
            "a single trial keeps the legacy no-timeout behavior."
        ),
    )
    p.add_argument(
        "--enter-to-end-trial",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow Enter to end the current trial early without stopping the session.",
    )
    p.add_argument(
        "--wait-for-trial-start",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Wait for Enter after object repositioning. Defaults to enabled when --trials > 1.",
    )

    p.add_argument("--return-home-s", type=float, default=3.0, help="minimum seconds to park the arm after a trial or on exit (0 disables)")
    p.add_argument(
        "--return-home-settle-timeout-s",
        type=float,
        default=6.0,
        help=(
            "Additional seconds to hold the final park command while motor telemetry "
            "converges (default: 6)."
        ),
    )
    p.add_argument("--home-target", default="dataset_start", choices=["dataset_start", "mean", "start", "zeros"],
                   help="where to park after a trial or on exit (see resolve_home_pose; dataset_start recommended)")
    p.add_argument("--no-return-home", action="store_true", help="do not park the arm after trials or on exit")
    p.add_argument(
        "--move-to-dataset-start-state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Move into the demonstrated episode-start distribution before connecting the policy "
            "(default: enabled)."
        ),
    )
    p.add_argument("--start-state-move-s", type=float, default=3.0)
    p.add_argument(
        "--start-state-max-delta",
        type=float,
        default=1.0,
        help="Maximum per-command joint change during the pre-roll start-state move.",
    )
    p.add_argument("--start-state-settle-s", type=float, default=1.0)
    p.add_argument(
        "--start-state-tolerance",
        type=float,
        default=1.0,
        help="Maximum verified start/park pose error in normalized joint units (default: 1).",
    )
    p.add_argument("--start-state-max-z", type=float, default=3.0)
    p.add_argument(
        "--start-state-wrist-flex",
        type=float,
        default=None,
        help=(
            "Override only the demonstrated wrist-flex start target. The value "
            "must remain within the selected dataset's episode-start range. "
            "Defaults to the dataset's own start pose, which is always in "
            "range; a fixed default silently goes stale when the start-state "
            "dataset changes."
        ),
    )
    p.add_argument(
        "--start-state-repo-id",
        default=None,
        help=(
            "Dataset used to compute the demonstrated start pose. Defaults to the "
            "policy training dataset. DAgger aggregates with mid-task correction "
            "windows should override this with the seed dataset."
        ),
    )
    p.add_argument(
        "--start-state-revision",
        default=None,
        help="Revision for --start-state-repo-id (default: main).",
    )
    p.add_argument("--log-dir", default=str(REPO_ROOT / "outputs/eval_async_logs"))
    p.add_argument("--no-file-log", action="store_true", help="do not write client/server log files")
    p.add_argument(
        "--trace-path",
        default="auto",
        help="Output JSONL trace path (step/qpos/target_action/sent_action), same schema as "
        "models/act/eval_async_custom.py's trace, for apples-to-apples comparison. Use 'auto' for "
        "outputs/eval_traces/act_eval_async_<timestamp>.jsonl or 'none' to disable.",
    )
    p.add_argument(
        "--record-rollout",
        action="store_true",
        help=(
            "For each trial, save synchronized camera MP4s, observations.jsonl, "
            "actions.jsonl, and metadata.json, then prompt for success/failure."
        ),
    )
    p.add_argument(
        "--rollout-output-root",
        type=Path,
        default=REPO_ROOT / "outputs/eval_rollouts",
        help="Parent directory used by --record-rollout.",
    )

    p.add_argument("--server-address", default="127.0.0.1:8080")
    p.add_argument("--no-launch-server", action="store_true", help="connect to an already-running/remote server")
    p.add_argument("--server-start-timeout", type=float, default=60.0)
    p.add_argument(
        "--first-action-timeout",
        type=float,
        default=10.0,
        help="Abort if no first action chunk arrives within this many seconds. <=0 disables.",
    )
    p.add_argument(
        "--action-starvation-timeout",
        type=float,
        default=2.0,
        help="Abort if the action queue stays empty this long after playback starts. <=0 disables.",
    )
    return p.parse_args()


def visualize_action_queue_size_if_present(
    action_queue_size: list[int],
    *,
    visualize=visualize_action_queue_size,
) -> None:
    """Plot the action-queue-size trace, skipping degenerate/empty queues.

    ``visualize_action_queue_size`` calls ``max(action_queue_size)``, which
    raises ``ValueError`` on an empty list -- easily reached when a trial is
    aborted before any step is recorded. This call site sits in
    ``run_robot_client``'s cleanup ``finally`` block, so an unguarded raise
    here would replace (mask) whatever exception is already propagating
    through cleanup, such as a watchdog safety-abort reason.
    """
    if not action_queue_size:
        print(
            "[eval_async] skipping action queue visualization: no queue samples recorded.",
            flush=True,
        )
        return
    visualize(action_queue_size)


def run_guarded_cleanup(label: str, action, default=None):
    """Run a non-critical cleanup step, logging (never raising) on failure.

    Used for UI/file teardown (trial-capture writer close, camera preview
    close) that sits ahead of the robot shutdown/return-home/disconnect
    sequence in ``run_robot_client``'s ``finally`` block. Without this guard,
    an exception from one of those cleanup steps (e.g. a video writer flush
    error or an OpenCV window-close failure) would propagate and skip parking,
    disconnect, and torque-disable entirely.
    """
    try:
        return action()
    except Exception as exc:  # noqa: BLE001 -- cleanup failures must not block shutdown
        print(f"[eval_async] {label} failed: {exc}", flush=True)
        return default


def run_robot_client(
    args: argparse.Namespace,
    policy_path: Path,
    camera_names: list[str],
    fourcc: str | None,
) -> None:
    client_cfg = RobotClientConfig(
        policy_type=args.policy_type,
        pretrained_name_or_path=str(policy_path),
        robot=OmxFollowerConfig(
            port=args.robot_port,
            id=args.robot_id,
            cameras={name: camera_config(name, fourcc, args.camera_warmup_s) for name in camera_names},
        ),
        actions_per_chunk=args.actions_per_chunk,
        task=args.task,
        server_address=args.server_address,
        policy_device=args.policy_device,
        client_device=args.client_device,
        chunk_size_threshold=args.chunk_size_threshold,
        fps=args.fps,
        aggregate_fn_name=args.aggregate_fn,
        debug_visualize_queue_size=args.visualize_queue,
    )
    print(
        f"[eval_async] policy={policy_path.name} device={args.policy_device} fps={args.fps} "
        f"actions_per_chunk={args.actions_per_chunk} threshold={args.chunk_size_threshold} "
        f"aggregate={args.aggregate_fn} cameras={camera_names} fourcc={fourcc}",
        flush=True,
    )

    client = None
    trace_file = None
    trace_path = None
    home_pose = None
    home_label = args.home_target
    fallback_pose = None
    parked = False
    watchdog = None
    session_interrupted = False
    session_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    rollout_session_dir = None
    active_capture = None
    top_camera_preview = None

    def start_trial_capture(trial: int):
        nonlocal active_capture, rollout_session_dir
        if not args.record_rollout:
            return None
        if active_capture is not None:
            raise RuntimeError("A rollout capture is already active")
        if rollout_session_dir is None:
            args.rollout_output_root.mkdir(parents=True, exist_ok=True)
            rollout_session_dir = (
                args.rollout_output_root
                / f"act_eval_async_{session_timestamp}"
            )
            rollout_session_dir.mkdir(parents=False, exist_ok=False)
            print(
                f"[eval_async] rollout artifacts -> {rollout_session_dir}",
                flush=True,
            )

        writer = RolloutArtifactWriter(
            rollout_session_dir / f"trial-{trial:03d}",
            camera_names,
            args.fps,
        )
        capture = {
            "trial": trial,
            "writer": writer,
            "started_at": datetime.now(timezone.utc),
            "started_perf": time.perf_counter(),
        }
        active_capture = capture

        def record_observation(observation) -> None:
            frames = {
                camera: observation[camera]
                for camera in camera_names
            }
            qpos = {
                key: float(value)
                for key, value in observation.items()
                if key.endswith(".pos")
            }
            with client.latest_action_lock:
                policy_timestep = max(client.latest_action, 0)
            writer.record_observation(
                frames,
                {
                    "captured_at": datetime.now(timezone.utc).isoformat(),
                    "source": "policy_request",
                    "policy_timestep": policy_timestep,
                    "qpos": qpos,
                },
            )

        def record_action(entry) -> None:
            record_rollout_action_transition(
                client=client,
                writer=writer,
                entry=entry,
                camera_names=camera_names,
            )

        client.observation_callback = record_observation
        client.action_callback = record_action
        return capture

    def detach_trial_capture():
        nonlocal active_capture
        if client is not None:
            client.observation_callback = None
            client.action_callback = None
        capture = active_capture
        active_capture = None
        if capture is not None:
            capture["rollout_finished_at"] = datetime.now(timezone.utc)
            capture["duration_s"] = (
                time.perf_counter() - capture["started_perf"]
            )
            capture["writer"].close()
        return capture

    def finish_trial_capture(capture, reason: str, outcome_label: str | None) -> None:
        if capture is None:
            return
        writer = capture["writer"]
        finished_at = datetime.now(timezone.utc)
        metadata = {
            "schema_version": 1,
            "policy_type": args.policy_type,
            "advantage": args.advantage,
            "policy_path": str(policy_path),
            "task": args.task,
            "trial": capture["trial"],
            "outcome_label": outcome_label,
            "success": (
                outcome_label == "success"
                if outcome_label is not None
                else None
            ),
            "exit_reason": reason,
            "trial_started_at": capture["started_at"].isoformat(),
            "rollout_finished_at": capture["rollout_finished_at"].isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_s": capture["duration_s"],
            "control_hz": args.fps,
            "video_encoded_fps": args.fps,
            "average_observation_hz": (
                writer.observation_count / capture["duration_s"]
                if capture["duration_s"] > 0
                else 0.0
            ),
            "camera_names": list(camera_names),
            "joint_names": list(client.robot.action_features),
            "observation_sampling": (
                "policy requests plus a cached-camera/state observation after "
                "every executed action"
            ),
            "video_timing": (
                "constant-rate encoding; use observations.jsonl captured_at "
                "timestamps for real observation timing"
            ),
            "observation_count": writer.observation_count,
            "action_count": writer.action_count,
            "camera_frame_count": writer.video_recorder.frame_count,
            "videos": {
                camera: str(path.relative_to(writer.output_dir))
                for camera, path in writer.video_paths.items()
                if path.exists()
            },
            "observations_path": writer.observations_path.name,
            "actions_path": writer.actions_path.name,
            "session_trace_path": (
                str(trace_path.resolve())
                if trace_path is not None
                else None
            ),
            "actions_per_chunk": args.actions_per_chunk,
            "chunk_size_threshold": args.chunk_size_threshold,
            "aggregate_fn": args.aggregate_fn,
            "action_smoothing": args.action_smoothing,
            "max_joint_delta": args.max_joint_delta,
        }
        metadata_path = writer.output_dir / "metadata.json"
        write_metadata_atomic(metadata_path, metadata)
        print(
            f"[eval_async] Trial {capture['trial']} metadata -> {metadata_path}",
            flush=True,
        )

    try:
        client = LockedRobotClient(client_cfg)  # connects the robot
        if args.show_top_camera:
            top_camera_preview = TopCameraPreview(client.robot.cameras["top"])
            client.loop_callback = top_camera_preview.update
            print(
                "[eval_async] showing live top-camera preview "
                "(q/Esc closes the preview; Ctrl-C or Enter stops the trial).",
                flush=True,
            )
        fallback_pose = joint_positions(client.robot)
        camera_view_checker = None
        scene_coverage = None
        if "top" in client.robot.cameras:
            try:
                camera_view_checker, scene_coverage = load_session_diagnostics(
                    policy_path,
                    repo_id_override=args.diagnostic_dataset_repo_id,
                    revision_override=args.diagnostic_dataset_revision,
                    camera_view_check=args.camera_view_check,
                    camera_reference_count=args.camera_reference_count,
                )
            except Exception as exc:  # noqa: BLE001 -- diagnostic must not block eval
                print(
                    f"[eval_async] camera view reference load skipped: {exc!r}",
                    flush=True,
                )

        dataset_start_pose = None
        statistics = None
        needs_dataset_start = args.move_to_dataset_start_state or (
            not args.no_return_home
            and args.return_home_s > 0
            and args.home_target == "dataset_start"
        )
        if needs_dataset_start:
            dataset_start_pose, statistics, repo_id, revision = resolve_dataset_start_pose(
                client.robot,
                policy_path,
                repo_id_override=args.start_state_repo_id,
                revision_override=args.start_state_revision,
                statistics_loader=dataset_start_statistics,
            )
            dataset_start_pose = apply_wrist_flex_start_override(
                dataset_start_pose,
                statistics,
                args.start_state_wrist_flex,
            )
            if (
                not args.no_return_home
                and args.return_home_s > 0
                and args.home_target == "dataset_start"
            ):
                home_pose = dataset_start_pose
        if args.move_to_dataset_start_state:
            assert dataset_start_pose is not None and statistics is not None
            print(
                f"[eval_async] moving to demonstrated start distribution from {repo_id}@{revision}",
                flush=True,
            )
            return_home(
                client.robot,
                dataset_start_pose,
                args.start_state_move_s,
                args.fps,
                "dataset_start",
                max_step_delta=args.start_state_max_delta,
                tolerance=args.start_state_tolerance,
                settle_timeout_s=args.return_home_settle_timeout_s,
            )
            current = joint_positions(client.robot)
            std = {
                key: float(statistics["std"][index])
                for index, key in enumerate(client.robot.action_features)
            }
            violations = start_state_violations(
                current,
                dataset_start_pose,
                std,
                tolerance_floor=args.start_state_tolerance,
                max_z=args.start_state_max_z,
            )
            if violations:
                raise RuntimeError(
                    f"dataset start-state move remained outside demonstrated support: {violations}"
                )
            print(f"[eval_async] demonstrated start-state support reached: {current}", flush=True)
            if args.start_state_settle_s > 0:
                time.sleep(args.start_state_settle_s)

        if args.action_smoothing < 1.0:
            install_action_smoother(client.robot, args.action_smoothing)
            print(
                f"[eval_async] action EMA smoothing a={args.action_smoothing} (gripper exempt)",
                flush=True,
            )
        install_action_rate_limiter(client.robot, args.max_joint_delta)
        print(
            f"[eval_async] hard action safety clamp={args.max_joint_delta:g} "
            "joint units/command "
            "(all joints, including gripper)",
            flush=True,
        )
        if args.trace_path.lower() != "none":
            trace_path = (
                Path("outputs/eval_traces")
                / f"act_eval_async_{datetime.now().strftime('%Y%m%d-%H%M%S')}.jsonl"
                if args.trace_path == "auto"
                else Path(args.trace_path)
            )
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_file = trace_path.open("w")
            print(f"[eval_async] saving eval trace to {trace_path}", flush=True)
        if trace_file is not None or args.record_rollout:
            def dispatch_trial_action(entry) -> None:
                callback = client.action_callback
                if callback is not None:
                    callback(entry)

            install_trace_logger(
                client.robot,
                trace_file,
                dispatch_trial_action,
            )

        if not args.no_return_home and args.return_home_s > 0:
            if args.home_target == "dataset_start" and dataset_start_pose is not None:
                home_pose = dataset_start_pose
            else:
                home_pose = resolve_home_pose(args.home_target, client.robot, policy_path)
            print(f"[eval_async] Ctrl-C home '{args.home_target}': {home_pose}", flush=True)

        if not client.start():
            raise RuntimeError("Failed to connect RobotClient to the policy server.")
        print(
            f"[eval_async] connected; persistent session ready for {args.trials} trial(s).",
            flush=True,
        )
        threading.Thread(target=client.receive_actions, daemon=True).start()
        threading.Thread(target=heartbeat, args=(client,), daemon=True).start()
        watchdog = ActionStreamWatchdog(
            client,
            first_action_timeout=args.first_action_timeout,
            starvation_timeout=args.action_starvation_timeout,
        )
        threading.Thread(target=watchdog.run, daemon=True).start()

        trial_duration = (
            args.trial_duration
            if args.trial_duration is not None
            else (60.0 if args.trials > 1 else 0.0)
        )
        wait_for_trial_start = (
            args.wait_for_trial_start
            if args.wait_for_trial_start is not None
            else args.trials > 1
        )

        def before_trial(trial: int) -> None:
            nonlocal parked
            if wait_for_trial_start:
                prompt_for_trial_start(
                    trial=trial,
                    total_trials=args.trials,
                )
            if "top" in client.robot.cameras:
                try:
                    from models.act.scene_check import print_scene_check

                    frame = client.robot.cameras["top"].read_latest()
                    if scene_coverage is None:
                        print(
                            f"[eval_async trial={trial}] SCENE CHECK SKIPPED: "
                            "training coverage unavailable.",
                            flush=True,
                        )
                    else:
                        print_scene_check(
                            frame,
                            label=f"eval_async trial={trial}",
                            coverage=scene_coverage,
                        )
                    if camera_view_checker is not None:
                        from models.act.camera_alignment import print_camera_view_check

                        print_camera_view_check(
                            camera_view_checker,
                            frame,
                            label=f"eval_async trial={trial}",
                        )
                except Exception as exc:  # noqa: BLE001 -- informational only
                    print(f"[eval_async] scene check skipped: {exc!r}", flush=True)
            start_trial_capture(trial)
            parked = False
            print(
                f"[eval_async] Trial {trial} running; "
                f"{trial_stop_instructions(duration_s=trial_duration, enter_enabled=args.enter_to_end_trial, stdin_interactive=sys.stdin.isatty())}.",
                flush=True,
            )

        def after_trial(trial: int, reason: str) -> None:
            nonlocal parked
            capture = detach_trial_capture()
            print(f"[eval_async] Trial {trial} ended: {reason}.", flush=True)
            try:
                if (
                    should_park_after_trial(reason)
                    and home_pose is not None
                    and not args.no_return_home
                    and args.return_home_s > 0
                ):
                    return_home(
                        client.robot,
                        home_pose,
                        args.return_home_s,
                        args.fps,
                        home_label,
                        max_step_delta=args.start_state_max_delta,
                        tolerance=args.start_state_tolerance,
                        settle_timeout_s=args.return_home_settle_timeout_s,
                    )
                    parked = True
                    if args.start_state_settle_s > 0:
                        time.sleep(args.start_state_settle_s)
            except BaseException:
                finish_trial_capture(
                    capture,
                    f"{reason}_parking_failed",
                    None,
                )
                raise
            outcome_label = None
            prompt_interrupted = False
            if (
                capture is not None
                and trial_has_recorded_actions(capture["writer"].action_count)
                and reason in {"operator", "duration"}
            ):
                try:
                    outcome_label = prompt_for_outcome()
                except KeyboardInterrupt:
                    prompt_interrupted = True
                    print(
                        "\n[eval_async] outcome prompt canceled; leaving trial unlabeled.",
                        flush=True,
                    )
            finish_trial_capture(capture, reason, outcome_label)
            if prompt_interrupted:
                raise KeyboardInterrupt

        run_trial_sequence(
            client,
            trials=args.trials,
            task=args.task,
            duration_s=trial_duration,
            stop_requested_factory=lambda _trial: (
                stdin_enter_pressed if args.enter_to_end_trial else (lambda: False)
            ),
            before_trial=before_trial,
            after_trial=after_trial,
        )
        if watchdog.reason is not None:
            raise RuntimeError(watchdog.reason)
    except KeyboardInterrupt:
        session_interrupted = True
        print("\n[eval_async] interrupted; stopping policy and parking arm.", flush=True)
    finally:
        # Capture/preview teardown must never block the robot shutdown below:
        # an exception here (e.g. a video writer flush error or an OpenCV
        # window-close failure) would otherwise skip parking, disconnect, and
        # torque-disable cleanup entirely.
        pending_capture = run_guarded_cleanup("trial capture cleanup", detach_trial_capture)
        if client is not None:
            client.loop_callback = None
        if top_camera_preview is not None:
            run_guarded_cleanup("top camera preview close", top_camera_preview.close)
        if client is not None:
            client.shutdown_event.set()
            cleanup_pose = home_pose
            if (
                cleanup_pose is None
                and not args.no_return_home
                and args.return_home_s > 0
                and fallback_pose is not None
            ):
                cleanup_pose = fallback_pose
                home_label = "startup"
            if (
                cleanup_pose is not None
                and not parked
                and (watchdog is None or watchdog.reason is None)
            ):
                try:
                    return_home(
                        client.robot,
                        cleanup_pose,
                        args.return_home_s,
                        args.fps,
                        home_label,
                        max_step_delta=args.max_joint_delta,
                        tolerance=args.start_state_tolerance,
                        settle_timeout_s=args.return_home_settle_timeout_s,
                    )
                except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001
                    print(f"[eval_async] could not return home: {exc}", flush=True)
            try:
                client.stop()
            except Exception as exc:  # noqa: BLE001 -- continue closing logs/server
                print(f"[eval_async] client stop failed: {exc}", flush=True)
        if pending_capture is not None:
            outcome_label = None
            if (
                session_interrupted
                and trial_has_recorded_actions(pending_capture["writer"].action_count)
            ):
                try:
                    outcome_label = prompt_for_outcome()
                except KeyboardInterrupt:
                    print(
                        "\n[eval_async] outcome prompt canceled; leaving trial unlabeled.",
                        flush=True,
                    )
            finish_trial_capture(
                pending_capture,
                "keyboard_interrupt" if session_interrupted else "aborted",
                outcome_label,
            )
        if trace_file is not None:
            trace_file.close()
        if args.visualize_queue and client is not None:
            visualize_action_queue_size_if_present(client.action_queue_size)


def main() -> None:
    args = parse_args()
    if args.trials <= 0:
        raise ValueError("--trials must be positive.")
    if args.trial_duration is not None and args.trial_duration <= 0:
        raise ValueError("--trial-duration must be positive when provided.")
    if args.return_home_settle_timeout_s < 0:
        raise ValueError("--return-home-settle-timeout-s must be non-negative.")
    if args.camera_reference_count <= 0:
        raise ValueError("--camera-reference-count must be positive.")
    policy_path = Path(args.policy_path).resolve()
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy path not found: {policy_path}")
    camera_names = [c.strip() for c in args.cameras.split(",") if c.strip()]
    if args.show_top_camera and "top" not in camera_names:
        raise ValueError("--show-top-camera requires 'top' in --cameras.")
    fourcc = args.camera_fourcc or None
    host, port = args.server_address.split(":")
    if not np.isfinite(args.max_joint_delta) or args.max_joint_delta <= 0:
        raise ValueError("--max-joint-delta must be a positive finite value.")

    force_no_calibration()

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    server_log_path = None
    if not args.no_file_log:
        client_log, server_log_path = setup_file_logging(Path(args.log_dir), ts)
        print(f"[eval_async] client log -> {client_log}", flush=True)

    if not args.no_camera_preflight:
        print(f"[eval_async] preflighting cameras {camera_names} (fourcc={fourcc})...", flush=True)
        preflight_cameras(camera_names, fourcc, args.camera_warmup_s, args.camera_preflight_retries)

    server_proc = None
    if not args.no_launch_server:
        server_proc = launch_server(
            host,
            int(port),
            args.fps,
            server_log_path,
            args.server_start_timeout,
            args.policy_type,
            args.advantage,
        )

    try:
        run_robot_client(args, policy_path, camera_names, fourcc)
    finally:
        if server_proc is not None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_proc.kill()


if __name__ == "__main__":
    main()
