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
import select
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime
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

from models.act.eval_common import (
    clear_stuck_port,
    dataset_start_statistics,
    joint_positions,
    move_to_joint_state,
    start_state_violations,
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
    print(f"[eval_async] launching policy server: {' '.join(cmd)}", flush=True)
    stdout = open(log_path, "w") if log_path else None  # noqa: SIM115 (open for process lifetime)
    if log_path:
        print(f"[eval_async] policy server log -> {log_path}", flush=True)
    proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=stdout,
                            stderr=subprocess.STDOUT if stdout else None)
    deadline = time.time() + timeout
    while time.time() < deadline:
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


def install_action_smoother(robot, alpha: float) -> None:
    """Wrap ``robot.send_action`` with a per-joint EMA low-pass to damp residual jitter.

    sent[j] = alpha*target[j] + (1-alpha)*prev[j]. The gripper is passed through
    unfiltered so open/close stays crisp. The filter state seeds from the first
    target (no startup lurch). The return_home sweep also goes through this filter,
    but it already interpolates slowly so the added lag is negligible.
    """
    raw_send = robot.send_action
    state: dict[str, float] = {}

    def smoothed_send(action: dict[str, float]):
        out = {}
        for k, v in action.items():
            if "gripper" in k or k not in state:
                out[k] = v
            else:
                out[k] = alpha * v + (1.0 - alpha) * state[k]
            state[k] = out[k]
        return raw_send(out)

    robot.send_action = smoothed_send


def return_home(
    robot,
    target: dict[str, float],
    duration_s: float,
    fps: int,
    label: str,
    *,
    max_step_delta: float | None = None,
) -> None:
    """Smoothly interpolate from the current pose to ``target`` over ``duration_s`` at ``fps``."""
    if fps <= 0:
        raise ValueError("fps must be positive.")
    if max_step_delta is not None and max_step_delta <= 0:
        raise ValueError("max_step_delta must be positive when provided.")
    print(f"\n[eval_async] returning to {label} pose over {duration_s:.1f}s...", flush=True)
    try:
        current = joint_positions(robot)
    except ConnectionError as exc:
        if "Port is in use" not in str(exc) or not clear_stuck_port(robot):
            raise
        print("[eval_async] recovered from a stuck serial port (Ctrl-C landed mid-transaction); retrying.", flush=True)
        current = joint_positions(robot)
    for k in sorted(target):
        print(f"[eval_async]   {k:18s} {current.get(k, float('nan')):8.2f} -> {target[k]:8.2f} "
              f"(Δ{target[k] - current.get(k, target[k]):+.2f})", flush=True)
    target_vector = np.array([target[f"{name}.pos"] for name in JOINT_NAMES], dtype=np.float32)
    move_to_joint_state(
        robot,
        target=target_vector,
        timeout=max(duration_s, 1 / fps),
        ramp_seconds=max(duration_s, 0),
        hz=fps,
        max_delta=max_step_delta or 0,
        tolerance=None,
    )
    print(f"[eval_async] {label} pose reached.", flush=True)
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
        super().__init__(*args, **kwargs)
        self.action_queue_lock = threading.RLock()
        self.trial_active = threading.Event()
        self._control_loop_started = False
        self.trial_started_at = 0.0

    def _aggregate_action_queues(self, *args, **kwargs):
        trial_active = getattr(self, "trial_active", None)
        if trial_active is not None and not trial_active.is_set():
            return
        trial_started_at = getattr(self, "trial_started_at", 0.0)
        if args and trial_started_at:
            incoming_actions = [
                action
                for action in args[0]
                if action.get_timestamp() >= trial_started_at
            ]
            if not incoming_actions:
                return
            args = (incoming_actions, *args[1:])
        # Upstream aliases the deque under this lock, then iterates it after
        # releasing the lock while the control thread may pop from it.
        with self.action_queue_lock:
            return super()._aggregate_action_queues(*args, **kwargs)

    def pause_trial(self) -> None:
        self.trial_active.clear()
        with self.action_queue_lock:
            while not self.action_queue.empty():
                self.action_queue.get_nowait()

    def prepare_trial(self) -> None:
        self.pause_trial()
        self.stub.Ready(services_pb2.Empty())
        with self.latest_action_lock:
            self.latest_action = -1
        self.action_chunk_size = -1
        self.must_go.set()
        self.trial_started_at = time.time()
        self.trial_active.set()

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
                reason = "operator"
                break
            if duration_s > 0 and loop_started - started >= duration_s:
                reason = "duration"
                break

            if self.actions_available():
                self.control_loop_action(verbose)
            if self._ready_to_send_observation():
                self.control_loop_observation(task, verbose)

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
    return reasons


def stdin_enter_pressed(stream=sys.stdin) -> bool:
    if not stream.isatty():
        return False
    readable, _, _ = select.select([stream], [], [], 0)
    if not readable:
        return False
    stream.readline()
    return True


def install_trace_logger(robot, trace_file) -> None:
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
        trace_file.write(json.dumps({
            "step": state["step"],
            "elapsed_s": time.perf_counter() - state["start"],
            "qpos": qpos,
            "target_action": {k: float(v) for k, v in action.items()},
            "sent_action": {k: float(v) for k, v in sent.items()},
        }) + "\n")
        state["step"] += 1
        return sent

    robot.send_action = traced_send


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy-path", default=str(DEFAULT_CKPT), help="pretrained_model checkpoint dir")
    p.add_argument("--policy-type", default="act")
    p.add_argument("--task", default="Place the sponge in the bowl")
    p.add_argument("--robot-port", default=FOLLOWER_PORT, help="OMX follower serial port")
    p.add_argument("--robot-id", default="omx_follower_arm")
    p.add_argument("--cameras", default="top,wrist", help="comma-separated names from utils.config.CAMERAS")
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

    p.add_argument("--return-home-s", type=float, default=3.0, help="seconds to park the arm on Ctrl-C (0 disables)")
    p.add_argument("--home-target", default="dataset_start", choices=["dataset_start", "mean", "start", "zeros"],
                   help="where to park on Ctrl-C (see resolve_home_pose; dataset_start recommended)")
    p.add_argument("--no-return-home", action="store_true", help="do not park the arm on exit")
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
    p.add_argument("--start-state-tolerance", type=float, default=0.5)
    p.add_argument("--start-state-max-z", type=float, default=3.0)
    p.add_argument("--log-dir", default=str(REPO_ROOT / "outputs/eval_async_logs"))
    p.add_argument("--no-file-log", action="store_true", help="do not write client/server log files")
    p.add_argument(
        "--trace-path",
        default="auto",
        help="Output JSONL trace path (step/qpos/target_action/sent_action), same schema as "
        "models/act/eval_async_custom.py's trace, for apples-to-apples comparison. Use 'auto' for "
        "outputs/eval_traces/act_eval_async_<timestamp>.jsonl or 'none' to disable.",
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
    home_pose = None
    home_label = args.home_target
    fallback_pose = None
    parked = False
    try:
        client = LockedRobotClient(client_cfg)  # connects the robot
        fallback_pose = joint_positions(client.robot)
        camera_view_checker = None
        scene_coverage = None
        if args.camera_view_check and "top" in client.robot.cameras:
            try:
                from models.act.camera_alignment import CameraAlignmentChecker
                from models.act.scene_check import coverage_from_frames

                camera_repo_id, camera_revision = dataset_reference(policy_path)
                print(
                    "[eval_async] loading top-camera alignment references from "
                    f"{camera_repo_id}@{camera_revision}...",
                    flush=True,
                )
                camera_view_checker = CameraAlignmentChecker.from_dataset(
                    camera_repo_id,
                    revision=camera_revision,
                    camera_name="top",
                    max_references=args.camera_reference_count,
                )
                scene_coverage = coverage_from_frames(
                    camera_view_checker.coverage_frames
                )
            except Exception as exc:  # noqa: BLE001 -- diagnostic must not block eval
                print(
                    f"[eval_async] camera view reference load skipped: {exc!r}",
                    flush=True,
                )

        dataset_start_pose = None
        if args.move_to_dataset_start_state:
            repo_id, revision = dataset_reference(policy_path)
            statistics = dataset_start_statistics(repo_id, revision)
            dataset_start_pose = {
                key: float(statistics["mean"][index])
                for index, key in enumerate(client.robot.action_features)
            }
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
            install_trace_logger(client.robot, trace_file)

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
                input(
                    f"\n[eval_async] Trial {trial}/{args.trials}: position the objects, "
                    "then press Enter to start..."
                )
            if "top" in client.robot.cameras:
                try:
                    from models.act.scene_check import print_scene_check

                    frame = client.robot.cameras["top"].read_latest()
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
            parked = False
            timeout_text = (
                f" or after {trial_duration:g}s"
                if trial_duration > 0
                else ""
            )
            print(
                f"[eval_async] Trial {trial} running; press Enter to end{timeout_text}.",
                flush=True,
            )

        def after_trial(trial: int, reason: str) -> None:
            nonlocal parked
            print(f"[eval_async] Trial {trial} ended: {reason}.", flush=True)
            if home_pose is not None and not args.no_return_home and args.return_home_s > 0:
                return_home(
                    client.robot,
                    home_pose,
                    args.return_home_s,
                    args.fps,
                    home_label,
                    max_step_delta=args.start_state_max_delta,
                )
                parked = True
                if args.start_state_settle_s > 0:
                    time.sleep(args.start_state_settle_s)

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
        print("\n[eval_async] interrupted; stopping policy and parking arm.", flush=True)
    finally:
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
            if cleanup_pose is not None and not parked:
                try:
                    return_home(client.robot, cleanup_pose, args.return_home_s, args.fps, home_label)
                except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001
                    print(f"[eval_async] could not return home: {exc}", flush=True)
            try:
                client.stop()
            except Exception as exc:  # noqa: BLE001 -- continue closing logs/server
                print(f"[eval_async] client stop failed: {exc}", flush=True)
        if trace_file is not None:
            trace_file.close()
        if args.visualize_queue and client is not None:
            visualize_action_queue_size(client.action_queue_size)


def main() -> None:
    args = parse_args()
    policy_path = Path(args.policy_path).resolve()
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy path not found: {policy_path}")
    camera_names = [c.strip() for c in args.cameras.split(",") if c.strip()]
    fourcc = args.camera_fourcc or None
    host, port = args.server_address.split(":")

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
