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
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from lerobot.async_inference.configs import RobotClientConfig
from lerobot.async_inference.helpers import visualize_action_queue_size
from lerobot.async_inference.robot_client import RobotClient
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.omx_follower.config_omx_follower import OmxFollowerConfig
from lerobot.robots.omx_follower.omx_follower import OmxFollower

from utils.config import CAMERAS, CAMERA_HEIGHT, CAMERA_WIDTH, FOLLOWER_PORT, FPS

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


def launch_server(host: str, port: int, fps: int, log_path: Path | None, timeout: float) -> subprocess.Popen:
    """Launch the local policy server and block until its port is accepting connections."""
    cmd = [sys.executable, "-m", "lerobot.async_inference.policy_server",
           f"--host={host}", f"--port={port}", f"--fps={fps}"]
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


def joint_positions(robot) -> dict[str, float]:
    """Current absolute joint state as {'<joint>.pos': deg} (the send_action format).

    Reads the motor bus directly rather than robot.get_observation(), which reads motor
    positions successfully first and only THEN reads cameras -- if any camera raises (e.g. a
    transient stale-frame TimeoutError), the whole call fails and the already-successful
    position read is discarded. Homing/parking is safety-critical and must not depend on
    camera health.
    """
    raw = robot.bus.sync_read("Present_Position")
    return {f"{motor}.pos": float(val) for motor, val in raw.items()}


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

    import numpy as np

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


def clear_stuck_port(robot) -> bool:
    """Recover from a Ctrl-C landing mid-serial-transaction.

    dynamixel_sdk guards every read/write with a plain `port.is_using` bool (set True before the
    transaction, False after), with NO try/finally. If SIGINT is delivered while that flag is
    True, it never resets, and every subsequent sync_read/sync_write immediately fails with
    ConnectionError("... [TxRxResult] Port is in use!") even though the port itself is fine.
    robot.disconnect() resets this same flag (motors_bus.py disconnect(), before closing the
    port) -- this does the equivalent reset without a full disconnect/reconnect cycle.
    Returns True if a port_handler was found and reset.
    """
    port_handler = getattr(getattr(robot, "bus", None), "port_handler", None)
    if port_handler is None:
        return False
    port_handler.is_using = False
    return True


def return_home(robot, target: dict[str, float], duration_s: float, fps: int, label: str) -> None:
    """Smoothly interpolate from the current pose to ``target`` over ``duration_s`` at ``fps``."""
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
    steps = max(1, int(duration_s * fps))
    for step in range(1, steps + 1):
        t0 = time.perf_counter()
        alpha = step / steps
        robot.send_action({k: current[k] + alpha * (target[k] - current[k]) for k in target})
        time.sleep(max(0.0, 1.0 / fps - (time.perf_counter() - t0)))
    print(f"[eval_async] {label} pose reached.", flush=True)


def heartbeat(client, interval: float = 2.0) -> None:
    """Periodic liveness line, since the control loop is otherwise silent."""
    start = time.time()
    while not client.shutdown_event.wait(interval):
        with client.latest_action_lock:
            la = client.latest_action
        print(f"[eval_async] alive t={time.time() - start:5.1f}s  latest_action={la}  "
              f"queue={client.action_queue.qsize()}", flush=True)


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

    p.add_argument("--return-home-s", type=float, default=3.0, help="seconds to park the arm on Ctrl-C (0 disables)")
    p.add_argument("--home-target", default="dataset_start", choices=["dataset_start", "mean", "start", "zeros"],
                   help="where to park on Ctrl-C (see resolve_home_pose; dataset_start recommended)")
    p.add_argument("--no-return-home", action="store_true", help="do not park the arm on exit")
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
    return p.parse_args()


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
        server_proc = launch_server(host, int(port), args.fps, server_log_path, args.server_start_timeout)

    client_cfg = RobotClientConfig(
        policy_type=args.policy_type,
        pretrained_name_or_path=str(policy_path),
        robot=OmxFollowerConfig(
            port=args.robot_port, id=args.robot_id,
            cameras={n: camera_config(n, fourcc, args.camera_warmup_s) for n in camera_names},
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
    print(f"[eval_async] policy={policy_path.name} device={args.policy_device} fps={args.fps} "
          f"actions_per_chunk={args.actions_per_chunk} threshold={args.chunk_size_threshold} "
          f"aggregate={args.aggregate_fn} cameras={camera_names} fourcc={fourcc}", flush=True)

    client = RobotClient(client_cfg)  # connects the robot
    if "top" in client.robot.cameras:
        try:
            from models.act.scene_check import print_scene_check
            frame = client.robot.cameras["top"].read_latest()
            print_scene_check(frame, label="eval_async")
        except Exception as exc:  # noqa: BLE001 -- informational only, must never block the eval
            print(f"[eval_async] scene check skipped: {exc!r}", flush=True)
    if args.action_smoothing < 1.0:
        install_action_smoother(client.robot, args.action_smoothing)
        print(f"[eval_async] action EMA smoothing a={args.action_smoothing} (gripper exempt)", flush=True)
    trace_file = None
    if args.trace_path.lower() != "none":
        trace_path = (
            Path("outputs/eval_traces") / f"act_eval_async_{datetime.now().strftime('%Y%m%d-%H%M%S')}.jsonl"
            if args.trace_path == "auto"
            else Path(args.trace_path)
        )
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_file = trace_path.open("w")
        print(f"[eval_async] saving eval trace to {trace_path}", flush=True)
        install_trace_logger(client.robot, trace_file)  # installed after the smoother, see docstring
    home_pose = None
    if not args.no_return_home and args.return_home_s > 0:
        home_pose = resolve_home_pose(args.home_target, client.robot, policy_path)
        print(f"[eval_async] Ctrl-C home '{args.home_target}': {home_pose}", flush=True)

    try:
        if not client.start():
            raise RuntimeError("Failed to connect RobotClient to the policy server.")
        print("[eval_async] connected; control loop running (Ctrl-C to stop)", flush=True)
        threading.Thread(target=client.receive_actions, daemon=True).start()
        threading.Thread(target=heartbeat, args=(client,), daemon=True).start()
        try:
            client.control_loop(task=args.task)
        except KeyboardInterrupt:
            print("\n[eval_async] interrupted; stopping policy and parking arm.", flush=True)
        finally:
            client.shutdown_event.set()  # stop control loop + receiver from issuing more actions
            if home_pose is not None:
                try:
                    return_home(client.robot, home_pose, args.return_home_s, args.fps, args.home_target)
                except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001
                    print(f"[eval_async] could not return home: {exc}", flush=True)
            client.stop()
            if trace_file is not None:
                trace_file.close()
            if args.visualize_queue:
                visualize_action_queue_size(client.action_queue_size)
    except KeyboardInterrupt:
        print("\n[eval_async] interrupted before control loop; shutting down.", flush=True)
    finally:
        if server_proc is not None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_proc.kill()


if __name__ == "__main__":
    main()
