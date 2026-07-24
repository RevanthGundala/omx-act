"""Async (open-loop) live evaluation for the custom ACT checkpoint (models/act/act.py).

This is the custom-checkpoint equivalent of models/act/eval_async.py, which only works with
LeRobot-format ``pretrained_model`` directories (it launches LeRobot's PolicyServer/RobotClient
over gRPC). Our custom .pt checkpoints can't go through that stack -- PolicyServer only resolves
policy classes through LeRobot's own hardcoded factory (get_policy_class), which always points at
LeRobot's own modeling_*.py implementations, not ours.

Instead, this script reimplements the same producer/consumer architecture in-process, against our
own ACTPolicy directly:

- An INFERENCE THREAD continuously predicts fresh chunk_size-step chunks from the freshest
  camera/qpos observation, using the raw ``model.forward`` output (NOT temporal-ensembled --
  TE is a closed-loop mechanism that assumes state advances between queries; on this checkpoint
  family the very first ~10 chunk steps are a near-zero "settle" segment learned from teleop data,
  so re-querying every control tick just perpetually re-observes "still at the start" and never
  reaches the back-loaded reach later in the chunk. See docs/act_debugging_learnings.md.)
- A CONTROL THREAD (main thread) plays the resulting action queue at a fixed --hz, blending
  overlapping timesteps with the same aggregate_fn registry as LeRobot's
  async_inference/configs.py (weighted_average / conservative / latest_only / average), and
  triggers a replan once the queue drains below --chunk-size-threshold * chunk_size.

Usage:
    uv run python -m models.act.eval_async_custom \
      --checkpoint <run>/checkpoints/step_020000.pt
"""
import argparse
import json
import threading
import time
import traceback
from collections import deque
from dataclasses import fields
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets import LeRobotDatasetMetadata

from configs.act.base import ACTConfig
from models.act.act import ACTPolicy
from models.act.eval import (
    create_follower,
    parse_camera_list,
    read_obs,
    safe_disconnect,
    send_action,
    validate_checkpoint_shapes,
)
from models.act.eval_common import clear_stuck_port, dataset_start_statistics, joint_positions, move_to_joint_state
from utils.config import CAMERA_HEIGHT, CAMERA_WIDTH, FOLLOWER_PORT, FPS, JOINT_NAMES

AGGREGATE_FNS = {
    # Same registry as lerobot.async_inference.configs, applied per-overlapping-timestep.
    "weighted_average": lambda old, new: 0.3 * old + 0.7 * new,
    "conservative": lambda old, new: 0.7 * old + 0.3 * new,
    "latest_only": lambda old, new: new,
    "average": lambda old, new: 0.5 * old + 0.5 * new,
}


def limit_action_chunk(chunk: np.ndarray, actions_per_chunk: int) -> np.ndarray:
    if not 1 <= actions_per_chunk <= chunk.shape[0]:
        raise ValueError(
            f"actions_per_chunk must be in [1, {chunk.shape[0]}], got {actions_per_chunk}"
        )
    return chunk[:actions_per_chunk]


class ChunkQueue:
    """Thread-safe buffer of upcoming absolute-timestep actions.

    Every entry is tagged with an absolute step index so a chunk that finishes predicting AFTER
    the queue has already advanced (inference takes ~100ms, i.e. a few control ticks) gets
    aligned correctly instead of being blended against the wrong (already-consumed) positions.
    merge_chunk() takes `captured_at_step` -- the front_abs_step() snapshot taken right when the
    observation was captured, BEFORE the (slow) model.forward call -- and discards the prefix of
    the chunk that has since been consumed before blending the rest. Mirrors
    lerobot.async_inference.robot_client's per-timestep blend (robot_client.py:256-262), which
    similarly discards already-executed timesteps before merging (policy_server.py:312-320).
    """

    def __init__(self, aggregate_fn):
        self._lock = threading.Lock()
        self._buffer: deque[np.ndarray] = deque()
        self._aggregate_fn = aggregate_fn
        self._front_abs_step = 0  # absolute step index of self._buffer[0] (next pop_front result)

    def front_abs_step(self) -> int:
        with self._lock:
            return self._front_abs_step

    def merge_chunk(self, chunk: np.ndarray, captured_at_step: int) -> None:
        with self._lock:
            # Discard the prefix of `chunk` that's already been popped since captured_at_step.
            stale = self._front_abs_step - captured_at_step
            if stale >= chunk.shape[0]:
                # The observation this chunk was predicted from is now so old (e.g. the queue
                # fully drained while a camera was stalled/dead and captured_at_step stopped
                # advancing) that a literal discard-the-prefix alignment would throw away the
                # ENTIRE chunk. Left as-is this becomes a permanent deadlock: front_abs_step
                # freezes once the buffer hits empty, so drift never shrinks and every future
                # chunk keeps getting fully discarded forever. Re-anchor instead: treat the
                # chunk as if captured right now, so playback can resume (degraded quality is
                # preferable to a silent, permanent freeze).
                stale = 0
            stale = max(0, min(stale, chunk.shape[0]))
            chunk = chunk[stale:]
            old_len = len(self._buffer)
            overlap = min(old_len, chunk.shape[0])
            for i in range(overlap):
                self._buffer[i] = self._aggregate_fn(self._buffer[i], chunk[i])
            for i in range(overlap, chunk.shape[0]):
                self._buffer.append(chunk[i])

    def pop_front(self):
        with self._lock:
            if not self._buffer:
                return None
            self._front_abs_step += 1
            return self._buffer.popleft()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)


def make_action_smoother(alpha: float):
    """Stateful EMA smoother: smoothed = prev + alpha*(new-prev). Gripper (last joint) is exempt."""
    state = {"prev": None}

    def smooth(action: np.ndarray) -> np.ndarray:
        if state["prev"] is None:
            state["prev"] = action.copy()
            return action
        smoothed = state["prev"] + alpha * (action - state["prev"])
        smoothed[-1] = action[-1]
        state["prev"] = smoothed.copy()
        return smoothed

    return smooth


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--duration", type=float, default=60.0, help="Seconds to run before auto-stopping.")
    parser.add_argument("--hz", type=float, default=float(FPS), help="Control-loop playback rate.")
    parser.add_argument("--camera-fps", type=int, default=FPS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--follower-port", default=FOLLOWER_PORT)
    parser.add_argument("--camera-width", type=int, default=CAMERA_WIDTH)
    parser.add_argument("--camera-height", type=int, default=CAMERA_HEIGHT)
    parser.add_argument("--model-width", type=int, default=CAMERA_WIDTH)
    parser.add_argument("--model-height", type=int, default=CAMERA_HEIGHT)
    parser.add_argument("--start-delay", type=float, default=3.0)
    parser.add_argument(
        "--move-to-dataset-start-state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Before inference, move the follower to the mean/median demonstrated episode-start state "
            "(default: enabled; use --no-move-to-dataset-start-state to opt out)."
        ),
    )
    parser.add_argument(
        "--start-state-reducer",
        choices=["mean", "median"],
        default="mean",
        help="Dataset episode-start statistic used with --move-to-dataset-start-state.",
    )
    parser.add_argument(
        "--start-state-repo-id",
        default=None,
        help="Dataset used for start-state statistics. Defaults to the checkpoint training dataset.",
    )
    parser.add_argument(
        "--start-state-revision",
        default=None,
        help="Dataset revision used for start-state statistics. Defaults to the checkpoint training revision.",
    )
    parser.add_argument("--start-state-duration", type=float, default=8.0)
    parser.add_argument("--start-state-ramp-seconds", type=float, default=3.0)
    parser.add_argument("--start-state-max-delta", type=float, default=2.0)
    parser.add_argument(
        "--start-state-tolerance",
        type=float,
        default=0.5,
        help="Minimum absolute per-joint start tolerance, combined with --start-state-max-z.",
    )
    parser.add_argument(
        "--start-state-max-z",
        type=float,
        default=3.0,
        help="Accept each joint once it is within this many demonstrated start-state standard deviations.",
    )
    parser.add_argument("--start-state-settle-seconds", type=float, default=1.0)
    parser.add_argument("--start-state-log-every", type=float, default=1.0)
    parser.add_argument(
        "--action-space",
        choices=["absolute", "delta"],
        default="absolute",
        help="'delta' reconstructs the command as qpos-at-capture-time + policy_output.",
    )
    parser.add_argument("--max-joint-delta", type=float, default=0.0,
                         help="Clamp each sent action to +/- this many degrees from the previously sent target. "
                         "<=0 disables (default; matches eval_async.py, which has no such clamp, for a fair A/B "
                         "comparison against the official checkpoint's async eval). --action-smoothing already "
                         "provides a soft velocity limiter.")
    parser.add_argument("--chunk-size-threshold", type=float, default=0.4,
                         help="Trigger a replan once the queue drains below this fraction of actions_per_chunk.")
    parser.add_argument(
        "--actions-per-chunk",
        type=int,
        default=None,
        help=(
            "Number of predicted actions to enqueue per inference. Defaults to "
            "min(30, checkpoint chunk_size), matching eval_async.py."
        ),
    )
    parser.add_argument("--aggregate-fn", choices=list(AGGREGATE_FNS), default="conservative")
    parser.add_argument("--action-smoothing", type=float, default=0.7,
                         help="EMA alpha applied to sent actions (1.0 disables). Gripper is exempt.")
    parser.add_argument("--return-home-s", type=float, default=3.0,
                         help="Seconds to interpolate back to the pose observed at startup on Ctrl-C. <=0 disables.")
    parser.add_argument("--camera-failure-timeout", type=float, default=2.0,
                         help="Abort and return home if no camera frame has been successfully captured for this "
                         "many seconds (e.g. a dead/disconnected camera). Continuing to command the robot from "
                         "vision that's several seconds stale is unsafe; this also prevents the queue from "
                         "permanently starving once observation-capture drift exceeds chunk_size. <=0 disables.")
    parser.add_argument("--zero-cameras", default="")
    parser.add_argument("--freeze-cameras", default="")
    parser.add_argument("--trace-path", default="auto")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device != "auto"
        else "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ACTConfig(**{f.name: ckpt["config"][f.name] for f in fields(ACTConfig) if f.name in ckpt["config"]})
    actions_per_chunk = (
        min(30, cfg.chunk_size)
        if args.actions_per_chunk is None
        else args.actions_per_chunk
    )
    if not 1 <= actions_per_chunk <= cfg.chunk_size:
        raise ValueError(
            f"--actions-per-chunk must be in [1, {cfg.chunk_size}], got {actions_per_chunk}"
        )

    zero_cameras = parse_camera_list(args.zero_cameras)
    freeze_cameras = parse_camera_list(args.freeze_cameras)
    unknown = (zero_cameras | freeze_cameras) - set(cfg.camera_names)
    if unknown:
        raise ValueError(f"camera ablation requested unknown cameras {sorted(unknown)}; checkpoint cameras={cfg.camera_names}")
    if zero_cameras & freeze_cameras:
        raise ValueError(f"cameras cannot be both zeroed and frozen: {sorted(zero_cameras & freeze_cameras)}")

    stats = ckpt.get("norm_stats") or LeRobotDatasetMetadata(cfg.dataset_repo_id, revision=cfg.dataset_revision).stats
    q_mean = torch.as_tensor(stats["observation.state"]["mean"], device=device, dtype=torch.float32).flatten()
    q_std = torch.as_tensor(stats["observation.state"]["std"], device=device, dtype=torch.float32).flatten().clamp_min(1e-6)
    a_mean = torch.as_tensor(stats["action"]["mean"], dtype=torch.float32).flatten().numpy()
    a_std = torch.as_tensor(stats["action"]["std"], dtype=torch.float32).flatten().clamp_min(1e-6).numpy()
    validate_checkpoint_shapes(
        cfg=cfg,
        q_mean=q_mean,
        q_std=q_std,
        a_mean=torch.as_tensor(a_mean),
        a_std=torch.as_tensor(a_std),
    )

    im_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    im_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)

    model = ACTPolicy(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(
        "eval_async_custom_config "
        f"checkpoint={args.checkpoint} device={device} chunk_size={cfg.chunk_size} "
        f"actions_per_chunk={actions_per_chunk} "
        f"cameras={cfg.camera_names} hz={args.hz:g} threshold={args.chunk_size_threshold} "
        f"aggregate={args.aggregate_fn} action_smoothing={args.action_smoothing} "
        f"action_space={args.action_space}",
        flush=True,
    )
    print(f"joint_order={JOINT_NAMES}", flush=True)

    follower = create_follower(
        port=args.follower_port,
        camera_names=cfg.camera_names,
        fps=args.camera_fps,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
    )
    follower.connect(calibrate=False)
    if "top" in follower.cameras:
        try:
            from models.act.scene_check import print_scene_check
            frame = follower.cameras["top"].read_latest()
            print_scene_check(frame, label="eval_async_custom")
        except Exception as exc:  # noqa: BLE001 -- informational only, must never block the eval
            print(f"[eval_async_custom] scene check skipped: {exc!r}", flush=True)

    trace_file = None
    if args.trace_path.lower() != "none":
        trace_path = (
            Path("outputs/eval_traces") / f"act_eval_async_custom_{datetime.now().strftime('%Y%m%d-%H%M%S')}.jsonl"
            if args.trace_path == "auto"
            else Path(args.trace_path)
        )
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_file = trace_path.open("w")
        print(f"Saving eval trace to {trace_path}", flush=True)

    frozen_camera_frames: dict[str, np.ndarray] = {}

    def read_qpos_only() -> dict[str, float]:
        """Motor-telemetry-only read, bypassing follower.get_observation()'s camera loop entirely.
        get_observation() reads Present_Position successfully first and only THEN reads cameras --
        if any camera raises (e.g. a transient stale-frame TimeoutError), the whole call fails and
        the already-successful position read is discarded. Return-home is safety-critical and must
        never depend on camera health."""
        return joint_positions(follower)

    def capture_obs():
        """Hardware read. Called ONLY from the control thread -- OmxFollower/motors_bus have no
        internal locking, so concurrent reads/writes from a second thread would race on the
        shared Dynamixel serial bus."""
        return read_obs(follower, cfg.camera_names, image_width=args.model_width, image_height=args.model_height)

    def capture_obs_safe():
        """capture_obs() wrapped so a single transient camera stale-frame doesn't abort the whole
        rollout. Returns (obs_or_None, error_or_None); caller falls back to the previous cached
        observation on failure instead of crashing the run."""
        try:
            return capture_obs(), None
        except TimeoutError as exc:
            return None, exc

    def preprocess(obs):
        """Pure tensor prep, no hardware I/O -- safe to call from the inference thread."""
        qpos = torch.as_tensor(obs["qpos"], device=device, dtype=torch.float32).flatten()
        qpos_normalized = ((qpos - q_mean) / q_std).unsqueeze(0)
        imgs = []
        for camera in cfg.camera_names:
            image_array = obs["images"][camera]
            if camera in zero_cameras:
                image_array = np.zeros_like(image_array)
            elif camera in freeze_cameras:
                if camera not in frozen_camera_frames:
                    frozen_camera_frames[camera] = np.array(image_array, copy=True)
                image_array = frozen_camera_frames[camera]
            image = torch.as_tensor(np.ascontiguousarray(image_array), device=device, dtype=torch.float32)
            image = image.permute(2, 0, 1) / 255.0
            imgs.append((image - im_mean) / im_std)
        return qpos_normalized, torch.stack(imgs, dim=0).unsqueeze(0)

    class LatestObs:
        """Thread-safe holder for the most recent observation + the queue's front_abs_step at
        capture time, updated by the control thread only and read by the inference thread."""

        def __init__(self):
            self._lock = threading.Lock()
            self._obs = None
            self._captured_at_step = 0

        def update(self, obs, captured_at_step: int) -> None:
            with self._lock:
                self._obs = obs
                self._captured_at_step = captured_at_step

        def get(self):
            with self._lock:
                return self._obs, self._captured_at_step

    queue = ChunkQueue(AGGREGATE_FNS[args.aggregate_fn])
    latest_obs = LatestObs()
    smoother = make_action_smoother(args.action_smoothing) if args.action_smoothing < 1.0 else (lambda a: a)
    stop_event = threading.Event()
    replan_event = threading.Event()
    inference_stats = {"count": 0, "last_ms": None}

    def inference_worker():
        while not stop_event.is_set():
            if not replan_event.wait(timeout=0.5):
                continue
            if stop_event.is_set():
                break
            replan_event.clear()
            obs, captured_at_step = latest_obs.get()
            if obs is None:
                replan_event.set()  # control thread hasn't captured a first observation yet
                time.sleep(0.02)
                continue
            try:
                t0 = time.perf_counter()
                with torch.inference_mode():
                    qpos_normalized, images = preprocess(obs)
                    act_pred, _, _ = model.forward(images, qpos_normalized)
                action_normalized = limit_action_chunk(
                    act_pred.squeeze(0).cpu().numpy(),
                    actions_per_chunk,
                )
                action = action_normalized * a_std + a_mean
                if args.action_space == "delta":
                    action = action + obs["qpos"][None, :]
                queue.merge_chunk(action, captured_at_step)
                inference_stats["count"] += 1
                inference_stats["last_ms"] = (time.perf_counter() - t0) * 1000.0
                print(
                    f"[eval_async_custom] replanned #{inference_stats['count']} "
                    f"in {inference_stats['last_ms']:.0f}ms, queue_len={len(queue)}",
                    flush=True,
                )
            except Exception:
                print("[eval_async_custom] inference_worker crashed; stopping run.", flush=True)
                traceback.print_exc()
                stop_event.set()
                break

    worker = threading.Thread(target=inference_worker, name="act-inference", daemon=True)

    exit_reason = "startup_error"
    step = 0
    home_pose = None
    period = 1.0 / args.hz
    try:
        worker.start()

        fallback_home = read_qpos_only()
        home_pose = fallback_home
        if args.move_to_dataset_start_state:
            repo_id = args.start_state_repo_id or cfg.dataset_repo_id
            revision = args.start_state_revision or cfg.dataset_revision
            start_statistics = dataset_start_statistics(repo_id, revision)
            target_start_state = start_statistics[args.start_state_reducer]
            support_tolerance = np.maximum(
                args.start_state_tolerance,
                args.start_state_max_z * start_statistics["std"],
            )
            print(
                "Moving follower to dataset "
                f"{args.start_state_reducer} start state from {repo_id}@{revision}: "
                f"{dict(zip(JOINT_NAMES, target_start_state.tolist(), strict=True))}; "
                "accepted per-joint error="
                f"{dict(zip(JOINT_NAMES, support_tolerance.tolist(), strict=True))}",
                flush=True,
            )
            final_entry = move_to_joint_state(
                follower,
                target=target_start_state,
                timeout=args.start_state_duration,
                ramp_seconds=args.start_state_ramp_seconds,
                hz=args.hz,
                max_delta=args.start_state_max_delta,
                tolerance=support_tolerance,
                log_every=args.start_state_log_every,
            )
            if final_entry.get("reached"):
                print(
                    "Reached demonstrated start-state support "
                    f"(max support ratio={final_entry['max_support_ratio']:.3f}).",
                    flush=True,
                )
            else:
                final_state = np.asarray(final_entry["state"], dtype=np.float32)
                errors = np.abs(target_start_state - final_state)
                violations = {
                    JOINT_NAMES[i]: {
                        "error": float(errors[i]),
                        "allowed": float(support_tolerance[i]),
                    }
                    for i in range(len(JOINT_NAMES))
                    if errors[i] > support_tolerance[i]
                }
                raise RuntimeError(
                    "dataset start-state move remained outside demonstrated support: "
                    f"{violations}."
                )
            if args.start_state_settle_seconds > 0:
                print(f"Settling for {args.start_state_settle_seconds:.1f}s before rollout.", flush=True)
                time.sleep(args.start_state_settle_seconds)

        if args.start_delay > 0:
            print(f"Starting custom async ACT eval in {args.start_delay:.1f}s. Ctrl-C to stop.", flush=True)
            time.sleep(args.start_delay)

        initial_obs = None
        for attempt in range(5):
            initial_obs, err = capture_obs_safe()
            if initial_obs is not None:
                break
            print(f"[eval_async_custom] initial camera read failed ({err!r}); retrying ({attempt + 1}/5)...", flush=True)
            time.sleep(0.3)
        if initial_obs is None:
            raise RuntimeError("could not capture an initial observation (camera not delivering fresh frames).")
        home_pose = {f"{name}.pos": float(initial_obs["qpos"][i]) for i, name in enumerate(JOINT_NAMES)}
        latest_obs.update(initial_obs, queue.front_abs_step())
        replan_event.set()  # force the first prediction now that an observation is available

        wait_start = time.perf_counter()
        while len(queue) == 0 and time.perf_counter() - wait_start < 10.0:
            time.sleep(0.02)
        if len(queue) == 0:
            raise RuntimeError("no chunk predicted within 10s; aborting.")
        print(f"[eval_async_custom] first chunk ready; starting playback at {args.hz:g}Hz.", flush=True)

        start_time = time.perf_counter()
        last_heartbeat = start_time
        last_successful_capture = start_time
        last_target = np.array([home_pose[f"{n}.pos"] for n in JOINT_NAMES], dtype=np.float32)
        exit_reason = "duration_elapsed"
        while True:
            if worker.is_alive() is False and stop_event.is_set():
                exit_reason = "inference_worker_crashed"
                break
            now = time.perf_counter()
            if now - start_time >= args.duration:
                break
            if args.camera_failure_timeout > 0 and now - last_successful_capture > args.camera_failure_timeout:
                raise RuntimeError(
                    f"no successful camera capture in {now - last_successful_capture:.1f}s "
                    f"(> --camera-failure-timeout {args.camera_failure_timeout:g}s); camera likely dead/disconnected."
                )
            tick_start = now

            # Send the already-queued action FIRST, before doing anything else this tick.
            # LeRobot's proven robot_client.py control_loop does the same: action-send happens
            # before observation-capture every tick (control_loop_action then
            # control_loop_observation, robot_client.py:469-474). Capturing the observation
            # BEFORE sending (the original order here) directly coupled motor-command timing to
            # camera/motor-bus READ latency variance every single tick -- any jitter in that read
            # (OS scheduling, USB polling, driver hiccups) delayed the send_action call itself,
            # injecting timing jitter straight into the actuator command cadence. That's a likely
            # cause of visibly jittery/jiggly arm motion. Capturing AFTER send instead means that
            # read's latency is absorbed into the tick's leftover time budget (before the
            # end-of-loop sleep) rather than delaying actuation.
            policy_action = queue.pop_front()
            queue_was_empty = policy_action is None
            if policy_action is None:
                policy_action = last_target  # queue starved; hold last target rather than sending nothing
            action = policy_action
            if args.max_joint_delta > 0:
                action = np.clip(action, last_target - args.max_joint_delta, last_target + args.max_joint_delta)
            smoothed = smoother(action)
            sent_action = send_action(follower, smoothed)
            last_target = np.array([sent_action[f"{n}.pos"] for n in JOINT_NAMES], dtype=np.float32)

            if len(queue) <= args.chunk_size_threshold * actions_per_chunk:
                replan_event.set()

            obs, cam_err = capture_obs_safe()
            if obs is not None:
                latest_obs.update(obs, queue.front_abs_step())
                last_successful_capture = now
            elif step % 30 == 0:  # avoid log spam if the camera stays stale for a while
                print(f"[eval_async_custom] warning: camera read failed this tick ({cam_err!r}); "
                      f"reusing last observation for inference.", flush=True)

            if trace_file is not None:
                # Prefer qpos from this tick's successful camera-inclusive capture (no extra
                # hardware call); only fall back to a standalone motor-only read when the camera
                # capture failed this tick, since read_obs() discards an already-successful motor
                # read whenever any camera raises.
                if obs is not None:
                    current_qpos = {f"{name}.pos": float(obs["qpos"][i]) for i, name in enumerate(JOINT_NAMES)}
                else:
                    try:
                        current_qpos = read_qpos_only()
                    except Exception:  # noqa: BLE001 -- trace logging must never crash the control loop
                        current_qpos = None
                trace_file.write(json.dumps({
                    "step": step,
                    "elapsed_s": now - start_time,
                    "queue_len": len(queue),
                    "queue_was_empty": queue_was_empty,
                    "qpos": current_qpos,
                    "policy_action": {name: float(policy_action[i]) for i, name in enumerate(JOINT_NAMES)},
                    "sent_action": {name: float(sent_action[f"{name}.pos"]) for name in JOINT_NAMES},
                }) + "\n")

            if now - last_heartbeat >= 2.0:
                print(f"[eval_async_custom] alive t={now - start_time:5.1f}s step={step} queue_len={len(queue)}", flush=True)
                last_heartbeat = now

            step += 1
            elapsed = time.perf_counter() - tick_start
            time.sleep(max(0.0, period - elapsed))
    except KeyboardInterrupt:
        print("\n[eval_async_custom] interrupted; stopping policy and parking arm.", flush=True)
        exit_reason = "keyboard_interrupt"
    except Exception as exc:
        print(f"\n[eval_async_custom] error: {exc!r}; stopping and parking arm.", flush=True)
        exit_reason = f"error: {exc!r}"
    finally:
        stop_event.set()
        replan_event.set()  # wake the worker so it can observe stop_event and exit
        worker.join(timeout=2.0)
        if worker.is_alive():
            print("[eval_async_custom] warning: inference worker did not stop within 2s.", flush=True)
        if args.return_home_s > 0 and home_pose is not None:
            print(f"\n[eval_async_custom] returning to start pose over {args.return_home_s:.1f}s...", flush=True)
            try:
                try:
                    current = read_qpos_only()  # camera-independent: parking must not depend on camera health
                except ConnectionError as exc:
                    if "Port is in use" not in str(exc) or not clear_stuck_port(follower):
                        raise
                    print("[eval_async_custom] recovered from a stuck serial port "
                          "(Ctrl-C landed mid-transaction); retrying.", flush=True)
                    current = read_qpos_only()
                steps = max(1, int(args.return_home_s * args.hz))
                for i in range(1, steps + 1):
                    t0 = time.perf_counter()
                    alpha = i / steps
                    follower.send_action({
                        k: current[k] + alpha * (home_pose[k] - current[k]) for k in home_pose
                    })
                    time.sleep(max(0.0, period - (time.perf_counter() - t0)))
                print("[eval_async_custom] start pose reached.", flush=True)
            except Exception as exc:  # noqa: BLE001 -- best-effort; must not block disconnect below
                print(f"[eval_async_custom] could not return home: {exc!r}", flush=True)
        if trace_file is not None:
            trace_file.close()
        safe_disconnect(follower)
        print(f"[eval_async_custom] exit_reason={exit_reason} steps={step}", flush=True)


if __name__ == "__main__":
    main()
