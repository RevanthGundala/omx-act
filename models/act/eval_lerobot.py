"""Run a stock LeRobot policy live on the OMX follower arm."""

import argparse
from datetime import datetime
import json
import sys
import time
from pathlib import Path

import numpy as np
import packaging.version
import torch
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.configs import PreTrainedConfig
from lerobot.datasets import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.robots.omx_follower.config_omx_follower import OmxFollowerConfig
from lerobot.robots.omx_follower.omx_follower import OmxFollower

try:
    from lerobot.utils.device_utils import get_safe_torch_device
except ImportError:
    from lerobot.utils.utils import get_safe_torch_device

try:
    import rerun as rr
except ImportError:
    rr = None

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import models.act.lerobot_policy  # noqa: F401 -- register custom_act before config parsing
from models.act.eval_common import dataset_start_statistics
from utils.config import CAMERA_HEIGHT, CAMERA_WIDTH, CAMERAS, FOLLOWER_PORT, FPS, JOINT_NAMES
from utils.rerun_utils import init_rerun


def disable_auto_calibration(robot):
    original_connect = robot.connect

    def connect_without_calibration(*args, **kwargs):
        kwargs["calibrate"] = False
        return original_connect(*args, **kwargs)

    robot.connect = connect_without_calibration
    return robot


def create_follower(
    *,
    port: str,
    camera_names: list[str],
    camera_fps: int,
    camera_width: int,
    camera_height: int,
) -> OmxFollower:
    cameras = {}
    for name in camera_names:
        if name not in CAMERAS:
            raise KeyError(f"camera '{name}' is not configured in utils.config.CAMERAS")
        cameras[name] = OpenCVCameraConfig(
            index_or_path=CAMERAS[name],
            fps=camera_fps,
            width=camera_width,
            height=camera_height,
            warmup_s=5,
        )
    config = OmxFollowerConfig(port=port, id="omx_follower_arm", cameras=cameras)
    return disable_auto_calibration(OmxFollower(config))


def safe_disconnect(robot) -> None:
    try:
        robot.disconnect()
    except RuntimeError:
        print("Warning: clean disconnect failed (motor hardware error). Power-cycling recommended.", flush=True)


def resolve_device(requested_device: str) -> torch.device:
    requested_device = requested_device.lower()
    if requested_device == "auto":
        if torch.cuda.is_available():
            requested_device = "cuda"
        elif torch.backends.mps.is_available():
            requested_device = "mps"
        else:
            requested_device = "cpu"
    return get_safe_torch_device(requested_device, log=True)


def detect_camera_features(policy: PreTrainedPolicy) -> list[str]:
    return [
        key.split("observation.images.")[-1]
        for key in policy.config.input_features
        if key.startswith("observation.images.")
    ]


def uses_env_state(policy: PreTrainedPolicy) -> bool:
    feature = policy.config.input_features.get("observation.state")
    return bool(feature is not None and getattr(feature.type, "value", feature.type) == "ENV")


def action_queue_len(policy: PreTrainedPolicy) -> int | None:
    # ACTPolicy keeps a plain deque at `_action_queue` (populated only when
    # temporal_ensemble_coeff is None). Other policy families (e.g. diffusion) use a
    # `_queues` dict instead; check both so this stays useful across policy types.
    action_queue = getattr(policy, "_action_queue", None)
    if action_queue is not None:
        return len(action_queue)
    queues = getattr(policy, "_queues", None)
    if isinstance(queues, dict) and "action" in queues:
        return len(queues["action"])
    return None


def vector_by_joint(values) -> dict[str, float]:
    return {name: float(values[i]) for i, name in enumerate(JOINT_NAMES)}


def save_camera_frames(output_dir: Path, camera_frames: dict[str, np.ndarray], *, prefix: str, step: int) -> None:
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)
    for camera, frame in camera_frames.items():
        path = output_dir / f"{prefix}_step_{step:05d}_{camera}.jpg"
        cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print(f"saved_camera_frame={path}", flush=True)


def parse_camera_list(value: str, available_cameras: list[str]) -> list[str]:
    if value == "":
        return []
    if value == "all":
        return list(available_cameras)
    cameras = [camera.strip() for camera in value.split(",") if camera.strip()]
    unknown = sorted(set(cameras) - set(available_cameras))
    if unknown:
        raise ValueError(f"unknown cameras {unknown}; available cameras={available_cameras}")
    return cameras


def default_video_dir(policy_path: Path) -> Path:
    run_dir = policy_path if policy_path.name != "pretrained_model" else policy_path.parent
    return run_dir / "eval_videos"


def action_from_vector(values) -> dict[str, float]:
    return {f"{name}.pos": float(values[i]) for i, name in enumerate(JOINT_NAMES)}


def send_interpolated_action(
    follower: OmxFollower,
    *,
    start: np.ndarray,
    target: np.ndarray,
    steps: int,
    command_hz: float,
) -> dict[str, float]:
    steps = max(1, int(steps))
    if steps == 1:
        return follower.send_action(action_from_vector(target))
    if command_hz <= 0:
        raise ValueError("command_hz must be > 0")

    sent_action = None
    for substep in range(1, steps + 1):
        tick_start = time.perf_counter()
        alpha = substep / steps
        command = start + alpha * (target - start)
        sent_action = follower.send_action(action_from_vector(command))
        sleep_s = 1.0 / command_hz - (time.perf_counter() - tick_start)
        if sleep_s > 0 and substep < steps:
            time.sleep(sleep_s)
    assert sent_action is not None
    return sent_action


def default_log_path(policy_path: Path) -> Path:
    run_dir = policy_path if policy_path.name != "pretrained_model" else policy_path.parent
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return run_dir / "eval_logs" / f"rollout_{timestamp}.jsonl"


def write_rollout_log(log_path: Path, metadata: dict, entries: list[dict], exit_reason: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as file:
        file.write(json.dumps({"type": "metadata", **metadata}, sort_keys=True) + "\n")
        for entry in entries:
            file.write(json.dumps({"type": "step", **entry}, sort_keys=True) + "\n")

    summary_path = log_path.with_suffix(".summary.json")
    summary = {
        **metadata,
        "exit_reason": exit_reason,
        "num_logged_steps": len(entries),
        "log_path": str(log_path),
    }
    if entries:
        summary["last_step"] = entries[-1]["step"]
        elapsed_entries = [entry for entry in entries if "elapsed_seconds" in entry]
        if elapsed_entries:
            summary["duration_seconds"] = elapsed_entries[-1]["elapsed_seconds"]
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"rollout_log={log_path}", flush=True)
    print(f"rollout_summary={summary_path}", flush=True)


def load_training_run_config(policy_path: Path) -> dict:
    metadata_path = policy_path / "training_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        return metadata.get("run_config", {})

    for parent in [policy_path, *policy_path.parents]:
        config_path = parent / "config.json"
        if not config_path.exists():
            continue
        config = json.loads(config_path.read_text())
        if "run_config" in config:
            return config["run_config"]
    return {}


def dataset_start_state(repo_id: str, revision: str, reducer: str) -> np.ndarray:
    statistics = dataset_start_statistics(repo_id, revision)
    if reducer in {"mean", "median"}:
        return statistics[reducer]
    raise ValueError(f"unknown start-state reducer: {reducer}")


def move_to_start_state(
    follower: OmxFollower,
    *,
    target: np.ndarray,
    timeout: float,
    ramp_seconds: float,
    hz: float,
    max_delta: float,
    tolerance: float,
    log_every: float,
) -> list[dict]:
    if timeout <= 0:
        raise ValueError("--start-state-duration must be > 0")
    if hz <= 0:
        raise ValueError("--hz must be > 0")
    if tolerance <= 0:
        raise ValueError("--start-state-tolerance must be > 0")
    if ramp_seconds < 0:
        raise ValueError("--start-state-ramp-seconds must be >= 0")

    initial_obs = follower.get_observation()
    initial_state = np.array([initial_obs[f"{name}.pos"] for name in JOINT_NAMES], dtype=np.float32)
    entries = []
    start = time.perf_counter()
    next_log_time = start
    step = 0
    while True:
        loop_start = time.perf_counter()
        elapsed = loop_start - start
        live_obs = follower.get_observation()
        live_state = np.array([live_obs[f"{name}.pos"] for name in JOINT_NAMES], dtype=np.float32)
        error = target - live_state
        max_abs_error = float(np.max(np.abs(error)))
        worst_joint_index = int(np.argmax(np.abs(error)))
        worst_joint = JOINT_NAMES[worst_joint_index]
        reached = max_abs_error <= tolerance
        if reached:
            entries.append(
                {
                    "phase": "move_to_dataset_start",
                    "step": step,
                    "target": vector_by_joint(target),
                    "state": vector_by_joint(live_state),
                    "sent_action": vector_by_joint(live_state),
                    "delta_to_target": vector_by_joint(error),
                    "max_abs_error": max_abs_error,
                    "worst_joint": worst_joint,
                    "reached": True,
                    "loop_seconds": time.perf_counter() - loop_start,
                }
            )
            print(
                "start_state reached "
                f"step={step} max_error={max_abs_error:.3f} "
                f"worst={worst_joint} state={live_state[worst_joint_index]:.3f} "
                f"target={target[worst_joint_index]:.3f}",
                flush=True,
            )
            break

        if ramp_seconds > 0 and elapsed < ramp_seconds:
            alpha = min(1.0, elapsed / ramp_seconds)
            command_target = initial_state + alpha * (target - initial_state)
        else:
            command_target = target

        command = command_target.copy()
        if max_delta > 0:
            command = live_state + np.clip(command_target - live_state, -max_delta, max_delta)
        sent_action = follower.send_action(action_from_vector(command))
        loop_dt = time.perf_counter() - loop_start
        entries.append(
            {
                "phase": "move_to_dataset_start",
                "step": step,
                "target": vector_by_joint(target),
                "command_target": vector_by_joint(command_target),
                "state": vector_by_joint(live_state),
                "sent_action": {name: float(sent_action[f"{name}.pos"]) for name in JOINT_NAMES},
                "delta_to_target": vector_by_joint(error),
                "max_abs_error": max_abs_error,
                "worst_joint": worst_joint,
                "reached": False,
                "loop_seconds": loop_dt,
            }
        )
        now = time.perf_counter()
        if log_every > 0 and now >= next_log_time:
            print(
                "start_state progress "
                f"step={step} elapsed={now - start:.1f}s max_error={max_abs_error:.3f} "
                f"worst={worst_joint} state={live_state[worst_joint_index]:.3f} "
                f"target={target[worst_joint_index]:.3f} "
                f"ramp_target={command_target[worst_joint_index]:.3f} "
                f"sent={sent_action[f'{worst_joint}.pos']:.3f}",
                flush=True,
            )
            next_log_time = now + log_every
        step += 1
        if now - start >= timeout:
            break
        time.sleep(max(0.0, 1.0 / hz - loop_dt))
    return entries


def build_observation(
    follower: OmxFollower,
    *,
    policy_camera_names: list[str],
    capture_camera_names: list[str],
    model_width: int,
    model_height: int,
    include_environment_state: bool,
) -> tuple[dict[str, torch.Tensor], np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    obs = follower.get_observation()
    state = np.array([obs[f"{name}.pos"] for name in JOINT_NAMES], dtype=np.float32)
    state_tensor = torch.from_numpy(state)
    obs_dict: dict[str, torch.Tensor] = {"observation.state": state_tensor}
    if include_environment_state:
        obs_dict["observation.environment_state"] = state_tensor

    raw_camera_frames = {}
    model_camera_frames = {}
    for camera in capture_camera_names:
        if camera not in obs:
            raise RuntimeError(f"follower did not return camera '{camera}'; available keys: {list(obs.keys())}")
        raw_frame = obs[camera]
        raw_camera_frames[camera] = raw_frame
        frame = raw_frame
        if camera in policy_camera_names:
            height, width = frame.shape[:2]
            if (width, height) != (model_width, model_height):
                import cv2

                frame = cv2.resize(frame, (model_width, model_height))
            model_camera_frames[camera] = frame
            image = torch.from_numpy(np.ascontiguousarray(frame)).float() / 255.0
            obs_dict[f"observation.images.{camera}"] = image.permute(2, 0, 1)
        else:
            model_camera_frames[camera] = frame
    return obs_dict, state, raw_camera_frames, model_camera_frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a stock LeRobot policy on the OMX follower arm.")
    parser.add_argument("--policy-path", required=True, help="Path to a LeRobot pretrained_model directory.")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--max-duration", type=float, default=60.0)
    parser.add_argument("--hz", type=float, default=float(FPS))
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
        action="store_true",
        help="Before rollout, move the follower to the mean/median first-frame state from the training dataset.",
    )
    parser.add_argument(
        "--start-state-reducer",
        choices=["mean", "median"],
        default="mean",
        help="Dataset start-state statistic to use when --move-to-dataset-start-state is set.",
    )
    parser.add_argument(
        "--start-state-repo-id",
        default=None,
        help="Dataset repo for start-state stats. Defaults to the training dataset recorded with the policy.",
    )
    parser.add_argument(
        "--task",
        default=None,
        help=(
            "Language instruction for VLA policies (e.g. pi0.5) that condition on a task string. "
            "Injected into each observation under the 'task' key. Leave unset for ACT/diffusion policies."
        ),
    )
    parser.add_argument(
        "--start-state-revision",
        default=None,
        help="Dataset revision for start-state stats. Defaults to the training revision recorded with the policy.",
    )
    parser.add_argument(
        "--start-state-duration",
        type=float,
        default=5.0,
        help="Maximum seconds to servo from current pose to dataset start state.",
    )
    parser.add_argument(
        "--start-state-ramp-seconds",
        type=float,
        default=3.0,
        help="Seconds to interpolate from current pose to dataset start target before holding the target.",
    )
    parser.add_argument(
        "--start-state-max-delta",
        type=float,
        default=0.0,
        help="Maximum per-joint command change during start-state motion. Defaults to 0, which sends the exact target.",
    )
    parser.add_argument(
        "--start-state-tolerance",
        type=float,
        default=0.5,
        help="Stop start-state servoing once every joint is within this absolute error.",
    )
    parser.add_argument(
        "--start-state-warning-tolerance",
        type=float,
        default=2.0,
        help="Only warn on start-state timeout if the final max joint error is above this value.",
    )
    parser.add_argument(
        "--start-state-settle-seconds",
        type=float,
        default=1.0,
        help="Seconds to wait after reaching dataset start state before policy rollout.",
    )
    parser.add_argument(
        "--start-state-log-every",
        type=float,
        default=1.0,
        help="Print start-state homing progress every N seconds. Use <=0 to disable.",
    )
    parser.add_argument("--no-rerun", action="store_true")
    parser.add_argument("--no-save-rrd", action="store_true")
    parser.add_argument(
        "--log-path",
        default=None,
        help="Write structured rollout logs to this JSONL path. Defaults next to the policy checkpoint.",
    )
    parser.add_argument(
        "--save-camera-frames-dir",
        type=Path,
        default=None,
        help="Save top/wrist camera stills from the first rollout observation to this directory.",
    )
    parser.add_argument(
        "--save-camera",
        default="",
        help="Camera name to save as raw MP4, comma-separated cameras, or 'all'. Example: top,wrist",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=None,
        help="Directory for saved MP4 videos. Defaults next to the policy checkpoint.",
    )
    parser.add_argument("--video-fps", type=float, default=None, help="FPS metadata for saved MP4. Defaults to --hz.")
    parser.add_argument(
        "--action-smoothing-alpha",
        type=float,
        default=1.0,
        help="Low-pass filter factor for policy actions. 1 disables smoothing; lower values smooth more.",
    )
    parser.add_argument(
        "--warm-start-steps",
        type=int,
        default=0,
        help="For the first N rollout steps, ramp from current state to the filtered policy command.",
    )
    parser.add_argument(
        "--action-interpolation-steps",
        type=int,
        default=1,
        help="Split each policy command into N motor substeps. 1 disables interpolation.",
    )
    parser.add_argument(
        "--action-interpolation-hz",
        type=float,
        default=30.0,
        help="Motor command rate used for --action-interpolation-steps.",
    )
    parser.add_argument(
        "--policy-n-action-steps",
        type=int,
        default=None,
        help="Override policy.config.n_action_steps before loading the policy, useful for safer ACT eval.",
    )
    parser.add_argument(
        "--policy-temporal-ensemble-coeff",
        type=float,
        default=None,
        help="Override policy.config.temporal_ensemble_coeff before loading the policy, for ACT temporal ensembling.",
    )
    parser.add_argument(
        "--policy-disable-temporal-ensemble",
        action="store_true",
        help="Force policy.config.temporal_ensemble_coeff to None, even if the checkpoint was trained/saved with "
        "temporal ensembling enabled. Required to actually get --policy-n-action-steps queue-based open-loop "
        "chunk execution: ACTPolicy.select_action always re-queries every step when temporal_ensemble_coeff is "
        "not None (n_action_steps is silently ignored in that branch), regardless of what n_action_steps is set to.",
    )
    parser.add_argument("--log-every", type=int, default=1, help="Print every N steps. Use 0 to silence per-step logs.")
    parser.add_argument(
        "--max-joint-delta",
        type=float,
        default=2.0,
        help="Maximum absolute per-joint command change from current observed state per step. Use <=0 to disable.",
    )
    parser.add_argument(
        "--action-space",
        choices=["absolute", "delta"],
        default="absolute",
        help="Interpretation of the policy output. 'delta' reconstructs the command as "
        "current_state + policy_output (use for policies trained on delta-action datasets).",
    )
    args = parser.parse_args()
    if not (0.0 < args.action_smoothing_alpha <= 1.0):
        raise ValueError("--action-smoothing-alpha must be in (0, 1]")

    policy_path = Path(args.policy_path)
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy path not found: {policy_path}")
    run_config = load_training_run_config(policy_path)

    device = resolve_device(args.device)
    policy_config = PreTrainedConfig.from_pretrained(policy_path)
    policy_config.device = device.type
    if args.policy_n_action_steps is not None:
        if not hasattr(policy_config, "n_action_steps"):
            raise ValueError(f"policy type {policy_config.type!r} does not support n_action_steps")
        policy_config.n_action_steps = args.policy_n_action_steps
    if args.policy_temporal_ensemble_coeff is not None:
        if not hasattr(policy_config, "temporal_ensemble_coeff"):
            raise ValueError(f"policy type {policy_config.type!r} does not support temporal_ensemble_coeff")
        policy_config.temporal_ensemble_coeff = args.policy_temporal_ensemble_coeff
    if args.policy_disable_temporal_ensemble:
        if not hasattr(policy_config, "temporal_ensemble_coeff"):
            raise ValueError(f"policy type {policy_config.type!r} does not support temporal_ensemble_coeff")
        if args.policy_temporal_ensemble_coeff is not None:
            raise ValueError("--policy-disable-temporal-ensemble conflicts with --policy-temporal-ensemble-coeff")
        policy_config.temporal_ensemble_coeff = None
    policy_cls = get_policy_class(policy_config.type)
    policy = policy_cls.from_pretrained(policy_path, config=policy_config)
    policy.to(device)
    policy.eval()
    policy.reset()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(policy_path),
        preprocessor_overrides={"device_processor": {"device": device.type}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )

    policy_camera_names = detect_camera_features(policy)
    capture_camera_names = policy_camera_names or list(CAMERAS)
    include_environment_state = uses_env_state(policy)
    action_dim = policy.config.output_features["action"].shape[0]
    if action_dim != len(JOINT_NAMES):
        raise ValueError(f"policy action dim={action_dim}, but OMX follower has {len(JOINT_NAMES)} joints")

    print(
        "lerobot_eval_config "
        f"policy_path={policy_path} device={device} steps={args.steps} max_duration={args.max_duration:g} "
        f"hz={args.hz:g} camera_fps={args.camera_fps} cameras={capture_camera_names} "
        f"capture_size={args.camera_width}x{args.camera_height} "
        f"model_size={args.model_width}x{args.model_height}",
        flush=True,
    )
    print(f"joint_order={JOINT_NAMES}", flush=True)
    metadata = {
        "policy_path": str(policy_path),
        "policy_type": policy.config.type,
        "device": str(device),
        "steps": args.steps,
        "max_duration": args.max_duration,
        "hz": args.hz,
        "camera_fps": args.camera_fps,
        "cameras": capture_camera_names,
        "joint_order": list(JOINT_NAMES),
        "max_joint_delta": args.max_joint_delta,
        "action_space": args.action_space,
        "n_action_steps": getattr(policy.config, "n_action_steps", None),
        "horizon": getattr(policy.config, "horizon", None),
        "policy_n_action_steps_override": args.policy_n_action_steps,
        "policy_temporal_ensemble_coeff_override": args.policy_temporal_ensemble_coeff,
        "policy_disable_temporal_ensemble": args.policy_disable_temporal_ensemble,
        "effective_temporal_ensemble_coeff": getattr(policy.config, "temporal_ensemble_coeff", None),
        "action_smoothing_alpha": args.action_smoothing_alpha,
        "warm_start_steps": args.warm_start_steps,
        "action_interpolation_steps": args.action_interpolation_steps,
        "action_interpolation_hz": args.action_interpolation_hz,
        "training_dataset_repo_id": run_config.get("dataset_repo_id"),
        "training_dataset_revision": run_config.get("dataset_revision"),
    }
    log_path = Path(args.log_path) if args.log_path else default_log_path(policy_path)
    video_dir = args.video_dir if args.video_dir is not None else default_video_dir(policy_path)
    save_video_cameras = parse_camera_list(args.save_camera, capture_camera_names)
    video_paths = {
        camera: video_dir / f"{camera}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.mp4"
        for camera in save_video_cameras
    }
    video_writers = {}
    rollout_entries: list[dict] = []
    exit_reason = "completed"
    saved_rollout_camera_frames = False

    follower = create_follower(
        port=args.follower_port,
        camera_names=capture_camera_names,
        camera_fps=args.camera_fps,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
    )
    if not args.no_rerun:
        if rr is None:
            raise RuntimeError("rerun-sdk is not installed; pass --no-rerun or install rerun-sdk")
        rrd_path = init_rerun(
            "lerobot_eval",
            has_camera=bool(capture_camera_names),
            camera_names=capture_camera_names,
            save_rrd=not args.no_save_rrd,
        )
        if rrd_path is not None:
            print(f"rerun_rrd={rrd_path}", flush=True)

    follower.connect(calibrate=False)
    try:
        for camera, video_path in video_paths.items():
            video_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Saving raw {camera} camera video to {video_path}", flush=True)

        if args.move_to_dataset_start_state:
            repo_id = args.start_state_repo_id or run_config.get("dataset_repo_id")
            revision = args.start_state_revision or run_config.get("dataset_revision") or "main"
            if not repo_id:
                raise ValueError(
                    "No training dataset repo found for --move-to-dataset-start-state. "
                    "Pass --start-state-repo-id explicitly."
                )
            target_start_state = dataset_start_state(repo_id, revision, args.start_state_reducer)
            metadata["dataset_start_state_repo_id"] = repo_id
            metadata["dataset_start_state_revision"] = revision
            metadata["dataset_start_state_reducer"] = args.start_state_reducer
            metadata["dataset_start_state"] = vector_by_joint(target_start_state)
            metadata["dataset_start_state_tolerance"] = args.start_state_tolerance
            metadata["dataset_start_state_warning_tolerance"] = args.start_state_warning_tolerance
            metadata["dataset_start_state_timeout"] = args.start_state_duration
            metadata["dataset_start_state_ramp_seconds"] = args.start_state_ramp_seconds
            print(
                "Moving follower to dataset "
                f"{args.start_state_reducer} start state from {repo_id}@{revision}: "
                f"{vector_by_joint(target_start_state)}",
                flush=True,
            )
            rollout_entries.extend(
                move_to_start_state(
                    follower,
                    target=target_start_state,
                    timeout=args.start_state_duration,
                    ramp_seconds=args.start_state_ramp_seconds,
                    hz=args.hz,
                    max_delta=args.start_state_max_delta,
                    tolerance=args.start_state_tolerance,
                    log_every=args.start_state_log_every,
                )
            )
            start_state_entries = [
                entry for entry in rollout_entries if entry.get("phase") == "move_to_dataset_start"
            ]
            if start_state_entries and start_state_entries[-1].get("reached"):
                print(
                    "Reached dataset start state "
                    f"within {args.start_state_tolerance:g} joint units.",
                    flush=True,
                )
            elif start_state_entries:
                final_error = start_state_entries[-1].get("max_abs_error", float("nan"))
                if final_error <= args.start_state_warning_tolerance:
                    print(
                        "Dataset start-state move timed out but is close enough; "
                        f"max_abs_error={final_error:.3f}. Continuing rollout.",
                        flush=True,
                    )
                else:
                    print(
                        "WARNING: dataset start-state move timed out; "
                        f"max_abs_error={final_error:.3f}. Continuing rollout.",
                        flush=True,
                    )
            if args.start_state_settle_seconds > 0:
                print(f"Settling for {args.start_state_settle_seconds:.1f}s before rollout.", flush=True)
                time.sleep(args.start_state_settle_seconds)

        if args.start_delay > 0:
            print(f"Starting stock LeRobot rollout in {args.start_delay:.1f}s. Press Ctrl+C to stop.", flush=True)
            time.sleep(args.start_delay)

        start_time = time.perf_counter()
        previous_smoothed_action = None
        previous_sent_target = None
        for step in range(args.steps):
            elapsed = time.perf_counter() - start_time
            if elapsed >= args.max_duration:
                print(f"\nReached max_duration={args.max_duration:.1f}s at step={step}.", flush=True)
                exit_reason = "max_duration"
                break

            loop_start = time.perf_counter()
            try:
                obs_dict, state, raw_camera_frames, model_camera_frames = build_observation(
                    follower,
                    policy_camera_names=policy_camera_names,
                    capture_camera_names=capture_camera_names,
                    model_width=args.model_width,
                    model_height=args.model_height,
                    include_environment_state=include_environment_state,
                )
            except TimeoutError as exc:
                print(f"\nStopping stock LeRobot rollout: stale camera frame at step {step}: {exc}", flush=True)
                exit_reason = "stale_camera_frame"
                break

            if args.save_camera_frames_dir is not None and not saved_rollout_camera_frames:
                save_camera_frames(args.save_camera_frames_dir, raw_camera_frames, prefix="eval_raw", step=step)
                save_camera_frames(args.save_camera_frames_dir, model_camera_frames, prefix="eval_model", step=step)
                saved_rollout_camera_frames = True
            for camera, video_path in video_paths.items():
                import cv2

                frame = raw_camera_frames[camera]
                if camera not in video_writers:
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(video_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        args.video_fps or args.hz,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"failed to open video writer: {video_path}")
                    video_writers[camera] = writer
                video_writers[camera].write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            action_queue_before = action_queue_len(policy)
            with torch.inference_mode():
                if args.task is not None:
                    obs_dict["task"] = args.task
                obs_dict = preprocessor(obs_dict)
                action_values = postprocessor(policy.select_action(obs_dict)).squeeze(0).cpu()

            if args.action_space == "delta":
                # Policy was trained on action = recorded_action - state, so the raw
                # output is a per-joint offset; reconstruct the absolute command from
                # the current observed state. Everything downstream stays absolute.
                action_values = action_values + torch.as_tensor(state, dtype=action_values.dtype)

            action = {}
            raw_action_np = action_values.numpy()
            if previous_smoothed_action is None:
                smoothed_action = raw_action_np.copy()
            else:
                alpha = args.action_smoothing_alpha
                smoothed_action = previous_smoothed_action + alpha * (raw_action_np - previous_smoothed_action)
            if args.warm_start_steps > 0 and step < args.warm_start_steps:
                warm_alpha = (step + 1) / args.warm_start_steps
                smoothed_action = state + warm_alpha * (smoothed_action - state)
            previous_smoothed_action = smoothed_action.copy()

            target_action = smoothed_action.copy()
            for i, name in enumerate(JOINT_NAMES):
                value = float(target_action[i])
                if args.max_joint_delta > 0:
                    value = float(np.clip(value, state[i] - args.max_joint_delta, state[i] + args.max_joint_delta))
                target_action[i] = value
                action[f"{name}.pos"] = value
            interpolation_start = previous_sent_target if previous_sent_target is not None else state
            sent_action = send_interpolated_action(
                follower,
                start=interpolation_start,
                target=target_action,
                steps=args.action_interpolation_steps,
                command_hz=args.action_interpolation_hz,
            )
            previous_sent_target = target_action.copy()
            loop_dt = time.perf_counter() - loop_start
            loop_hz = 1.0 / loop_dt if loop_dt > 0 else float("inf")
            delta_values = {name: float(sent_action[f"{name}.pos"] - state[i]) for i, name in enumerate(JOINT_NAMES)}
            rollout_entries.append(
                {
                    "step": step,
                    "elapsed_seconds": elapsed,
                    "loop_seconds": loop_dt,
                    "loop_hz": loop_hz,
                    "policy_replanned": action_queue_before == 0 if action_queue_before is not None else None,
                    "state": vector_by_joint(state),
                    "raw_action": vector_by_joint(action_values),
                    "smoothed_action": vector_by_joint(smoothed_action),
                    "sent_action": {name: float(sent_action[f"{name}.pos"]) for name in JOINT_NAMES},
                    "delta": delta_values,
                }
            )

            if not args.no_rerun:
                rr.set_time("step", sequence=step)
                for i, name in enumerate(JOINT_NAMES):
                    rr.log(f"joints/{name}/state", rr.Scalars([float(state[i])]))
                    rr.log(f"joints/{name}/policy_action", rr.Scalars([float(action_values[i])]))
                    rr.log(f"joints/{name}/sent_action", rr.Scalars([float(sent_action[f'{name}.pos'])]))
                rr.log("metrics/loop_hz", rr.Scalars([loop_hz]))
                for camera in raw_camera_frames:
                    rr.log(f"camera/{camera}/raw", rr.Image(raw_camera_frames[camera]))
                    rr.log(f"camera/{camera}/model", rr.Image(model_camera_frames[camera]))

            if args.log_every > 0 and step % args.log_every == 0:
                state_preview = "  |  ".join(f"{name}: {state[i]:7.2f}" for i, name in enumerate(JOINT_NAMES))
                raw_preview = "  |  ".join(f"{name}: {action_values[i].item():7.2f}" for i, name in enumerate(JOINT_NAMES))
                sent_preview = "  |  ".join(f"{name}: {sent_action[f'{name}.pos']:7.2f}" for name in JOINT_NAMES)
                delta_preview = "  |  ".join(
                    f"{name}: {sent_action[f'{name}.pos'] - state[i]:7.2f}"
                    for i, name in enumerate(JOINT_NAMES)
                )
                print(
                    f"step={step:05d} hz={loop_hz:5.1f}\n"
                    f"  state: {state_preview}\n"
                    f"  raw:   {raw_preview}\n"
                    f"  sent:  {sent_preview}\n"
                    f"  delta: {delta_preview}",
                    flush=True,
                )

            time.sleep(max(0.0, 1.0 / args.hz - loop_dt))
    except KeyboardInterrupt:
        print("\nStopping stock LeRobot rollout.", flush=True)
        exit_reason = "keyboard_interrupt"
    finally:
        safe_disconnect(follower)
        for camera, writer in video_writers.items():
            writer.release()
            print(f"saved_camera_video={video_paths[camera]}", flush=True)
        write_rollout_log(log_path, metadata, rollout_entries, exit_reason)


if __name__ == "__main__":
    main()
