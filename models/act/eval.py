import argparse
from datetime import datetime
from dataclasses import fields
import json
from pathlib import Path
import time
import zlib

import cv2
import numpy as np
import rerun as rr
import torch
from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.omx_follower.config_omx_follower import OmxFollowerConfig
from lerobot.robots.omx_follower.omx_follower import OmxFollower

from configs.act.base import ACTConfig
from models.act.act import ACTPolicy
from utils.config import CAMERA_HEIGHT, CAMERA_WIDTH, CAMERAS, FOLLOWER_PORT, FPS, JOINT_NAMES
from utils.rerun_utils import init_rerun


def _disable_auto_calibration(robot):
    original_connect = robot.connect

    def connect_without_calibration(*args, **kwargs):
        kwargs["calibrate"] = False
        return original_connect(*args, **kwargs)

    robot.connect = connect_without_calibration
    return robot


def create_follower(
    *,
    port: str,
    camera_names: tuple[str, ...],
    fps: int,
    camera_width: int,
    camera_height: int,
) -> OmxFollower:
    cameras = {}
    for name in camera_names:
        if name not in CAMERAS:
            raise KeyError(f"camera '{name}' is not configured in utils.config.CAMERAS")
        cameras[name] = OpenCVCameraConfig(
            index_or_path=CAMERAS[name],
            fps=fps,
            width=camera_width,
            height=camera_height,
            warmup_s=5,
        )
    config = OmxFollowerConfig(port=port, id="omx_follower_arm", cameras=cameras)
    return _disable_auto_calibration(OmxFollower(config))


def read_obs(
    follower: OmxFollower,
    camera_names: tuple[str, ...],
    *,
    image_width: int,
    image_height: int,
) -> dict[str, object]:
    obs = follower.get_observation()
    qpos = np.array([obs[f"{name}.pos"] for name in JOINT_NAMES], dtype=np.float32)
    images = {}
    raw_images = {}
    camera_debug = {}
    for camera in camera_names:
        if camera not in obs:
            raise RuntimeError(f"follower did not return camera '{camera}'; available keys: {list(obs.keys())}")
        raw_frame = obs[camera]
        raw_images[camera] = raw_frame
        frame = raw_frame
        height, width = raw_frame.shape[:2]
        if (width, height) != (image_width, image_height):
            frame = cv2.resize(raw_frame, (image_width, image_height))
        images[camera] = frame
        camera_debug[camera] = summarize_camera_frame(follower, camera, raw_frame)
    return {"qpos": qpos, "images": images, "raw_images": raw_images, "camera_debug": camera_debug}


def summarize_camera_frame(follower: OmxFollower, camera_name: str, frame: np.ndarray) -> dict[str, object]:
    camera = follower.cameras.get(camera_name)
    timestamp = None
    if camera is not None:
        frame_lock = getattr(camera, "frame_lock", None)
        if frame_lock is None:
            timestamp = getattr(camera, "latest_timestamp", None)
        else:
            with frame_lock:
                timestamp = getattr(camera, "latest_timestamp", None)

    age_ms = None if timestamp is None else (time.perf_counter() - float(timestamp)) * 1000.0
    sample = np.ascontiguousarray(frame[::16, ::16])
    return {
        "age_ms": None if age_ms is None else float(age_ms),
        "shape": [int(value) for value in frame.shape],
        "dtype": str(frame.dtype),
        "min": int(frame.min()),
        "max": int(frame.max()),
        "mean": float(frame.mean()),
        "std": float(frame.std()),
        "sample_crc32": int(zlib.crc32(sample.tobytes()) & 0xFFFFFFFF),
    }


def send_action(follower: OmxFollower, action: np.ndarray) -> dict[str, float]:
    action = np.asarray(action, dtype=np.float32).flatten()
    if action.shape[0] != len(JOINT_NAMES):
        raise ValueError(f"policy returned {action.shape[0]} actions, but OMX follower expects {len(JOINT_NAMES)}")
    return follower.send_action({f"{name}.pos": float(action[i]) for i, name in enumerate(JOINT_NAMES)})


def parse_camera_list(value: str) -> set[str]:
    return {camera.strip() for camera in value.split(",") if camera.strip()}


def safe_disconnect(robot) -> None:
    try:
        robot.disconnect()
    except RuntimeError:
        print("Warning: clean disconnect failed (motor hardware error). Power-cycling recommended.", flush=True)


def validate_checkpoint_shapes(
    *,
    cfg: ACTConfig,
    q_mean: torch.Tensor,
    q_std: torch.Tensor,
    a_mean: torch.Tensor,
    a_std: torch.Tensor,
) -> None:
    expected_shape = (len(JOINT_NAMES),)
    if cfg.joint_dim != len(JOINT_NAMES):
        raise ValueError(f"checkpoint joint_dim={cfg.joint_dim}, but OMX follower has {len(JOINT_NAMES)} joints")
    for name, value in {
        "observation.state mean": q_mean,
        "observation.state std": q_std,
        "action mean": a_mean,
        "action std": a_std,
    }.items():
        if tuple(value.shape) != expected_shape:
            raise ValueError(f"checkpoint {name} shape {tuple(value.shape)} does not match OMX joints {expected_shape}")
    for camera in cfg.camera_names:
        if camera not in CAMERAS:
            raise KeyError(f"checkpoint expects camera '{camera}', but utils.config.CAMERAS only has {tuple(CAMERAS)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--hz", type=float, default=float(FPS))
    parser.add_argument("--camera-fps", type=int, default=FPS, help="Camera FPS requested from OpenCV. Independent from control-loop --hz.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--follower-port", default=FOLLOWER_PORT)
    parser.add_argument("--camera-width", type=int, default=CAMERA_WIDTH, help="Camera capture width requested from OpenCV.")
    parser.add_argument("--camera-height", type=int, default=CAMERA_HEIGHT, help="Camera capture height requested from OpenCV.")
    parser.add_argument("--model-width", type=int, default=CAMERA_WIDTH, help="Image width fed to ACT.")
    parser.add_argument("--model-height", type=int, default=CAMERA_HEIGHT, help="Image height fed to ACT.")
    parser.add_argument("--start-delay", type=float, default=3.0)
    parser.add_argument("--log-every", type=int, default=1, help="Print debug logs every N steps. Use 0 to disable detailed per-step logs.")
    parser.add_argument("--no-rerun", action="store_true", help="Disable Rerun viewer and .rrd logging.")
    parser.add_argument("--no-save-rrd", action="store_true", help="Show Rerun live viewer without saving outputs/rerun/act_eval.rrd.")
    parser.add_argument("--no-temporal-ensemble", action="store_true", help="Execute only the first action from each predicted chunk.")
    parser.add_argument("--save-camera", default=None, help="Camera name to save as MP4, e.g. top or wrist.")
    parser.add_argument("--video-path", default=None, help="Output MP4 path. Defaults to outputs/eval_videos/<camera>_<timestamp>.mp4.")
    parser.add_argument("--video-fps", type=float, default=None, help="FPS metadata for saved MP4. Defaults to --hz.")
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
        "current_qpos + policy_output (use for checkpoints trained on delta-action datasets).",
    )
    parser.add_argument(
        "--warm-start-steps",
        type=int,
        default=0,
        help="For the first N control steps, ramp from current observed qpos to the policy command. Use 0 to disable.",
    )
    parser.add_argument(
        "--zero-cameras",
        default="",
        help="Comma-separated checkpoint camera names to replace with black frames for model input only, e.g. wrist or top,wrist.",
    )
    parser.add_argument(
        "--freeze-cameras",
        default="",
        help="Comma-separated checkpoint camera names to hold at their first observed frame for model input only.",
    )
    parser.add_argument(
        "--trace-path",
        default="auto",
        help="Output JSONL trace path for per-step qpos/action/timing data. Use 'auto' for outputs/eval_traces/act_eval_<timestamp>.jsonl or 'none' to disable.",
    )
    args = parser.parse_args()

    device = torch.device(args.device if args.device != "auto" else "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ACTConfig(**{f.name: ckpt["config"][f.name] for f in fields(ACTConfig) if f.name in ckpt["config"]})
    zero_cameras = parse_camera_list(args.zero_cameras)
    freeze_cameras = parse_camera_list(args.freeze_cameras)
    unknown_ablation_cameras = (zero_cameras | freeze_cameras) - set(cfg.camera_names)
    if unknown_ablation_cameras:
        raise ValueError(f"camera ablation requested unknown cameras {sorted(unknown_ablation_cameras)}; checkpoint cameras={cfg.camera_names}")
    overlap_ablation_cameras = zero_cameras & freeze_cameras
    if overlap_ablation_cameras:
        raise ValueError(f"cameras cannot be both zeroed and frozen: {sorted(overlap_ablation_cameras)}")
    stats = ckpt.get("norm_stats") or LeRobotDatasetMetadata(cfg.dataset_repo_id, revision=cfg.dataset_revision).stats
    q_mean = torch.as_tensor(stats["observation.state"]["mean"], device=device, dtype=torch.float32).flatten()
    q_std = torch.as_tensor(stats["observation.state"]["std"], device=device, dtype=torch.float32).flatten().clamp_min(1e-6)
    a_mean = torch.as_tensor(stats["action"]["mean"], dtype=torch.float32).flatten()
    a_std = torch.as_tensor(stats["action"]["std"], dtype=torch.float32).flatten().clamp_min(1e-6)
    validate_checkpoint_shapes(cfg=cfg, q_mean=q_mean, q_std=q_std, a_mean=a_mean, a_std=a_std)
    print(
        "eval_config "
        f"checkpoint={args.checkpoint} device={device} joint_dim={cfg.joint_dim} "
        f"chunk_size={cfg.chunk_size} cameras={cfg.camera_names} "
        f"temporal_ensemble={not args.no_temporal_ensemble} "
        f"action_space={args.action_space} "
        f"zero_cameras={sorted(zero_cameras)} freeze_cameras={sorted(freeze_cameras)} "
        f"follower_port={args.follower_port} hz={args.hz:g} camera_fps={args.camera_fps} "
        f"capture_size={args.camera_width}x{args.camera_height} "
        f"model_size={args.model_width}x{args.model_height}",
        flush=True,
    )
    print(f"joint_order={JOINT_NAMES}", flush=True)
    print(
        "norm_stats "
        f"q_mean={np.round(q_mean.cpu().numpy(), 4).tolist()} "
        f"q_std={np.round(q_std.cpu().numpy(), 4).tolist()} "
        f"a_mean={np.round(a_mean.numpy(), 4).tolist()} "
        f"a_std={np.round(a_std.numpy(), 4).tolist()}",
        flush=True,
    )
    im_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    im_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)

    model = ACTPolicy(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.reset_action_selection()

    follower = create_follower(
        port=args.follower_port,
        camera_names=cfg.camera_names,
        fps=args.camera_fps,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
    )
    if not args.no_rerun:
        rrd_path = init_rerun(
            "act_eval",
            has_camera=True,
            camera_names=list(cfg.camera_names),
            save_rrd=not args.no_save_rrd,
        )
        if rrd_path is not None:
            print(f"rerun_rrd={rrd_path}", flush=True)
    video_writer = None
    video_path = None
    if args.save_camera is not None:
        if args.save_camera not in cfg.camera_names:
            raise ValueError(f"--save-camera {args.save_camera!r} is not in checkpoint cameras {cfg.camera_names}")
        video_path = Path(args.video_path) if args.video_path else Path("outputs/eval_videos") / f"{args.save_camera}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving raw {args.save_camera} camera video to {video_path}", flush=True)
    trace_file = None
    trace_path = None
    if args.trace_path.lower() != "none":
        trace_path = (
            Path("outputs/eval_traces") / f"act_eval_{datetime.now().strftime('%Y%m%d-%H%M%S')}.jsonl"
            if args.trace_path == "auto"
            else Path(args.trace_path)
        )
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_file = trace_path.open("w")
        print(f"Saving eval trace to {trace_path}", flush=True)
    follower.connect(calibrate=False)
    frozen_camera_frames = {}
    try:
        if args.start_delay > 0:
            print(f"Starting live ACT eval in {args.start_delay:.1f}s. Press Ctrl+C to stop.", flush=True)
            time.sleep(args.start_delay)
        with torch.inference_mode():
            for step in range(args.steps):
                start = time.perf_counter()
                try:
                    obs = read_obs(
                        follower,
                        cfg.camera_names,
                        image_width=args.model_width,
                        image_height=args.model_height,
                    )
                except TimeoutError as exc:
                    print(f"\nStopping live ACT eval: stale camera frame at step {step}: {exc}", flush=True)
                    break
                t_obs = time.perf_counter()
                qpos = torch.as_tensor(obs["qpos"], device=device, dtype=torch.float32).flatten()
                if qpos.shape != q_mean.shape:
                    raise ValueError(f"observation qpos shape {tuple(qpos.shape)} does not match stats shape {tuple(q_mean.shape)}")
                qpos_normalized = ((qpos - q_mean) / q_std).unsqueeze(0)
                imgs = []
                camera_ablation = {}
                for camera in cfg.camera_names:
                    image_array = obs["images"][camera]
                    ablation = "live"
                    if camera in zero_cameras:
                        image_array = np.zeros_like(image_array)
                        ablation = "zero"
                    elif camera in freeze_cameras:
                        if camera not in frozen_camera_frames:
                            frozen_camera_frames[camera] = np.array(image_array, copy=True)
                        image_array = frozen_camera_frames[camera]
                        ablation = "freeze"
                    camera_ablation[camera] = ablation
                    image = torch.as_tensor(np.ascontiguousarray(image_array), device=device, dtype=torch.float32)
                    image = image.permute(2, 0, 1) / 255.0
                    imgs.append((image - im_mean) / im_std)
                t_preproc = time.perf_counter()
                action_normalized = model.select_action(
                    torch.stack(imgs, dim=0).unsqueeze(0),
                    qpos_normalized,
                    use_temporal_ensemble=not args.no_temporal_ensemble,
                ).squeeze(0).cpu()
                t_model = time.perf_counter()
                action = (action_normalized * a_std + a_mean).numpy()
                if args.action_space == "delta":
                    # Checkpoint trained on action = recorded_action - qpos; reconstruct
                    # the absolute command from the current observed qpos. Warm-start and
                    # clamp below then operate in absolute space as usual.
                    action = action + obs["qpos"]
                if not np.isfinite(action).all():
                    raise FloatingPointError(f"non-finite action at step {step}: {action}")
                unclipped_action = action.copy()
                if args.warm_start_steps > 0 and step < args.warm_start_steps:
                    alpha = (step + 1) / args.warm_start_steps
                    action = obs["qpos"] + alpha * (action - obs["qpos"])
                if args.max_joint_delta > 0:
                    action = np.clip(action, obs["qpos"] - args.max_joint_delta, obs["qpos"] + args.max_joint_delta)
                sent_action = send_action(follower, action)
                t_send = time.perf_counter()
                video_dt = 0.0
                if args.save_camera is not None:
                    t_video_start = time.perf_counter()
                    frame = obs["raw_images"][args.save_camera]
                    if video_writer is None:
                        height, width = frame.shape[:2]
                        video_writer = cv2.VideoWriter(
                            str(video_path),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            args.video_fps or args.hz,
                            (width, height),
                        )
                        if not video_writer.isOpened():
                            raise RuntimeError(f"failed to open video writer: {video_path}")
                    video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    video_dt = time.perf_counter() - t_video_start
                rerun_dt = 0.0
                if not args.no_rerun:
                    t_rerun_start = time.perf_counter()
                    rr.set_time("step", sequence=step)
                    for i, name in enumerate(JOINT_NAMES):
                        rr.log(f"joints/{name}/state", rr.Scalars([float(obs["qpos"][i])]))
                        rr.log(f"joints/{name}/policy_action", rr.Scalars([float(action[i])]))
                        rr.log(f"joints/{name}/policy_action_normalized", rr.Scalars([float(action_normalized[i])]))
                        rr.log(f"joints/{name}/sent_action", rr.Scalars([float(sent_action[f'{name}.pos'])]))
                    for camera, frame in obs["raw_images"].items():
                        rr.log(f"camera/{camera}/raw", rr.Image(frame))
                    for camera, frame in obs["images"].items():
                        rr.log(f"camera/{camera}/model", rr.Image(frame))
                    for camera, debug in obs["camera_debug"].items():
                        if debug["age_ms"] is not None:
                            rr.log(f"camera/{camera}/age_ms", rr.Scalars([float(debug["age_ms"])]))
                        rr.log(f"camera/{camera}/sample_crc32", rr.Scalars([float(debug["sample_crc32"])]))
                        rr.log(f"camera/{camera}/mean", rr.Scalars([float(debug["mean"])]))
                        rr.log(f"camera/{camera}/std", rr.Scalars([float(debug["std"])]))
                    rerun_dt = time.perf_counter() - t_rerun_start
                loop_dt = time.perf_counter() - start
                sleep_dt = max(0.0, 1.0 / args.hz - loop_dt)
                timing = {
                    "obs_ms": (t_obs - start) * 1000.0,
                    "preproc_ms": (t_preproc - t_obs) * 1000.0,
                    "model_ms": (t_model - t_preproc) * 1000.0,
                    "send_ms": (t_send - t_model) * 1000.0,
                    "video_ms": video_dt * 1000.0,
                    "rerun_ms": rerun_dt * 1000.0,
                    "loop_ms": loop_dt * 1000.0,
                    "sleep_ms": sleep_dt * 1000.0,
                }
                if not args.no_rerun:
                    rr.log("metrics/loop_hz", rr.Scalars([1.0 / loop_dt if loop_dt > 0 else float("inf")]))
                    for name, value in timing.items():
                        rr.log(f"timing/{name}", rr.Scalars([float(value)]))
                if trace_file is not None:
                    trace_record = {
                        "step": step,
                        "timestamp_s": time.time(),
                        "joint_names": JOINT_NAMES,
                        "qpos": [float(value) for value in obs["qpos"]],
                        "qpos_normalized": [float(value) for value in qpos_normalized.squeeze(0).cpu().numpy()],
                        "policy_action_normalized": [float(value) for value in action_normalized.numpy()],
                        "policy_action_unclipped": [float(value) for value in unclipped_action],
                        "policy_action": [float(value) for value in action],
                        "sent_action": [float(sent_action[f"{name}.pos"]) for name in JOINT_NAMES],
                        "timing_ms": {name: float(value) for name, value in timing.items()},
                        "cameras": obs["camera_debug"],
                        "camera_ablation": camera_ablation,
                    }
                    trace_file.write(json.dumps(trace_record, sort_keys=True) + "\n")
                    trace_file.flush()
                if args.log_every > 0:
                    if step % args.log_every == 0:
                        raw_camera_summary = {
                            name: {
                                "shape": tuple(frame.shape),
                                "dtype": str(frame.dtype),
                                "min": int(frame.min()),
                                "max": int(frame.max()),
                            }
                            for name, frame in obs["raw_images"].items()
                        }
                        model_camera_summary = {
                            name: {
                                "shape": tuple(frame.shape),
                                "dtype": str(frame.dtype),
                                "min": int(frame.min()),
                                "max": int(frame.max()),
                            }
                            for name, frame in obs["images"].items()
                        }
                        camera_freshness = {
                            name: {
                                "age_ms": None if debug["age_ms"] is None else round(float(debug["age_ms"]), 1),
                                "mean": round(float(debug["mean"]), 2),
                                "std": round(float(debug["std"]), 2),
                                "sample_crc32": int(debug["sample_crc32"]),
                            }
                            for name, debug in obs["camera_debug"].items()
                        }
                        print(
                            "debug_step "
                            f"step={step} hz={(1.0 / loop_dt if loop_dt > 0 else float('inf')):.1f} "
                            f"qpos={np.round(obs['qpos'], 4).tolist()} "
                            f"qpos_norm={np.round(qpos_normalized.squeeze(0).cpu().numpy(), 4).tolist()} "
                            f"action_norm={np.round(action_normalized.numpy(), 4).tolist()} "
                            f"action_unclipped={np.round(unclipped_action, 4).tolist()} "
                            f"action={np.round(action, 4).tolist()} "
                            f"sent_action={sent_action} "
                            f"timing_ms={{{', '.join(f'{name}: {value:.1f}' for name, value in timing.items())}}} "
                            f"camera_ablation={camera_ablation} "
                            f"raw_cameras={raw_camera_summary} "
                            f"model_cameras={model_camera_summary} "
                            f"camera_freshness={camera_freshness}",
                            flush=True,
                        )
                    else:
                        print(f"step={step} sent_action={sent_action}", flush=True)
                time.sleep(sleep_dt)
    except KeyboardInterrupt:
        print("\nStopping live ACT eval.", flush=True)
    finally:
        if video_writer is not None:
            video_writer.release()
            print(f"Saved video: {video_path}", flush=True)
        if trace_file is not None:
            trace_file.close()
            print(f"Saved eval trace: {trace_path}", flush=True)
        safe_disconnect(follower)


if __name__ == "__main__":
    main()
