"""Offline counterfactual probe for custom ACT checkpoints.

Loads one LeRobot dataset frame, applies controlled camera/qpos mutations, and
reports how much the predicted custom-ACT action chunk changes. This is intended
to answer whether a policy is object/scene-conditioned before running hardware.

Example:
    uv run python evaluation/act_counterfactual_probe.py \
      --checkpoint outputs/vastai/instance-41363766/outputs/act_experiments/act_custom_train_chunk100_steps20000/20260617-182438/checkpoints/step_020000.pt \
      --episode 0 --frame-index 0 \
      --mask top:220,160,120,120 \
      --move top:220,160,120,120,120,0
"""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
import torch
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.act.base import ACTConfig
from models.act.act import ACTPolicy


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def parse_rect(value: str) -> tuple[str, int, int, int, int]:
    try:
        camera, rest = value.split(":", 1)
        x, y, width, height = [int(part) for part in rest.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "rect must be formatted as camera:x,y,width,height, e.g. top:220,160,120,120"
        ) from exc
    return camera, x, y, width, height


def parse_move(value: str) -> tuple[str, int, int, int, int, int, int]:
    try:
        camera, rest = value.split(":", 1)
        x, y, width, height, dx, dy = [int(part) for part in rest.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "move must be formatted as camera:x,y,width,height,dx,dy, e.g. top:220,160,120,120,120,0"
        ) from exc
    return camera, x, y, width, height, dx, dy


def clamp_rect(image: np.ndarray, x: int, y: int, width: int, height: int) -> tuple[int, int, int, int]:
    h, w = image.shape[:2]
    x0 = max(0, min(w, x))
    y0 = max(0, min(h, y))
    x1 = max(x0, min(w, x + width))
    y1 = max(y0, min(h, y + height))
    return x0, y0, x1, y1


def image_to_hwc_float(image: torch.Tensor) -> np.ndarray:
    array = image.detach().cpu().numpy()
    if array.ndim == 3 and array.shape[0] in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    if array.max() > 1.0:
        array = array / 255.0
    return np.ascontiguousarray(array.astype(np.float32))


def fill_value(camera: str, stats: dict[str, dict[str, Any]], mode: str) -> np.ndarray | float:
    if mode == "black":
        return 0.0
    key = f"observation.images.{camera}"
    mean = np.asarray(stats[key]["mean"], dtype=np.float32).reshape(1, 1, 3)
    return mean


def mask_rect(image: np.ndarray, camera: str, rect: tuple[int, int, int, int], stats, fill_mode: str) -> np.ndarray:
    out = image.copy()
    x0, y0, x1, y1 = clamp_rect(out, *rect)
    out[y0:y1, x0:x1] = fill_value(camera, stats, fill_mode)
    return out


def move_rect(image: np.ndarray, rect: tuple[int, int, int, int], dx: int, dy: int, fill: np.ndarray | float) -> np.ndarray:
    out = image.copy()
    x0, y0, x1, y1 = clamp_rect(out, *rect)
    patch = out[y0:y1, x0:x1].copy()
    out[y0:y1, x0:x1] = fill

    nx0, ny0, nx1, ny1 = clamp_rect(out, x0 + dx, y0 + dy, x1 - x0, y1 - y0)
    patch = patch[: ny1 - ny0, : nx1 - nx0]
    out[ny0:ny1, nx0:nx1] = patch
    return out


def load_sample(dataset: LeRobotDataset, *, dataset_index: int | None, episode: int, frame_index: int):
    if dataset_index is not None:
        return dataset_index, dataset[dataset_index]
    for idx in range(len(dataset)):
        row = dataset.hf_dataset[idx]
        if int(row["episode_index"]) == episode and int(row["frame_index"]) == frame_index:
            return idx, dataset[idx]
    raise ValueError(f"could not find episode={episode} frame_index={frame_index}")


def build_model_inputs(images: dict[str, np.ndarray], qpos: torch.Tensor, cfg: ACTConfig, device: torch.device):
    tensors = []
    for camera in cfg.camera_names:
        image = torch.from_numpy(np.ascontiguousarray(images[camera])).float().permute(2, 0, 1)
        tensors.append((image - IMAGENET_MEAN) / IMAGENET_STD)
    return torch.stack(tensors, dim=0).unsqueeze(0).to(device), qpos.to(device).unsqueeze(0)


def predict_chunk(
    model: ACTPolicy,
    images: dict[str, np.ndarray],
    qpos_normalized: torch.Tensor,
    cfg: ACTConfig,
    a_mean: torch.Tensor,
    a_std: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    image_tensor, qpos_tensor = build_model_inputs(images, qpos_normalized, cfg, device)
    with torch.inference_mode():
        pred_normalized, _, _ = model(image_tensor, qpos_tensor)
    pred_normalized = pred_normalized.squeeze(0).cpu()
    pred = pred_normalized * a_std + a_mean
    return pred_normalized.numpy(), pred.numpy()


def summarize_variant(
    name: str,
    pred: np.ndarray,
    pred_norm: np.ndarray,
    base_pred: np.ndarray,
    base_norm: np.ndarray,
    first_n: int,
) -> dict[str, Any]:
    n = min(first_n, pred.shape[0], base_pred.shape[0])
    raw_delta = pred[:n] - base_pred[:n]
    norm_delta = pred_norm[:n] - base_norm[:n]
    return {
        "name": name,
        "first_n": n,
        "first_action_l2": float(np.linalg.norm(raw_delta[0])),
        "first_action_linf": float(np.max(np.abs(raw_delta[0]))),
        "chunk_mean_l2": float(np.linalg.norm(raw_delta, axis=1).mean()),
        "chunk_max_l2": float(np.linalg.norm(raw_delta, axis=1).max()),
        "normalized_first_action_l2": float(np.linalg.norm(norm_delta[0])),
        "normalized_chunk_mean_l2": float(np.linalg.norm(norm_delta, axis=1).mean()),
        "first_action_delta": raw_delta[0].tolist(),
        "per_step_l2": np.linalg.norm(raw_delta, axis=1).tolist(),
        "predicted_actions": pred[:n].tolist(),
    }


def save_contact_sheet(path: Path, variants: dict[str, dict[str, np.ndarray]], camera_names: tuple[str, ...]) -> None:
    thumbs = []
    for variant_name, images in variants.items():
        for camera in camera_names:
            image = (np.clip(images[camera], 0.0, 1.0) * 255).astype(np.uint8)
            thumb = cv2.resize(image, (240, 180), interpolation=cv2.INTER_AREA)
            cv2.rectangle(thumb, (0, 0), (240, 26), (0, 0, 0), -1)
            cv2.putText(
                thumb,
                f"{variant_name}:{camera}",
                (4, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            thumbs.append(thumb)
    cols = len(camera_names)
    rows = int(np.ceil(len(thumbs) / cols))
    sheet = np.zeros((rows * 180, cols * 240, 3), dtype=np.uint8)
    for idx, thumb in enumerate(thumbs):
        y = (idx // cols) * 180
        x = (idx % cols) * 240
        sheet[y : y + 180, x : x + 240] = cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sheet)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline counterfactual probe for custom ACT checkpoints.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--dataset-index", type=int, default=None)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--first-n", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mask", action="append", type=parse_rect, default=[], help="Add mask variant camera:x,y,w,h")
    parser.add_argument("--move", action="append", type=parse_move, default=[], help="Add move variant camera:x,y,w,h,dx,dy")
    parser.add_argument("--mask-fill", choices=("black", "mean"), default="black")
    parser.add_argument("--output-json", default="outputs/counterfactual_probes/act_counterfactual_probe.json")
    parser.add_argument("--output-image", default="outputs/counterfactual_probes/act_counterfactual_probe.jpg")
    args = parser.parse_args()

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_name)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ACTConfig(**{f.name: checkpoint["config"][f.name] for f in fields(ACTConfig) if f.name in checkpoint["config"]})
    metadata = LeRobotDatasetMetadata(args.dataset, revision=args.revision)
    stats = checkpoint.get("norm_stats") or metadata.stats
    q_mean = torch.as_tensor(stats["observation.state"]["mean"], dtype=torch.float32).flatten()
    q_std = torch.as_tensor(stats["observation.state"]["std"], dtype=torch.float32).flatten().clamp_min(1e-6)
    a_mean = torch.as_tensor(stats["action"]["mean"], dtype=torch.float32).flatten()
    a_std = torch.as_tensor(stats["action"]["std"], dtype=torch.float32).flatten().clamp_min(1e-6)

    dataset = LeRobotDataset(args.dataset, revision=args.revision)
    dataset_index, sample = load_sample(
        dataset,
        dataset_index=args.dataset_index,
        episode=args.episode,
        frame_index=args.frame_index,
    )

    qpos = sample["observation.state"].float().flatten()
    qpos_normalized = (qpos - q_mean) / q_std
    base_images = {
        camera: image_to_hwc_float(sample[f"observation.images.{camera}"])
        for camera in cfg.camera_names
    }

    variants: dict[str, dict[str, np.ndarray]] = {"original": {k: v.copy() for k, v in base_images.items()}}
    for camera in cfg.camera_names:
        zeroed = {k: v.copy() for k, v in base_images.items()}
        zeroed[camera] = np.zeros_like(zeroed[camera])
        variants[f"zero_{camera}"] = zeroed
    variants["zero_all"] = {camera: np.zeros_like(image) for camera, image in base_images.items()}

    for camera in cfg.camera_names:
        image = base_images[camera]
        h, w = image.shape[:2]
        for name, rect in {
            "left": (0, 0, w // 2, h),
            "right": (w // 2, 0, w - w // 2, h),
            "center": (w // 4, h // 4, w // 2, h // 2),
        }.items():
            masked = {k: v.copy() for k, v in base_images.items()}
            masked[camera] = mask_rect(masked[camera], camera, rect, stats, args.mask_fill)
            variants[f"mask_{camera}_{name}"] = masked

    for idx, (camera, x, y, width, height) in enumerate(args.mask):
        if camera not in cfg.camera_names:
            raise ValueError(f"mask requested camera {camera!r}, but checkpoint cameras are {cfg.camera_names}")
        masked = {k: v.copy() for k, v in base_images.items()}
        masked[camera] = mask_rect(masked[camera], camera, (x, y, width, height), stats, args.mask_fill)
        variants[f"mask_{idx}_{camera}_{x}_{y}_{width}_{height}"] = masked

    for idx, (camera, x, y, width, height, dx, dy) in enumerate(args.move):
        if camera not in cfg.camera_names:
            raise ValueError(f"move requested camera {camera!r}, but checkpoint cameras are {cfg.camera_names}")
        moved = {k: v.copy() for k, v in base_images.items()}
        moved[camera] = move_rect(
            moved[camera],
            (x, y, width, height),
            dx,
            dy,
            fill_value(camera, stats, args.mask_fill),
        )
        variants[f"move_{idx}_{camera}_{x}_{y}_{width}_{height}_{dx}_{dy}"] = moved

    model = ACTPolicy(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    predictions = {}
    for name, images in variants.items():
        pred_norm, pred = predict_chunk(model, images, qpos_normalized, cfg, a_mean, a_std, device)
        predictions[name] = {"pred_norm": pred_norm, "pred": pred}

    base_pred = predictions["original"]["pred"]
    base_norm = predictions["original"]["pred_norm"]
    results = [
        summarize_variant(name, value["pred"], value["pred_norm"], base_pred, base_norm, args.first_n)
        for name, value in predictions.items()
    ]

    # Add qpos-only variants after image variants so their outputs are easy to compare.
    qpos_variants = {
        "qpos_dataset_mean": torch.zeros_like(qpos_normalized),
    }
    for name, qpos_variant in qpos_variants.items():
        pred_norm, pred = predict_chunk(model, base_images, qpos_variant, cfg, a_mean, a_std, device)
        results.append(summarize_variant(name, pred, pred_norm, base_pred, base_norm, args.first_n))

    output = {
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "revision": args.revision,
        "dataset_index": dataset_index,
        "episode": int(sample["episode_index"]),
        "frame_index": int(sample["frame_index"]),
        "camera_names": list(cfg.camera_names),
        "qpos": qpos.tolist(),
        "original_predicted_actions": base_pred[: args.first_n].tolist(),
        "variants": sorted(results, key=lambda item: item["chunk_mean_l2"], reverse=True),
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, indent=2) + "\n")
    save_contact_sheet(Path(args.output_image), variants, cfg.camera_names)

    print(f"Wrote {output_json}")
    print(f"Wrote {args.output_image}")
    print("Top counterfactual changes:")
    for item in output["variants"][:12]:
        print(
            f"{item['name']:32s} first_l2={item['first_action_l2']:.3f} "
            f"chunk_mean_l2={item['chunk_mean_l2']:.3f} norm_chunk_mean_l2={item['normalized_chunk_mean_l2']:.3f}"
        )


if __name__ == "__main__":
    main()
