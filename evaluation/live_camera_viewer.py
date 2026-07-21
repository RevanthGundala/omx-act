"""Live OpenCV viewer for configured OMX cameras.

Shows top/wrist camera feeds continuously so the rig can be aligned before
recording or eval. Press q or Esc to quit, s to save a snapshot.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config import CAMERA_HEIGHT, CAMERA_WIDTH, CAMERAS, FPS


def parse_camera_names(value: str) -> list[str]:
    if value == "all":
        return list(CAMERAS)
    names = [name.strip() for name in value.split(",") if name.strip()]
    unknown = sorted(set(names) - set(CAMERAS))
    if unknown:
        raise ValueError(f"unknown cameras {unknown}; configured cameras={list(CAMERAS)}")
    return names


def draw_overlay(frame: np.ndarray, label: str, *, grid: bool, crosshair: bool) -> np.ndarray:
    out = frame.copy()
    height, width = out.shape[:2]
    if grid:
        for x in (width // 3, 2 * width // 3):
            cv2.line(out, (x, 0), (x, height), (60, 220, 60), 1)
        for y in (height // 3, 2 * height // 3):
            cv2.line(out, (0, y), (width, y), (60, 220, 60), 1)
    if crosshair:
        cv2.line(out, (width // 2, 0), (width // 2, height), (0, 255, 255), 1)
        cv2.line(out, (0, height // 2), (width, height // 2), (0, 255, 255), 1)
    cv2.rectangle(out, (0, 0), (260, 34), (0, 0, 0), -1)
    cv2.putText(out, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Show configured top/wrist camera feeds.")
    parser.add_argument("--cameras", default="top,wrist", help="Comma-separated camera names or 'all'.")
    parser.add_argument("--width", type=int, default=CAMERA_WIDTH)
    parser.add_argument("--height", type=int, default=CAMERA_HEIGHT)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--no-grid", action="store_true")
    parser.add_argument("--no-crosshair", action="store_true")
    parser.add_argument("--snapshot-dir", type=Path, default=Path("outputs/camera_snapshots"))
    args = parser.parse_args()

    camera_names = parse_camera_names(args.cameras)
    captures = {}
    try:
        for name in camera_names:
            cap = cv2.VideoCapture(CAMERAS[name])
            if not cap.isOpened():
                raise RuntimeError(f"failed to open camera {name} index={CAMERAS[name]}")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
            cap.set(cv2.CAP_PROP_FPS, args.fps)
            captures[name] = cap

        print("Press q/Esc to quit, s to save snapshots.")
        while True:
            frames = []
            for name, cap in captures.items():
                ok, frame = cap.read()
                if not ok:
                    frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)
                    cv2.putText(frame, f"{name}: read failed", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                frame = cv2.resize(frame, (args.width, args.height))
                frames.append(draw_overlay(frame, name, grid=not args.no_grid, crosshair=not args.no_crosshair))

            canvas = np.hstack(frames) if len(frames) > 1 else frames[0]
            cv2.imshow("act-sim cameras", canvas)
            key = cv2.waitKey(max(1, int(1000 / max(args.fps, 1)))) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                args.snapshot_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                path = args.snapshot_dir / f"cameras_{timestamp}.jpg"
                cv2.imwrite(str(path), canvas)
                print(f"saved {path}")
    finally:
        for cap in captures.values():
            cap.release()
        cv2.destroyAllWindows()
        time.sleep(0.1)


if __name__ == "__main__":
    main()
