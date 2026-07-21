"""Build a montage of top-camera start frames whose sponge/bowl positions fall in a given
in-distribution bucket (as classified by scene_check.xy_bucket / the 011 position audit).

Filters outputs/dataset_audits/011_position_detections.json (per-episode sponge_bucket /
bowl_bucket, produced by the 011 position audit -- see scene_check.py's load_coverage) and
grabs the matching outputs/dataset_audits/011_start_frames/episode_XXXX_top_start.jpg frames,
tiling them into a single labeled grid image. Useful for eyeballing "what does an in-distribution
'left-mid sponge x right-mid bowl' start actually look like" instead of reasoning about buckets
in the abstract.

Buckets are 3x3 grid cells: {left,center,right}-{near,mid,far} (see scene_check.xy_bucket).
Omit --sponge-bucket/--bowl-bucket to match any value for that object.

Example:
    uv run python -m models.act.build_bucket_montage --sponge-bucket left-mid --bowl-bucket right-mid
    uv run python -m models.act.build_bucket_montage --bowl-bucket right-mid --output outputs/dataset_audits/bowl_right_mid_montage.jpg
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COVERAGE_JSON = REPO_ROOT / "outputs/dataset_audits/011_position_detections.json"
DEFAULT_FRAMES_DIR = REPO_ROOT / "outputs/dataset_audits/011_start_frames"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/dataset_audits/bucket_montage.jpg"

TILE_SIZE = 220
LABEL_H = 22


def load_matches(
    coverage_json: Path,
    sponge_bucket: str | None,
    bowl_bucket: str | None,
) -> list[dict]:
    data = json.loads(coverage_json.read_text())
    matches = []
    for row in data:
        if sponge_bucket is not None and row.get("sponge_bucket") != sponge_bucket:
            continue
        if bowl_bucket is not None and row.get("bowl_bucket") != bowl_bucket:
            continue
        matches.append(row)
    return matches


def build_montage(matches: list[dict], frames_dir: Path, cols: int = 6) -> Image.Image:
    if not matches:
        raise ValueError("No episodes matched the requested bucket(s).")

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    rows = (len(matches) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * TILE_SIZE, rows * (TILE_SIZE + LABEL_H)), "white")
    draw = ImageDraw.Draw(canvas)

    for i, row in enumerate(matches):
        r, c = divmod(i, cols)
        frame_path = frames_dir / f"{row['episode']}.jpg"
        if not frame_path.exists():
            frame_path = frames_dir / f"{row['episode']}.png"
        x0, y0 = c * TILE_SIZE, r * (TILE_SIZE + LABEL_H)
        if frame_path.exists():
            img = Image.open(frame_path).convert("RGB").resize((TILE_SIZE, TILE_SIZE))
            canvas.paste(img, (x0, y0))
        else:
            draw.rectangle([x0, y0, x0 + TILE_SIZE, y0 + TILE_SIZE], outline="red", width=2)
            draw.text((x0 + 8, y0 + TILE_SIZE // 2), "missing frame", fill="red", font=font)
        label = f"{row['episode']}  s={row.get('sponge_bucket')} b={row.get('bowl_bucket')}"
        draw.text((x0 + 4, y0 + TILE_SIZE + 4), label, fill="black", font=font)

    return canvas


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sponge-bucket", default=None,
                   help="e.g. left-mid, center-near, right-far (omit to match any)")
    p.add_argument("--bowl-bucket", default=None,
                   help="e.g. left-mid, center-near, right-far (omit to match any)")
    p.add_argument("--coverage-json", type=Path, default=DEFAULT_COVERAGE_JSON)
    p.add_argument("--frames-dir", type=Path, default=DEFAULT_FRAMES_DIR)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--cols", type=int, default=6)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.sponge_bucket is None and args.bowl_bucket is None:
        raise SystemExit("Specify at least one of --sponge-bucket / --bowl-bucket.")

    matches = load_matches(args.coverage_json, args.sponge_bucket, args.bowl_bucket)
    print(f"Matched {len(matches)} episode(s) for "
          f"sponge_bucket={args.sponge_bucket!r} bowl_bucket={args.bowl_bucket!r}")
    for row in matches:
        print(f"  {row['episode']}  sponge={row.get('sponge_bucket')} bowl={row.get('bowl_bucket')}")

    montage = build_montage(matches, args.frames_dir, cols=args.cols)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    montage.save(args.output)
    print(f"Saved montage -> {args.output}")


if __name__ == "__main__":
    main()
