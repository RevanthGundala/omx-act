# omx-act

Custom [ACT](https://arxiv.org/abs/2304.13705) for an OpenMANIPULATOR-X arm
learning **place the sponge on a fixed hot pad**. This repo is the policy,
training, and eval. Teleop and data collection live in
`omx-training` (`--policy-code-root` points here).

The ACT model and its training loop are custom (`train/act/custom.py`). We use
[LeRobot](https://github.com/huggingface/lerobot) for dataset I/O
(`LeRobotDataset`), image/action pre- and post-processors, the OMX
follower and OpenCV camera drivers, and async eval
(`PolicyServer` / `RobotClient`). Stock LeRobot ACT — trained with
LeRobot's own loop — is the comparison baseline.
`models/act/lerobot_policy.py` wraps our weights as a LeRobot policy so
eval can go through that stack.

## Results

Controlled physical evaluation on 30 matched sponge placements (5 layouts × 6 trials):

- **Base ACT** — 48 demonstrations. **15/30 = 50%** (95% CI: 33–67%)
- **DAgger (Round 1)** — +18 human corrections. **19/30 = 63%** (95% CI: 46–78%)
- **DAgger (Round 2)** — +12 targeted corrections on failed layouts. **20/30 = 67%** (95% CI: 49–81%)

All 36 failures were pick-up failures (sponge never moved or nudged slightly); targeted corrections eliminated misses on the specific layouts they targeted.

## Install

Python 3.12+, [uv](https://docs.astral.sh/uv/). macOS (MPS eval) or Linux (CUDA train).

```bash
git clone https://github.com/RevanthGundala/omx-act.git
cd omx-act
uv sync --group dev
uv run pytest -q
```

`lerobot` is pulled from git. `torchcodec` does not load here (missing
`libavutil.56`); dataset video goes through OpenCV.

Hardware for live eval: OMX follower, two USB cameras (`top`, `wrist`),
serial port in `utils/config.py`. List cameras with:

```bash
uv run python -c "from lerobot.cameras.opencv.camera_opencv import OpenCVCamera; print(OpenCVCamera.find_cameras())"
```

## Run a trained policy

Needs the arm. Recommended path is async eval: inference is ~10 Hz on MPS,
training was 30 Hz, so we replay a 30-step chunk and blend overlaps.

**LeRobot export** (`pretrained_model/` directory):

```bash
uv run python models/act/eval/async_lerobot.py \
  --policy-path /path/to/pretrained_model \
  --policy-type custom_act \
  --task "Place the sponge on the hot pad" \
  --cameras top,wrist \
  --fps 30 \
  --actions-per-chunk 30 \
  --policy-device mps \
  --start-state-wrist-flex-offset 0 \
  --trials 15
```

Export a `.pt` checkpoint first if you only have the training file:

```bash
uv run python -m models.act.lerobot_policy \
  --checkpoint /path/to/last.pt \
  --output-dir /path/to/pretrained_model \
  --action-space absolute
```

**Raw `.pt`** (no LeRobot server):

```bash
uv run python models/act/eval/async_custom.py \
  --checkpoint /path/to/last.pt
```

| script | clocking | loads |
|---|---|---|
| `models/act/eval/async_lerobot.py` | async chunk replay | `pretrained_model/` |
| `models/act/eval/async_custom.py` | async chunk replay | custom `.pt` |
| `models/act/eval/local.py` | sync loop | custom `.pt` |
| `models/act/eval/stock_lerobot.py` | sync loop | stock LeRobot ACT |

Wrist-flex has directional stiction. Always approach the start pose from
below. Never raise motor current or PID gains.

## Train

Paid GPU jobs go through Modal. An independent watchdog is required
(`train/modal/call_watchdog.py`). A `finally` in the launcher is not enough.

```bash
# base ACT
uv run modal run --detach train/modal/custom_act.py \
  --dataset-repo-id RevanthGundala/026-place-sponge-on-hot-pad-clean \
  --job-name act_custom_clean48_chunk30_20k --steps 20000

# DAgger fine-tune
uv run modal run --detach train/modal/custom_act.py \
  --dataset-repo-id RevanthGundala/028-custom48-dagger18 \
  --pretrained-checkpoint act_experiments/<run>/checkpoints/last.pt \
  --job-name act_custom_clean48_dagger18_ft10k --steps 10000
```

## Layout

```
models/act/          custom ACT (do not move lerobot_policy.py — omx-training
                     loads it by exact path)
  eval/              live-robot evaluation
scripts/dataset/     dataset surgery, DAgger aggregate
train/act/           ACT training loops
train/modal/         Modal launchers + watchdog
train/vastai/        vast.ai launchers
utils/config.py      OMX ports / cameras
configs/act/base.py  hyperparameters
tests/               pytest, grouped like the code
```

This repo and `omx-training` both expose `configs`, `models`, `utils`. Pin
this repo first on `sys.path` in entry scripts. Never put this repo on
`PYTHONPATH` for an omx-training run.

## License

MIT. See `LICENSE`.
