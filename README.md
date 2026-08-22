# act-sim

Custom [ACT](https://arxiv.org/abs/2304.13705) for an OpenMANIPULATOR-X arm
learning **place the sponge on a fixed hot pad**. This repo is the policy,
training, and eval. Teleop and data collection live in
`omx-training` (`--policy-code-root` points here).

We then asked whether [RECAP](https://arxiv.org/abs/2511.14759) advantage
conditioning helps a small deterministic ACT. **The token never fired.** Live
"RECAP" numbers below are behaviour cloning with an unused sticker.

## Results

| method | extra data | live | grid |
|---|---|---|---|
| custom ACT | 48 demos | 4/22 = 18% | unmatched |
| + DAgger | 18 expert corrections | 9/15 = 60% | unmatched |
| RECAP-ACT parent | mixed labels, unused token | 8/15 = 53% | 5×3 tape |
| + 7 SPACE takeovers | extra corrections | 12/15 = 80% | same 5×3 |
| + 4 autos + 3 SPACE | cloned on-policy fails | 3/15 = 20% | same 5×3 |

Held-out L1 moves ≤ 0.07% when the advantage bit is flipped. Do not quote the
withdrawn 82% SFT number. DAgger 9/15 vs RECAP 8/15 is **not** a method
comparison (different placements, 18 vs 7 corrections).

## Install

Python 3.12+, [uv](https://docs.astral.sh/uv/). macOS (MPS eval) or Linux (CUDA train).

```bash
git clone https://github.com/RevanthGundala/act-sim.git
cd act-sim
uv sync --group dev
uv run pytest -q
```

`lerobot` is pulled from git. `torchcodec` does not load here (missing
`libavutil.56`); dataset video goes through OpenCV
(`models/recap/lerobot_source.py`).

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
  --action-space absolute \
  --recap
```

**Raw `.pt`** (no LeRobot server):

```bash
uv run python models/act/eval/async_custom.py \
  --checkpoint /path/to/last.pt \
  --recap-advantage pos
```

| script | clocking | loads |
|---|---|---|
| `models/act/eval/async_lerobot.py` | async chunk replay | `pretrained_model/` |
| `models/act/eval/async_custom.py` | async chunk replay | custom `.pt` |
| `models/act/eval/local.py` | sync loop | custom `.pt` |
| `models/act/eval/stock_lerobot.py` | sync loop | stock LeRobot ACT |

Wrist-flex has directional stiction. Always approach the start pose from
below. Never raise motor current or PID gains. Use
`--start-state-wrist-flex-offset 0` on the 034/036/037 start range.

## No robot

```bash
uv run pytest -q

# does this checkpoint actually use its advantage token?
uv run python -m utils.analysis.token_sensitivity \
  --checkpoint /path/to/step_020000.pt \
  --dataset-repo-id RevanthGundala/034-recap-balanced \
  --labels /path/to/advantage_labels.json \
  --batches 150

# held-out L1 when the token is flipped / dropped
uv run python -m utils.analysis.token_loss \
  --checkpoint /path/to/step_020000.pt \
  --dataset-repo-id RevanthGundala/034-recap-balanced \
  --labels /path/to/advantage_labels.json \
  --train-split 0.8
```

Public datasets: `RevanthGundala/026-place-sponge-on-hot-pad-clean` (48 demos),
`RevanthGundala/028-custom48-dagger18`, `RevanthGundala/034-recap-balanced`.

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

# RECAP: critic → labels → policy
uv run modal run --detach train/modal/recap_critic.py
uv run python -m train.recap.label_advantage --help
uv run modal run --detach train/modal/custom_act.py::recap_policy \
  --pretrained-checkpoint act_experiments/<run>/checkpoints/last.pt \
  --dataset-repo-id RevanthGundala/034-recap-balanced \
  --labels-path act_experiments/recap_labels/034-recap-balanced.json
```

Local dataset that is not on Hugging Face:

```bash
RECAP_POLICY_DATASET_ROOT=/path/to/036-recap-corr7 \
  uv run python -m train.recap.policy
```

Architecture and optimizer come from the warm-start checkpoint.

## Layout

```
models/act/          custom ACT (do not move lerobot_policy.py — omx-training
                     loads it by exact path)
  eval/              live-robot evaluation
models/recap/
  policy/            live architecture: 3-state token + CVAE
  masked/            token can be omitted
  critic/            V(o) only — no CVAE
scripts/dataset/     dataset surgery, DAgger aggregate
scripts/recap/       build RECAP mixes from demos / rollouts / corrections
train/act/           ACT training loops
train/recap/         critic, labeling, RECAP policy training
train/modal/         Modal launchers + watchdog
train/vastai/        vast.ai launchers
utils/config.py      OMX ports / cameras
utils/analysis/      token checks (sensitivity, L1, I(a;I|o))
configs/act/base.py  hyperparameters
tests/               pytest, grouped like the code
```

This repo and `omx-training` both expose `configs`, `models`, `utils`. Pin
this repo first on `sys.path` in entry scripts. Never put act-sim on
`PYTHONPATH` for an omx-training run.

## License

MIT. See `LICENSE`.
