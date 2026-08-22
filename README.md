# act-sim

Policy code for an OpenMANIPULATOR-X (OMX) arm learning **"place the sponge on
the hot pad"**: model, training, offline analysis, and evaluation.

This repo holds everything that runs on a GPU or against a dataset. The robot
itself — teleop, cameras, data collection — lives in the companion repo
`omx-training`, which loads this one at runtime via `--policy-code-root`.

The research question is how best to improve a behaviour-cloned policy from its
own failures. Short version of what we actually measured:

| method | what it adds | live success | grid |
|---|---|---|---|
| custom ACT | 48 demonstrations | 4/22 = 18% | unmatched |
| + DAgger | 18 human correction episodes | 9/15 = 60% | unmatched |
| RECAP-ACT parent | mixed labels on 66 eps, token unused | 8/15 = 53% | 5×3 tape |
| + 7 SPACE takeovers | extra corrections, still unused token | 12/15 = 80% | same 5×3 |
| + 4 autos + 3 SPACE | cloned on-policy fails | 3/15 = 20% | same 5×3 |

**The advantage token never fired.** Held-out L1 moves ≤ 0.07% when the bit
is flipped. Every "RECAP" live number is behaviour cloning with a sticker.
DAgger 9/15 and RECAP 8/15 are **not** a method comparison (different sponge
placements, 18 vs 7 corrections). Do not quote the withdrawn 82% SFT number.

**Read `docs/sponge_hot_pad_experiments.md` first** — it is the log of record.
The narrative write-up is `docs/blog.md`.

## Layout

```
models/act/          the custom ACT policy
  act.py             backbone + CVAE encoder + transformer encoder/decoder
  train.py           training loop
  helper.py          dataloaders
  episode_split.py   the seeded train/holdout episode split
  lerobot_policy.py  adapter exposing the custom model as a LeRobot policy;
                     omx-training loads this by exact path, so do not move it
  eval/              live-robot evaluation (see "Evaluation" below), plus
                     pre-rollout scene validation and rollout capture
  tools/             compare_policies: policy-vs-policy end-effector diagnostics

models/recap/        RECAP (arXiv:2511.14759) on top of that policy
  act_recap.py       ACT plus a three-state advantage-conditioning token
  act_recap_masked.py  variant that can omit the token entirely
  act_critic.py      observation-only distributional value model, 201 bins
  train_critic.py    critic training
  label_advantage.py N-step advantage labelling and thresholding
  train_policy.py    advantage-conditioned policy training
  build_dataset.py   composes demos + rollouts + corrections into a dataset
  build_gen_dataset.py  mix A/B builders (corrections vs rollouts+corrections)
  check_token_sensitivity.py  action displacement when the token flips
  check_token_loss.py  held-out L1 when the token is ablated
  check_action_label_mi.py  I(a ; I | o) via critic-space neighbours
  probe_advantage_from_obs.py  can I be read from o alone?

scripts/             command-line tools, none imported by the library code
  dataset/           dataset surgery
    shortcut_audit.py  does this dataset require object-conditioned actions?
    clone_split.py     copy selected episodes into a new dataset
    export_video.py    episode stills and MP4s, for figures
    lerobot_schema.py  merge-compatibility rules for combining datasets
    dagger/            aggregate demos with corrections, and validate the result
    trimming/          drop the near-static settle frames at episode edges
  set_hf_token.py, set_vast_key.py   credential setup

configs/act/base.py  ACTConfig — the single source of hyperparameters
train/               launchers for paid compute, plus their watchdogs
  modal/             Modal: custom ACT, RECAP critic, RECAP policy
  vastai/            vast.ai: launcher, resume, instance watchdog, image
tests/               pytest suite
docs/                experiment logs and write-ups
```

## Training

All current training runs on Modal:

```bash
# base ACT from demonstrations
modal run --detach train/modal/custom_act.py \
  --dataset-repo-id RevanthGundala/026-place-sponge-on-hot-pad-clean \
  --job-name act_custom_clean48_chunk30_20k --steps 20000

# fine-tune from a checkpoint (DAgger)
modal run --detach train/modal/custom_act.py \
  --dataset-repo-id RevanthGundala/028-custom48-dagger18 \
  --pretrained-checkpoint act_experiments/<run>/checkpoints/last.pt \
  --job-name act_custom_clean48_dagger18_ft10k --steps 10000

# RECAP: critic, then labels, then the advantage-conditioned policy
modal run --detach train/modal/recap_critic.py
python -m models.recap.label_advantage --help
modal run --detach train/modal/custom_act.py::recap_policy \
  --pretrained-checkpoint act_experiments/<run>/checkpoints/last.pt \
  --dataset-repo-id RevanthGundala/034-recap-balanced \
  --labels-path act_experiments/recap_labels/034-recap-balanced.json
```

Architecture and optimizer settings are inherited from the warm-start
checkpoint, so those commands only carry run identity and budget.

**Any paid GPU run needs an independent watchdog** — see
`train/modal/call_watchdog.py`. A `finally` block in the launcher is not
enough, because it dies with the launcher.

## Evaluation

There are several evaluation entrypoints because they target different
runtimes; they are not duplicates:

All four drive the real arm. They differ on two axes — how actions are
clocked, and which checkpoint format they can load:

| script | clocking | checkpoint |
|---|---|---|
| `eval/local.py` | synchronous control loop | custom `.pt` |
| `eval/async_custom.py` | async, replays a full action chunk | custom `.pt` |
| `eval/async_lerobot.py` | async, via LeRobot `PolicyServer`/`RobotClient` over gRPC | LeRobot `pretrained_model/` |
| `eval/stock_lerobot.py` | synchronous | stock LeRobot policy |

The async pair exists because inference on MPS runs at ~10 Hz while the policy
was trained at 30 Hz; replaying a chunk at the trained rate removes the
resulting jerk. `async_custom.py` is separate from `async_lerobot.py` because
LeRobot's `PolicyServer` resolves policy classes through its own hardcoded
factory and cannot load a custom checkpoint.

After training a RECAP policy, verify it actually uses its conditioning before
spending robot time:

```bash
python -m models.recap.check_token_sensitivity \
  --checkpoint <checkpoint>.pt \
  --dataset-repo-id RevanthGundala/034-recap-balanced \
  --labels <labels>.json --episodes 48-65 --batches 150
```

It exits non-zero if the token does not move the actions. It fails on every
*real-label* policy we have trained. That is not "ACT cannot condition": a
planted +0.5 action shift is recovered at 0.519 (§5.10). The real labels do
not change the optimal action given the observation (`I(a ; I | o)` ≈ 2%,
held-out L1 flip ≤ 0.07%). See §5.9 and §5.15.

For a local LeRobot dataset that is not on Hugging Face:

```bash
RECAP_POLICY_DATASET_ROOT=/path/to/036-recap-corr7 \
  python -m models.recap.train_policy
```

Use at least ~150 batches. The default of 16 reads 128 sequential frames, which
is one episode and not a representative sample.

## Tests

```bash
pytest              # full suite
pytest tests/recap  # just the RECAP suite
```

Tests are grouped by the code they cover:

| folder | covers |
|---|---|
| `tests/act/` | the ACT policy, eval, and training |
| `tests/recap/` | critic, advantage labelling, conditioned policy |
| `tests/dataset/` | dataset surgery and DAgger aggregation |
| `tests/train/` | Modal and vast.ai launchers |

Use the `repo_root` fixture rather than counting parent directories from
`__file__`; test files are one level deeper than they look.

## Gotchas

- **This repo and `omx-training` share top-level package names** (`configs`,
  `docs`, `models`, `tests`, `utils`). Whichever is
  first on `sys.path` wins, and it has silently resolved `utils.config` to the
  wrong repo's copy before. Pin the repo root first in entry scripts; never put
  act-sim on `PYTHONPATH` for an omx-training run.
- **Wrist-flex has directional stiction**: it tracks to ~0.11 units positive but
  stalls 0.9–1.6 units short going negative. Always approach a target from
  below. Never raise motor current or PID gains to compensate.
- **`torchcodec` does not load in these environments** (missing `libavutil.56`),
  so dataset video reading goes through OpenCV — see
  `models/recap/lerobot_source.py`.
- **Validate on held-out episodes, never held-out frames.** Consecutive frames
  are near-identical and leak straight across the split.

## Docs

| file | contents |
|---|---|
| `docs/sponge_hot_pad_experiments.md` | log of record — every measured number |
| `docs/blog.md` | narrative: the advantage token never fired |
| `docs/act_debugging_learnings.md` | ACT debugging notebook, then the RECAP mechanism kills |
| `docs/recap_research_question.md` | why RECAP was chosen; current status at the top |
