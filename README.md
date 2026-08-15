# act-sim

Policy code for an OpenMANIPULATOR-X (OMX) arm learning **"place the sponge on
the hot pad"**: model, training, offline analysis, and evaluation.

This repo holds everything that runs on a GPU or against a dataset. The robot
itself — teleop, cameras, data collection — lives in the companion repo
`omx-training`, which loads this one at runtime via `--policy-code-root`.

The research question is how best to improve a behaviour-cloned policy from its
own failures. Three methods are compared under a matched data budget:

| method | what it adds | live success |
|---|---|---|
| custom ACT | 48 demonstrations | 4/22 = 18% |
| + DAgger | 18 human correction episodes | 9/15 = 60% |
| + RECAP | advantage-conditioned training on rollouts + corrections | see caveat below |

**Read `docs/sponge_hot_pad_experiments.md` first** — it is the log of record,
with every measured number and the caveats attached to each. Most importantly,
the RECAP and DAgger evaluations used non-overlapping sponge placements and are
not yet comparable.

## Layout

```
models/act/          the custom ACT policy
  act.py             backbone + CVAE encoder + transformer encoder/decoder
  train.py           training loop
  helper.py          dataloaders and the seeded train/holdout episode split
  lerobot_policy.py  adapter exposing the custom model as a LeRobot policy
  eval*.py           evaluation entrypoints (see "Evaluation" below)
  camera_alignment.py, scene_check.py   pre-rollout scene validation

models/recap/        RECAP (arXiv:2511.14759) on top of that policy
  act_recap.py       ACT plus a three-state advantage-conditioning token
  act_critic.py      observation-only distributional value model, 201 bins
  train_critic.py    critic training
  label_advantage.py N-step advantage labelling and thresholding
  train_policy.py    advantage-conditioned policy training
  check_token_sensitivity.py  asserts a trained policy uses its token
  build_dataset.py   composes demos + rollouts + corrections into a dataset

dataset_tools/       dataset surgery: episode splits, no-op trimming,
                     DAgger aggregation. Source, despite the name of the
                     data/ directory it used to live in.
configs/act/base.py  ACTConfig — the single source of hyperparameters
deploy/              Modal and vast.ai launchers, plus run watchdogs
evaluation/          one-off analysis scripts (dataset audits, probes)
tests/               pytest suite
docs/                experiment logs and write-ups
```

## Training

All current training runs on Modal:

```bash
# base ACT from demonstrations
modal run --detach deploy/train_custom_act_modal.py \
  --dataset-repo-id RevanthGundala/026-place-sponge-on-hot-pad-clean \
  --job-name act_custom_clean48_chunk30_20k --steps 20000

# fine-tune from a checkpoint (DAgger)
modal run --detach deploy/train_custom_act_modal.py \
  --dataset-repo-id RevanthGundala/028-custom48-dagger18 \
  --pretrained-checkpoint act_experiments/<run>/checkpoints/last.pt \
  --job-name act_custom_clean48_dagger18_ft10k --steps 10000

# RECAP: critic, then labels, then the advantage-conditioned policy
modal run --detach deploy/train_recap_critic_modal.py
python -m models.recap.label_advantage --help
modal run --detach deploy/train_custom_act_modal.py::recap_policy \
  --pretrained-checkpoint act_experiments/<run>/checkpoints/last.pt \
  --dataset-repo-id RevanthGundala/034-recap-balanced \
  --labels-path act_experiments/recap_labels/034-recap-balanced.json
```

Architecture and optimizer settings are inherited from the warm-start
checkpoint, so those commands only carry run identity and budget.

**Any paid GPU run needs an independent watchdog** — see
`deploy/modal_call_watchdog.py`. A `finally` block in the launcher is not
enough, because it dies with the launcher.

## Evaluation

There are several evaluation entrypoints because they target different
runtimes; they are not duplicates:

All four drive the real arm. They differ on two axes — how actions are
clocked, and which checkpoint format they can load:

| script | clocking | checkpoint |
|---|---|---|
| `eval.py` | synchronous control loop | custom `.pt` |
| `eval_async_custom.py` | async, replays a full action chunk | custom `.pt` |
| `eval_async.py` | async, via LeRobot `PolicyServer`/`RobotClient` over gRPC | LeRobot `pretrained_model/` |
| `eval_lerobot.py` | synchronous | stock LeRobot policy |

The async pair exists because inference on MPS runs at ~10 Hz while the policy
was trained at 30 Hz; replaying a chunk at the trained rate removes the
resulting jerk. `eval_async_custom.py` is separate from `eval_async.py` because
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

It exits non-zero if the token does not move the actions. It currently fails on
every policy we have trained; `docs/sponge_hot_pad_experiments.md` §5.9 explains
why, and why we believe that is a property of ACT rather than a bug.

Use at least ~150 batches. The default of 16 reads 128 sequential frames, which
is one episode and not a representative sample.

## Tests

```bash
PYTHONPATH=. python -m pytest tests/ -q
```

Two failures in `tests/test_act_eval.py` (wrist-flex seating) are known and
pre-existing.

## Gotchas

- **This repo and `omx-training` share top-level package names** (`configs`,
  `deploy`, `docs`, `evaluation`, `models`, `tests`, `utils`). Whichever is
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
| `docs/sponge_hot_pad_experiments.md` | the log of record for every experiment |
| `docs/act_debugging_learnings.md` | why the original ACT failed, and the ALOHA sim comparison that preceded this work |
| `docs/recap_research_question.md` | why RECAP was chosen for an ACT policy |
| `docs/blog.md` | narrative write-up in progress |
