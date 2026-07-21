# Debugging the custom ACT — learnings

Record of the investigation into why the custom `models/act` ACT policy was
training poorly, what we changed, and (importantly) what we did and did not
actually measure.

## TL;DR

- The custom ACT was failing: a hard ~0.08 normalized-L1 floor on a
  single-episode overfit, with a weak/ignored latent.
- We ran the **genuine** tonyzhaozh/act DETRVAE on our **exact** dataloader and
  objective (vendored under `models/act_official/`) as a decisive A/B and fixed
  the custom model piece by piece. On the multi-episode `sim_transfer_cube_reference`
  benchmark its **held-out validation L1 matched** the official implementation
  (~0.196 vs ~0.220).
- **BUT the sim rollout overturns the "we're done" framing.** On the actual
  MuJoCo task, **both** models are catastrophically below the paper's ~90%:
  patched custom **4%** success, faithful official ACT **0%** (50 rollouts each).
  Since even the genuine official implementation fails here, **the bottleneck is
  our recipe/harness, not the custom architecture.** Val L1 ~0.2 is itself an
  underfitting symptom (both policies mostly hold pose; ~14% cube touch).
- Net: the architecture fixes were real and necessary (latent now works, custom
  == official on val L1), but a working policy still requires fixing the
  training/eval **recipe** to reproduce the paper.

## Fixes applied to `models/act` (in order)

1. **Main transformer encoder** (`act.py`). The decoder cross-attended to raw
   projected tokens; there was no DETR/ACT-style encoder. Added a self-attention
   stack over the `[z, joint, image]` memory before the decoder. Highest-impact
   structural fix.
2. **Reparameterization sampling** (`act.py`). Latent uses `z ~ N(mu, sigma)`
   (not `z = mu`).
3. **ACT-style KL** (`train.py`). KL is now sum-over-latent-dims, mean-over-batch
   (matches official). The previous mean-over-all-dims form was ~`d_z`x weaker and
   let the posterior collapse.
4. **Attention mask inversion fix** (`attention.py`). `future_mask` is built with
   `1 = keep / 0 = pad`, but `masked_fill` used `== 1`, masking the **valid**
   tokens (and collapsing to uniform attention when there was no padding). Fixed
   to `== 0`.
5. **2-D sine positional embedding** for image tokens (`act.py`), replacing the
   1-D flatten-index table. Row/col encoded on separate channel halves.
6. **Per-layer positional injection** (`attention.py`, `act.py`). Positions are
   added to q/k at **every** encoder/decoder attention layer (`with_pos_embed`
   style), and the decoder cross-attention adds pos to memory keys + `query_pos`
   to queries, instead of baking position into the residual once. Split
   `cross_kv_proj` into separate `cross_k_proj`/`cross_v_proj` so the value stays
   position-free; decoder starts from zeros with `q_emb` as the learned query pos.
7. **FrozenBatchNorm2d backbone** (`act.py`), like official ACT, instead of
   trainable BatchNorm (unstable at small batch sizes).
8. **ImageNet image normalization** (`train.py`). Images are normalized with
   ImageNet mean/std before the pretrained ResNet (previously raw `[0,1]`).
   Opt-out via `normalize_images=` and disabled for the official harness, whose
   `ACTPolicy` normalizes internally (avoids double-normalizing).

## Decisive A/B harness

- `models/act_official/` is a faithful vendored copy of tonyzhaozh/act
  (detr_vae / transformer / backbone / position_encoding / policy), adapted only
  for: removed `IPython` debug hooks, torchvision `pretrained=` -> `weights=`,
  parametrized the ALOHA-hardcoded dim 14 to `state_dim`/`action_dim`, and a
  device-agnostic builder.
- `models/act_official/train_official.py` reuses our **exact** dataloader, norm
  stats, and L1+KL objective from `models/act/train.py` — only the model is
  swapped. This is what made the comparison trustworthy.
- Modal launchers: `deploy/train_custom_act_modal.py` and
  `deploy/train_official_act_modal.py` (mirrors of each other, same image and
  benchmark volume).

## Results (held-out validation L1, `sim_transfer_cube_reference`, 40 train / 10 val)

| step  | Custom (patched) | Official ACT |
|------:|-----------------:|-------------:|
| 1000  | 0.420            | 0.247        |
| 3000  | 0.219            | 0.237        |
| 5000  | 0.213            | 0.235        |
| 7000  | 0.196            | 0.231        |
| 10000 | **0.196**        | **0.220**    |

Final train L1 was essentially identical (custom 0.030 / official 0.0275).

## Key methodological lesson: overfit L1 was a red herring

On a **single-episode overfit**, the custom model was pinned at ~0.025–0.029
while official reached ~0.014 — and **none** of fixes 4–8 (mask, 2-D pos,
per-layer pos, FrozenBN, ImageNet norm) moved that number. A single fully-valid
episode is memorizable regardless of these details, so the overfit metric was
**blind** to the fixes. The **multi-episode held-out val L1** was the metric that
actually reflected the improvements (padding, spatial generalization, a working
latent). Evaluate ACT changes on multi-episode val, not single-episode overfit.

## Sim eval (actually run, locally) — the decisive result

We ran the MuJoCo rollout eval locally for **both** models, 50 rollouts each,
with identical sim plumbing + temporal-ensemble action selection:

- Patched custom: `models/act/eval.py`
- Official ACT:   `models/act_official/eval_official.py` (mirror; builds
  `ACTPolicy`, feeds raw `[0,1]` images since the policy ImageNet-normalizes
  internally)

| metric (`sim_transfer_cube`) | Patched custom | Official ACT | Paper target |
|---|---|---|---|
| success (reward 4) | **4%** (2/50) | **0%** (0/50) | **~90%** |
| lifted (reward >= 2) | 4% (2/50) | 4% (2/50) | — |
| touched (reward >= 1) | 14% (7/50) | 14% (7/50) | — |
| avg return | 34.0 | 3.6 | — |

(Paper/README: "success rate should be around 90% for transfer cube, ~50% for
insertion".)

### Interpretation (this overturns the "custom == official, done" framing)

- **Both models are catastrophically below ~90%, and the faithful official ACT
  gets 0%.** Since the genuine implementation also fails in our setup, the
  bottleneck is the **recipe/harness, not the custom architecture.**
- Val L1 ~0.2 parity (0.196 vs 0.220) does **not** mean the policy works — it is
  an **underfitting symptom**. Both policies essentially hold pose (~14% touch),
  whereas a converged ACT touches ~100% and reaches far lower L1.
- The architecture fixes were still real and necessary (latent works; custom
  matches official on val L1), but they were never the thing blocking a working
  policy.

### Eval is trustworthy (ruled out artifacts)

- Harness runs end-to-end locally (`mujoco`/`dm_control` in the venv; renders OK).
- **No dimension bug.** `action_spec=(16,)` is raw mujoco actuators; the task's
  `before_step` consumes a **14-dim** policy action. Our 14-dim actions are right.
- **No distribution mismatch.** Training HDF5 `top` image vs eval-rendered `top`
  image are identical (`(480,640,3)`, mean 39.9); reset qpos matches; actions are
  finite/in-range.

## Likely recipe culprits (shared by both models, since both fail)

- **Demos / training length vs the paper:** paper uses 50 demos / 2000 epochs; we
  used 40 train demos / 10k steps with val L1 plateaued ~0.2 (under-converged).
- **Eval protocol:** query frequency / temporal-aggregation vs official
  `imitate_episodes.py`.
- **Normalization:** `NORM_STD_MIN=1e-2` clamp can distort near-constant action
  dims.
- **Objective/data:** whether chunk sampling / loss lets the model collapse to
  "predict current pose".

## Real robot diagnosis: pretrained backbone matters (2026-06-16)

On the real OMX robot (`pour_water` task, 50 demos), the custom ACT was
"memorizing trajectories" — it ignored visual input and replayed a fixed motion
regardless of object positions.

### Root cause: randomly initialized ResNet18 backbone

The custom `train.py` called `ACTPolicy(cfg)` without `pretrained_backbone=True`,
so the ResNet18 visual encoder started from **random weights** instead of
ImageNet-pretrained weights.

**Evidence from causal ablation tests:**
- Shuffling all images, keeping qpos: first-action change ~5.63 L2
- Shuffling qpos, keeping images: first-action change ~55.74 L2
- **Ratio: qpos affects output ~10x more than images**

**Evidence from weight comparison:**
| Metric | Random init | ImageNet pretrained | Our trained model |
|--------|-------------|---------------------|-------------------|
| conv1 std | 0.025 | **0.130** | 0.025 |
| Distance to our model | **0.028** | 0.083 | — |

The trained model's backbone weights were 3x closer to random init than to
ImageNet pretrained — confirming it never had meaningful visual features.

### Fix applied

```python
# models/act/train.py line 44
model = ACTPolicy(cfg, pretrained_backbone=True).to(device)
```

### Retrained results (10k steps, RTX 4090, chunk_size=100)

| Metric | Random backbone (old) | Pretrained backbone (new) |
|--------|----------------------|---------------------------|
| Final loss | ~0.3 | **0.12** |
| Final action L1 | ~0.15 | **0.08** |
| Final val loss | ~0.2 | **0.10** |

Checkpoint: `outputs/vastai/instance-41278194/outputs/act_experiments/act_custom_train_chunk100_steps10000/20260617-042657/checkpoints/last.pt`

### Why this matters

With only ~50 demonstrations, a randomly initialized ResNet cannot learn useful
visual features. The model compensates by relying on qpos (proprioception),
which causes it to replay memorized trajectories. ImageNet pretraining provides
a strong visual prior that lets the model actually use camera input.

## Recent OMX/towel official LeRobot ACT results (2026-06-24/25)

We moved from the custom ACT codepath to stock `lerobot-train` / LeRobot
`pretrained_model` checkpoints for the towel task:

- Main dataset: `RevanthGundala/009-place-towel-in-bowl-10fps`
  - 25 episodes, 3298 frames, 10 fps.
- Single-episode clone with fresh stats:
  - `RevanthGundala/009-place-towel-in-bowl-10fps-ep1-only`
  - 1 episode, 146 frames, 10 fps.

### Checkpoints and losses

| run | dataset | key config | checkpoint | logged loss / notes |
|---|---|---|---|---|
| official TE 10k | full towel 10fps | `chunk_size=30`, `n_action_steps=1`, `temporal_ensemble_coeff=0.01`, `use_vae=true` | `outputs/modal_official_act_towel_10k/010000/pretrained_model` | ~0.120 combined loss |
| official TE 20k | full towel 10fps | same, resumed from 10k | `outputs/modal_official_act_towel_20k/020000/pretrained_model` | ~0.075-0.082 combined loss; lower loss but worse start-frame offline behavior |
| official no-TE n30 10k | full towel 10fps | `chunk_size=30`, `n_action_steps=30`, `temporal_ensemble_coeff=None` | `outputs/modal_official_act_towel_10k_no_te_n30/010000/pretrained_model` | ~0.120-0.127 combined loss; worse offline than TE 10k |
| ep1 overfit 7.5k | full towel dataset filtered with `--dataset.episodes=[1]` | TE, VAE | `outputs/modal_official_act_towel_ep1_overfit_7500/007500/pretrained_model` | saved before interrupted 10k run |
| ep1 overfit 10k | full towel dataset filtered with `--dataset.episodes=[1]` | TE, VAE | `outputs/modal_official_act_towel_ep1_overfit_10k/010000/pretrained_model` | ~0.102 combined loss |
| ep1-only fresh-stats 5k | true cloned one-episode dataset | TE, VAE | `outputs/modal_official_act_towel_ep1_fresh_stats_5k/005000/pretrained_model` | 0.243 combined loss; dominated by KL; inference-path normalized L1 ~0.084 |
| ep1-only fresh-stats no-VAE 5k | true cloned one-episode dataset | `use_vae=false`, `kl_weight=0.01` (ignored), TE | `outputs/modal_official_act_towel_ep1_novae_5k/005000/pretrained_model` | 0.050 pure L1, but worse raw offline than VAE |

Modal volume paths mirror these local output directories under:

```text
/act_lerobot_<dataset-slug>[_suffix]/<steps>/checkpoints/<step>/pretrained_model
```

Examples:

```text
/act_lerobot_009-place-towel-in-bowl-10fps/10000/checkpoints/010000/pretrained_model
/act_lerobot_009-place-towel-in-bowl-10fps/10000/checkpoints/020000/pretrained_model
/act_lerobot_009-place-towel-in-bowl-10fps-ep1-only_fresh-stats-5k/5000/checkpoints/005000/pretrained_model
/act_lerobot_009-place-towel-in-bowl-10fps-ep1-only_novae-kl001-5k/5000/checkpoints/005000/pretrained_model
```

### Offline eval metrics we used

The offline sanity checks use LeRobot's official policy/preprocessor/
postprocessor, but custom comparison code in
`evaluation/offline_lerobot_action_check.py`:

```python
action = postprocessor(policy.select_action(preprocessor(obs))).squeeze(0).cpu()
state = sample["observation.state"].cpu()
recorded_action = sample["action"].cpu()

pred_delta = action - state
demo_delta = recorded_action - state
pred_error = action - recorded_action
```

So the "delta" metrics compare both prediction and demo action to the same
recorded dataset state at the same frame. This is not model rollout and does not
compound observation drift.

| run | start / sampled offline result |
|---|---|
| official TE 10k | start-frame mean max `|pred-state|` 4.56; broad-sample mean 5.13 |
| official TE 20k | start-frame mean 6.41; broad-sample mean 5.38 |
| no-TE n30 10k | start-frame mean 6.38; broad-sample mean 5.53 |
| ep1 overfit 10k (full stats) | on episode 1 sampled frames: mean `|pred-state|` 4.78, mean `|pred-recorded|` 3.15, pred delta L2 6.68 vs demo delta L2 3.87 |
| ep1-only fresh-stats VAE 5k | on ep1-only sampled frames: mean `|pred-state|` 4.87, mean `|pred-recorded|` 3.57, pred delta L2 7.22 vs demo delta L2 3.87 |
| ep1-only fresh-stats no-VAE 5k | mean `|pred-state|` 5.61, mean `|pred-recorded|` 4.17, pred delta L2 7.84 vs demo delta L2 3.87 |

Takeaway: lower logged loss did not translate to better raw one-step control
precision. The TE 10k full-dataset checkpoint remained the best of the full
dataset official runs by the start-frame sanity metric, but it was still not
good enough for reliable live pickup.

### What we ruled out

- **Camera FPS / resolution mismatch:** stock rollout needed camera capture at
  30 fps because the OpenCV cameras refused 10 fps. Running cameras at 30 fps
  while the rollout loop runs at 10 Hz is fine. The policy was trained on
  `3x480x640` top/wrist images, and eval uses 640x480 model inputs.
- **Action queue / chunk index bug:** for TE ACT, `select_action()` has no
  `_action_queue`; first call after `policy.reset()` returns chunk index 0
  exactly. We compared `select_action()` to the direct full chunk and got
  distance 0.0 to `chunk[0]`.
- **Off-by-N action timing:** frame-0 predictions did not appear later in the
  recorded trajectory. Best future recorded-action vector match remained poor
  (L2 ~6.48).
- **Episode leakage:** `LeRobotDataset(..., episodes=[1])` produced 146 frames
  with only `episode_index=1`. The true ep1-only clone produced 146 frames with
  only `episode_index=0`. Sampled dataloader batches contained only that single
  episode.
- **Stats mismatch for the cloned ep1-only run:** checkpoint action stats match
  raw ep1-only action stats. The earlier `--dataset.episodes=[1]` run used
  full-dataset stats, but the fresh clone fixed this and still did not fix raw
  offline precision.
- **Gripper-only scaling:** per-joint breakdown showed overlarge prediction
  deltas across arm joints too, especially `shoulder_lift` and `elbow_flex`.
- **Image blindness:** counterfactual swaps/zeroing showed meaningful image
  sensitivity, especially top camera. The model is not simply ignoring images.
- **VAE removal as a fix:** no-VAE had lower logged loss because KL was removed,
  but apples-to-apples inference metrics were worse than the VAE model.

### Important interpretation

The core measurement conflict is now clear:

- Training/logged ACT loss is normalized chunk loss, often with KL mixed in.
- Robot success needs raw, one-step absolute target precision.
- With absolute joint actions, action std is large because stats are computed
  over the whole absolute pose trajectory. A small normalized error can become
  multiple raw joint units:

```text
raw_error ~= normalized_error * action_std
```

Example from frame 0 of the ep1-only VAE checkpoint:

```text
normalized elbow error ~= 0.124
elbow action std ~= 30.8
raw elbow error ~= 3.8 joint units
```

Thus "reasonable" normalized L1 (~0.05-0.09) is still not precise enough for
pickup.

### Live eval learnings

- Stock `lerobot-rollout` exposed raw/clamped action targets and produced many:

```text
Relative goal position magnitude had to be clamped to be safe.
```

- Custom eval interpolation/smoothing made motion **way smoother**, confirming a
  real execution smoothness problem.
- Interpolation did **not** fix missed pickup. It smooths the path to the
  policy's target, but cannot fix a target that is spatially wrong.

### Current best diagnosis

There is probably no remaining obvious preprocessing or LeRobot-pipeline bug.
The failure is a precision / supervision representation issue:

- ACT predicts **absolute next joint positions**.
- The task needs small, precise corrective motion at pickup.
- Normalized absolute-action L1 can look good while raw one-step action error is
  still too large for grasping.
- On one-episode overfit, the model is learning but does not yet drive raw
  `chunk[0]` errors near zero.

### Remaining high-signal experiments

1. **Longer ep1-only convergence test with the VAE model.** The fresh ep1-only
   VAE 5k run was still learning. Train much longer and judge on inference-path
   normalized L1 plus raw per-joint `chunk[0]` error, not combined loss.
2. **Delta-action / velocity target training.** Predict `action - state` and add
   it back at eval time. This directly increases signal-to-noise on the small
   corrective motion instead of supervising absolute pose. **(Implemented — see
   "Normalization round-trip rule-out + delta-action workflow" below.)**
3. **If delta actions also fail:** recollect/relabel demonstrations with slower,
   more precise approach/pickup and more consistent trajectories.

## Normalization round-trip rule-out + delta-action workflow (2026-06-25)

### Round-trip rule-out (the normalization is NOT the bug)

We had repeatedly suspected a normalization floor (`NORM_STD_MIN=1e-2` clamp).
`evaluation/normalization_roundtrip_check.py` decisively ruled this out for the
stock LeRobot towel checkpoints:

- The preprocessor normalizer and postprocessor unnormalizer `action.mean` /
  `action.std` are **byte-identical** (`max_abs_diff = 0.0`).
- `denormalize(normalize(a))` over the full action range has **max error
  2.1e-14** (machine precision). All action stds are 12.5–35.2, far above the
  `1e-2` clamp, so the clamp never bites.

**Conclusion:** the large raw one-step error on a single-episode overfit is
genuine model under-fit (LR `1e-5`, only ~274 epochs vs the paper's ~2000) and
**absolute-action std amplification**, not a pipeline bug. Image augmentation was
also already off (`image_transforms.enable=false`), so it was not blocking
memorization either.

### Why delta actions (quantified)

Same script measured, on `009-place-towel-in-bowl-10fps-ep1-only`:

| joint | absolute std | delta std (`action-state`) | ratio |
|---|---:|---:|---:|
| shoulder_pan | 16.0 | 0.85 | 0.053 |
| shoulder_lift | 35.2 | 1.84 | 0.052 |
| elbow_flex | 30.8 | 0.57 | **0.018** |
| wrist_flex | 12.5 | 1.85 | 0.148 |
| wrist_roll | 15.9 | 0.79 | 0.050 |
| gripper | 16.8 | 2.45 | 0.146 |

Delta std is ~0.08x absolute std on average (elbow ~0.018x = 54x tighter). So at
the **same normalized L1**, a delta-action policy yields ~12x smaller raw joint
error on average (elbow ~54x) — directly improving grasp precision.

### End-to-end delta-action recipe (stock LeRobot, temporal-ensemble safe)

The relabel is per-frame (`action[t] -= observation.state[t]`), so the delta
target is a pure function of the frame and temporal ensembling stays correct
(all query times estimate the same delta). At eval the absolute command is
reconstructed from the **live** state, which is the desired relative-control
behavior.

```bash
# 1. Relabel the dataset to delta actions (fresh stats over deltas).
uv run python evaluation/make_delta_action_dataset.py \
  --source-repo RevanthGundala/009-place-towel-in-bowl-10fps \
  --target-repo RevanthGundala/009-place-towel-in-bowl-10fps-delta \
  --root outputs/filtered_datasets/009-place-towel-in-bowl-10fps-delta \
  --push

# 2. Train stock lerobot-train on the delta dataset (NO code change).
uv run python deploy/train_lerobot_official_vastai.py \
  --dataset-repo-id RevanthGundala/009-place-towel-in-bowl-10fps-delta ...

# 3. Offline sanity (errors reported in absolute joint units, comparable to
#    absolute runs). Point --dataset at the delta dataset.
uv run python evaluation/offline_lerobot_action_check.py \
  --policy-path <delta_run>/pretrained_model \
  --dataset RevanthGundala/009-place-towel-in-bowl-10fps-delta \
  --action-space delta

# 4. Live eval reconstructs command = current_state + policy_output.
uv run python -m models.act.eval_lerobot \
  --policy-path <delta_run>/pretrained_model --action-space delta
```

Judge success on raw per-joint one-step error and live pickup, not combined loss.

The relabel tool runs a built-in self-check that reconstructs `delta + state` and
asserts it recovers the original absolute action (verified max error 1.8e-7 on
`009 ep1-only`). The same relabeled dataset also works for the **custom** ACT
path (`models/act/train.py` consumes the delta dataset unchanged; run the custom
live eval with `python -m models.act.eval --action-space delta`). The offline
check supports `--action-space delta` too and reports errors in absolute joint
units for apples-to-apples comparison with absolute runs.

### Result: delta vs absolute, official TE 10k (2026-06-26)

> **CORRECTION (2026-06-27): the "~9.7x win" below is WRONG — it was a units
> artifact, and delta actions FAIL live (worse than absolute). See
> "Delta-action post-mortem (2026-06-27)" below. Keep this section only as a
> record of the mistake.**

Trained stock `lerobot-train` ACT on `009-place-towel-in-bowl-10fps-delta`
(10k steps, chunk=30, n_action_steps=1, te_coeff=0.01, VAE kl=10, lr=1e-5) —
identical recipe to the absolute TE 10k baseline except absolute→delta actions.
Final training loss 0.27. Swept all 25 episode start frames, errors in absolute
joint units (delta reconstructed via `pred + state`):

| metric (mean over 25 start frames) | absolute TE 10k | delta TE 10k | improvement |
|---|---|---|---|
| max `|pred-state|` (the old sanity metric) | 4.69 | 2.45 | 1.9x |
| **mean `|pred-recorded|`** (true one-step error) | 2.247 | **0.232** | **~9.7x** |
| max `|pred-recorded|` | 4.55 | 0.725 | ~6.3x |

The absolute sweep reproduced the documented 4.56 (got 4.69), confirming the
comparison. Delta actions cut true one-step joint error ~10x — matching the
predicted ~12x amplification win. `predicted_minus_state` shows real motion
(e.g. gripper +3.1 matching recorded +2.9), so the policy did not collapse to
zero deltas. Checkpoint: `RevanthGundala` delta run, instance-42791229. Next:
live OMX eval with `--action-space delta` to confirm the offline gain transfers
to real pickup.

> Launcher note: `deploy/train_lerobot_official_vastai.py` monkey-patches the
> remote command and (on the bare CUDA image) must install lerobot **with** deps
> and extras, not `--no-deps`. It now runs
> `pip install "lerobot[diffusion,dynamixel,training] @ git+...@<ref>"`. Any shell
> args added to that `bash -lc '...'` command must use double quotes (nested
> single quotes break it).

### Delta-action post-mortem (2026-06-27): delta is the WRONG lever

Ran the delta TE-10k checkpoint live on the real OMX twice and it failed worse
than absolute. Root-caused it; conclusions:

1. **The "~9.7x offline win" was a units artifact, not a real improvement.**
   `outputs/startframe_sweep.py` measured the delta error in *delta* action units
   (target std ~1.4) and the absolute error in *absolute* units (target std ~21).
   Comparing 0.232 to 2.247 is apples-to-oranges. Normalized by each target's std:
   delta 0.232/1.4 ≈ **0.17**, absolute 2.247/21.4 ≈ **0.11** — the absolute
   policy is actually *slightly better*. The offline metric never indicated task
   success.

2. **Live, the delta policy drifts monotonically and never grasps.** Two runs
   (clamp 2.0 @30Hz, and clamp 10.0 @10Hz): the arm drives one direction and runs
   off (elbow_flex 53→6, shoulder_lift −65→−34), gripper barely moves. Raising the
   clamp did not help — the clamp was not the cause.

3. **Mechanism: delta relabeling made the policy IGNORE proprioception.**
   Vision-sensitivity probe (`outputs/vision_sensitivity_probe.py`, swap only
   images vs only state across the 25 start frames):

   | policy | Δaction when only images change | Δaction when only state changes |
   |---|---|---|
   | absolute TE 10k | 2.64 | **1.34** (uses state) |
   | delta TE 10k | 0.42 | **0.0008** (ignores state) |

   Subtracting state from the target (`delta = action − state`) removes the
   action↔state correlation, so the network learns to predict deltas from vision
   alone. But delta reconstruction at eval is `command = live_state + output`, and
   closed-loop stability *requires* the delta to depend on state (a goal-seeking
   delta must shrink as the arm nears the target). A vision-only delta is ~constant
   per scene → constant-velocity drift → runaway. **Delta is structurally wrong for
   a state-ignoring ACT policy.**

4. **Eval-time knobs are exhausted** (confirmed from the full rollout-summary
   history): `--max-joint-delta` tried at 0.0/1.5/2/4/8/10, `--hz` at 10 and 30,
   absolute and delta action spaces, plus smoothing/interpolation. None fix it,
   because none can fix a spatially-wrong target.

5. **009 images are degenerate too.** Top-camera per-channel pixel std ≈ **0.01**
   (wrist ≈ 0.013) — same low-variability pathology the docs flagged for 007. The
   scene barely varies, so there is little visual signal to precisely localize the
   towel. Consistent with a policy that drives an average trajectory and misses.

**Net:** action representation is settled (delta is worse than absolute; both fail
live). Combined with the earlier finding that even genuine official ACT scores 0%
in sim, the remaining lever is the **data** — coverage/diversity/precision of the
demonstrations — not any policy or eval knob. Next action: full audit of the
penne/towel datasets to quantify start-state coverage and near-duplicate bias.

## Full dataset audit (2026-06-27)

Audited every available OMX dataset (`outputs/full_dataset_audit.py`,
results in `outputs/dataset_audits/full_audit/`). 006-place-in-cup and
008-move-to-pasta are empty cache stubs (deleted from HF, 0 parquet) — skipped.

| dataset | eps | top/wrist pixel-std | start pairwise dist | **qpos→action ratio** | penne obj pairwise px |
|---|---|---|---|---|---|
| 007-penne base | 50 | 0.015/0.029 | 0.67 | **0.048** | 151 |
| 007-bowl-left-21 | 21 | 0.013/0.021 | 0.66 | **0.043** | 103 (median 35!) |
| 007-diverse75 | 38 | 0.016/0.030 | 0.70 | **0.046** | 154 |
| 008-touch-bowl | 26 | 0.015/0.047 | 1.32 | **0.063** | 203 |
| 009-towel | 25 | 0.010/0.013 | 0.73 | **0.037** | n/a |
| 009-towel-delta | 25 | 0.010/0.013 | 0.73 | 0.567 | n/a |

**Two systemic pathologies in ALL absolute datasets:**

1. **Strong qpos shortcut.** A *linear* map from current joint positions predicts
   the action with only ~4–6% of the mean-baseline error (`qpos→action ratio`
   0.037–0.063). The action is almost fully determined by proprioception, so the
   policy is never forced to use the cameras during training. (The delta dataset
   breaks this — ratio 0.567 — because `action−state` removes the correlation;
   that is also why the delta policy went state-blind and drifted.)

2. **Degenerate images.** Top-camera per-channel pixel std 0.010–0.016 (009 worst).
   Even if the policy tried to use vision, there is little visual signal — the
   object is small against a near-static scene.

Plus **limited start-state diversity**: `elbow_flex` start std ≈ 0.19–0.23° in
*every* dataset (near-frozen); 008-touch-bowl also freezes the gripper; the
`bowl-left-21` subset clusters the penne (median pairwise only 35 px).

**Why this is the real root cause.** The start joint pose is *correlated* with the
object position (the operator pre-positions the arm per layout), so qpos encodes
the task and the network learns a proprioception-driven average trajectory that
ignores low-variance cameras → it cannot localize the object → misses the grasp.
No action representation or eval knob can fix data with this structure.

**Data-collection fixes (in priority order):**

1. **Decorrelate proprioception from the task: always start every demo from the
   SAME fixed home pose**, regardless of object position. Then start-qpos is
   constant and cannot predict the object-dependent action — the policy is forced
   to use vision. (Highest leverage; directly kills the shortcut.)
2. **Increase visual signal**: better lighting/contrast, object larger in frame,
   tighter workspace crop so the object isn't a few low-variance pixels.
3. **Increase + balance object-position coverage** (farthest-point like
   diverse75; avoid bowl-left clustering); record left/center/right × near/far.
4. **More episodes** (paper uses ~50; towel has 25).

### Pour-water (old) vs penne/towel (new) — the rig got WORSE

Audited the pour-water datasets too (`outputs/pour_water_audit.py`). The
`*-globalstats` variants fail to load (HF metadata error) but equal their bases.

| dataset | eps | top/wrist pixel-std | qpos→action |
|---|---|---|---|
| 002-pour-water | 53 | **0.103**/0.097 | 0.075 |
| 003-pour-water | 70 | **0.060**/0.072 | 0.066 |
| 003-pour-new20-only | 20 | 0.016/0.068 | 0.067 |
| 004-pour-water-relative | 50 | 0.022/0.065 | 0.562 (relative) |
| 007-penne base | 50 | 0.015/0.029 | 0.048 |
| 009-towel | 25 | **0.010**/0.013 | 0.037 |

**Key difference — visual signal collapsed over collection generations.**
Old pour-water (002/003) top-camera pixel std is **0.06–0.10**; the newest towel
is **0.010** — a **6–10× drop** in visual variability. Old sets are also larger
(53–70 eps vs 25). **The pour-water rig produced far more learnable vision data
than the current penne/towel setup.** Whatever changed between then and now
(camera distance/zoom, object size, background, lighting) destroyed visual signal.

**Similarity:** the qpos shortcut is present in *both* eras (ratio 0.05–0.075) —
it is inherent to absolute position-control teleop where the arm is pre-positioned
per layout — but it is actually slightly *worse* (stronger) in the newest towel
set. `004-pour-water-relative` (ratio 0.562) shows relative actions were already
tried back at v004 and abandoned — consistent with the 2026-06-27 finding that
delta/relative actions are the wrong lever.

**Implication:** for a near-term working policy, `003-pour-water` (70 eps, ~6×
the image variance of towel) is the best existing substrate. For new collection,
first restore the old pour-water camera/scene setup's visual variability, then
add the fixed-home-pose protocol to also kill the qpos shortcut.

## Suggested next steps

1. ~~Reproduce the paper recipe first with the **official** model (all 50 demos,
   match epochs/eval protocol) until it hits ~90% — that validates the harness.
   Only then is a custom-vs-official success-rate comparison meaningful.~~
2. ~~Once the harness reproduces the paper, re-evaluate the patched custom model.~~
3. ~~Retrain on the real OMX pour-water dataset with these fixes and check the
   latent (KL not collapsing) and val behavior there.~~
4. **Run real robot eval with pretrained backbone checkpoint** to verify visual
   conditioning is working.
5. If still underperforming, consider: reduce chunk_size, use relative actions,
   add more demonstrations.

## Reproduce

```bash
# Multi-episode reference comparison (Modal A10G), val L1:
modal run --detach deploy/train_official_act_modal.py --profile sim_transfer_cube_reference --run-name ref_official_seed0
modal run --detach deploy/train_custom_act_modal.py   --profile sim_transfer_cube_reference --run-name ref_patched_seed0

# Metrics land in the volumes (root = /outputs; do NOT prefix /outputs in `modal volume get`):
modal volume get omx-custom-act-training-logs   /act_experiments/sim_transfer_cube_reference/ref_patched_seed0/metrics.jsonl .
modal volume get omx-official-act-training-logs /act_experiments/sim_transfer_cube_reference/ref_official_seed0/metrics.jsonl .

# Real MuJoCo success-rate eval (needs ACT_REPO_DIR + MuJoCo) — NOT yet run:
python -m models.act.eval --checkpoint <run_dir>/checkpoint_last.pt --act-repo-dir /path/to/tonyzhaozh/act
```

## pi0.5 capability probe (2026-06-28)

**Why:** after proving *no* ACT/diffusion towel model beats the trivial "freeze at
current state" baseline (uniform offline scoreboard: baseline norm_mae 0.062;
every absolute ACT/diffusion model 1.3–12× *worse*; only the delta model "beats"
it by memorizing the qpos shortcut, which is why it drifts live), the last
untested lever was a strong pretrained VLA. Decision logic: pi0.5 works → ACT
capacity was the bottleneck; pi0.5 also fails → data/setup/task problem.

**Training (DONE, succeeded):** full finetune of `lerobot/pi05_base` on
`RevanthGundala/009-place-towel-in-bowl` (30fps, 25 ep, 9869 frames), stock
`lerobot-train` on Modal A100-80GB, batch 32, 3000 steps (~9.6 epochs), bf16,
gradient checkpointing. Final loss **0.008** (clean convergence from 0.070).
Checkpoint pushed to **`RevanthGundala/pi05-towel`** (HF, 9 files).

Two infra fixes were required in `../omx-training/deploy/train_modal_lerobot.py`:
1. Pin lerobot to git commit `87242cfced` (== local venv 0.5.2). PyPI
   `lerobot[pi]>=0.5.1` renamed the processor step `relative_actions_processor`
   and cannot load `pi05_base`'s saved pipeline ("not found in registry").
2. Install `lerobot[pi,training]` (not just `[pi]`): the git build splits out
   `datasets`/wandb/accelerate that the PyPI wheel bundled.
(Also: Modal `--detach` is mandatory — the launcher uses `Function.spawn()`.)

**EVAL BLOCKER (RESOLVED via Modal) — local hardware cannot run pi0.5:** this dev
Mac has only **8.6 GB RAM** (~2.6 GB free). A 3B-param PaliGemma VLA needs ~6 GB
(bf16) / ~12 GB (fp32) just for weights, so the process is OOM-killed (exit 137)
on BOTH MPS and CPU before completing a single inference. Consequences:
- Offline scoring of pi0.5 cannot run on this machine (the ACT scoreboard works
  only because ACT is tiny ~50M params).
- **Live OMX eval also cannot run here** — `models/act/eval_lerobot.py` drives the
  robot from this same 8 GB Mac and would OOM loading pi0.5.

**Resolution:** offline scoring was moved to a Modal GPU container
(`../omx-training/deploy/eval_pi05_offline_modal.py`, A10G, reuses the pinned
lerobot image). It loads `RevanthGundala/pi05-towel`, runs single-step open-loop
prediction on 40 frames, and compares normalized MAE vs the freeze baseline —
the same metric as the ACT scoreboard.

**OFFLINE VERDICT — pi0.5 BEATS the freeze baseline (decisive positive):**

| model (action space)        | offline norm_mae | vs freeze baseline 0.062 |
|-----------------------------|------------------|--------------------------|
| **pi0.5 towel 3k (abs)**    | **0.025**        | **0.40× — BEATS**        |
| every absolute ACT/diffusion| 0.083–0.770      | 1.3–12× — WORSE          |
| delta ACT 10k               | 0.014            | 0.22× — beats, but only by memorizing the qpos shortcut |

Why this matters: pi0.5 is an **absolute-action** policy, yet it is the *only*
absolute model that beats "freeze." Every absolute ACT/diffusion model was 1.3–12×
*worse* than doing nothing. pi0.5's win is **not** the delta trick (the delta ACT
model beats baseline trivially because its targets are ≈0). So pi0.5 genuinely
extracts predictive signal from the cameras + state that ACT could not.
(Result: `outputs/dataset_audits/pi05_towel_score.json`.)

**Verdict: pi0.5 learned the mapping ACT could not → the bottleneck was MODEL
CAPACITY, not (primarily) the data.** A strong pretrained VLA fits this 25-episode
towel dataset where every from-scratch ACT/diffusion run failed. Caveats:
- Offline single-step open-loop is *necessary but not sufficient*; the real proof
  is closed-loop on the robot (compounding error, gripper timing, drift).
- **Live OMX eval still pending** — must run off this 8 GB Mac, from a machine with
  ≥16 GB RAM. `eval_lerobot.py` is policy-agnostic; use `--task "Place the towel
  in the bowl"` (pi0.5's tokenizer requires the instruction) plus `--action-space
  absolute --move-to-dataset-start-state --start-state-repo-id
  RevanthGundala/009-place-towel-in-bowl`.

**Next step if live eval confirms:** the path forward for ACT-class policies is to
either (a) use pi0.5 / a VLA as the deployed policy, or (b) attack the qpos
shortcut that cripples ACT — record every demo from a *fixed* home pose so start
state is decorrelated from object position, forcing the policy to use vision.

## Sponge 014 in-distribution async eval (2026-07-12)

**Run:** official LeRobot ACT trained for 20k steps on the no-op-trimmed
`RevanthGundala/014-place-sponge-in-bowl-diverse-50` dataset (50 episodes,
15,066 frames), chunk size 30. This checkpoint was trained without dataset image
transforms. Live evaluation used `models/act/eval_async.py` with its default
closed-loop async settings: 30 actions per chunk, queue threshold 0.4,
`conservative` chunk aggregation, and action smoothing 0.7.

| in-distribution layout | successes |
|---:|---:|
| 1 | 2/3 |
| 2 | 1/3 |
| 3 | 0/3 |
| 4 | 2/3 |
| 5 | 1/3 |
| 6 (later identified as likely OOD) | 0/3 |
| **Verified/claimed ID subtotal (layouts 1-5)** | **6/15 (40.0%)** |
| **All attempted layouts** | **6/18 (33.3%)** |

**Failure observations:**

- Layout 6 was completely mis-aimed in all three attempts, but its original
  classification as in-distribution was based on an overly coarse 3x3 grid.
  It is now treated as likely OOD and excluded from the ID subtotal. True ID
  requires the joint sponge position, bowl position, sponge angle/separation,
  and robot start pose to match dataset support—not merely each object's
  independent grid cell.
- The other failed attempts generally approached the sponge but failed to pick
  it up.
- The initial predicted trajectory often aimed incorrectly. Closed-loop async
  replanning then moved the arm closer to the sponge, but after the missed grasp
  the arm eventually stopped making useful progress.
- Working hypothesis: replanning helps compensate for a poor initial chunk, but
  a missed grasp moves the robot/object relationship outside the states covered
  by successful demonstrations. The policy then has no demonstrated recovery
  behavior and stalls. This is a hypothesis from rollout behavior, not yet
  proven by an offline counterfactual.

### Controlled fully open-loop comparison

Set the queue threshold to zero so the client executes all 30 actions before
requesting another observation. `latest_only` makes chunk boundaries explicit;
all other important eval settings remain unchanged:

```bash
uv run python models/act/eval_async.py \
  --policy-path outputs/vastai/instance-44620633/outputs/lerobot_official/lerobot_official_act_chunk30_steps20000/checkpoints/020000/pretrained_model \
  --task "Place the sponge in the bowl" \
  --chunk-size-threshold 0 \
  --aggregate-fn latest_only
```

This is open-loop within each one-second, 30-action chunk, not one open-loop
trajectory for the entire task. Compare it on the same six layouts without
changing smoothing or placement; otherwise the result will not isolate whether
mid-chunk replanning is helping or hurting.

### Offline image-swap diagnostic

To determine whether the crafted visual diversity was learned, the 20k
checkpoint was evaluated on the first frame of all 50 episodes. For each
episode, both camera images were replaced with images from a different episode
while keeping the original robot state fixed.

- Swapping images changed the predicted 30-step chunk by **7.36 degrees mean
  L2 per step**.
- This is **62.6%** of the checkpoint's natural prediction difference between
  two different episode starts (11.77 degrees).
- Swapping robot start state while keeping images fixed had a similarly large
  effect (6.98 degrees, 59.3% of natural variation).

**Conclusion: the policy is materially conditioned on the images.** The 50
episodes were not wasted, and this is not a simple camera-ignored/qpos-only
collapse.

The following first-chunk imitation diagnostic was also measured:

| offline metric over 50 demonstrated starts | ACT 20k | hold-current-state baseline |
|---|---:|---:|
| first-action MAE | 1.97 deg | **0.58 deg** |
| first 10 actions MAE | 1.71 deg | **0.64 deg** |
| full 30-action chunk MAE | **1.71 deg** | 1.78 deg |

This metric must **not** be interpreted as task success. Holding is an unusually
strong baseline during the first second because demonstrations begin slowly and
the trimmer intentionally retains a four-frame margin. There are also multiple
valid trajectories for the same scene, while MAE compares the policy against one
specific operator trajectory. A 1.71-degree mismatch can still be task-valid,
and successful async rollouts use later camera-conditioned replans rather than
only this initial chunk. The result is consistent with the observed initial
mis-aim, but by itself does not prove that the checkpoint is underfit or explain
the rollout success rate.

### Default image transforms live comparison

A fresh official LeRobot ACT model was trained on the same 50 episodes for 50k
steps with LeRobot's default image transforms enabled. The same five claimed-ID
layouts were evaluated at both the 20k and 50k checkpoints with the same default
async settings:

| layout | 20k, no transforms | 20k, default transforms | 50k, default transforms |
|---:|---:|---:|---:|
| 1 | 2/3 | 2/3 | 0/3 |
| 2 | 1/3 | 0/3 | 0/3 |
| 3 | 0/3 | 0/3 | 1/3 |
| 4 | 2/3 | 0/3 | 0/3 |
| 5 | 1/3 | 3/3 | 2/3 |
| **Total** | **6/15 (40%)** | **5/15 (33.3%)** | **3/15 (20%)** |

At the controlled 20k duration, transforms changed observed success from 6/15 to
5/15: only one trial, so this does not establish that augmentation helps or
hurts. The augmented model then fell to 3/15 at 50k, but these samples remain too
small to distinguish a real training-duration effect from rollout variance. The
large per-layout changes also suggest sensitivity to exact placement. The
default transform set includes random affine translation/rotation while leaving
action labels unchanged, which may be counterproductive for pixel-precise object
localization, but these live results do not prove that hypothesis.

A second three-attempt batch produced `3/3, 0/3, 2/3, 0/3, 0/3` for the older
unaugmented 20k checkpoint and `2/3, 0/3, 0/3, 0/3, 3/3` for the augmented 20k
checkpoint. Combined with the first batches, the per-layout results are:

| layout | geometry (bowl; sponge) | unaugmented 20k combined | augmented 20k |
|---:|---|---:|---:|
| 1 | left-middle; right-middle horizontal | 5/6 | 4/6 |
| 2 | right-middle; left-middle vertical | 1/6 | 0/6 |
| 3 | center-middle; left-top diagonal `/` | 2/6 | 0/6 |
| 4 | center-top; right-middle diagonal `\` | 2/6 | 0/6 |
| 5 | left-middle; center-top vertical | 1/6 | 6/6 |
| **Total** | | **11/30 (36.7%)** | **10/30 (33.3%)** |

The nearly equal aggregate rates hide a strong model-specific spatial bias. Both
checkpoints handle layout 1 well and both fail layout 2. The augmented checkpoint
is consistently perfect on layout 5 while the older checkpoint usually fails it,
but augmentation does not generalize that success to the other vertical layout
(2), so sponge orientation alone is not the explanation. Conversely, only the
older checkpoint succeeds at all on layouts 3 and 4. This looks like each
checkpoint learned a different narrow set of pickup regions/trajectories rather
than robust visual localization across the workspace.

The older checkpoint also changed sharply between batches on layouts 3, 4, and
5 (`0→2`, `2→0`, and `1→0` successes). Since inference is effectively
deterministic, this points to sensitivity to small setup/execution differences
and grasp-boundary behavior, not random policy sampling. The evidence supports a
precision/localization problem: each policy's aiming bias happens to work in
some layouts and miss in others. It does not support a general orientation rule
or show that default augmentation improves robustness.

### Photometric-only 20k ablation

A third matched 20k checkpoint was trained with seed 1000 and only brightness,
contrast, saturation, hue, and sharpness transforms. Geometric affine rotation
and translation were excluded. The run resumed from its saved 10k optimizer/RNG
state after Vast host failures and finished at loss 0.076. Final checkpoint:

`outputs/vastai/instance-44697377/outputs/lerobot_official/lerobot_official_act_chunk30_steps20000_photometric/checkpoints/020000/pretrained_model`

Live evaluation on the same five layouts, six attempts each:

| layout | no transforms | all default transforms | photometric only |
|---:|---:|---:|---:|
| 1 | 5/6 | 4/6 | 1/6 |
| 2 | 1/6 | 0/6 | 0/6 |
| 3 | 2/6 | 0/6 | 0/6 |
| 4 | 2/6 | 0/6 | 1/6 |
| 5 | 1/6 | 6/6 | 4/6 |
| **Total** | **11/30 (36.7%)** | **10/30 (33.3%)** | **6/30 (20.0%)** |

Removing affine transforms did not restore the unaugmented model's spatial
coverage. Photometric-only retained the transformed model's layout-5 bias and
still collapsed on layouts 2 and 3. Therefore affine augmentation alone does
not explain the specialization. The controlled conclusion is that neither
transform configuration improved this task; the unaugmented 20k checkpoint
remains the best and broadest base policy. The photometric result also cautions
against attributing the transformed models' behavior to a single transform:
training-time augmentation changed the learned solution, but this small
dataset/model can converge to a narrow layout-dependent policy even when pixel
geometry is unchanged.
