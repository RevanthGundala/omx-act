# Sponge → hot pad: experiment log

Everything we ran on the "Place the sponge on the hot pad" task, in order, with
the numbers we actually measured. Anything not measured is marked as such.

The through-line: a behaviour-cloned ACT policy failed, we found *why*, fixed
the data rather than the model, and then compared DAgger against RECAP under a
matched data budget.

---

## 0. Task and hardware

- OpenMANIPULATOR-X (OMX), 6 joints, 30 Hz control.
- Two cameras: `top` and `wrist`, 480×640 RGB.
- Policy: custom ACT (`models/act`), chunk size 30.
- Hot pad position is **fixed by design**; only the sponge moves.

---

## 1. The original failure: dataset 017

The first policy reached for a point consistently *outside* the sponge. The
error was systematic, not random, which ruled out simple undertraining noise.

### 1.1 Was it a bad checkpoint?

Scanned all ten checkpoints (`base_checkpoint_bias_scan.json`). Outward bias at
episode end, in millimetres:

| step | 2k | 6k | 10k | 14k | 18k | 20k |
|---|---|---|---|---|---|---|
| outward bias (mm) | 32.7 | 26.4 | 19.9 | 19.0 | 11.1 | **10.7** |

Falling monotonically, best at the final checkpoint. **Not checkpoint
selection** — the model was still improving when training stopped.

### 1.2 Was it undertraining?

Resumed 20k → 50k (`base_20k_to_50k_bias_scan.json`):

| step | 20k | 30k | 40k | 50k |
|---|---|---|---|---|
| outward bias (mm) | 10.66 | 6.77 | 9.41 | **5.40** |
| EE MAE (mm) | 15.75 | 16.12 | 11.94 | **10.46** |

Improved but did not resolve. Training longer helped; it was not the root cause.

### 1.3 The actual root cause: temporal misalignment

Audited motion onset across the 48 episodes:

- Median onset: **frame 20**
- Max onset: **frame 48**
- **11 of 48 episodes** began moving *after frame 30*

ACT's chunk size is 30. In those 11 episodes the entire reach fell outside the
prediction horizon, so the policy was trained to predict "hold still" from the
start state.

Secondary issue: wrist-flex start pose spread of **23.2 units** (range
30.4–53.6, std 5.46), so the same visual scene mapped to many different arm
configurations.

**Conclusion: the demonstrations were bad, not the architecture.**

---

## 2. Fixing data collection (dataset 026)

Changes to `omx-training/data/record.py`:

| Problem | Fix |
|---|---|
| Variable start pose | `--fixed-start-profile hot-pad-v1` canonical reset |
| Late motion onset | Enter-to-arm + sustained-motion trigger, 6-frame pre-roll |
| Duplicate scene layouts | `--strict-coverage`, no duplicate override |
| Wrist-flex stiction on reset | Directional seating: approach target from below |

### 2.1 Two collection bugs worth remembering

**Cumulative displacement cannot detect motion onset.** Drift crosses any fixed
threshold given enough time. Replaced with speed over a sliding window: a real
reach moves 1.6–3.5 units per 3 frames, drift moves ~0.11.

**The leader arm sags 1.54 units when torque releases** — already above the 1.5
trigger, so recording fired while stationary. Fixed with a settle-wait (4
consecutive reads drifting < 0.15) before capturing the reference pose.

### 2.2 Result

`RevanthGundala/026-place-sponge-on-hot-pad-clean` — 48 episodes, 13,484 frames.

- Motion onset: median **1**, max **3** (was median 20, max 48)
- Scene coverage: **48/48 planned buckets** (6 positions × 8 orientations)

Outward bias after 20k steps on the clean data (`clean48_checkpoint_scan.json`):

| step | 2k | 6k | 10k | 14k | 18k | 20k |
|---|---|---|---|---|---|---|
| outward bias (mm) | −0.36 | 3.74 | 3.51 | 2.27 | 1.90 | **2.10** |
| EE MAE (mm) | 21.4 | 15.5 | 12.9 | 11.1 | 9.2 | **9.5** |

**2.10 mm vs 10.66 mm on the old dataset — a 5× reduction from data alone.**

---

## 3. Custom ACT vs stock LeRobot ACT

Trained both on dataset 026, 20k steps, chunk 30.

| | stock LeRobot | custom |
|---|---|---|
| final train loss | 0.071 | 0.070 |
| joint MAE (normalized units) | 1.2112 | **1.1868** |

Verified equivalent: normalizer *and* unnormalizer tensors bit-identical
(0.00e+00 across 90 tensors), identical processor chains, identical ResNet18
backbone (11,186,112 params), matching attention math, weight-transfer
equivalence to ~3e-7.

**Unresolved:** the custom model showed more vertical reversals per second live
(8.6 vs 7.0) despite *smaller* per-step motion and *lower* jerk. Two independent
auditors found no semantic bug. Best explanation: ACT's loss never constrains
chunk-to-chunk consistency, so two models can tie on loss and differ in
closed-loop stability. Not resolved; would need 3 seeds each to separate
systematic effect from variance.

---

## 4. DAgger

Collected 18 correction episodes against the custom base policy
(`rollout_027-custom48-dagger`, 4,229 frames, 141 s of teleop).

Aggregate `028-custom48-dagger18`: 48 demos + 18 corrections = 66 episodes,
17,549 frames, 24.1% correction frames.

Fine-tuned 10k steps from the custom 20k base (`dagger18_ft_scan.json`):

| checkpoint | seed MAE | correction MAE | outward (mm) |
|---|---|---|---|
| BASE (20k) | 1.187 | 3.865 | 1.79 |
| ft 4k | 1.165 | 1.596 | 3.50 |
| ft 8k | 0.816 | 1.542 | 0.37 |
| **ft 10k** | **0.744** | **1.420** | 1.17 |

Correction tracking improved 2.7× *while* seed accuracy also improved — it
learned recovery without forgetting the demonstrations.

### 4.1 Live result

| model | success rate |
|---|---|
| custom base | 4/22 = **18%** |
| custom + DAgger-18 | 9/15 = **60%** |

**Fisher exact test: p = 0.0145.** Using only the clean 15-trial baseline
session (4/15 = 27%) it is 27% → 60%, still a clear gain.

Caveats: the baseline pools 5 sessions with differing settings, and the two
evaluations used independently chosen sponge placements rather than identical
paired scenes.

---

## 5. RECAP

Paper: **π\*₀.₆: a VLA That Learns From Experience**, arXiv:2511.14759
(Physical Intelligence, Nov 2025). RECAP = RL with Experience and Corrections
via Advantage-conditioned Policies.

### 5.1 What the paper prescribes

| Parameter | Value | Source |
|---|---|---|
| Value bins | 201 | Sec. IV-A |
| Discount | none (γ = 1) | Sec. III |
| Reward | −1/step, 0 terminal success, −C_fail terminal failure | Eq. 5 |
| Critic | observation-only V(o) | Sec. IV-A |
| Advantage | binary token, N=50 lookahead | Sec. V-B, App. A-F |
| Threshold | percentile → ~40% positive in fine-tuning | App. A-F |
| Conditioning dropout | 30% | App. A-F |
| Corrections | forced `I_t = True` | Sec. IV-B |
| Autonomous:correction ratio | ~1.6:1 (450:287, 600:360) | App. A-F |

**Not specified by the paper:** exact C_fail, batch mixing ratio between demos
and rollouts, bin spacing (uniform vs HL-Gauss), initial demo counts.

### 5.2 Why the critic has no CVAE latent

The advantage is `A_t = Σr[t:t+N] + V(o_{t+N}) − V(o_t)`.

A critic conditioned on future actions predicts the realized return exactly, so
`V(o_t) = R_t`. Since `R_t = Σr[t:t+N] + R_{t+N}` by definition:

```
A_t = Σr + R_{t+N} − R_t = 0
```

Measured on our data: **advantage std = 0.000000** across every episode. Not
small — exactly zero. A future-conditioned critic makes RECAP inert.

This is a proof, not an ablation: it holds for any weights. Note the RECAP
*policy* keeps ACT's CVAE unchanged (all 17,373,248 params); only the critic
omits it.

### 5.3 Why the distributional head is required

Terminal frames in our dataset take only two values: **0 or −704**
(8 successes, 10 failures). The MSE-optimal scalar prediction is **−391.1**, and
**0 of 18** episodes end near it. A scalar critic is forced to predict a value
that never occurs. The 201-bin head represents both modes directly.

### 5.4 Data collection

DAgger's 18 correction episodes stored **only** intervention windows
(`record_autonomous=False`), 100% intervention frames. RECAP needs the
autonomous segments — they are the only source of negative-advantage examples.
So new collection was required.

Split 11 autonomous + 7 corrections = **18 episodes**, matching DAgger's budget
exactly and landing near the paper's ~1.6:1 ratio.

Position coverage, sponge bucket counts:

| | old eval rollouts | new collection |
|---|---|---|
| buckets covered | 5/6 | **6/6** |
| max skew | **17×** | 5× |

`031-recap-gen1`: 18 episodes, 6,354 frames, C_fail = 704 (longest episode 703
+ 1), 8 successes / 10 failures.

### 5.5 The MLP sanity check

Before spending GPU time, trained a ~140K-param state-only MLP on the same
returns.

- Normalization verified: returns map to exactly [−1.0000, 0.0000], bins span
  [0, 200], 160 distinct bins used.
- Train CE 5.29 → 1.06 — **overfitting is the point**: it proves the pipeline,
  labels, and gradients are sound.
- Validation peaked early then collapsed:

| step | 0 | **250** | 1000 | 3000 |
|---|---|---|---|---|
| val MAE | 347.1 | **256.4** | 315.6 | 339.1 |
| vs baseline | +1.9% | **+27.5%** | +10.8% | +4.1% |

Conclusion: signal exists, and **validation-based checkpoint selection is
mandatory**. This directly set `save_freq=500` for the critic runs.

Adding elapsed time as an input moved improvement from +3.9% → +29.1%, i.e. most
predictable structure is the countdown implied by −1/step, not scene content.

### 5.6 Critic training

Two Modal A10G runs, 10k steps, warm-started from the custom ACT base
(**99.8%** of critic params transfer; only the 103K value head is new).

| | `freeze=encoder` (18% trainable) | `freeze=none` (100%) |
|---|---|---|
| **best val MAE** | **182.8** @ step 3500 | 209.5 @ step 7000 |
| vs baseline (370.4) | **+50.6%** | +43.4% |
| best val loss | **4.137** | 5.184 |
| success AUC | **0.848** | 0.842 |

**Freezing the encoder beat full fine-tuning on every metric** — a deviation
from the paper, which fine-tunes fully but had 400–1000 episodes per task
against our 18.

The winner peaked at step 3500 of 10000 and degraded to 233.8 by the end.
Without validation-based selection we would have shipped a checkpoint **28%
worse**.

Selected: `outputs/recap_critic/best_critic.pt`. Cost ≈ $1.60.

---

## 6. Transferable lessons

1. **Audit demonstrations before blaming the architecture.** Motion onset past
   the chunk horizon was invisible in the loss curve and caused the whole
   failure.
2. **A monotonically improving metric at the final checkpoint means undertrained,
   not mis-selected.**
3. **Cumulative-displacement thresholds cannot separate drift from motion.** Use
   speed over a sliding window.
4. **Wrist-flex has directional stiction:** positive tracks to ~0.11 units,
   negative stalls 0.9–1.6 short. Always approach from below. Never raise
   current or PID gains.
5. **Overfitting a tiny model is a *positive* pipeline signal.** Failing to
   overfit means something is broken.
6. **Validate on held-out episodes, never held-out frames.** Consecutive frames
   are near-identical and leak.
7. **Baselines must be computed from ground truth.** A baseline that drifts
   during training is being computed from predictions — we shipped exactly that
   bug and caught it mid-run.
8. **With ~5k frames, freeze most of a warm-started model.**

---

## 7. Artifacts

Datasets:

| id | contents |
|---|---|
| `026-place-sponge-on-hot-pad-clean` | 48 clean demos, 13,484 frames |
| `rollout_027-custom48-dagger` | 18 DAgger corrections, 4,229 frames |
| `028-custom48-dagger18` | demos + corrections, 66 eps, 17,549 frames |
| `rollout_030-recap-corrections7` | 7 RECAP corrections, 1,854 frames |
| `031-recap-gen1` | RECAP training set, 18 eps, 6,354 frames |

Models:

| path | what |
|---|---|
| `outputs/modal_custom_act_clean48_chunk30_20k/` | custom ACT base, loss 0.070 |
| `outputs/modal_custom_act_dagger18_ft10k/` | DAgger fine-tune, 9/15 live |
| `outputs/recap_critic/best_critic.pt` | RECAP critic, val MAE 182.8 |

Raw scan JSON lives in the session state directory
(`base_checkpoint_bias_scan.json`, `clean48_checkpoint_scan.json`,
`dagger18_ft_scan.json`, `custom_vs_stock_clean48.json`,
`base_20k_to_50k_bias_scan.json`).

---

## 8. Open items

- RECAP policy trainer is not written: advantage labelling with N=50 lookahead,
  ~40th-percentile threshold, 30% conditioning dropout, corrections forced
  positive.
- Custom-vs-stock closed-loop smoothness difference remains unexplained.
- The RECAP-vs-DAgger comparison holds data budget equal (18 episodes each) but
  RECAP receives fewer expert corrections (7 vs 18) and 11 autonomous episodes
  DAgger structurally cannot use. That asymmetry is the thing being tested and
  should be stated plainly in any writeup.
- Whether to run the ablation adding the 37 older, position-biased rollouts to
  RECAP as a second condition.
