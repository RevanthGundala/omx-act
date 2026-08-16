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
returns. The script (`models/recap/sanity_mlp.py`) has since been removed; it
was a one-off probe, and recovering it means `git log --diff-filter=D --
models/recap/sanity_mlp.py`.

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

A second run on the larger balanced dataset (§5.7) reached **val MAE 86.6**
(+52.1% over its baseline, success AUC 0.815):
`outputs/recap_critic_20k/best_critic.pt`. This is the critic used for
labelling.

### 5.7 Rebalancing the dataset (034)

`031-recap-gen1` was 8 successes / 10 failures. That is a reasonable spread for
learning a value function but a poor one for behaviour cloning, so the training
set was rebuilt to combine everything collected to date.

`034-recap-balanced`: 48 demos + 11 autonomous rollouts + 7 corrections = **66
episodes, 19,437 frames**, 60 successes / 6 failures, C_fail = 476.

### 5.8 Advantage labelling

Labelled with N=50 lookahead against the 20k critic, threshold calibrated on
non-correction frames to the paper's ~40% positive target, corrections forced
positive (Sec. IV-B). Threshold ε_ℓ = **6.49**.

| group | frames | positive |
|---|---|---|
| demonstrations | 13,484 | 40.1% |
| autonomous rollouts | 3,469 | 34.7% |
| correction episodes (all frames) | 2,484 | 79.7% |
| — human-intervention frames | 1,352 | **100%** (forced) |
| — policy frames within those episodes | 1,132 | 55.4% |
| frames from successful episodes | 17,261 | 46.0% |
| frames from failed episodes | 2,176 | **29.4%** |
| **overall** | **19,437** | **44.2%** |

The labels are not noise: failed episodes are 16.6 points less positive than
successful ones, and the calibration target was hit on the frames it was
calibrated over (40.1% / 34.7%). The 44.2% overall is higher only because the
forced-positive correction frames sit on top.

**Deviation, flagged for the writeup:** calibrating the threshold on
non-correction frames only is our reading of App. A-F, which is not explicit on
whether forced-positive frames participate in the percentile.

### 5.9 The RECAP policy ignores its advantage token

Trained the advantage-conditioned policy on all 66 episodes, warm-started from
the custom ACT base, 20k steps, lr 1e-5, `train_split=1.0`, no image transforms
(matched to the DAgger fine-tune for comparability). Loss fell 0.070 → 0.046.

Training succeeded. **The resulting policy does not use its conditioning
token**, which makes it behaviour cloning on a mixed dataset rather than an
advantage-conditioned policy. Nothing in the loss detects this;
`models/recap/check_token_sensitivity.py` was written to catch it and did.

The check flips the token on identical observations and measures how far the
predicted actions move, expressed as a fraction of how far they move when the
*observation* changes. All values in standard deviations of the action
distribution, 150 batches:

| model | trained on | eval frames | sensitivity |
|---|---|---|---|
| baseline | all 66 eps (19,437 fr) | 48–65 | **0.93%** |
| arm A | rollouts + corrections (5,953 fr) | 48–65 | **0.98%** |
| arm B | rollouts only (3,469 fr) | 48–58 | **0.96%** |

Pass threshold is 5%. All three fail, and in all three the residual effect
points the **wrong way** on positive-labelled frames.

Ablations that changed nothing:

| variation | result |
|---|---|
| conditioning dropout 0.3 → 0 | 0.137% → 0.138% internal influence; loss 0.0555 → 0.0549 |
| 10k vs 20k steps | influence 0.137% → 0.123% (*decreasing*) |
| vs. random initialisation | 0.147% — training **reduced** token influence |
| shrink dataset 5.6× (arms A/B) | 0.93% → 0.98% / 0.96% |

Ruled out as causes: label/frame misalignment (verified exact 1:1), one-hot
construction, dead token weights (embedding norm difference 10.27), conditioning
dropout, class balance, and dataset composition.

**Where the signal dies.** Tracing one token flip through the network:

```
11.17   at the token's own encoder slot
 0.065  leaked to the other 602 tokens
 0.0008 at the output
```

The token is loud where it enters and the transformer simply does not route it.

**Why.** For a conditioning token to reduce the loss, the label must change the
optimal action *given the observation* — i.e. `E[a | o, POS] ≠ E[a | o, NEG]`.
A matched-neighbour test (each frame's 40 nearest neighbours in normalised state
space, different episodes only; null = within-episode circular label shift)
measures exactly this:

| subset | frames | effect | z | p |
|---|---|---|---|---|
| all 66 eps | 19,437 | +2.86% | 3.82 | 0.005 |
| demonstrations | 13,484 | +3.53% | 4.21 | 0.005 |
| autonomous rollouts | 3,469 | −1.25% | −0.83 | 0.408 |
| corrections | 2,484 | −3.11% | −1.89 | 0.055 |
| rollouts + corrections | 5,953 | −0.17% | −0.62 | 0.562 |

Even where significant, the effect is ~3% — against a policy already predicting
actions to ~1 motor unit, that is roughly 0.03 units of headroom. Gradient
descent correctly assigns the token almost no influence.

Our policy is near-deterministic given an observation, and the advantage is
driven largely by *sponge placement*, which the instantaneous action does not
respond to. So the same action is labelled positive in one episode and negative
in another. The label varies; the action does not.

**What we know and what we do not.** We know **that** the real-label token is
inert, across every ablation above. We do **not** have a single proven cause.
Hypotheses still live: the critic labels do not change the action target given
the observation; the CVAE `z` already carries the action during training, so the
token is redundant; chunk dilution. An earlier draft of this section asserted
that ACT is unimodal and therefore *cannot* condition — §5.10 refutes that, and
the claim is withdrawn.

We also cannot claim, on this evidence, that PI's method requires matched
good/bad trajectories from the same state. That is a hypothesis we have not
tested, not a finding.

The remaining untested option is AWR-style advantage *weighting*, which changes
the loss rather than the input and so does not require the label to explain
residual variance.

### 5.10 Test A: the trainer can learn a token

To separate "this trainer cannot learn *any* token" from "our labels are not a
token task," we discarded the critic labels and planted a bit that is
**required** to reduce L1: a deterministic coin flip per frame, plus +0.5
z-units on every joint when the bit is POSITIVE. Same architecture, same warm
start, 10k steps, dropout 0. This is a capability probe, **not** RECAP — RECAP
still clones whatever action was taken and only hopes the tag lines up with
better actions.

| | planted | measured after 10k |
|---|---|---|
| token effect (POS − NEG) | 0.50 | **0.519** |
| relative to changing the scene | — | **1387%** |
| direction | POS matches the shifted target | **PASS** |

App `ap-Q5o55iTBj4ZyBqgDUVys8X`, A10, ~$0.64. Checkpoint
`recap_policy_synth_shift05_10k/20260815-205025/step_010000.pt`.

So the wiring works: when using the token cuts the loss, this trainer uses it
almost exactly. The 0.93% real-label result is therefore **not** "ACT cannot
condition." What remains open is why *these* labels do not create that
incentive.

### 5.11 The all-positive SFT also ignores its token

`recap_sft_positive_10k` trained on the 48 demonstrations with every indicator
fixed to POSITIVE and no conditioning dropout.

Reading that checkpoint needed an architecture that can *omit* the token rather
than substitute a third value. `models/recap/act_recap_masked.py` is that
architecture: `nn.Embedding(2, d_model)` over NEGATIVE and POSITIVE only, with a
`MASKED_ADVANTAGE` sentinel whose slot is masked out of encoder self-attention
and decoder cross-attention. It is a separate file; the legacy model is
untouched. Its parameter layout differs deliberately, so a legacy checkpoint is
ported with `convert_legacy_state_dict`, which sets each embedding row to the
legacy `adv_proj` column plus bias and reproduces the legacy outputs exactly for
visible tokens. That parity is asserted before any masked number is reported.

On 256 identical observations, removing the POSITIVE token had essentially no
causal effect:

| checkpoint | visible-versus-masked output effect |
|---|---:|
| SFT step 2k | **0.04%** of observation-driven variation |
| SFT step 10k | **0.04%** |

Encoder attention to the advantage slot was approximately uniform (1.12× uniform
at 2k, 1.01× at 10k); decoder cross-attention was 0.27× and 0.29×. A few heads
attended more strongly, yet masking the slot still did not move the actions — so
attention maps here are descriptive, not evidence that the condition affects
behaviour. Attention maps are session artifacts, not repo files.

This means the SFT's 82% live result should be read as ordinary fine-tuning with
a constant, unused token — not as evidence that all-positive SFT pretrained a
useful advantage channel.

### 5.12 Two measurement errors worth recording

**Subset evaluation inflated the effect.** Arm A first measured 1.45% against a
0.75% baseline — apparently a near-doubling. Running the *baseline* on the same
episode subset gave 1.52%, higher than arm A. The gain came from evaluating on a
narrower slice, not from training on one. Any conditioning-strength number is
only interpretable against a control measured on identical frames.

**The check under-sampled.** Its default of 16 batches is 128 frames read
sequentially, so two different episode subsets returned identical numbers to
five decimals — both were reading the start of episode 48. At 150 batches the
baseline moved 1.52% → 0.93%. **The earlier 0.75% figure should not be quoted;
use 0.93%.**

### 5.13 Training from scratch: the model learns to ignore the token

Every earlier RECAP run warm-started from a policy that had already converged
without a token, which left one objection open: perhaps the token was inert
because it arrived too late to matter, not because it was unusable.

`recap_masked_scratch_2k` closes that. Random init, no warm start, token present
from step 0, 2,000 steps on `031-recap-gen1`, 30% omission, chunk 30 and
train_split 1.0 to match the baseline recipe. A10G, 7m33s, ~$0.14. The trainer
logs attention on the advantage slot from the weights the forward pass already
cached, so these are the training steps themselves rather than a later probe.

The architecture question first: the masked model differs from the base custom
ACT by exactly two tensors, `decoder.advantage_embedding.weight` (2x512) and
`decoder.adv_pos_emb.weight` (1x512) — 1,536 of 52,350,854 parameters, or
0.0029%. Every one of the base's 239 tensors has an identically-shaped
counterpart. Warm start is therefore a scientific choice, not a compatibility
one.

**Both mechanisms work.** Omission was 32.5% over 640 sampled examples
(z = +1.38 against the 30% target). Attention on an omitted slot was
**exactly 0.000e+00** at all 79 logged steps, in encoder self-attention and
decoder cross-attention alike — absence, not a small number.

**And the model still trained itself out of using the token.** Memory is 603
slots, so uniform attention is 1/603 = 1.658e-03:

| steps | decoder attention on the slot | versus uniform |
|---|---:|---:|
| 0-500 | 3.687e-03 | **2.22x** |
| 500-1000 | 2.045e-03 | 1.23x |
| 1000-1500 | 1.883e-03 | 1.14x |
| 1500-2000 | 1.492e-03 | **0.90x** |

It starts *above* uniform and decays to *below* it, a 59.5% fall, while action
loss drops 0.863 to 0.142. The token was available, was initially attended, and
was then actively abandoned.

That is the predicted outcome rather than a surprise: when every visible token
is POSITIVE and all examples share the same expert target, ignoring the token is
loss-minimising. The value is in ruling out the alternative explanation. The
token is not inert because it was introduced late — it is inert because
all-positive SFT gives it nothing to encode. Any conditioning signal has to come
from labels that vary with behaviour.

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
9. **A conditioning input only helps if it changes the optimal output given the
   other inputs.** Class balance is irrelevant for an input — appending a
   perfectly balanced coin flip to every observation teaches a model nothing.
   Check `E[a | o, c=1] ≠ E[a | o, c=0]` in the data *before* training on it.
10. **Measure conditioning strength against a control on identical frames.**
    Evaluating on a narrower slice inflates the apparent effect; we briefly
    believed a subset had doubled token use when the baseline scored higher on
    the same slice.
11. **A diagnostic with a small default sample can return identical numbers for
    different data.** Ours read 128 sequential frames, so two different subsets
    both measured the start of the same episode.
12. **To test whether a conditioner *can* be learned, plant a bit the loss
    requires.** Test A is not RECAP; it only asks whether the trainer will use a
    token when ignoring it is impossible. Ours does (0.519 against a planted
    0.50), which rules out "the trainer cannot condition" without identifying
    what is wrong with the real labels.
13. **Omitting a conditioning token is not the same as setting it to a third
    value.** A learned UNCONDITIONED embedding is still a token the encoder
    attends to. Testing real absence needed a separate architecture that masks
    the slot.

---

## 7. Artifacts

Datasets:

| id | contents |
|---|---|
| `026-place-sponge-on-hot-pad-clean` | 48 clean demos, 13,484 frames |
| `rollout_027-custom48-dagger` | 18 DAgger corrections, 4,229 frames |
| `028-custom48-dagger18` | demos + corrections, 66 eps, 17,549 frames |
| `rollout_030-recap-corrections7` | 7 RECAP corrections, 1,854 frames |
| `031-recap-gen1` | RECAP critic set, 18 eps, 6,354 frames, C_fail 704 |
| `034-recap-balanced` | RECAP policy set, 66 eps, 19,437 frames, C_fail 476 |

Models:

| path | what |
|---|---|
| `outputs/modal_custom_act_clean48_chunk30_20k/` | custom ACT base, loss 0.070 |
| `outputs/modal_custom_act_dagger18_ft10k/` | DAgger fine-tune, 9/15 live |
| `outputs/recap_critic/best_critic.pt` | RECAP critic (gen1), val MAE 182.8 |
| `outputs/recap_critic_20k/best_critic.pt` | RECAP critic (034), val MAE 86.6 |
| `recap_policy_balanced66_20k` | RECAP policy, token sensitivity 0.93% (fails) |
| `recap_policy_rollcorr_10k` | arm A, rollouts + corrections, 0.98% |
| `recap_policy_rollonly_10k` | arm B, rollouts only, 0.96% |
| `recap_policy_synth_shift05_10k` | Test A, planted +0.5 shift, token effect 0.519 **PASS** |

Raw scan JSON lives in the session state directory
(`base_checkpoint_bias_scan.json`, `clean48_checkpoint_scan.json`,
`dagger18_ft_scan.json`, `custom_vs_stock_clean48.json`,
`base_20k_to_50k_bias_scan.json`).

---

## 8. Open items

- **The RECAP-vs-DAgger success comparison is not yet valid.** The RECAP SFT
  eval ran at sponge y ≈ 0.247 and the DAgger eval at y ≈ 0.348 — distributions
  that barely overlap. The camera is stable (the fixed hot pad reads y =
  0.468–0.469 across all sessions), so this is genuine placement variation, not
  setup drift. Within existing data sponge y only weakly predicts success
  (Mann-Whitney p = 0.121), but the fix is to re-evaluate the DAgger checkpoint
  at the planned positions for a paired comparison. **This blocks the writeup.**
- AWR-style advantage weighting is the one mechanism §5.9 does not rule out.
- The cause of the real-label token being inert is still unidentified. §5.10
  rules out "the trainer cannot learn a token" and §5.13 rules out "the token was
  introduced too late"; neither says which of the remaining hypotheses is right.
- Masked all-positive SFT has now been run from scratch (§5.13). Masking and
  omission are verified working, and the model still abandons the token, so the
  remaining question is whether *varying* labels change that. The natural next
  run is masked training on real critic labels rather than all-POSITIVE.
- Custom-vs-stock closed-loop smoothness difference remains unexplained.
- The RECAP-vs-DAgger comparison holds data budget equal (18 episodes each) but
  RECAP receives fewer expert corrections (7 vs 18) and 11 autonomous episodes
  DAgger structurally cannot use. That asymmetry is the thing being tested and
  should be stated plainly in any writeup.
- Whether to run the ablation adding the 37 older, position-biased rollouts to
  RECAP as a second condition.
