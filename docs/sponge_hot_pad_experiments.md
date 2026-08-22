# Sponge → hot pad: experiment log

Everything we ran on the "Place the sponge on the hot pad" task, in order, with
the numbers we actually measured. Anything not measured is marked as such.

The through-line: a behaviour-cloned ACT policy failed, we found *why*, and
fixed the **data** rather than the model. DAgger then helped. RECAP's
advantage token never fired on this task (`I(a ; I | o)` ≈ 0), so every
"RECAP" live number is behaviour cloning with a sticker. DAgger and RECAP
were **not** evaluated on a matched sponge grid — that comparison is still
open.

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

### 5.13 Mechanism check: masking and dropout work, and the token still decays

**Scope.** This is a mechanism check, not a training result, and it should not be
quoted as one. Three limits, all of which were chosen for speed and none of which
were noticed before the run:

- It trained on `031-recap-gen1`, which is the **critic** set (8 successes, 10
  failures), not the 48 demonstrations. Forcing every label POSITIVE there labels
  ten failed episodes as good, so this is not a coherent SFT stage. The earlier
  `recap_sft_positive_10k` of section 5.11 used the demonstrations, so the two
  are **not** a matched pair and their numbers should not be read as replicates.
- `train_split=1.0`, so no validation set exists and every number below was
  measured on data the model trained on. On training data the observation alone
  largely determines the action, which is the regime most favourable to finding a
  conditioning token redundant. The bias runs toward the conclusion drawn.
- 2,000 steps against the 20,000 the baseline recipe uses.

What survives all three is the part that does not depend on the data: a constant
label carries zero mutual information with anything, so "the token is ignored
because every visible label is identical" holds whatever it was trained on.

**Setup.** `recap_masked_scratch_2k`. Random init, no warm start, token present
from step 0, 30% omission, chunk 30, lr 1e-5. A10G, 7m33s, ~$0.14. The trainer
logs attention on the advantage slot from the weights the forward pass already
cached, so these are the training steps themselves rather than a later probe.

**The architecture is all but shared with the baseline.** The masked model
differs from the base custom ACT by exactly two tensors,
`decoder.advantage_embedding.weight` (2x512) and `decoder.adv_pos_emb.weight`
(1x512) — 1,536 of 52,350,854 parameters, or 0.0029%. Every one of the base's 239
tensors has an identically-shaped counterpart, so warm start is a scientific
choice rather than a compatibility one.

**Both mechanisms work.** Omission was 32.5% over 640 sampled examples (z = +1.38
against the 30% target). Attention on an omitted slot was **exactly 0.000e+00**
at all 79 logged steps, in encoder self-attention and decoder cross-attention
alike — absence, not a small number. That is the claim this run does establish,
and it is a claim about the implementation rather than about RECAP.

**The token decayed during training.** Memory is 603 slots, so uniform attention
is 1/603 = 1.658e-03:

| steps | decoder attention on the slot | versus uniform |
|---|---:|---:|
| 0-500 | 3.687e-03 | **2.22x** |
| 500-1000 | 2.045e-03 | 1.23x |
| 1000-1500 | 1.883e-03 | 1.14x |
| 1500-2000 | 1.492e-03 | **0.90x** |

It starts *above* uniform and decays to *below* it, a 59.5% fall, while action
loss drops 0.863 to 0.142. The token was available, was initially attended, and
was then abandoned.

On the final checkpoint, evaluated on the training data with the caveat above,
the advantage slot ranked **309 of 603** by attention received (1.100e-03 against
a median slot's 1.112e-03) while the joints slot ranked **1** at 143.50x uniform.
So the architecture concentrates hard on a slot that carries something, which
removes "a transformer cannot single out one token" as an explanation. Masking
the token moved the actions by 0.11% of observation-driven variation.

**What this does and does not show.** It shows the masking and dropout
implementation is correct, and it removes one competing explanation for the
inert token: the token was present from step 0 here, so "it was introduced too
late to matter" is not why. It does not show anything about RECAP, because RECAP
conditions on labels that vary and every visible label here was POSITIVE. A
near-zero effect is what an all-positive run predicts, not a finding about the
method.

### 5.14 The advantage is the critic's residual, and the critic is too coarse

Two independent model reviews (GPT-5.6 Sol, Grok 4.6) converged on a hypothesis
this log had under-weighted: the labels may be critic noise rather than weak
signal. Every specific claim either review made about this repo was checked and
held.

**The identity.** By definition of the return, `R_t = sum(r[t:t+N]) + R_{t+N}`.
Substituting into the advantage gives

```
A_t = sum(r[t:t+N]) + V(o_{t+N}) - V(o_t)
    = [V(o_{t+N}) - R_{t+N}] - [V(o_t) - R_t]
    = residual_{t+N} - residual_t
```

The advantage is **exactly** the difference of the critic's residuals, for any
critic. A critic that predicted returns perfectly would make every advantage
identically zero — the same collapse as the CVAE-latent proof in section 5.2.
Asserted numerically in `tests/recap/test_recap_models.py`.

This is not a defect. Part of the residual is real signal, because the future is
stochastic and `V(o)` predicts its expectation. But it makes the critic's
accuracy the thing that decides whether the binary label carries information.

**The bound.** The labels every policy run has used came from
`outputs/recap_critic_20k/best_critic.pt`, trained on `034-recap-balanced` —
which contains **60 successes and 6 failures**. Its val return MAE is 86.6,
implying a residual std of ~108.5 return units, against a resolved threshold of
eps_l = **6.49**. The threshold is 17x smaller than the critic's own error.

Residual correlation across the lookahead is what could rescue this: errors that
persist from `t` to `t+50` cancel in the subtraction. Running
`models/recap/advantage_noise_bound.py`:

| residual correlation across N=50 | advantage noise std | vs threshold |
|---|---:|---:|
| 0.0 | 153.5 | 23.7x |
| 0.9 | 48.5 | 7.5x |
| 0.99 | 15.4 | 2.4x |
| 0.999 | 4.9 | 0.7x |

For the noise to fit inside the threshold, the critic's residuals would have to
be correlated at **rho > 0.998** across 50 steps — 1.7 seconds at 30 Hz.

**Status: bound, not measurement.** The actual correlation has not been measured,
and it is the cheapest way to settle this. If it is below ~0.998, the binary
labels are substantially coin flips, and every "the token is inert" result is
reinterpreted: the token would be routing noise, not failing to route signal.
That is a different finding from "these labels have no action-conditional
content," and this log should not assert the latter until the former is excluded.

**A protocol error found in the same pass.** Appendix A specifies two separate
calibrations: ~30% positive on *demonstration* data for pre-training, ~40% on
*evaluation rollouts* for fine-tuning. `label_advantage.py` resolves a single
threshold at the 40% target over `mask=~intervention`, which pools demos and
rollouts. Demonstrations are ~80% of that pool, so the threshold is set largely
by the demonstration distribution and then applied to rollouts, which received
34.7% positive against the paper's 40%. The module docstring states the correct
rule; the implementation does not follow it.

**Two claims in this log are withdrawn as overstated:**

- Section 5.10 does not show "the trainer can learn a token." It planted 0.5
  z-units on every joint, a 1387% relative effect, against a real signal of
  1-3%. It rules out an inability to learn a *large* token. A dose-response at
  0.03 and 0.05 would say whether a realistically small one is learnable at all.
- The attention decay in section 5.13 is not evidence about RECAP. Every visible
  label in that run was POSITIVE, and a constant carries no information, so
  abandoning it is correct behaviour rather than a finding.

**AWR is withdrawn as the promising remaining option.** Both reviews derived the
same result independently: L1 advantage-weighted regression at a fixed
observation is the *weighted median* of `a | o`. If the label does not change the
action distribution given the observation, the weighted median equals the
unweighted one and AWR reduces to behaviour cloning. What it can still change is
the distribution *over* observations, which on this data upweights easy sponge
placements — the same confound that invalidates the 82%. It is a control, not a
fix.

### 5.15 The critic is coarse, but that is not why the token is inert (2026-08-18)

Four checks asked whether a bad critic is a sufficient explanation. They were
run on `034-recap-balanced` (19,437 frames, 66 episodes) and the labels already
used for every policy run (`eps_ℓ = 6.49`, interventions forced POSITIVE).
Continuous `A` was never saved in `advantage_labels.json`; the critic-feature
job below recomputed it.

**The algebraic form.** Every non-terminal reward is exactly −1
(`recap_rewards`). On any frame with a full lookahead, `R_t − R_{t+50}` is
therefore identically −50 — that difference is the 50-step reward *sum*, not a
baseline. Substituting into `n_step_advantages` gives

```
A_t = −50 + V(o_{t+50}) − V(o_t)
```

on bootstrapped frames (~83%). The executed action does not appear. A perfect
critic makes `A ≡ 0` (the collapse already proven for a future-conditioned
critic in §5.2). A noisy critic makes `A` a noisy function of two observations.
There is no critic accuracy at which this estimator becomes action-conditional.

On the last 50 frames the bootstrap is dropped and `A_t = R_t − V(o_t)`, so
the terminal 0 / −476 can enter. That is the only slice where episode outcome
is in the label by construction.

**1. The binary labels are not collapsed.** 8,586 POS / 10,851 NEG = **44.2%**
POS. Non-intervention frames are exactly 40.0% by construction; all 1,352
intervention frames are POS by rule. Failures 29.4% POS vs successes 46.0%.
Median POS run length is 2 frames, but 69% of POS frames sit in runs ≥ 10.
Last-50 / MC-tail frames are 77.3% POS (success 78%, failure 69%); first-50 of
failures are 14% POS vs 52% for successes. The bit flickers and leans POS at
the end of almost every episode. It is not a single class.

**2. `I` is partly already in the observation.** Frozen critic encoder, tiny
head, interventions excluded, 8 held-out episodes / 2,176 frames
(`recap_probe_i_from_o`, A10, `ap-GyPIZgN8qFA1U8rnawA6HX`):

| probe | held-out AUC | acc (majority 0.58) |
|---|---:|---:|
| vision + qpos MLP | **0.780** | 0.727 |
| vision + qpos linear | 0.739 | 0.688 |
| qpos MLP | 0.655 | 0.658 |
| qpos linear | 0.576 | 0.585 |
| critic scalar `V(o)` | **0.523** | 0.600 |

The cameras predict the bit. The critic's own number does not. That is what
`A = −50 + ΔV` says: `V(o_t)` is the wrong object; progress over 1.7 s is
partly visible in the picture. AUC 0.78 is not chance and not a deterministic
function of `o`.

**3. High-A episodes succeed more and are not faster.** Non-forced POS tertiles
(22 / 22 / 22): top **22/22** success, bottom **17/22**. Success vs fail
non-forced POS 0.423 vs 0.299 (pooled AUC 0.817, MW p=0.004). Autonomous-only
(5 vs 6) AUC 0.767, p=0.089. **N_fail = 6**, all rollouts — directional only.
Among 60 successes, Spearman(POS, steps) = +0.18 (p=0.18): higher POS is not
quicker. Forced corrections inflate episode POS from 0.57 to 0.80.

**4. Held-out L1 does not move when the token is ablated.**
`recap_policy_balanced66_20k` step 20k, 13 held-out episodes / 3,455 frames,
`train_split=0.8` (`ap-5aKgs4L5ms4d4sPOOhmRpe`). Five tokens on identical
batches; CVAE `z = 0` because `model.eval()`:

| token | L1 | Δ% vs true |
|---|---:|---:|
| true label | 0.051330 | — |
| flipped | 0.051331 | **+0.00%** |
| all POS | 0.051366 | +0.07% |
| all NEG | 0.051296 | −0.07% |
| unconditioned | 0.051357 | +0.05% |

Largest |Δ%| = **0.07%**. The earlier 0.93% action-sensitivity was on training
data; this is the same conclusion on a holdout, in the actual training loss.
Open item "re-measure token effect on held-out episodes" is closed.

**5. `I(a ; I | o)` at similar scenes is barely above the null.** Cross-episode
k=40 neighbours, circular-within-episode label shift (200 perms), interventions
excluded. Joints-only is the known-bad control (sponge pose lives in the
images). Critic-encoder cosine space is the real matcher
(`recap_action_label_mi`, `ap-gvf2y0IwmhxkjuL3BunbRz`; first launch
`ap-ldCrHFm0ynvkNmwLPhHFt9` died on `ACTConfig(**ckpt["config"])` rejecting
`dataset_format` — use the probe's field-filtering loader):

| matcher | lift | rel | z | p |
|---|---:|---:|---:|---:|
| qpos (control) | +0.0027 | +1.07% | 1.08 | **0.30** |
| critic features | +0.0103 | +2.06% | 2.26 | **0.035** |

Opposite-label neighbours are 2% farther in action space than same-label ones.
That is statistically non-null and practically empty — the same scale as the
old joints+chunk test's ~3%, now with cameras, and far too small to move L1.

The same job recomputed continuous `A`: mean 0.57, **std 48.3**, median 1.53,
10–90% = [−33, +34]. Threshold ε=6.49 sits inside one blob, not between two
modes. Std 48.3 matches the §5.14 noise-bound row for residual correlation
ρ≈0.9 (48.5). The labels are noisy *and* mostly a function of `o`.

**Verdict.** "The critic isn't good enough" is partly true and not the cause of
the inert token. The bit is balanced, weakly ranks success, and is already in
the cameras. Flipping it does not change this policy's loss. A better critic
under the same reward still produces an observation-only label. What remains
is whether a multimodal policy could use a 2% residual, and the matched robot
eval that still blocks the writeup.

Scripts: `models/recap/probe_advantage_from_obs.py`,
`models/recap/check_token_loss.py`, `models/recap/check_action_label_mi.py`.

### 5.16 Live eval of RECAP-ACT as implemented (2026-08-20)

Rolled out `recap_policy_balanced66_20k` (exported `--recap`, inference token
POSITIVE) with the same async stack as the DAgger eval: `eval.async_lerobot`,
`--policy-type custom_act`, `--max-joint-delta 40`, `--actions-per-chunk 30`.
`--start-state-wrist-flex-offset 0` because 034's start wrist-flex support is
`[46.28, 48.38]` and the default +2 offset is out of range.

Session `outputs/eval_rollouts_recap_balanced66/act_eval_async_20260820-111513`,
15 operator-ended trials, all labeled:

| trial | outcome | duration (s) |
|---|---|---:|
| 1–3 | fail | 14.6 / 13.2 / 13.6 |
| 4–8 | success | 12.5 / 13.4 / 12.9 / 8.8 / 8.9 |
| 9 | success | 43.1 |
| 10–12 | fail | 11.5 / 8.0 / 8.3 |
| 13 | success | 13.3 |
| 14 | fail | 23.4 |
| 15 | success | 25.5 |

**8/15 = 53%.** Pattern FFF SSSSS FFF S F S.

Five sponge sites × three tries (top-camera `sponge_xy_norm` from scene check):

| site | trials | xy (approx) | success |
|---|---|---|---|
| right-near | 1–3 | (0.83, 0.32) | **0/3** |
| right-mid | 4–6 | (0.78, 0.36) | **3/3** |
| center-near | 7–9 | (0.54, 0.28) | **3/3** |
| left-near A | 10–12 | (0.27, 0.30) | **0/3** |
| left-near B | 13–15 | (0.22–0.27, 0.15–0.23) | **2/3** |

Macro over sites: (0+1+1+0+2/3)/5 = **53%**, same as micro. Two sites are never solved.

This is RECAP-ACT *as implemented*: warm start from the custom ACT base, 20k
steps on 034 (48 demos + 11 autonomous + 7 RECAP corrections), mixed labels,
30% dropout. It is not all-positive SFT, and it is not DAgger-18 (18 expert
corrections, no token). Because the token is inert (§5.15), the live policy is
behaviour cloning on 034 with an unused POSITIVE sticker.

Do not read 8/15 vs DAgger 9/15 as a method comparison. Placements were not
matched (same confound that invalidates the withdrawn 82%). The two recipes
also do not share a correction budget (7 vs 18). Fisher on 8/15 vs 9/15 is
p ≈ 1; vs base 4/22 is p ≈ 0.035 — same qualitative pattern as DAgger vs
base, with the same unmatched-grid asterisk.

### 5.17 Mix ablation live: 7 SPACE vs 4 auto + 3 SPACE (2026-08-21)

Same RECAP-ACT init (`balanced66_20k`), same extra **episode** count (7),
same 5×3 tape grid as §5.16. Critics FT 4k freeze-encoder, labels from
**step_001000** (A val MAE 68, B 51). Policies 10k from `balanced66`
(A checkpoint is **8k**: no `step_010000` was written).

**A** `036-recap-corr7` (7 SPACE only). Session
`outputs/eval_rollouts_036_corr7/act_eval_async_20260821-102055`:

| trials | outcomes |
|---|---|
| 1–2 | F F |
| 3–7 | S S S S S |
| 8 | F |
| 9–15 | S S S S S S S |

**12/15 = 80%** after operator correction: trials 3–5 were successes
mislabeled F at the prompt (same class of error as §5.16 auto t3). Pattern
FF SSSSS F SSSSSSS. Scene-check buckets were not written (`sponge_bucket`
null). Same five tape marks as §5.16.

**B** `037-recap-mix7` (4 autos + 035 ep0/3/6). Session
`outputs/eval_rollouts_037_mix7/act_eval_async_20260821-112425`:

FFFFFFS FFFFFF SS → **3/15 = 20%** (success on 7, 14, 15). Same 5×3 as A.

| | extra | live | token |
|---|---|---|---|
| parent `balanced66` | — | 8/15 = 53% | inert (§5.15) |
| **A** 7 SPACE | 7 corr | **12/15 = 80%** | not re-measured |
| **B** 4 auto + 3 SPACE | mix | **3/15 = 20%** | still inert |

On this **matched 5×3**, extra **corrections** beat extra **on-policy fails + fewer corrections**. B is worse than the parent.

**Not a token result.** B `step_010000`, 128 train frames, mask-vs-visible output effect **0.03%** of observation effect; advantage slot rank **591 / 603** (0.29× uniform); joints slot rank **1 / 603** (74×). Same diagnosis as the parent: the transformer can concentrate on a slot; it does not on this one.

**Train loss does not explain 12/15 vs 3/15.** Logged every 100 steps; A never wrote `step_010000` (last 8k). At 1k-step marks:

| step | A loss | B loss |
|---:|---:|---:|
| 0 | 0.072 | 0.056 |
| 2000 | 0.051 | 0.044 |
| 6000 | 0.043 | **0.059** |
| 8000 | **0.040** (eval ckpt) | 0.045 |
| last | 0.049 @ 9.5k | **0.054** @ 9.9k |

A actually fitted (0.072 → 0.040). B ended where it started. Last-1k means are similar (~0.047 vs ~0.045). The live gap is the mix: A cloned seven successful takeovers; B cloned three of those plus four on-policy rollouts, three of which are failures at the hard sites.

Do not put A 12/15 in a RECAP-vs-DAgger table. The token did not wake up. This is DAgger-shaped data inside a RECAP-shaped trainer.

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
14. **An n-step advantage with step reward −1 is −N plus a value difference.**
    `R_t − R_{t+N}` is the window sum, not a baseline. On this reward it is a
    constant, so `A` has no action term on bootstrapped frames. Improving `V`
    cannot invent `I(a ; I | o) > 0`.
15. **Match neighbours in the observation the policy sees.** Joint-angle kNN
    ignores sponge pose. A 6-joint "matched" test is not `I(a ; I | o)`.
16. **Flip the real token and measure held-out L1**, not only action
    displacement on the train set. Displacement can look non-zero while the
    loss does not care; the reverse bias also exists. We needed both, on a
    holdout.
17. **Cloning extra on-policy failures can tank live success while train loss
    looks fine.** Same init, same extra episode count, same 5×3: 7 SPACE
    takeovers → 12/15; 4 autos (3 fail) + 3 SPACE → 3/15, worse than the
    parent. The token stayed unused, so this is a data-mix result, not
    advantage conditioning.

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
| `rollout_035-recap-balanced-dagger` | 7 SPACE takeovers on parent, all success |
| `036-recap-corr7` | 034 + 7 SPACE, 73 eps, 22,195 fr, 67/6 (local, not on HF) |
| `037-recap-mix7` | 034 + 4 autos + 3 SPACE, 73 eps, 22,018 fr, 64/9 (local) |

Models:

| path | what |
|---|---|
| `outputs/modal_custom_act_clean48_chunk30_20k/` | custom ACT base, loss 0.070 |
| `outputs/modal_custom_act_dagger18_ft10k/` | DAgger fine-tune, 9/15 live |
| `outputs/recap_critic/best_critic.pt` | RECAP critic (gen1), val MAE 182.8 |
| `outputs/recap_critic_20k/best_critic.pt` | RECAP critic (034), val MAE 86.6 |
| `recap_policy_balanced66_20k` | RECAP-ACT as implemented, token inert; live **8/15 = 53%** |
| `recap_policy_036_corr7_ft10k` | mix A, last=8k; live **12/15 = 80%** (token not re-measured) |
| `recap_policy_037_mix7_ft10k` | mix B, 10k; live **3/15 = 20%**; token 0.03% causal |
| `recap_policy_rollcorr_10k` | arm A, rollouts + corrections, 0.98% |
| `recap_policy_rollonly_10k` | arm B, rollouts only, 0.96% |
| `recap_policy_synth_shift05_10k` | Test A, planted +0.5 shift, token effect 0.519 **PASS** |
| `recap_probe_i_from_o` | I-from-o probe, vision+qpos MLP AUC 0.780 |
| `recap_token_loss/20260818-042125` | held-out token L1, max \|Δ\| 0.07% |
| `recap_action_label_mi/20260819-010846` | critic-space I(a;I\|o), lift +2.06%, p=0.035 |

Raw scan JSON lives in the session state directory
(`base_checkpoint_bias_scan.json`, `clean48_checkpoint_scan.json`,
`dagger18_ft_scan.json`, `custom_vs_stock_clean48.json`,
`base_20k_to_50k_bias_scan.json`).

---

## 8. Open items

- **RECAP-vs-DAgger is still unmatched.** Live numbers: base 4/22, DAgger-18
  9/15, parent RECAP-ACT 8/15 (§5.16), mix A 12/15 / mix B 3/15 on the
  **same 5×3 as the parent** (§5.17). The old SFT 82% stays withdrawn. DAgger
  was not run on that grid (82% eval y ≈ 0.247, DAgger y ≈ 0.348). **That
  still blocks a method table.** Re-eval DAgger-18 on the same cells. Do not
  collect another RECAP iteration hoping the token starts working.
- AWR-style advantage weighting is withdrawn (§5.14): L1-AWR at fixed `o` is
  the weighted median, which equals BC when `I` does not move `a | o`.
- **The inert token is no longer "unidentified."** §5.15: labels are balanced
  and weakly rank success; `I` is readable from the cameras (AUC 0.78) not from
  `V(o)` (0.52); flipping the token moves held-out L1 by ≤0.07%; critic-space
  neighbours show only a +2% action lift (p=0.035). The bit is mostly a
  function of `o`. Remaining live questions are L1 unimodality / paired
  actions at the same visual state, a 0.03 dose-response (failure branch
  only), and missing CFG (cannot be the cause of a 0.07% effect).
- Masking and conditioning dropout are verified correct (§5.13). The
  all-POSITIVE masked runs are not evidence about varying labels; §5.15's
  flip/all-POS/all-NEG L1 table is.
- **The CVAE-latent hypothesis is dead** (`docs/act_debugging_learnings.md`,
  2026-08-17). `zero_latent` is a control, not a diagnostic.
- **Held-out token L1 is measured** (§5.15): 13 episodes, 3,455 frames, max
  |Δ| 0.07%. The older 0.93% figure was train-set action displacement and
  should not be quoted alone.
- Custom-vs-stock closed-loop smoothness difference remains unexplained.
- The RECAP-vs-DAgger comparison holds data budget equal (18 episodes each) but
  RECAP receives fewer expert corrections (7 vs 18) and 11 autonomous episodes
  DAgger structurally cannot use. That asymmetry is the thing being tested and
  should be stated plainly in any writeup.
- Whether to run the ablation adding the 37 older, position-biased rollouts to
  RECAP as a second condition.
