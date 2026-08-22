# The advantage token never fired

We implemented RECAP (Physical Intelligence, [arXiv:2511.14759](https://arxiv.org/abs/2511.14759)) on a real OpenMANIPULATOR-X: 6 joints, 30 Hz, top + wrist cameras, task = put a sponge on a **fixed** hot pad. The paper's move is an advantage token — a POS/NEG bit on every transition — so a policy can train on mixed-quality robot data instead of expert corrections only.

The architecture learned the token when we **planted** a signal it needed. On the real critic labels it ignored the slot completely. Every live "RECAP" number below is therefore behaviour cloning with a sticker, not advantage conditioning.

## First we thought the model was broken

Custom ACT vs stock LeRobot looked like a training bug: offline L1 was low, the arm missed the sponge. It was the demonstrations. In 11 of 48 episodes the reach started **after** ACT's chunk of 30, so the policy was trained to hold still from the start state. Wrist-flex start pose was also all over the place.

We fixed collection (canonical reset, motion trigger, coverage, seat wrist-flex from below — never raise current or PID). Clean 48 demos:

| policy | live success | grid |
|---|---|---|
| custom ACT | 4/22 = 18% | unmatched |
| + DAgger-18 expert corrections | 9/15 = 60% | unmatched |

DAgger works. Fisher vs base p = 0.0145. That is the only method comparison we will stand behind today, and even that is **not** on the later 5×3 tape grid.

## What RECAP was supposed to add

DAgger throws away the robot's action and clones the human. RECAP keeps everything — demos, autonomous rollouts, failures, corrections — and tags each frame with whether the n-step advantage was positive. At inference you condition on POSITIVE (and, in the paper, classifier-free guidance, which we never implemented).

We warm-started from the custom ACT, trained 20k steps on 66 mixed episodes (`034-recap-balanced`), 30% conditioning dropout. Loss 0.070 → 0.046. Training "worked."

## The token is a no-op

Flip POS ↔ NEG on the **same** observation. If the bit matters, actions move.

| check | result |
|---|---|
| action sensitivity, train set | **0.93%** of observation effect (pass bar 5%) |
| held-out L1, 13 episodes | flip / all-POS / all-NEG / drop, max \|Δ\| **0.07%** |
| encoder slot rank | 309 / 603 (0.66× uniform) |
| joints slot | **1 / 603** (143× uniform) |

The network *can* concentrate on a memory slot. It chose the joints. The advantage slot is loud at its own encoder position (norm gap 10.3) and ~0 at the output. Training *reduced* token influence vs random init.

This is not "ACT cannot condition." Plant a coin-flip bit and add +0.5 z-units to the action when it is POS. Same trainer, CVAE fully on: measured effect **0.519**. Wiring works. These labels do not create that incentive.

The all-positive SFT (the "82%" checkpoint) is the same story: masking the token moves actions by **0.04%**. Do not put 82% in a comparison table. It was ordinary fine-tune on a non-overlapping sponge slice.

## Why the bit is unused

A conditioning token is learnable **iff** `I(a ; I | o) > 0`: the same observation, two different actions, labelled differently. Three ways that fails — `I` is a function of `o`, `I` is noise, `a` is determined by `o` — look the same in the policy.

On this reward (`−1`/step) the bootstrapped advantage is

```
A_t = −50 + V(o_{t+50}) − V(o_t)
```

No action term. `I` is a thresholded value residual. A linear probe reads it from the cameras (AUC **0.78**) and not from scalar `V(o)` (0.52). Neighbours matched in critic-encoder space: opposite-label actions are **2.06%** farther than same-label (p = 0.035). Statistically not zero; far too small to move L1.

Corrections do not supply the missing pairs. The human takes over **after** the state is already bad.

The CVAE latent is not the thief. KL upper-bounds what `z` carries; on the real-label runs that is **0.0001 bits** against the token's 1 bit. Test A learned the planted 0.50 with the CVAE fully active.

## Live, as implemented

Same async stack as DAgger (`eval_async`, chunk 30, EMA, wrist-flex offset 0). Inference token pinned POSITIVE.

Five tape marks × three tries:

| checkpoint | extra data | live | token |
|---|---|---|---|
| parent `balanced66` | — | **8/15 = 53%** | inert |
| mix A, 8k | 7 SPACE takeovers | **12/15 = 80%** | not re-measured |
| mix B, 10k | 4 on-policy + 3 SPACE | **3/15 = 20%** | **0.03%** causal |

A vs B is the first **matched** grid. Extra expert rescues helped. Extra on-policy failures (3 of 4 autos failed at the hard sites) made the policy worse than the parent. B's train loss never left the start band (0.056 → 0.054). A's did (0.072 → 0.040). The token stayed unused, so this is a **data-mix** result inside a RECAP-shaped trainer.

Do not read 8/15 or 12/15 vs DAgger 9/15 as RECAP vs DAgger. Placements were not matched; correction budgets were not either (7 vs 18).

## What we can actually say

1. **Data quality beat architecture.** Chunk-horizon onset explained the first failure; extra corrections explained the last gain.
2. **RECAP's token is learnable only when the label tells you about the action given the observation.** Ours does not. A better critic under the same reward cannot invent that mutual information.
3. **Cloning failures is not free.** Same step budget, same extra episode count: 12/15 vs 3/15.
4. **A live number with an unused conditioner is BC.** Report it that way.

The writeup still needs DAgger-18 on the same 5×3. Until then there is no method table, only a mechanism result: on this ACT, this reward, and this sponge task, the advantage token never fired.
