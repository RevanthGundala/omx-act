# Why Evaluate RECAP with ACT?

## Motivation

RECAP is not being selected because it is universally better than other
reinforcement-learning algorithms. It is a particularly good match for the
constraints of this project:

- Our ACT policy emits continuous action chunks but is deterministic at
  inference and does not provide a tractable action log-probability.
- RECAP can reuse ACT's existing supervised `L1 + KL` objective instead of
  requiring a stochastic policy head for PPO or SAC.
- Real-robot interaction is expensive. RECAP performs iterated offline updates
  and can reuse demonstrations, old autonomous rollouts, failures, and human
  corrections.
- RECAP does not require exploratory policy-gradient rollouts on the physical
  robot, where unconstrained exploration could be unsafe.
- It preserves our current async, closed-loop execution and EMA chunk blending.
  A method such as PAC-ACT would require sampled action chunks with well-defined
  log-probabilities and different chunk-execution semantics.
- Unlike success-filtered behavior cloning or advantage-weighted regression,
  RECAP is designed to train on all collected experience by identifying whether
  each action has positive or negative advantage.

For the immediate goal of improving robot performance, DAgger remains the
simpler and safer baseline. The research contribution is evaluating whether
RECAP adds value beyond DAgger once the behavior-cloned policy and evaluation
conditions are reliable.

## Primary Research Question

> Can a deterministic, L1-trained ACT policy improve from mixed-quality
> real-robot experience using RECAP-style advantage conditioning, without flow
> matching, and outperform DAgger and advantage-weighted or success-filtered
> behavior cloning under the same rollout and correction budget?

This asks whether RECAP's advantage-conditioning mechanism transfers from an
expressive flow-matching VLA to a smaller deterministic action-chunking policy.

## Hypothesis

Given the same demonstrations, autonomous rollouts, and human corrections,
RECAP-ACT will improve balanced task success because it can:

1. Assign credit within trajectories instead of labeling an entire failed
   episode as useless.
2. Retain all collected data rather than discarding or heavily downweighting
   unsuccessful behavior.
3. Learn from autonomous experience while remaining anchored to demonstrated
   behavior.

## Required Comparisons

All methods should start from the same ACT checkpoint and receive the same data
and optimization budget.

| Method | Purpose |
|---|---|
| Frozen ACT | Measures the original behavior-cloning baseline |
| DAgger/BC | Tests whether corrections and additional expert labels are sufficient |
| Success-filtered BC or AWR | Tests whether selecting or weighting good actions is sufficient |
| RECAP-ACT | Tests whether explicit advantage conditioning adds value |

PPO, PAC-ACT, and SAC are not the first comparisons because they require a
stochastic actor, reliable action log-probabilities, more online interaction,
and changes to the current chunk-execution controller.

## Evaluation

The primary metric should be macro-averaged success over a fixed, balanced
position-and-orientation evaluation grid. Secondary metrics should include:

- Worst-layout success rate.
- Completion time or throughput.
- Performance on layouts the original policy already solved.
- Human corrections and autonomous episodes required.

Lighting should be treated as a controlled evaluation condition initially so
that observation shift does not confound the RL comparison.

## Interpretation

If RECAP-ACT outperforms the controls, the result shows that
advantage-conditioned policy improvement can transfer beyond flow-matching
policies. If it does not outperform DAgger or AWR, that is still informative:
the benefit of RECAP may depend on a more expressive generative action
distribution, or the task's remaining errors may be dominated by data coverage
rather than credit assignment.
