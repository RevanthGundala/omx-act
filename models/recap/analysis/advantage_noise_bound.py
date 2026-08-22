"""How much of the RECAP advantage is critic error rather than signal?

The advantage is defined as

    A_t = sum(r[t:t+N]) + V(o_{t+N}) - V(o_t)

and the realized return satisfies, by definition,

    R_t = sum(r[t:t+N]) + R_{t+N}

Substituting the second into the first gives an identity:

    A_t = [V(o_{t+N}) - R_{t+N}] - [V(o_t) - R_t]
        = residual_{t+N} - residual_t

So the advantage is *exactly* the difference of the critic's residuals. A critic
that predicted returns perfectly would make every advantage identically zero.
This is the same structure as the CVAE-latent proof in section 5.2: whenever the
value head can reconstruct the realized return, the advantage collapses.

That is not a defect -- part of the residual is genuine signal, because the
future is stochastic and V(o) predicts its expectation. But it means the usable
signal is bounded by how much of the residual is systematic rather than
approximation error, and it makes the critic's accuracy the thing that decides
whether a binary label carries information.

This module quantifies the resulting bound. Run it directly for the table.
"""

from __future__ import annotations

import math


def advantage_noise_std(value_error_std: float, residual_correlation: float) -> float:
    """Std of the advantage induced purely by critic error.

    ``A = residual_{t+N} - residual_t``, so for residuals with equal variance and
    correlation ``rho`` the difference has variance ``2 * (1 - rho) * var``.
    Correlation is what saves this: errors that persist across the lookahead
    cancel in the subtraction, and only the part that changes survives.
    """
    if not -1.0 <= residual_correlation <= 1.0:
        raise ValueError(f"correlation must be in [-1, 1], got {residual_correlation}")
    if value_error_std < 0:
        raise ValueError(f"value error std must be non-negative, got {value_error_std}")
    return math.sqrt(2.0 * (1.0 - residual_correlation)) * value_error_std


def correlation_required(value_error_std: float, threshold: float, margin: float = 1.0) -> float:
    """Residual correlation needed for advantage noise to stay under a threshold.

    ``margin`` is how many noise standard deviations must fit inside the
    threshold before a sign decision at that threshold is better than a coin
    flip. Returns the required correlation, which may exceed 1 -- meaning no
    achievable correlation is enough.
    """
    if threshold <= 0:
        raise ValueError(f"threshold must be positive, got {threshold}")
    target = threshold / margin
    return 1.0 - (target / value_error_std) ** 2 / 2.0


def main() -> None:
    # Measured on this project: outputs/recap_critic_20k/best_critic.pt, the
    # critic that produced the labels every policy run has used.
    val_mae = 86.6
    # A normal error has MAE = sigma * sqrt(2/pi), so sigma = MAE * 1.2533.
    value_error_std = val_mae * math.sqrt(math.pi / 2.0)
    threshold = 6.49  # eps_l resolved by the 40% positive-fraction target

    print("Critic that produced the labels: val return MAE = 86.6")
    print(f"implied residual std           = {value_error_std:.1f} return units")
    print(f"advantage threshold eps_l      = {threshold} return units")
    print(f"ratio                          = {value_error_std / threshold:.0f}x")
    print()
    print("Advantage noise from critic error alone, by how correlated the")
    print("critic's residuals are across the N=50 lookahead:")
    print()
    print(f"  {'correlation':>12} {'noise std':>11} {'vs threshold':>14}")
    for rho in (0.0, 0.5, 0.9, 0.99, 0.999, 0.9999):
        noise = advantage_noise_std(value_error_std, rho)
        print(f"  {rho:>12.4f} {noise:>11.2f} {noise / threshold:>13.1f}x")

    required = correlation_required(value_error_std, threshold)
    print()
    print(f"For the noise to fit inside the threshold at all, the residuals")
    print(f"would have to be correlated at rho > {required:.5f} across 50 steps")
    print("(1.7 seconds at 30 Hz).")
    print()
    print("This is a bound, not a measurement: it says what the critic's accuracy")
    print("permits, and the actual correlation has not been measured. Measuring it")
    print("is the cheapest way to find out whether the labels carry signal at all.")


if __name__ == "__main__":
    main()
