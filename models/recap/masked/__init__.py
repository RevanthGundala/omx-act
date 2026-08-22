"""Masked-token RECAP variant.

Same ACT/CVAE backbone as ``models.recap.policy``, but the advantage token
can be genuinely absent (encoder/decoder mask) instead of a third learned
UNCONDITIONED embedding. Used for the from-scratch / SFT mechanism checks,
not for the live mix-ablation policies.
"""

from .model import ACTPolicy, AdvantageCondition, MASKED_ADVANTAGE, CVAE_Encoder, Decoder

__all__ = [
    "ACTPolicy",
    "AdvantageCondition",
    "MASKED_ADVANTAGE",
    "CVAE_Encoder",
    "Decoder",
]
