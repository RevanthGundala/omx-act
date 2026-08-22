"""Compatibility shim. The model lives in ``models.recap.masked``."""

from models.recap.masked.model import (
    ACTPolicy,
    AdvantageCondition,
    MASKED_ADVANTAGE,
    CVAE_Encoder,
    Decoder,
)

__all__ = [
    "ACTPolicy",
    "AdvantageCondition",
    "MASKED_ADVANTAGE",
    "CVAE_Encoder",
    "Decoder",
]
