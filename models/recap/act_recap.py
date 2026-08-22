"""Compatibility shim. The model lives in ``models.recap.policy``."""

from models.recap.policy.model import ACTPolicy, AdvantageCondition, CVAE_Encoder, Decoder

__all__ = ["ACTPolicy", "AdvantageCondition", "CVAE_Encoder", "Decoder"]
