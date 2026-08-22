"""Compatibility shim. The model lives in ``models.recap.critic``."""

from models.recap.critic.model import ACTCritic, Decoder

__all__ = ["ACTCritic", "Decoder"]
