"""Observation-only distributional critic. No CVAE, no action input.

``V(o)`` only. The n-step advantage is a value residual, so this model
cannot invent ``I(a ; I | o)``.
"""

from .model import ACTCritic, Decoder

__all__ = ["ACTCritic", "Decoder"]
