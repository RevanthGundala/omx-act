"""RECAP policy as trained on the robot.

ACT + a 3-state advantage token (NEG / POS / UNCONDITIONED) + the original
CVAE encoder. Every live checkpoint uses this class. The CVAE is unused at
inference (``z = 0``) and collapsed in training; it is still part of the
architecture so those checkpoints load.
"""

from .model import ACTPolicy, AdvantageCondition, CVAE_Encoder, Decoder

__all__ = ["ACTPolicy", "AdvantageCondition", "CVAE_Encoder", "Decoder"]
