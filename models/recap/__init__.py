"""RECAP policy and value models built from the custom ACT implementation.

Architectural variants live in their own folders:

* ``policy/``   — 3-state token + CVAE. What we trained and eval'd live.
* ``masked/``   — token can be omitted. Mechanism / SFT checks only.
* ``critic/``   — observation-only ``V(o)``. No CVAE, no action input.
* ``analysis/`` — token-sensitivity, held-out L1, I(a;I|o), attention.

Old module paths (``act_recap``, ``check_token_sensitivity``, …) re-export
from those folders.
"""
