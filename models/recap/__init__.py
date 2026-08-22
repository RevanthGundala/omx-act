"""RECAP policy and value models built from the custom ACT implementation.

Architectural variants live in their own folders:

* ``policy/``  — 3-state token + CVAE. This is what we trained and eval'd live.
* ``masked/``  — token can be omitted. Mechanism / SFT checks only.
* ``critic/``  — observation-only ``V(o)``. No CVAE, no action input.

The old module paths (``act_recap``, ``act_recap_masked``, ``act_critic``)
re-export from those folders.
"""
