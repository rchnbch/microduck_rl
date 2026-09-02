"""PGA-MAP-Elites (Nilsson & Cully, GECCO 2021) in PyTorch.

Same archive, same objective and same behaviour descriptor as the Phase-2
vanilla run in :mod:`qd.run_map_elites` — what changes is the genome (a
closed-loop MLP over the repo's 61-D observation contract instead of an
open-loop CPG) and how offspring are produced: half by directional GA
variation between two elites, half by taking policy-gradient steps against a
TD3 critic trained on every transition the archive has ever collected.

Reference implementation consulted for hyperparameters: QDax's ``pga_me``
(``qdax/core/emitters/pga_me_emitter.py`` and ``qdax/baselines/td3.py``). No
JAX dependency here — everything is PyTorch, to match mjlab's stack.
"""
