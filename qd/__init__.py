"""Quality-Diversity (MAP-Elites) gait discovery for the Microduck biped.

Phase 2 (``cpg_genome`` / ``evaluate`` / ``run_map_elites``) runs vanilla
MAP-Elites over an open-loop CPG genome.  Phase 3 (``qd.pga``) reuses the same
batched evaluation harness, archive and behaviour descriptor with a closed-loop
MLP genome and policy-gradient variation.

Everything here is deliberately separate from the PPO training stack: the only
things imported from ``mjlab_microduck`` are the robot config and the MJCF, so
a change under ``qd/`` can never perturb a training run.
"""

# Import order matters. ``import mjlab`` runs mjlab's plugin loader, which
# imports ``mjlab_microduck.tasks`` -> ``...robot.microduck_constants``. If a
# ``qd`` module imports ``microduck_constants`` first, the loader re-enters a
# partially initialized module and mjlab prints
# "[WARN] Failed to load task package mjlab_microduck". Nothing here needs the
# task registry, but the warning is confusing noise, so let mjlab finish first.
import mjlab as _mjlab  # noqa: F401
