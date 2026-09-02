"""CPU regression tests for walking-v2: honest physics, survival gate, seeding.

Three claims are worth locking down because each of them is invisible when it
breaks — the run keeps producing an archive, just a dishonest one:

* the rollout is simulated on the **all-collisions** model and stops at the
  fall, so nothing post-fall reaches fitness, descriptor or replay buffer;
* only full-episode survivors enter the archive, whatever their displacement;
* the seed path reads a real rsl_rl checkpoint and writes the forward command
  into the twist slot the 61-D contract puts it in.
"""

import numpy as np
import pytest
import torch
from qd.common import FitnessCfg, RolloutMetrics
from qd.pga.policy_genome import PolicySpec

SPEC = PolicySpec()


# --------------------------------------------------------------------------- #
# Honest physics
# --------------------------------------------------------------------------- #


def test_honest_physics_is_the_default_on_both_harnesses():
    """The v2 defaults are the point of the job; a silent revert is a bug.

    Both pipelines must agree, or a CPG archive and an MLP archive stop being
    measured under the same physics and the README's comparison dies.
    """
    from qd.evaluate import HarnessCfg
    from qd.pga.evaluate import PolicyHarnessCfg

    assert HarnessCfg().full_collision is True
    assert PolicyHarnessCfg().full_collision is True
    # The fall check has to actually run, and not on every step (host sync).
    assert HarnessCfg().fall_check_every > 1
    assert PolicyHarnessCfg().fall_check_every > 1
    # Full collision means more contacts, and mjlab's default constraint
    # allocation overflows on a robot lying on its side. An overflow silently
    # drops constraints — it makes the floor soft again, which is the exact bug
    # full_collision exists to fix — so the low-level harness pins its own.
    assert HarnessCfg().njmax >= 100


def test_full_collision_selects_the_all_collisions_model():
    """`robot_walk.xml` collides at the soles only; the fix is the other MJCF."""
    from qd.evaluate import HarnessCfg, _deterministic_robot_cfg

    walk = _deterministic_robot_cfg(HarnessCfg(full_collision=False))
    full = _deterministic_robot_cfg(HarnessCfg(full_collision=True))
    assert walk.spec_fn.__name__ == "get_walk_spec"
    assert full.spec_fn is not walk.spec_fn
    # Same robot otherwise: identical HOME frame and identical actuators, so
    # the only thing the switch changes is which geoms touch the ground.
    assert full.init_state == walk.init_state
    assert full.articulation.actuators == walk.articulation.actuators


def test_the_two_models_differ_only_in_ground_collision_geoms():
    """If the all-collisions export ever grew a joint, the genome would break."""
    import mujoco

    from mjlab_microduck.robot import microduck_constants as mc

    walk = mujoco.MjModel.from_xml_path(str(mc.MICRODUCK_WALK_XML))
    full = mujoco.MjModel.from_xml_path(str(mc.MICRODUCK_ALLCOLLISIONS_XML))
    assert walk.njnt == full.njnt
    assert walk.nu == full.nu
    names = lambda m: [
        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)
    ]
    assert names(walk) == names(full)
    # ...and the all-collisions model really does add ground-colliding geoms.
    colliding = lambda m: sum(
        1 for i in range(m.ngeom) if m.geom_contype[i] & 1 and m.geom_conaffinity[i] & 1
    )
    assert colliding(full) > colliding(walk)


def test_fallen_fraction_uses_the_full_episode_when_the_loop_stops_early():
    """Early-stopping the rollout must not make every env look 100% alive."""
    cfg = FitnessCfg(fall_penalty=0.25, fall_height=0.09)
    # 10 steps simulated out of a 100-step episode; the env fell on step 5.
    m = RolloutMetrics(1, cfg, "cpu", episode_steps=100)
    m.begin(torch.tensor([[0.0, 0.0, 0.12]]))
    for step in range(10):
        z = 0.12 if step < 5 else 0.02
        m.update(
            torch.tensor([[0.0, 0.0, z]]),
            torch.tensor([[0.0, 0.0, -1.0]]),
            torch.tensor([[True, True]]),
        )
    _, _, info = m.finalize()
    # 6 counted steps — the 5 upright ones plus the step the fall was detected
    # on, which counts so an env that fails immediately still has a non-zero
    # duty-factor denominator — out of the full 100-step episode, NOT out of
    # the 10 that were actually stepped.
    assert info["alive_steps"][0] == 6
    assert info["survival_fraction"][0] == pytest.approx(0.06)
    assert info["fell"][0]


def test_metrics_ignore_everything_after_the_fall():
    """A face-plant that skids forward must not keep earning displacement, and
    a robot lying on both soles must not report duty (1, 1)."""
    cfg = FitnessCfg(fall_penalty=0.0, fall_height=0.09)
    m = RolloutMetrics(1, cfg, "cpu", episode_steps=4)
    m.begin(torch.tensor([[0.0, 0.0, 0.12]]))
    steps = [(0.1, 0.12, False), (0.2, 0.02, True), (5.0, 0.02, True), (9.0, 0.02, True)]
    for x, z, on_floor in steps:
        m.update(
            torch.tensor([[x, 0.0, z]]),
            torch.tensor([[0.0, 0.0, -1.0]]),
            torch.tensor([[on_floor, on_floor]]),
        )
    fitness, measures, info = m.finalize()
    # Frozen at the fall step (x=0.2), not at the 9.0 m skid.
    assert info["displacement"][0] == pytest.approx(0.2)
    assert fitness[0] == pytest.approx(0.2)
    # Contacts: 1 of the 2 counted steps had feet down, not 3 of 4.
    assert measures[0].tolist() == pytest.approx([0.5, 0.5])


# --------------------------------------------------------------------------- #
# Survival gate
# --------------------------------------------------------------------------- #


def test_survival_gate_admits_only_full_episode_survivors():
    pytest.importorskip("ribs")
    from qd.common import make_archive
    from qd.pga.run_pga_me import _insert

    archive = make_archive(solution_dim=4, grid_dims=(4, 4), qd_score_offset=-5.0)
    genomes = torch.zeros(4, 4)
    # The faller travels FURTHEST — v1 would have taken it, and did.
    fitness = np.array([0.9, 0.1, 0.2, 0.3])
    measures = np.array([[0.1, 0.1], [0.4, 0.4], [0.6, 0.6], [0.9, 0.9]])
    survived = np.array([False, True, True, True])

    rate, feasible = _insert(archive, genomes, fitness, measures, survived, gate=True)
    assert feasible == pytest.approx(0.75)
    assert archive.stats.num_elites == 3
    assert rate == pytest.approx(3 / 4), "insertion rate keeps v1's denominator"
    assert archive.stats.obj_max == pytest.approx(0.3), "the 0.9 faller stayed out"

    ungated = make_archive(solution_dim=4, grid_dims=(4, 4), qd_score_offset=-5.0)
    _insert(ungated, genomes, fitness, measures, survived, gate=False)
    assert ungated.stats.num_elites == 4
    assert ungated.stats.obj_max == pytest.approx(0.9)


def test_gate_of_an_all_fallen_block_inserts_nothing_without_crashing():
    pytest.importorskip("ribs")
    from qd.common import make_archive
    from qd.pga.run_pga_me import _insert

    archive = make_archive(solution_dim=4, grid_dims=(4, 4), qd_score_offset=-5.0)
    rate, feasible = _insert(
        archive,
        torch.zeros(3, 4),
        np.array([0.1, 0.2, 0.3]),
        np.full((3, 2), 0.5),
        np.zeros(3, dtype=bool),
    )
    assert (rate, feasible) == (0.0, 0.0)
    assert archive.stats.num_elites == 0


def test_gated_archive_objective_is_plain_displacement():
    """The gate makes the fall penalty arithmetically inert, so v1's fitness
    formula can stay untouched and the two archives remain comparable."""
    cfg = FitnessCfg(fall_penalty=0.25)
    m = RolloutMetrics(1, cfg, "cpu", episode_steps=3)
    m.begin(torch.tensor([[0.0, 0.0, 0.12]]))
    for x in (0.1, 0.2, 0.3):
        m.update(
            torch.tensor([[x, 0.0, 0.12]]),
            torch.tensor([[0.0, 0.0, -1.0]]),
            torch.tensor([[True, True]]),
        )
    fitness, _, info = m.finalize()
    assert not info["fell"][0]
    assert fitness[0] == pytest.approx(info["displacement"][0], abs=1e-6)


def test_empty_archive_falls_back_to_random_parents():
    """Under the gate an archive can be legitimately empty; iteration 1 must
    keep searching rather than raise."""
    pytest.importorskip("ribs")
    from qd.common import make_archive
    from qd.pga.variation import sample_parents

    archive = make_archive(solution_dim=SPEC.genome_dim, grid_dims=(4, 4))
    g = torch.Generator().manual_seed(0)
    parents = sample_parents(archive, 5, g, "cpu", spec=SPEC)
    assert parents.shape == (5, SPEC.genome_dim)
    assert not torch.allclose(parents[0], parents[1])


# --------------------------------------------------------------------------- #
# Seeding
# --------------------------------------------------------------------------- #


def test_seed_jitter_spreads_without_leaving_the_neighbourhood():
    from qd.pga.variation import ISO_SIGMA
    from qd.seed import SeedCfg, jitter

    cfg = SeedCfg()
    g = torch.Generator().manual_seed(0)
    seed = torch.zeros(1, SPEC.genome_dim)
    family = jitter(seed, cfg.jitter_count, cfg.jitter_sigma, g)
    assert family.shape == (cfg.jitter_count, SPEC.genome_dim)
    assert family.std().item() == pytest.approx(cfg.jitter_sigma, rel=0.1)
    # Wider than GA variation's iso term, or every variant lands in one cell.
    assert cfg.jitter_sigma > ISO_SIGMA
    assert jitter(seed, 0, cfg.jitter_sigma, g).shape == (0, SPEC.genome_dim)


def test_seed_writes_the_teacher_command_into_the_twist_slot():
    """The forward command is baked into the seed's weights, not its input —
    so the slot the teacher is queried at must be the twist block."""
    from qd.seed import TWIST_SLICE

    # 48 proprioception + [twist(3), head(4), body(6)] = the 61-D contract.
    assert (TWIST_SLICE.start, TWIST_SLICE.stop) == (48, 51)
    assert SPEC.obs_dim == 61


def test_teacher_vx_is_inside_the_velocity_task_command_range():
    """A teacher queried outside the range it trained on is extrapolating."""
    from qd.seed import SeedCfg

    assert 0.01 < SeedCfg().teacher_vx <= 0.4


def test_ppo_teacher_rebuilds_an_rsl_rl_actor_state_dict():
    """The seed path reads a checkpoint without importing rsl_rl."""
    from qd.seed import PpoTeacher

    sd = {
        "obs_normalizer._mean": torch.zeros(1, 61),
        "obs_normalizer._std": torch.ones(1, 61),
        "mlp.0.weight": torch.randn(8, 61),
        "mlp.0.bias": torch.zeros(8),
        "mlp.2.weight": torch.randn(14, 8),
        "mlp.2.bias": torch.zeros(14),
    }
    teacher = PpoTeacher(sd)
    assert teacher.obs_dim == 61
    assert teacher(torch.randn(3, 61)).shape == (3, 14)
    # Linear output head: an rsl_rl actor is not tanh-bounded, which is exactly
    # why the seed is distilled rather than weight-copied.
    assert not isinstance(teacher.mlp[-1], torch.nn.Tanh)


def test_ppo_teacher_applies_the_observation_normalizer():
    """Forgetting the normalizer is the classic silent-wrong-policy bug
    (AGENTS.md: obs normalization is ON, and must be baked in)."""
    from qd.seed import PpoTeacher

    weight = torch.eye(4, 4)
    sd = {
        "obs_normalizer._mean": torch.full((1, 4), 2.0),
        "obs_normalizer._std": torch.full((1, 4), 3.0),
        "mlp.0.weight": weight,
        "mlp.0.bias": torch.zeros(4),
    }
    teacher = PpoTeacher(sd, eps=0.0)
    obs = torch.full((1, 4), 5.0)
    torch.testing.assert_close(teacher(obs), torch.ones(1, 4))


def test_commands_default_to_a_single_forward_walk():
    from qd.seed import SeedCfg, commands_of

    assert commands_of(SeedCfg()) == ((SeedCfg().teacher_vx, 0.0, 0.0),)
    multi = SeedCfg(teacher_commands=((0.1, 0.0, 0.0), (0.4, 0.0, 0.0)))
    assert commands_of(multi) == ((0.1, 0.0, 0.0), (0.4, 0.0, 0.0))


def test_seed_family_splits_the_jitter_budget_across_seeds():
    """Every seed needs its own neighbourhood: they sit far apart in descriptor
    space, and the cells between them are filled by their clouds."""
    from qd.seed import seed_family

    g = torch.Generator().manual_seed(0)
    seeds = torch.arange(3 * 8, dtype=torch.float32).reshape(3, 8)
    family = seed_family(seeds, count=30, sigmas=0.02, generator=g)
    assert family.shape == (3 + 30, 8)
    # The seeds themselves come first, unperturbed — a distilled walker must be
    # offered to the archive exactly as it was verified.
    torch.testing.assert_close(family[:3], seeds)
    # Each seed's cloud sits on that seed, not on seed 0.
    for i in range(3):
        cloud = family[3 + i * 10 : 3 + (i + 1) * 10]
        assert (cloud - seeds[i]).abs().max() < 0.2


def test_seed_family_of_one_seed_matches_plain_jitter():
    from qd.seed import jitter, seed_family

    seeds = torch.zeros(1, 8)
    a = seed_family(seeds, 10, 0.02, torch.Generator().manual_seed(1))
    b = torch.cat([seeds, jitter(seeds, 10, 0.02, torch.Generator().manual_seed(1))])
    torch.testing.assert_close(a, b)


def test_seed_family_splits_across_the_sigma_ladder_too():
    """The useful jitter radius is not known in advance, so the block probes
    several — the budget divides over (seed, sigma) pairs."""
    from qd.seed import seed_family

    g = torch.Generator().manual_seed(0)
    seeds = torch.zeros(2, 8)
    family = seed_family(seeds, count=40, sigmas=(0.01, 0.08), generator=g)
    # 2 seeds + 4 clouds of 10.
    assert family.shape == (2 + 40, 8)
    clouds = [family[2 + i * 10 : 2 + (i + 1) * 10] for i in range(4)]
    narrow = [c.std().item() for c in (clouds[0], clouds[2])]
    wide = [c.std().item() for c in (clouds[1], clouds[3])]
    assert max(narrow) < min(wide), "the ladder collapsed to one radius"
    assert min(wide) > 5 * max(narrow)
