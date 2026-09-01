"""CPU regression tests for the MAP-Elites CPG genome (qd/cpg_genome.py).

These lock the two things that silently rot: the joint set/order resolving
against the *actual* model, and the two-layer bound enforcement.
"""

import math

import mujoco
import numpy as np
import pytest
import torch
from qd import cpg_genome
from qd.evaluate import cpg_target_trajectory

from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_XML


def test_genome_dimension_is_31():
    assert cpg_genome.NUM_LEG_JOINTS == 10
    assert cpg_genome.GENOME_DIM == 31
    space = cpg_genome.genome_space()
    assert space.dim == 31


def test_leg_joints_exist_and_follow_the_servo_layout():
    """Left leg then right leg, hip_yaw/roll/pitch/knee/ankle — no neck/head."""
    model = mujoco.MjModel.from_xml_path(str(MICRODUCK_WALK_XML))
    for name in cpg_genome.LEG_JOINT_NAMES:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) >= 0, name

    order = ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")
    assert cpg_genome.LEG_JOINT_NAMES[:5] == tuple(f"left_{j}" for j in order)
    assert cpg_genome.LEG_JOINT_NAMES[5:] == tuple(f"right_{j}" for j in order)
    assert not any("head" in n or "neck" in n for n in cpg_genome.LEG_JOINT_NAMES)


def test_leg_joint_ids_resolve_on_the_entity_in_the_declared_order():
    """The harness selects by NAME; this pins that the names still resolve.

    ``Entity.joint_names`` interleaves neck/head between the legs (servo indices
    5-8), so a hard-coded 0..9 slice would silently grab the head.
    """
    from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_ROBOT_CFG

    entity = MICRODUCK_WALK_ROBOT_CFG.build()
    ids, names = entity.find_joints(list(cpg_genome.LEG_JOINT_NAMES), preserve_order=True)
    assert tuple(names) == cpg_genome.LEG_JOINT_NAMES
    # Left leg 0-4 and right leg 9-13 in the 14-servo layout; 5-8 are neck/head.
    assert ids == [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]


def test_offset_bounds_sit_inside_the_mjcf_limits():
    space = cpg_genome.genome_space()
    raw_lo, raw_hi = cpg_genome.leg_joint_limits()
    assert np.all(space.lower[cpg_genome.OFFSET_SLICE] > raw_lo)
    assert np.all(space.upper[cpg_genome.OFFSET_SLICE] < raw_hi)
    # 0.9 soft factor about the midpoint.
    mid = 0.5 * (raw_lo + raw_hi)
    np.testing.assert_allclose(
        space.upper[cpg_genome.OFFSET_SLICE] - mid,
        0.45 * (raw_hi - raw_lo),
        rtol=1e-9,
    )


def test_amplitude_and_frequency_bounds():
    space = cpg_genome.genome_space()
    raw_lo, raw_hi = cpg_genome.leg_joint_limits()
    assert np.all(space.lower[cpg_genome.AMP_SLICE] == 0.0)
    np.testing.assert_allclose(
        space.upper[cpg_genome.AMP_SLICE], 0.25 * (raw_hi - raw_lo)
    )
    assert space.lower[0] == pytest.approx(0.5)
    assert space.upper[0] == pytest.approx(3.0)
    np.testing.assert_allclose(space.lower[cpg_genome.PHASE_SLICE], 0.0)
    np.testing.assert_allclose(space.upper[cpg_genome.PHASE_SLICE], 2 * math.pi)


def test_sigma_is_ten_percent_of_each_range():
    space = cpg_genome.genome_space()
    np.testing.assert_allclose(space.sigma(0.1), 0.1 * (space.upper - space.lower))


def test_cpg_targets_match_the_closed_form():
    space = cpg_genome.genome_space()
    rng = np.random.default_rng(0)
    g = space.sample(4, rng)
    t = 0.37
    got = cpg_genome.cpg_targets(g, t)
    freq, amp, phase, offset = cpg_genome.unpack(g)
    want = offset + amp * np.sin(2 * np.pi * freq * t + phase)
    np.testing.assert_allclose(got, want)
    assert got.shape == (4, 10)


def test_torch_trajectory_matches_the_numpy_reference():
    """The GPU path in qd/evaluate.py must not drift from the reference."""
    space = cpg_genome.genome_space()
    rng = np.random.default_rng(1)
    g = space.sample(6, rng)
    times = np.arange(11) * 0.02
    traj = cpg_target_trajectory(
        torch.as_tensor(g, dtype=torch.float64),
        torch.as_tensor(times, dtype=torch.float64),
    ).numpy()
    assert traj.shape == (11, 6, 10)
    for k, t in enumerate(times):
        np.testing.assert_allclose(traj[k], cpg_genome.cpg_targets(g, float(t)), atol=1e-9)


def test_clip_targets_saturates_at_the_soft_limits():
    """Bound enforcement #2: an offset+amplitude excursion is clamped, not
    silently absorbed by the actuator model."""
    soft_lo, soft_hi = cpg_genome.soft_leg_joint_limits()
    wild = np.stack([np.full(10, 10.0), np.full(10, -10.0)])
    clipped = cpg_genome.clip_targets(wild)
    np.testing.assert_allclose(clipped[0], soft_hi)
    np.testing.assert_allclose(clipped[1], soft_lo)


def test_space_sample_and_clip_respect_the_bounds():
    space = cpg_genome.genome_space()
    rng = np.random.default_rng(2)
    g = space.sample(64, rng)
    assert np.all(g >= space.lower) and np.all(g <= space.upper)
    out_of_range = g + 100.0
    np.testing.assert_array_less(space.clip(out_of_range) - space.upper, 1e-9)


def test_unpack_rejects_the_wrong_width():
    with pytest.raises(ValueError):
        cpg_genome.unpack(np.zeros((3, 30)))
