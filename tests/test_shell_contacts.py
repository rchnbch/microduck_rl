"""The v4 contact-shell fix, locked against the ways it silently un-fixes itself.

Three failure modes this guards, all of which look like nothing at import time:

* a re-export from Onshape drops a mesh, and a classifier feature quietly stops
  matching anything;
* somebody "tidies" ``FULL_COLLISION``'s friction dict back to a scalar, and the
  shell ``mu`` written in ``microduck_constants`` stops being the one MuJoCo
  uses because the floor wins the elementwise mix;
* ``priority`` goes back to 0 for the shells, with the same silent effect.

CPU only: everything here reads a compiled ``MjModel``.
"""

from __future__ import annotations

import mujoco
import pytest

from mjlab_microduck.robot.microduck_constants import (
    MICRODUCK_ALLCOLLISIONS_XML,
    SHELL_FRICTION,
    SHELL_FRICTION_RANGE,
    get_standup_spec,
    get_walk_spec,
)
from mjlab_microduck.robot.shell_contacts import (
    FOOT_GEOM_NAMES,
    SHELL_GROUND_MESHES,
    collision_geom_name,
    collision_geoms_on_bodies,
    ground_collision_geom_names,
    name_collision_geoms,
    prepare_contacts,
)


@pytest.fixture(scope="module")
def fixed_model():
    from mjlab.entity import Entity

    from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG

    return Entity(MICRODUCK_STANDUP_ROBOT_CFG).spec.compile()


def _ground_geoms(model):
    out = {}
    for i in range(model.ngeom):
        if not (model.geom_contype[i] & 1 or model.geom_conaffinity[i] & 1):
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        out[name] = i
    return out


# --------------------------------------------------------------------------- #
# Naming
# --------------------------------------------------------------------------- #


def test_the_export_really_does_leave_the_shells_unnamed():
    """The premise of the whole fix, asserted against the raw MJCF.

    If a future export starts naming them, this fails and the fix should be
    re-read rather than kept on faith."""
    raw = mujoco.MjSpec.from_file(str(MICRODUCK_ALLCOLLISIONS_XML))
    unnamed = [
        g
        for g in raw.geoms
        if not g.name and g.classname is not None and g.classname.name == "collision"
    ]
    assert len(unnamed) == 8, [g.meshname for g in unnamed]
    named = [g.name for g in raw.geoms if g.name]
    assert sorted(named) == sorted(FOOT_GEOM_NAMES)


def test_every_collision_geom_ends_up_named():
    spec = get_standup_spec()
    for geom in spec.geoms:
        cls = geom.classname.name if geom.classname is not None else ""
        if cls in ("collision", "self_collision_only"):
            assert geom.name, f"unnamed {cls} geom on body {geom.parent.name}"


def test_naming_is_idempotent():
    spec = get_standup_spec()
    before = [g.name for g in spec.geoms]
    name_collision_geoms(spec)
    prepare_contacts(spec)
    assert [g.name for g in spec.geoms] == before


def test_names_are_unique():
    spec = get_standup_spec()
    names = [g.name for g in spec.geoms if g.name]
    assert len(names) == len(set(names))


def test_the_collapse_rule_does_not_lie_about_which_part_it_is():
    # jaw_soft carries three meshes; collapsing on "body starts with mesh"
    # would name the jaw geom `jaw_soft_collision`, which reads as the head.
    assert collision_geom_name("leg", "leg") == "leg_collision"
    assert collision_geom_name("leg_2", "leg") == "leg_2_leg_collision"
    assert collision_geom_name("jaw_soft", "jaw") == "jaw_soft_jaw_collision"


# --------------------------------------------------------------------------- #
# The geoms that were missing
# --------------------------------------------------------------------------- #


def test_the_thighs_and_trunk_side_shells_gain_ground_geoms():
    spec = get_standup_spec()
    names = ground_collision_geom_names(spec)
    for mesh in SHELL_GROUND_MESHES:
        assert any(mesh in n for n in names), f"{mesh} has no ground-collision geom"


def test_ground_geom_count_went_from_ten_to_fourteen(fixed_model):
    raw = mujoco.MjSpec.from_file(str(MICRODUCK_ALLCOLLISIONS_XML))
    raw_ground = sum(
        1
        for g in raw.geoms
        if g.classname is not None and g.classname.name == "collision"
    )
    assert raw_ground == 10
    assert len(_ground_geoms(fixed_model)) == 14


def test_head_geoms_are_discoverable_by_body_not_by_name_pattern():
    spec = get_standup_spec()
    head = collision_geoms_on_bodies(spec, ("jaw_soft",))
    assert len(head) == 3
    assert all(n.endswith("_collision") for n in head)


def test_the_walk_model_gets_the_same_treatment():
    # robot_walk.xml strips most collision geoms, but whatever it has must be
    # named too — the backlash twins build on it and the invariant is family-wide.
    # The spec must be held in a local: iterating `get_walk_spec().geoms`
    # directly frees the MjSpec while its geom pointers are still live, which
    # segfaults rather than failing.
    spec = get_walk_spec()
    for geom in spec.geoms:
        cls = geom.classname.name if geom.classname is not None else ""
        if cls in ("collision", "self_collision_only"):
            assert geom.name


# --------------------------------------------------------------------------- #
# The friction actually reaching the solver
# --------------------------------------------------------------------------- #


def test_shells_carry_the_configured_friction(fixed_model):
    geoms = _ground_geoms(fixed_model)
    for name, i in geoms.items():
        expected = 1.0 if name in FOOT_GEOM_NAMES else SHELL_FRICTION
        assert fixed_model.geom_friction[i][0] == pytest.approx(expected), name


def test_every_ground_geom_has_priority_so_its_friction_is_the_one_used(fixed_model):
    # With equal priority MuJoCo mixes the pair elementwise (max for friction),
    # so a mu = 0.4 shell against a mu = 1 floor still slides at 1.0.
    for name, i in _ground_geoms(fixed_model).items():
        assert fixed_model.geom_priority[i] >= 1, name


def test_shells_are_condim_3_not_frictionless(fixed_model):
    # The cfg used to ask for condim 1 on the shells. It never matched a geom,
    # and a frictionless belly is the wrong model for a crawl that pushes off it.
    for name, i in _ground_geoms(fixed_model).items():
        assert fixed_model.geom_condim[i] == 3, name


def test_the_dr_range_brackets_the_nominal():
    lo, hi = SHELL_FRICTION_RANGE
    assert lo < SHELL_FRICTION < hi
    assert lo > 0.0


def test_feet_keep_their_rubber_friction(fixed_model):
    geoms = _ground_geoms(fixed_model)
    for name in FOOT_GEOM_NAMES:
        assert geoms[name] is not None
        assert fixed_model.geom_friction[geoms[name]][0] == pytest.approx(1.0)
