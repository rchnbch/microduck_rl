"""Naming and completing the robot's ground-contact shell.

Three facts about the exported MJCF, all read off the compiled model rather
than assumed (see ``qd.check_shell_contacts`` for the before/after dump):

1. **Only the two soles are named.** ``onshape-to-robot`` names a geom only
   when the CAD part is tagged; everything else comes out as an anonymous
   ``<geom class="collision" mesh="...">``. ``FULL_COLLISION`` addresses geoms
   by *name* (``geom_names_expr=[".*_collision"]``), so its ``condim=1`` rule —
   written to make the shells frictionless — matched exactly two geoms, both of
   which are exempted by the foot rule ahead of it. Every other ground contact
   on this robot has been running MuJoCo's defaults: ``condim 3``,
   ``priority 0``, ``mu = 1``.

   The same anonymity defeats ``CollisionCfg.disable_other_geoms``: it collects
   the non-matching names into a *set*, so all ~70 unnamed geoms collapse to a
   single ``''`` entry and at most one of them is ever disabled. The shells
   were live by accident, not by design.

2. **``priority 0`` means the robot's friction is not the one used.** With
   equal priority MuJoCo mixes the pair elementwise (``max`` for friction), so
   a shell set to ``mu = 0.4`` against a ``mu = 1`` floor still slides at 1.
   Only ``priority >= 1`` on the robot geom makes the number written here the
   number the solver uses — which is why the soles already carry it.

3. **The upper legs and the trunk side shells have no ground geom at all.**
   The exported collision set is trunk battery pack, both hip brackets, both
   shanks, both soles and three head-shell meshes. A prone robot rests on its
   thighs and its side shells; on the model as exported, those parts are
   ghosts, and a crawl would be pushing off geometry that does not exist.

This module fixes all three on the ``mujoco.MjSpec``, not in the generated XML,
so a re-export from Onshape does not silently undo it (AGENTS.md: the MJCFs in
``robot/microduck/`` are build artefacts).
"""

from __future__ import annotations

import mujoco

COLLISION_CLASS = "collision"
SELF_COLLISION_CLASS = "self_collision_only"

FOOT_GEOM_NAMES: tuple[str, ...] = ("left_foot_collision", "right_foot_collision")
"""The two sole geoms the export already names. Rubber-ish pads, ``mu = 1``."""

SHELL_GROUND_MESHES: tuple[str, ...] = (
    "upper_leg_left",
    "upper_leg_right",
    "left_shell",
    "right_shell",
)
"""Visual meshes that need a ground-collision twin.

The thighs and the two trunk side shells: what a robot lying prone or on its
side actually rests on, and the only ground-facing surfaces the export left
without a collision geom. Deliberately *not* the neck (it is shielded by the
head shells in every rest pose measured) and not the many fastener/bearing
meshes (interior).
"""


def collision_geom_name(body_name: str, mesh_name: str, suffix: str = "collision") -> str:
    """Deterministic name for an anonymous collision geom.

    ``<body>_<mesh>_<suffix>``, collapsing the redundant half only when the two
    are *equal* — so the left shank is ``leg_collision`` and its mirror is
    ``leg_2_leg_collision``. A looser "body starts with mesh" collapse reads
    better and lies: the jaw geom on body ``jaw_soft`` would become
    ``jaw_soft_collision``, which every reader takes for the whole head.
    """
    if body_name == mesh_name:
        return f"{body_name}_{suffix}"
    return f"{body_name}_{mesh_name}_{suffix}"


def _classname(geom) -> str:
    cls = geom.classname
    return cls.name if cls is not None else ""


def name_collision_geoms(spec: mujoco.MjSpec) -> dict[str, str]:
    """Give every anonymous collision geom a stable name. Idempotent.

    Returns ``{geom name: class name}`` for the geoms this call named, so a
    caller (or a test) can assert on what the export actually contained rather
    than on a hardcoded list.
    """
    named: dict[str, str] = {}
    taken = {g.name for g in spec.geoms if g.name}
    for geom in spec.geoms:
        if geom.name:
            continue
        cls = _classname(geom)
        if cls not in (COLLISION_CLASS, SELF_COLLISION_CLASS):
            continue
        body = geom.parent.name
        mesh = geom.meshname or "geom"
        suffix = "collision" if cls == COLLISION_CLASS else "selfcollision"
        name = collision_geom_name(body, mesh, suffix)
        candidate, k = name, 1
        while candidate in taken:
            k += 1
            candidate = f"{name}_{k}"
        geom.name = candidate
        taken.add(candidate)
        named[candidate] = cls
    return named


def add_shell_ground_geoms(spec: mujoco.MjSpec) -> tuple[str, ...]:
    """Clone the missing ground-facing visual meshes into collision geoms.

    Same mesh, same body frame, same pose as the visual geom it copies, so the
    contact surface is the part's real outer surface rather than a hand-fitted
    primitive. Idempotent: a geom whose name already exists is skipped.
    """
    added: list[str] = []
    existing = {g.name for g in spec.geoms if g.name}
    # `spec.geoms` is live; adding while iterating it is not safe, so the
    # sources are collected first.
    sources = [
        g
        for g in spec.geoms
        if g.meshname in SHELL_GROUND_MESHES and _classname(g) != COLLISION_CLASS
    ]
    for src in sources:
        name = collision_geom_name(src.parent.name, src.meshname)
        if name in existing:
            continue
        geom = src.parent.add_geom()
        geom.name = name
        geom.type = src.type
        geom.meshname = src.meshname
        geom.pos = src.pos
        geom.quat = src.quat
        geom.classname = spec.find_default(COLLISION_CLASS)
        geom.group = 3
        geom.contype = 1
        geom.conaffinity = 1
        existing.add(name)
        added.append(name)
    return tuple(added)


def prepare_contacts(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """Both fixes, in the order the second depends on the first.

    Every ``get_*_spec`` in :mod:`mjlab_microduck.robot.microduck_constants`
    runs this, so a robot cfg built anywhere in the repo carries the same,
    fully-named contact shell — and ``FULL_COLLISION``'s per-name rules finally
    address the geoms they were written for.
    """
    name_collision_geoms(spec)
    add_shell_ground_geoms(spec)
    return spec


def ground_collision_geom_names(spec: mujoco.MjSpec) -> tuple[str, ...]:
    """Names of the geoms that can touch the terrain, after :func:`prepare_contacts`."""
    return tuple(
        g.name
        for g in spec.geoms
        if g.name.endswith("_collision") and _classname(g) == COLLISION_CLASS
    )


def collision_geoms_on_bodies(
    spec: mujoco.MjSpec, body_names: tuple[str, ...]
) -> tuple[str, ...]:
    """Ground-collision geoms carried by the named bodies.

    Asked of the spec rather than pattern-matched on the geom name: the naming
    rule is an implementation detail of this module, and a classifier feature
    that silently stops matching after a re-export is the kind of bug that only
    shows up as a mode that stopped existing.
    """
    wanted = set(body_names)
    return tuple(
        g.name
        for g in spec.geoms
        if g.name.endswith("_collision")
        and _classname(g) == COLLISION_CLASS
        and g.parent.name in wanted
    )
