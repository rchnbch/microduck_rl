"""Before/after evidence for the v4 shell-contact fix (design draft, Q4).

Three questions, three sections, all measured rather than asserted:

1. **What does the model actually compile to?** The geom table, legacy and
   fixed side by side: which geoms can touch the ground, and with what
   ``condim`` / ``priority`` / ``mu``. This is where the fix is visible as a
   fact rather than as a diff.
2. **Does it create contacts that were not there?** New collision geoms on the
   thighs and the trunk side shells can touch *each other* as well as the
   floor, and the velocity task charges a self-collision penalty. A permanent
   new self-contact would be a silent constant tax on every future PPO run.
3. **Where do the rest poses settle, and do they stay?** AGENTS.md: a rest pose
   must be a stable equilibrium, checked on TILT and not only on height, and a
   target height must never be carried across a model revision. The contact
   shell *is* a model revision, so every pose the probes spawn in is
   re-measured here.

Run::

    uv run python -m qd.check_shell_contacts --out logs/qd/shell_contacts
    uv run python -m qd.check_shell_contacts --sweep-friction 0.2 0.4 0.8

The friction sweep is the honest companion to a number nobody has measured on
hardware: it reports how much a scripted prone push actually depends on the
shell ``mu``, so the choice can be argued about with a sensitivity instead of a
citation.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import tyro

from qd.common import FitnessCfg, write_json


@dataclass
class Args:
    out: Path | None = None
    num_envs: int = 64
    device: str = "cuda:0"
    settle_seconds: float = 3.0
    """How long a rest pose is held before it is called stable.

    3 s, the duration AGENTS.md prescribes for an equilibrium check — long
    enough that a pose which is merely *slow* to topple is caught."""

    sweep_friction: tuple[float, ...] = ()
    """Shell ``mu`` values to re-measure the prone push at."""

    self_collision_samples: int = 2000
    """Random joint configurations used for the self-contact comparison."""

    tilt_tolerance_deg: float = 15.0
    """How far a rest pose may drift and still count as the same pose."""

    fitness: FitnessCfg = field(default_factory=FitnessCfg)


# --------------------------------------------------------------------------- #
# 1. The compiled geom table
# --------------------------------------------------------------------------- #


def _legacy_robot_cfg():
    """The robot cfg exactly as v1-v3 built it: unnamed shells, old rules.

    Reconstructed here rather than left behind a flag in production code —
    there is one contact model from v4 on, and this is a measurement
    instrument."""
    import mujoco
    from mjlab.utils.spec_config import CollisionCfg

    from mjlab_microduck.robot.microduck_constants import (
        MICRODUCK_ALLCOLLISIONS_XML,
        MICRODUCK_STANDUP_ROBOT_CFG,
    )

    legacy_collision = CollisionCfg(
        geom_names_expr=[".*_collision"],
        condim={r"^(left|right)_foot_collision$": 3, ".*_collision": 1},
        priority={r"^(left|right)_foot_collision$": 1},
        friction={r"^(left|right)_foot_collision$": (1.0,)},
    )
    return dataclasses.replace(
        MICRODUCK_STANDUP_ROBOT_CFG,
        spec_fn=lambda: mujoco.MjSpec.from_file(str(MICRODUCK_ALLCOLLISIONS_XML)),
        collisions=(legacy_collision,),
    )


def geom_table(robot_cfg) -> list[dict]:
    """Every geom that can touch the terrain in a compiled model."""
    import mujoco
    from mjlab.entity import Entity

    model = Entity(robot_cfg).spec.compile()
    rows = []
    for i in range(model.ngeom):
        contype = int(model.geom_contype[i])
        conaffinity = int(model.geom_conaffinity[i])
        # The terrain plane is contype/conaffinity 1; a geom that shares no bit
        # with it cannot touch the ground whatever else it collides with.
        if not (contype & 1 or conaffinity & 1):
            continue
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[i])
        rows.append(
            {
                "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "<unnamed>",
                "body": body,
                "condim": int(model.geom_condim[i]),
                "priority": int(model.geom_priority[i]),
                "mu": round(float(model.geom_friction[i][0]), 4),
            }
        )
    return rows


def compare_geom_tables() -> dict:
    from qd.evaluate import HarnessCfg, _deterministic_robot_cfg

    legacy = geom_table(_legacy_robot_cfg())
    fixed = geom_table(_deterministic_robot_cfg(HarnessCfg(full_collision=True)))
    legacy_names = {r["name"] for r in legacy}
    fixed_names = {r["name"] for r in fixed}
    return {
        "legacy": legacy,
        "fixed": fixed,
        "legacy_ground_geoms": len(legacy),
        "fixed_ground_geoms": len(fixed),
        "legacy_named": sorted(n for n in legacy_names if n != "<unnamed>"),
        "added_geoms": sorted(fixed_names - legacy_names - {"<unnamed>"}),
        "legacy_unnamed_count": sum(1 for r in legacy if r["name"] == "<unnamed>"),
        "fixed_unnamed_count": sum(1 for r in fixed if r["name"] == "<unnamed>"),
    }


def print_geom_tables(cmp: dict) -> None:
    print("\n=== 1. Ground-contact geoms, as compiled ===")
    for label in ("legacy", "fixed"):
        rows = cmp[label]
        print(f"\n-- {label}: {len(rows)} geoms that can touch the terrain --")
        print(f"   {'name':40s} {'body':18s} condim prio    mu")
        for r in rows:
            print(
                f"   {r['name']:40s} {r['body']:18s} {r['condim']:6d} "
                f"{r['priority']:4d} {r['mu']:6.3f}"
            )
    print(
        f"\n   legacy: {cmp['legacy_unnamed_count']} of {cmp['legacy_ground_geoms']} "
        f"unnamed -> unreachable by FULL_COLLISION's per-name rules; "
        f"named were {cmp['legacy_named']}"
    )
    print(f"   added by the fix: {cmp['added_geoms']}")


# --------------------------------------------------------------------------- #
# 2 & 3. What the fixed model does when it is actually simulated
# --------------------------------------------------------------------------- #


def rest_pose_report(args: Args, legacy: bool = False) -> dict:
    """Settle each spawn pose and report where it lands and what carries it.

    With ``legacy`` the same poses are settled on the *unfixed* contact shell,
    which is the before half of the before/after: a prone robot resting on
    thighs that have no collision geom is resting on nothing.
    """
    from qd import spawn
    from qd.evaluate import HarnessCfg, MicroduckRolloutHarness

    cfg = HarnessCfg(
        num_envs=args.num_envs,
        device=args.device,
        mode_channels=True,
        njmax=192,
    )
    if legacy:
        # The legacy model cannot even *build* the per-geom sensor: nine of its
        # eleven ground-contact geoms have no name, so there is nothing to
        # address them by. That failure is part of the finding — on the model
        # v1-v3 ran, "which part of the robot is touching the floor" was not an
        # observable quantity — so the legacy column reports height and tilt
        # only.
        harness = _legacy_harness(
            dataclasses.replace(cfg, mode_channels=False), args.fitness
        )
        names = ()
    else:
        harness = MicroduckRolloutHarness(cfg, args.fitness)
        names = harness.contact_columns[0]
    steps = round(args.settle_seconds / harness.control_dt)
    out: dict[str, dict] = {}
    for pose_name, pose in spawn.POSES.items():
        harness.reset(pose)
        targets = harness.spawn_servo_targets
        z0 = harness.base_pos()[:, 2].clone()
        # Contacts are counted over the SECOND half only: the first half is the
        # drop onto the floor, and a transient touch during a landing is not
        # "what carries this pose".
        contact_steps = torch.zeros(
            harness.num_envs, len(names), device=harness.device
        )
        z_mid = z0
        for k in range(steps):
            harness.set_servo_targets(targets)
            harness.step()
            if k == steps // 2:
                z_mid = harness.base_pos()[:, 2].clone()
            if k >= steps // 2 and names:
                ch = harness.mode_channels()
                assert ch is not None
                contact_steps += (ch["contact_found"] > 0).float()
        z1 = harness.base_pos()[:, 2]
        grav = harness.projected_gravity()
        up = -grav[:, 2]
        tilt_deg = torch.rad2deg(torch.arccos(up.clamp(-1.0, 1.0)))
        frac = (contact_steps / max(steps - steps // 2, 1)).mean(dim=0)
        carrying = {
            names[i]: round(float(frac[i]), 3)
            for i in range(len(names))
            if float(frac[i]) > 0.05
        }
        out[pose_name] = {
            "spawn_z_m": round(float(z0.mean()), 5),
            "z_at_half_m": round(float(z_mid.mean()), 5),
            "settled_z_m": round(float(z1.mean()), 5),
            "settled_z_sd_m": round(float(z1.std()), 5),
            "drift_m": round(float((z1 - z_mid).abs().mean()), 5),
            "tilt_deg_mean": round(float(tilt_deg.mean()), 2),
            "tilt_deg_max": round(float(tilt_deg.max()), 2),
            "upright_cos_mean": round(float(up.mean()), 4),
            "carried_by": carrying,
        }
    harness.close()
    return out


def _legacy_harness(cfg, fitness):
    """A rollout harness on the pre-v4 contact shell, for the before column."""
    import dataclasses as _dc

    from qd import evaluate as _ev

    original = _ev._deterministic_robot_cfg
    try:
        _ev._deterministic_robot_cfg = lambda _c: _dc.replace(
            _legacy_robot_cfg(),
            articulation=_dc.replace(
                _legacy_robot_cfg().articulation,
                actuators=tuple(
                    _dc.replace(
                        act,
                        vin=cfg.vin,
                        vin_range=None,
                        vin_drop_gain_range=None,
                        delay_min_lag=cfg.command_lag,
                        delay_max_lag=cfg.command_lag,
                    )
                    for act in _legacy_robot_cfg().articulation.actuators
                ),
            ),
        )
        # The legacy model has no named shells, so the per-geom sensor can only
        # address the two soles; the harness resolves whatever exists.
        return _ev.MicroduckRolloutHarness(cfg, fitness)
    finally:
        _ev._deterministic_robot_cfg = original


def _count_self_contacts(model, data, mujoco) -> tuple[int, dict[str, int]]:
    n = 0
    pairs: dict[str, int] = {}
    for c in range(data.ncon):
        con = data.contact[c]
        if model.geom_bodyid[con.geom1] == 0 or model.geom_bodyid[con.geom2] == 0:
            continue  # world / plane: not a self-contact
        n += 1
        g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1) or "<unnamed>"
        g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2) or "<unnamed>"
        key = " x ".join(sorted((g1, g2)))
        pairs[key] = pairs.get(key, 0) + 1
    return n, pairs


def self_collision_report(args: Args) -> dict:
    """Does the completed contact shell create self-contacts that were not there?

    New collision geoms on the thighs and the trunk side shells can touch *each
    other*, and six env cfgs in this repo charge a ``self_collision`` penalty —
    so a new permanent self-contact would be a silent constant tax on every
    future PPO run, surfacing as "the reward baseline moved" long after anyone
    remembers why.

    Uniform-over-limits sampling answers the wrong question: it visits joint
    combinations a tanh-bounded policy around HOME never reaches, and it will
    find contacts in *any* model detailed enough to have them. So the sweep is
    run at several distances from HOME, and the number that matters is the one
    at the radius policies actually occupy. The uniform row stays in the table
    as the worst case.
    """
    import mujoco
    from mjlab.entity import Entity

    from qd.evaluate import HarnessCfg, _deterministic_robot_cfg

    hinges = None
    out: dict[str, dict] = {}
    for label, robot_cfg in (
        ("legacy", _legacy_robot_cfg()),
        ("fixed", _deterministic_robot_cfg(HarnessCfg(full_collision=True))),
    ):
        model = Entity(robot_cfg).spec.compile()
        data = mujoco.MjData(model)
        if hinges is None:
            hinges = [
                (j, int(model.jnt_qposadr[j]), *model.jnt_range[j])
                for j in range(model.njnt)
                if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
            ]
        home = np.array(model.qpos0, dtype=float)
        rows: dict[str, dict] = {}
        for sigma in (0.0, 0.1, 0.3, 0.6, None):
            rng = np.random.default_rng(0)
            n_samples = 1 if sigma == 0.0 else args.self_collision_samples
            counts, pairs = [], {}
            for _ in range(n_samples):
                data.qpos[:] = home
                data.qpos[2] = 1.0  # clear of the plane: only self-contacts remain
                for _j, adr, lo, hi in hinges:
                    if hi <= lo:
                        continue
                    if sigma is None:
                        data.qpos[adr] = rng.uniform(lo, hi)
                    elif sigma > 0:
                        data.qpos[adr] = np.clip(
                            home[adr] + rng.normal(0.0, sigma), lo, hi
                        )
                mujoco.mj_forward(model, data)
                n, p = _count_self_contacts(model, data, mujoco)
                counts.append(n)
                for k, v in p.items():
                    pairs[k] = pairs.get(k, 0) + v
            key = "HOME" if sigma == 0.0 else (
                "uniform over limits" if sigma is None else f"HOME + N(0, {sigma})"
            )
            rows[key] = {
                "mean_self_contacts": round(float(np.mean(counts)), 4),
                "max_self_contacts": int(np.max(counts)),
                "fraction_with_any": round(float(np.mean(np.asarray(counts) > 0)), 4),
                "top_pairs": dict(sorted(pairs.items(), key=lambda kv: -kv[1])[:4]),
            }
        out[label] = rows
    out["samples"] = args.self_collision_samples
    return out


def friction_sweep(args: Args) -> dict:
    """How far a scripted prone push travels, as a function of shell ``mu``.

    The honest companion to a literature value: it says how much the choice
    matters before any policy is trained on it."""
    from qd import probes, spawn
    from qd.evaluate import HarnessCfg, MicroduckRolloutHarness

    results = {}
    # The tuned chin drag, not the hand-written one: the hand-written probes
    # crawl backwards, and a sensitivity measured on a probe that does not
    # crawl says nothing about how much crawling depends on shell friction.
    probe = probes.by_name("crawl_chin_drag_tuned")
    for mu in args.sweep_friction:
        harness = MicroduckRolloutHarness(
            HarnessCfg(
                num_envs=args.num_envs,
                device=args.device,
                mode_channels=True,
                njmax=192,
                shell_friction=float(mu),
            ),
            args.fitness,
        )
        feats, _ = probes.run_probe(harness, probe, spawn.get(probe.spawn))
        results[f"mu={mu:g}"] = {
            "median_displacement_m": round(float(np.median(feats.displacement)), 4),
            "worst_window_m": round(
                float(np.median(feats.window_dx[1:].min(axis=0))), 4
            ),
            "f_body": round(float(np.median(feats.f_body)), 3),
            "p95_az": round(float(np.median(feats.p95_az)), 2),
        }
        harness.close()
    return results


def main(args: Args | None = None) -> None:
    args = args or tyro.cli(Args)
    report: dict = {}

    cmp = compare_geom_tables()
    print_geom_tables(cmp)
    report["geoms"] = cmp

    print("\n=== 2. Self-collision, by distance from HOME ===")
    report["self_collision"] = self_collision_report(args)
    keys = list(report["self_collision"]["legacy"])
    print(f"   {'joint sampling':24s} {'legacy mean':>12s} {'fixed mean':>11s} "
          f"{'legacy any':>11s} {'fixed any':>10s}")
    for k in keys:
        lg = report["self_collision"]["legacy"][k]
        fx = report["self_collision"]["fixed"][k]
        print(
            f"   {k:24s} {lg['mean_self_contacts']:12.3f} "
            f"{fx['mean_self_contacts']:11.3f} {lg['fraction_with_any']:11.3f} "
            f"{fx['fraction_with_any']:10.3f}"
        )
    worst = report["self_collision"]["fixed"]["uniform over limits"]["top_pairs"]
    print(f"   pairs the fix introduces, worst case: {list(worst)[:3]}")

    print("\n=== 3. Rest poses, before and after the fix ===")
    report["rest_poses_legacy"] = rest_pose_report(args, legacy=True)
    report["rest_poses"] = rest_pose_report(args)
    for label, key in (("legacy", "rest_poses_legacy"), ("fixed", "rest_poses")):
        print(f"\n-- {label} --")
        print(
            f"   {'pose':8s} {'spawn z':>9s} {'settled z':>10s} {'drift':>8s} "
            f"{'tilt deg':>9s}  carried by"
        )
        for name, row in report[key].items():
            stable = row["tilt_deg_max"] - row["tilt_deg_mean"] < args.tilt_tolerance_deg
            print(
                f"   {name:8s} {row['spawn_z_m']:9.4f} {row['settled_z_m']:10.4f} "
                f"{row['drift_m']:8.4f} {row['tilt_deg_mean']:9.1f}  "
                f"{sorted(row['carried_by'])}"
                + ("" if stable else "   <-- SPREAD")
            )
    print("\n   sink (legacy settled z - fixed settled z), positive = the legacy")
    print("   model let the robot settle LOWER because a surface was missing:")
    for name in report["rest_poses"]:
        sink = (
            report["rest_poses_legacy"][name]["settled_z_m"]
            - report["rest_poses"][name]["settled_z_m"]
        )
        print(f"      {name:8s} {sink * 1000:+7.1f} mm")

    if args.sweep_friction:
        print("\n=== 4. Shell-friction sensitivity of a scripted prone push ===")
        report["friction_sweep"] = friction_sweep(args)
        for k, v in report["friction_sweep"].items():
            print(f"   {k}: {v}")

    if args.out:
        write_json(Path(args.out) / "shell_contacts.json", report)
        print(f"\nwrote {args.out}/shell_contacts.json")


if __name__ == "__main__":
    main()
