"""Does anything end up under the floor? The honest-physics acceptance check.

v1's rollouts kept simulating after the fall on a model whose only ground
contacts were the two foot soles, so a toppled robot's trunk and head passed
straight through the plane — visibly, in every clip of a falling elite. Walking-
v2 changes both halves of that (all-collisions model, rollout stops at the
fall), and this measures whether it worked rather than asserting it.

Two metrics, because the cheap one is not conclusive on its own:

* a **screen** run every control step: the lowest ``geom_xpos.z - rbound`` over
  all geoms. ``rbound`` encloses the whole mesh in a sphere, so a foot resting
  normally on the plane already scores about -0.03 m. Useful for finding the
  worst frame, useless as an absolute verdict.
* the **exact lowest mesh vertex** on that worst frame, computed on CPU by
  transforming every geom's vertices into the world. This one *is* an absolute
  verdict: negative means part of the robot was below z=0.

``--dump-frames`` writes the worst frame of each configuration as a PNG, which
is the acceptance criterion in its original form — look at the picture.

Run it on genomes that actually fall — random MLPs do, and that is the point:

    MUJOCO_GL=glfw uv run python -m qd.check_floor --num-envs 64 --dump-frames out/
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import tyro

from qd.common import FitnessCfg, load_archive, write_json
from qd.pga.evaluate import PolicyHarnessCfg, PolicyRolloutHarness
from qd.pga.policy_genome import DEFAULT_SPEC

CONFIGS = {
    "v1 (feet only, runs past the fall)": {
        "full_collision": False,
        "fall_check_every": 0,
    },
    "v2 (full collision, stops at the fall)": {
        "full_collision": True,
        "fall_check_every": 25,
    },
}


@dataclass
class Args:
    archive: Path | None = None
    """Take genomes from an archive; random MLPs (which fall) if omitted."""

    num_envs: int = 64
    device: str = "cuda:0"
    seed: int = 0
    out: Path | None = None
    dump_frames: Path | None = None
    """Render each configuration's worst frame here (needs MUJOCO_GL set)."""

    fitness: FitnessCfg = field(default_factory=FitnessCfg)


def exact_lowest_vertex(mj_model, qpos: np.ndarray) -> tuple[float, str]:
    """Lowest world-z of any geom vertex at ``qpos``, and the geom it belongs to.

    Meshes are transformed vertex by vertex; primitives fall back to their
    bounding radius, which for this robot is only the terrain plane.
    """
    import mujoco

    data = mujoco.MjData(mj_model)
    data.qpos[:] = qpos
    mujoco.mj_forward(mj_model, data)

    best, best_name = float("inf"), ""
    for gid in range(mj_model.ngeom):
        if mj_model.geom_bodyid[gid] == 0:  # world/terrain
            continue
        pos = data.geom_xpos[gid]
        mat = data.geom_xmat[gid].reshape(3, 3)
        mesh_id = mj_model.geom_dataid[gid]
        if mj_model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH and mesh_id >= 0:
            start = mj_model.mesh_vertadr[mesh_id]
            verts = mj_model.mesh_vert[start : start + mj_model.mesh_vertnum[mesh_id]]
            z = float((verts @ mat.T + pos)[:, 2].min())
        else:
            z = float(pos[2] - mj_model.geom_rbound[gid])
        if z < best:
            best = z
            best_name = (
                mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom{gid}"
            )
    return best, best_name


def _render(mj_model, qpos: np.ndarray, path: Path) -> None:
    import imageio.v2 as imageio
    import mujoco

    data = mujoco.MjData(mj_model)
    data.qpos[:] = qpos
    mujoco.mj_forward(mj_model, data)
    camera = mujoco.MjvCamera()
    camera.distance, camera.elevation, camera.azimuth = 0.5, -6.0, 120.0
    # The free joint's position, not a body index: worlds are laid out at env
    # origins, so body 1 is not where this particular robot is.
    camera.lookat[:] = [qpos[0], qpos[1], 0.0]
    with mujoco.Renderer(mj_model, height=480, width=640) as renderer:
        renderer.update_scene(data, camera)
        path.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(path, renderer.render())


def main(args: Args | None = None) -> None:
    args = args or tyro.cli(Args)
    spec = DEFAULT_SPEC

    if args.archive is not None:
        data = load_archive(args.archive)
        rows = np.argsort(-data["objective"])[: args.num_envs]
        source = f"{args.archive} (top {len(rows)} elites)"
        genomes_np, num_envs = data["solution"][rows], len(rows)
    else:
        source = f"{args.num_envs} random MLPs (they fall — that is the point)"
        genomes_np, num_envs = None, args.num_envs

    results = {}
    for label, overrides in CONFIGS.items():
        harness = PolicyRolloutHarness(
            PolicyHarnessCfg(num_envs=num_envs, device=args.device, **overrides),
            args.fitness,
            spec,
        )
        mj_model = harness.env.sim.mj_model
        rbound = torch.as_tensor(mj_model.geom_rbound, device=args.device)
        generator = torch.Generator(device=args.device).manual_seed(args.seed)
        genomes = (
            spec.initial_population(num_envs, generator, args.device)
            if genomes_np is None
            else torch.as_tensor(genomes_np, dtype=torch.float32, device=args.device)
        )

        state = {"worst": float("inf"), "qpos": None, "worst_scored": float("inf")}

        # `harness` and `rbound` are bound as defaults, not captured: the loop
        # rebinds (and deletes) them, and a late-bound closure would read the
        # next configuration's harness.
        def recorder(phase, _step, alive, state=state, sim=harness.env.sim, rbound=rbound):
            z = (sim.data.geom_xpos[..., 2] - rbound).min(dim=-1).values
            # Only frames a clip would show: the settle, then the upright part.
            visible = z if phase == "settle" else torch.where(
                alive, z, torch.full_like(z, float("inf"))
            )
            worst_env = int(torch.argmin(z))
            if float(z[worst_env]) < state["worst"]:
                state["worst"] = float(z[worst_env])
                state["qpos"] = (
                    sim.data.qpos[worst_env].detach().cpu().numpy().copy()
                )
            state["worst_scored"] = min(state["worst_scored"], float(visible.min()))

        _, _, info, _ = harness.rollout(genomes, collect=False, recorder=recorder)
        exact_z, exact_geom = exact_lowest_vertex(mj_model, state["qpos"])
        results[label] = {
            "screen_lowest_point_m": state["worst"],
            "screen_lowest_point_rendered_frames_m": state["worst_scored"],
            "exact_lowest_vertex_m": exact_z,
            "exact_lowest_vertex_geom": exact_geom,
            "survivors": int((~info["fell"]).sum()),
        }
        print(
            f"{label:<40} worst frame: screen {state['worst']:+.4f} m, "
            f"EXACT lowest vertex {exact_z:+.4f} m ({exact_geom})",
            flush=True,
        )
        if args.dump_frames is not None:
            name = "v1" if "v1" in label else "v2"
            path = args.dump_frames / f"worst_frame_{name}.png"
            _render(mj_model, state["qpos"], path)
            results[label]["frame"] = str(path)
            print(f"  wrote {path}", flush=True)
        del harness
        torch.cuda.empty_cache()

    payload = {"source": source, "num_envs": num_envs, "results": results}
    if args.out is not None:
        write_json(args.out, payload)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
