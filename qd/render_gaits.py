"""Render archive elites to video clips, for the interactive gait viewer.

    MUJOCO_GL=glfw uv run python -m qd.render_gaits \\
        --archive logs/qd/map_elites/archive_final.npz --out logs/qd/gaits/cpg

**The ``MUJOCO_GL`` prefix is required.** MuJoCo picks its GL backend when the
module is imported, so it has to be in the environment before Python starts —
setting it in code is too late. On this WSL2 box ``egl`` fails outright
(PyOpenGL finds no libEGL) and there is no OSMesa, but WSLg provides a real
display, so the ``glfw`` backend renders offscreen fine. On a genuinely
headless machine use ``osmesa`` instead.

Fidelity: the clips are **not** a CPU re-simulation. Each elite is rolled out in
the same batched MuJoCo-Warp harness that produced its archived fitness, with
``qpos`` logged every control step; rendering then replays those exact poses
through CPU MuJoCo. So what you watch is the trajectory that was scored, not an
approximation of it — the only thing CPU-side is the rasteriser.

Elites are rolled out one-per-world, so a whole archive costs a handful of
batched rollouts; the wall clock is dominated by rasterising frames.

**v5 (latent CVT) archives.** A ``qd.verify_aurora`` output has no grid: its
elites live in the Voronoi cells of a 4-d centroid set, and the file itself
carries no ``index`` (verification re-encodes into the frozen evaluation space,
where 970 elites share 40 cells). Pass the search archive as ``--cells`` and
every verified elite is placed in the cell it was *filed* in, matched through
``all_verified``. The clip is then rolled out in the harness verification used
— contact channels on, no fall latch, the full 7 s scored — so it is still the
trajectory that produced the median on the page, and ``trim_at_fall`` is
ignored: a crawl is below ``fall_height`` from its first step, and the scored
trajectory is the whole episode. Time upright is recomputed from the logged
poses under the v1–v3 rule (trunk above 0.075 m, tilt under 60°), so a walker
reads 7 s and a crawler ~0 s on the same scale as every other tab.

    MUJOCO_GL=glfw uv run python -m qd.render_gaits \\
        --archive logs/qd/v5/verify_v5/verified.npz \\
        --cells logs/qd/aurora_v5/final.npz --out logs/qd/v5/gaits/v5
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import tyro

from qd.common import FitnessCfg, load_archive
from qd.descriptors import DescriptorCfg


@dataclass
class Args:
    archive: Path
    out: Path = Path("logs/qd/gaits")

    cells: Path | None = None
    """For a ``qd.verify_aurora`` archive (no ``index``): the CVT archive whose
    ``index`` / ``centroids`` say which cell each elite was filed in."""

    top: int = 0
    """Render only the N best elites (0 = every filled cell)."""

    width: int = 320
    height: int = 240
    frame_stride: int = 2
    """Control steps per rendered frame; 2 gives 25 fps from the 50 Hz loop."""

    quality: int = 6
    """imageio/ffmpeg quality, 0-10. Lower is smaller."""

    camera_distance: float = 0.8
    camera_elevation: float = -12.0
    camera_azimuth: float = 135.0
    camera_lookat_z: float = 0.09
    """Fixed height for the camera target.

    The camera follows the trunk in x/y but NOT in z. Under v1's foot-only
    collision model the trunk frame ended up ~0.1 m *below* the floor after a
    face-plant and a z-tracking camera dived underground with it; walking-v2's
    full-collision model keeps the trunk above the plane, but a fixed target
    height still frames a fall better than one that follows a bouncing trunk.
    """

    max_envs: int = 128
    """Worlds per batched rollout; also the render chunk size."""

    device: str = "cuda:0"
    genome: str = "auto"
    """'cpg', 'mlp', or 'auto' (inferred from the solution width)."""

    full_collision: bool = True
    """Render under walking-v2 physics (every shell collides with the ground).

    Set ``False`` only to reproduce a v1 clip under v1 physics."""

    trim_at_fall: bool = True
    """Cut each clip on the frame that world's fall was detected.

    The clip is then exactly the trajectory that was scored — no frames the
    fitness, descriptor and replay buffer all refused to look at."""

    resume: bool = False
    """Continue an interrupted run from ``out/manifest.partial.json``.

    Batches whose clips and numbers are all on disk are kept as they are;
    the batch that was interrupted is redone in full. Nothing is stitched
    across rollouts: a kept clip and its manifest numbers always come from the
    same rollout, because a re-roll would NOT reproduce them — the batched
    contact solve differs between harness instances by metres, not
    millimetres, for a walker on a tipping point."""

    fitness: FitnessCfg = field(default_factory=FitnessCfg)


def _select(data: dict, top: int) -> np.ndarray:
    order = np.argsort(-data["objective"])
    return order[:top] if top else order


def _load(path: Path) -> dict:
    """:func:`qd.common.load_archive`, tolerating a ``qd.verify_aurora`` file.

    Grid checkpoints always carry ``meta_json``; a verification output does
    not, and it is the one file this renderer is asked for that never does.
    """
    with np.load(path, allow_pickle=False) as f:
        if "meta_json" not in f.files:
            return {k: f[k] for k in f.files} | {"meta": {}}
    return load_archive(path)


def _place_in_cells(data: dict, cells: Path | None) -> dict:
    """Cell, search-space latent and posture for every elite of a verified CVT archive.

    ``verified.npz`` is ``final.npz`` filtered by ``all_verified`` (checked here
    on the genomes, not assumed), so the search archive's ``index`` and
    ``measures`` carry over row by row. Posture is j017's DBSCAN mode, named by
    the v4 label its members carry (the purity table in ``summary.json``):
    nothing routes on the v4 classifier, it only names the cluster.
    """
    if cells is None:
        raise SystemExit(
            "this archive has no grid: pass --cells <the CVT archive it was verified from>"
        )
    from qd.modes import MODES

    with np.load(cells, allow_pickle=False) as f:
        rows = np.flatnonzero(data["all_verified"])
        if not np.array_equal(f["solution"][rows], data["solution"]):
            raise SystemExit(f"{cells} is not the archive this one was verified from")
        index = f["index"][rows]
        latent = f["measures"][rows]
        centroids = f["centroids"]
    if len(np.unique(index)) != len(index):
        raise SystemExit("two verified elites share a cell; the CVT archive is not one-per-cell")

    mode = np.asarray(data["mode"])
    postures: dict[str, str] = {}
    for m in np.unique(mode):
        labels = np.asarray(data["v4_label"])[mode == m]
        postures[str(int(m))] = MODES[int(np.bincount(labels, minlength=len(MODES)).argmax())]
    return {
        "index": index,
        "latent": latent,
        "centroids": centroids,
        "posture": [postures[str(int(m))] for m in mode],
        "postures": postures,
    }


def _rollout_with_qpos(
    args: Args,
    batch: np.ndarray,
    kind: str,
    descriptor: DescriptorCfg | None = None,
    latent: bool = False,
):
    """Roll a batch out and return ``(qpos, alive, fitness, measures, info, mj_model)``.

    ``qpos`` is ``(T, N, nq)`` and ``alive`` the matching ``(T, N)`` mask of
    which worlds were still upright on each frame. Walking-v2 cuts each clip at
    its own fall: a clip is the trajectory that was *scored*, and the scored
    trajectory ends at the fall.

    ``latent`` selects the harness :mod:`qd.verify_aurora` scored the archive
    in: contact channels on, no periodic all-fallen stop, and no fall latch, so
    the whole episode is simulated and scored for every world.
    """
    import torch

    descriptor = descriptor or DescriptorCfg()
    frames: list[np.ndarray] = []
    alive_frames: list[np.ndarray] = []

    def make_recorder(sim):
        def recorder(_phase, _step, alive):
            frames.append(sim.data.qpos.detach().cpu().numpy().copy())
            alive_frames.append(
                np.ones(len(batch), dtype=bool)
                if alive is None
                else alive.detach().cpu().numpy().copy()
            )

        return recorder

    if kind == "cpg":
        from qd.evaluate import CpgEvaluator, HarnessCfg, MicroduckRolloutHarness

        harness = MicroduckRolloutHarness(
            HarnessCfg(
                num_envs=len(batch),
                device=args.device,
                full_collision=args.full_collision,
                full_gait_stats=True,
            ),
            args.fitness,
            descriptor,
        )
        sim = harness.sim
        evaluator = CpgEvaluator(harness)
        fitness, measures, info = evaluator._evaluate_chunk(
            evaluator.space.clip(batch), recorder=make_recorder(sim)
        )
        mj_model = sim.mj_model
    else:
        from qd.pga.evaluate import PolicyHarnessCfg, PolicyRolloutHarness

        fitness = replace(args.fitness, latch_fall=False) if latent else args.fitness
        harness = PolicyRolloutHarness(
            PolicyHarnessCfg(
                num_envs=len(batch),
                device=args.device,
                full_collision=args.full_collision,
                full_gait_stats=True,
                mode_channels=latent,
                fall_check_every=0 if latent else PolicyHarnessCfg.fall_check_every,
            ),
            fitness,
            descriptor=descriptor,
        )
        sim = harness.env.sim
        genomes = torch.as_tensor(batch, dtype=torch.float32, device=args.device)
        fitness, measures, info, _ = harness.rollout(
            genomes, collect=False, recorder=make_recorder(sim)
        )
        mj_model = sim.mj_model

    return np.stack(frames), np.stack(alive_frames), fitness, measures, info, mj_model


def _upright_mask(mj_model, qpos: np.ndarray, fitness: FitnessCfg) -> np.ndarray:
    """``(T,)`` "would v1–v3 have called this frame upright?" from logged poses.

    Under the v5 harness nothing latches on a fall, so the harness's own
    ``alive`` mask is all-true for a crawler. This re-derives the v1–v3 rule
    (trunk above ``fall_height``, base +z within ``fall_tilt_deg`` of world +z)
    from the free-joint pose so every tab reports time upright on one scale.
    """
    import mujoco

    if mj_model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
        raise RuntimeError("expected a free root joint at qpos[0:7]")
    z = qpos[:, 2]
    qx, qy = qpos[:, 4], qpos[:, 5]
    # World-z component of the body's +z axis: R[2, 2] of the (w, x, y, z) quaternion.
    upright = 1.0 - 2.0 * (qx * qx + qy * qy)
    return (z >= fitness.fall_height) & (
        upright >= np.cos(np.radians(fitness.fall_tilt_deg))
    )


def _render_clip(
    mj_model, qpos: np.ndarray, args: Args, trunk_body: int
) -> np.ndarray:
    """``(T, nq)`` poses -> ``(F, H, W, 3)`` uint8 frames, camera tracking the trunk."""
    import mujoco

    data = mujoco.MjData(mj_model)
    camera = mujoco.MjvCamera()
    camera.distance = args.camera_distance
    camera.elevation = args.camera_elevation
    camera.azimuth = args.camera_azimuth

    out = []
    with mujoco.Renderer(mj_model, height=args.height, width=args.width) as renderer:
        for pose in qpos[:: args.frame_stride]:
            data.qpos[:] = pose
            mujoco.mj_forward(mj_model, data)
            camera.lookat[0:2] = data.xpos[trunk_body][0:2]
            camera.lookat[2] = args.camera_lookat_z
            renderer.update_scene(data, camera)
            out.append(renderer.render().copy())
    return np.stack(out)


def main(args: Args | None = None) -> None:
    import os

    if os.environ.get("MUJOCO_GL") in (None, "", "egl"):
        raise SystemExit(
            "Set MUJOCO_GL before starting Python, e.g.\n"
            "  MUJOCO_GL=glfw uv run python -m qd.render_gaits ...\n"
            "MuJoCo resolves its GL backend at import time, so setting it here "
            "would be too late. 'egl' has no libEGL on this box; 'glfw' works "
            "through WSLg, 'osmesa' on a headless machine."
        )

    args = args or tyro.cli(Args)
    import imageio.v2 as imageio
    import mujoco

    data = _load(args.archive)
    kind = args.genome
    if kind == "auto":
        kind = "cpg" if data["solution"].shape[1] < 100 else "mlp"

    latent = "grid_dims" not in data
    cvt = _place_in_cells(data, args.cells) if latent else None
    if latent:
        # The number on the page is the verified median; the filed objective
        # rides along as `filed_fitness` so the readout can show the optimism.
        data["filed_fitness"] = data["objective"]
        data["objective"] = np.asarray(data["median_displacement"], dtype=np.float64)
        if args.trim_at_fall:
            print(
                "latent archive: the whole episode was scored, so clips are not "
                "trimmed at the fall (--trim-at-fall ignored)",
                flush=True,
            )

    rows = _select(data, args.top)
    # The axes this archive was *built* on. A v1/v2 checkpoint carries no such
    # key and comes back as duty factor, which is what it was binned on.
    descriptor = DescriptorCfg.from_meta(data.get("meta"))
    dims = None if latent else tuple(int(x) for x in data["grid_dims"])
    clips_dir = args.out / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    fps = max(1, round(1.0 / (0.02 * args.frame_stride)))
    print(f"rendering {len(rows)} elites from {args.archive} ({kind}) at "
          f"{args.width}x{args.height}, {fps} fps", flush=True)

    entries: list[dict] = []
    partial = args.out / "manifest.partial.json"
    if args.resume and partial.exists():
        saved = json.loads(partial.read_text())
        if saved["max_envs"] != args.max_envs or saved["rows"] != [int(r) for r in rows]:
            raise SystemExit(f"{partial} was written with a different batch layout; drop --resume")
        entries = saved["elites"]
        print(f"resuming: {len(entries)} clips kept from {partial}", flush=True)

    for start in range(0, len(rows), args.max_envs):
        chunk = rows[start : start + args.max_envs]
        chunk_rows = {int(r) for r in chunk}
        kept = [e for e in entries if e["row"] in chunk_rows]
        if len(kept) == len(chunk) and all((args.out / e["clip"]).exists() for e in kept):
            continue
        entries = [e for e in entries if e["row"] not in chunk_rows]
        batch = data["solution"][chunk]
        qpos, alive, fitness, measures, info, mj_model = _rollout_with_qpos(
            args, batch, kind, descriptor, latent=latent
        )
        trunk_body = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_BODY, "robot/trunk_base"
        )
        if trunk_body < 0:
            raise RuntimeError("body 'robot/trunk_base' not found for camera tracking")

        for i, row in enumerate(chunk):
            if latent:
                cell = [int(cvt["index"][row])]
                name = f"cell_{cell[0]:04d}"
            else:
                rc = np.unravel_index(int(data["index"][row]), dims)
                cell = [int(rc[0]), int(rc[1])]
                name = f"cell_r{cell[0]:02d}_c{cell[1]:02d}"
            path = clips_dir / f"{name}.mp4"
            frames = qpos[:, i, :]
            if args.trim_at_fall and not latent:
                frames = frames[: int(alive[:, i].sum())]
            pixels = _render_clip(mj_model, frames, args, trunk_body)
            imageio.mimwrite(
                path, pixels, fps=fps, quality=args.quality, macro_block_size=1
            )
            entry = {
                "row": int(row),
                "cell": cell,
                "clip": f"clips/{path.name}",
                "bytes": path.stat().st_size,
                "archived_fitness": float(data["objective"][row]),
                "replay_fitness": float(fitness[i]),
                "displacement_m": float(info["displacement"][i]),
                # The archive's own two axes, whatever they are...
                "measure_x": float(measures[i, 0]),
                "measure_y": float(measures[i, 1]),
                # ...and duty factor regardless, so a v3 clip still reports
                # the quantity every earlier archive was indexed by and the
                # viewer can compare a v2 gait with a v3 one on it.
                "duty_left": float(info["axis/duty_left"][i]),
                "duty_right": float(info["axis/duty_right"][i]),
                "upright_s": float(info["alive_steps"][i] * 0.02),
                "survived": bool(~info["fell"][i]),
            }
            if latent:
                # Nothing latched, so `alive_steps` is the episode length for
                # every world; re-derive the v1-v3 count from the poses.
                settle = len(frames) - round(args.fitness.episode_seconds / 0.02)
                up = _upright_mask(mj_model, frames[settle:], args.fitness)
                entry["upright_s"] = float(up.sum() * 0.02)
                entry["survived"] = bool(up.all())
                entry["filed_fitness"] = float(data["filed_fitness"][row])
                entry["latent"] = [float(v) for v in cvt["latent"][row]]
                entry["eval_latent"] = [float(v) for v in data["latent"][row]]
                entry["posture"] = cvt["posture"][row]
                entry["viable_count"] = int(data["viable_count"][row])
            entries.append(entry)
        done = min(start + args.max_envs, len(rows))
        # Checkpoint after every batch: a crash (WSLg's GL has segfaulted 900
        # clips into a run) then costs one batch, and the kept numbers and
        # clips come from the same rollout, which a re-render could not
        # promise — the batched solve is not deterministic across instances.
        partial.write_text(
            json.dumps(
                {"max_envs": args.max_envs, "rows": [int(r) for r in rows], "elites": entries}
            )
        )
        print(f"  {done}/{len(rows)} clips", flush=True)
    order = {int(r): i for i, r in enumerate(rows)}
    entries.sort(key=lambda e: order[e["row"]])

    manifest = {
        "archive": str(args.archive),
        "genome": kind,
        "kind": "cvt" if latent else "grid",
        "grid_dims": None if latent else list(dims),
        # True when the archive came out of `qd.verify_archive`, i.e. its
        # objective column is a median over fresh replicas rather than the one
        # lucky sample the search inserted on. The viewer labels it accordingly:
        # calling a verified median "archived fitness" would hide the whole
        # point of verifying.
        # A `qd.verify_aurora` file IS the verified set: no meta, medians in
        # `median_displacement`, and nothing else in it was ever a sample.
        "verified": latent or bool((data.get("meta") or {}).get("verified", False)),
        "descriptor": {
            "axes": list(descriptor.names),
            "labels": list(descriptor.labels),
            "ranges": [list(r) for r in descriptor.ranges],
        },
        "episode_seconds": args.fitness.episode_seconds,
        "fps": fps,
        "resolution": [args.width, args.height],
        "elites": entries,
    }
    if latent:
        # The archive's own geometry: the viewer projects these centroids to
        # draw the map, and names the modes by the postures found here.
        manifest["cells"] = str(args.cells)
        manifest["centroids"] = cvt["centroids"].tolist()
        manifest["postures"] = cvt["postures"]
        summary = args.archive.with_name("summary.json")
        manifest["replicas"] = (
            int(json.loads(summary.read_text()).get("replicas", 8)) if summary.exists() else 8
        )
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    partial.unlink(missing_ok=True)
    total_mb = sum(e["bytes"] for e in entries) / 1e6
    print(
        f"\nwrote {len(entries)} clips to {clips_dir} ({total_mb:.1f} MB total, "
        f"median {np.median([e['bytes'] for e in entries]) / 1e3:.0f} KB)\n"
        f"manifest: {args.out / 'manifest.json'}"
    )


if __name__ == "__main__":
    main()
