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
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tyro

from qd.descriptors import DescriptorCfg
from qd.common import FitnessCfg, load_archive


@dataclass
class Args:
    archive: Path
    out: Path = Path("logs/qd/gaits")

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

    fitness: FitnessCfg = field(default_factory=FitnessCfg)


def _select(data: dict, top: int) -> np.ndarray:
    order = np.argsort(-data["objective"])
    return order[:top] if top else order


def _rollout_with_qpos(
    args: Args, batch: np.ndarray, kind: str, descriptor: DescriptorCfg | None = None
):
    """Roll a batch out and return ``(qpos, alive, fitness, measures, info, mj_model)``.

    ``qpos`` is ``(T, N, nq)`` and ``alive`` the matching ``(T, N)`` mask of
    which worlds were still upright on each frame. Walking-v2 cuts each clip at
    its own fall: a clip is the trajectory that was *scored*, and the scored
    trajectory ends at the fall.
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

        harness = PolicyRolloutHarness(
            PolicyHarnessCfg(
                num_envs=len(batch),
                device=args.device,
                full_collision=args.full_collision,
                full_gait_stats=True,
            ),
            args.fitness,
            descriptor=descriptor,
        )
        sim = harness.env.sim
        genomes = torch.as_tensor(batch, dtype=torch.float32, device=args.device)
        fitness, measures, info, _ = harness.rollout(
            genomes, collect=False, recorder=make_recorder(sim)
        )
        mj_model = sim.mj_model

    return np.stack(frames), np.stack(alive_frames), fitness, measures, info, mj_model


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

    data = load_archive(args.archive)
    kind = args.genome
    if kind == "auto":
        kind = "cpg" if data["solution"].shape[1] < 100 else "mlp"

    rows = _select(data, args.top)
    # The axes this archive was *built* on. A v1/v2 checkpoint carries no such
    # key and comes back as duty factor, which is what it was binned on.
    descriptor = DescriptorCfg.from_meta(data.get("meta"))
    dims = tuple(int(x) for x in data["grid_dims"])
    clips_dir = args.out / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    fps = max(1, round(1.0 / (0.02 * args.frame_stride)))
    print(f"rendering {len(rows)} elites from {args.archive} ({kind}) at "
          f"{args.width}x{args.height}, {fps} fps", flush=True)

    entries = []
    for start in range(0, len(rows), args.max_envs):
        chunk = rows[start : start + args.max_envs]
        batch = data["solution"][chunk]
        qpos, alive, fitness, measures, info, mj_model = _rollout_with_qpos(
            args, batch, kind, descriptor
        )
        trunk_body = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_BODY, "robot/trunk_base"
        )
        if trunk_body < 0:
            raise RuntimeError("body 'robot/trunk_base' not found for camera tracking")

        for i, row in enumerate(chunk):
            cell = np.unravel_index(int(data["index"][row]), dims)
            name = f"cell_r{int(cell[0]):02d}_c{int(cell[1]):02d}"
            path = clips_dir / f"{name}.mp4"
            frames = qpos[:, i, :]
            if args.trim_at_fall:
                frames = frames[: int(alive[:, i].sum())]
            pixels = _render_clip(mj_model, frames, args, trunk_body)
            imageio.mimwrite(
                path, pixels, fps=fps, quality=args.quality, macro_block_size=1
            )
            entries.append(
                {
                    "row": int(row),
                    "cell": [int(cell[0]), int(cell[1])],
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
            )
        done = min(start + args.max_envs, len(rows))
        print(f"  {done}/{len(rows)} clips", flush=True)

    manifest = {
        "archive": str(args.archive),
        "genome": kind,
        "grid_dims": list(dims),
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
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    total_mb = sum(e["bytes"] for e in entries) / 1e6
    print(
        f"\nwrote {len(entries)} clips to {clips_dir} ({total_mb:.1f} MB total, "
        f"median {np.median([e['bytes'] for e in entries]) / 1e3:.0f} KB)\n"
        f"manifest: {args.out / 'manifest.json'}"
    )


if __name__ == "__main__":
    main()
