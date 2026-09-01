"""Inspect and play back an elite from a saved MAP-Elites archive.

List the best elites::

    uv run python -m qd.play_elite --archive logs/qd/map_elites/archive_final.npz --list

Replay one in the MuJoCo viewer (needs a display)::

    uv run python -m qd.play_elite --archive .../archive_final.npz --rank 0 --viewer

Or, headless, dump the gait for offline inspection (contact pattern, joint
traces, trunk path)::

    uv run python -m qd.play_elite --archive .../archive_final.npz --cell 6,14 \
        --save-npz /tmp/elite_6_14.npz

Selection is by ``--rank`` (0 = highest fitness), ``--cell R,C`` (archive grid
cell, nearest filled cell wins) or ``--index`` (raw row in the checkpoint).
The replay re-simulates the genome, so the printed fitness/descriptor is an
independent check that the archive entry reproduces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tyro

from qd.common import FitnessCfg, load_archive
from qd.evaluate import CpgEvaluator, HarnessCfg, MicroduckRolloutHarness


@dataclass
class Args:
    archive: Path
    """Path to an ``archive_*.npz`` written by ``qd.run_map_elites``."""

    rank: int | None = None
    """Select the n-th best elite (0 = best)."""

    cell: tuple[int, int] | None = None
    """Select the elite nearest to grid cell ``(row, col)``."""

    index: int | None = None
    """Select by raw row index in the checkpoint."""

    list: bool = False
    """Print the top elites and exit without simulating."""

    top: int = 20
    """How many rows ``--list`` prints."""

    viewer: bool = False
    """Replay the recorded trajectory in ``mujoco.viewer`` (needs a display)."""

    loops: int = 3
    """Viewer replays of the recorded trajectory."""

    save_npz: Path | None = None
    """Write the recorded qpos trajectory + contact/base traces here."""

    device: str = "cuda:0"
    num_envs: int = 8
    """Replay worlds. All run the identical genome; world 0 is recorded. Small
    is fine — the batch only exists because the harness is built for batches."""

    fitness: FitnessCfg = field(default_factory=FitnessCfg)


def _grid_shape(data: dict) -> tuple[int, int]:
    return tuple(int(x) for x in data["grid_dims"])  # type: ignore[return-value]


def _print_table(data: dict, top: int) -> None:
    obj = data["objective"]
    meas = data["measures"]
    idx = data["index"]
    dims = _grid_shape(data)
    order = np.argsort(-obj)[:top]
    print(f"{len(obj)} elites; grid {dims[0]}x{dims[1]}")
    print(f"{'rank':>4} {'row':>4} {'cell':>9} {'fitness_m':>10} {'duty_L':>7} {'duty_R':>7}")
    for rank, i in enumerate(order):
        cell = np.unravel_index(int(idx[i]), dims)
        print(
            f"{rank:>4} {int(i):>4} {tuple(int(c) for c in cell)!s:>9} "
            f"{obj[i]:>10.4f} {meas[i, 0]:>7.3f} {meas[i, 1]:>7.3f}"
        )


def _select(data: dict, args: Args) -> int:
    if args.index is not None:
        return int(args.index)
    if args.cell is not None:
        dims = _grid_shape(data)
        cells = np.stack(np.unravel_index(data["index"].astype(int), dims), axis=-1)
        target = np.asarray(args.cell)
        return int(np.argmin(np.abs(cells - target).sum(axis=-1)))
    return int(np.argsort(-data["objective"])[args.rank or 0])


def main(args: Args | None = None) -> None:
    args = args or tyro.cli(Args)
    data = load_archive(args.archive)

    if args.list:
        _print_table(data, args.top)
        return

    row = _select(data, args)
    genome = data["solution"][row]
    dims = _grid_shape(data)
    cell = np.unravel_index(int(data["index"][row]), dims)
    print(
        f"elite row {row}: cell {tuple(int(c) for c in cell)}  "
        f"archived fitness {data['objective'][row]:.4f} m  "
        f"duty (L,R) = ({data['measures'][row, 0]:.3f}, {data['measures'][row, 1]:.3f})"
    )

    harness = MicroduckRolloutHarness(
        HarnessCfg(num_envs=args.num_envs, device=args.device), args.fitness
    )
    evaluator = CpgEvaluator(harness)

    qpos_log: list[np.ndarray] = []
    contact_log: list[np.ndarray] = []
    base_log: list[np.ndarray] = []

    def recorder(phase: str, _step: int) -> None:
        qpos_log.append(harness.sim.data.qpos[0].detach().cpu().numpy().copy())
        contact_log.append(harness.foot_contact()[0].detach().cpu().numpy().copy())
        base_log.append(harness.base_pos()[0].detach().cpu().numpy().copy())

    fitness, measures, info = evaluator.replay(genome, recorder=recorder)
    print(
        f"replay:  fitness {fitness:.4f} m  duty (L,R) = "
        f"({measures[0]:.3f}, {measures[1]:.3f})  fell={bool(info['fell'])}  "
        f"alive {int(info['alive_steps'])}/{round(args.fitness.episode_seconds / harness.control_dt)} steps"
    )

    qpos = np.stack(qpos_log)
    contacts = np.stack(contact_log)
    base = np.stack(base_log)
    settle_steps = round(args.fitness.settle_seconds / harness.control_dt)

    if args.save_npz is not None:
        args.save_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.save_npz,
            genome=genome,
            qpos=qpos,
            foot_contact=contacts,
            base_pos=base,
            control_dt=harness.control_dt,
            settle_steps=settle_steps,
            fitness=fitness,
            measures=measures,
        )
        print(f"wrote {args.save_npz}")

    if args.viewer:
        _launch_viewer(harness, qpos, harness.control_dt, args.loops)


def _launch_viewer(
    harness: MicroduckRolloutHarness, qpos: np.ndarray, dt: float, loops: int
) -> None:
    import time

    import mujoco
    import mujoco.viewer

    model = harness.sim.mj_model
    mj_data = mujoco.MjData(model)
    with mujoco.viewer.launch_passive(model, mj_data) as viewer:
        for _ in range(loops):
            for frame in qpos:
                if not viewer.is_running():
                    return
                mj_data.qpos[:] = frame
                mujoco.mj_forward(model, mj_data)
                viewer.sync()
                time.sleep(dt)


if __name__ == "__main__":
    main()
