# `qd/` — Quality-Diversity gait discovery for Microduck

Gradient-free (and, in Phase 3, gradient-*assisted*) search for a **diverse
archive** of forward-walking gaits, on the same MuJoCo-Warp simulation the PPO
recipes train in.

Where PPO returns one policy that maximizes a reward, MAP-Elites returns a
**grid of policies**, one per bin of a behaviour descriptor, each the best of
its kind. Here the descriptor is per-foot ground-contact duty factor, so the
archive spans everything from a shuffle that never lifts a foot to a hopping
gait with both feet airborne most of the time.

```
qd/
├── common.py          fitness + behaviour descriptor + archive/plot/checkpoint helpers
├── cpg_genome.py      Phase 2 genome: 31-parameter open-loop CPG
├── evaluate.py        batched mjlab rollout harness: genomes -> (fitness, descriptor)
├── run_map_elites.py  Phase 2 CLI: pyribs GridArchive + GaussianEmitter ask/tell loop
├── play_elite.py      inspect / replay one elite from a saved archive
├── check_harness.py   physics sanity checks — run before any long run
└── bench.py           throughput + evaluation-noise benchmark: pick a batch size
```

Before the first long run on a new machine or after touching the MJCF:

```bash
uv run python -m qd.check_harness   # standing height, time-to-topple, descriptor spread
uv run python -m qd.bench           # ms/genome vs batch size, fitness noise
```

## Install

pyribs lives in an opt-in dependency group so the PPO/HF-Jobs dependency set is
untouched (`numba`/`scikit-learn`/`pandas` come along with `ribs[visualize]`,
and AGENTS.md is emphatic that a fresh `uv sync` is the ground truth for
training):

```bash
uv sync --group qd
```

## Run

```bash
# smoke test first — always. ~1 min, proves the loop end to end.
uv run python -m qd.run_map_elites --generations 5 --batch-size 32 \
    --initial-solutions 64 --out-dir logs/qd/smoke

# a real run
uv run python -m qd.run_map_elites --generations 200 --batch-size 1024 \
    --initial-solutions 2048 --out-dir logs/qd/map_elites
```

Batch size is close to free: a generation is a 350-step Python loop over the
whole batch, so on an RTX 3060 it costs ~6 s at 64 worlds and ~13 s at 2048
(95 → 6 ms per genome). Pick the largest batch that fits — 2048 worlds use
about 2 GB. `uv run python -m qd.bench` measures this for your GPU.

`qd/` is **not** part of the installed distribution (`pyproject.toml` builds
only `src/`), so there is no console script — `python -m` puts the repo root on
`sys.path`, which is what the `qd.*` imports need. Keeping it out of the wheel
is also the guarantee that nothing here can perturb a training run.

Outputs land in `--out-dir`:

| file | what |
| --- | --- |
| `archive_gen####.npz`, `archive_final.npz` | every elite: genome, fitness, descriptor, cell index |
| `heatmap_gen####.png`, `heatmap_final.png` | duty-factor heatmap coloured by fitness |
| `history.json` | per-generation coverage / QD-score / best fitness / fall rate |
| `summary.json` | final stats + the exact args the run used |

## Playing back a gait

```bash
A=logs/qd/map_elites/archive_final.npz

uv run python -m qd.play_elite --archive $A --list              # top elites + their cells
uv run python -m qd.play_elite --archive $A --rank 0 --viewer   # best gait, MuJoCo viewer
uv run python -m qd.play_elite --archive $A --cell 6,14 --viewer   # a specific archive cell
uv run python -m qd.play_elite --archive $A --rank 0 --save-npz /tmp/best.npz  # headless dump
```

`--viewer` replays the recorded `qpos` trajectory in `mujoco.viewer` (needs a
display; over WSL2 that means an X server). `--save-npz` is the headless path:
it writes `qpos`, per-step `foot_contact` and `base_pos`, which is enough to
plot a gait diagram or a trunk path without a GUI.

The replay **re-simulates** the genome rather than trusting the archive, so the
fitness and descriptor it prints are an independent reproduction check. Expect
them to land within a few millimetres of the archived values rather than on top
of them — see "Reproducibility" below.

To compare cells, pick two that are far apart on the grid, e.g. a
balanced-duty cell (`~0.6, ~0.6` — a walking gait, feet alternating) against a
lopsided one (`~1.0, ~0.2` — one foot planted, the other doing the work).

## Design

### Genome — open-loop CPG (31 parameters)

Ten leg joints (5 per leg; neck and head are pinned at HOME), each a sinusoid
sharing one global frequency:

```
target_j(t) = offset_j + amplitude_j * sin(2*pi*freq*t + phase_j)
```

`[freq | amplitude_0..9 | phase_0..9 | offset_0..9]`, in the servo order
`left_{hip_yaw,hip_roll,hip_pitch,knee,ankle}` then the same five on the right.
The layout is **blocked**, not interleaved, so a whole batch evaluates as three
contiguous slices.

Why open-loop: it makes Phase 2 a pure black-box search — no observations, no
network, no normalizer — which isolates the QD machinery from the policy
machinery. Phase 3 swaps in a closed-loop MLP over the repo's 61-D observation
contract and keeps everything else.

Bounds: `freq ∈ [0.5, 3.0] Hz`; `amplitude_j ∈ [0, 0.25 * range_j]`;
`phase_j ∈ [0, 2π]`; `offset_j` inside the joint's *soft* limits (MJCF range
shrunk 0.9 about its midpoint, matching `soft_joint_pos_limit_factor`). Limits
are read from `robot_walk.xml` at import time rather than copied into a
constant, so an MJCF re-export can't silently invalidate them.

Bounds are enforced **twice** — per-dimension `bounds=` on the emitter, and a
defensive `np.clip` in `CpgEvaluator.evaluate` plus a `clamp` on every write to
the sim. Without the second layer an out-of-range target is silently absorbed
by the actuator model and the archive fills with entries that don't reproduce.

### Fitness

```
fitness = forward_displacement_at_fall - fall_penalty * fraction_of_episode_fallen
```

Forward (+x) displacement of the trunk over a 7 s rollout at 50 Hz, after a
0.25 s settle at HOME (so the spawn drop isn't scored).

`robot_walk.xml` has the trunk and head collision geoms stripped — only the
feet collide — so **a fall is not observable as a trunk-ground contact.** It is
detected from base state instead: trunk below `fall_height` (0.075 m — the
robot stands at a **measured** 0.115 m, see `qd/check_harness.py`), or the base
+z axis tilted more than `fall_tilt_deg` from world +z (read off
`projected_gravity_b`). Non-finite state counts as a fall too. On a fall the
displacement is **frozen** at the moment of detection, so a face-plant that
skids forward stops earning.

**Why the penalty is pro-rata rather than a flat subtraction.** On this robot a
passive HOME hold is *not* a stable equilibrium: with the servos simply holding
HOME the trunk droops forward and topples at **~1.34 s** (measured — the head is
~38% of body mass and the position loop sags ~0.17 rad under load). With no
control everything falls, so a constant penalty is a constant offset and the
search degenerates into "who dives forward fastest". Charging by *time spent
down* makes falling late strictly cheaper than falling early, so a gait that
stays up beats a ballistic dive that covers the same ground and lies there.
It is also rate-limited by construction — there is no state you can reach early
and then farm (AGENTS.md, "No jackpots").

### Behaviour descriptor — per-foot duty factor (2-D, 20×20)

`(left_foot_duty_factor, right_foot_duty_factor)`, each the fraction of control
steps that foot was in ground contact, from the same `feet_ground_contact`
sensor the velocity task uses (left slot first, right second). Cully et al.'s
hexapod descriptor, generalized to a biped.

Steps after a fall are excluded from the average — otherwise a robot lying on
its face with both soles touching would report duty `(1, 1)` and squat in that
corner of the archive.

### Reproducibility

Domain randomization is **off**. The BAM actuator keeps its physics (voltage
control, load-dependent friction — the joint-friction DR knob would be a no-op
under anything else, see AGENTS.md) but its randomization knobs are pinned:
fixed 7.4 V supply, no voltage-sag gain, and a fixed 4-step command lag (the
midpoint of the 3–6 range PPO trains under — the lag is real hardware latency,
not randomization, so it stays). Every world spawns at the identical HOME pose.

That removes every *deliberate* source of variation, but **the archive is still
not noise-free**: MuJoCo-Warp's batched contact/constraint solve is
order-sensitive, so re-running the same genome moves its fitness by up to
~4 mm (`uv run python -m qd.bench` measures it; on an RTX 3060 the spread over
repeated evaluations of one elite ran 3e-5 to 4.4e-3 m). MAP-Elites keeps the
best sample per cell, so the archive is mildly *optimistic* — an elite's
replayed fitness is on average a hair below its archived one. The usual fixes
are re-evaluating on insertion or storing a running mean per cell; neither is
implemented here, and the effect is small next to the fitness spread across
cells. Worth knowing before reading two archives as exactly comparable.

`sim.expand_model_fields(("dof_frictionloss", "dof_damping"))` is called after
scene initialization: BAM writes a per-env friction budget into those model
fields every step, and mjlab only allocates them per-world on request. This is
the standalone-harness equivalent of the `expand_bam_friction_fields` startup
event every env cfg in this repo registers.

### Batched evaluation

One genome per parallel world; the CPG target trajectory for the whole
generation is precomputed on the GPU as a `(T, B, 10)` tensor, so the rollout
loop does no host↔device copies. A generation is a single batched rollout.
`--num-envs` defaults to `--batch-size`, so one generation is exactly one
rollout. Larger batches are chunked and smaller ones padded (the CUDA graph
pins the world count at build time, so the harness is built once and reused for
the whole run) — which is why raising `--batch-size` is the way to use more of
the GPU, not raising `--num-envs`.

## Tests

`tests/test_qd_cpg_genome.py` and `tests/test_qd_common.py` are CPU-only and
run with the rest of the suite:

```bash
uv run --with pytest pytest tests/ -q
```

They lock the things that rot silently: the leg-joint set resolving **by name**
against the actual model in the documented servo order (`Entity.joint_names`
interleaves neck/head between the legs, so a hard-coded `0..9` slice grabs the
head), the two-layer bound enforcement, the fall latch, and the descriptor
staying inside `[0, 1]²` where the `GridArchive` can see it.

## Later upgrades (not built)

* **CMA-ME** — swap the `GaussianEmitter` for pyribs'
  `EvolutionStrategyEmitter`, which adapts a covariance matrix per emitter and
  usually beats isotropic mutation badly on a 31-D search space. Deliberately
  out of scope here: the point of Phase 2 is a vanilla MAP-Elites baseline for
  Phase 3 to be measured against.
* Richer descriptors (stride length, lateral drift, energy) or a 3-D archive.
* CPG coupling terms (a phase-locked oscillator network) instead of independent
  per-joint phases.
