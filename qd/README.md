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
├── bench.py           throughput + evaluation-noise benchmark: pick a batch size
├── survival_report.py   re-evaluate the top elites: survival + archive optimism
├── compare_archives.py  side-by-side CPG vs PGA-ME comparison
├── render_gaits.py      rollout -> mp4 clips of each elite's gait
├── build_viewer.py      the clickable archive viewer page
└── pga/               Phase 3: PGA-MAP-Elites (see below)
```

Before the first long run on a new machine or after touching the MJCF:

```bash
uv run python -m qd.check_harness   # standing height, time-to-topple, descriptor spread
uv run python -m qd.bench           # ms/genome vs batch size, fitness noise
```

After a run, to see whether the elites actually *walk* rather than just score
well — the pro-rata penalty means a high fitness can come either from covering
ground fast before falling or from staying up, and the archive does not record
which:

```bash
uv run python -m qd.survival_report --archive logs/qd/map_elites/archive_final.npz
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

## Results

Both pipelines run to **~207,000 evaluations** on one RTX 3060 (MAP-Elites
206,848 in 31 min; PGA-ME 207,048 in 74 min), same archive, same objective,
same descriptor, same deterministic-actuator physics.

**Every number in the "honest" block is a re-evaluation.** MAP-Elites stores the
luckiest sample per cell and this simulator is not bit-reproducible, so archived
fitness is biased upward — by +0.002 m for the CPG and **+0.133 m for the MLP**,
a 60× difference (see "Reproducibility"). Comparing the two archives on archived
values would have handed PGA-ME a 3.3× win on peak fitness that does not survive
re-running the genomes. Reproduce with
`uv run python -m qd.compare_archives --a ... --b ...`.

| metric | MAP-Elites (CPG) | PGA-ME (MLP) |
| --- | --- | --- |
| **Honest — every elite re-evaluated** | | |
| best-cell fitness | +0.143 m | **+0.212 m** |
| QD-score | **1679.8** | 1441.4 |
| mean fitness | −0.059 m | −0.064 m |
| positive-fitness elites | 67 | **84** |
| survived the full 7 s | 0 | **1** |
| longest upright | 2.10 s | **7.00 s** |
| median upright | 0.80 s | **1.04 s** |
| furthest travelled | +0.339 m | **+0.403 m** |
| **furthest by a survivor** | **—** | **+0.088 m** |
| *Archived — optimistic, for reference* | | |
| archived best-cell fitness | +0.153 m | +0.504 m |
| archived QD-score | 1680.6 | 1480.4 |
| archive optimism (mean) | +0.002 m | +0.133 m |
| **Structure** | | |
| elites | 340 | 292 |
| coverage | 85.0% | 73.0% |
| cells the other never reached | 48 | 0 |

![archive comparison](../logs/qd/comparison/comparison.png)

### The headline: it balances, or it travels — never both

**The single policy that stays upright for the full 7 s covers 8.8 cm.** That is
1.3 cm/s, about a third of a body length in the whole episode — standing still
with extra steps. The elite that travels furthest (+0.403 m) falls before the
episode ends. Nothing in either archive both stays up and goes anywhere.

That is the real state of this work, and it is worth stating plainly because
every aggregate metric above hides it: fitness rewards distance *and* staying
up, so a high score can come from either, and only the survivor-displacement row
separates them.

### What PGA-ME won and lost

**Won: peak quality and the ability to stay upright at all.** Best-cell fitness
+0.212 m vs +0.143 m, 84 positive-fitness elites vs 67, and — the qualitative
difference — 561 full-episode survivors across the run against **zero, ever**,
for the CPG. A passive HOME hold topples at 1.34 s; the open-loop CPG stretches
that to 2.10 s and no further, because an open-loop genome has no feedback and
*structurally cannot* correct a topple. A closed-loop policy can, and does.

**Lost: coverage and QD-score.** 73.0% vs 85.0%, and QD 1441 vs 1680. The
archive is also visibly noisier — the replay heatmap is speckled with dark cells
where the CPG's is smooth, which is the reactivity amplifying simulator
non-determinism, rendered as an image. And 48 cells that the CPG filled were
never reached by PGA-ME, while PGA-ME reached none the CPG missed.

So against the acceptance criterion — *match or beat on QD-score **and**
best-cell fitness* — PGA-ME **beats on best-cell fitness and loses on
QD-score**, at matched budget. Not a clean win, and the two are good at
different things: gradient variation concentrates quality where the critic can
find it, and pays for that in exploration.

### Per-operator insertion rates

Logged every iteration (`history.json`) and averaged in `summary.json`. Over the
matched run: **GA 3.7%, PG 1.8%**, greedy actor inserted 3 times. Both operators
contribute — PG is not the ~zero that would mean the critic or the reward wiring
is broken. Both start far higher (PG 95% at iteration 2) and decay together as
the archive saturates, which is the expected shape; PG runs at roughly half GA's
rate throughout.

## Phase 3 — PGA-MAP-Elites (`qd/pga/`)

Same 20×20 duty-factor archive, same objective, same behaviour descriptor. What
changes is the genome and how offspring are made — which is exactly what makes
the two archives comparable.

```
qd/pga/
├── policy_genome.py  MLP 61->64->64->14 (tanh), flattened to a 9038-param vector
├── evaluate.py       batched rollout on the STRIPPED Velocity env + transition collection
├── td3.py            replay buffer, twin critics, target nets, greedy actor
├── variation.py      iso+lineDD directional GA variation, and PG variation
└── run_pga_me.py     the iteration loop
```

```bash
uv run python -m qd.pga.run_pga_me --iterations 200 --batch-size 100
```

### Genome — a closed-loop MLP

`61 → 64 → 64 → 14`, tanh everywhere, flattened to **9038 parameters**. The 61
inputs are the repo's shared observation contract (48 proprioception + the 13-D
command block, commands pinned at zero), so an evolved policy stays compatible
with the existing ONNX export and runtime. tanh on the *output* bounds actions
to ±1 rad around HOME, which is both a sane range for this robot and what makes
TD3 correct — target-policy smoothing needs a bounded action space to clip its
noise against.

A whole population is one `(P, 9038)` tensor and one batched forward
(`einsum` over a per-policy weight axis), because a generation is evaluated in
one rollout and PG variation trains ~50 offspring simultaneously.

### Observations come from the real env, stripped

Phase 2 drives mjlab's low-level `Scene`/`Simulation`; Phase 3 needs the 61-D
observation pipeline and hand-rolling it is the mistake AGENTS.md warns about.
So `qd/pga/evaluate.py` builds the actual `Mjlab-Velocity-Flat-MicroDuck` cfg
and removes what a QD evaluation must not have:

| removed | why |
| --- | --- |
| all 9 DR event terms | a genome must have one fitness |
| observation corruption | same |
| spawn jitter and random yaw | "+x displacement" is meaningless from a random heading |
| all command ranges (→ 0) | the deployment idle state, and Phase 2's condition |
| curricula | they re-widen the ranges that were just zeroed |
| rewards, terminations | the rollout is fixed-length and scored by `qd.common` |

`expand_bam_friction_fields` is deliberately **kept** — it is not DR but the
per-world model-field expansion BAM cannot run without. A test asserts every
event term in the velocity cfg is explicitly classified as one or the other, so
a term added upstream fails loudly instead of leaving DR quietly on. The robot
is also swapped for Phase 2's deterministic-actuator config, so both archives
are measured under identical physics.

### Per-step reward — an exact decomposition of the episodic fitness

```
r_t        = forward_velocity * dt                          while upright
r_terminal = -fall_penalty * (steps_left / total_steps)     on the fall, done=1
```

Summed, this is `displacement - fall_penalty * fraction_of_episode_fallen` —
*precisely* what the archive ranks on. That identity is the point: a critic
trained on anything else would push PG variation somewhere the archive does not
reward, and the symptom would be a PG insertion rate near zero. A test pins the
two expressions together.

Post-fall steps contribute no transitions, so the critic never learns from the
frozen tail.

### Variation

**GA half — iso+lineDD** (Vassiliades & Mouret):

```
child = a + iso_sigma * N(0, I) + line_sigma * N(0, 1) * (b - a)
```

Phase 2's plain per-dimension Gaussian is fine on 31 CPG parameters and
hopeless on 9038 MLP weights: a fixed sigma either barely moves the policy or
destroys it. The line term mutates along the direction between two elites — a
direction the archive has already shown to be productive — and is scale-free,
so the step size adapts to how far apart the parents are.

**PG half** copies random elites and takes `num_pg_training_steps` Adam steps on
them to maximise the TD3 critic's Q. All offspring are stepped at once, each on
its own independently sampled transition batch, mirroring QDax's vmapped
emitter. The greedy actor is evaluated and inserted alongside them each
iteration.

Hyperparameters follow QDax's `pga_me` / `td3` defaults (consulted directly):
`proportion_mutation_ga` 0.5, `num_critic_training_steps` 300,
`num_pg_training_steps` 100, replay buffer 1e6, transition batch 256, critic and
greedy-actor LR 3e-4, offspring policy LR 1e-3, `policy_noise` 0.2,
`noise_clip` 0.5, `policy_delay` 2, `soft_tau_update` 0.005, discount 0.99,
`iso_sigma` 0.005, `line_sigma` 0.05. Critic hidden layers are `(256, 256)`.

### Per-operator insertion rates

Every iteration logs `ga_insert_rate`, `pg_insert_rate` and whether the greedy
actor was inserted, and `summary.json` carries the run means. This is a
correctness signal, not a curiosity: **PG insertions near zero means the critic
or the reward wiring is broken**, and the fix is to debug it, not to ship the
run.

## Watching the gaits

```bash
# render every filled cell to mp4 (needs the MUJOCO_GL prefix — see below)
MUJOCO_GL=glfw uv run python -m qd.render_gaits \
    --archive logs/qd/map_elites/archive_final.npz --out logs/qd/gaits/cpg

# build the clickable page
uv run python -m qd.build_viewer \
    --manifests logs/qd/gaits/cpg/manifest.json logs/qd/gaits/pga/manifest.json \
    --labels "MAP-Elites" "PGA-ME" --out logs/qd/viewer/index.html
```

The page is a 20×20 heatmap you click: pick a cell and that elite's gait plays
beside its fitness, duty factors, distance, time upright and outcome. Arrow keys
step to the nearest *filled* cell, since the archive has holes. Two manifests
give an archive switcher.

**`MUJOCO_GL` must be set before Python starts** — MuJoCo resolves its GL
backend at import time, so setting it in code is too late. On this WSL2 box
`egl` fails (PyOpenGL finds no libEGL) and there is no OSMesa, but WSLg provides
a real display, so `glfw` renders offscreen fine. Use `osmesa` on a genuinely
headless machine.

The clips are **not** a CPU re-simulation. Each elite is rolled out in the same
batched MuJoCo-Warp harness that produced its archived fitness, with `qpos`
logged every control step; rendering replays those exact poses through CPU
MuJoCo. What you watch is the trajectory that was scored — only the rasteriser
is CPU-side.

Clips inline as data: URIs, because a published artifact cannot fetch external
media and the page has a 16 MB ceiling. `--budget-mb` fills it deliberately:
top elites by fitness first, then a spatial sweep across the descriptor space.
Every filled cell is still rendered to disk at full resolution, and cells
without an embedded clip stay clickable, showing their stats and the file path.

## Comparing the two archives

```bash
uv run python -m qd.compare_archives \
    --a logs/qd/map_elites/archive_final.npz --a-label "MAP-Elites (CPG)" \
    --b logs/qd/pga_me/archive_final.npz     --b-label "PGA-ME (MLP)" \
    --out logs/qd/comparison
```

Writes `comparison.png` — both heatmaps on a shared colour scale plus a
difference map showing which cells each pipeline reached — and
`comparison.json` with coverage, QD-score, best-cell fitness, and the mean
fitness delta over the cells both filled. Pass the same `--qd-score-offset` the
runs used, or the QD-scores are not comparable.

## Later upgrades (not built)

* **CMA-ME** — swap the `GaussianEmitter` for pyribs'
  `EvolutionStrategyEmitter`, which adapts a covariance matrix per emitter and
  usually beats isotropic mutation badly on a 31-D search space. Deliberately
  out of scope here: the point of Phase 2 is a vanilla MAP-Elites baseline for
  Phase 3 to be measured against.
* Richer descriptors (stride length, lateral drift, energy) or a 3-D archive.
* CPG coupling terms (a phase-locked oscillator network) instead of independent
  per-joint phases.
