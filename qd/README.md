# `qd/` — Quality-Diversity gait discovery for Microduck

Gradient-free (and, in Phase 3, gradient-*assisted*) search for a **diverse
archive** of forward-walking gaits, on the same MuJoCo-Warp simulation the PPO
recipes train in.

Where PPO returns one policy that maximizes a reward, MAP-Elites returns a
**grid of policies**, one per bin of a behaviour descriptor, each the best of
its kind. v1 and v2 used per-foot ground-contact duty factor, Cully et al.'s
hexapod descriptor; v2 then measured that a biped which can only stay upright
by alternating its feet at roughly even duty barely moves in it, and v3
replaces it with axes chosen by measurement.

> **Read [Walking v3](#walking-v3--a-descriptor-this-robot-can-move-in-and-a-gate-that-matches-the-bar)
> first if you want the current state.** v1 (Phases 2-3, below) is the baseline
> record: archives full of policies that *fall over*, because falling was priced
> rather than forbidden. v2 makes not-falling a feasibility constraint, fixes
> the physics the fall was measured under, and starts the search from the PPO
> walker — and ends with two measurements that v3 acts on. Everything below
> v1's Results section still describes machinery v2 and v3 use unchanged.

```
qd/
├── common.py          fitness + archive/plot/checkpoint helpers
├── descriptors.py     v3: the candidate behaviour axes, and which two an archive uses
├── select_descriptor.py  v3: pick the axes by measurement, not by taste
├── cpg_genome.py      Phase 2 genome: 31-parameter open-loop CPG
├── evaluate.py        batched mjlab rollout harness: genomes -> (fitness, descriptor)
├── run_map_elites.py  Phase 2 CLI: pyribs GridArchive + GaussianEmitter ask/tell loop
├── seed.py            v2: distil the PPO walker into the genome (DAgger)
├── play_elite.py      inspect / replay one elite from a saved archive
├── check_harness.py   physics sanity checks — run before any long run
├── check_floor.py     v2: is any part of the robot under the plane?
├── check_repeatability.py  v2: how much is one evaluation worth? (not much)
├── verify_archive.py  v2: re-roll every elite N times; keep the ones that are real
├── bench.py           throughput + evaluation-noise benchmark: pick a batch size
├── bench_collision.py v2: what full-body ground collision costs
├── survival_report.py   re-evaluate elites: survival + archive optimism
├── compare_archives.py  side-by-side archive comparison
├── render_gaits.py      rollout -> mp4 clips of each elite's gait
├── build_viewer.py      the clickable archive viewer page
└── pga/               Phase 3: PGA-MAP-Elites (see below)
    └── tune_pg.py     v2: how hard PG variation can push before it breaks a walker
```

Before the first long run on a new machine or after touching the MJCF:

```bash
uv run python -m qd.check_harness   # standing height, time-to-topple, descriptor spread
uv run python -m qd.bench           # ms/genome vs batch size, fitness noise
```

After a run, to see whether the elites actually *walk* rather than just score
well. In a v1 archive that is a question about the objective — the pro-rata
penalty means a high fitness can come either from covering ground fast before
falling or from staying up, and the archive does not record which. In a gated
v2 archive every member survived on insertion, so the question becomes whether
they *still* do, and how far they get, over replicas:

```bash
uv run python -m qd.survival_report --archive logs/qd/map_elites/archive_final.npz
uv run python -m qd.survival_report --archive logs/qd/pga_me_v2/archive_final.npz \
    --sample 64 --replicas 8
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

### Behaviour descriptor — two axes out of a catalogue (2-D, 20×20)

`qd/descriptors.py` holds nineteen candidate axes — contact statistics, posture,
motion, actuator effort — and one accumulator that folds a batched rollout into
all of them at once. `DescriptorCfg` names the two an archive is binned on plus
their grid ranges, and every checkpoint records that choice, so verification,
replay, rendering and the viewer all measure an archive on the axes it was
*built* on.

The default is **per-foot duty factor** — `(left, right)`, each the fraction of
control steps that foot was in ground contact, from the same
`feet_ground_contact` sensor the velocity task uses. Cully et al.'s hexapod
descriptor, generalized to a biped; it is what v1 and v2 ran on, and a v1/v2
checkpoint that names no descriptor comes back as this one rather than being
silently re-binned. Walking-v3 runs on
[measured axes instead](#stage-a--choosing-the-axes-by-measurement).

Steps after a fall are excluded from **every** axis — otherwise a robot lying on
its face with both soles touching would report duty `(1, 1)` and squat in that
corner of the archive, and the same trick is available to every axis added
since.

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

`tests/test_qd_cpg_genome.py`, `tests/test_qd_common.py` and
`tests/test_qd_descriptors.py` are CPU-only and run with the rest of the suite:

```bash
uv run --with pytest pytest tests/ -q
```

They lock the things that rot silently: the leg-joint set resolving **by name**
against the actual model in the documented servo order (`Entity.joint_names`
interleaves neck/head between the legs, so a hard-coded `0..9` slice grabs the
head), the two-layer bound enforcement, the fall latch, and the descriptor
staying inside `[0, 1]²` where the `GridArchive` can see it.

`test_qd_descriptors.py` locks the v3 axes the same way: the arithmetic of each
one against a hand-computable rollout (including the touchdown edge detection
step frequency and stride length are built on), that a missing input channel
reads NaN rather than zero, that nothing after the fall-detection frame reaches
any axis, and that a descriptor round-trips through a checkpoint's `meta` — the
last of which is what stops an old archive being verified on axes it was never
binned on.

## Results — v1

**This section is the baseline record, kept as it was written.** Its numbers
were measured under v1's physics (ground collision on the two foot soles only,
rollouts continuing past the fall). Re-measured under v2's physics they get
worse, and [Walking v2](#walking-v2--survival-gated-ppo-seeded-pga-me) says by
how much.

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

> **And that lone survivor does not survive honest physics.** Re-measured under
> v2's model — every shell colliding with the ground rather than the two soles
> — the top 64 elites of *both* v1 archives contain **zero** full-episode
> survivors: longest upright 4.70 s for the MLP archive and 1.82 s for the CPG,
> against archived claims of 7.00 s and 2.10 s. The v1 MLP archive is also
> +0.277 m optimistic on that re-measurement. So the count of survivors in v1
> is not one. It is none.

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
# smoke test first
uv run python -m qd.pga.run_pga_me --iterations 2 --batch-size 8 \
    --initial-solutions 16 --out-dir logs/qd/pga_smoke

# the run the Results table reports: ~207k evaluations, budget-matched to the
# MAP-Elites run above so the two archives can be compared at all
uv run python -m qd.pga.run_pga_me --iterations 200 --batch-size 1024 \
    --initial-solutions 2048 --td3.replay-buffer-size 2000000 \
    --out-dir logs/qd/pga_me_matched
```

A bigger batch costs little (the rollout loop dominates) but fills the replay
buffer far faster — 1024 worlds produce ~358k transitions per iteration, so the
default 1e6 buffer turns over every three iterations and the critic only ever
sees a narrow recency window. Hence `--td3.replay-buffer-size 2000000` here.

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

> **Superseded in v2** — see
> [the shaped per-step reward](#3-ppo-seeding-and-a-shaped-critic). Once
> survival is a constraint rather than a term in the objective, the critic's
> job is to teach balance, and this reward does not contain balance. What
> survives is the velocity term's scale.

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
`proportion_mutation_ga` 0.5, `num_critic_training_steps` 300, replay buffer
1e6, transition batch 256, critic LR 3e-4, `policy_noise` 0.2, `noise_clip`
0.5, `policy_delay` 2, `soft_tau_update` 0.005, discount 0.99, `iso_sigma`
0.005, `line_sigma` 0.05. Critic hidden layers are `(256, 256)`.

**Three of QDax's defaults do not survive v2** — `num_pg_training_steps` 100 ->
30, offspring policy LR 1e-3 -> 3e-5, greedy-actor LR 3e-4 -> 1e-6. They are
tuned for a search that starts from random policies, where a large policy move
costs nothing because there is nothing yet to break; here they destroy the
seeded walker outright.
[QDax's step sizes destroy a walker](#what-the-gate-exposed-qdaxs-step-sizes-destroy-a-walker)
has the measurements.

### Per-operator insertion rates

Every iteration logs `ga_insert_rate`, `pg_insert_rate` and whether the greedy
actor was inserted, and `summary.json` carries the run means. This is a
correctness signal, not a curiosity: **PG insertions near zero means the critic
or the reward wiring is broken**, and the fix is to debug it, not to ship the
run.


## Walking v2 — survival-gated, PPO-seeded PGA-ME

v1's verdict, in one line: *the ducks fall over*. 561 training-time survivors
across the whole PGA-ME run, exactly one of which survived a replay, and that
one covered 8.8 cm in seven seconds. The archive was full of policies optimised
to dive well.

Nothing was broken. The objective was doing exactly what it said:

```
fitness = displacement_at_fall - 0.25 * fraction_of_episode_fallen
```

A policy that covers 0.4 m and falls at 2 s of a 7 s episode scores +0.22. A
policy that stays upright and goes nowhere scores 0.00. **The dive wins**, and
MAP-Elites is very good at finding what wins. Charging by time-down made
falling *late* cheaper than falling early, which is the right shape, but it
never made falling *unacceptable*.

Three changes, and they are meant to be read together — the gate is only
meaningful if the fall is measured honestly, and the gate is only survivable if
the search starts somewhere feasible.

### 1. Honest physics

Two problems, one in the model and one in the loop.

**The model.** `robot_walk.xml` gives ground-collision geoms to the two foot
soles; the trunk, head, hips and thighs are `contype=0` or self-collision-only.
That is sound for PPO, which *terminates* the episode at the fall and never
simulates what happens next. A QD rollout kept going for the full 7 s, and a
toppled robot sank through the plane. v2 uses `robot_allcollisions.xml` — the
same robot, same HOME frame, same BAM actuators, the shells given ground
contacts (`qd.evaluate.HarnessCfg.full_collision`, default on for both
pipelines so CPG and MLP archives stay measured under identical physics).

**The loop.** The rollout now stops at the fall: a fallen world stops
contributing to fitness, descriptor, replay transitions and recorded `qpos`,
and the loop breaks entirely once nothing is upright. `RolloutMetrics` keeps
the *full* episode length as the denominator for the fallen fraction, so
stopping early does not make everything read as 100% alive.

How far under the floor did v1 go? `qd.check_floor` rolls out 64 falling random
MLPs and computes the exact lowest mesh vertex on the worst frame — every
geom's vertices transformed into the world, not a bounding-sphere estimate:

| | lowest point of the robot | what the frame looks like |
| --- | --- | --- |
| v1 (feet only, runs past the fall) | **-0.276 m** | two foot geoms floating on an empty plane; the duck is *gone* |
| v2 (full collision, stops at the fall) | **-0.013 m** | a duck lying face-down on the floor |

-0.013 m is a head shell at a contact, i.e. ordinary solver interpenetration on
a 25 cm robot. -0.276 m is the whole robot below the world.

```bash
MUJOCO_GL=glfw uv run python -m qd.check_floor --num-envs 64 \
    --dump-frames logs/qd/floor_frames --out logs/qd/floor_check.json
```

**One thing full collision does cost: constraint budget.** More geoms touching
the ground means more constraints, and mjlab's default allocation is sized for
the foot-only model. Under `full_collision`, `qd.check_harness` printed
`nefc overflow - please increase njmax to 91` on a robot lying on its side. An
nefc overflow silently *drops* constraints — it quietly makes the floor soft
again, which is exactly the bug `full_collision` exists to fix, reappearing on
the frames where honest physics matters most. `HarnessCfg.njmax` is pinned to
128. Only the low-level Phase-2 path was affected; the Phase-3
`ManagerBasedRlEnv` path sizes this itself and logged zero overflows across
both full runs.

The measured constants survive the model swap, which is the check AGENTS.md
insists on: settled trunk z **0.1151 m** under all-collisions against v1's
0.115 m, 40 mm of margin over the 0.075 m fall threshold, and a passive HOME
hold still toppling at **1.34 s**.

**What it costs in time: nothing.** `qd.bench_collision` at batch 1024 on an RTX 3060,
identical population in all four rows:

| collision | stops at fall | s/generation | ms/genome |
| --- | --- | --- | --- |
| feet only (v1) | no | 15.60 | 15.23 |
| feet only | yes | 14.58 | 14.24 |
| full body | no | 16.34 | 15.96 |
| **full body (v2)** | **yes** | **15.61** | **15.24** |

Full-body collision costs 1.05x per generation. Stopping at the fall gives it
back (a random MLP is down inside 1.5 s of a 7 s episode). **Net v1 -> v2:
1.00x.** The honest physics was free; it just had to be asked for.

### 2. The survival gate

Only solutions upright for the **whole** episode are inserted. Not-falling is a
feasibility constraint, not a term in the objective.

The fitness formula is **unchanged** — deliberately. For a full-episode
survivor `fallen_fraction` is 0, so the pro-rata penalty is arithmetically inert
inside a gated archive and the archived objective *is* forward displacement.
Leaving v1's expression alone is what keeps a v1 archive and a v2 archive
comparable at all: same objective, same descriptor, same grid, different
admission rule.

Two rates are logged per operator per iteration:

* **feasibility rate** — what fraction of this operator's offspring stayed up.
  This is the curve that says whether the search is living inside the
  constraint or bouncing off it.
* **insertion rate** — measured over the *whole* block, v1's denominator, so
  the per-operator numbers remain comparable across the two runs.

If the feasible set were empty at initialisation the run would need an
episode-length ramp. It never was: even 1024 randomly initialised MLPs produce
~65 full-episode survivors (a random tanh net often holds a static crouch), and
with the seed below the archive opens with dozens.

### 3. PPO seeding, and a shaped critic

**The seed.** `qd/seed.py` takes the Phase-1 PPO velocity walker and distils it
into the genome architecture. A weight copy is not available: the rsl_rl actor
is `61 -> 512 -> 256 -> 128 -> 14` with ELU and a **linear** output head behind
an `EmpiricalNormalization` layer, while the genome is `61 -> 64 -> 64 -> 14`
with tanh throughout. Retraining PPO at 64x64 would fix the widths and the
normalizer folds exactly into the first layer, but the output nonlinearity would
still not match — and tanh on the output is not cosmetic, it is what bounds the
action space TD3's target-policy smoothing clips its noise against.

So: behaviour cloning, with **DAgger**. Observation slot `[34:48]` is the
previous action, so a student trained only on the teacher's trajectories sees
the teacher's action history at training time and its own at deployment. That
compounds, and here it compounds into a fall — visible in the log below as
rounds 1 and 2, where the student drives and *nothing* survives, before round 3
recovers:

```
round 0 (teacher driving)  89600 states  bc loss 0.00245  survivors 256/256  max displ +1.689 m
round 1 (student driving) 106891 states  bc loss 0.00609  survivors   0/256  max displ +0.663 m
round 2 (student driving) 124140 states  bc loss 0.00849  survivors   0/256  max displ +0.691 m
round 3 (student driving) 213409 states  bc loss 0.00458  survivors 254/256  max displ +1.414 m
```

**The forward command lives in the weights, not the input.** The QD env pins all
13 command slots to zero — v1's convention, and the deployment idle state — and
under a zero twist command the PPO walker correctly stands still. So the teacher
is queried at the observation with the twist-vx slot overwritten by
`teacher_vx = 0.3` m/s, while the student is trained on the **unmodified**
zero-command observation. The student that comes out walks forward when its
input says "stand". The 61-D obs contract is untouched.

The resulting seed, under v2 physics:

| | |
| --- | --- |
| survives 7 s | yes |
| displacement | **+1.27 m** |
| duty factor (L, R) | (0.55, 0.51) — a real alternating gait |
| across 256 identical replicas | **99.6% survive**, +0.94 m mean, -0.40 .. +1.62 m |

That replica spread is not noise in the measurement; it *is* this simulator's
closed-loop non-determinism, measured on the policy that has to live with it.

```bash
uv run python -m qd.seed \
    --checkpoint logs/rsl_rl/qd_phase1_baseline/<run>/model_399.pt \
    --out logs/qd/seeds/ppo_seed.npz
```

**The shaped per-step reward.** v1's critic was trained on bare forward velocity
with a terminal fall penalty, chosen so the undiscounted return equalled the
episodic fitness *exactly*. That identity was the right call while fitness was
the only thing standing between the search and a face-plant. With survival now a
constraint, the critic's job changes: what PG variation needs from it is a dense
signal for **balance**, which bare forward velocity does not contain — v1's
critic could not tell a controlled step from the first frame of a dive. So
(`qd.pga.evaluate.ShapedRewardCfg`):

```
r_t        = 1.0 * v_x * dt                                    # metres of progress
           + (0.10 + 0.30 * clip(-projected_gravity_z, 0, 1)) * dt   # alive + upright
r_terminal = -1.0 * (steps_left / total_steps)                 # on the fall
```

Weights are in metres, so the velocity term is still literally displacement and
the critic keeps v1's scale. The 1:1 ratio between travel and posture is
borrowed from the velocity task's own stack (`track_linear_velocity` weight 2.0
against `upright` weight 2.0): at the 0.3 m/s the twist command tops out around,
`upright_weight` pays the same per second as travel does. The upright term reads
the same `projected_gravity_z` the fall check reads, so the critic is taught
balance in the coordinate the gate actually measures. The fall penalty is 4x
v1's, because it no longer has to stay small to avoid distorting an archive
objective — inside a gated archive the terminal penalty is unreachable.

**The exact-decomposition identity is deliberately abandoned**, and the test
that pinned it now pins the narrower claim that survives: at `vel_weight` 1.0
the velocity term alone still sums to displacement.

### What the gate exposed: QDax's step sizes destroy a walker

The first gated run had **PG insertion rate 0.0%, every iteration**, against GA
at 3-5%. Per-operator feasibility said why: GA offspring survived ~40% of the
time, PG offspring **0%**. PG variation was not failing to help — it was
destroying the policies it was handed.

The arithmetic is not subtle. Adam's per-parameter step is bounded by the
learning rate, so `steps * lr` is a budget on how far the genome travels. QDax's
default 100 offspring steps at 1e-3 moves every one of 9038 weights by up to
0.1, against an initial weight scale of `1/sqrt(61) = 0.13`. That is a ~70%
perturbation of the entire policy. In v1 this was invisible: everything fell
anyway, so a destroyed offspring and an intact one scored about the same and PG
still inserted 1.8% into fall-dominated cells.

`qd/pga/tune_pg.py` measures the cliff instead of guessing at it — one real
generation into the buffer, the critic trained exactly as an iteration would,
then the same 40 parents mutated at each setting and all of them evaluated in
one batched rollout. Parents were feasible 40% of the time:

| steps | lr | mean per-weight drift | offspring feasible |
| --- | --- | --- | --- |
| 5 | 3e-5 | 1.1e-4 | 48% |
| 5 | 3e-4 | 1.1e-3 | 45% |
| **30** | **3e-5** | **6.3e-4** | **55%** |
| 10 | 3e-4 | 2.1e-3 | 40% |
| 30 | 1e-4 | 2.1e-3 | 52% |
| 30 | 3e-4 | 5.8e-3 | **0%** |
| 100 | 1e-3 *(QDax default)* | 2.9e-2 | **0%** |

Feasibility holds to about 2e-3 of mean per-weight drift and collapses by 6e-3.
The defaults are now **30 steps at 3e-5** — drift 6.3e-4, comfortably inside the
safe region and still moving the genome ~5x further than the GA's iso term.

```bash
uv run python -m qd.pga.tune_pg --seed-genome logs/qd/seeds/ppo_seed.npz
```

### What the gate exposed, part two: one seed is not enough

The first full gated run stalled. Elites went 7 -> 15 over nine iterations and
then 15 -> 17 over the next ten, with insertion rates down at 0.0-0.2%. The
tempting reading is "the search is converging"; the per-operator logs say
otherwise. **Feasibility stayed healthy at 30-40%** — roughly 170 of 512 GA
offspring per iteration were surviving the full episode, and essentially every
one of them landed in a cell that was already filled.

So the offspring were fine. They just were not *different*.

**Near a walker, the map from MLP weights to duty factor is flat.** GA variation
is already perturbing at the largest per-weight step that leaves the policy
upright — `iso_sigma` 0.005 gives a mean per-weight drift of 4e-3, and
`tune_pg` measured feasibility collapsing by 6e-3 — and at that step the duty
factors barely move. Push harder and the policy dies. Push softer and the
behaviour does not change. There is no setting of an isotropic weight-space
mutation that buys behavioural diversity here, and that is a property of the
genome-to-descriptor map, not of the hyperparameters.

What *does* produce structurally different, feasible gaits is the thing that
produced the seed in the first place. The PPO teacher walks differently **on
command** — a 0.1 m/s shuffle, a 0.4 m/s stride, a forward-plus-turn — and every
one of those is feasible by construction, which is what weight-space mutation
cannot manufacture. So v2 distils several of them:

```bash
uv run python -m qd.seed \
    --checkpoint logs/rsl_rl/qd_phase1_baseline/<run>/model_399.pt \
    --out logs/qd/seeds/ppo_seeds.npz \
    --seeding.teacher-commands "0.10 0.0 0.0" "0.20 0.0 0.0" "0.30 0.0 0.0" \
                               "0.40 0.0 0.0" "0.25 0.15 0.0" "0.25 0.0 0.5"
```

Each seed keeps its command baked into its weights, so all of them are still
policies the archive evaluates under a **zero** observation and the 61-D
contract is untouched. `run_pga_me` gives every seed its own jittered
neighbourhood (`jitter_count` split evenly), which costs no extra evaluations —
the seed block is one fixed-size rollout either way, it simply spends fewer of
its worlds on random MLPs.

**Did it work? Partly, and not for the reason predicted.** The archive improved
a lot — 38 elites by iteration 20 against the single-seed run's 17 by iteration
19, with feasibility 55-87% against 30-40%. But the six commands did **not**
spread the duty factors: they land within 0.03 of each other, which is under two
cells. What helped was six different *weight-space* neighbourhoods and a ladder
of jitter radii around each, not the commands separating in descriptor space.
The next section is the measurement, and it is the more important result.

The general point survives intact, and generalises past this robot: once
not-falling is a constraint, the feasible set is a thin manifold, isotropic
mutation cannot travel along it, and the archive's spread is set by how diverse
its *feasible seeding* was. A richer answer would be a variation operator that
moves along the manifold rather than across it — which is what PG variation is
supposed to be, and would be if the critic knew about the descriptor. That is
the obvious next thing to build, and it is not built here.

### Negative result: per-foot duty factor barely varies among upright gaits

This one is worth reading before designing another run on this archive, because
it says the behaviour descriptor is not measuring much.

Three independent measurements, all pointing the same way.

**Commanded gaits that differ a lot in speed do not differ in duty factor.**
The six distilled seeds come from twist commands spanning the velocity task's
whole forward range, and the teacher's own displacement over 7 s spans 6x across
them. Their duty factors span 0.03:

| teacher command (vx, vy, wz) | teacher travels | seed duty (L, R) |
| --- | --- | --- |
| (0.10, 0, 0) | +0.41 m | (0.48, 0.57) |
| (0.20, 0, 0) | +0.83 m | (0.46, 0.57) |
| (0.30, 0, 0) | +1.64 m | (0.48, 0.55) |
| (0.40, 0, 0) | +2.42 m | (0.49, 0.54) |
| (0.25, 0.15, 0) — forward + strafe | +1.30 m | (0.47, 0.56) |
| (0.25, 0, 0.5) — forward + turn | +1.24 m | (0.48, 0.57) |

A shuffle and a stride, a straight walk and a turn, all inside two cells of a
20x20 grid. Whatever distinguishes those gaits, this descriptor cannot see it.

**Mutation cannot move it either.** GA variation runs at the largest per-weight
step that leaves the policy upright (`iso_sigma` 0.005, mean drift 4e-3, against
the ~6e-3 at which `tune_pg` measured feasibility collapsing), and at that step
offspring stay feasible ~40-60% of the time and land almost entirely in cells
that are already filled — that is the single-seed stall above, and it is a
property of the genome-to-descriptor map rather than of the hyperparameters.

**Chance cannot move it.** 256 byte-identical copies of one genome, whose
*displacement* spreads with a standard deviation of 0.605 m, have a duty-factor
standard deviation of **0.014** — about a quarter of a cell. The objective is
chaotic and the descriptor is stable.

So: the reachable-while-upright region of this descriptor is genuinely small,
and v2's low raw coverage is mostly that, not a failure of search. v1's 73-85%
raw coverage was overwhelmingly *fallen* policies, whose duty factors are set by
how they happened to collapse — a robot lying on both soles reads (1, 1), a
face-plant reads whatever its last upright step did. Coverage over a
fall-populated archive is a measure of the ways this robot can fall.

**Implication for the next run: change the descriptor, not the search.** Stride
length, lateral drift, cost of transport, turning rate or trunk-pitch amplitude
would all vary across the six gaits above, where duty factor does not. Per-foot
duty factor is Cully et al.'s hexapod descriptor and it earns its reputation on
a robot with six legs and many ways to distribute contact; on a biped that can
only stay upright by alternating its two feet at roughly even duty, it is close
to a constant. That is not an argument against QD here — it is an argument that
the axes were the wrong ones, and picking better ones is cheap.

### The greedy actor, and what TD3 is actually for here

The greedy actor needed the same treatment for a subtler and more important
reason. It takes ~150 updates *per iteration* (`num_critic_training_steps /
policy_delay`), so its budget is 150x an offspring's. Measured survival
fraction per iteration, over 8 iterations, starting from the seeded walker:

```
3e-4 (QDax)   1.00 0.06 ...            destroyed inside one iteration
1e-5          1.00 1.00 0.20 0.16 ...  gone by iteration 3
1e-6          1.00 1.00 1.00 1.00 ...  upright throughout
```

This matters far beyond the greedy actor's own insertions (it has essentially
none, in v1 or v2). TD3 bootstraps its target off `greedy_target`:

```
target = r + gamma * min(Q1, Q2)(s', greedy_target(s'))
```

The critic therefore learns the value of *the greedy actor's* behaviour. Boot-
strapping off a policy that face-plants at 1.3 s teaches a critic whose Q surface
is about falling — and PG variation then gradient-ascends every elite toward
that. Moving the greedy actor from 1e-5 to 1e-6 took PG offspring feasibility
from ~45% to ~58%.

**Be clear about what this means.** At 1e-6 the greedy actor is effectively
*frozen near the seed*: over 200 iterations its total drift is on the order of
the safe single-variation budget, so it improves barely if at all, and in the
validation runs its own displacement decayed toward zero (the cold critic likes
the alive and upright terms more than it likes travelling). TD3 in walking-v2 is
therefore **not** an actor-learning algorithm. It is a critic-fitting algorithm
whose actor's job is to keep the bootstrap anchored on states a survivor can
actually reach. The gradient half of PGA-ME earns its place through PG variation
of archive elites, not through the greedy actor. A version that wanted a genuinely
learning greedy actor would need the critic warmed on survivor data before the
actor is allowed to move — a curriculum on the learning rate — which is out of
scope here and worth trying next.

### The finding that reframes every number here: a walker's fitness is chaotic

v1's Reproducibility section measured this simulator's evaluation noise on an
open-loop CPG and got ~4 mm, then noted that a closed-loop MLP amplifies it
~60x. Walking-v2's genomes are *walkers*, and the amplification is not 60x.

`qd.check_repeatability` runs **N byte-identical copies of one genome in one
batched rollout**, with every DR knob off, the spawn pinned, the actuator
deterministic, and every world reset to the same state. The only thing that
differs between worlds is MuJoCo-Warp's contact/constraint solve order. On the
0.4 m/s seed, 256 copies:

| | |
| --- | --- |
| survival | 98.8% |
| displacement, median | **+1.565 m** |
| displacement, standard deviation | **0.605 m** |
| displacement, 5th-95th percentile | -0.051 .. +1.937 m |
| displacement, full range | -0.796 .. +2.168 m |
| duty factor, standard deviation | (0.014, 0.014) |
| trunk-x spread between worlds | 1 mm at 0.04 s, 10 mm at 0.12 s, 100 mm at **0.52 s** |

That divergence profile is exponential, which is what a marginally stable
bipedal walker integrating 350 control steps is: a chaotic system. The same
policy walks 2 m or falls backwards depending on the last bit of a contact
force.

Three consequences, and they run through everything above:

1. **A single evaluation is nearly uninformative about displacement.** Report
   medians over replicas. `qd.survival_report --replicas N` does this; the
   displacement columns at `--replicas 1` are a coin flip and should be read
   as such.

2. **MAP-Elites is not merely optimistic here, it is systematically
   luck-ranked.** It inserts on one sample and keeps the maximum per cell, so a
   cell's archived value estimates its elite's *best luck* — measured at
   **+0.60 m** above the median on this genome. This also explains the
   insertion-rate collapse the run shows from about iteration 10: once every
   cell holds a lucky draw, a genuinely better offspring loses to an incumbent's
   good day, and the logged insertion rate stops being a measure of whether the
   operators work. **Re-evaluation on insertion, or a running mean per cell, is
   the fix, and it is not implemented here** — v1's README already listed it as
   a known gap, and this measurement is what makes it urgent rather than
   theoretical. Doing it properly costs a second evaluation per candidate, i.e.
   half the archive at the same budget; that trade is worth measuring and was
   out of scope for this pass.

3. **Survival is much steadier than distance.** 98.8% versus a 0.6 m spread.
   That asymmetry is why the survival gate works as a constraint at all: it is a
   near-deterministic predicate on a chaotic system, so gating on it does not
   inherit the chaos. It is also why the acceptance question "do the archived
   elites really survive" gets a clean answer while "how far does elite k go"
   only gets one over replicas.

The duty-factor standard deviation of 0.014 is worth reading twice, next to the
0.6 m in displacement. The **descriptor is stable while the objective is not** —
about a quarter of a 0.05-wide cell, so an elite occasionally hops one cell.
Combined with the flatness finding above, the picture is that this robot's
behaviour space, as this descriptor sees it, is both narrow and stubborn: hard
to move deliberately, and barely moved by chance.

### The gate is necessary and not sufficient: re-evaluation on insertion

The 200-iteration survival-gated run met the physics, wiring and locomotion
criteria and failed the two honesty ones. Of 64 elites sampled uniformly from
the archive and replayed 8 times each, **14 survived**. Verifying all 81 elites
the same way left **7** that survive at least 7 of 8 replicas, holding **6**
cells above 0.25 m.

An archive that is 100% survivors by construction was 91% coin-flips in fact.

**The proof of what went wrong is the seeds.** Six genomes went into the archive
at iteration 0 with 99-100% measured replica survival, the most robust policies
in the whole run by a wide margin. At iteration 200, **not one of them is in the
archive**:

```
seed 0 (vx 0.10): evicted    seed 3 (vx 0.40): evicted
seed 1 (vx 0.20): evicted    seed 4 (fwd+strafe): evicted
seed 2 (vx 0.30): evicted    seed 5 (fwd+turn): evicted
```

Each was replaced, in its own cell, by a descendant that got a luckier single
rollout. With displacement carrying a 0.605 m standard deviation, a marginal
policy's good day beats a robust policy's typical one, and MAP-Elites keeps the
good day. **The survival gate cannot prevent this** — a lucky policy really did
survive the one rollout it was gated on. Gating on a single sample of a chaotic
predicate filters nothing in the long run; it just changes which lottery is
being run.

**The rule.** `--insertion-replicas N` rolls every candidate out N times and
judges it on all of them:

* **survival is unanimous** — a candidate counts as a survivor only if it stayed
  upright in *every* replica. A policy that falls one time in N is a policy that
  falls;
* **fitness is the median** displacement — not the max, which is precisely the
  luck-ranking being removed, and not the mean, which one catastrophic replica
  drags around;
* every replica's transitions still go to the replay buffer, so the critic sees
  N times the data rather than less.

`N=1` is passed through untouched, so the two runs differ in exactly one rule
and stay comparable; a test pins that.

**The cost is half the search.** N replicas is exactly N times the evaluations,
so a budget-matched run divides its iterations by N: **99 iterations instead of
200**, 1024 offspring each, 207,048 evaluations either way. That is not a
footnote — it is the trade. Half as many genomes are tried, and each result is
worth believing. Whether the archive that comes out is better is an empirical
question, answered in the comparison below rather than assumed.

**Its limitation, stated plainly** (and understated, as
[v3 later measured](#the-replicas-were-not-independent-and-that-is-most-of-why-v2-failed):
the two replicas were not two independent draws, because re-rolling the same
batch puts every replica in the same world, and a world index carries a
persistent bias worth ~6x the between-rollout spread). Two replicas is a much
weaker filter than the eight the verification uses. A policy that survives 2 of 2 has a survival
rate whose 95% interval still stretches down to about 0.22; only about 3 in 4
genuinely-90%-robust policies pass it on any given attempt. So the re-evaluated
run is *not* expected to produce an archive that is 100% verified — it is
expected to shift the distribution, and the honest test is the same 8-replica
verification applied to both runs. Pushing N higher trades search harder still:
at N=8 the run would get 25 iterations.


### Results — v1 vs v2, all on medians

> **Read this first, or the table below will mislead you.** On this simulator a
> walking policy's displacement has a **standard deviation of 0.605 m** across
> byte-identical worlds, and MAP-Elites inserts on one sample and keeps the
> maximum — so an *archived* value estimates an elite's best luck, by a measured
> **+0.64 to +0.94 m**. Every "verified" number below is instead the **median of
> 8 fresh rollouts** per elite, and an elite counts as surviving only if it
> stayed upright in at least 7 of those 8. Archived values appear only in the
> row labelled as such.

Three runs, all at ~207,000 evaluations on one RTX 3060:

| | v1 MAP-Elites (CPG) | v1 PGA-ME (MLP) | v2 single-sample | v2 re-evaluated |
| --- | --- | --- | --- | --- |
| **what it is** | open-loop CPG, fall penalty | closed-loop MLP, fall penalty | gated + seeded, 200 it | gated + seeded, **99 it x 2 replicas** |
| **VERIFIED — 8 fresh rollouts per elite** | | | | |
| elites surviving >= 7 of 8 | **0** | **0** | 4 | **5** |
| best verified median displacement | — | — | +1.620 m | **+2.117 m** |
| cells with a verified elite >= 0.25 m | 0 | 0 | 4 | **5** |
| **STRUCTURE — raw, counts nothing about robustness** | | | | |
| elites | 340 | 292 | 81 | 62 |
| raw coverage | 85.0% | 73.0% | 20.2% | 15.5% |
| *archived best (optimistic)* | *+0.153 m* | *+0.504 m* | *+2.970 m* | *+3.554 m* |
| *archive optimism* | *-0.217 m* | *-0.064 m* | *+0.643 m* | *+0.897 m* |

**v1's archives contain no robust walkers at all.** Not one of 632 elites across
both v1 archives survives 7 of 8 fresh rollouts under honest physics. v1's raw
coverage of 85% and 73% is a map of the ways this robot can fall over.

**v2 produces walkers that travel two metres and keep their feet.** Against
j002's headline — "the single policy that stays upright covers 8.8 cm" — the
best verified v2 elite covers **+2.117 m**, about 24x further, and does it in
five of eight independent rollouts.

#### Verification strictness, reported whole

A single pass/fail threshold hides the distribution, and choosing one after
seeing the numbers is how a report flatters itself. So here is the whole sweep,
the same for both v2 archives:

| survives at least | v2 single-sample | | | v2 re-evaluated | | |
| --- | --- | --- | --- | --- | --- | --- |
| | elites | cells >=0.25 m | cells >=0.50 m | elites | cells >=0.25 m | cells >=0.50 m |
| 4 of 8 | 23 | 13 | 11 | 23 | **20** | **19** |
| 5 of 8 | 16 | 11 | 9 | 17 | **17** | **16** |
| 6 of 8 | 7 | 7 | 5 | 10 | **10** | **10** |
| 7 of 8 | 4 | 4 | 2 | 5 | **5** | **5** |
| 8 of 8 | 3 | 3 | 1 | 3 | 3 | **3** |

Two things to read off it. First, **the re-evaluated run dominates at every
threshold** on cells above a distance, and its best verified elite goes +2.117 m
against +1.620 m. Second, and more telling: in the re-evaluated archive the
"elites" and "cells >= 0.50 m" columns are *the same number* at every strictness
level. It has no weak survivors. The single-sample archive's do not match at any
level — a third to a half of what survives there goes nowhere.

**The verification is itself a sample.** Running the same 8-replica check twice
on the same archives gave 7 and 8 verified elites the first time and 4 and 5 the
second. That is not a bug; it is the same chaos, one level up. Treat every count
in these tables as carrying an uncertainty of a few elites, and treat the
*ordering* between the two runs — which held in both passes — as the result.

#### The cost of the fix, stated as a trade

The re-evaluated run bought its reliability with **half the search**: 99
iterations instead of 200, at 1024 offspring each, for the same 207,048
evaluations. It ended with 62 elites against 81 and 15.5% raw coverage against
20.2%. Fewer genomes tried, each worth believing. On the verified columns —
the only ones that mean anything here — it wins outright.

#### Where the honest numbers leave the acceptance criteria

Two of this job's criteria are met and two are not, and the failures share one
cause:

* **PASS** — best replay-verified elite >= 0.5 m in 7 s, and >= 5x j002's best
  survivor (0.088 m). Measured: **+2.117 m**, 24x.
* **PASS** — physics honest, no clip below the floor, seeds replay-verified at
  iteration 0, both GA and PG contributing (re-evaluated run: GA 0.54% and PG
  0.32% mean insertion, GA 34.0% and PG 31.5% mean feasibility — the same ~2:1
  GA:PG ratio v1 reported at 3.7%/1.8%).
* **FAIL** — ">= 90% of 64 sampled elites survive on independent replay."
  Measured: 8% at 7-of-8, 37% at 4-of-8. The archive is 100% survivors *by
  construction* and a minority of them in fact.
* **FAIL** — ">= 20 distinct cells hold elites >= 0.25 m" on verified medians.
  Measured: 5 at 7-of-8, and exactly 20 at 4-of-8 — so it passes only at the
  loosest reading of "survives", which is not the reading worth having.

Both failures are the same problem at different depths: **two insertion replicas
is a much weaker filter than eight-replica verification.** A policy that survives
2 of 2 still has a 95% interval on its true survival rate reaching down to ~0.22.
The fix worked in the direction and roughly the magnitude predicted before it
was run — measured on the same verification pass, mean survival rate per elite
**0.296 -> 0.395** and elites that never survive a single replay **17 -> 5** —
and it did not reach 90%, which at matched budget would need something like
N=8 and 25 iterations. That is a different experiment, and Alex's call rather
than this job's.

### Budget and reproduction

Budget-matched to v1's 207,048 evaluations:

```bash
# 1. distil the seed (once, ~4 min)
uv run python -m qd.seed \
    --checkpoint logs/rsl_rl/qd_phase1_baseline/<run>/model_399.pt \
    --out logs/qd/seeds/ppo_seed.npz

# 2a. the single-sample run: 1025 (seed block) + 1024 (random init)
#     + 200*1025 = 207,049 evaluations
uv run python -m qd.pga.run_pga_me --iterations 200 --batch-size 1024 \
    --initial-solutions 1024 --seed-genome logs/qd/seeds/ppo_seeds.npz \
    --seeding.jitter-count 240 --td3.replay-buffer-size 2000000 \
    --out-dir logs/qd/pga_me_v2

# 2b. the same budget spent on re-evaluation instead of iterations:
#     99 iterations x 2 replicas = 207,048 evaluations
uv run python -m qd.pga.run_pga_me --iterations 99 --batch-size 1024 \
    --initial-solutions 1024 --seed-genome logs/qd/seeds/ppo_seeds.npz \
    --seeding.jitter-count 240 --insertion-replicas 2 \
    --td3.replay-buffer-size 2000000 --out-dir logs/qd/pga_me_v2_reeval

# 3. the honest numbers: every elite re-rolled 8 times, survivors kept
uv run python -m qd.verify_archive --archive logs/qd/pga_me_v2/archive_final.npz
uv run python -m qd.verify_archive --archive logs/qd/pga_me_v2_reeval/archive_final.npz
```

The seed block is one rollout holding the seed, 100 jittered variants
(`jitter_sigma` 0.02, 4x the GA's iso term so the variants do not all land in
the seed's own cell) and 924 random MLPs. Together with `--initial-solutions
1024` that is 1948 random initialisations against v1's 2048 — the random-init
budget is essentially preserved, with 101 seeded genomes added.

Random MLPs are still evaluated with a seed present. Under the gate almost none
of them are inserted, but they are precisely the falling-over experience the
critic needs in its buffer to learn what not to do.


## Walking v3 — a descriptor this robot can move in, and a gate that matches the bar

v2 ended with two measurements and one honest failure. The measurements:
per-foot duty factor is nearly constant across everything this robot can do
while upright, and a 2-replica insertion gate is a much weaker filter than an
8-replica verification. The failure: an archive that was 100% survivors by
construction and **8%** by verification, holding **5** robust elites in **5**
cells.

v3 changes exactly those two things, and they turn out to be one idea.

### Stage A — choosing the axes by measurement

The question v2 never asked costs about ten minutes:

> does this axis separate gaits we already know are different, by more than one
> genome's own replica noise — and can the search *move* along it?

`qd/select_descriptor.py` asks it of all nineteen candidate axes in
`qd/descriptors.py`, on a fixed **measurement set** of gaits that are known to
be distinct and known to be robust: the six PPO teacher gaits distilled by
`qd.seed` (twist commands spanning a 6x range in how far the teacher travels)
and the five j003 elites that survived 7-of-8 verification. Sixty-four
byte-identical replicas each, 704 rollouts, and a fallen replica is dropped —
a truncated episode is not a measurement of a gait.

Three numbers per axis:

* **between-gait spread** — the range of the eleven gaits' *median* values. How
  much of the axis real, feasible, structurally different gaits actually use.
* **within-genome replica noise** — the mean across gaits of one genome's
  standard deviation over its replicas. What MuJoCo-Warp's contact-solve order
  moves the axis by, with the genome held byte-identical.
* **mutation reach** — mutate the whole measurement set with the run's own GA
  operator (iso+lineDD at the tuned sigmas), keep the 909 of 1024 offspring
  that stay upright, and measure how far the axis travels among them, in units
  of its own replica noise. This is the half v2 got wrong twice over: duty
  factor neither separated gaits *nor* moved under mutation, and the second
  failure is why the archive stalled at 15 elites.

The full table is in `qd/measurements/descriptor_selection.md`; the axes that
cleared both thresholds (replica sd under 10% of the axis's measured range,
spread-to-noise at least 3):

| axis | between-gait spread | replica sd | sd / range | spread / sd | \|corr\| with displacement | mutation reach (replica sds) |
| --- | --- | --- | --- | --- | --- | --- |
| mean \|joint velocity\| [rad/s] | 0.889 | 0.0748 | 5.6% | 11.9 | 0.93 | 13.3 |
| cost of transport | 4.97 | 0.435 | 3.8% | 11.4 | 0.72 | 13.3 |
| actuator power [W] | 2.40 | 0.225 | 5.4% | 10.7 | 0.90 | 12.4 |
| mean \|yaw rate\| [rad/s] | 0.791 | 0.0833 | 6.9% | 9.5 | 0.87 | 10.6 |
| stride length [m/step] | 0.0563 | 0.0062 | 8.0% | 9.0 | **1.00** | 9.5 |
| **mean trunk height [m]** | **0.00278** | **0.00032** | **7.4%** | **8.7** | **0.61** | **11.7** |
| right-foot duty factor | 0.117 | 0.0145 | 7.7% | 8.1 | 0.82 | 8.1 |
| trunk-height oscillation [m] | 0.00148 | 0.00028 | 8.9% | 5.2 | 0.83 | 7.4 |
| *left-foot duty factor* | *0.070* | *0.0156* | *11.4%* | *4.5* | *0.66* | *5.5* |
| *step frequency [Hz]* | *0.571* | *0.270* | *7.9%* | *2.1* | *0.58* | *4.9* |
| *mean forward lean* | *0.0302* | *0.0377* | *14.4%* | *0.8* | *0.27* | *3.5* |

**Chosen: mean trunk height x mean |joint velocity|** — *how crouched it walks*
by *how fast its limbs move* — on a 20x20 grid over

| axis | range | why this range |
| --- | --- | --- |
| mean trunk height | **[0.11667, 0.12184] m** | measurement-set extremes, padded 10% |
| mean \|joint velocity\| | **[0.89228, 2.50831] rad/s** | same |

Against v2's descriptor, on the same eleven gaits and the same 20x20 grid:

| | duty factor (v2) | trunk height x limb speed (v3) |
| --- | --- | --- |
| the 11 known gaits occupy | **4 cells** | **10 cells** |
| the 6 teacher gaits occupy | **1 cell** | **6 cells** |
| cells reachable by feasible GA mutants | — | **201 of 400** |
| marginal bins filled by feasible mutants | — | 20/20 and 19/20 |

#### The axis that topped the ranking and was rejected anyway

`stride_length x torso_height_mean` scored best on every raw column — 230 cells
reached, 11 of 11 gaits separated — and it is not the descriptor. Stride length
here is planar displacement divided by the touchdown count, and the touchdown
count barely varies across the whole measurement set (step frequency spans
4.57-5.14 Hz). So stride length is the objective rescaled: its correlation with
displacement is **1.00**. One of the archive's two axes would *be* fitness,
MAP-Elites would fill it trivially, and "coverage" would degenerate into a
histogram of distances. An axis has to be a property of *how* the robot walks,
not a restatement of how far.

#### What the chosen pair costs, stated plainly

Limb speed correlates with displacement at **0.93** and trunk height at 0.61,
so the high-limb-speed half of the archive partially mirrors the fitness
ladder: fast cells hold fast walkers largely because moving faster requires
moving the limbs faster. That is a real cost of this pair and it is visible in
the table above. It was accepted because every more-independent alternative
failed somewhere worse — cost of transport (|corr| 0.72, and the most
independent of trunk height at 0.18) is ~1/distance, heavy-tailed, and on a
linear 20-bin grid the feasible mutants pile 53% of themselves into one bin;
step frequency and forward lean fail the noise threshold outright. Trunk height,
the axis that carries the *behavioural* half of the pair, is the least
fitness-correlated axis that passed.

### Stage B — the gate matches the bar

`--insertion-replicas 8`, unanimous survival, median fitness. Same rule v2
introduced at N=2, run at the N the verification actually uses. The reasoning
v2 wrote down and did not act on: a candidate that survives 2 of 2 still has a
95% interval on its true survival rate reaching down to ~0.22, so the archive
fills with policies that pass the gate and fail the bar.

**The two halves of v3 are the same idea.** The chosen axes are fine-grained —
0.26 mm bins on trunk height against a 0.52 mm single-rollout standard
deviation — so a *single* rollout's descriptor jitters about two bins, and the
median over the eight insertion replicas is what makes a cell mean anything.
The gate stabilises the geography as well as the fitness, and this descriptor
would have been a bad choice under v2's rule.

#### The replicas were not independent, and that is most of why v2 failed

Setting `--insertion-replicas 8` is not the same as having eight independent
draws, and this took a measurement to notice. Insertion replicates by rolling
the identical block out N times; `qd.verify_archive` replicates by putting a
genome in N different **worlds** of one batch. Those are the same thing only if
a world index carries no bias.

It does. Same genome, same harness, deliberately measured both ways:

| | displacement sd | descriptor sd (height, limb speed) |
| --- | --- | --- |
| 32 different **worlds**, one rollout | **0.469 m** | — |
| same **slot**, 6 sequential rollouts | **0.071 m** | 0.00012 m, 0.0115 rad/s |
| **permuted slots**, 6 replicas | **0.226 m** | 0.00020 m, 0.0401 rad/s |

A world's position in the batched contact solve is a persistent property of the
world, not a fresh coin flip per rollout. So v2's replication mechanism — which
v3 inherited — sampled roughly a *sixth* of the variance the verification bar
measures, and the gate built to match the bar was mechanically a fraction of
it.

**This is part of the answer to why v2's re-evaluated run still verified at 8%.**
Its two replicas were not two draws of a chaotic quantity; they were one draw
plus a small perturbation of it. The N=2 rule was weaker than even its own
stated limitation ("a policy that survives 2 of 2 has a 95% interval reaching
down to ~0.22") assumed, because that interval assumes independence.

The fix costs nothing: shuffle which world each candidate occupies on each
replica and un-permute the results — same worlds, same replica count, same wall
clock. `--insertion-permute-worlds` is **on by default** from v3 onward;
`--no-insertion-permute-worlds` reproduces v2's behaviour exactly, which is what
the ablation below uses.

The generalisable version, for anyone replicating a stochastic evaluation on a
batched simulator: **re-running the same batch is not the same as resampling.**
If the batch index is part of the state, replicate across indices.

### Results — v2 vs v3, all on medians

> Same warning as v2's table, and it still governs everything below. On this
> simulator a walking policy's displacement has a standard deviation of 0.605 m
> across byte-identical worlds, so every "verified" number here is the **median
> of 8 fresh rollouts** per elite, produced by a verification pass that is
> independent of insertion, and an elite counts as surviving only if it stayed
> upright in at least 7 of those 8.
>
> Cell counts are given **resolvable / raw**. Raw counts cells of the 20x20
> grid. Resolvable re-bins the same elites at the resolution the descriptor's
> own reproducibility supports — bins `2*sqrt(2)` standard errors of the
> archived median wide, which on the final archive is a 9x15 grid. **Quote the
> resolvable number**: a cell count on a grid finer than the measurement is a
> count of quantization, not of behaviours.

| | v2 single-sample | v2 re-evaluated (N=2) | v3 in-place ablation (it 10) | v3 permuted (it 10) | v3 budget-matched (213k evals) | **v3 full (426k evals)** |
| --- | --- | --- | --- | --- | --- | --- |
| **VERIFIED — 8 fresh rollouts per elite** | | | | | | |
| elites surviving 7-of-8 | 4 | 5 | 193 | 223 | 242 | **268** |
| archive robustness (7-of-8) | 5% | 8% | 65% | **83%** | 74% | 75% |
| best verified median displacement | +1.620 m | +2.117 m | +2.099 m | +2.222 m | +2.167 m | **+2.286 m** |
| cells with a verified elite >= 0.25 m | 4 | 5 | 90 / 186 | 116 / 219 | 110 / 241 | **109 / 263** |
| cells with a verified elite >= 0.50 m | 2 | 5 | 84 / 170 | 113 / 211 | 110 / 238 | **109 / 261** |
| mean survival rate per elite | 0.296 | 0.395 | 0.836 | **0.925** | 0.885 | 0.883 |
| elites that never survived a replay | 17 | 5 | — | — | — | **0** |
| **STRUCTURE — raw, counts nothing about robustness** | | | | | | |
| elites | 81 | 62 | 299 | 268 | 327 | 356 |
| raw coverage | 20.2% | 15.5% | 74.8% | 67.0% | 81.8% | 89.0% |
| *archive optimism (archived − verified median)* | *+0.643 m* | *+0.887 m* | *+0.516 m* | *+0.159 m* | *+0.209 m* | *+0.261 m* |
| cell stability (exact / within one) | — | — | 18% / 64% | 35% / 91% | 39% / 88% | 33% / 90% |

**The headline.** Against v2's best archive — 5 robust elites in 5 cells — v3
holds **268 robust elites in 109 resolvable cells**, and its best verified
walker covers **+2.286 m** against +2.117 m. The elites that never survive a
single replay, 17 in v2's single-sample archive and 5 after its N=2 fix, are
**zero**. At *equal* evaluation budget (the 207k snapshot) the numbers are
242 elites in 110 cells: the win is not bought with the extra budget. (That
snapshot fires on the first iteration *past* j003's 207,049 evaluations, so it
is 213,192 — 3% more, in v3's favour by 3%, which is smaller than the few-elite
uncertainty the verification itself carries.)

Both halves contributed and they are separable. The descriptor is why there are
cells to fill at all — v2's search was not failing to find gaits, it was
finding them in a space that could not tell them apart. The gate is why the
ones in the archive are real: archive optimism falls from +0.887 m to +0.26 m,
and mean per-elite survival from 0.40 to 0.88.

#### Verification strictness, reported whole

The same sweep v2 published, on the final v3 archive (356 raw elites), cells as
resolvable / raw:

| survives at least | elites | cells >= 0.25 m | cells >= 0.50 m | best median |
| --- | --- | --- | --- | --- |
| 4 of 8 | 354 | 114 / 348 | 114 / 343 | +2.286 m |
| 5 of 8 | 348 | 114 / 342 | 114 / 338 | +2.286 m |
| 6 of 8 | 332 | 114 / 326 | 114 / 324 | +2.286 m |
| **7 of 8** | **268** | **109 / 263** | **109 / 261** | **+2.286 m** |
| 8 of 8 | 146 | 89 / 146 | 88 / 144 | +2.266 m |

The property v2 could only claim at its best threshold — that "elites" and
"cells above 0.50 m" are nearly the same number — holds here at *every*
strictness level. There are essentially no weak survivors: an elite that stays
up in this archive is an elite that travels.

#### What the replication fix bought, measured

Two runs identical in every respect except how the eight insertion replicas
were drawn, compared at the same iteration (10) and the same evaluation count:

| at iteration 10 | replicas in place (v2's mechanism) | replicas across permuted worlds |
| --- | --- | --- |
| archive robustness (7-of-8) | 65% | **83%** |
| mean survival rate per elite | 0.836 | **0.925** |
| archive optimism | +0.516 m | **+0.159 m** |
| cell stability (exact / within one) | 18% / 64% | **35% / 91%** |
| resolvable cells >= 0.25 m | 90 | **116** |
| raw elites | 299 | 268 |

The permuted gate admits 31 fewer elites and 18 percentage points more of them
are real. It also more than halves the archive's optimism and doubles its
cell stability — which is the second half of the same story: an insertion
median over eight *correlated* replicas is a worse estimate of the descriptor,
not only of the fitness, so the geography was fuzzier too.

#### Robustness fell short of the bar, and the reason is measurable

The pre-registered criterion was **>= 90%** of final-archive elites surviving
7-of-8. Measured: **75%**. Not a pass, and worth being precise about why,
because the shortfall is structural rather than a tuning miss.

Mean true survival across the archive is **0.883**. For an elite with true
survival `p`, the chance of clearing a 7-of-8 bar is `8p⁷(1−p) + p⁸`:

| true survival `p` | P(passes 7 of 8) |
| --- | --- |
| 0.80 | 0.503 |
| 0.85 | 0.657 |
| 0.883 | **0.762** |
| 0.90 | 0.813 |
| 0.95 | 0.943 |

At the archive's measured mean of 0.883 the expected pass rate is **0.762**,
against **0.753** observed — the archive is behaving exactly like a population
whose typical member is 88% robust, with no residual to explain. **A 90% pass rate at a 7-of-8 bar requires nearly
every elite to be genuinely 95%-robust**, which is a much stronger demand than
"the gate matches the bar" — the criterion and the gate are not the same
threshold, and I did not notice that when the criterion was set.

And there is a mechanism pushing the archive *down* as the search runs, visible
in the table above: robustness goes **83% (it 10) -> 74% (it 24) -> 75% (it 50)**
while raw coverage climbs 67% -> 82% -> 89%. **The winner's curse applies to
the survival predicate, not only to the fitness.** j003 removed luck-ranking
from fitness by taking a median instead of a maximum; survival is still a
max-like operator — an elite is in the archive because it passed unanimously
*once*, out of ~51,000 candidate evaluations. A genuinely 88%-robust genome
passes unanimous-8 with probability 0.36, so with enough attempts the archive
fills with exactly those. Raising N raises the exponent but never removes the
selection: it is the same asymmetry, one level up.

**The fix this points at is not a bigger N.** It is what v2's README already
named as the open direction and neither run built: a **running survival
estimate per cell that accumulates across an elite's lineage**, or simply
re-testing incumbents so that an elite has to keep passing rather than pass
once. That converts survival from a one-shot maximum into an average, which is
precisely what taking the median did for fitness. It is the obvious next thing
to build, and it is not built here.

### Reproduction

```bash
# Stage A — the descriptor table (~10 min, writes selection.md/.json)
uv run python -m qd.select_descriptor \
    --seeds logs/qd/seeds/ppo_seeds.npz \
    --elites qd-run-archives/j003/qd/pga_me_v2_reeval/archive_final_verified.npz \
    --replicas 64 --mutation-probe 1024 --out logs/qd/descriptor_selection

# Stage B — 50 iterations x 1024 offspring x 8 replicas = 426,392 evaluations
uv run python -m qd.pga.run_pga_me --iterations 50 --batch-size 1024 \
    --initial-solutions 1024 --seed-genome logs/qd/seeds/ppo_seeds.npz \
    --seeding.jitter-count 240 --insertion-replicas 8 \
    --td3.replay-buffer-size 2000000 \
    --descriptor.axis-x torso_height_mean --descriptor.x-range 0.11667 0.12184 \
    --descriptor.axis-y joint_speed --descriptor.y-range 0.89228 2.50831 \
    --budget-checkpoint-evals 207049 --out-dir logs/qd/pga_me_v3

# the honest numbers: fresh replicas, independent of insertion
uv run python -m qd.verify_archive --archive logs/qd/pga_me_v3/archive_final.npz
uv run python -m qd.verify_archive --archive logs/qd/pga_me_v3/archive_budget.npz
```

```bash
# the replication ablation: identical, except replicas stay in their world
uv run python -m qd.pga.run_pga_me --iterations 10 --batch-size 1024 \
    --initial-solutions 1024 --seed-genome logs/qd/seeds/ppo_seeds.npz \
    --seeding.jitter-count 240 --insertion-replicas 8 \
    --no-insertion-permute-worlds --td3.replay-buffer-size 2000000 \
    --descriptor.axis-x torso_height_mean --descriptor.x-range 0.11667 0.12184 \
    --descriptor.axis-y joint_speed --descriptor.y-range 0.89228 2.50831 \
    --out-dir logs/qd/pga_me_v3_inplace_ablation
```

`--budget-checkpoint-evals 207049` snapshots the archive the moment the run
passes j003's evaluation count — the first iteration boundary past it, which
here is iteration 24 at 213,192 evaluations — so the v2-vs-v3 comparison can be read both at
equal cost and at full budget. Measured on an RTX 3060: iteration 0 (seed block
plus random initialisation, 16 rollouts) 278 s, then **141 s per iteration** —
**426,392 evaluations in 2.03 h**, 2.06x j003's evaluations for 1.4x its wall
clock, because eight replicas of one batch amortise better than eight separate
generations.

## Watching the gaits

```bash
# render every filled cell to mp4 (needs the MUJOCO_GL prefix — see below)
MUJOCO_GL=glfw uv run python -m qd.render_gaits \
    --archive logs/qd/pga_me_v2/archive_final.npz --out logs/qd/gaits/v2

# the two v1 archives, rendered under v1 physics so the clips match the
# numbers that archive was searched under (`--no-full-collision`
# `--no-trim-at-fall` reproduces exactly what v1 published)
MUJOCO_GL=glfw uv run python -m qd.render_gaits \
    --archive logs/qd/map_elites/archive_final.npz --out logs/qd/gaits/cpg \
    --no-full-collision --no-trim-at-fall
MUJOCO_GL=glfw uv run python -m qd.render_gaits \
    --archive logs/qd/pga_me_matched/archive_final.npz --out logs/qd/gaits/pga \
    --no-full-collision --no-trim-at-fall

# v3: render the VERIFIED archive, so the colour is a median over 8 rollouts
MUJOCO_GL=glfw uv run python -m qd.render_gaits \
    --archive logs/qd/pga_me_v3/archive_final_verified.npz --out logs/qd/gaits/v3

# build the clickable page: four archives, one switcher
uv run python -m qd.build_viewer \
    --manifests logs/qd/gaits/cpg/manifest.json \
                logs/qd/gaits/pga/manifest.json \
                logs/qd/gaits/v2/manifest.json \
                logs/qd/gaits/v3/manifest.json \
    --labels "v1 MAP-Elites (CPG)" "v1 PGA-ME (MLP)" "v2 gated + seeded" \
             "v3 measured axes + independent 8-replica gate" \
    --budget-mb 2.6 --out logs/qd/viewer/index.html
```

The page is a 20×20 heatmap you click: pick a cell and that elite's gait plays
beside its distance, time upright, outcome and both of **that archive's own**
behaviour axes — the labels under the grid change with the switcher, because a
v3 archive is not binned on the same quantities a v1 or v2 archive was. Duty
factor is still reported for every clip whatever the archive's axes are, so a
v2 gait and a v3 gait can be compared on it. Arrow keys step to the nearest
*filled* cell, since the archive has holes. Each manifest adds an entry to the
archive switcher, and `--budget-mb` is **per archive**, so four archives on one
page need about a quarter of the per-archive budget one did.

Rendering the **verified** archive rather than the raw one is the honest choice
for a gated run, and the page says so: the colour is then the elite's median
over eight fresh rollouts, while the clip beside it is a single one of those
rollouts, which on this simulator can differ from the median by half a metre.

**v2 clips end at the fall.** `--trim-at-fall` (on by default) cuts each clip on
the frame that world's fall was detected, so a clip is exactly the trajectory
that was scored — no frames the fitness, descriptor and replay buffer all
refused to look at. In a gated v2 archive every elite is a survivor, so nothing
is trimmed there; it is the *v1* archives, replayed, where the difference shows.

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
**every elite that survived the full episode first**, then the top elites by
fitness, then a spatial sweep across the descriptor space. Survivors lead
because they are the rarest thing in either archive — one gait out of 632 — and
they are *not* the highest-scoring, so a fitness-ranked selection drops exactly
the clip a reader most needs to see. Every filled cell is still rendered to disk
at full resolution, and cells without an embedded clip stay clickable, showing
their stats and the file path.

The one survivor sits at cell (18, 19) — both feet on the ground ~95% of the
episode. Watching it is the fastest way to understand the headline: it is
standing, with a shuffle.

## Comparing archives

`qd.compare_archives` leads with **surviving-elite coverage**, not raw
coverage, and the distinction is the whole reason a v1 archive and a v2 archive
can be compared at all. Raw coverage counts every filled cell whatever is in
it; under v1's fall-penalty objective nearly every cell held a policy that
falls. Surviving-elite coverage counts only cells whose elite is a
replay-verified full-episode survivor — the same question asked of both
pipelines: *how much of the behaviour space can this robot reach without
falling over.* Raw coverage is still printed underneath.

```bash
uv run python -m qd.compare_archives \
    --a logs/qd/map_elites/archive_final.npz --a-label "MAP-Elites (CPG)" \
    --b logs/qd/pga_me_matched/archive_final.npz --b-label "PGA-ME (MLP)" \
    --out logs/qd/comparison
```

Writes `comparison.png` — both heatmaps on a shared colour scale plus a
difference map showing which cells each pipeline reached — and
`comparison.json` with coverage, QD-score, best-cell fitness, and the mean
fitness delta over the cells both filled. Pass the same `--qd-score-offset` the
runs used, or the QD-scores are not comparable.

## Later upgrades (not built)

* ~~Re-evaluation on insertion.~~ **Built** — `--insertion-replicas`, see
  [above](#the-gate-is-necessary-and-not-sufficient-re-evaluation-on-insertion),
  and made to sample independently in v3
  ([`--insertion-permute-worlds`](#the-replicas-were-not-independent-and-that-is-most-of-why-v2-failed)).
  What remains open is the *shape* of it, and v3 measured why it matters. This
  implementation spends N rollouts on every candidate, including the ones that
  fall in the first replica and can never pass; a sequential rule — one rollout,
  and only survivors earn a second — would buy most of the filtering at a
  fraction of the cost.
* **Survival as an average, not a maximum.** The most important open item, and
  v3's clearest negative result. An elite is in the archive because it passed a
  unanimous-N gate *once*, out of ~51,000 candidate evaluations, so the
  winner's curse that j003 removed from the fitness column is still fully
  operative on the survival column: measured archive robustness *falls* from
  83% to 75% as the search runs and the number of attempts grows. Raising N
  raises the exponent and does not remove the selection. A **per-cell running
  survival estimate that accumulates across an elite's lineage** — or simply
  re-testing incumbents, so an elite has to keep passing rather than pass once
  — converts survival into an average, which is exactly what the median did for
  fitness.
* ~~A descriptor this robot can actually move in.~~ **Built** — see
  [Walking v3](#stage-a--choosing-the-axes-by-measurement). The prediction that
  "stride length, lateral drift, energy or a turning rate would all vary more"
  was half right: energy, limb speed, turning rate and trunk height all beat
  duty factor by a wide margin, lateral drift failed the noise threshold, and
  stride length turned out to be the *objective* in disguise (|corr| 1.00 with
  displacement) and was rejected for it. What remains open is a **log-scaled or
  non-uniform axis**: cost of transport is the most objective-independent
  quantity measured here and was passed over only because it is ~1/distance and
  piles 53% of feasible mutants into one bin of a linear grid.
* **A descriptor-aware PG operator.** PG variation currently ascends Q, which
  knows nothing about the behaviour descriptor, so it improves elites in place
  instead of moving them to new cells. A critic conditioned on a target
  descriptor would make the gradient half of PGA-ME an *exploration* operator
  rather than a refinement one — which is exactly what a survival-gated search
  over a thin feasible manifold needs.
* **CMA-ME** — swap the `GaussianEmitter` for pyribs'
  `EvolutionStrategyEmitter`, which adapts a covariance matrix per emitter and
  usually beats isotropic mutation badly on a 31-D search space. Deliberately
  out of scope here: the point of Phase 2 is a vanilla MAP-Elites baseline for
  Phase 3 to be measured against.
* Richer descriptors (stride length, lateral drift, energy) or a 3-D archive.
* CPG coupling terms (a phase-locked oscillator network) instead of independent
  per-joint phases.
