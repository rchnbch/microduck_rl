# `qd/` — Quality-Diversity gait discovery for Microduck

Gradient-free (and, in Phase 3, gradient-*assisted*) search for a **diverse
archive** of forward-walking gaits, on the same MuJoCo-Warp simulation the PPO
recipes train in.

Where PPO returns one policy that maximizes a reward, MAP-Elites returns a
**grid of policies**, one per bin of a behaviour descriptor, each the best of
its kind. Here the descriptor is per-foot ground-contact duty factor, so the
archive spans everything from a shuffle that never lifts a foot to a hopping
gait with both feet airborne most of the time.

> **Read [Walking v2](#walking-v2--survival-gated-ppo-seeded-pga-me) first if you
> want the current state.** v1 (Phases 2-3, below) is the baseline record: it
> produced archives full of policies that *fall over*, because falling was
> priced rather than forbidden. v2 makes not-falling a feasibility constraint,
> fixes the physics the fall was measured under, and starts the search from the
> PPO walker. Everything below v1's Results section still describes machinery
> v2 uses unchanged.

```
qd/
├── common.py          fitness + behaviour descriptor + archive/plot/checkpoint helpers
├── cpg_genome.py      Phase 2 genome: 31-parameter open-loop CPG
├── evaluate.py        batched mjlab rollout harness: genomes -> (fitness, descriptor)
├── run_map_elites.py  Phase 2 CLI: pyribs GridArchive + GaussianEmitter ask/tell loop
├── seed.py            v2: distil the PPO walker into the genome (DAgger)
├── play_elite.py      inspect / replay one elite from a saved archive
├── check_harness.py   physics sanity checks — run before any long run
├── check_floor.py     v2: is any part of the robot under the plane?
├── check_repeatability.py  v2: how much is one evaluation worth? (not much)
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

**What it costs: nothing.** `qd.bench_collision` at batch 1024 on an RTX 3060,
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

### Budget and reproduction

Budget-matched to v1's 207,048 evaluations:

```bash
# 1. distil the seed (once, ~4 min)
uv run python -m qd.seed \
    --checkpoint logs/rsl_rl/qd_phase1_baseline/<run>/model_399.pt \
    --out logs/qd/seeds/ppo_seed.npz

# 2. the run: 1025 (seed block) + 1024 (random init) + 200*1025 = 207,049 evals
uv run python -m qd.pga.run_pga_me --iterations 200 --batch-size 1024 \
    --initial-solutions 1024 --seed-genome logs/qd/seeds/ppo_seed.npz \
    --td3.replay-buffer-size 2000000 --out-dir logs/qd/pga_me_v2
```

The seed block is one rollout holding the seed, 100 jittered variants
(`jitter_sigma` 0.02, 4x the GA's iso term so the variants do not all land in
the seed's own cell) and 924 random MLPs. Together with `--initial-solutions
1024` that is 1948 random initialisations against v1's 2048 — the random-init
budget is essentially preserved, with 101 seeded genomes added.

Random MLPs are still evaluated with a seed present. Under the gate almost none
of them are inserted, but they are precisely the falling-over experience the
critic needs in its buffer to learn what not to do.


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

# build the clickable page: three archives, one switcher
uv run python -m qd.build_viewer \
    --manifests logs/qd/gaits/cpg/manifest.json \
                logs/qd/gaits/pga/manifest.json \
                logs/qd/gaits/v2/manifest.json \
    --labels "v1 MAP-Elites (CPG)" "v1 PGA-ME (MLP)" "v2 gated + seeded" \
    --budget-mb 3.3 --out logs/qd/viewer/index.html
```

The page is a 20×20 heatmap you click: pick a cell and that elite's gait plays
beside its fitness, duty factors, distance, time upright and outcome. Arrow keys
step to the nearest *filled* cell, since the archive has holes. Each manifest
adds an entry to the archive switcher, and `--budget-mb` is **per archive**, so
three archives on one page need about a third of the per-archive budget two did.

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

* **Re-evaluation on insertion.** The single most valuable thing not built
  here, and the measurement that says so is
  [above](#the-finding-that-reframes-every-number-here-a-walkers-fitness-is-chaotic):
  one rollout of a walker has a 0.6 m standard deviation in displacement, and
  MAP-Elites inserts on one sample and keeps the maximum, so archived values
  rank luck as much as ability (+0.60 m of it, measured). Re-evaluating a
  candidate before insertion, or keeping a running mean per cell, fixes it at
  the cost of a second evaluation per candidate — half the archive at the same
  budget. Whether that trade is worth it is itself an experiment worth running,
  and it is the one to run next.
* **A descriptor this robot can actually move in.** Per-foot duty factor is
  nearly invariant across everything the robot can do without falling: six
  teacher commands spanning a 0.41 m to 2.42 m stride all distil to duty
  factors within 0.03 of each other, and 256 identical replicas of one genome
  spread only 0.014. Stride length, lateral drift, energy or a turning rate
  would all vary more, and a QD archive is only as interesting as the axes it
  is binned on.
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
