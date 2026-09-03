# QD modes — design draft: descriptors and viability for genuinely distinct locomotion modes

**Status:** draft for discussion, no implementation. Branch `qd-modes-design`.
**Feeds:** the implementation job that follows Alex's sign-off (j006+).
**Reading time:** ~15 min for sections 0–7; appendices are derivations,
probe logs and the sim-state inventory.

Alex's verdict on v3: *"the behaviors are not different enough… I don't just
want walking."* This draft is the design that turns the v3 machinery into an
archive of *modes of forward motion* — walking, crawling, rolling, and
whatever else moves forward — without readmitting the junk that filled v1.
Hopping is analysed too, and the measurement in Appendix A says this robot
cannot do it; it stays in the design only as a label the search may surprise
us with, not as a seeded mode.

Everything quantitative below is either a v1–v3 measurement (cited), a number
read off the model files (`robot_allcollisions.xml`, BAM `xl330/m6.json`,
`microduck_constants.py`), or a throwaway scripted probe run for this draft
on the v3 harness (Appendix D lists exactly what was run). Nothing landed in
`qd/`.

---

## 0. The recommendation in one paragraph

Replace the upright gate with a **sustained-progress predicate**: the robot
must advance in *every* 2 s window of the 7 s episode (first window exempt for
the stand-to-mode transition), keep the *same mode label* in every window
after the first, and pass unanimously across 8 world-permuted replicas, with
a measured cap on impact violence. Classify every viable rollout into a
**mode** by three contact/posture quantities (non-foot ground-contact
fraction, whole-body airborne fraction, net supported rotation rate), require
the label to agree across replicas *and across windows*, and keep **one 20×20
sub-archive per mode** — walk on v3's measured axes, the other modes on axes
chosen by the same Stage-A measurement run on mode-specific probes, plus an
"other" bucket for anything that moves forward and fits no rule. Fitness stays
median +x displacement over the replicas; modes never compete with each
other. Seed walk from the PPO walker as today; seed crawl from a new
prone-locomotion PPO task that *also* spawns standing (so the seed knows how
to get down); seed roll from a distilled roulade policy plus a chained-roll
variant; do **not** seed hop — a scripted full-effort launch from the deepest
stable pose lifts the trunk 2 cm at 0.19 m/s and never leaves the ground
(Appendix A), so a hop archive would be empty by physics, not by search.
Nothing here is expected to be *discovered* from walking seeds: v2 measured
that isotropic mutation cannot leave the feasible manifold it starts on, so
every mode needs its own seed. Scripted open-loop probes for Stage A′ must
start from a stable ground rest pose (prone, supine, seated), because no
open-loop posture survives more than ~1 s from a standing start on this
robot (measured, Appendix D).

---

## 1. Viability redesign

### 1.1 What the upright gate was actually doing

Three jobs, and only one of them is "upright":

1. **Excluding v1's divers.** v1's objective was `displacement_at_fall −
   0.25·fallen_fraction`; a ballistic dive covering 0.4 m and lying down scored
   +0.22 against a standing 0.00. 562 of 632 v1-era elites were policies
   optimised to dive well. The gate made falling *inadmissible* rather than
   priced.
2. **Keeping the gate off the chaos.** Survival is near-deterministic (98.8 %
   over 256 identical replicas of the 0.4 m/s seed) while displacement is
   chaotic (sd 0.605 m, trunk-x spread reaching 100 mm by 0.52 s). Gating on a
   stable predicate meant the gate did not inherit the chaos. That asymmetry
   is the whole reason the gate worked.
3. **Equalising the descriptor's integration window.** Every archived elite is
   a full-episode passer, so every descriptor is a 7 s time average and the
   axes are comparable across elites.

A replacement predicate has to keep all three. It also has to stop doing the
fourth thing the upright gate does by construction: declaring a crawl dead at
`base z < 0.075 m` and a roll dead at `tilt > 60°`, before the first step.
(Scale of the problem: the verified-stable SIT keyframe sits at trunk
z = 0.061 m, already below `fall_height`; prone rest is 0.075 m, supine rest
0.048 m. Every non-walking rest pose this robot has is "fallen" to v3.)

### 1.2 The anatomy of a degenerate diver

Every degenerate v1 policy, whatever it looked like, shared one signature:
**its forward progress was front-loaded and then stopped.** A dive covers its
distance in the first second and lies still; a face-plant that skids covers a
few centimetres more and then twitches in place; a topple-forward covers
whatever the trunk's arc gives it. None of them keeps going.

Legitimate crawls and rolls share the opposite signature: progress keeps
accruing, window after window, because the *mechanism* repeats.

That is the predicate. Not "what posture is it in" but "is it still going".

### 1.3 Candidate predicates, and what each admits

| # | predicate | dive & skid | face-plant & twitch | crash-lunge loop | static crouch | walk | crawl | roll | late fall (walks 6 s, face-plants at 6.5 s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P1 | **upright** all episode (v3) | excluded | excluded | excluded | admitted (fitness 0) | admitted | **excluded** | **excluded** | excluded |
| P2 | **sustained progress**: ≥ d_min in every W-window | excluded (later windows fail) | excluded | admitted if each lunge clears d_min | excluded | admitted | admitted | admitted | **admitted** (last window's progress was earned before the fall) |
| P2′ | P2 **+ mode label constant across windows** | excluded | excluded | admitted as "other" | excluded | admitted | admitted | admitted | excluded (label flips walk → other in the last window) |
| P3 | **bounded stress**: p95 trunk \|a_z\|, peak non-foot contact force, joint-limit proximity | excluded only if the impact is measured as violent | admitted (twitching is gentle) | excluded | admitted | admitted | admitted | depends on threshold | depends |
| P4 | **mode-conditional**: walk→upright, crawl→prone+progress, roll→rotation+progress | excluded | excluded | excluded unless a "lunge" mode is defined | admitted (walk) | admitted | admitted | admitted | excluded |
| P5 | **no gate**, insert on robust median fitness | **admitted** (a dive is *repeatable*) | excluded (median ≈ 0) | admitted | excluded | admitted | admitted | admitted | admitted |
| P6 | **never inert**: joint motion in every window | excluded | **admitted** (twitching counts) | admitted | excluded | admitted | admitted | admitted | admitted |

Notes on the rows that matter:

- **P5 is the trap.** It is tempting because it removes the "what counts as
  alive" question entirely. But v1's divers are *open-loop-like* ballistic
  motions over a short horizon, and short-horizon motions are exactly what this
  simulator reproduces well (open-loop CPG noise ≤ 4.4 mm; the 0.605 m chaos
  is a *closed-loop walker integrating 350 steps*). A dive's median over 8
  replicas is a confident +0.4 m. P5 readmits v1 wholesale, with better
  statistics.
- **P2 alone has a hole at the end of the episode.** Any window-based
  progress rule can be satisfied early in the last window and then abandoned:
  a walker that covers 5 cm in [5.0, 6.5] s and face-plants at 6.5 s passes.
  Under the v3 gate this was impossible; under P2 it is the cheapest way to
  end an episode. **P2′** closes it without naming modes in the predicate:
  the mode classifier (§2.4) is evaluated per window, and the label must be
  the same in every window from the second onward. A late fall changes
  `f_body` from ≈ 0 to ≈ 1 in the last window, the label flips, the rollout
  is out. A crawl that stands up at the end flips the other way and is out
  too — correctly, it is not a robust crawl.
- **P3 alone cannot carry viability.** A gentle face-plant is gentle. Stress
  bounds are a *style* constraint (they say which crawls we want), not a
  liveness constraint. They belong in the gate as an added clause, threshold
  measured (§6), never as the only clause.
- **P4 is clean but circular.** It needs the modes enumerated in advance and
  defines "viable" differently per mode, so it can never admit "anything else
  creative" — the thing Alex asked for. It also makes cross-mode junk
  (a walker that collapses into something crawl-shaped) admissible under the
  crawl rule. Mode labels are still needed for the archive (§3), but as a
  *classifier of viable rollouts*, not as the definition of viability.
- **P6 is P2's cheap cousin and fails on twitching.** Keep it as a diagnostic.

### 1.4 The proposed predicate (P2′)

A rollout is **viable** iff, in every replica:

1. **Finite state** throughout (existing NaN guard).
2. **Sustained progress.** With windows of length W = 2 s at stride 1 s over
   the 7 s scored episode (six windows), the trunk's +x displacement over each
   window k ≥ 2 is at least d_min. Window 1 is exempt: every mode spawns
   standing at HOME (§4.1) and needs the first second to become the thing it
   is. Initial d_min = 0.05 m (2.5 cm/s); **the final value is chosen in
   Stage A′ (§6)**, not here.
3. **Constant mode.** The mode label (§2.4), computed from the classifier
   features accumulated *per window*, is identical for windows 2–6. A label
   of "other" is allowed but must also be constant.
4. **Bounded violence.** The 95th percentile over the episode of trunk
   vertical acceleration |a_z| (from the `imu_accel` sensor already wired for
   `trunk_vertical_accel_penalty`) is below a cap. The cap is set in Stage A′
   as 1.5× the worst *intended-mode* probe, and is kept only if it excludes
   ≥ 90 % of the v1 divers at that setting; otherwise it is dropped and
   clauses 2–3 carry the load alone. A percentile, not a max: a max is a
   luck-ranked operator (v3's winner's-curse lesson applies to any max-like
   predicate).

Aggregation across the 8 permuted replicas is **unanimous by default**, with
one deliberate escape hatch: if Stage A′ measures the pass rate of the *known
robust* probes below 0.95 per replica, the rule becomes 7-of-8, and the reason
is written next to the number. v3's arithmetic is the reason to be explicit:
a policy that is 88 % robust per replica passes unanimous-8 only 36 % of the
time, and a 90 % archive pass rate at 7-of-8 already demands 95 %-robust
elites. The progress predicate is *less* deterministic than "upright" — the
0.4 m/s seed's 5th percentile displacement is −0.05 m, so a few percent of its
replicas would fail any absolute window floor — and the rule has to be set
from that measurement rather than assumed.

Why windows over the whole episode and not just "still moving at the end":
a final-window-only rule admits a policy that stands for five seconds and
dives at t = 5. Requiring progress in every window is what makes the front-
loaded profile of a dive inadmissible wherever it is placed in the episode;
requiring a constant label is what makes the *end* of the episode as honest
as the middle.

What P2′ gives up, stated plainly:

- **Standing still is no longer viable.** An in-place stepping "stand" scored
  0 in v3 and sat in the archive as a legitimate cell. Under P2′ it is out.
  For an archive of *forward locomotion* that is the right call; the
  zero-command stand is a runtime policy, not a QD mode.
- **A crash-lunge loop is admissible if each lunge clears d_min and the
  impact cap, and its label is a constant "other".** If the search finds one,
  it is a locomotion mode by the letter of this rule. I would let it in and
  let the human look at it; the impact cap is what stops it being *violent*,
  and the "other" bucket is where it lands.
- **The fall latch goes.** Descriptors and fitness integrate over the full
  7 s for every candidate (property 3 of §1.1 is preserved because every
  archived elite is a full-episode passer, as before).
- **The predicate now depends on the classifier.** A bad classifier threshold
  becomes a bad viability rule. That is why §6 measures label stability on
  every probe before either is used.

### 1.5 Why not simply widen the thresholds

Because the thresholds are not the problem. `fall_height = 0.075` could be
set to 0.02 and `fall_tilt_deg` to 179 and the predicate would still say
"a robot lying on its face is alive", which is what v1 said. The upright gate
worked by *excluding* a set of states; P2′ works by *requiring* a behaviour.
Only the second generalises to modes whose states look like falling.

---

## 2. Descriptor proposal for mode separation

### 2.1 What the harness already measures, and what must be added

Per control step, `RolloutMetrics.update` receives base position, projected
gravity, per-foot contact `found`, and the optional extras `lin_vel_w`,
`ang_vel_b`, `joint_vel`, `qfrc_actuator` (`qd.descriptors.EXTRA_CHANNELS`).
`GaitStats` accumulates nineteen axes from these as running sums.

Available from `robot.data` and the scene but **not yet plumbed** into the
extras (Appendix B has the full inventory):

| channel | source | needed by |
| --- | --- | --- |
| `body_contact` (N, n_bodies) bool | new `ContactSensorCfg(primary=ContactMatch(mode="body", pattern=<list>), secondary=terrain, fields=("found",), reduce="none", num_slots=1)` — same construction the roulade env uses for `robot_ground_contact` and `head_ground_contact` | support-class fractions, airborne fraction, head involvement |
| `foot_force` (N, 2) | the feet sensor's `force` field (already requested by the velocity/standup cfgs, dropped by the QD cfg) | force-thresholded contact, flight detection without chatter |
| `root_quat_w` (N, 4) | `robot.data.root_link_quat_w` | lateral-axis flatness, inversion fraction |
| `trunk_acc` (N, 3) | `sensordata` at the `imu_accel` address (as `trunk_vertical_accel_penalty` reads it) | impact cap, landing gentleness axis |
| `root_angmom` (N, 3) | `subtreeangmom` sensor `root_angmom` (already in `sensors.xml`) | rotation-based axes without integration |
| `z_trace` (T, N) | a trajectory buffer, not a running sum | spectral / periodicity axes (§2.3, A6) |
| per-window accumulators (6, N) of the classifier features and of +x displacement | new running sums, reset at each window boundary | P2′ clauses 2–3 |

Which bodies can touch the ground on `robot_allcollisions.xml` as exported
(11 contact-enabled geoms, read off the compiled model): `trunk_base` (the
battery-pack mesh `np_f970`; the `power_support` mesh is self-collision-only),
`hip_l` / `hip_l_2`, `leg` / `leg_2` (the shanks — what a "knee" contact
actually is), the two soles, and `jaw_soft` (top head shell, jaw, bottom head
shell). **The upper legs and the trunk side shells have no ground-contact
geom** — a crawl will rest on exactly those, so this needs checking before
any crawl is trained (§7, open question 4). And **none of the shell geoms is
named**, so `FULL_COLLISION`'s `condim=1` rule for `.*_collision` (intended
frictionless shells) matches only the two named soles; every shell contact
runs on MuJoCo's defaults (condim 3, μ = 1, priority 0).

### 2.2 Two kinds of axis, and why the archive needs both

v3's Stage A found that axes worth binning on have replica sd under 10 % of
the between-gait spread. Among *walkers* that spread is tiny: 5 mm of trunk
height, 1.6 rad/s of limb speed. Between *modes* the same quantities move by
an order of magnitude more (trunk height 0.03 m prone vs 0.115 m standing),
which makes them superb mode separators and useless within-mode axes on the
same grid — 20 bins over 90 mm puts every walker in one cell, which is the
v2 duty-factor failure in a new costume. So:

- **Mode-separating quantities** feed a *classifier* (§2.4), not a grid axis.
- **Within-mode axes** are chosen per mode by the Stage-A procedure and get
  their own grid (§3).

### 2.3 Candidate axes, evaluated

Ranges are expectations to be *measured*; where a number is quoted it is
from v3's measurements (walk) or from Appendix A.

**A1 — support-class occupancy** (from `body_contact`; three fractions of
control steps): `f_feet` (only soles touching), `f_body` (any of trunk / hip /
shank / head touching), `f_air` (nothing touching).

| mode | f_feet | f_body | f_air |
| --- | --- | --- | --- |
| walk | ≈ 1 | ≈ 0 | ≈ 0 (v3 feet-only flight fraction: 0.013 spread, 0.005 sd) |
| crawl | ≈ 0 | ≈ 1 | 0 |
| roll (somersault chain) | 0.1–0.3 | 0.6–0.9 | ≤ 0.1 |
| hop (if it ever exists) | 0.7–0.9 | ≈ 0 | 0.1–0.3 |

Expected replica noise: contact fractions are thresholded and time-averaged,
the same family as duty factor, whose sd was 0.014. Expect ≈ 0.01–0.02.
Failure modes: a fallen-and-twitching robot reads `f_body ≈ 1` exactly like a
crawl — this axis *cannot* tell them apart and never should; P2′ does.
Touchdown chatter inflates `f_air` if `found` is used raw (the probe logs in
Appendix D show single-step foot-contact dropouts during ordinary settling);
use the force field with a threshold (≥ 0.5 N) or require two consecutive
airborne steps.

**A2 — whole-body airborne fraction** is `f_air` above, singled out because
the existing `flight_fraction` is feet-only and reads "flight" for a robot
lying on its back with its feet in the air. Only the whole-body version means
anything.

**A3 — net supported rotation rate** about the body lateral axis:
`|Σ ω_y·dt·supported| / T`, integrated exactly as the roulade accumulator does
(`root_link_ang_vel_b[:,1]`, gated on any-body ground contact so ballistic
spins do not count), reported as rad/s. Also the roll-axis version
`|Σ ω_x·dt·supported| / T` for a log-roll (lying on the side, rotating about
the long axis, moving in +x — physically the more plausible roll on a body
this shape). Use the larger of the two as "rotation rate".

| mode | rotation rate |
| --- | --- |
| walk | ≈ 0 (pitch oscillation cancels in a *net* integral) |
| crawl | ≈ 0 |
| roll | 2π per roll: 0.9 rad/s at one roll per 7 s, ~2.7 rad/s at one per 2.3 s |

Noise: the integral is smooth, but the *count of completed rolls* is discrete,
so a policy near the boundary between two and three rolls per episode straddles
a bin. Mitigation for a within-mode grid: bins aligned to whole revolutions, or
a wide-bin rate axis. Failure mode: a policy that rocks forward and back while
supported nets ≈ 0 — correct, that is not rolling.

**A4 — trunk height distribution**: `torso_height_mean` (v3's axis, sd
0.32 mm within walking) plus `torso_height_osc`. Cross-mode span ≈ 0.09 m
against 0.3 mm noise — a spread/noise of ~300. Keep as *the* height-based
mode feature and as the within-walk axis it already is. Within crawl it will
also carry information (belly-drag vs knee-crawl differ by ~2 cm).

**A5 — orientation distribution**: `trunk_lean` (mean gravity_x) and a new
`f_inverted` (fraction of steps with −gravity_z < 0). Walk: lean ≈ 0, inverted
0. Crawl: lean ≈ ±1 (prone or supine), inverted 0. Roll: inverted 0.3–0.5.
Lean failed v3's noise threshold *within walking* (sd/range 14 %) because
walkers barely lean; across modes it is unambiguous. Classifier feature, not a
grid axis.

**A6 — periodicity / spectral energy**: dominant frequency of trunk z or of
whole-body contact, from a (T, N) trace buffer (350 × 1025 floats — trivial).
Cross-mode: walk touchdowns 4.6–5.1 Hz, roll ~0.5 Hz, crawl unknown. v3
measured touchdown-based `step_frequency` at sd 0.27 Hz against a 0.57 Hz
spread, so within-mode it is marginal; the spectral version integrates over
the whole trace and should be steadier, but that is a claim to measure. Worth
building because it is the one axis that says *how* a mode repeats.

**A7 — limb speed** (`joint_speed`, v3's second axis, sd 0.075 rad/s). Keep
as a within-mode axis candidate everywhere; expect it to correlate with
fitness inside every mode as it does in walking (0.93).

**A8 — head involvement**: fraction of steps with `jaw_soft` ground contact.
Roll 0.15–0.3, crawl 0 or ≈ 1 (a chin-drag crawl is a real option on a robot
whose head is 38 % of its mass), walk 0. A component of A1 worth keeping
separate because it is what distinguishes an over-the-head roll from a
shoulder roll — the roulade env needed a head-top latch for exactly that.

**A9 — impact**: p95 trunk |a_z|, and mean non-foot contact force. Used as the
viability cap (§1.4); *also* a candidate within-mode axis for roll ("how hard
does it land"), where it is not a restatement of fitness.

**Axes to drop from consideration**: `stride_length` (|corr| 1.00 with
fitness — the objective in disguise, v3's finding, and worse under modes where
touchdown counting is meaningless); `energy_per_meter` / `cost_of_transport`
across modes (heavy-tailed, and dominated by which mode you are in rather than
how you do it).

### 2.4 The mode classifier

From the median over replicas of the classifier features — and, for clause 3
of P2′, from the same features accumulated per window — rules applied in
this order (thresholds are *initial values*; Stage A′ sets them at the
midpoint between the measured probe clusters and requires a margin of at least
five replica sds on both sides):

| order | mode | rule |
| --- | --- | --- |
| 1 | **roll** | rotation rate ≥ 0.8 rad/s (≈ one revolution per episode) |
| 2 | **hop** | `f_air` ≥ 0.10 — kept as a label so an unexpected bounding gait is filed rather than lost; no seed (§5) |
| 3 | **crawl** | `f_body` ≥ 0.5 |
| 4 | **walk** | `f_body` ≤ 0.1 and `f_air` ≤ 0.05 |
| 5 | **other** | everything viable that matched no rule (e.g. `f_body` in 0.1–0.5: a knee-assisted shuffle, a head-drag walk) |

Two properties the classifier must have, both measured before use:

- **Label stability, across replicas and across windows.** Each replica and
  each window gets its own label; a candidate whose replicas disagree (fewer
  than 7 of 8 agree on the episode label) is rejected as mode-unstable, and a
  replica whose windows disagree fails clause 3. A robot that sometimes walks
  and sometimes crawls is not a robust anything. This is where the chaos
  would otherwise leak into the archive geography.
- **Negative-probe rejection.** v1's 632 elites (292 diving MLPs, 340 CPGs),
  the passive HOME hold, and random MLPs are free negative probes. Every one
  of them must fail P2′. Any that pass are reported by name, not tuned away.

---

## 3. Archive geometry

### 3.1 Options

| geometry | mode separation | within-mode diversity | resolution honesty (v3 lesson) | cost |
| --- | --- | --- | --- | --- |
| **A. one 2D grid** on (f_body, trunk height) | yes | lost — all walkers in ~1 cell | fails: bins sized to the cross-mode span | none |
| **B. 3D/4D grid** (support, height, limb speed, rotation) | yes | some | 20⁴ = 160k cells, almost all empty; the resolvable grid per axis is still set by the noisiest mode | moderate |
| **C. CVT archive** in 4–5D | yes | yes if centroids follow the data | centroids from uniform sampling over ranges land mostly in empty space; needs data-seeded centroids, which need the modes to exist first | pyribs `CVTArchive`, moderate |
| **D. hierarchical**: classifier → per-mode 2D grid, axes chosen per mode | yes, explicit | yes — v3's walk grid survives untouched | each sub-grid gets its own resolvable-resolution check; correlated pairs rejected per mode | a dict of `GridArchive`s, small |

### 3.2 Recommendation: D

Hierarchical, for three reasons that all trace back to v3's measurements:

1. **The walk archive is a known good thing.** 268 verified elites in 109
   resolvable cells on (trunk height × limb speed). A single cross-mode grid
   throws that away; a sub-archive keeps it byte-for-byte.
2. **The resolvable-resolution lesson is per axis pair.** v3's 0.82 axis
   correlation reduced 20×20 to an effective 9×15. Under D each mode runs its
   own Stage A, rejects its own correlated pairs, and reports its own
   resolvable grid. Under A–C one noisy mode sets the resolution for all.
3. **Modes do not compete.** With one objective (displacement) across modes a
   2 m walker would out-rank every 0.4 m crawl wherever their descriptors
   overlap. Separate archives make "best crawl" a meaningful cell.

Mechanics: `archives: dict[mode, GridArchive]`; each candidate is inserted
into the archive its label names; parents are sampled **per mode with a
budget** (e.g. equal share per non-empty mode, so 300 walkers do not supply
95 % of the parents while crawl has five). "Other" is a fifth sub-archive
binned on a generic pair — trunk height × rotation rate — until it holds
enough elites for its own Stage A. The viewer gets a mode tab.

Default within-mode axes for new modes, until each mode's own measurement
exists: trunk height × limb speed, **re-ranged per mode** from its probes. That
pair separated walkers 6-of-6; the expectation is that it also separates
crawls (belly vs knee height; slow vs fast limbs), and the measurement will say
whether it does.

### 3.3 Why not a mode *axis* on one grid

Making mode a categorical axis (5 columns × 20 rows) is the same as D with a
worse interface: it forces one shared within-mode axis on all modes and
hides the per-mode resolution check. D costs a dictionary.

---

## 4. Fitness and gates per mode

### 4.1 Fitness

**Median +x trunk displacement over the 8 permuted replicas**, over the full
7 s, unchanged. Every mode spawns standing at HOME at the same spawn height
(0.125 m), so displacement is measured from the same start and includes the
stand-to-mode transition. Two consequences, both accepted:

- A crawl loses ~1 s to lowering itself. That is honest — it is the cost the
  real robot pays on a policy switch from stand — and window 1 of P2′ is
  exempt for exactly this reason. It also means **every seed must know how to
  get down** (§5.2): a crawl distilled from a prone-only PPO task would spend
  its whole episode toppling from HOME, since no open-loop descent survives
  (Appendix D).
- The 7 s episode is short for slow modes (a 5 cm/s crawl covers 0.35 m).
  Lengthening to 10 s costs 43 % per rollout; keep 7 s for the first archive
  and revisit once crawl speeds are measured (open question 6).

### 4.2 Gates, restated as the insertion rule

A candidate is inserted iff, over 8 world-permuted replicas:

1. every replica is finite, passes sustained progress in windows 2–6, and
   carries one mode label across windows 2–6 (unanimous, or 7-of-8 by
   measured exception);
2. the median p95 |a_z| is under the impact cap (if the cap survives Stage A′);
3. at least 7 of 8 replicas carry the same episode mode label;
4. its median descriptor lands in a cell of that mode's grid whose incumbent
   has lower median fitness (pyribs' rule).

Feasibility and insertion rates are logged **per mode and per operator**, as
v2/v3 log them per operator. A mode whose feasibility rate is zero for ten
iterations is a mode with no seed, and the log should say so rather than let
"coverage" hide it.

### 4.3 Luck-ranking under the new predicate

Fitness is already de-lucked by the median. Survival is not — v3 measured the
winner's curse on the survival column directly (archive robustness decaying
83 % → 75 % as the number of attempts grew), and P2′ is a *less* deterministic
predicate than upright, so the curse will bite harder. Two mitigations, in
order of preference:

- **Re-test incumbents.** Each iteration, re-roll a random tenth of the
  archive (8 permuted replicas each) and fold the results into a per-elite
  running pass count; evict when the running pass rate drops below the gate.
  With ~1000 elites that is ~800 evaluations per iteration, about one extra
  batched rollout on top of the eight the offspring cost — roughly +12 %.
  This is v3's named-but-unbuilt fix and it converts survival from a
  one-shot max into an average, which is what the median did for fitness.
- **Sequential replicas.** Roll once; only a passer earns the other seven.
  Cuts the cost of the 60–70 % of offspring that fail on the first replica.
  Orthogonal to the above; either can ship first.

### 4.4 What changes for the TD3 critic

`ShapedRewardCfg` pays `0.10 + 0.30·upright` per second. Under modes that is
a critic that thinks crawling is bad, and PG variation would gradient-ascend
every crawl toward standing up. Replace with `v_x·dt − λ·|a_z|·dt` (progress
and gentleness, mode-agnostic) and accept that PG variation will mostly refine
walkers until a descriptor- or mode-conditioned critic exists — which is the
same open item v3 named. One critic and one buffer; tagging transitions by
mode is cheap and makes a per-mode critic a later option.

---

## 5. Seeding strategy per mode

### 5.1 Position

**No mode will be discovered from walking seeds.** v2 measured that isotropic
weight mutation at the largest step that keeps a walker upright does not move
its behaviour, and that a survival-gated search lives on a thin feasible
manifold it cannot leave. P2′ widens the feasible set (crawls are in it now)
but does not build a bridge to it from the walkers. Every mode needs a seed
that already does the thing — *from a standing start*.

### 5.2 Per mode

Numbers below are from Appendix A (motor: kt 0.366 N·m/A, R 2.81 Ω, 7.4 V,
firmware current cap 1.75 A → torque cap 0.64 N·m, no-load 20.2 rad/s;
firmware P-gain 200 → 0.575 duty per rad of position error, so under the
genome's ±1 rad action bound the most a policy can *ask for* is ≈ 0.55 N·m at
stall and zero torque at ≈ 11.6 rad/s; robot 0.737 kg from the MJCF
inertials, head assembly 0.280 kg = 38.0 %).

| mode | physical plausibility | seed | risk |
| --- | --- | --- | --- |
| **walk** | proven (2.29 m / 7 s verified) | existing six distilled PPO seeds | none |
| **crawl** | **feasible.** Prone on the battery pack; hip-pitch torque ≈ 0.4 N·m net over a ~5–9 cm lever gives 4–8 N per leg against 7.2 N of weight, so even a 100 %-on-trunk drag at μ = 1 is pushable. Expect 5–15 cm/s. The uncertainty is *style*, not existence: which parts drag, and whether the leg recovery stroke drags backward (hip roll/yaw are ±22° / −25..30°, so legs must lift by knee flexion). | **(a)** a new PPO prone-locomotion task built on the standup/velstand template — spawn mix via `set_random_ground_state` with a **standing bucket** (so the seed learns the descent; the standup env's mix table is the template), forward-velocity tracking, height *ceiling*, no upright term — distilled like the walker; **(b)** open-loop CPG crawls from a prone spawn as the *physics probe* and Stage-A′ measurement set — cheap, low-chaos (CPG noise ≤ 4 mm), and they exist before (a) is trained. Do both; (b) first. | medium: (a) is a new reward design; (b) probes only measure the crawl itself, never the transition, and may distil poorly into the closed-loop MLP. |
| **roll** | **plausible, hardest.** The roulade PPO task exists and the runtime demo reel shows it on the real robot; its cfg records measured over-the-top transit at 3.5–5.5 rad/s and a completed-episode parking basin at z ≈ 0.105, so one supported forward roll *happens* from a standing start. Forward locomotion by rolling needs *chained* rolls: roll → land tucked → roll again, without the full stand the roulade env demanded. Distance per roll ≈ 0.2–0.4 m (a tucked body is roughly a 7 cm-radius wheel: 0.44 m per turn, less for a non-round body), so 3 rolls in 7 s ≈ 0.6–1.2 m. The log-roll (lying on the side, rotating about the long axis, moving +x) is mechanically simpler — 0.28 m per turn on a ~9 cm cross-section, driven by leg and head swings — and nobody has tried it. | **(a)** distil the best roulade checkpoint into the MLP genome (same 61-D obs contract, starts standing — the transition is built in); **(b)** a "chained roll" PPO variant of the roulade env with the landing gate replaced by P2′-style progress and `ROULADE_FORWARD_VEL_RANGE` opened; **(c)** a scripted CPG log-roll probe from a side-lying spawn (hip roll + head yaw sinusoids in quadrature) for Stage A′. | high: roulade needed five runs for one roll and the rise was the hard part; chaining without a rise is a *new* skill. The log-roll probe is the cheap hedge. |
| **hop** | **out of reach — measured, not estimated.** From the deepest stable pose this robot has (the SIT keyframe, trunk z 0.061 m, verified stable for 1 s in the probe), a full-effort simultaneous extension of hip, knee and ankle raises the trunk 2 cm in 0.16 s with a peak vertical speed of **0.19 m/s** (apex of a ballistic flight would be 2 mm) and both feet stay loaded throughout; the knee tops out at 5.9 rad/s with the actuator on its 0.64 N·m current limit for one control step and decaying with the position error afterwards. A 1 cm hop needs 0.44 m/s at takeoff, 2.3× what the best open-loop launch delivers; the torque available to a *policy* is bounded by the firmware gain (≈ 0.55 N·m at a 1 rad error, zero at 11.6 rad/s) and load friction takes ~27 % of that. A closed-loop policy can sequence the joints better than a simultaneous step and add a head throw, but the shortfall is a factor, not a margin. | none. Keep the `hop` label so a bounding gait the search stumbles on is filed, and offer a PPO hop task only as an explicit experiment Alex chooses to fund (open question 1). | — |
| **other** | by definition unknown | none — this is where the search's own inventions land, and where a human looks | — |

### 5.3 The neck

The MLP genome already controls all 14 servos (`ACTION_DIM = 14`); only the
CPG harness pins the neck at HOME (`LEG_JOINT_NAMES`, 10 joints). The job spec
describes the neck as "pinned at HOME through v3" — true for the CPG path,
not for the PGA-ME path that produced the v2/v3 archives. Rolling needs the
chin tuck (the roulade env's head-top latch required neck_pitch −1 /
head_pitch +1), so **the CPG probe genome grows to 43 parameters** (four more
sinusoids). No change to the MLP path.

### 5.4 Open-loop probes cannot start standing (measured)

Every scripted posture change from a standing HOME spawn toppled the robot
before it could be evaluated: a 0.4 s ramp to SIT topples at 0.36 s, 2 s
ramps to any of 30 flat-foot squats topple before the 3 s mark, and the
passive HOME hold itself is down by 1.1–1.3 s (the v2 measurement was
1.34 s). Balance on this robot is closed-loop only, even for a crouch. Two
consequences for the plan:

- **Stage A′ probes spawn in their mode's rest pose** (prone z 0.075, supine
  0.048, seated 0.061, side-lying for the log-roll), using the same
  `set_random_ground_state` machinery the standup env uses. The archive's
  own evaluations spawn standing, as v3 does; the window-1 exemption is what
  makes the two comparable on windows 2–6.
- **Only closed-loop seeds can be evaluated in the archive harness.** A CPG
  crawl is a measurement instrument and a distillation teacher, never an
  archive candidate.

---

## 6. Bootstrap measurement plan (Stage A′)

v3's rule: descriptors chosen by taste fail, descriptors chosen by
spread-to-noise on known-distinct probes work. For modes nobody has produced,
the probes are built by hand and the same discipline runs on them. Everything
below is one job's worth of batched rollouts — ~30 probes × 64 permuted
replicas ≈ 2000 rollouts, minutes on the RTX 3060 — plus the negative set.

### 6.1 Probe set

| set | probes | spawn | source |
| --- | --- | --- | --- |
| walk (positive) | six distilled seeds + five verified v3 elites | standing | exists (`qd-run-archives/j004`) |
| crawl (positive) | 3–6 hand-tuned CPG crawls (hip-pitch/knee sinusoids, left–right antiphase and in-phase; belly-drag and knee-crawl variants); later, the distilled PPO crawl at two speeds | prone (CPG); standing (PPO) | scripted now, PPO later |
| roll (positive) | distilled roulade checkpoint; scripted tuck-and-flop CPG; scripted log-roll CPG | standing; seated; side-lying | one exists, two scripted |
| negatives (must fail) | all 292 v1 PGA-ME divers, all 340 v1 CPG elites, passive HOME hold, 1024 random MLPs, and — deliberately — a scripted *dive* CPG (lean forward, push, lie still) and a scripted *twitcher* (prone, joints oscillating, no progress) | standing; prone for the twitcher | exist / trivial |

A scripted probe that turns out not to move forward is not a failure of the
plan; it is the physics answer for that mode, reported as such (the hop probe
in Appendix D is the first such answer).

### 6.2 Measurements and thresholds

1. **Predicate calibration.** For each positive probe, pass rate of P2′ per
   replica over a grid of (W ∈ {1.5, 2, 3} s, d_min ∈ {0.02, 0.05, 0.10} m);
   for each negative, the same. Choose the (W, d_min) maximising the margin
   between the *worst positive* and the *best negative*, and require positives
   ≥ 0.95 and negatives ≤ 0.05 per replica. Report the whole sweep, as v2/v3
   reported the verification strictness sweep. If no setting clears both
   bars, that is the finding; do not lower them.
2. **Impact cap.** p95 |a_z| per probe. Cap = 1.5 × max over positives; keep
   iff ≥ 90 % of the negatives that *pass* clauses 2–3 (if any) exceed it. If
   nothing degenerate passes in the first place, the cap is unnecessary and
   is reported as such.
3. **Classifier features.** For `f_body`, `f_air`, rotation rate, `f_inverted`,
   head fraction: per-probe median and replica sd, *per window and per
   episode*; between-mode separation in replica sds (expect ≫ 10); thresholds
   at the geometric midpoint between adjacent mode clusters with ≥ 5 sd of
   margin each side. Label agreement per probe over 64 replicas ≥ 0.95 and
   across windows 2–6 ≥ 0.95, else the feature is not usable as a rule.
4. **Within-mode axes**, per mode with ≥ 5 positive probes (walk now, crawl
   once the PPO seeds exist, others when they exist): the unmodified v3
   `select_descriptor` procedure — sd/range < 10 %, spread/sd ≥ 3, mutation
   reach among *P2′-feasible* offspring, |corr| with fitness reported, pair
   correlation reported, resolvable grid computed. Modes with fewer probes
   run on the default pair (trunk height × limb speed, re-ranged) and say so.
5. **Chaos profile per mode.** `check_repeatability` on one probe per mode:
   displacement sd and the divergence profile. Crawls and CPG rolls should be
   *far* less chaotic than walkers (short contact-solve horizon, no marginal
   balance); if a mode is *more* chaotic than walking, its replica count and
   gate rule need their own setting.

### 6.3 What Stage A′ can fail on, and what each failure means

- **A positive probe fails P2′ at every (W, d_min).** The mode does not move
  forward under open-loop control; it needs a PPO seed or is physically out.
  Say which.
- **A negative probe passes P2′.** Either it is a genuine mode (a lunge loop:
  inspect it, it goes in "other") or the predicate has a hole (report the
  policy, do not patch the threshold to exclude it by name).
- **Labels are unstable on a positive probe.** The mode boundary runs through
  a real behaviour; move the threshold, or merge the modes, and re-measure.

---

## 7. Recommendation and open questions

### 7.1 The one design

- **Viability (P2′):** sustained progress (W = 2 s, stride 1 s,
  d_min ≈ 0.05 m, window 1 exempt; values from Stage A′), one mode label
  across windows 2–6, finite state, impact cap on median p95 |a_z| if it
  discriminates; unanimous over 8 world-permuted replicas unless the robust
  probes measure below 0.95 per replica, then 7-of-8 with the reason written
  down.
- **Mode label:** `f_body`, `f_air`, supported rotation rate, in the order
  roll → hop → crawl → walk → other, thresholds measured, ≥ 7-of-8 replica
  agreement and per-window constancy required.
- **Archive:** one 20×20 `GridArchive` per mode; walk keeps v3's axes and
  ranges; others on trunk height × limb speed re-ranged until their own
  Stage A; "other" on trunk height × rotation rate; per-mode parent budget;
  per-mode resolvable-resolution reporting; incumbents re-tested (a tenth per
  iteration) with a running pass rate.
- **Fitness:** median +x displacement over the replicas, per-mode archives so
  modes never compete; critic reward becomes progress minus impact.
- **Seeding:** walk from the existing PPO seeds; crawl from CPG probes (prone
  spawn) now and a prone-locomotion PPO task *with standing spawns* next; roll
  from a distilled roulade checkpoint plus a log-roll CPG probe, with a
  chained-roll PPO variant if a single roll stalls; **no hop seed** — the
  label stays, the physics says the archive would be empty.
- **Genome:** unchanged 61→64→64→14 MLP; the CPG probe genome grows to 43
  parameters to free the neck.

**Why this one.** It is the smallest change to v2/v3 that keeps their honesty
argument intact: the gate stays a predicate on a *behaviour that must repeat
and stay what it is*, which is the only kind of predicate that excludes divers
and late fallers and admits crawlers at the same time, and it stays evaluated
the way v3 learned to evaluate — medians over world-permuted replicas,
thresholds from measured probe clusters, sweeps reported whole. The hierarchy
is what keeps v3's resolvable-resolution lesson true per mode instead of
letting the noisiest mode set the grid for all. And the seeding takes v2's
measurement seriously: the search will not build a bridge to a new mode, so
each mode gets its own, with the cheap open-loop probe run before the
expensive PPO task so that a physically implausible mode is found out for the
price of a batch rather than a training run — which is exactly what happened
to hop while writing this.

### 7.2 Open questions for Alex, ranked

1. **Hop: drop it, or fund one PPO attempt to prove the measurement wrong?**
   The scripted launch (Appendix A, D) delivers 0.19 m/s against the 0.44 m/s
   a 1 cm hop needs, and the firmware gain caps what a policy can ask for at
   ≈ 0.55 N·m. My recommendation is drop, and spend the budget on roll; a
   PPO hop task is a bounded experiment (~one training run) if you want the
   closed-loop answer on record.
2. **Every mode spawns standing, transition included?** Recommended (it is
   what a policy switch on the robot does), but it costs a crawl ~1 s of its
   7 s, makes window 1 of the predicate exempt, and obliges every seed task to
   include standing spawns. The alternative is a per-mode spawn pose, which
   breaks displacement comparability across modes and adds a spawn-pose
   design per mode.
3. **Two new PPO tasks (prone crawl with standing spawns, chained roll) — yes
   or CPG-only?** PPO seeds are what made walking work; CPG probes are cheap
   and low-chaos but cannot be archive candidates at all (they cannot start
   standing). My recommendation is CPG probes first for crawl and log-roll,
   then both PPO tasks, roll first because its seed already half-exists.
4. **Shell contact model before crawl/roll.** The shells are unnamed in the
   export, so `FULL_COLLISION`'s intended frictionless `condim=1` never
   applies and every shell runs at MuJoCo's default μ = 1; nothing randomises
   it (`foot_friction` DR is feet-only); and the upper legs and trunk side
   shells have no ground geom at all. A crawl trained on μ = 1 PLA is
   pessimistic on drag and optimistic on knee traction. Recommended: name the
   shell geoms, set a measured shell μ with DR, and add the missing geoms
   *before* any crawl/roll seed is trained; this is a sim2real footgun of
   exactly the kind AGENTS.md lists.
5. **Impact cap as a viability clause, or impact as a descriptor axis?** A cap
   excludes lunge loops; an axis keeps them visible. Recommended: cap, with
   the sweep reported so the choice is revisable.
6. **Episode length 7 s or 10 s for slow modes?** Recommended: 7 s for the
   first archive, revisit with measured crawl speeds.
7. **Incumbent re-testing in the first implementation job or later?** It is
   the fix v3 named for the survival winner's curse and it will matter more
   under P2′. Recommended: in — it is +12 % cost and a running counter.

---

## Appendix A — can this robot hop? Arithmetic, then measurement

**Motor (BAM `xl330/m6.json`, harness at 7.4 V, 4-step command lag).**
kt = 0.366 N·m/A, R = 2.81 Ω. Voltage-limited stall torque kt·V/R =
0.964 N·m (the MJCF `forcerange` ±0.96). BAM's XL330 actuator carries a
firmware current limit of 1.75 A by default (`bam/dynamixel/actuator.py`;
the commented-out `max_current` line in `microduck_constants.py` does not
disable it), so the binding torque cap is kt·I_max = **0.64 N·m**; the probe
below measured `qfrc_actuator` saturating at exactly 0.641 N·m. No-load speed
V/kt = **20.2 rad/s**. Load-dependent friction takes ≈ 0.267·τ_motor + 0.005
off the top (m6 `load_friction_motor`). Reflected rotor inertia (`armature`)
0.0018 kg·m² dominates any link.

**Firmware loop.** Duty = (4096/2π)/(256·885) · kp · Δq = 0.00288 · kp · Δq;
at the robot's kp_fw = 200 that is **0.575 duty per rad**, saturating only at
|Δq| = 1.74 rad. The genome's tanh output bounds the commanded position to
±1 rad around HOME, so the most torque a *policy* can request is
0.964 · 0.575 · (1 rad) ≈ **0.55 N·m at stall**, falling to zero at
0.575·7.4/0.366 ≈ **11.6 rad/s** — before the 27 % load-friction loss. This
is the number that matters for explosive motion; the 0.64 N·m current cap is
only reachable with a > 1.1 rad error.

**Robot:** 0.737 kg from the MJCF inertials (head assembly 0.280 kg, 38.0 %);
weight 7.23 N. Thigh ≈ 4.2 cm, shank ≈ 4.2 cm (plus a 2.6 cm lateral offset
along the ankle axis), ankle-to-sole ≈ 3.9 cm; standing trunk z 0.115 m
(measured), SIT keyframe 0.061 m. Usable vertical push-off stroke d ≈ 0.04 m
with hip, knee and ankle all extending.

**What a hop would need** (h → takeoff speed v = √(2gh); ground force
F = mg·(1 + h/d)):

| h | v | F (total) | per-leg knee torque at stroke start (arm ≈ 2.4 cm) | joint speed needed, three joints sharing | torque a policy can request at that speed |
| --- | --- | --- | --- | --- | --- |
| 1 cm | 0.44 m/s | 9.0 N | 0.11 N·m | ≈ 9 rad/s | 0.55·(1 − 9/11.6) ≈ 0.12 N·m, ≈ 0.09 after friction — **below need** |
| 2 cm | 0.63 m/s | 10.8 N | 0.13 N·m | ≈ 13 rad/s | 0 — past the policy's no-load speed |
| 5 cm | 0.99 m/s | 16.3 N | 0.20 N·m | ≈ 20 rad/s | 0 — at the motor's no-load speed |

Joint speed from the two-link leg: dL/dθ = ab·sinθ / L ≈ 0.017 m/rad at 45°
knee flexion, 0.026 at 70°; a single joint would need 36 rad/s for h = 2 cm,
which is why the estimate lets all three pitch joints share the stroke.

**What the simulator says** (probe 4, Appendix D; median of 2 replicas, from
the SIT keyframe after a 1 s hold that stayed at up = 1.000):

| snap target | trunk rise | peak trunk v_z | apex if ballistic | feet unloaded before the topple | peak knee speed | peak knee torque |
| --- | --- | --- | --- | --- | --- | --- |
| straight legs | +2.0 cm in 0.16 s | **0.19 m/s** | 1.8 mm | never | 5.8 rad/s | 0.641 N·m (1 step, then decays with the error) |
| HOME | +1.6 cm | 0.16 m/s | 1.3 mm | never | 5.9 rad/s | 0.640 N·m |
| straight + ankle push (both signs) | +1.0 / +2.2 cm | 0.18 / 0.30 m/s | 1.6 / 4.5 mm | never | 5.5 / 6.0 rad/s | 0.641 N·m |
| hip-forward / hip-back variants | +1.4 / +2.3 cm | 0.15 / 0.20 m/s | — | never | ≈ 6 rad/s | 0.63–0.64 N·m |

Every variant then topples (trunk up-cosine goes negative within 0.3 s),
because the launch is open-loop; that is the balance problem, separate from
the strength one. The best launch reaches 0.30 m/s of trunk speed, 0.44 m/s
is a 1 cm hop, and none of it produces flight. **Conclusion: hopping is out
of the seeded design.** A closed-loop policy could sequence the three joints
and throw the head, but the gap is a factor of 1.5–2.3 on takeoff speed
against a torque budget that is already saturated, and the airtime for a
1 cm hop (90 ms) is barely longer than the 80 ms command lag.

## Appendix B — sim-state inventory for the harness

Everything below is on `robot.data` (mjlab `Entity`) or on the scene, at the
50 Hz control step, for all N worlds, device-side, no host sync:

- base: `root_link_pos_w`, `root_link_quat_w`, `root_link_lin_vel_w`,
  `root_link_ang_vel_b`, `projected_gravity_b`
- joints: `joint_pos`, `joint_vel`, `qfrc_actuator` (14 servo joints on the
  all-collisions model; use the `_servo_joint_ids` convention if a backlash
  twin is ever evaluated)
- per-body: `body_link_pos_w`, `body_link_quat_w` (used by the roulade head-top
  test)
- contact sensors (`ContactSensorCfg`): per-geom or per-body `found`, `force`
  (netforce), air time; matching by geom name, body name, or subtree, against
  the terrain or the robot itself — this is how `feet_ground_contact`,
  `self_collision`, `head_ground_contact` and `robot_ground_contact` are built
- MJCF sensors via `sensordata`: `imu_accel` (accelerometer),
  `imu_ang_vel`, `imu_lin_vel`, `root_angmom` (subtree angular momentum)

Ground-colliding bodies on `robot_allcollisions.xml` as exported (11 geoms
with `contype`/`conaffinity` = 1): `trunk_base` (battery-pack mesh only),
`hip_l`, `hip_l_2`, `leg`, `leg_2`, `ankle_left`, `ankle_right` (soles),
`jaw_soft` (three head-shell meshes). Not colliding with the ground: upper
legs, trunk side shells, neck. Only the two soles are *named*
(`left_foot_collision`, `right_foot_collision`); they get condim 3, μ = 1.0,
priority 1 from `FULL_COLLISION`, and every other ground contact runs on
MuJoCo's defaults (condim 3, μ = 1, priority 0) because the cfg's override
patterns match names.

Additions the design needs, all of the same shape as the existing extras:
`body_contact` (N, n_bodies), `foot_force` (N, 2), `root_quat_w` (N, 4),
`trunk_acc` (N, 3), `root_angmom` (N, 3), a (T, N) trunk-z trace for the
spectral axis, and per-window (6, N) accumulators of +x displacement and of
the classifier features for P2′.

## Appendix C — the chaos numbers this design is built against

| measurement | value | where |
| --- | --- | --- |
| identical-genome displacement sd, 0.4 m/s walker, 256 replicas | 0.605 m (5–95 %: −0.05 .. +1.94 m) | v2, `check_repeatability` |
| its survival rate | 98.8 % | same |
| trunk-x divergence between identical worlds | 1 mm at 0.04 s, 100 mm at 0.52 s | same |
| open-loop CPG re-evaluation noise | ≤ 4.4 mm | v1, `bench` |
| world-index bias: same slot vs different worlds | 0.071 m vs 0.469 m | v3 |
| descriptor replica sd: trunk height, limb speed | 0.32 mm, 0.075 rad/s | v3 Stage A |
| duty-factor replica sd | 0.014 | v2 |
| v3 axis correlation → resolvable grid | 0.82 → 9×15 of 20×20 | v3 verify |
| survival winner's curse over a run | 83 % → 75 % robustness | v3 |
| P(pass unanimous-8) for an 88 %-robust policy | 0.36 | v3 arithmetic |
| roulade over-the-top transit speed | 3.5–5.5 rad/s | roulade cfg, run-3 eval |
| passive HOME hold topples at | 1.34 s (v2); 1.1–1.3 s at the 60° tilt threshold in this draft's probes | v2 `check_harness`; Appendix D |

## Appendix D — throwaway probes run for this draft

All on the v3 low-level harness (`qd/evaluate.py` from branch
`qd-walking-v3`, staged in a scratch directory), `robot_allcollisions.xml`,
deterministic BAM m6 at 7.4 V, 4-step lag, 50 Hz, `full_gait_stats` on,
early-fall stop disabled. Scripts live in the job scratchpad and are not
committed.

1. **Model probe** (CPU MuJoCo): total mass 0.737 kg; head subtree 0.280 kg;
   16 bodies, 14 actuators with `forcerange` ±0.96; 11 ground-contact geoms
   (list in Appendix B); standing trunk z 0.115.
2. **Scripted crouch-and-snap from standing** (8 variants × 4 replicas): every
   crouch variant toppled at 0.36–0.42 s, before the snap; the passive HOME
   hold toppled at 1.08 s (60° threshold). No variant was evaluable as a
   jump.
3. **Slow flat-foot squats from standing** (2 s ramp, 1 s hold; 12 then 30
   hip/knee combinations, ankle = −(hip + knee)): 0 of 42 held to the 3 s
   mark. Combined with (2): no open-loop posture change from standing
   survives on this robot; balance is closed-loop only.
4. **Seated launch** (spawn in the SIT keyframe at z 0.062, hold 1 s, snap to
   8 targets, 2 replicas each): SIT held at up = 1.000 for the full second;
   results in Appendix A. No flight in any variant; peak trunk v_z 0.30 m/s
   (ankle-push variant), 0.19 m/s (straight legs); actuator saturates at
   0.641 N·m; knee ≤ 6 rad/s.

**Where this draft and the earlier hand-off draft disagreed.** The earlier
draft (same job, previous session) reasoned that a 1–2 cm hop was inside the
torque–speed envelope. Two things changed the conclusion: the firmware gain
arithmetic (0.575 duty/rad at kp 200, not 1.15 — so a ±1 rad policy action
can request at most ≈ 0.55 N·m, and the current cap is rarely the binding
limit), and the probe, which measured 0.19–0.30 m/s where 0.44 m/s is the
floor. The earlier draft also let Stage A′ CPG probes spawn standing and
treated them as potential archive seeds; the toppling measurements moved
probes to rest-pose spawns and made "seeds must include the descent" a
requirement of the crawl PPO task. Finally, the temporal-consistency clause
(P2′, clause 3) was added after noticing that a windowed progress rule alone
admits a walker that falls in the last half-second. Everything else —
predicate family, classifier, hierarchy, fitness, roll and crawl seeding —
is the earlier draft's analysis, kept because it held up against the code
and the numbers.
