# j007 checkpoint 1 — shell contact fix + Stage A′

Branch `qd-modes-v4`, PR #5 (base `qd-walking-v3`). Design doc merged in from
`qd-modes-design` so it travels with its implementation.

**Headline: the shell fix landed and matters more than expected; Stage A′ did
NOT clear its pre-registered bars, and the reason is measured and structural.
Two decisions belong to HQ/Alex before checkpoint 2 — §6.**

---

## 1. Q4 — the shell contact fix

### What was wrong

`FULL_COLLISION` addresses geoms by **name**; `onshape-to-robot` names exactly
two, the soles. So its `condim=1` rule for the shells has never matched a geom.

| | legacy | fixed |
| --- | --- | --- |
| geoms that can touch the ground | 10 | **14** |
| of those, named | 2 | **14** |
| shell `condim` / `priority` / `mu` | 3 / 0 / **1.0** | 3 / **1** / **0.4** |

Two further silent consequences: `disable_other_geoms` collects non-matching
names into a **set**, so ~70 anonymous geoms collapse to one `''` entry and at
most one is ever disabled (the shells were live by accident); and `priority 0`
makes MuJoCo mix the pair elementwise, so a shell at `mu = 0.4` against a
`mu = 1` floor still slides at 1.0. The export also gives **no ground geom at
all** to the upper legs or the trunk side shells — what a prone or side-lying
robot rests on.

Fixed in `robot/shell_contacts.py`, applied in every `get_*_spec`, on the
`MjSpec` rather than in the generated MJCF so a re-export cannot undo it.

### Before/after (`qd.check_shell_contacts`, 64 worlds, 3 s settle)

| pose | legacy settled z | fixed settled z | sink | carried by (fixed) |
| --- | --- | --- | --- | --- |
| stand (topples, as it must) | 0.0474 | 0.0438 | +3.6 mm | soles, jaw |
| prone | 0.0352 | 0.0353 | −0.1 mm | hips, soles, jaw |
| supine | 0.0471 | 0.0476 | −0.5 mm | **trunk side shells**, top head shell, soles |
| **side** | 0.0408 | **0.0669** | **−26.1 mm** | **upper leg**, jaw |
| sit | 0.0651 | 0.0643 | +0.8 mm | soles, shanks, **upper legs** |

A side-lying robot used to settle **26.1 mm too low, resting on a thigh that
did not exist**. On the legacy model the per-geom contact sensor cannot even be
*built* — nine of eleven ground geoms have no name — so "which part is touching
the floor" was not an observable quantity.

### How much the friction number matters

`crawl_chin_drag_tuned`, 64 replicas, re-run at four shell frictions:

| shell mu | displacement / 7 s | worst 2 s window | p95 \|a_z\| |
| --- | --- | --- | --- |
| 0.25 (DR floor) | +0.749 m | 0.215 m | 11.4 |
| **0.40 (nominal)** | **+0.725 m** | **0.214 m** | 11.3 |
| 0.65 (DR ceiling) | +0.582 m | 0.167 m | 12.1 |
| **1.00 (legacy default)** | **+0.293 m** | **0.084 m** | 12.7 |

Within the DR range the crawl varies ~25 %, which is what makes shipping a
literature value tolerable. At the **legacy** `mu = 1.0` the same crawl travels
2.5× less and its worst window falls to 0.084 m — **below the `d_min` of
0.10 m the sweep would otherwise have chosen**. On the model v1–v3 ran, the
best scripted crawl is not viable. The contact fix is the difference between
crawl existing and not existing.

**`SHELL_FRICTION = 0.4` is a literature value for PLA, not a hardware
measurement**, and says so in its own docstring. This is the one number in
checkpoint 1 that is chosen rather than measured.

### The cost, stated

Adding the thigh and side-shell geoms makes them touchable **by each other**,
and six env cfgs charge a `self_collision` penalty. Measured over random joint
configurations at several distances from HOME:

| joint sampling | legacy mean | fixed mean |
| --- | --- | --- |
| HOME | 0.000 | 0.000 |
| HOME + N(0, 0.1) | 0.005 | 0.007 |
| HOME + N(0, 0.3) | 0.291 | 0.523 |
| HOME + N(0, 0.6) | 0.747 | 1.732 |
| uniform over limits | 0.885 | 2.385 |

**No constant tax on walking.** The increase is in folded poses and the pairs
are physically real (head shell against trunk side shell — the head is 38 % of
body mass). Tasks that deliberately fold (standup, roulade, a crawl task)
should be re-baselined rather than surprised.

---

## 2. Two corrections to the accepted design

Neither is a preference; each was found by a probe behaving absurdly.

**(a) The impact clause differences world-frame `v_z`, not the accelerometer.**
The draft said to read `imu_accel` "already wired for
`trunk_vertical_accel_penalty`". That penalty does not use it — it differences
`root_link_lin_vel_w[:, 2]` — and it is right not to: an accelerometer measures
specific force and reads 9.81 m/s² on a robot lying perfectly still, so a cap
on it would charge a resting crawl more than a gentle landing.

**(b) Rotation is accumulated in the WORLD frame about horizontal axes.** The
draft's `root_link_ang_vel_b[:, 1]` is right for a somersault by an upright
robot and wrong for anything lying down: in the side-lying spawn the body's
lateral axis points straight *up*. Measured before the fix — a probe
**pirouetting on the floor** scored **1.69 rad/s and classified as a roll**
while covering **−3 cm**. World z is now measured separately
(`rotation_rate_yaw`) rather than mistaken for locomotion.

---

## 3. Stage A′ — what the probe set measured

Full artefact: `qd/measurements/stage_a_prime.md` (+ `.json`).

### Probes that do not move (reported, not tuned away)

Every hand-written parameterisation crawled backwards and no roll probe
rotated, so the whole (base × direction × frequency × amplitude) grid was swept
— 396 variants over six batched rollouts.

| probe | intended | median dx | rotation |
| --- | --- | --- | --- |
| `crawl_belly_push` | crawl | −0.132 m | 0.003 rad/s |
| `crawl_belly_push_inphase` | crawl | −0.064 m | 0.016 rad/s |
| `crawl_knee` | crawl | −0.054 m | 0.032 rad/s |
| `crawl_chin_drag` | crawl | −0.240 m | 0.016 rad/s |
| `log_roll` | roll | −0.063 m | 0.190 rad/s |
| `log_roll_fast` | roll | −0.053 m | 0.174 rad/s |
| `tuck_and_flop` | roll | +0.052 m | 0.104 rad/s |

**Roll does not roll open-loop.** Best supported world-horizontal rotation over
129 swept roll variants: **0.22 rad/s** against the 0.8 rad/s rule. The roll
threshold is therefore **UNCALIBRATED** — there is no roll cluster — and stays
at the draft's arithmetic until a distilled roulade checkpoint gives it a
member. This is §6.3's "the mode does not move forward under open-loop control;
it needs a PPO seed or is physically out", and the design's Q3 already routes
roll through the roulade distillation.

### The classifier threshold that cut through a real cluster

`hop_air_min` 0.10 → **0.16**. The draft's initial value runs straight through
the measured crawl cluster (per-window `f_air` reaches 0.130), and two genuine
crawls flipped crawl↔hop between windows, failing clause 3 in 19–29 % of
replicas. This is exactly §6.3's prescribed remedy. Reported as **one-sided**:
there is no hop cluster to bound it from above, because hop is dropped (Q1) and
Appendix A measured the physics out of reach. Raising it admits no negative.

`crawl_body_min` unchanged at 0.5 (clearance, not a cut).
The walk|crawl `f_body` boundary has **3.6 sd below / 1.4 sd above** — the
crawl side is under the 5-sd bar, driven by the chin-drag crawls that ride
partly on their feet.

### Chaos per mode — the number that explains everything below

| mode | probes | median dx | mean replica sd |
| --- | --- | --- | --- |
| crawl | 9 | +0.384 m | **0.0028 m** |
| roll | 3 | −0.053 m | 0.0026 m |
| **walk** | 11 | +1.430 m | **0.5591 m** |

Walking is **200× more chaotic** than the open-loop crawls, reproducing v2's
0.605 m. §6.2-5 asked whether any mode needs its own replica count; the answer
is yes, and it is walk.

---

## 4. The honest failure: no setting cleared both bars

Pre-registered: positives ≥ 0.95 and negatives ≤ 0.05 **per replica**.
Measured, over 16 calibration positives and 1661 negatives (292 v1 MLP divers,
340 v1 CPG elites, 1024 random MLPs, 5 scripted degenerates):

| setting | worst positive | best negative | positives ≥ .95 | negatives ≤ .05 |
| --- | --- | --- | --- | --- |
| W=1.5, d=0.02 | 0.531 | 1.000 | 5/16 | 1621/1661 |
| W=2, d=0.02 | 0.719 | 1.000 | 5/16 | 1564/1661 |
| W=2, d=0.05 | 0.227 | 1.000 | 5/16 | 1637/1661 |
| W=2, d=0.10 | 0.000 | 0.867 | 5/16 | 1657/1661 |
| W=3, d=0.02 | 0.781 | 1.000 | 5/16 | 1476/1661 |
| W=3, d=0.10 | 0.133 | 1.000 | 5/16 | 1635/1661 |

**The five that reach ≥0.95 at every setting are the five scripted crawls
(1.000 each). The eleven that never do are the eleven walk positives.**

Clause breakdown at W=2, d_min=0.05 — all failures are the **progress** clause,
none are `finite`:

| probe | viable | progress | label | median dx |
| --- | --- | --- | --- | --- |
| walk_seed_0 | 0.227 | 0.227 | 1.000 | 0.190 m |
| walk_seed_1..5 | 0.71–0.81 | 0.71–0.81 | 1.000 | 0.68–1.43 m |
| v3_elite_0..4 | 0.71–0.80 | 0.78–0.81 | 0.87–0.98 | 1.72–2.09 m |
| the 5 tuned crawls | **1.000** | **1.000** | **1.000** | 0.38–0.72 m |

**A Microduck walker passes P2′ about 0.78 of the time per replica, because a
Microduck walker falls.** v3 independently measured mean per-elite survival at
**0.883** under the *upright* gate; P2′ is slightly stricter because it also
demands sustained progress. Loosening `d_min` to 0.02 raises walk only to
0.72–0.84 — the grid does not control what they fail on.

**The ≥0.95-per-replica bar is achievable by a near-deterministic open-loop
crawl and unachievable by any closed-loop walker on this simulator. It is not a
property of P2′.** No threshold was lowered to hide this.

---

## 5. The finding that reframes the negatives

Of 1661 negatives, 24 exceed 0.05 per replica at W=2/d=0.05, and five pass
unanimously over 8 replicas. Re-run at **128 world-permuted replicas**:

| genome | viable | label | agreement | window constancy | median dx | sd | f_body | p95 \|a_z\| |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **v1_cpg_175** | **1.000** | crawl | 1.000 | 1.000 | **+0.543 m** | 0.0018 m | 0.789 | 6.1 |
| **v1_cpg_277** | **1.000** | crawl | 1.000 | 1.000 | **+0.359 m** | 0.0057 m | 0.914 | 3.1 |
| v1_cpg_169 | 0.859 | crawl | 1.000 | 1.000 | +0.306 m | 0.0159 m | 0.923 | 4.0 |
| v1_cpg_4 | 0.594 | crawl | 1.000 | 1.000 | +0.447 m | 0.0193 m | 0.874 | 5.9 |
| v1_cpg_6 | 0.461 | crawl | 1.000 | 1.000 | +0.320 m | 0.0131 m | 0.871 | 4.2 |

**These are not a hole in the predicate. They are crawls** — and they have been
sitting in v1's MAP-Elites archive since j002, filed as junk because the upright
gate called them fallen. They spawn **standing at HOME**, get down, and travel
0.31–0.54 m on their shells with gentle impacts, which is the same range as the
hand-tuned scripted crawls and better than three of them.

Two consequences:

* **Q3's crawl seeding may not need a PPO task at all** for a first seed. These
  are 31-D CPG genomes, so they need distilling into the 9038-D MLP genome the
  archive uses — the same behaviour-cloning path `qd.seed` already runs for the
  PPO walker — but unlike the scripted probes they start standing, so the
  distillation is well-posed.
* The 8-replica estimate said all five were at 1.000; at 128 replicas three
  dropped to 0.46–0.86. Small-sample optimism, exactly the v3 lesson, visible
  again.

---

## 6. What I need from HQ before checkpoint 2

### Decision 1 — the aggregation rule

The insertion gate is what the archive actually applies. Measured admission
rates at W=2 (binomial from the measured per-replica rates; "crawl" is the five
tuned probes, all at 1.000):

| d_min | rule | walk admitted | crawl admitted | negatives admitted |
| --- | --- | --- | --- | --- |
| 0.05 | 8 of 8 (unanimous) | **0.126** | 1.000 | 5.3 of 1661 |
| 0.05 | 7 of 8 | **0.400** | 1.000 | 5.7 of 1661 |
| 0.05 | 6 of 8 | 0.671 | 1.000 | 5.9 of 1661 |
| 0.05 | **5 of 8** | **0.831** | 1.000 | 6.1 of 1661 |
| 0.10 | 7 of 8 | 0.312 | 1.000 | 0.7 of 1661 |
| 0.02 | 5 of 8 | 0.934 | 1.000 | 29.5 of 1661 |

Note the shape: **the aggregation threshold barely moves the negatives** (5.3 →
6.1 across the whole column) because their per-replica rates are bimodal —
either ~0 or 1.0 — and five of the six "admitted" are the real crawls above. It
moves *walk* from 0.13 to 0.83.

**My recommendation: W = 2 s, stride 1 s, d_min = 0.05 m, and insertion at
5-of-8** rather than the design's unanimous-8, invoking §1.4's escape hatch
("if Stage A′ measures the pass rate of the known robust probes below 0.95 per
replica, the rule becomes k-of-8, and the reason is written next to the
number") with the reason being the measured 0.78. Unanimous-8 would admit 13 %
of *known-good* walkers and the walk sub-archive would be a fraction of v3's.

The risk is v2's: a weaker gate readmits marginal policies via the winner's
curse over ~51,000 offspring. **v4 already has the designed answer built** —
incumbent re-testing with eviction (`qd/hierarchy.py`, Q7) — which converts
survival from a one-shot maximum into a running average. I would set the
eviction bar from the measured walker rate (evict below ~0.60 running pass
rate: far under a real walker's 0.78, far above the ~0.125 of the marginal
junk) rather than at the 0.875 I currently have hardcoded.

### Decision 2 — the walk-regression criterion is measuring a different question

The acceptance criterion says the v4 walk sub-archive must hold ≥80 resolvable
cells of verified elites (v3 achieved 109). But **v3's 109 was verified under
the upright predicate, and v3's own top elites pass P2′ at 7-of-8 only 40 % of
the time.** Verifying v4's walk archive under P2′ and comparing the count to
v3's upright-verified count compares two different questions — the exact
"criterion and gate are not the same threshold" mistake v3's README documents
having made.

I see three options and would like a ruling:

1. **Verify v4's walk sub-archive under P2′ and re-baseline** by re-verifying
   v3's final archive under P2′ too, so the comparison is like-for-like. Costs
   one extra verification pass (~356 elites × 8 replicas, ~20 min).
2. Keep the v3 number as the bar and accept that the criterion is much harder
   than it reads.
3. Verify walk under the *upright* predicate (v3's rule) and the other modes
   under P2′, so "walk did not regress" is answered in v3's own terms.

**My recommendation is (1)** — it is cheap, it is the only comparison that means
anything, and it produces the number the README should quote.

### Decision 3 (minor, I have a clear recommendation) — keep the impact cap?

Measured at W=2, cap = 1.5 x the worst intended-mode positive = **17.0 m/s^2**:

| d_min | degenerates clearing clauses 2-3 | excluded by the cap | design's 90 % rule |
| --- | --- | --- | --- |
| 0.02 | 97 | 1 (`thrash`) | 1.0 % -> DROP |
| 0.05 | 24 | 1 (`thrash`) | 4.2 % -> DROP |
| 0.10 | 4 | 1 (`thrash`) | 25.0 % -> DROP |

By the design's own rule (keep the cap iff it excludes >= 90 % of the
degenerates that clear clauses 2-3) the cap is **dropped**. But look at the
denominator: every other degenerate that clears clauses 2-3 is *gentle*
(p95 \|a_z\| <= 14.6), so clauses 2-3 already do the anti-violence work on this
population, and the 90 % rule was written expecting many violent degenerates
when the measured count is **one**.

That one is real: `thrash` sustains +0.209 m per window and p95 \|a_z\| = 28.2,
and at 5-of-8 the cap is the only thing that excludes it (0.99 of the 6.14
expected admissions at d_min=0.05, all of it `thrash`). The cap costs the
positives nothing — the worst is 11.3 against a cap of 17.0, a 1.5x margin by
construction.

**Recommendation: keep the cap at 17.0 m/s^2**, and record that it fails the
design's 90 % rule because that rule's denominator turned out to be ~1. A
clause that costs nothing and closes a known hole is worth more than a
percentage calibrated against an imagined population. Flagging it because it
is a deviation from the accepted design, not because I am unsure.

### Not a question, just a flag

`SHELL_FRICTION = 0.4` is a literature value. Everything crawl-related inherits
its uncertainty. One tribometer measurement on a real shell would retire it.

---

## Status

* Shell fix: **landed**, with before/after evidence. ✅
* Stage A′ §6.2 measurements 1–5: **executed and reported whole**. ✅
* Predicate calibrated: **partially** — cleanly on the crawl cluster, and the
  walk positives' failure to reach the pre-registered bar is reported as a
  measured structural fact rather than resolved by lowering it. ⚠️
* Impact cap: measured. It fails the design's 90 % rule (it excludes exactly
  one degenerate at every setting) but that one is the only violent degenerate
  measured, and the cap costs the positives nothing — see Decision 3. ⚠️
* Roll threshold: **uncalibrated**, no probe rotates. Deferred to checkpoint 3's
  roulade distillation, as Q3 plans.
* 317 tests green; changed files ruff-clean; logs in `qd-run-archives/j007/`.

**Paused for sign-off on Decision 1 and Decision 2.**
