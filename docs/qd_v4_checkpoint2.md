# j007 checkpoint 2 — v4 infrastructure, verified on knowns

Branch `qd-modes-v4`, PR #5. HQ's checkpoint-1 ruling is applied; the amended
acceptance criteria are in the job file, citing that ruling.

**Headline: every checkpoint-2 criterion passes, and the walk sub-archive
already beats the re-baselined bar after eight iterations at half batch.**
One further change to the accepted design is flagged in §2 rather than made
silently. Budget projection and the checkpoint-3 plan are in §6.

---

## 1. The ruling, as shipped

| knob | value | where the number came from |
| --- | --- | --- |
| window / stride | 2 s / 1 s | Stage A' sweep |
| `d_min` | 0.05 m | Stage A' sweep; 24 of 1661 negatives exceed 0.05/replica |
| impact cap | 17.0 m/s² | 1.5× the worst intended-mode positive (11.3) |
| `hop_air_min` | 0.16 | above every measured crawl/walk window + 5 replica sds |
| insertion | **5-of-8** viable | a real walker measures 0.78/replica; unanimous-8 admits 13 % of them |
| label agreement | 7-of-8 | every probe measured 1.000 except the deliberately borderline |
| retest eviction | **0.60** | walker 0.78, marginal junk 0.125 — 0.60 is the empty middle |
| verification | **5-of-8**, sweep always printed | insertion evidence must match verification demands (j004) |

Each of these sits in the docstring of the thing it configures, with its
measurement, so the next person does not have to find this document.

---

## 2. One further change, flagged rather than silent

**The label-constancy clause is now asked of the *viable* replicas, not of all
of them.** An early fall is both non-viable (no progress) *and* label-flipped
(the robot spends the episode on its shell) — because the fall is what flips
it. The design's literal reading charges one event to two clauses.

Measured on v3's own elites:

| | label over all replicas | label over viable replicas |
| --- | --- | --- |
| v3_elite_0 admitted | 0.595 | **0.905** |
| v3_elite_2 admitted | 0.614 | **0.819** |
| mean over 11 walk positives | 0.744 | **0.831** |
| negatives expected admitted (of 1656) | 0.15 | **0.15** |

Nine points of walk admission recovered, negatives unchanged. The clause keeps
its teeth where it was aimed: a candidate that walks viably in five replicas
and crawls viably in three still fails, because that disagreement is among
rollouts that all succeeded. Both readings are computed and reported
(`qd.check_knowns`), and the strict one is a flag away.

---

## 3. Criterion 6a — every known negative fails

`qd.check_knowns`, computed from the Stage A' feature caches (no GPU, re-runs
in seconds whenever a threshold moves).

| | |
| --- | --- |
| known negatives | **1656** (292 v1 PGA-ME divers, 335 v1 CPG elites, 1024 random MLPs, 5 scripted degenerates) |
| worst per-replica pass rate | **0.250** |
| expected admitted by the gate | **0.15 (0.009 %)** |
| admitted more likely than not | **none** |

**The five named exceptions** — `v1_cpg_{175,277,169,4,6}` — are reported, not
subtracted. They pass because they are **crawls** (checkpoint 1 §5, verified at
128 permuted replicas), and HQ approved distilling them as crawl seeds.

## 4. Criterion 6b — every known walker passes and classifies as walk

| probe | per-replica | label | agreement | median dx | admitted |
| --- | --- | --- | --- | --- | --- |
| walk_seed_0 | 0.227 | walk | 1.000 | 0.190 m | **0.017** |
| walk_seed_1 | 0.797 | walk | 0.992 | 0.678 m | 0.942 |
| walk_seed_2 | 0.711 | walk | 0.984 | 1.049 m | 0.842 |
| walk_seed_3 | 0.812 | walk | 1.000 | 1.430 m | 0.950 |
| walk_seed_4 | 0.797 | walk | 1.000 | 0.889 m | 0.943 |
| walk_seed_5 | 0.812 | walk | 0.992 | 0.828 m | 0.947 |
| v3_elite_0..4 | 0.711–0.797 | walk | 0.836–0.922 | 1.72–2.09 m | 0.819–0.945 |

**All eleven classify as walk.** Mean admission **0.831**.

**One seed is excluded and it should be:** `walk_seed_0` travels 0.190 m in 7 s
(2.7 cm/s) against a predicate demanding 2.5 cm/s *sustained in every window*.
It is admitted 1.7 % of the time. That is P2' doing its job — the slowest
teacher gait is barely forward locomotion — but it means **five of six
distilled seeds will seed the archive, not six**, and the walk sub-archive
starts from a slightly narrower base than v3's.

## 5. Criterion 6c — a walk-only run under the v4 gate

8 iterations × 512 offspring × 8 replicas, seeded from the six PPO seeds.

```
it  0 | evals  8200 | wa  28 | feas  0.0% | ins  0.0% | retest   0                    | 256s
it  1 | evals 13008 | wa  71 | feas 64.1% | ins 48.1% | retest  88 pass 80.7% evict 17 | 507s
it  2 | evals 18080 | wa  94 | feas 61.0% | ins 30.6% | retest 121 pass 77.7% evict 27 | 766s
it  4 | evals 28664 | wa 145 | feas 52.0% | ins 22.8% | retest 160 pass 86.2% evict 15 | 1271s
it  6 | evals 40000 | wa 183 | feas 44.6% | ins 17.9% | retest 201 pass 86.1% evict 18 | 1774s
it  8 | evals 51808 | wa 205 | feas 59.3% | ins 22.8% | retest 230 pass 86.5% evict 25 | 2273s
```

* **Feasibility 44.6–64.1 %** — inside v3's measured 40–65 % band. ✅
* **Every elite lands in the walk sub-archive** (crawl/roll/hop/other all 0):
  the classifier is not misfiling walkers. ✅
* **Incumbent re-testing is running from iteration 1** (Q7), re-testing the
  whole archive each iteration, **pass rate 77.7–86.5 %** — which independently
  reproduces the 0.78 per-replica walker rate the gate was set from — and
  **evicting 15–33 elites per iteration**, 179 in total. The winner's curse is
  being corrected in flight rather than measured after the fact. ✅

### The archive it produced, verified under P2'/5-of-8

| | v3 re-baselined | **v4 walk smoke (8 iters, half batch)** |
| --- | --- | --- |
| elites raw | 356 | 205 |
| elites verified | 301 | 178 |
| archive robustness | 84.6 % | **86.8 %** |
| mean viable replicas | 0.763 | **0.805** |
| **resolvable cells ≥ 0.25 m** | **57** | **83** |
| best verified median | +2.322 m | +2.015 m |
| archive optimism | +0.263 m | **+0.119 m** |
| resolvable grid | 7×9 | **11×16** |

**The walk sub-archive already beats the amended regression bar (57 resolvable
cells) after eight iterations at half the batch.** Two secondary readings worth
having:

* **Archive optimism halves** (+0.263 → +0.119 m). That is the incumbent
  re-testing: an elite that got in on a lucky draw is re-rolled and evicted
  before it can be reported.
* **The resolvable grid is finer** (7×9 → 11×16) because the elites are more
  robust, so their descriptors are less noisy. Resolution is not a property of
  the grid; it is a property of how reliable the things in it are.

Best median is below v3's (+2.02 vs +2.32) — expected at 8 iterations of 512
against 50 of 1024, and the bar (≥1.5 m) is cleared.

---

## 6. Budget — measured, not extrapolated

The 512-batch run gives 252 s/iteration. Scaling that linearly to 1024 would
say 504 s and a 6.8–8.5 h full run. **That extrapolation is wrong**, and v3's
ledger already said so — batch size is nearly free on this box (64 → 2048
worlds only doubles generation time). So it was measured instead: two
iterations at the full batch.

| | batch 512 | **batch 1024** |
| --- | --- | --- |
| iteration 0 (seeds + random init) | 256 s | 325 s |
| per iteration | 252 s | **316 s** (309, 324) |
| offspring evals / iteration | 4104 | **8200** |
| feasibility | 44.6–64.1 % | 64.9–65.4 % |

Doubling the batch costs 1.25×, not 2×. The re-test is already priced in: it
pads to a full batch whether it re-tests 100 incumbents or 1000, so 316 s is
the **steady-state** number rather than an early-iteration one.

### Projection for the remaining work

| item | cost |
| --- | --- |
| crawl seed distillation (5 CPG teachers × 4 DAgger rounds, 128 envs) | **~8 min** |
| full multi-modal run, 400k offspring evals (49 iterations) | **4.3 h** |
| full multi-modal run, 500k offspring evals (61 iterations) | **5.4 h** |
| final 8-replica P2' verification, ~1000 elites across sub-archives | **~35 min** |
| rendering + viewer | ~20 min |
| **total, checkpoints 3–4** | **~5.5–6.5 h** |

That is inside the spec's 6–10 h envelope. For scale: v3 spent 2.03 h on
426k evaluations; v4 is ~2.1× dearer per evaluation, and the whole of that is
the incumbent re-test doubling the rollouts per iteration. It is buying the
thing v3 named as its clearest negative result, and §5 shows it working —
optimism halved, 179 elites evicted in 8 iterations.

**Recommendation: 400k offspring evals (49 iterations at batch 1024).** The
smoke run's insertion rate was already down to 20 % by iteration 8, and v3's
full run added only 26 verified elites between its budget-matched snapshot and
its final one. I would rather spend the margin on checkpoint 4's verification
and on a second seeded mode than on iterations 50–61.

---

## 7. The roll seed is blocked on a missing artefact — a decision for HQ

Q3 routes the roll seed through "distil the best roulade checkpoint". **There
is no roulade checkpoint on this machine.** `logs/` is gitignored and was not
retained; the only surviving trained artefact from the earlier jobs is the
already-distilled `ppo_seeds.npz` in `qd-run-archives/j004`. wandb is not
authenticated in this session, so I cannot look for one there either — and I
am not going to ask for credentials.

This is not the physics conversation the spec anticipated for checkpoint 3. It
is a missing file. Three ways forward:

1. **Alex runs `wandb login`, or points me at a roulade run path**, and I distil
   it as planned. Cost: ~10 min. By far the best outcome if the run exists.
2. **Train the roulade task fresh.** The cfg exists and encodes a five-run
   lesson arc. AGENTS.md budgets curriculum-heavy tasks at 4000–6000 iterations
   — **4–6.5 h**, which pushes the job to 10–13 h and past the envelope.
3. **Ship roll unseeded and report the account**, which the acceptance criterion
   explicitly permits: *"≥3 verified roll cells OR a measured account of why the
   roll archive is thin/empty (physics vs search, with numbers)."* The account
   is already measured: no open-loop probe exceeds **0.22 rad/s** of supported
   world-horizontal rotation against the 0.8 rad/s rule over 129 swept
   variants, and v2 measured that isotropic mutation cannot leave the feasible
   manifold it starts on — so an unseeded roll sub-archive is expected empty by
   *search*, with the physics question left open rather than answered.

**My recommendation: (1) if it is cheap for Alex, otherwise (3).** Option 2
spends more than half the remaining budget on the mode the design already rates
"high risk", and it would come out of the crawl work, which is the mode that is
demonstrably real.

Note that crawl is unaffected — its seed exists and needed no training run at
all, which is the checkpoint-1 finding paying for itself.

---

## Status against checkpoint-2 criteria

* P2' gate, classifier, hierarchical archives, per-mode parent budget,
  incumbent re-testing, progress-minus-impact critic: **implemented**, 335
  tests green, changed files ruff-clean. ✅
* Every known negative fails, five exceptions named and explained: ✅
* Every known walker passes and labels as walk: ✅
* Walk-only run reproduces a healthy sub-archive, feasibility in band: ✅
* Budget projected from measurement: ✅ (§6)

**Ready for checkpoint 3 on crawl.** Blocked on the roll seed pending §7.
