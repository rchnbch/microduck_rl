# j007 checkpoint 4 — the v4 multi-modal archive, finished

Branch `qd-modes-v4`, PR #5 (base `qd-walking-v3`). Logs in
`qd-run-archives/j007/` (232 MB). Viewer:
https://claude.ai/code/artifact/cb5bf696-508c-4f5e-ab46-b88c4534506d
(six tabs: v1 CPG, v1 PGA-ME, v2, v3, **v4 walk**, **v4 crawl**).

**Headline: two modes, 83 resolvable cells of verified forward locomotion,
where v3 had 57 cells of walking. Crawl is new, real, and 100 % robust. Walk
regressed by 16 cells and the arithmetic below says exactly why.**

---

## The run

49 iterations × 1024 offspring × 8 world-permuted replicas =
**547,464 offspring evaluations in 4.6 h** (measured projection was 4.3 h).
Seeded with five distilled PPO walkers and one distilled crawl.

| | v3 walk, re-verified under P2' | **v4 walk** | **v4 crawl** |
| --- | --- | --- | --- |
| elites raw | 356 | 263 | 58 |
| elites verified (5-of-8) | 301 | 234 | **58** |
| archive robustness | 84.6 % | **89.0 % — below the 90 % bar** | **100.0 %** |
| mean viable replicas | 0.763 | 0.758 | **0.963** |
| resolvable grid | 7×9 = 63 | 6×8 = 48 | 20×20 = 400 |
| **resolvable cells ≥ 0.25 m** | **57** | **41** | **42** |
| best verified median | +2.322 m | +2.254 m | **+1.271 m** |
| archive optimism | +0.263 m | +0.204 m | **+0.010 m** |
| verifying as another mode | — | 12 (crawl) | 0 |
| failed label agreement | 0 | 0 | 0 |

---

## Against the amended acceptance criteria

| criterion | result |
| --- | --- |
| Shell fix landed first, documented, before/after evidence | ✅ side-lying pose sat 26.1 mm too low; at legacy µ the best crawl is not viable at all |
| Stage A' executed per §6.2, sweeps reported whole | ✅ — and the pre-registered bars were **NOT met**; reported as the honest failure, no bar lowered |
| Negative-probe rejection, exceptions by name | ✅ 0.15 expected admissions of 1656; five named exceptions that turned out to be real crawls |
| **Walk does not regress** (≥57 resolvable cells, best ≥1.5 m) | ❌ **41 cells** (best +2.254 m ✅) — see below |
| **Crawl is real** (≥10 resolvable cells, best ≥0.25 m) | ✅✅ **42 cells, best +1.271 m** — 4× and 5× the bars |
| Roll attempted honestly (≥3 cells **or** a measured account) | ✅ via the account: 0.22 rad/s over 129 variants against the 0.8 rad/s rule |
| Mode integrity (≥7/8 label agreement, per-window constancy) | ✅ zero failures in either sub-archive |
| **Robustness ≥90 % across all sub-archives** | ✅ aggregate **91.0 %** (292/321) — but **walk alone is 89.0 % and fails**; crawl is 100 %. The aggregate passes because crawl is perfect and small. |
| Incumbent re-testing from iteration 1, pass rate reported | ✅ every iteration, **1,118 evictions**, 80–88 % pass |
| All distances medians over permuted replicas | ✅ |
| Viewer mode tabs, README, branch, PR, logs | ✅ |
| Tests green, changed files ruff-clean | ✅ 337 tests |

**Ten of eleven pass. The miss is walk.**

---

## Why walk regressed — and a wrong explanation, corrected

**Walk's 41 cells against the 57 bar is a genuine regression.** My first
account of it said the bar was *unreachable* at this archive's resolution.
**That is refuted by this job's own checkpoint-2 measurement**, and the
correction matters because the wrong version partly excused the result.

The checkpoint-2 walk-only smoke run, on **the same gate and the same
verifier**, produced an **11×16 resolvable grid and 83 resolvable cells** from
178 verified elites. A walk sub-archive can support far more than 48 cells, so
the final run's 6×8 grid is an **outcome**, not a pre-existing constraint:

| walk archive | raw | verified | mean viable replicas | resolvable grid | cells ≥ 0.25 m |
| --- | --- | --- | --- | --- | --- |
| checkpoint-2 smoke (walk-only, 8 iters) | 205 | 178 | **0.805** | **11×16** | **83** |
| v3, re-verified | 356 | 301 | 0.763 | 7×9 | 57 |
| **v4 final (multi-modal, 49 iters)** | 263 | 234 | **0.758** | **6×8** | **41** |

The resolvable grid tracks **how reliable the elites are**, not how many there
are. The final walk elites clear the gate slightly more often than the smoke's
(89.0 % vs 86.8 %) but are individually less reliable (0.758 viable replicas vs
0.805); a less reliable elite spends more of its episode fallen, its descriptor
is noisier, and the grid its geography can be counted on coarsens. 83 collapses
to 41 through the *resolution*, not the coverage.

**The likely driver is the parent budget**, and that part is by design: walk
drew **256 of 512 parents** from iteration 0 where v3's walk drew all of them.
Shallower search → less reliable elites → noisier descriptors → coarser grid.
**That chain is inference from the table, not a controlled measurement** — the
controlled version is a walk-only run at the same batch and iteration count,
which this job did not spend.

Without the budget, 263 walkers against 58 crawls would have handed crawl 18 %
of the parents instead of 50 % and the crawl archive would not exist. **The
trade is 42 crawl cells for 16 walk cells.**

**Separately, the bar was mis-set — a different error, and mine.** I
transplanted v3's re-verified count into a bar for a different archive without
checking the resolution it would be counted at. A count on a 7×9 grid and a
count on a 6×8 grid are not comparable, so the bar as written was
**uncountable**, not unreachable. State a cell bar with the grid it is counted
on, or as a fraction of the resolvable grid: v3 filled 90 % of its 63, the
smoke 47 % of its 176, v4 walk 85 % of its 48.

---

## What the strictness sweep says about the two modes

| survives at least | walk elites | walk cells | crawl elites | crawl cells |
| --- | --- | --- | --- | --- |
| 1 of 8 | 251 | 41 | 58 | 42 |
| **5 of 8** (the gate) | **234** | **41** | **58** | **42** |
| 6 of 8 | 180 | 40 | 56 | 41 |
| 7 of 8 | 105 | 36 | 55 | 41 |
| 8 of 8 (unanimous) | 36 | 24 | **46** | **33** |

The two modes are not the same kind of object. **Crawl is nearly
deterministic**: 58 of 58 at the gate, still 46 at unanimous-8, and archive
optimism of **+0.010 m** — essentially none. Walking falls and crawling does
not, so walk drops 234 → 36 between 5-of-8 and 8-of-8.

**This retro-justifies checkpoint 1's ruling precisely.** The relaxation from
unanimous-8 to 5-of-8 was needed **for walk and only for walk** — the crawl
sub-archive would have satisfied the design's original unanimous gate as
written.

---

## The empty modes

* **roll — 0, unseeded.** No roulade checkpoint exists on this machine and no
  scripted probe rotates (0.22 rad/s over 129 swept variants against the
  0.8 rad/s rule). v2 measured that isotropic mutation cannot leave its
  manifold, so an unseeded mode is empty **by search**. The physics question is
  left open, not answered.
* **hop — 0, unseeded by design** (Q1). Appendix A measured 0.19–0.30 m/s
  against the 0.44 m/s a 1 cm hop needs.
* **other — 0, and this is the informative zero.** Across 547k evaluations the
  classifier produced no viable behaviour it could not name. The five-way
  partition is doing real work rather than using "other" as a dumping ground.

---

## The crawl plateau — the finding that shapes the follow-up

```
it  0: 17    it 12: 51    it 24: 56    it 36: 55
it  4: 43    it 16: 53    it 28: 56    it 49: 58
it  8: 48    it 20: 56    it 32: 55
```

Crawl opened at 17 cells, reached 56 by iteration 20, and **stayed flat for the
remaining 29 iterations** while walk climbed to 281. That is v2's manifold
result reproduced as a curve rather than argued: **one seed buys one manifold.**
Walk had five seeds; crawl had one.

It is also the concrete case for the follow-up. A prone-locomotion PPO task
(the design's Q3 path (a)) would produce a state-conditioned, in-action-box,
standing-spawn crawl policy — several genuinely different crawl seeds instead
of one — and the clipping test already proved a genome-expressible crawl exists
(all five scripted crawls stay viable at 1.000 clipped into HOME ± 1 rad).

---

## What I would flag to Alex

1. **The walk miss is a real regression**, and I initially explained it in a
   way that let it off too lightly — the "unreachable bar" account is refuted
   by this job's own smoke run (11×16 grid, 83 cells, same gate). The trade
   that caused it is the one the job was commissioned to make ("the behaviors
   are not different enough… I don't just want walking"): 83 cells in two modes
   against 57 in one. Your call whether that is the right side of it.
2. **The bar was also mis-set, separately.** I transplanted v3's cell count
   without the grid it was counted on, which made it uncountable rather than
   unreachable — my error at checkpoint 1, and the same class of error v3
   documented.
3. **Archive robustness passes only in aggregate.** 91.0 % pooled; walk alone
   is 89.0 % and misses the bar.
4. **`SHELL_FRICTION = 0.4` is a literature value, not a measurement**, and
   **no task randomizes it yet** — the DR event term is future work for the
   crawl-PPO follow-up. Every
   crawl number inherits its uncertainty. One tribometer reading on a real
   shell retires it; the sensitivity is measured (across the DR range the crawl
   varies ~25 %, but at the legacy µ = 1.0 it is not viable at all).
5. **Roll remains open.** If a roulade checkpoint exists in wandb, seeding roll
   is ~10 minutes of work and the question gets a real answer.
