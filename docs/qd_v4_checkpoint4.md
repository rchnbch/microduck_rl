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
| archive robustness | 84.6 % | 89.0 % | **100.0 %** |
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
| **Robustness ≥90 % across all sub-archives** | ✅ **91.0 %** (292 of 321) |
| Incumbent re-testing from iteration 1, pass rate reported | ✅ every iteration, **1,118 evictions**, 80–88 % pass |
| All distances medians over permuted replicas | ✅ |
| Viewer mode tabs, README, branch, PR, logs | ✅ |
| Tests green, changed files ruff-clean | ✅ 337 tests |

**Ten of eleven pass. The miss is walk.**

---

## Why walk regressed, in two measured parts

**1. The 57-cell bar was unreachable at this archive's own resolution.** v4
walk's resolvable grid is **6×8 = 48 cells**. 57 > 48, so no amount of coverage
could have reached the bar. The archive fills **41 of the 48 cells its own
descriptor reproducibility supports (85 %)**, against v3's 57 of 63 (90 %). The
grid coarsened because the descriptor was measured more noisily here, and a
coarser grid caps the count before coverage is even in question.

This is the same class of mistake v3's README documents making, and I did not
catch it when the bar was set at checkpoint 1: **a cell-count bar is only
meaningful alongside the resolution it is counted at.**

**2. Walk got half the search, by design.** The per-mode parent budget splits
GA parents evenly across non-empty modes, and crawl was non-empty from
iteration 0 — so walk drew **256 of 512 parents** from the first iteration
where v3's walk drew all of them. v4's walk archive is the product of roughly
half of v3's walking search: 263 raw elites against 356.

That is the budget doing its job, not failing. Without it, 263 walkers against
58 crawls would have handed crawl **18 %** of the parents instead of 50 %, and
the crawl archive would not exist. **The trade is 42 crawl cells for 16 walk
cells.** Whether that is a good trade is a judgement about what the archive is
for; the criterion as written says walk must not regress, and it did.

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

1. **The walk miss is a real regression against the criterion**, and the trade
   that caused it is exactly the one the job was commissioned to make ("the
   behaviors are not different enough… I don't just want walking"). 83 cells in
   two modes against 57 in one. Your call whether that is the right side of
   the trade.
2. **The 57-cell bar could not have been met** at v4 walk's measured resolution
   (48 cells total). I set that bar at checkpoint 1 without checking it against
   the resolution the new archive would have — my error, and the same class of
   error v3 documented.
3. **`SHELL_FRICTION = 0.4` is a literature value, not a measurement.** Every
   crawl number inherits its uncertainty. One tribometer reading on a real
   shell retires it; the sensitivity is measured (across the DR range the crawl
   varies ~25 %, but at the legacy µ = 1.0 it is not viable at all).
4. **Roll remains open.** If a roulade checkpoint exists in wandb, seeding roll
   is ~10 minutes of work and the question gets a real answer.
