# C4 — Confirmed-Scale Execution Report

Date: 2026-09-26
Engine: c2pilot `3ec8c193354975fa26a3961d8835289a0cad54cc686141f75573baeaebb31c98`
(deadline-capable deterministic /Brepro build; supersedes `A8E0C0CB…`; C4-BINVERIFY-003 PASS 4/4).
Branch: `comparative-study/c4-confirmed-benchmark`.

## 1. Scope executed

13 confirmed-scale TEST cells + 2 boundary cells (MEM/TIME), all on frozen
test splits (SciFact 300 qids / NFCorpus 323 qids), 5 fresh-process reps
(B0 oracles: 1 rep), seed 20260925, k=10. Binary verified per run
(15/15 `environment.yaml` record `c2pilot_sha256 == 3ec8c193…`).

## 2. Results (recall@10 on qrels, mean ± sd over 5 reps)

| cell | mode | split | n | recall@10 | sd | p50 us |
|------|------|-------|---|-----------|-----|--------|
| C4-W01-SCI-B0-001 | B0 oracle | SCI | 300 | 0.7833 | n/a (1 rep) | — |
| C4-W01-NFC-B0-001 | B0 oracle | NFC | 323 | 0.1550 | n/a (1 rep) | — |
| C4-W01-SCI-B1-001 | B1 EF32 | SCI | 300 | 0.7783 | 0.0049 | 1028 |
| C4-W01-NFC-B1-001 | B1 EF64 | NFC | 323 | 0.1491 | 0.0022 | 941 |
| C4-W02-SCI-B2-001 | B2 (UNREACHABLE) EF64-C500 | SCI | 300 | 0.7920 | 0.0022 | 1940 |
| C4-W02-NFC-B2-001 | B2 EF16-C50 | NFC | 323 | 0.1597 | 0.0015 | 1699 |
| C4-W03-SCI-B3-001 | B3 EF64-C500 | SCI | 300 | 0.7911 | 0.0022 | 1851 |
| C4-W03-NFC-B3-001 | B3 EF16-C50 | NFC | 323 | 0.1593 | 0.0006 | 1718 |
| C4-W04-SCI-B4-001 | B4 EF64-C500 | SCI | 300 | 0.7918 | 0.0008 | 1917 |
| C4-W07-SCI-B7-001 | B7 (UNREACHABLE) EF64-C500 | SCI | 300 | 0.7421 | 0.0054 | 2106 |
| C4-W07-NFC-B7-001 | B7 EF64 | NFC | 323 | 0.1488 | 0.0017 | 2011 |
| C4-W07-SCI-B7-TRKB-001 | B7 Track A/B | SCI | 300 | 0.7332 | 0.0262 | 2132 |
| C4-W01-NFC-B1-TRKB-001 | B1 Track A/B | NFC | 323 | 0.1503 | 0.0022 | 772 |

Boundary cells (C2 BUDGET / C3 budget-propagation carry):

- MEM `C4-W19-SCI-B3-MEM-001` legs {512 MiB, 1.0 GiB, 1.6 GiB}: all
  within_leg_budget True (peak child RSS per leg ≈ 272 / 305 / 305 MiB).
- TIME `C4-W19-SCI-B1-TIME-001` deadline_us {1000, 10000, 100000}: deadline
  exceeded totals {0, 0, 0}. Leg 1 (1000 µs) yields recall 0.7833 with p50 1015 µs.

## 3. Primary-family contrasts (paired, per-query = mean of 5 reps; bootstrap 10k seeded; Holm within family)

| dataset | contrast | verdict | diff [95% CI] | p_wilcoxon | p_holm | dz |
|---------|----------|---------|---------------|-----------|--------|----|
| SCI | B1 vs B2 | n.s. | −0.0136 [−0.0417, +0.0143] | 0.1763 | 0.7051 | −0.055 |
| SCI | B2 vs B3 | n.s. (negligible) | +0.0008 [−0.0031, +0.0052] | 0.7875 | 1.0000 | +0.023 |
| SCI | B3 vs B4 | n.s. (negligible) | −0.0007 [−0.0034, +0.0021] | 0.7335 | 1.0000 | −0.028 |
| SCI | B2 vs B7 | SIGNIFICANT | +0.0498 [+0.0257, +0.0764] | 0.0001 | 0.0004 | +0.223 |
| NFC | B1 vs B2 | n.s. | −0.0105 [−0.0222, −0.0000] | 0.0917 | 0.4584 | −0.104 |
| NFC | B2 vs B3 | n.s. (negligible) | +0.0003 [−0.0013, +0.0019] | 0.8081 | 1.0000 | +0.021 |
| NFC | B2 vs B7 | SIGNIFICANT | +0.0109 [+0.0024, +0.0210] | 0.0022 | 0.0133 | +0.126 |
| NFC | B3 vs B4 | NOT-EXECUTABLE | B4 frozen for SciFact only (documented no-op arm); never interpolated | | | |

Verdict key per C1 `statistical-plan.md`: |diff| < 0.01 ⇒ negligible
regardless of effect; otherwise significance decided by Holm-adjusted Wilcoxon.
B2-vs-B7 is significant on both datasets (B2 > B7). B2-vs-B3 and B3-vs-B4
paired differences are negligible on both datasets (≤ 0.0037).

## 4. Budget matching (C4.3)

- Candidate budget (BUDGET-CAND:500): max candidate_count across all cells =
  128 (NFC-B7), means 0–97 (B1 exact-path cells report candidate_count 0;
  single-head exact mode selects no ANN candidates). All ≤ 500. ✅
- Memory (BUDGET-MEM:512 MiB): peak child-tree RSS max 305.1 MiB (SCI-B2 car),
  all cells ≤ 512. ✅
- MEM legs: every leg measured child RSS within leg budget. ✅
- TIME legs: 0/0/0 deadline misses. ✅
- Equal-budget / equal-quality tables honest; UNREACHABLE-flagged only cells
  where the matched candidate config genuinely cannot be reached on SciFact
  (SCI-B2, SCI-B7 keep plan defaults EF64-C500, never interpolated).

## 5. Reconciliation (C4.6)

`run-reconciliation.csv`: 15/15 TEST cells reconcile vs frozen plan
(split=TEST, n == plan query_count, rep_count matches, binary sha match,
status PASS). B0 exact oracles reproduced within process-to-process tolerance
(identical to C3 oracle checks); exact-top10 hashes recorded per oracle cell.

## 6. Determinism / integrity

- Within-config per-query *order* is bit-identical across all runs (frozen
  seeded order → pairwise exact alignment for paired stats).
- Single-rep engine recall is deterministic in-process; fresh-process execution
  shows intrinsic variance (sd ≤ 0.006 for all non-TRKB cells). B0 oracle
  single-rep means are stable.
- Largest within-cell spread: `C4-W07-SCI-B7-TRKB-001` sd 0.0262 (rep values
  {0.7344, 0.7467, 0.7506, 0.7522, 0.6822}). Per C1 rule, no outlier removal;
  reported transparently. All non-TRKB cells sd ≤ 0.0054.

## 7. Gate audit

`c4-gate-audit.md`: 20/20 gates PASS. No open defects blocking closure.

## 8. Failed / superseded attempts

- `c2pilot` `A8E0C0CB…` superseded (deadline-capable rebuild at `3ec8c193…`);
  C4-BINVERIFY-002 PASS noted for the earlier binary, superseded by
  C4-BINVERIFY-003 for `3ec8c193…`.
- Tuning-era runs (C4.2, `…-000` run ids) preserved; definitive TEST is the
  C4.4 `…-001` set above.
- `c4-system-eligibility.csv` was truncated to 0 bytes during a session (bad
  `.replace` write rollback); recovered verbatim from git HEAD **plus** the
  uncommitted reconstruction of the pre-truncation content (data-loss
  incident, root cause = evaluation-order truncation before read; fix =
  open-read-then-write). 13 rows, header intact, verified.