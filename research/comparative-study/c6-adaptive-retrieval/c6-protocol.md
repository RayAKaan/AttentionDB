# C6 Protocol: Adaptive Retrieval Allocation

**Version**: 1.0.0 (frozen)  
**Branch**: `comparative-study/c6-adaptive-retrieval`  
**Depends on**: C0–C5 complete; C5 artifacts frozen in `raw/`

---

## 1. Research Question

**H0**: Adaptive cross-head allocation does not materially improve recall/efficiency over independent multi-head union (C5-B) when candidate budget and latency are controlled.

**H1**: Cross-head signals can dynamically allocate candidate-generation effort between heads, producing equal-or-better recall at lower/equal cost.

---

## 2. Mandatory Arms (7)

| Arm | Name | Mechanism | Adaptive? |
|-----|------|-----------|-----------|
| C6-A | Canonical | Single-head (CANONICAL) | No |
| C6-B | Independent Multi-Head | 3-head union (TITLE/BODY/CITE) | No |
| C6-C | Static Equal Budget | Equal split across 3 heads | No (no redistribution) |
| C6-D | Query-Adaptive | Query→centroid cosine → softmax | No (no redistribution) |
| C6-E | Interaction-Guided | Stage1(30%) → overlap/entropy → Stage2(70%) | **Yes** |
| C6-F | E + C5 Cross-Refine | E + λ=0.10 cross-refine | **Yes** |
| C6-RE | Randomized Control | Random weights | No (no redistribution) |

---

## 3. Budget Constraints (FROZEN)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `total_candidates` | 500 | Union cap (same as C5) |
| `total_ef_work` | 192 | 64 × 3 heads (same as C5-B total EF) |
| `min_per_head` | 20 | Floor per head |
| `max_per_head` | 300 | Ceiling per head |
| `stage1_fraction` | 0.3 | C6-E/F: 30% budget for Stage 1 |

All arms operate under **identical total budget**.

---

## 4. Allocation Policies

### C6-C: StaticEqualPolicy
```
initial: candidates = total / n_heads, ef = total_ef / n_heads
redistribute: none
```

### C6-D: QueryAdaptivePolicy
```
signal_h = max(0, cosine(query, centroid_h))
initial: candidates ∝ signal_h, ef ∝ signal_h (with floor/ceil)
redistribute: none
```

### C6-E: InteractionGuidedPolicy
```
Stage 1 (30% budget): equal split
Compute per-head: overlap_with_others, score_entropy
signal_h = (1 - overlap) × (1 + entropy/entropy_threshold)
Stage 2 (70% budget): candidates ∝ signal_h
```

### C6-F: InteractionGuidedPolicy + C5 Cross-Refine (λ=0.10)
Same as C6-E + C5 one-step cross-refine on Stage-2 results.

### C6-RE: RandomizedPolicy
```
initial: random weights (seed 20260925)
redistribute: random weights (seed 20260925 + 0x9E3779B9)
```

---

## 5. Experimental Design

### Splits (Per C4/C5 Protocol)
| Split | SciFact | NFCorpus |
|-------|---------|----------|
| TEST | 300 qrels (test.tsv) | 323 qrels (test.tsv) |
| VALID | 200 seeded train-sample | 324 dev.tsv |
| PROBE/SMOKE | 20 from VAL | 20 from VAL |

### Replications
- 5 fresh-process runs per cell (seed 20260925)
- Query order: `shuffled_indices(n, seed + rep × 0x9E3779B9)`

### Plan Cells (30 total)

| Track | Cells | Description |
|-------|-------|-------------|
| SAFETY | 4 | EFPROBE: ef 16 vs 128 on both datasets (MODE B) |
| SAFETY | 14 | SMOKE: all 7 arms on both datasets (20 qids) |
| TUNE | 12 | VALIDATION: B, C, D, E, F, RE on both datasets |
| TRK-A | 14 | TEST: 7 arms × 2 datasets × 5 reps |

---

## 6. Budget Conservation (Mandatory)

For every adaptive query record:
- `initial_allocation`: per-head candidates, EF
- `redistributions`: per-head Stage-2 deltas
- `final_allocation`: initial + redistributions
- `budget_conserved`: `sum(candidates) ≤ total_candidates` AND `sum(ef) ≤ total_ef_work`
- `total_candidates_used`, `total_ef_work_used`

---

## 7. Prove Adaptivity (Mandatory)

For adaptive arms (E, F), demonstrate:
- Allocation varies across queries (entropy > 0)
- Per-head `candidates`, `ef`, `round`, `reason` recorded
- Redistribution triggers logged (overlap threshold, entropy threshold)

---

## 8. No Test-Set Tuning

Hyperparameters frozen before TEST:
- `stage1_fraction = 0.3`
- `overlap_threshold = 0.3`
- `entropy_threshold = 1.0`
- `randomized_seed = 20260925`

Tuned on VALIDATION only (12 cells).

---

## 9. Stop Conditions (Any → Halt + Preserve + Report)

1. Anchor hash mismatch (B0)
2. Exact-oracle discrepancy
3. ef-sensitivity probe failing (ef not real knob)
4. Budget leak in adaptive arms
5. Raw evidence overwritten
6. 3 consecutive harness crashes
7. Silent wrong result

---

## 10. Statistics (Per C4/C5 Framework)

- Unit = query; per-query recall = mean of 5 reps
- Paired bootstrap 10,000 (seed 20260925) → 95% CI
- Wilcoxon signed-rank (two-sided)
- Holm-Bonferroni across 10 primary contrasts (B vs C/D/E/F/RE × 2 datasets)
- Cohen's dz effect sizes
- |Δ| < 0.01 → negligible
- No winner labels; no post-hoc exclusion

---

## 11. Deliverables

1. `c6-protocol.md` (this file)
2. `c6-architecture.md` (implementation details)
3. `c6-run-plan.csv` (30 cells)
4. `probe/c6pilot.rs` (reproducible `/Brepro` build)
5. `harness/c6_test_run.py` (orchestrator)
6. `harness/c6_analyze.py` (statistics)
7. 9 reports in `analysis/reports/`:
   - 1_results_report.md
   - 2_statistical_report.md
   - 3_candidate_report.md
   - 4_environment_report.md
   - 5_dataset_report.md
   - 6_resource_report.md
   - 7_implementation_report.md
   - 8_failure_report.md
   - 9_reproducibility_report.md
   - 10_closure_report.md
8. `analysis/statistical_results.json`
9. Raw evidence in `raw/C6-*/artifacts/`
10. `raw/RUN-INDEX.yaml` registrations
11. Updated `research/comparative-study/README.md`
12. PR into main (no auto-merge)