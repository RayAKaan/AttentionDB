# C1 Validation Report

Study `comparative-study-001` · protocol v1.0.0 · validated at the C1 commit.

## Gate results (charter §16, C1 scope)

| # | Gate | Result | Method |
|---|---|---|---|
| 1 | YAML syntax valid | **PASS** | PyYAML safe_load on mode-registry.yaml, workload-registry.yaml, metric-registry.yaml, preregistration.yaml (2 initial parse errors found and FIXED: unquoted scalars containing `: `/`[0,1]`; metric-registry rewritten block-style) |
| 2 | CSV syntax valid | **PASS** | csv.DictReader: dataset-matrix.csv (13 rows × 18 cols), baseline-feasibility.csv (10 rows × 13 cols), no malformed rows |
| 3 | All referenced IDs resolve | **PASS** | machine check: preregistration ↔ modes/workloads/metrics/baselines/datasets; workloads ↔ datasets/modes — 13 datasets, 8 modes, 19 workloads, 21 metrics, 10 baselines, zero unresolved |
| 4 | Every B0–B7 has status + precise definition | **PASS** | all 8 modes carry implementation, status (implemented ×5 / conditional ×1 / requires-implementation ×2 / documented-noop ×1), inputs, candidate procedure, budget, fusion, training requirements, ModelCard flag, interaction flag, resource cost, limitations, tracks |
| 5 | Every baseline has feasibility status | **PASS** | statuses ∈ {READY, READY-PLAN, CONDITIONAL, CONDITIONAL-LOW, BLOCKED-AUTH, EXCLUDED}, each with reason + evidence column |
| 6 | Every dataset decision has evidence + rationale | **PASS** | all 13 rows carry official source, license verify-at-download flag, label structure, per-head provenance, decision + rationale; narrative in dataset-decisions.md with citations |
| 7 | Train/val/test isolation explicit | **PASS** | training-and-leakage.md (split policy, LODO transfer, INV-L1..L5, honest-status rule) + preregistration.split_rules |
| 8 | Fairness accounting includes per-head cost | **PASS** | fairness-and-resource-accounting.md CONFIG-FACTS block (vectors_per_record, dim_per_head, total_indexed_dims, representation cost) + disclosure D1 |
| 9 | Quality-matched vs budget-matched distinct | **PASS** | recall-and-budget-matching.md sections A/B with tolerance, sweeps, unreachable-target handling |
| 10 | All metrics have units + aggregation | **PASS** | metric-registry.yaml machine check (unit + direction on all 21) |
| 11 | No benchmark results generated or fabricated | **PASS** | no runs exist under comparative-study/raw/; automated scan of C1 docs for result-like keys found none; no performance claims in any C1 document |
| 12 | No C0 or Phase 3E artifacts modified | **PASS** | `git diff fe4f92b HEAD -- research/phase3` = 0 lines at C1 build time; C0 documents untouched by C1 (new files only under c1/); Phase 3E checkers not run this phase (no code changed — nothing to re-verify) |
| 13 | All C1 artifacts under the namespace | **PASS** | everything under research/comparative-study/{c1,README.md} |
| 14 | Preregistration internally consistent | **PASS** | datasets/modes/workloads/metrics/hypotheses cross-reference cleanly; hypotheses are null/directional/exploratory and presuppose no winner |
| 15 | Lightweight schema/lint checks only | **PASS** | PyYAML + csv + custom resolver as above; no heavyweight tooling |
| 16 | C2 not started | **PASS** | no harness code written; no servers started; no datasets downloaded |

## Issues found during validation (all resolved)

1. mode-registry.yaml: two YAML parse failures (unquoted scalars with `: `)
   → fixed by quoting.
2. metric-registry.yaml: flow-mapping parse hazard (`[0,1]` inside `{…}`)
   → file rewritten in block style with identical content.
3. workload-registry.yaml: two imprecise dataset references (W-15
   "DS-SCIFACT-scale clone", W-16 "DS-ANN ladder") → exact IDs written.
4. Cross-resolution initially flagged the above; re-run clean.

## Verdict

**C1 COMPLETE.** All acceptance gates pass; no critical gate failed; C2 has
not been begun. Unresolved items are preregistered blockers (BLK-1..BLK-4),
not gate failures.
