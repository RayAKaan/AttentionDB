# C5 Candidate-Set Analysis Report

**Question**: Does the cross-head interaction change the candidate SET (not just scores)?

**Method**: Per-query binary indicator = 1 if `round2_added_ids_total > 0` in any of 5 reps (majority vote). Protocol: interaction must change candidate set to support causal claim.

---

## TEST Phase (Frozen)

| Dataset | Queries | Changed | Rate | Binomial p (vs H₀: ≤1%) |
|---------|---------|---------|------|-------------------------|
| SciFact | 300 | 300 | **100.0%** | < 1e-300 |
| NFCorpus | 323 | 322 | **99.7%** | < 1e-300 |

**Result**: Candidate set changes on >99% of queries on BOTH datasets. Mechanism is **NOT inert** (protocol §9: inert if >99% unchanged; here >99% CHANGED).

---

## VALIDATION Phase (λ Tuning)

| Dataset | λ | Queries | Changed | Rate |
|---------|---|---------|---------|------|
| SciFact | 0.10 | 200 | 180 | 90.0% |
| SciFact | 0.25 | 200 | 194 | 97.0% |
| SciFact | 0.50 | 200 | 198 | 99.0% |
| NFCorpus | 0.10 | 324 | 312 | 96.3% |
| NFCorpus | 0.25 | 324 | 323 | 99.7% |
| NFCorpus | 0.50 | 324 | 324 | 100.0% |

Glue arm (λ=0): 0% changed on both datasets (matches control B exactly).

---

## Per-Query Change Magnitude (TEST, mean over 5 reps)

| Dataset | Mean added IDs/query | Median | Max | % queries with ≥1 added |
|---------|---------------------|--------|-----|------------------------|
| SciFact | 6.2 | 5 | 31 | 100% |
| NFCorpus | 7.1 | 6 | 28 | 99.7% |

`round2_added_ids_total` = total new candidate IDs added in round 2 via cross-head interaction.

---

## Causal Claim Assessment

**Protocol requirement**: "causal claim requires the interaction to change the candidate SET (not just scores)"

- ✅ **Satisfied**: Candidate set changes on >99% of TEST queries
- ✅ **Glue arm validation**: λ=0 glue arm produces 0% change (matches B exactly)
- ⚠️ **But**: Recall@10 effect is negligible (|Δ| < 0.01) and not statistically significant

**Conclusion**: The interaction **does** change the candidate set (causal mechanism operates), but the **practical recall benefit is negligible**. This is a valid neutral/negative result per protocol: "neutral results are valid."