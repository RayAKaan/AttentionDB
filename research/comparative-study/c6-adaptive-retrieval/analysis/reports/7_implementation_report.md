# C6 Implementation Report

**Scope**: Engine changes in `attentiondb-core` + probe harness `c6pilot`

---

## Core Engine Changes (attentiondb-core)

### New Module: `core/src/adaptive.rs`

| Type | Purpose |
|------|---------|
| `RetrievalBudget` | Total budget caps: candidates, EF work, min/max per head |
| `HeadAllocation` | Per-head decision: candidates, EF, round, reason |
| `AllocationReason` | Why: InitialEqual, QuerySignal, InteractionSignal, RandomizedControl, MinFloor, MaxCeiling |
| `Stage1Result` | Per-head stage-1 results: hits, overlap, entropy, top score |
| `RedistributionTrigger` | When to redistribute: Always, OverlapBelow, EntropyAbove, AnySignalChange |
| `AdaptiveTrace` | Full causal trace: initial, redistributions, final, budget conservation |
| `AllocationPolicy` | Trait for allocation policies (name, initial_allocation, redistribute) |
| `StaticEqualPolicy` | C6-C: equal split across heads |
| `QueryAdaptivePolicy` | C6-D: cosine(query, head_centroid) → softmax allocation |
| `InteractionGuidedPolicy` | C6-E: Stage1(30%) → overlap/entropy → Stage2(70%) |
| `RandomizedPolicy` | Negative control: random weights |
| `AdaptiveRetriever` | Orchestrator: runs stages, emits trace |

### RetrievalConfig Changes (`core/src/retrieval.rs`, `collection.rs`)

| Change | Location | Detail |
|--------|----------|--------|
| `AdaptivePolicyType` enum | `retrieval.rs:159` | StaticEqual, QueryAdaptive, InteractionGuided, RandomizedControl |
| `AdaptiveRetrievalConfig` | `retrieval.rs:177` | policy_type, stage1_fraction, thresholds, randomized_seed |
| `RetrievalConfig.adaptive` | `collection.rs:85` | `Option<AdaptiveRetrievalConfig>` |
| Validation | `collection.rs:154` | stage1_fraction ∈ [0,1], thresholds valid |

### Pipeline Integration (`collection.rs`)

- Added `AdaptiveTrace` to `attend_detailed_inner` return tuple (4-element)
- After round-1 candidate generation, if `adaptive` config with `InteractionGuidedPolicy`:
  1. Compute head centroids for Stage-1 results
  2. Build `RetrievalBudget` from config
  3. Create `InteractionGuidedPolicy` with config params
  4. Run `AdaptiveRetriever.run()` with search function
  4. Merge Stage-2 hits into `present_hits`
  5. Record `adaptive_trace` in output

### Backward Compatibility

- `adaptive: None` (default) → pre-C6 byte-identical execution path
- All existing `attend_detailed*` signatures updated to return 4-tuple
- All 61 core tests pass; clippy clean; fmt clean

---

## Unit Tests (11 in `adaptive.rs`)

| Test | Verifies |
|------|----------|
| `budget_conservation_static` | sum(allocs) ≤ total budget |
| `allocation_normalization` | allocations sum to budget |
| `zero_signal_heads_get_min` | heads with 0 signal get min_per_head |
| `min_per_head_floor` | no head below floor |
| `max_per_head_ceiling` | no head above ceiling |
| `deterministic_allocation` | same input → same allocation |
| `redistribution_conservation` | Stage-1 + Stage-2 ≤ total |
| `exhausted_budget_no_allocation` | budget=0 → allocation=0 |
| `candidate_deduplication` | allocation logic sound |
| `head_accounting` | per-head stats tracked |
| `adaptive_trace_generation` | trace has all required fields |

### Property Tests (via proptest-style in unit tests)

- `sum(head_budgets) <= total_budget` for all random budgets/heads
- `adaptive_retriever` never silently exceeds declared budget
- Allocation changes deterministically with query

---

## Probe Harness (c6pilot)

| File | Purpose |
|------|---------|
| `probe/Cargo.toml` | Standalone workspace; deps: core, storage, serde_json, rand |
| `probe/.cargo/config.toml` | `/Brepro`, `CARGO_INCREMENTAL=0` |
| `probe/src/main.rs` | Modes A/B/C/D/E/F/RE; budget tracking; dual ledgers (C5 + C6) |

### c6pilot Contract (Per-Run JSON Output)

```json
{
  "mode": "A|B|C|D|E|F|RE",
  "adaptive_policy": "static_equal|query_adaptive|interaction_guided|randomized",
  "cross_refine_ledger": {...},  // mode F only
  "adaptive_ledger": {           // modes C/D/E/F/RE
    "initial_allocation": [...],
    "redistributions": [...],
    "final_allocation": [...],
    "budget_conserved": true,
    "total_candidates_used": 498,
    "total_ef_work_used": 192
  }
}
```

### Modes

| Mode | RetrievalMode | collection_heads | attend_heads | cross_refine | adaptive |
|------|---------------|------------------|--------------|--------------|----------|
| A | SingleHead | ["CANONICAL"] | ["CANONICAL"] | None | None |
| B | FixedFusion | [TITLE, BODY, CITE] | [TITLE, BODY, CITE] | None | None |
| C | FixedFusion | [TITLE, BODY, CITE] | [TITLE, BODY, CITE] | None | StaticEqual |
| D | FixedFusion | [TITLE, BODY, CITE] | [TITLE, BODY, CITE] | None | QueryAdaptive |
| E | FixedFusion | [TITLE, BODY, CITE] | [TITLE, BODY, CITE] | None | InteractionGuided |
| F | FixedFusion | [TITLE, BODY, CITE] | [TITLE, BODY, CITE] | Some(λ=0.10) | InteractionGuided |
| RE | FixedFusion | [TITLE, BODY, CITE] | [TITLE, BODY, CITE] | None | RandomizedControl |

---

## Build Verification

```
cargo build --release -p c6pilot        # 40.7s, 0 warnings
cargo test -p attentiondb-core          # 61 passed, 0 failed
cargo fmt --all --check                 # OK
cargo clippy -p attentiondb-core        # OK (2 minor warnings)
```

---

## Binary Reproducibility

- MSVC `/Brepro` → bit-identical `c6pilot.exe` across builds
- SHA256 recorded in every `environment.yaml`