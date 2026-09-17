# Phase 3E — E6 Deviations & Corrections (PH3E-TXN-001)

All deviations are documented addenda; none silently reinterprets a format or
weakens an earlier contract. A1–A5 are untouched (A6 appended).

## D1 — `tx_after_commit_fsync` is coincident with `tx_after_commit_wal` (not separately instrumented)

The E6 spec listed `TX_AFTER_COMMIT_FSYNC` as a nominal window. In the actual
code, the WAL append call **is** the durability step: `Sync` fsyncs inside the
append, `Group` flushes to the OS, `Async` buffers (A1/A2). There is no
separate post-append fsync step to gate, so a distinct window would be a
fabricated row. Consequence: the `tx_after_commit_wal` window *is* the
post-fsync window for sync/group, and the buffered-append window for async.
Coverage matrix reports 7 windows, not 8. No separate row was invented.

## D2 — `AFTER_APPLY` / `BEFORE_ACK` are generic gates shared with the plain insert path

`GATE_AFTER_APPLY`/`GATE_BEFORE_ACK` predate E6 (plain insert/update paths).
On the transaction commit path they fire once, after the idempotent apply. To
land the kill inside the txn window (not the baseline-insert window), the E6
crash cells target child-process hit numbers 6 and 7 (5 baseline inserts +
1 txn apply). This is recorded per-cell in raw config; the boundary labels in
`e6-crash.csv` are `gate: after_apply` / `gate: before_ack`.

## D3 — async legality at post-WAL windows

At `tx_after_commit_wal` / `after_apply` / `before_ack` / post-ACK under
`async`, both `ABSENT_ATOMIC` and `PRESENT_ATOMIC` are legal outcomes (the
append is buffered; a group kill may lose it). This is A2's ACK≠machine-
durability boundary, not a new weakening: sync/group remain strictly PRESENT
at those windows and strictly ABSENT before the commit record. The generator
and checker gate 23 encode exactly this legal-state map.

## D4 — AbortTxn WAL kind is never emitted; rollback is logical only

The audit (spec §audit) found `AbortTxn` in the WAL record enum, never
produced by any code path; rollback discards the staged in-memory txn and
never writes a WAL record. E6 documents this as-is (rollback = logical
discard; uncommitted end-of-log groups discarded at replay) rather than
adding a new abort record — no WAL format reinterpretation.

## D5 — In-transaction update/upsert is UNSUPPORTED by type

`TxnOp = Insert | Delete` only. E6 does not add in-txn update/upsert (that
would be silent scope invention). The E6k cell classifies the *standalone*
update (exists-only, uuid-preserving, fields-map replaced wholesale, old
numeric id retired) and upsert (exists ? update : insert, both branches), and
records the in-txn gap as UNSUPPORTED. Generator + gate 23 fail if the E6k
row ever loses the UNSUPPORTED marker (unsupported-PASS guard).

## D6 — E6k exported-version semantics

`export_state` reports `version` from the **fields map**; `update_document`
replaces the fields map wholesale, so the exported version reads 0 after an
update (the durable `Record.version` increment is not exported). The E6k
expectation therefore asserts observable semantics (field values num/cat,
uuid preservation, old-id retirement) instead of the exported version number.
Durable version behavior remains covered by E3/E4-era checks on the record
path.

## D7 — Corruption cells must run before any `close()`; torn-tail source dir

`close()` checkpoints and trims the WAL, leaving no multi-record group to
tear. The E6s cells therefore commit and `drop()` the engine without close
(sync appends are already fsynced), then copy the directory and corrupt the
largest WAL segment (≥64 B asserted). The torn-tail cell reuses the directory
from the `tx_before_commit_wal` crash cell (a real partial group at
groupkill), placed AFTER the crash families in the driver for dependency
ordering. Both cells verified: torn tail → whole group discarded atomically;
garbled committed segment → open REFUSED (E1 policy unchanged).

## D8 — CSV column encodings for state-machine cells

Two cells encode multiple sub-results positionally in the fixed CSV columns
(the harness writes a fixed 9-column schema):
- `illegal-transitions`: `commit_status=c1=true`, `checker_clean=c2=false`,
  `restart_ok=r=false`, `model_match=c3=false` = double-commit ok; the three
  illegal attempts (double-commit, rollback-after-commit, commit-after-
  rollback) all returned false = no-ops.
- `staged-then-ckpt`: `observed_state=false` + `commit_status=NEVER-COMMITTED`
  + notes "checkpoint never..." = staged txn absent after checkpoint.
The generator (E6_TXN_CELLS + per-case rules) and gate 23 pin these encodings;
unregistered encodings fail loudly.

## D9 — Phase-2 filter test corrected to the documented two-valued NOT contract (test fix; engine semantics UNCHANGED)

Root-caused during E6 triage of intermittent CI failures (reproduced ~10% on
clean cdfd282 via stash test — pre-existing, not caused by E6):
`core/tests/phase2_filtering.rs::type_mismatch_and_missing_never_match`
asserted that `Or(cat="x", NOT cat="x")` excludes documents **missing**
`category`. The filter module's documented contract since baseline
(`query/src/filter.rs` header: "NOT(...) composes the two-valued result … a
deliberate, documented departure from SQL three-valued logic") makes that
expression a tautology: `NOT(cmp)` matches missing-field documents. The
module is internally consistent with its sibling `In { negated }` rule only
under its own docs; the TEST was the outlier. Retrieval semantics are frozen,
so **the test was corrected, not the code**: the missing-exclusion intent is
now expressed via the documented `IsNotNull` operator (`AND(IsNotNull(category),
Or(...))`), and 4 deterministic unit tests pinning the two-valued semantics
were added to `query/src/filter.rs` (`not_matches_missing_two_valued`,
`not_in_missing_never_matches`, `is_null_presence_operators`,
`missing_field_comparisons_all_false`). The 100%-precision invariant (§13) is
unaffected: with the documented semantics, no returned document is
non-matching. An alternative (Kleene/three-valued NOT) was considered and
rejected as a frozen-surface semantics change.

## D10 — Exact-recall test asserts replaced with contract-true subset/isolation asserts (test fix; engine UNCHANGED)

Two further intermittent failures root-caused to asserts demanding **exact
recall** from the approximate ANN (hnsw_rs draws per-insert layer assignments
from an RNG; graph shape and borderline scores vary run to run):
- `phase1_persistence::t07_multi_collection` asserted `attend(...).len() == 5`
  on a 5-doc collection (observed 4 under an unlucky graph);
- `query::executor::tests::test_execute_basic` asserted `ids.len() == 2` where
  one doc sits at the `min_weight` cutoff (observed 1).
Exact recall is not part of the ANN contract (and claiming it would violate
the honesty rules). The asserts now check the deterministic content: only
indexed/in-collection ids are returned, no cross-collection leakage
(t07: every returned id ∈ alpha's docs), non-empty where the contract
guarantees it, and the exact-match doc always returned (executor).
Storage-count asserts (total_vectors) were already deterministic and are
unchanged.

## D11 — Flake-probe artifact removed

The temporary reproduction harness (`core/tests/e6_flake_probe.rs`, ~2400
looped iterations + 8 concurrent-with-suite rounds, 0 failures standalone —
evidence that the flakes were recall/entropy-dependent, not load-dependent)
was removed after diagnosis; its role is filled by the deterministic unit
tests of D9.

## Verification of the fixes

- phase2_filtering 30×, phase1_persistence 15×, query crate 30×: 0 failures.
- Full workspace suite ×5: 320 passed, 0 FAILED (previously ~2/8 runs flaked).
- clippy `-D warnings`: clean.
- E1–E5 regressions after the fixes: PH3E-WAL-007 ≡ WAL-001 (byte),
  PH3E-DUR-010 ≡ DUR-001/002/003/004 (byte), PH3E-E3-004 ≡ E3-001 (byte),
  PH3E-BACKUP-006 ≡ BACKUP-004 (integrity byte + matrix classification),
  PH3E-COMPACT-005 ≡ COMPACT-004 (classification) — all identical, proving
  the fixes touched test code only.
