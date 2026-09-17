# Recovery Validation (Phase 3D)

## Claim
After every tested termination and every tested WAL corruption, the database either reopens
to a contract-consistent state (intact prefix, all acknowledged mutations under the selected
durability mode) or refuses to open with a reported error — never silently fabricating data
and never partially applying a transaction.

## Method
Three legs. (1) WAL corruption surgery (`PH3D-WALCORRUPT-001`): on an isolated 40-document
database, truncate the final segment to 60%, append garbage bytes, flip an in-frame byte,
delete the only segment, and duplicate a segment under a gapped name — reopen and report
(open/clean/refused), never repairing. (2) Crash recovery (`PH3D-CRASH-001..003`, 21 points):
reopen after each crash point, compare against the sidecar ack set, run the consistency
gate. (3) Replay equivalence (`PH3D-STATE-001..003`, `PH3D-INTEGRATION-001`): restart after
mixed mutation streams — observed state must equal the reference model exactly (≈30
restart/compaction gates per 1000-op run; graceful close→reopen in INTEGRATION).

## Results
- Torn tail → truncated with WARNING; intact prefix recovers (23 docs, checker clean).
- Corrupt frame (garbage tail, byte flip) → open REFUSES (`ERROR_ON_OPEN`).
- Gapped segment name (claims seq 999, holds seq 1) → open REFUSES (continuity check).
- Pre-checkpoint segment deletion → opens as an EMPTY database with a clean checker:
  structurally undetectable (the catalog stores no WAL high-water mark). Documented OPEN
  limitation; no mitigation attempted.
- Crash recovery: 21/21 contract-consistent, zero resurrection, zero partial transactions.
- Replay: exact logical state after every restart gate; transactions replay all-or-nothing.

## Fuzz coverage (§34)
Deterministic parser fuzz added as preserved regression tests: 64 random-byte segments and
32 truncations of a valid WAL (`replay_fuzz_*` in storage tests) — replay never panics;
truncated replays yield strict sequence prefixes or errors. Filter validate/eval fuzz
(256 random expression trees + over-deep rejection) in query tests.

## Limitations
Corruption is driver-performed surgery (no mid-rotation tearing); the undetectable
pre-checkpoint deletion remains OPEN (would require a WAL high-water mark in the manifest).
