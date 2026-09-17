# Durability Validation (Phase 3D)

## Claim
Acknowledged-write durability is a function of the explicitly selected durability mode, and
the engine recovers from every tested failure point exactly as its contract states: all
acknowledged writes present (Sync/GroupCommit), a proper acknowledged prefix with zero
unacknowledged leakage (Async), and all-or-nothing transactions at every point.

## Method
Acked-sidecar crash harness (`PH3D-CRASH-001..003`): the driver's child process performs 40
inserts, writing a flushed `ACK i` line to a sidecar file after each engine ack, then reaches
one of seven crash points (self-exit with `exit(137)` — Drop skipped — for terminal points;
parked and SIGKILLed for mid-stream points). A verifier reopens the directory, derives the
expected set from the sidecar, checks presence/prefix/no-resurrection (scan acked..acked+64),
runs the engine+directory consistency gate, and additionally validates a 10-insert
transaction's all-or-nothing outcome against a post-commit marker. 7 points × 3 durability
modes = 21 points. Graceful-shutdown durability (write → ack → close() → reopen → verify)
for insert/update/delete/upsert is covered by `PH3D-INTEGRATION-001`.

## Results
21/21 crash points consistent with the documented contract. GroupCommit and Sync: ALL_ACKED
everywhere; committed transactions durable (10/10). Async: proper prefixes (35/40, 11/21)
while unflushed, ALL_ACKED after flush/checkpoint; a committed-but-unflushed transaction
vanished after its commit returned (`COMMITTED_NOT_DURABLE_ASYNC`) — atomic (0/10), never
partial. Graceful close preserved all mutation kinds (PH3D-INTEGRATION-001 16/16).

## What "durable" means here (precise, §47-compliant)
- GroupCommit/Sync: an acknowledged mutation survives process death of the database process.
- Sync additionally fsyncs per append; the machine-crash (power-loss) axis is implemented
  but was not harnessed → durability for Sync is verified on the process axis only.
- Async: an acknowledgment does NOT imply process-crash durability; flush_wal/checkpoint
  are the promotion points.

## Limitations
Process-crash axis only for group/async; crash points are API-boundary-granular, not
instruction-level; no device-level error injection (ENOSPC/fsync failure).
