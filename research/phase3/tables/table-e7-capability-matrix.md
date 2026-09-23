# E7 capability matrix (23 rows) — verdicts derived from PH3E-CONC-002 raw status columns

| capability | verdict | evidence |
|---|---|---|
| Concurrent readers | VERIFIED | E7a 5 thread-counts, 0 errors, 0 gate blocks |
| Readers + writer | VERIFIED | E7b committed-only values, never torn |
| Concurrent writers | VERIFIED | E7g 6 cells + E7t 3 modes; mutation-gate serialized |
| Transaction staging concurrency | VERIFIED | E7s 3 concurrent stagers, distinct ids |
| Commit serialization | VERIFIED | E7t completion orders + E7v commit-order replay |
| Dirty reads | VERIFIED-ABSENT | E7b census + E7k ordered: no uncommitted value ever read |
| Atomic transaction visibility | PARTIAL | E7e/E7f: per-doc atomic; mixed pairs observable (NO per-txn snapshot) |
| Repeated-read behavior | PARTIAL | single-version keys: repeatable by construction; no txn snapshot (E7e) |
| Same-key conflict behavior | VERIFIED | E7g later-commit-wins, both orders x 3 modes |
| Lost-update behavior | VERIFIED-OCCURS | E7h both commits Ok, one write silently lost |
| Write skew | UNSUPPORTED | E7i: TxnOp = Insert|Delete, no txn reads |
| Phantom behavior | UNSUPPORTED | E7j: no transactional query API |
| Rollback visibility | VERIFIED-ABSENT | E7l 20 ordered reps, ever-visible=0 |
| Commit visibility | VERIFIED | E7m: visibility = apply point; 0 post-ACK stability violations |
| Collection isolation | VERIFIED | E7o zero contamination across backup/ckpt/compact |
| Checkpoint concurrency | VERIFIED | E7p ckpt inside staged window never commits/exposes |
| Compaction concurrency | VERIFIED | E7q commit x compact serialize; no resurrection |
| Backup concurrency | VERIFIED | E7r pre-or-post snapshot, never partial |
| Single-operation linearizability | PARTIAL | E7u P1/P2/P3 clean on point-register subset; not system-wide |
| Transaction serializability | PARTIAL | E7v blind-write histories serial by construction; general claim untestable |
| Formal isolation level | NOT CLAIMED | no read txns -> no ANSI level expressible or claimable |
| MVCC | UNSUPPORTED | single visible version per uuid; no snapshots |
| Conflict detection | UNSUPPORTED | E7g/E7h: no version checks, no aborts, commit-order-wins |
