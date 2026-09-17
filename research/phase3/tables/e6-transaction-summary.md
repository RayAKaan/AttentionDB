| family | case | mode | commit status | checker | restart | model | match |
|---|---|---|---|---|---|---|---|
| E6a-basic | t1-single-insert | sync | COMMITTED | true/true | ok | true/true | MATCH |
| E6a-basic | t2-multi-insert | sync | COMMITTED | true/true | ok | true/true | MATCH |
| E6a-basic | t3-insert-delete | sync | COMMITTED | true/true | ok | true/true | MATCH |
| E6a-basic | t4-multi-delete | sync | COMMITTED | true/true | ok | true/true | MATCH |
| E6a-basic | t5-empty-txn | sync | COMMITTED | true/true | ok | true/true | MATCH |
| E6b-rollback | rollback-invisible | sync | ROLLED_BACK | true/true | ok | true/true | MATCH |
| E6b-rollback | illegal-transitions | sync | c1=true | c2=false | r=false | c3=false | MATCH |
| E6c-multiop | n-ops-5 | sync | COMMITTED | true/true | ok | true/true | MATCH |
| E6c-multiop | n-ops-10 | sync | COMMITTED | true/true | ok | true/true | MATCH |
| E6c-multiop | n-ops-50 | sync | COMMITTED | true/true | ok | true/true | MATCH |
| E6c-multiop | n-ops-100 | sync | COMMITTED | true/true | ok | true/true | MATCH |
| E6d-multi-txn | commit-rollback-commit | sync | T1+T3 COMMITTED T2 ROLLED_BACK | true/true | ok | true/true | MATCH |
| E6i-ordering | commit-order-wins | sync | T2-after-T1=T3-after-T2 | true/true | ok | true/true | MATCH |
| E6i-ordering | delete-missing-noop | sync | COMMITTED | true | ok/- | true/- | MATCH |
| E6j-concurrency | 3-staged-txns | group | all committed (out of stage order) | true/true | ok | true/true | MATCH |
| E6k-update-upsert | update-upsert-semantics | sync | OK | true/true | ok | true/true | MATCH |
| E6l-same-key | ins-ins-del-ins-txn | sync | COMMITTED | true/true | ok | true/true | MATCH |
| E6m-tombstones | txn-ins-txn-del-compact | sync | 2 tombs reclaimed | true | ok | true | MATCH |
| E6m-tombstones | staged-delete-rollback | sync | ROLLED_BACK | true | ok | true/true | MATCH |
| E6n-compaction | commit-then-compact | sync | COMMITTED | true/true | ok | true/true | MATCH |
| E6n-compaction | compact-then-commit | sync | COMMITTED | true/true | ok | true/true | MATCH |
| E6o-checkpoint | txn-then-ckpt | sync | COMMITTED | true | ok | true/true | MATCH |
| E6o-checkpoint | ckpt-then-txn | sync | COMMITTED | true | ok | true/true | MATCH |
| E6o-checkpoint | staged-then-ckpt | sync | NEVER-COMMITTED | true | ok | false/false | MATCH |
| E6o-checkpoint | commit-ckpt-restart | sync | COMMITTED | true | ok | true/true | MATCH |
| E6p-rotation | 100-op-txn-2KiB-segments | group | COMMITTED | true/true | ok | true/true | MATCH |
| E6q-backup | before-during-after | sync | 3 backups | true/true/true | ok | true/true | MATCH |
| E6r-idempotence | restart-x3 | sync | COMMITTED | true/true/true | ok | true | MATCH |
| E6s-corruption | garbled-committed-group | sync | - | - | - | corruption never silently converted to a valid txn (E1 policy) | MATCH |
| E6s-corruption | torn-tail-partial-group | sync | DISCARDED | true | ok | true | MATCH |

Crash windows (fresh-process recovery, groupkill at gate):
| boundary | mode | aborted at boundary | txn state | atomicity | checker | model | match |
|---|---|---|---|---|---|---|---|
| pre-commit staging | sync | true | ABSENT | ATOMIC | clean | MATCH | MATCH |
| pre-commit staging | sync | true | ABSENT | ATOMIC | clean | MATCH | MATCH |
| pre-commit staging | group | true | ABSENT | ATOMIC | clean | MATCH | MATCH |
| gate: tx_before_commit_wal | sync | true | ABSENT_ATOMIC | ATOMIC | clean | MATCH | MATCH |
| gate: tx_before_commit_wal | group | true | ABSENT_ATOMIC | ATOMIC | clean | MATCH | MATCH |
| gate: tx_before_commit_wal | async | true | ABSENT_ATOMIC | ATOMIC | clean | MATCH | MATCH |
| gate: tx_after_commit_wal | sync | true | PRESENT_ATOMIC | ATOMIC | clean | MATCH | MATCH |
| gate: tx_after_commit_wal | group | true | PRESENT_ATOMIC | ATOMIC | clean | MATCH | MATCH |
| gate: tx_after_commit_wal | async | true | ABSENT_ATOMIC | ATOMIC | clean | MATCH | MATCH |
| gate: after_apply | sync | true | PRESENT_ATOMIC | ATOMIC | clean | MATCH | MATCH |
| gate: after_apply | group | true | PRESENT_ATOMIC | ATOMIC | clean | MATCH | MATCH |
| gate: after_apply | async | true | ABSENT_ATOMIC | ATOMIC | clean | MATCH | MATCH |
| gate: before_ack | sync | true | PRESENT_ATOMIC | ATOMIC | clean | MATCH | MATCH |
| gate: before_ack | group | true | PRESENT_ATOMIC | ATOMIC | clean | MATCH | MATCH |
| gate: before_ack | async | true | ABSENT_ATOMIC | ATOMIC | clean | MATCH | MATCH |
| abort after ACK | sync | true | PRESENT_ATOMIC | ATOMIC | clean | MATCH | MATCH |
| abort after ACK | group | true | PRESENT_ATOMIC | ATOMIC | clean | MATCH | MATCH |
| abort after ACK | async | true | ABSENT_ATOMIC | ATOMIC | clean | MATCH | MATCH |
| T1 committed; T3 staged only | sync | true | T1_PRESENT_T3_ABSENT | ATOMIC | clean | MATCH | MATCH |

Source: results/e6-transactions.csv + results/e6-crash-atomicity.csv (generated from raw/runs/PH3E-TXN-001 — no manual transcription).
