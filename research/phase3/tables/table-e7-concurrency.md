# E7 concurrency families — PH3E-CONC-002 (42 MATCH + 2 UNSUPPORTED + 4 crash ATOMIC)

| family | case | mode | threads | txns | observed | model | checker | status |
|---|---|---|---|---|---|---|---|---|
| E7a-readers | threads-1 | sync | 1 | 0 | errors=0 kinds(attend_err;attend_bad_id;scan_len;scan_err;get_fields)=0;0;0;0;0 bad-ids=[] | static-120-docs | clean | MATCH |
| E7a-readers | threads-2 | sync | 2 | 0 | errors=0 kinds(attend_err;attend_bad_id;scan_len;scan_err;get_fields)=0;0;0;0;0 bad-ids=[] | static-120-docs | clean | MATCH |
| E7a-readers | threads-4 | sync | 4 | 0 | errors=0 kinds(attend_err;attend_bad_id;scan_len;scan_err;get_fields)=0;0;0;0;0 bad-ids=[] | static-120-docs | clean | MATCH |
| E7a-readers | threads-8 | sync | 8 | 0 | errors=0 kinds(attend_err;attend_bad_id;scan_len;scan_err;get_fields)=0;0;0;0;0 bad-ids=[] | static-120-docs | clean | MATCH |
| E7a-readers | threads-16 | sync | 16 | 0 | errors=0 kinds(attend_err;attend_bad_id;scan_len;scan_err;get_fields)=0;0;0;0;0 bad-ids=[] | static-120-docs | clean | MATCH |
| E7b-reader-writer | 2r-1w-flips | sync | 3 | 0 | values-seen={1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 11; 12; 13; 15; 16; 17; 19; 20; 21; 22; 23; 25; 27; 28; 30; 31; 32 | final-num=40 | clean | MATCH |
| E7e-atomic-visibility | hot-reader-during-commits | sync | 2 | 30 |  per-doc-monotonic=YES mixed-pairs=0 | per-doc-atomic-monotonic | clean | MATCH |
| E7f-del-ins | hot-reader-during-commit | sync | 2 | 30 |  A0B0=3 A0B1=33 A1B0=1 per-doc-monotonic=0 mixed-pairs=3 | per-doc-monotonic-only | clean | MATCH |
| E7g-write-write | same-key-t1-then-t2 | sync | 2 | 2 | final=Some(200) expect=200 | later-commit-wins-1-doc/order-honored | clean | MATCH |
| E7g-write-write | same-key-t2-then-t1 | sync | 2 | 2 | final=Some(100) expect=100 | later-commit-wins-1-doc/order-honored | clean | MATCH |
| E7g-write-write | same-key-t1-then-t2 | group | 2 | 2 | final=Some(200) expect=200 | later-commit-wins-1-doc/order-honored | clean | MATCH |
| E7g-write-write | same-key-t2-then-t1 | group | 2 | 2 | final=Some(100) expect=100 | later-commit-wins-1-doc/order-honored | clean | MATCH |
| E7g-write-write | same-key-t1-then-t2 | async | 2 | 2 | final=Some(200) expect=200 | later-commit-wins-1-doc/order-honored | clean | MATCH |
| E7g-write-write | same-key-t2-then-t1 | async | 2 | 2 | final=Some(100) expect=100 | later-commit-wins-1-doc/order-honored | clean | MATCH |
| E7g-write-write | plain-2-writers | sync | 2 | 0 | acked=20 docs=20 | 20-survive | clean | MATCH |
| E7h-lost-update | read-outside-blind-write | sync | 2 | 2 | both-committed final=Some(2) expect=2 | commit-order-wins-no-detection | clean | MATCH |
| E7i-write-skew | classic-invariant | sync | 2 | 2 | not-expressible | UNSUPPORTED-NO-TXN-READS | - | - |
| E7j-phantom | predicate-requery-in-txn | sync | 2 | 2 | not-expressible | UNSUPPORTED-NO-TXN-QUERY | - | - |
| E7k-staged-visibility | stage-barrier-read-commit | sync | 2 | 30 | staged-new-visible=0 old-missing-while-staged=0 post-commit-missing=0 | absent-then-present | clean | MATCH |
| E7l-rollback-visibility | stage-barrier-rollback | sync | 2 | 20 | ever-visible=0 | never-visible | clean | MATCH |
| E7m-commit-visibility | watch-flip-during-commit | sync | 2 | 40 | window-observed=0 noticed-after-ack=39 early-retire=0 new-pre-ack=0 new-after-ack=39 unclassified=1 stability- | apply-point-visibility | clean | MATCH |
| E7n-del-reinsert | both-orders-compact-restart | sync | 2 | 4 | 6300=v2/42 6301=v3/43 | uuid-v2-v3-live-old-tombstoned | clean | MATCH |
| E7o-collections | T-A-T-B-concurrent-plus-ops | sync | 2 | 0 | A=[6400] B=[6401] | no-contamination | clean | MATCH |
| E7p-checkpoint | staged-then-concurrent-ckpt | sync | 2 | 1 | staged-visible-at-ckpt=0 | ckpt-never-commits | clean | MATCH |
| E7q-compaction | commits-vs-compact-loop | sync | 2 | 5 | compactions=4 docs=5 | no-partial-no-lost | clean | MATCH |
| E7r-backup | txn-during-staged | sync | 2 | 1 | snapshot=pre | pre-or-post-never-partial | clean | MATCH |
| E7r-backup | txn-during-commit | sync | 2 | 1 | snapshot=post | pre-or-post-never-partial | clean | MATCH |
| E7s-concurrent-staging | 3-stage-barrier-commit-3-1-2 | sync | 3 | 3 | distinct-ids=true docs=9 | WAL=commit-order | clean | MATCH |
| E7t-commit-contention | 4-barrier-commits | sync | 4 | 4 | finish-order=[3; 2; 1; 0] | all-committed-serialized | clean | MATCH |
| E7t-commit-contention | 4-barrier-commits | group | 4 | 4 | finish-order=[0; 1; 2; 3] | all-committed-serialized | clean | MATCH |
| E7t-commit-contention | 4-barrier-commits | async | 4 | 4 | finish-order=[1; 0; 3; 2] | all-committed-serialized | clean | MATCH |
| E7u-linearizability | register-realtime-history | sync | 4 | 0 | reads=1851 P1-future=0 P2-stale=0 P3-unknown=0 absent-window=1 | single-register-subset | clean | MATCH |
| E7v-serializability | blind-write-hist-seed-7 | sync | 3 | 3 | commit-order=[2; 0; 1] final-len=2 | history=serial-by-construction | clean | MATCH |
| E7v-serializability | blind-write-hist-seed-13 | sync | 3 | 3 | commit-order=[1; 2; 0] final-len=2 | history=serial-by-construction | clean | MATCH |
| E7w-schedule-enumeration | sched-0 | sync | 2 | 2 | A=2 B=2 | all-or-nothing | clean | MATCH |
| E7w-schedule-enumeration | sched-1 | sync | 2 | 2 | A=1 B=1 | all-or-nothing | clean | MATCH |
| E7w-schedule-enumeration | sched-2 | sync | 2 | 2 | A=2 B=2 | all-or-nothing | clean | MATCH |
| E7w-schedule-enumeration | sched-3 | sync | 2 | 2 | A=1 B=1 | all-or-nothing | clean | MATCH |
| E7x-randomized | seed-11 | sync | 2 | 3 | idx-len=4 record-len=4 bad-samples=0 dup-idx=0 dup-idx-replay=0 | commit-order-model | clean | MATCH |
| E7x-randomized | seed-23 | sync | 2 | 3 | idx-len=7 record-len=7 bad-samples=0 dup-idx=2 dup-idx-replay=2 | commit-order-model | clean | MATCH |
| E7x-randomized | seed-37 | sync | 2 | 3 | idx-len=7 record-len=7 bad-samples=0 dup-idx=2 dup-idx-replay=2 | commit-order-model | clean | MATCH |
| E7x-randomized | seed-52 | sync | 2 | 3 | idx-len=8 record-len=8 bad-samples=0 dup-idx=3 dup-idx-replay=3 | commit-order-model | clean | MATCH |
| E7x-randomized | seed-68 | sync | 2 | 3 | idx-len=5 record-len=5 bad-samples=0 dup-idx=0 dup-idx-replay=0 | commit-order-model | clean | MATCH |
| E7x-randomized | seed-84 | sync | 2 | 3 | idx-len=6 record-len=6 bad-samples=0 dup-idx=1 dup-idx-replay=1 | commit-order-model | clean | MATCH |
| E7y-crash | t1-t2-before | sync | - | 2 | T1_PRESENT_T2_ABSENT @ gate (aborted=true) | ATOMIC | clean | MATCH |
| E7y-crash | t1-t2-before | group | - | 2 | T1_PRESENT_T2_ABSENT @ gate (aborted=true) | ATOMIC | clean | MATCH |
| E7y-crash | t1-t2-after | sync | - | 2 | T1_PRESENT_T2_PRESENT @ gate (aborted=true) | ATOMIC | clean | MATCH |
| E7y-crash | t1-t2-after | group | - | 2 | T1_PRESENT_T2_PRESENT @ gate (aborted=true) | ATOMIC | clean | MATCH |
