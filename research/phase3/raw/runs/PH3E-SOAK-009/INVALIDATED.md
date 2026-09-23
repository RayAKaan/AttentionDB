# PH3E-SOAK-009 — INVALIDATED (harness defect, engine exonerated)

Family i (integrated 600s soak). Run completed 600s / 25,098 txns / 125,492 ops but:
1. Maintenance cadence never fired (ckpt=0, verify=0): the seq-modulo checks sit after
   the every-5-ops txn block which adds +2 to seq; every maintenance multiple (3000,
   5000, 9000, 15000, 20000) is also a multiple of 5, so the +2 always skipped it.
2. Reader violations = 218,051/440,962: the seq%20000 restart rebound the engine Arc
   WHILE the long-running reader still held a clone — the reader then polled a closed
   engine for the rest of the run. Violates E8's own restart constraint (never rebind
   under a live reader).
3. The txn path could stage two keys from different collection blocks into keys[0]'s
   collection (membership mis-tagging).

Engine behavior was never at fault (no refuses, checker clean on the live engine).
Artifacts retained for the record. Retried with the fixed harness as PH3E-SOAK-010.
