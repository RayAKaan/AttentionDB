# C4.5 — Confirmed-Scale Statistical Analysis

Derived exclusively from immutable raw per-query rows (`raw/C4-W0*/*/artifacts/RUN-repN.json`). Unit = query (paired across modes by the identical seeded randomized order).

Method (preregistered, C1 `statistical-plan.md`): per-query recall@10 = mean over 5 fresh-process reps; paired bootstrap 10,000 (seed 20260925) 95% CI on the mean paired difference; two-sided Wilcoxon signed-rank; Holm step-down within family; Cohen's dz; |diff| < 0.01 absolute = negligible regardless of p-value.

## SCI (n=300)

- B1: recall@10 mean=0.7783 (n=300), p50 latency=1020.9 us
- B2: recall@10 mean=0.7920 (n=300), p50 latency=1912.1 us
- B3: recall@10 mean=0.7911 (n=300), p50 latency=1901.4 us
- B4: recall@10 mean=0.7918 (n=300), p50 latency=1953.5 us
- B7: recall@10 mean=0.7421 (n=300), p50 latency=2166.8 us

## NFC (n=323)

- B1: recall@10 mean=0.1491 (n=323), p50 latency=947.7 us
- B2: recall@10 mean=0.1597 (n=323), p50 latency=1694.9 us
- B3: recall@10 mean=0.1593 (n=323), p50 latency=1782.3 us
- B7: recall@10 mean=0.1488 (n=323), p50 latency=2035.4 us

- B3-vs-B4: **NOT-EXECUTABLE** — B4 absent from frozen per-NFC TEST plan (documented no-op arm scoped to SciFact; never interpolated).

## SCI — paired contrasts (Holm-adjusted)

- B1-vs-B2: B1-vs-B2 mean diff=-0.0136 [-0.0417,+0.0143] (paired, 300 queries, 43 discordant); Wilcoxon p=0.1763 (Holm-adj p=0.7051); Cohen's dz=-0.055; |diff|<0.01 -> NOT negligible.
  verdict: n.s. (B1=0.7783 B2=0.7920)
  power note: min detectable mean recall diff (discordance 43/300) ~0.050 at 1-beta=0.8.

- B2-vs-B3: B2-vs-B3 mean diff=+0.0008 [-0.0031,+0.0052] (paired, 300 queries, 11 discordant); Wilcoxon p=0.7875 (Holm-adj p=1.0000); Cohen's dz=0.023; |diff|<0.01 -> NEGLIGIBLE.
  verdict: n.s. (B2=0.7920 B3=0.7911)
  power note: min detectable mean recall diff (discordance 11/300) ~0.027 at 1-beta=0.8.

- B3-vs-B4: B3-vs-B4 mean diff=-0.0007 [-0.0034,+0.0021] (paired, 300 queries, 7 discordant); Wilcoxon p=0.7335 (Holm-adj p=1.0000); Cohen's dz=-0.028; |diff|<0.01 -> NEGLIGIBLE.
  verdict: n.s. (B3=0.7911 B4=0.7918)
  power note: min detectable mean recall diff (discordance 7/300) ~0.022 at 1-beta=0.8.

- B2-vs-B7: B2-vs-B7 mean diff=+0.0498 [+0.0257,+0.0764] (paired, 300 queries, 42 discordant); Wilcoxon p=0.0001 (Holm-adj p=0.0004); Cohen's dz=0.223; |diff|<0.01 -> NOT negligible.
  verdict: SIGNIFICANT (B2=0.7920 B7=0.7421)

## NFC — paired contrasts (Holm-adjusted)

- B1-vs-B2: B1-vs-B2 mean diff=-0.0105 [-0.0222,+0.0000] (paired, 323 queries, 154 discordant); Wilcoxon p=0.0917 (Holm-adj p=0.4584); Cohen's dz=-0.104; |diff|<0.01 -> NOT negligible.
  verdict: n.s. (B1=0.1491 B2=0.1597)
  power note: min detectable mean recall diff (discordance 154/323) ~0.069 at 1-beta=0.8.

- B2-vs-B3: B2-vs-B3 mean diff=+0.0003 [-0.0013,+0.0019] (paired, 323 queries, 60 discordant); Wilcoxon p=0.8081 (Holm-adj p=1.0000); Cohen's dz=0.021; |diff|<0.01 -> NEGLIGIBLE.
  verdict: n.s. (B2=0.1597 B3=0.1593)
  power note: min detectable mean recall diff (discordance 60/323) ~0.054 at 1-beta=0.8.

- B2-vs-B7: B2-vs-B7 mean diff=+0.0109 [+0.0024,+0.0210] (paired, 323 queries, 126 discordant); Wilcoxon p=0.0022 (Holm-adj p=0.0133); Cohen's dz=0.126; |diff|<0.01 -> NOT negligible.
  verdict: SIGNIFICANT (B2=0.1597 B7=0.1488)


## Primary-family table

| dataset | contrast | verdict | evidence |
|---------|----------|---------|----------|
| NFC | B3-vs-B4 | NOT-EXECUTABLE | no frozen TEST cell (B4 frozen for SciFact only) |
| SCI | B1-vs-B2 | n.s. | diff=-0.0136 [-0.0417,+0.0143] p_holm=0.7051 |
| SCI | B2-vs-B3 | n.s. | diff=+0.0008 [-0.0031,+0.0052] p_holm=1.0000 |
| SCI | B3-vs-B4 | n.s. | diff=-0.0007 [-0.0034,+0.0021] p_holm=1.0000 |
| SCI | B2-vs-B7 | SIGNIFICANT | diff=+0.0498 [+0.0257,+0.0764] p_holm=0.0004 |
| NFC | B1-vs-B2 | n.s. | diff=-0.0105 [-0.0222,+0.0000] p_holm=0.4584 |
| NFC | B2-vs-B3 | n.s. | diff=+0.0003 [-0.0013,+0.0019] p_holm=1.0000 |
| NFC | B2-vs-B7 | SIGNIFICANT | diff=+0.0109 [+0.0024,+0.0210] p_holm=0.0133 |


## Honesty disclosures

- NFCorpus B3-vs-B4 recorded as NOT-EXECUTABLE (frozen plan scoped B4, the documented no-op arm, to SciFact only). It is **not** a failed cell and is never interpolated.
- Per-query recall is mean over 5 fresh-process reps because the C4.4 engine shows intrinsic run-to-run variance (sd ~0.0006-0.026 per cell). Single-rep recall is engineered to be deterministic, but fresh-process execution is not bit-stable; averaging within config before pairing is the preregistered-compatible stable estimator.

Headline no-winner-labels recall@10 (5-rep means):

| dataset | B1 | B2 | B3 | B4 | B7 |
|---------|----|----|----|----|----|
| SCI | 0.7783 | 0.7920 | 0.7911 | 0.7918 | 0.7421 |
| NFC | 0.1491 | 0.1597 | 0.1593 | — | 0.1488 |
