| Configuration | Docs | Dimension | Heads | Peak RSS | DB Size | Build Time | Retrieval p50 | Correctness | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| primary (001) | 80000 | 32 | 1 | 1381616 KB | 56008259 | 104.4 s | 492 us | OBSERVED_LIMIT | OBSERVED_LIMIT |
| primary (002) | 100000 | 32 | 1 | NA KB | NA | NA s | NA us | OBSERVED_LIMIT | OBSERVED_LIMIT |
| primary (003) | 150000 | 32 | 1 | NA KB | NA | NA s | NA us | OBSERVED_LIMIT | OBSERVED_LIMIT |
| primary (004) | 200000 | 32 | 1 | NA KB | NA | NA s | NA us | OBSERVED_LIMIT | OBSERVED_LIMIT |
| primary (005) | 40000 | 32 | 1 | 699872 KB | 27989953 | 38.2 s | 356 us | green | VERIFIED |
| slim (006) | 120000 | 32 | 1 | NA KB | NA | NA s | NA us | OBSERVED_LIMIT | OBSERVED_LIMIT |
| slim (007) | 60000 | 32 | 1 | 994568 KB | 40299640 | 73.9 s | 475 us | green | VERIFIED |
| primary-2head (008) | 40000 | 32 | 2 | 833324 KB | NA | 89.1 s | 634 us | SUPERSEDED | SUPERSEDED |
| primary-4head (009) | 40000 | 32 | 4 | 1371304 KB | NA | 172.2 s | 1215 us | SUPERSEDED | SUPERSEDED |
| primary-2head (012) | 40000 | 32 | 2 | 561076 KB | NA | 85.0 s | 580 us | green | VERIFIED |
| primary-4head (013) | 40000 | 32 | 4 | 862272 KB | NA | 166.4 s | 1372 us | green | VERIFIED |
| dim128 (010) | 20000 | 128 | 1 | 281228 KB | NA | 26.4 s | 475 us | green | VERIFIED |
| dim256 (011) | 20000 | 256 | 1 | 353140 KB | NA | 40.4 s | 683 us | green | VERIFIED |
| primary-churn (014) | 60k+200k ops | 32 | 1 | 877184 KB | NA | 70.7 s | 611 us | INVALIDATED | INVALIDATED |
| primary-churn (018) | 60k+200k ops | 32 | 1 | 914932 KB | NA | 70.7 s | 655 us | green | VERIFIED |
| primary-maint (016) | 60k-6k del | 32 | 1 | 957052 KB | NA | 61.9 s | 497 us | green | VERIFIED |
| primary-conc (015) | 130k attempted | 32 | 1 | NA KB | NA | NA s | NA us | OBSERVED_LIMIT | OBSERVED_LIMIT |
| primary-conc (019) | 40k+20k | 32 | 1 | 1072544 KB | NA | NA s | NA us | green | VERIFIED |
| primary-integrated (020) | 60k lifecycle | 32 | 1 | 1082496 KB | NA | 59.7 s | NA us | green | VERIFIED |
| primary-retrieval (017) | 80000 | 32 | 1 | NA KB | NA | 94.1 s | NA us | OBSERVED_LIMIT | OBSERVED_LIMIT |
| primary-retrieval (021) | 80000 | 32 | 1 | NA KB | NA | 93.1 s | NA us | green | VERIFIED |
