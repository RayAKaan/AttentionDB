# AttentionDB Benchmark Datasets

## ann-benchmarks.com HDF5 Datasets

Download standard ANN benchmark datasets for publication-grade evaluation:

```bash
bash datasets/download.sh
```

This downloads the following datasets from [ann-benchmarks.com](http://ann-benchmarks.com):

| Dataset | Vectors | Dimension | Distance | Size |
|---------|---------|-----------|----------|------|
| SIFT1M | 1,000,000 train / 10,000 test | 128 | Euclidean | ~512 MB |
| GloVe-100 | 1,183,514 train / 10,000 test | 100 | Cosine | ~900 MB |
| NYTimes-256 | 290,000 train / 10,000 test | 256 | Cosine | ~600 MB |
| GIST1M | 1,000,000 train / 10,000 test | 960 | Euclidean | ~3.8 GB |

## Custom Data Format

For custom datasets, prepare HDF5 files with the ann-benchmarks layout:

- `/train` — float32 `[N, D]` — corpus vectors
- `/test` — float32 `[Q, D]` — query vectors
- `/neighbors` — int32 `[Q, 100]` — ground truth (top-100 for each query)
- `/distances` — float32 `[Q, 100]` — distances to ground truth

## Dataset Citation

If using these datasets in a publication, cite:

- Aumüller, M., Bernhardsson, E., and Faithfull, A. (2020).
  "ANN-Benchmarks: A Benchmarking Tool for Approximate Nearest Neighbor Algorithms."
  Information Systems, 87, p.101374.
