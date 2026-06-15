# Recall Benchmark Guide for AttentionDB HNSW

This guide explains how to run high-quality recall benchmarks on real embedding data using the improved benchmark in `recall_bench.rs`.

## Why This Benchmark Matters

Most HNSW implementations are evaluated on synthetic or toy data. This benchmark is designed for serious research and supports:

- Real embedding datasets (`.fvecs` format)
- Multiple graph quality configurations in a single run
- Insert-time normalization (critical for real data)
- Professional JSON output for analysis and papers

## Quick Start (Real Data)

### 1. Download a Real Embedding Dataset

**Recommended: GloVe (Word Embeddings)**

```bash
mkdir -p data
cd data
wget https://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
```

**Alternative: SIFT (Image Descriptors)**

```bash
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xzf sift.tar.gz
```

### 2. Convert to `.fvecs` Format (if needed)

GloVe comes in `.txt` format. Convert it using this Python script:

Save as `convert_to_fvecs.py`:

```python
import struct
import numpy as np
import sys

def convert_glove(input_path, output_path, max_vectors=100000):
    print(f"Converting {input_path} -> {output_path}")
    with open(input_path, 'r', encoding='utf-8') as f_in, open(output_path, 'wb') as f_out:
        count = 0
        for line in f_in:
            if count >= max_vectors:
                break
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            dim = len(vector)
            f_out.write(struct.pack('i', dim))
            f_out.write(vector.tobytes())
            count += 1
    print(f"Converted {count} vectors.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_to_fvecs.py <input.txt> <output.fvecs> [max_vectors]")
        sys.exit(1)
    max_vec = int(sys.argv[3]) if len(sys.argv) > 3 else 100000
    convert_glove(sys.argv[1], sys.argv[2], max_vec)
```

Run it:

```bash
python convert_to_fvecs.py data/glove.6B.300d.txt data/glove-100k.fvecs 100000
```

### 3. Run the Benchmark

```bash
cargo bench --bench recall_bench -p attentiondb-hnsw
```

Results will be printed in the terminal and saved to `recall_results.json`.

## Understanding the Output

The benchmark tests multiple graph quality levels:

| Config Name  | max_nb_connection | ef_construction | Best For               |
|--------------|------------------|----------------|------------------------|
| Balanced     | 16               | 400            | Fast indexing, decent recall |
| HighQuality  | 32               | 600            | Good balance           |
| MaxQuality   | 48               | 800            | Highest recall (slower build) |

## Best Practices

### For Research / Papers
- Use the **MaxQuality** configuration
- Run with at least 300 queries
- Always enable `use_normalization`
- Save and commit the `recall_results.json` file

### For Performance Testing
- Use the **Balanced** configuration
- Focus on `ef=64` and `ef=128`
- Measure p99 latency

### For Production Tuning
- Start with **HighQuality** settings and adjust based on your recall vs latency requirements

## Troubleshooting

| Problem                 | Solution                                           |
|-------------------------|----------------------------------------------------|
| `dim` mismatch error    | Make sure `dim` in config matches your dataset     |
| Low recall on real data | Increase `ef_construction` and `max_nb_connection` |
| Out of memory           | Reduce `dataset_size` or `max_elements`            |
| Slow benchmark          | Reduce `num_queries` or `ef_values`                |

## Next Steps

After running the benchmark on real data:

1. Compare results across the three graph configurations
2. Tune parameters further if needed
3. Move on to building an end-to-end demo using the query and multihead modules
