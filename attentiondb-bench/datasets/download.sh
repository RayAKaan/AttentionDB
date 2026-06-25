#!/bin/bash
# Download ann-benchmarks.com datasets for AttentionDB benchmarking
# Usage: bash datasets/download.sh

set -euo pipefail

DATASETS_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$DATASETS_DIR"

declare -A DATASETS
DATASETS["sift-128-euclidean.hdf5"]="http://ann-benchmarks.com/sift-128-euclidean.hdf5"
DATASETS["glove-100-angular.hdf5"]="http://ann-benchmarks.com/glove-100-angular.hdf5"
DATASETS["nytimes-256-angular.hdf5"]="http://ann-benchmarks.com/nytimes-256-angular.hdf5"
DATASETS["gist-960-euclidean.hdf5"]="http://ann-benchmarks.com/gist-960-euclidean.hdf5"

for filename in "${!DATASETS[@]}"; do
    url="${DATASETS[$filename]}"
    path="$DATASETS_DIR/$filename"

    if [ -f "$path" ]; then
        echo "✓ $filename already exists, skipping"
        continue
    fi

    echo "↓ Downloading $filename..."
    if command -v wget &> /dev/null; then
        wget -q --show-progress "$url" -O "$path"
    elif command -v curl &> /dev/null; then
        curl -L --progress-bar "$url" -o "$path"
    else
        echo "Error: need wget or curl to download datasets"
        exit 1
    fi

    echo "  Downloaded to $path"
done

echo ""
echo "All datasets downloaded to $DATASETS_DIR"
ls -lh "$DATASETS_DIR"/*.hdf5 2>/dev/null || echo "(no hdf5 files found)"
