#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/mnt/storage/RSystemsBenchmarking/gitProject/Benchmark/data/testing/laionRetriveal/extra_exp/code/indexing_code"
OUTPUT_DIR="/mnt/storage/RSystemsBenchmarking/gitProject/Benchmark/data/testing/laionRetriveal/extra_exp/output_testing1_clip"
MYENV="${MYENV:-/mnt/storage/RSystemsBenchmarking/envs/39myenv}"
PYTHON="${PYTHON:-$MYENV/bin/python}"

export PATH="$MYENV/bin:$PATH"
cd "$BASE_DIR"

echo "[STEP 1/3] Building HNSW index..."
"$PYTHON" build_indexes32.py

echo "[STEP 2/3] Building IVFPQ index..."
"$PYTHON" build_indexes.py

echo "[STEP 3/3] Running retrieval for 1000 fixed queries (top-100)..."
"$PYTHON" run_retrieval_eval.py --output_root "$OUTPUT_DIR"

echo "[DONE] Outputs under: $OUTPUT_DIR/{hnsw,ivfpq}"
