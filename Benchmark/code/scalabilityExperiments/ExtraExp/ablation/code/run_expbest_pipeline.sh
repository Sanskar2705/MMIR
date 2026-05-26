#!/usr/bin/env bash
set -euo pipefail

CODE_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(cd "$CODE_DIR/.." && pwd)"
PY="/mnt/storage/RSystemsBenchmarking/envs/39myenv/bin/python"

mkdir -p "$BASE_DIR/results" "$BASE_DIR/evaluation" "$BASE_DIR/logs"

echo "[INFO] Step 1/3: Build high-recall indexes"
"$PY" "$CODE_DIR/build_high_recall_indexes.py" \
  --hnsw-out-dir "$BASE_DIR/index/hnsw_best" \
  --ivfpq-out-dir "$BASE_DIR/index/ivfpq_best" \
  --hnsw-configs "64:800,80:1200" \
  --ivfpq-configs "16384:96:8,32768:96:8" \
  --ivfpq-train-size 1200000

echo "[INFO] Step 2/3: Run retrieval sweep for best indexes"
"$PY" "$CODE_DIR/batch_retrieve_ablation_sweep.py" \
  --hnsw_dir "$BASE_DIR/index/hnsw_best" \
  --ivfpq_dir "$BASE_DIR/index/ivfpq_best" \
  --results_dir "$BASE_DIR/results" \
  --queries_file "$BASE_DIR/results/fixed_queries_1000.json" \
  --max_queries 1000 \
  --top_k 100 \
  --hnsw-efsearch "1024,2048,4096" \
  --ivfpq-nprobe "512,1024,2048,4096" \
  --skip-hnsw-efsearch "" \
  --skip-ivfpq-nprobe ""

echo "[INFO] Step 3/3: Evaluate everything with structured evaluator"
"$PY" "$CODE_DIR/structured_evaluation.py" \
  --results-dir "$BASE_DIR/results" \
  --knn-file "clip_image_flat.json" \
  --evaluation-dir "$BASE_DIR/evaluation" \
  --accuracy-csv "top100_overlap_accuracy_structured.csv" \
  --ranks-csv "top100_overlap_ranks_structured.csv"

echo "[DONE] ExpBest pipeline completed."
