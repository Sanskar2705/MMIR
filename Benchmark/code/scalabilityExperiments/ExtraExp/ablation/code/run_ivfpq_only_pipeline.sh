#!/usr/bin/env bash
set -euo pipefail

CODE_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(cd "$CODE_DIR/.." && pwd)"
PY="/mnt/storage/RSystemsBenchmarking/envs/39myenv/bin/python"

mkdir -p "$BASE_DIR/results" "$BASE_DIR/evaluation" "$BASE_DIR/logs" \
  "$BASE_DIR/index/ivfpq_best" "$BASE_DIR/index/hnsw_best"

echo "[INFO] Step 1/3: Build IVFPQ-only indexes (OPQ+IVF+PQ via Faiss index_factory)"
"$PY" "$CODE_DIR/build_high_recall_indexes.py" \
  --hnsw-configs "" \
  --ivfpq-out-dir "$BASE_DIR/index/ivfpq_best" \
  --ivfpq-configs "" \
  --ivfpq-factory-specs "OPQ192,IVF16384,PQ192|OPQ192,IVF32768,PQ192|OPQ256,IVF16384,PQ256|OPQ256,IVF32768,PQ256" \
  --ivfpq-train-size 1300000

echo "[INFO] Step 2/3: Run IVFPQ-only retrieval sweep"
"$PY" "$CODE_DIR/batch_retrieve_ablation_sweep.py" \
  --hnsw_dir "$BASE_DIR/index/hnsw_best" \
  --ivfpq_dir "$BASE_DIR/index/ivfpq_best" \
  --results_dir "$BASE_DIR/results" \
  --queries_file "$BASE_DIR/results/fixed_queries_1000.json" \
  --max_queries 1000 \
  --top_k 100 \
  --run-ivfpq \
  --ivfpq-nprobe "1024,2048,4096,8192,16384,32768" \
  --skip-ivfpq-nprobe ""

echo "[INFO] Step 3/3: Evaluate IVFPQ runs only (overlap vs flat KNN)"
"$PY" "$CODE_DIR/structured_evaluation.py" \
  --results-dir "$BASE_DIR/results" \
  --knn-file "clip_image_flat.json" \
  --evaluation-dir "$BASE_DIR/evaluation" \
  --index-types "ivfpq" \
  --accuracy-csv "top100_overlap_accuracy_ivfpq.csv" \
  --ranks-csv "top100_overlap_ranks_ivfpq.csv"

echo "[DONE] IVFPQ-only pipeline completed."
