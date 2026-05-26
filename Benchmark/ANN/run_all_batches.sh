#!/usr/bin/env bash
# Full ANN pipeline: build indexes -> KNN reference -> ANN retrieval -> overlap eval
# Run inside screen: screen -S Exp_ANN bash run_all_batches.sh

set -euo pipefail

ENV_PYTHON="/mnt/storage/RSystemsBenchmarking/envs/39myenv/bin/python"
PROJECT_ROOT="/mnt/storage/RSystemsBenchmarking/gitProject"
ANN_CODE="${PROJECT_ROOT}/Benchmark/code/ANN/Code"
LOG_DIR="${PROJECT_ROOT}/Benchmark/code/ANN/logs"
mkdir -p "$LOG_DIR"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

run_step() {
  local name="$1"
  shift
  log "START: ${name}"
  "$ENV_PYTHON" "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"
  log "DONE: ${name}"
}

# ── Batch 1: Build all FAISS indexes (CPU) ─────────────────────────────
log "=== BATCH 1: Build indexes ==="
run_step "build_indexes_all" \
  "${ANN_CODE}/build_indexes.py" \
  --encoder all --dataset all --index-type all --faiss-threads 8

# ── Batch 2: Flat KNN reference (GPU for query encoding) ───────────────
log "=== BATCH 2: KNN reference retrieval ==="
for enc in clip_image openclip_image uniir_joint; do
  for ds in coco flickr; do
    run_step "knn_${enc}_${ds}" \
      "${ANN_CODE}/run_knn.py" \
      --encoder "$enc" --dataset "$ds" \
      --top-k 100 --encode-batch-size 128 --device cuda
  done
done

# ── Batch 3: HNSW retrieval ────────────────────────────────────────────
log "=== BATCH 3: HNSW retrieval ==="
for enc in clip_image openclip_image uniir_joint; do
  for ds in coco flickr; do
    run_step "hnsw_${enc}_${ds}" \
      "${ANN_CODE}/run_retrieval.py" \
      --encoder "$enc" --dataset "$ds" --index-type hnsw \
      --top-k 100 --encode-batch-size 128 --device cuda
  done
done

# ── Batch 4: IVFPQ retrieval ───────────────────────────────────────────
log "=== BATCH 4: IVFPQ retrieval ==="
for enc in clip_image openclip_image uniir_joint; do
  for ds in coco flickr; do
    run_step "ivfpq_${enc}_${ds}" \
      "${ANN_CODE}/run_retrieval.py" \
      --encoder "$enc" --dataset "$ds" --index-type ivfpq \
      --top-k 100 --encode-batch-size 128 --device cuda
  done
done

# ── Batch 5: O-IVFPQ retrieval ─────────────────────────────────────────
log "=== BATCH 5: O-IVFPQ retrieval ==="
for enc in clip_image openclip_image uniir_joint; do
  for ds in coco flickr; do
    run_step "oivfpq_${enc}_${ds}" \
      "${ANN_CODE}/run_retrieval.py" \
      --encoder "$enc" --dataset "$ds" --index-type oivfpq \
      --top-k 100 --encode-batch-size 128 --device cuda
  done
done

# ── Batch 6: Overlap evaluation vs KNN ─────────────────────────────────
log "=== BATCH 6: Overlap evaluation ==="
run_step "compute_overlap_csv" \
  "${ANN_CODE}/compute_overlap_csv.py" \
  --encoder all --index-type all --top-n 100

log "=== ALL BATCHES COMPLETE ==="
