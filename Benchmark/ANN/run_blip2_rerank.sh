#!/usr/bin/env bash
# Direct in-process BLIP2-ITM rerank (batched GPU). No new HTTP service.
# Uses py3_9_neuralReranker1_0 (same stack as screen 59419.reranker).
#
# screen -S ExpBLIP2 -dm bash /mnt/storage/RSystemsBenchmarking/gitProject/Benchmark/code/ANN/run_blip2_rerank.sh

set -euo pipefail

ENV_PYTHON="${BLIP2_PYTHON:-/home/bharati/anaconda3/envs/py3_9_neuralReranker1_0/bin/python}"
PROJECT_ROOT="/mnt/storage/RSystemsBenchmarking/gitProject"
CODE="${PROJECT_ROOT}/Benchmark/code/ANN/Code"
ANN_ROOT="${PROJECT_ROOT}/Benchmark/code/ANN"
CSV="${ANN_ROOT}/Result/overlap_summary_all.csv"
LOG_DIR="${ANN_ROOT}/logs"

mkdir -p "$LOG_DIR" "${ANN_ROOT}/Reranker/HNSW" "${ANN_ROOT}/Reranker/IVFPQ" "${ANN_ROOT}/Reranker/OIVFPQ"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export LAVIS_ROOT="${LAVIS_ROOT:-/mnt/storage/bharati/Projects/RSystems/NeuralReranker/BLIP_family/LAVIS}"
cd "$PROJECT_ROOT"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

for idx in hnsw ivfpq oivfpq; do
  log "BLIP2 direct rerank: ${idx}"
  "$ENV_PYTHON" "${CODE}/blip2_rerank_batch.py" \
    --csv "$CSV" \
    --ann-root "$ANN_ROOT" \
    --index-type "$idx" \
    --top-k 100 \
    --image-batch-size 32 \
    --device cuda \
    --skip-existing \
    2>&1 | tee "${LOG_DIR}/blip2_rerank_${idx}.log"
done

log "ExpBLIP2 complete"
