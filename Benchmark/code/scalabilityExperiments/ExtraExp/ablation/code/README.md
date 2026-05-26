# Ablation code

CLIP text-to-image retrieval ablation on ~7M LAION embeddings: build indexes → retrieve top-100 → measure overlap vs flat KNN.

## Pipeline (run in order)

| Step | Script | Purpose |
|------|--------|---------|
| 0 | `batch_retrieve_knn_7m.py` | Flat KNN reference (`../results/clip_image_flat.json`) |
| 1 | `../build_ablation_indexes.py` | Grid HNSW + IVFPQ indexes under `../index/` |
| 1b | `build_high_recall_indexes.py` | High-recall `hnsw_best` / `ivfpq_best` (+ OPQ factory IVFPQ) |
| 2 | `batch_retrieve_ablation_sweep.py` | Sweep efSearch / nprobe → JSON under `../results/` |
| 3 | `structured_evaluation.py` | Overlap accuracy CSVs → `../evaluation/` |
| 4 | `build_master_evaluation_log.py` | Aggregate runs into `../evaluation/final/` |

## Orchestrators

- **`run_expbest_pipeline.sh`** — build best indexes → sweep → structured eval (main path).
- **`run_ivfpq_only_pipeline.sh`** — OPQ+IVF+PQ factory indexes only → IVFPQ sweep → eval.
- **`run_ablation_sweep.sh`** — retrieval sweep only (indexes must already exist).

## Core modules

- **`batch_retrieve_ablation.py`** — CLIP query encoding + FAISS search; used by sweep and KNN scripts.
- **`batch_retrieve_ablation_sweep.py`** — nested efSearch / nprobe sweep over index directories.
- **`batch_retrieve_knn_7m.py`** — exact flat IP search for ground-truth top-100 lists.
