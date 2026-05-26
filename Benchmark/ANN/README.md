# ANN benchmark (COCO / Flickr30k)

End-to-end pipeline for **approximate nearest-neighbor (ANN)** image retrieval from text queries, using FAISS indexes (**HNSW**, **IVFPQ**, **O-IVFPQ**). Results are compared to exact **flat KNN**, then optionally **BLIP2-ITM reranked** and scored with **recall@k** against query ground truth.

Designed for small corpora: **COCO Karpathy test** (~5k images) and **Flickr30k test** (~1k images).

## What this folder is for

| Stage | Goal |
|-------|------|
| Index build | Turn precomputed image embeddings into FAISS indexes (one config per file). |
| Retrieval | Encode test captions → search index → top-100 JSON per query. |
| Overlap eval | Measure how well ANN top-100 matches flat KNN top-100 (tune ANN params). |
| Reranking | Re-score top-100 candidates with BLIP2 image–text matching. |
| Recall eval | Compare reranked lists to ground-truth query images (recall@1/5/10). |

Python entry points live in **`Code/`** — see [Code/README.md](Code/README.md) for each script and run order.

## Encoders

| Encoder | Corpus vectors | Query side |
|---------|----------------|------------|
| `clip_image` | CLIP image embeddings | CLIP text |
| `openclip_image` | OpenCLIP image embeddings | OpenCLIP text |
| `uniir_joint` | UniIR joint image+text embeddings | UniIR text |

Embeddings are read from `Benchmark/data/embeddings/` (paths in `Code/configs.py`).

## Folder layout

```
ANN/
├── Code/                 # Python pipeline (build, retrieve, eval, rerank)
├── Index/                # Built FAISS indexes + names/meta JSON
│   ├── HNSW/
│   ├── IVFPQ/
│   └── OIVFPQ/
├── KNN/                  # Flat exact-search reference JSONs
├── HNSW/                 # ANN retrieval JSONs (per index config)
├── IVFPQ/
├── OIVFPQ/
├── Reranker/             # BLIP2-reranked JSONs
│   ├── HNSW/
│   ├── IVFPQ/
│   └── OIVFPQ/
├── Result/               # CSV summaries (overlap, reranker recall)
├── logs/                 # Logs from batch / rerank runs
└── run_all_batches.sh    # Main orchestrator (steps 1–6, no rerank)
```

### Artifacts (not in `Code/`)

- **`Index/{HNSW,IVFPQ,OIVFPQ}/{encoder}_{dataset}/`** — `.index`, `*_names.json`, `*_meta.json`
- **`{HNSW,IVFPQ,OIVFPQ}/`** — `results_{encoder}_{dataset}_{index_type}_{config}__0.json`
- **`KNN/`** — `results_{encoder}_{dataset}_knn_flat__0.json` (ground truth for overlap)
- **`Reranker/`** — same stems with `_rerank-blip2.json`
- **`Result/overlap_summary_all.csv`** — ANN vs KNN overlap (input job list for reranking)
- **`Result/reranker/reranker_recall_summary.csv`** — recall after reranking

## Pipeline overview

```text
embeddings (precomputed)
        │
        ▼
  [1] build_indexes.py  ──►  Index/
        │
        ▼
  [2] run_knn.py        ──►  KNN/          (exact reference)
  [3] run_retrieval.py  ──►  HNSW/ IVFPQ/ OIVFPQ/
        │
        ▼
  [4] compute_overlap_csv.py  ──►  Result/overlap_summary_all.csv
        │
        ▼
  [5] blip2_rerank_batch.py   ──►  Reranker/
        │
        ▼
  [6] evaluate_reranker_recall.py  ──►  Result/reranker/
```

Steps **1–4** are covered by `run_all_batches.sh`. Steps **5–6** need the BLIP2 environment (see `Code/README.md`).

## Quick start

From the Benchmark repo root:

```bash
export PYTHONPATH=/mnt/storage/RSystemsBenchmarking/gitProject
ENV_PYTHON=/mnt/storage/RSystemsBenchmarking/envs/39myenv/bin/python

# Full batch: build → KNN → HNSW/IVFPQ/OIVFPQ retrieve → overlap CSV
cd Benchmark/code/ANN
bash run_all_batches.sh
```

Or run scripts individually — order and flags are documented in **`Code/README.md`**.

### Reranking (optional, separate env)

```bash
# After overlap_summary_all.csv exists
bash run_blip2_rerank.sh          # single GPU, all index types
# or
bash run_blip2_parallel.sh        # 3 screen workers (HNSW / IVFPQ / OIVFPQ)

$ENV_PYTHON Benchmark/code/ANN/Code/evaluate_reranker_recall.py
```

## Shell helpers (repo root under `ANN/`)

| Script | Purpose |
|--------|---------|
| `run_all_batches.sh` | Steps 1–4 for all encoders/datasets/index types |
| `run_knn_batches.sh` | KNN reference only |
| `run_blip2_rerank.sh` | BLIP2 rerank using `Result/overlap_summary_all.csv` |
| `run_blip2_parallel.sh` | Parallel rerank (3 GPUs / screens) |
| `run_reranker_worker.sh` | One index type per worker |
| `run_expann.sh` | Experiment launcher (project-specific) |

LowerConfig / reverse-worker scripts are legacy experiment helpers; the supported path is the main pipeline above.

## Parameter grids

Tuned for small **N** (see `Code/configs.py`):

- **COCO:** HNSW M ∈ {16, 32, 48}; IVFPQ nlist ∈ {64, 128, 256}; O-IVFPQ IVF ∈ {64, 128, 256}
- **Flickr:** HNSW M ∈ {16, 32, 48}; IVFPQ nlist ∈ {32, 64, 128}; O-IVFPQ IVF ∈ {32, 64, 128}

Each grid point is one index file and one retrieval JSON.

## Git / storage notes

- **Commit:** `Code/`, shell scripts, small CSVs under `Result/`
- **Do not commit:** `Index/**/*.index`, large retrieval JSON trees, `logs/` — regenerate locally or use shared storage
- Embeddings and raw images live under `Benchmark/data/` and `data/datasets/` (outside this folder)

## Related docs

- Per-script usage and CLI: **[Code/README.md](Code/README.md)**
- Global Benchmark config (models, paths): `Benchmark/config/config.yaml`
