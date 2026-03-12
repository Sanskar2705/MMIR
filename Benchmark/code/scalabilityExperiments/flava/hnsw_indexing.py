#!/usr/bin/env python3
"""
Build FAISS HNSW index from FLAVA image embeddings (STREAMING, RAM-safe)
"""

import os
import json
import faiss
import numpy as np
from tqdm import tqdm

# ================= PATHS =================

EMB_DIR = (
    "/mnt/storage/RSystemsBenchmarking/gitProject/Benchmark/data/"
    "Flava10M/flava/embeddings"
)

BASE_DIR = (
    "/mnt/storage/RSystemsBenchmarking/gitProject/Benchmark/data/"
    "testing/testing1/flava"
)

OUT_DIR = os.path.join(BASE_DIR, "index")
INDEX_PATH = os.path.join(OUT_DIR, "flava_image_hnsw.index")
NAMES_PATH = os.path.join(OUT_DIR, "image_names.json")

os.makedirs(OUT_DIR, exist_ok=True)

# ================= HNSW PARAMS =================

HNSW_M = 32          # graph degree (16–48 typical)
EF_CONSTRUCTION = 200  # build-time accuracy/speed tradeoff

# ================= MAIN =================

def main():
    files = sorted(
        f for f in os.listdir(EMB_DIR)
        if f.startswith("embeddings_part_") and f.endswith(".json")
    )

    print(f"[INFO] Found {len(files)} embedding files")

    index = None
    image_names = []
    total_added = 0

    for fname in tqdm(files, desc="Indexing embeddings"):
        path = os.path.join(EMB_DIR, fname)

        with open(path, "r") as f:
            data = json.load(f)

        names = [row["image_name"] for row in data]
        embs = np.asarray([row["embedding"] for row in data], dtype="float32")

        if index is None:
            dim = embs.shape[1]
            index = faiss.IndexHNSWFlat(dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = EF_CONSTRUCTION
            print(
                f"[INFO] FAISS HNSW initialized "
                f"(dim={dim}, M={HNSW_M}, efC={EF_CONSTRUCTION})"
            )

        # safety: embeddings must be normalized
        norms = np.linalg.norm(embs, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError(f"Embeddings not normalized in {fname}")

        index.add(embs)
        image_names.extend(names)
        total_added += len(names)

        del embs, names, data

        if total_added % 1_000_000 == 0:
            print(f"[INFO] Indexed {total_added:,} images")

    print(f"[INFO] Total indexed images: {total_added:,}")

    print("[INFO] Saving FAISS HNSW index...")
    faiss.write_index(index, INDEX_PATH)

    print("[INFO] Saving image name mapping...")
    with open(NAMES_PATH, "w") as f:
        json.dump(image_names, f)

    assert index.ntotal == len(image_names)

    print("=" * 60)
    print("[DONE] FAISS HNSW index built successfully")
    print(f"Index: {INDEX_PATH}")
    print(f"Names: {NAMES_PATH}")
    print("=" * 60)

if __name__ == "__main__":
    main()
