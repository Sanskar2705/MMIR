#!/usr/bin/env python3
"""
Build FAISS HNSW index from CLIP image embeddings (STREAMING, RAM-safe)
"""

import os
import json
import faiss
import numpy as np
from tqdm import tqdm

# ================= PATHS =================

EMB_DIR = (
    "/mnt/storage/RSystemsBenchmarking/gitProject/Benchmark/data/"
    "CLIP10M/clip/embeddings"
)

BASE_DIR = (
    "/mnt/storage/RSystemsBenchmarking/gitProject/Benchmark/data/"
    "testing/testing1/clip"
)

OUT_DIR = os.path.join(BASE_DIR, "index")
INDEX_PATH = os.path.join(OUT_DIR, "clip_image_hnsw.index")
NAMES_PATH = os.path.join(OUT_DIR, "image_names.json")

os.makedirs(OUT_DIR, exist_ok=True)

# ================= HNSW PARAMS =================

HNSW_M = 32
EF_CONSTRUCTION = 200

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

    for fname in tqdm(files, desc="Indexing (HNSW)"):
        with open(os.path.join(EMB_DIR, fname), "r") as f:
            data = json.load(f)

        names = [row["image_name"] for row in data]
        embs = np.asarray([row["embedding"] for row in data], dtype="float32")

        if index is None:
            dim = embs.shape[1]
            index = faiss.IndexHNSWFlat(dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = EF_CONSTRUCTION
            print(
                f"[INFO] IndexHNSWFlat initialized "
                f"(dim={dim}, M={HNSW_M}, efC={EF_CONSTRUCTION})"
            )

        norms = np.linalg.norm(embs, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError(f"Embeddings not normalized in {fname}")

        index.add(embs)
        image_names.extend(names)
        total_added += len(names)

        del embs, names, data

    print(f"[INFO] Total indexed images: {total_added:,}")

    faiss.write_index(index, INDEX_PATH)
    with open(NAMES_PATH, "w") as f:
        json.dump(image_names, f)

    assert index.ntotal == len(image_names)

    print("[DONE] CLIP HNSW index built")
    print("Index:", INDEX_PATH)
    print("Names:", NAMES_PATH)

if __name__ == "__main__":
    main()
