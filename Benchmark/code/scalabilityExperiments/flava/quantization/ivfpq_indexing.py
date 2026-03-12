#!/usr/bin/env python3
"""
Build FAISS IVF-PQ inner-product index from FLAVA image embeddings (STREAMING, RAM-safe).

- Trains IVF-PQ on a sampled subset (RAM-safe)
- Adds all vectors streaming shard-by-shard
- Saves:
    flava_image_ivfpq.index
    image_names.json
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

OUT_DIR = os.path.join(BASE_DIR, "quantization")
INDEX_PATH = os.path.join(OUT_DIR, "flava_image_ivfpq.index")
NAMES_PATH = os.path.join(OUT_DIR, "image_names.json")

os.makedirs(OUT_DIR, exist_ok=True)

# ================= IVF-PQ PARAMS =================
NLIST = 4096          # IVF clusters
M = 16                # PQ subquantizers (bytes per vector when nbits=8)
NBITS = 8             # bits per subquantizer
NPROBE = 16           # search-time clusters to scan
TRAIN_SIZE = 200000   # vectors sampled for training (reduce if RAM tight)

# ================= HELPERS =================

def iter_embedding_files():
    return sorted(
        f for f in os.listdir(EMB_DIR)
        if f.startswith("embeddings_part_") and f.endswith(".json")
    )

def load_shard(path):
    with open(path, "r") as f:
        data = json.load(f)
    names = [row["image_name"] for row in data]
    embs = np.asarray([row["embedding"] for row in data], dtype="float32")
    return names, embs

def assert_normalized(embs, atol=1e-3):
    norms = np.linalg.norm(embs, axis=1)
    if not np.allclose(norms, 1.0, atol=atol):
        raise ValueError("Embeddings are not L2-normalized (cosine/IP needs normalized vectors).")

def sample_training_vectors(files, dim):
    """
    Collect TRAIN_SIZE vectors across shards (streaming).
    """
    train_vecs = []
    collected = 0

    for fname in tqdm(files, desc="Sampling train vectors"):
        shard_path = os.path.join(EMB_DIR, fname)
        _, embs = load_shard(shard_path)

        assert embs.shape[1] == dim
        assert_normalized(embs)

        remaining = TRAIN_SIZE - collected
        if remaining <= 0:
            break

        take = min(remaining, embs.shape[0])
        train_vecs.append(embs[:take])
        collected += take

        del embs

    if collected == 0:
        raise RuntimeError("Could not collect training vectors for IVF-PQ training.")

    return np.vstack(train_vecs)

# ================= MAIN =================

def main():
    files = iter_embedding_files()
    print(f"[INFO] Found {len(files)} embedding files")

    # ---- Step 1: get dim from first shard ----
    first_path = os.path.join(EMB_DIR, files[0])
    _, first_embs = load_shard(first_path)

    dim = first_embs.shape[1]
    assert_normalized(first_embs)
    del first_embs

    print(f"[INFO] dim={dim}")

    # ---- Step 2: build IVF-PQ skeleton ----
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFPQ(
        quantizer,
        dim,
        int(NLIST),
        int(M),
        int(NBITS),
        faiss.METRIC_INNER_PRODUCT
    )
    index.nprobe = int(NPROBE)

    # ---- Step 3: train IVF-PQ ----
    print(f"[INFO] Training IVF-PQ with TRAIN_SIZE={TRAIN_SIZE:,} vectors...")
    train_x = sample_training_vectors(files, dim)
    print(f"[INFO] Training matrix shape: {train_x.shape}")

    index.train(train_x)
    del train_x
    print("[INFO] Training done ")

    # ---- Step 4: add all vectors streaming ----
    image_names = []
    total_added = 0

    for fname in tqdm(files, desc="Adding vectors"):
        shard_path = os.path.join(EMB_DIR, fname)
        names, embs = load_shard(shard_path)

        assert embs.shape[1] == dim
        assert_normalized(embs)

        index.add(embs)
        image_names.extend(names)
        total_added += len(names)

        del embs, names

        if total_added % 1_000_000 == 0:
            print(f"[INFO] Indexed {total_added:,} images")

    print(f"[INFO] Total indexed images: {total_added:,}")
    assert index.ntotal == len(image_names)

    # ---- Step 5: save ----
    faiss.write_index(index, INDEX_PATH)
    with open(NAMES_PATH, "w") as f:
        json.dump(image_names, f)

    print("=" * 60)
    print("[DONE] FLAVA IVF-PQ index built successfully ")
    print(f"Index: {INDEX_PATH}")
    print(f"Names: {NAMES_PATH}")
    print(f"IVF-PQ params: nlist={NLIST}, m={M}, nbits={NBITS}, nprobe={NPROBE}")
    print("=" * 60)

if __name__ == "__main__":
    main()
