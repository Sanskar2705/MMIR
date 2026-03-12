#!/usr/bin/env python3
"""
Build FAISS IVF-PQ (IP / cosine) index from UniIR image embeddings
(STREAMING, RAM-safe)

- Uses IVF-PQ: approximate + compressed
- Trains IVF-PQ on a sample of vectors, then adds all vectors streaming.

Outputs:
  <BASE_DIR>/quantization/index/uniir_image_ivfpq.index
  <BASE_DIR>/quantization/index/image_names.json
"""

import os
import json
import faiss
import numpy as np
from tqdm import tqdm

# ================= PATHS =================

EMB_DIR = (
    "/mnt/storage/RSystemsBenchmarking/gitProject/Benchmark/data/"
    "UniIR10M/uniir/embeddings"
)

BASE_DIR = (
    "/mnt/storage/RSystemsBenchmarking/gitProject/Benchmark/data/"
    "testing/testing1/uniir"
)

OUT_DIR = os.path.join(BASE_DIR, "quantization", "index")
INDEX_PATH = os.path.join(OUT_DIR, "uniir_image_ivfpq.index")
NAMES_PATH = os.path.join(OUT_DIR, "image_names.json")

os.makedirs(OUT_DIR, exist_ok=True)

# ================= IVF-PQ PARAMS =================
# Since you said embedding dim = 768 and data size is large (10M)

NLIST = 4096      # IVF clusters
M = 16            # PQ subquantizers (768 % 16 == 0 )
NBITS = 8         # bits per subquantizer
NPROBE = 16       # clusters searched at query time
TRAIN_MAX = 200000  # how many vectors to train on (100k–500k typical)

# ================= MAIN =================

def main():
    files = sorted(
        f for f in os.listdir(EMB_DIR)
        if f.startswith("embeddings_part_") and f.endswith(".json")
    )

    print(f"[INFO] Found {len(files)} embedding files")
    print(f"[INFO] IVF-PQ params: nlist={NLIST}, m={M}, nbits={NBITS}, nprobe={NPROBE}")
    print(f"[INFO] Training sample max: {TRAIN_MAX}")

    # ---------- PASS 1: collect training vectors ----------
    train_vecs = []
    total_seen = 0
    dim = None

    print("[INFO] Collecting vectors for training IVF-PQ...")

    for fname in tqdm(files, desc="Training sample collection"):
        path = os.path.join(EMB_DIR, fname)
        with open(path, "r") as f:
            data = json.load(f)

        embs = np.asarray([row["embedding"] for row in data], dtype="float32")

        if dim is None:
            dim = embs.shape[1]
            print(f"[INFO] Detected embedding dim = {dim}")

        # safety check: normalized
        norms = np.linalg.norm(embs, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError(f"Embeddings not normalized in {fname}")

        # add to training pool until TRAIN_MAX
        remaining = TRAIN_MAX - total_seen
        if remaining > 0:
            train_vecs.append(embs[:remaining])
            total_seen += min(len(embs), remaining)

        del embs, data

        if total_seen >= TRAIN_MAX:
            break

    train_x = np.vstack(train_vecs)
    print(f"[INFO] Training vectors collected: {train_x.shape[0]:,} x {train_x.shape[1]}")

    # ---------- build IVF-PQ index ----------
    print("[INFO] Building IVF-PQ index...")

    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFPQ(
        quantizer,
        dim,
        NLIST,
        M,
        NBITS,
        faiss.METRIC_INNER_PRODUCT
    )

    print("[INFO] Training IVF-PQ index (this may take time)...")
    index.train(train_x)

    index.nprobe = NPROBE
    print("[INFO] IVF-PQ training complete ")

    # ---------- PASS 2: stream-add all vectors ----------
    print("[INFO] Adding all vectors to IVF-PQ index (streaming)...")

    image_names = []
    total_added = 0

    for fname in tqdm(files, desc="Indexing IVF-PQ"):
        path = os.path.join(EMB_DIR, fname)
        with open(path, "r") as f:
            data = json.load(f)

        names = [row["image_name"] for row in data]
        embs = np.asarray([row["embedding"] for row in data], dtype="float32")

        # safety check
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

    # ---------- save index + mapping ----------
    print("[INFO] Saving IVF-PQ index...")
    faiss.write_index(index, INDEX_PATH)

    print("[INFO] Saving image-name mapping...")
    with open(NAMES_PATH, "w") as f:
        json.dump(image_names, f)

    assert index.ntotal == len(image_names)

    print("=" * 60)
    print("[DONE] UniIR IVF-PQ index built successfully ")
    print(f"Index: {INDEX_PATH}")
    print(f"Names: {NAMES_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()

