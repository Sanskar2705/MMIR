#!/usr/bin/env python3
import json
import os
import tarfile
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np


def iter_embedding_files(emb_dir: str) -> List[str]:
    files = sorted(
        f
        for f in os.listdir(emb_dir)
        if f.startswith("embeddings_part_") and f.endswith(".json")
    )
    if not files:
        raise FileNotFoundError(f"No embedding shards found in: {emb_dir}")
    return files


def load_shard(path: str) -> Tuple[List[str], np.ndarray]:
    with open(path, "r") as f:
        data = json.load(f)
    names = [row["image_name"] for row in data]
    embs = np.asarray([row["embedding"] for row in data], dtype="float32")
    return names, embs


def assert_normalized(embs: np.ndarray, atol: float = 1e-3) -> None:
    norms = np.linalg.norm(embs, axis=1)
    if not np.allclose(norms, 1.0, atol=atol):
        raise ValueError("Embeddings are not L2-normalized.")


def build_fixed_queries(
    tar_dir: str,
    out_path: str,
    max_queries: int,
) -> List[str]:
    if os.path.isfile(out_path):
        with open(out_path, "r") as f:
            queries = json.load(f)
        if len(queries) >= max_queries:
            return queries[:max_queries]

    queries: List[str] = []
    seen = set()
    tar_files = sorted(
        os.path.join(tar_dir, x)
        for x in os.listdir(tar_dir)
        if x.endswith(".tar")
    )
    for tp in tar_files:
        with tarfile.open(tp, "r") as tf:
            members = sorted(m for m in tf.getnames() if m.endswith(".json"))
            for member in members:
                if len(queries) >= max_queries:
                    break
                fp = tf.extractfile(member)
                if not fp:
                    continue
                try:
                    d = json.loads(fp.read().decode("utf-8", errors="ignore"))
                except Exception:
                    continue
                q = (d.get("caption") or "").strip()
                if not q or q in seen:
                    continue
                seen.add(q)
                queries.append(q)
        if len(queries) >= max_queries:
            break

    if len(queries) < max_queries:
        raise RuntimeError(f"Only {len(queries)} unique queries found, expected {max_queries}.")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(queries, f, indent=2)
    return queries


def chunked(items: Sequence[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), size):
        yield list(items[i : i + size])