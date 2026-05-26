"""Shared utilities for ANN indexing and retrieval."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import faiss
import numpy as np

from Benchmark.code.ANN.Code.configs import (
    BASE_IMAGE_DIRS,
    EMBEDDING_FILES,
    EMBEDDING_ROOT,
    INDEX_ROOT,
    EncoderName,
    DatasetName,
    IndexType,
)


def embedding_path(encoder: EncoderName, dataset: DatasetName) -> str:
    return os.path.join(EMBEDDING_ROOT, EMBEDDING_FILES[encoder][dataset])


def index_dir(index_type: IndexType) -> str:
    return os.path.join(INDEX_ROOT, index_type.upper())


def index_paths(
    index_type: IndexType,
    encoder: EncoderName,
    dataset: DatasetName,
    cfg_tag: str,
) -> Tuple[str, str, str]:
    """Return (index_path, names_path, meta_path). meta is used for uniir_joint."""
    sub = os.path.join(index_dir(index_type), f"{encoder}_{dataset}")
    os.makedirs(sub, exist_ok=True)
    prefix = f"{encoder}_{dataset}_{cfg_tag}"
    return (
        os.path.join(sub, f"{prefix}.index"),
        os.path.join(sub, f"{prefix}_names.json"),
        os.path.join(sub, f"{prefix}_meta.json"),
    )


def results_dir(index_type: IndexType) -> str:
    d = os.path.join(os.path.dirname(INDEX_ROOT), index_type.upper())
    os.makedirs(d, exist_ok=True)
    return d


def load_embeddings_json(path: str) -> Tuple[List[str], List[Dict[str, Any]], np.ndarray]:
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
    if not items:
        raise ValueError(f"Empty embedding file: {path}")

    names: List[str] = []
    meta_rows: List[Dict[str, Any]] = []
    embs: List[List[float]] = []

    for row in items:
        img = row["image_name"]
        names.append(img)
        meta_rows.append(
            {
                "image_name": img,
                "caption": row.get("caption", ""),
            }
        )
        embs.append(row["embedding"])

    matrix = np.asarray(embs, dtype="float32")
    matrix = l2_normalize(matrix)
    return names, meta_rows, matrix


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def set_ivf_nprobe(index: faiss.Index, nprobe: int) -> None:
    if hasattr(faiss, "extract_index_ivf"):
        ivf = faiss.extract_index_ivf(index)
        if ivf is not None:
            ivf.nprobe = int(nprobe)
            return
    if hasattr(index, "nprobe"):
        index.nprobe = int(nprobe)


def set_hnsw_ef_search(index: faiss.Index, ef_search: int) -> None:
    if hasattr(index, "hnsw"):
        index.hnsw.efSearch = int(ef_search)


def save_index_bundle(
    index: faiss.Index,
    names: Sequence[str],
    meta_rows: Sequence[Dict[str, Any]],
    index_path: str,
    names_path: str,
    meta_path: str,
) -> None:
    faiss.write_index(index, index_path)
    with open(names_path, "w", encoding="utf-8") as f:
        json.dump(list(names), f)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(list(meta_rows), f)


def load_index_bundle(
    index_path: str,
    names_path: str,
    meta_path: str,
) -> Tuple[faiss.Index, List[str], List[Dict[str, Any]]]:
    index = faiss.read_index(index_path)
    with open(names_path, "r", encoding="utf-8") as f:
        names = json.load(f)
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    else:
        meta = [{"image_name": n, "caption": ""} for n in names]
    return index, names, meta


def load_queries(query_path: str) -> List[Dict[str, str]]:
    with open(query_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for row in data:
        if "caption" not in row or "image" not in row:
            raise KeyError("Each query must have 'image' and 'caption'")
    return data


def build_caption_lookup(meta_rows: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for row in meta_rows:
        img = row.get("image_name", "")
        cap = row.get("caption", "")
        if img and cap and img not in lookup:
            lookup[img] = cap
    return lookup


def load_ingest_caption_lookup(ingest_path: str) -> Dict[str, str]:
    """First caption per image from ingest JSON (for image-index result enrichment)."""
    with open(ingest_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    lookup: Dict[str, str] = {}
    for row in items:
        img = row.get("image") or row.get("image_name", "")
        caps = row.get("caption", "")
        if isinstance(caps, list):
            cap = caps[0] if caps else ""
        else:
            cap = str(caps)
        if img and cap:
            lookup.setdefault(img, cap.strip())
    return lookup


def merge_caption_lookups(*lookups: Dict[str, str]) -> Dict[str, str]:
    merged: Dict[str, str] = {}
    for lk in lookups:
        merged.update(lk)
    return merged


def format_hit(
    image_name: str,
    score: float,
    rank: int,
    base_image_dir: str,
    caption_lookup: Dict[str, str],
    target: str,
) -> Dict[str, Any]:
    return {
        "image_path": image_name,
        "image_abs_path": str((Path(base_image_dir) / Path(image_name).name).resolve()),
        "score": float(score),
        "rank": int(rank),
        "caption": caption_lookup.get(image_name, ""),
        "target": target,
    }
