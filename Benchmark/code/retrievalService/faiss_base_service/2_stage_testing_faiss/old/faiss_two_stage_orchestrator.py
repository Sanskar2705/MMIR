#!/usr/bin/env python3
import sys, os, json, argparse, logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import faiss
import numpy as np
import torch

# project imports
sys.path.append('/mnt/storage/RSystemsBenchmarking/gitProject')
from Benchmark.config.config_utils import load_config
from Benchmark.code.evaluation.time_util import get_time
from Benchmark.code.retrievalService.faiss_base_service.testing.embed_utils import get_embedder

os.environ["HF_HUB_DISABLE_XET"] = "1"

# ----------------- helpers -----------------

def _d():
    return "cuda" if torch.cuda.is_available() else "cpu"

def _engine_tag(engine: str) -> str:
    e = engine.lower()
    if e == "faiss":
        return "faiss"
    if e in ("faiss-hnsw", "hnsw"):
        return "faiss-hnsw"
    raise ValueError("engine must be 'faiss' or 'faiss-hnsw'")

def _kind_meta_files(index_dir: Path, model: str, dataset: str, engine_tag: str, target: str) -> Tuple[Path, Path, str]:
    """
    target ∈ {"image","caption","text","joint-image-text"}
    returns (index_path, meta_path, kind)
    """
    t = target.lower()
    if t == "text":
        t = "caption"

    if t == "image":
        kind = "img"
        idx = index_dir / f"{model}_{dataset}_{engine_tag}_{kind}.index"
        meta = index_dir / f"{model}_{dataset}_img_names.json"        # list[str]
    elif t == "caption":
        kind = "txt"
        idx = index_dir / f"{model}_{dataset}_{engine_tag}_{kind}.index"
        meta = index_dir / f"{model}_{dataset}_txt_meta.json"         # list[{"image_name","caption"}]
    elif t == "joint-image-text":
        kind = "joint-image-text"
        idx = index_dir / f"{model}_{dataset}_{engine_tag}_{kind}.index"
        meta = index_dir / f"{model}_{dataset}_joint-image-text_meta.json"
    else:
        raise ValueError("target must be one of: image, caption/text, joint-image-text")

    if not idx.exists() or not meta.exists():
        raise FileNotFoundError(f"Missing index/meta for {model=}, {dataset=}, {engine_tag=}, {target=} at {index_dir}")
    return idx, meta, kind

def _load_meta(meta_path: Path, target: str):
    data = json.loads(meta_path.read_text())
    if target == "image":
        names = data  # list[str]
        # return a unified list of dict rows to standardize downstream
        return [{"image_name": n, "caption": ""} for n in names]
    # else it is a list of dict rows already
    for r in data:
        r.setdefault("caption", "")
        r.setdefault("image_name", r.get("image_name", ""))
    return data

def _normalize(vec: np.ndarray) -> np.ndarray:
    # in practice your embed fns already L2-normalize; this is a safety net
    if vec.ndim == 1:
        v = vec / (np.linalg.norm(vec) + 1e-12)
        return v.astype("float32")
    v = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12)
    return v.astype("float32")

def _canon_name(name: str) -> str:
    # canonicalize keys so '/abs/path/IMG.jpg' matches 'img.jpg'
    return os.path.basename(str(name)).lower()

def _canon_caption(s: str) -> str:
    return (s or "").strip().lower()

# ----------------- FAISS wrappers -----------------

class FaissSearcher:
    """
    Wrap a FAISS index + aligned metadata, with query-embedding provided externally.
    """
    def __init__(self, index_path: Path, meta_path: Path, target: str, engine_tag: str, ef_search: Optional[int] = None):
        self.index = faiss.read_index(str(index_path))
        self.meta  = _load_meta(meta_path, target=("image" if target=="image" else "caption"))
        self.target = target
        self.engine_tag = engine_tag
        self.dim = self.index.d

        # optional per-searcher efSearch default for HNSW
        if engine_tag == "faiss-hnsw" and ef_search is not None:
            try:
                self.index.hnsw.efSearch = int(ef_search)
            except Exception:
                pass

        # map canonical image_name -> list of row ids (for Stage-2 subset scoring)
        self.ids_by_image: Dict[str, List[int]] = {}
        # map canonical caption -> list of row ids (for Stage-2 caption-subset scoring)
        self.ids_by_caption: Dict[str, List[int]] = {}

        for i, row in enumerate(self.meta):
            key_img = _canon_name(row["image_name"])
            self.ids_by_image.setdefault(key_img, []).append(i)

            cap = _canon_caption(row.get("caption", ""))
            if cap:
                self.ids_by_caption.setdefault(cap, []).append(i)

    def search(self, qvec: np.ndarray, k: int) -> List[Dict]:
        q = qvec[None, :].astype("float32")
        D, I = self.index.search(q, k)
        out = []
        for rank, (idx, score) in enumerate(zip(I[0], D[0]), start=1):
            if idx < 0:
                continue
            row = self.meta[idx]
            out.append({
                "image_path": row["image_name"],
                "caption": row.get("caption", ""),
                "score": float(score),
                "rank": rank,
                "target": self.target,
                "engine": self.engine_tag
            })
        return out

    # ---- global (no-collapse) exact re-score on a restricted image subset ----
    def score_all_rows_for_images(self, qvec: np.ndarray, image_names: List[str], top_k: int) -> List[Dict]:
        """
        Re-rank ALL rows associated with the provided image_names (no per-image collapse).
        Returns top_k rows globally by exact inner-product on reconstructed vectors.
        """
        q = qvec.astype("float32")
        scored_rows: List[Tuple[float, int, str]] = []  # (score, row_id, image_name)

        for name in image_names:
            key = _canon_name(name)
            ids = self.ids_by_image.get(key)
            if not ids:
                continue
            for rid in ids:
                try:
                    v = self.index.reconstruct(rid)
                except Exception:
                    # if reconstruct isn't supported for this index/id, skip this row
                    continue
                # normalize reconstructed vector for cosine/IP parity
                v = v / (np.linalg.norm(v) + 1e-12)
                s = float(np.dot(q, v))
                scored_rows.append((s, rid, name))

        # Sort by score desc, take global top_k rows
        scored_rows.sort(key=lambda x: x[0], reverse=True)
        scored_rows = scored_rows[:top_k]

        out: List[Dict] = []
        for rank, (score, rid, name) in enumerate(scored_rows, start=1):
            row = self.meta[rid]
            out.append(
                {
                    "image_path": name,
                    "caption": row.get("caption", ""),
                    "score": float(score),
                    "rank": rank,
                    "target": self.target,
                    "engine": self.engine_tag
                }
            )
        return out

    # ---- IVF-only true subset KNN + fallback to exact rescoring for Flat/HNSW (by IMAGE NAMES) ----
    def _subset_ids_for_images(self, image_names: List[str]) -> np.ndarray:
        ids: List[int] = []
        for name in image_names:
            key = _canon_name(name)
            rows = self.ids_by_image.get(key)
            if rows:
                ids.extend(rows)
        if not ids:
            return np.empty((0,), dtype="int64")
        return np.array(sorted(set(ids)), dtype="int64")

    def search_subset(self, qvec: np.ndarray, image_names: List[str], k: int) -> List[Dict]:
        """
        KNN restricted to 'image_names' only, mirroring Solr's `fq=image_path:(...)`.
        Uses FAISS IDSelector (IVF) when available; falls back to exact re-score (no collapse).
        """
        q = qvec[None, :].astype("float32")

        # Try IVF selector path
        try:
            if hasattr(self.index, "nprobe") and hasattr(faiss, "IDSelectorArray") and hasattr(faiss, "SearchParametersIVF"):
                allowed = self._subset_ids_for_images(image_names)
                if allowed.size > 0 and hasattr(self.index, "search_with_parameters"):
                    sel = faiss.IDSelectorArray(allowed.size, faiss.swig_ptr(allowed))
                    params = faiss.SearchParametersIVF(sel, None, None)
                    D, I = self.index.search_with_parameters(q, k, params)
                    out: List[Dict] = []
                    for rank, (idx, score) in enumerate(zip(I[0], D[0]), start=1):
                        if idx < 0:
                            continue
                        row = self.meta[idx]
                        out.append(
                            {
                                "image_path": row["image_name"],
                                "caption": row.get("caption", ""),
                                "score": float(score),
                                "rank": rank,
                                "target": self.target,
                                "engine": self.engine_tag
                            }
                        )
                    return out
        except Exception:
            pass

        # Fallback (Flat/HNSW or older FAISS): exact IP on subset, no per-image collapse
        return self.score_all_rows_for_images(qvec=qvec, image_names=image_names, top_k=k)

    # (Caption-subset helpers kept for completeness but NOT used when mimicking Solr)

# ----------------- Orchestrator -----------------

class TwoStageRetrievalOrchestratorFAISS:
    """
    Mirrors the Solr orchestrator:
    Stage-1: search full index
    Stage-2: ALWAYS restrict by Stage-1 image_paths and search within that subset
             on the Stage-2 index (all rows for those images can compete).
    """
    def __init__(self, ef_search_default: Optional[int] = None):
        self.cfg = load_config()
        self.dev = _d()
        self.embed_cache: Dict[str, callable] = {}
        self.ef_search_default = ef_search_default

        self.index_dir = Path(self.cfg["vector_store"]["faiss"]["index_dir"])
        if not self.index_dir.exists():
            raise FileNotFoundError(f"FAISS index_dir not found: {self.index_dir}")

    def _embedder(self, model_tag: str):
        if model_tag not in self.embed_cache:
            self.embed_cache[model_tag] = get_embedder(model_tag, self.cfg, self.dev)
        return self.embed_cache[model_tag]

    def _searcher(self, model: str, dataset: str, target: str, engine: str) -> FaissSearcher:
        eng_tag = _engine_tag(engine)
        idx_path, meta_path, _ = _kind_meta_files(self.index_dir, model, dataset, eng_tag, target)
        return FaissSearcher(idx_path, meta_path, target, eng_tag, ef_search=self.ef_search_default)

    def execute_two_stage_pipeline(
        self,
        query_text: str,
        stage1: Tuple[str, str, str, str],  # (model, dataset, target, engine)
        stage2: Tuple[str, str, str, str],  # (model, dataset, target, engine)
        stage1_k: int = 100,
        stage2_k: int = 10
    ) -> Dict:

        (m1, d1, t1, e1) = stage1
        (m2, d2, t2, e2) = stage2

        # ---- Stage 1: encode + search
        emb1 = self._embedder(m1)
        t0 = get_time()
        q1 = emb1(query_text)
        enc1 = get_time() - t0
        q1 = _normalize(q1)

        s1 = self._searcher(m1, d1, t1, e1)
        t1s = get_time()
        stage1_hits = s1.search(q1, stage1_k)
        s1_time = get_time() - t1s

        if not stage1_hits:
            return {
                "query": query_text,
                "final_results": [],
                "stage1_results": [],
                "total_time": enc1 + s1_time,
                "stage1": {"model": m1, "dataset": d1, "target": t1, "engine": e1,
                           "encode_time": enc1, "search_time": s1_time, "candidates_found": 0},
                "stage2": {"model": m2, "dataset": d2, "target": t2, "engine": e2,
                           "encode_time": 0.0, "search_time": 0.0, "results_returned": 0}
            }

        # ---- Build image-path candidate set from Stage-1 (Solr-style) ----
        candidate_images = [h["image_path"] for h in stage1_hits if h.get("image_path")]
        # (optional) de-dup to avoid redundant filtering
        # candidate_images = list(dict.fromkeys(candidate_images))

        # ---- Stage 2: encode + search restricted to Stage-1 image set ----
        emb2 = self._embedder(m2)
        t2t = get_time()
        q2 = emb2(query_text)
        enc2 = get_time() - t2t
        q2 = _normalize(q2)

        s2 = self._searcher(m2, d2, t2, e2)
        t2s = get_time()
        # ALWAYS image-path subset (matches Solr fq=image_path:(...))
        stage2_hits = s2.search_subset(q2, image_names=candidate_images, k=stage2_k)
        s2_time = get_time() - t2s

        total = enc1 + s1_time + enc2 + s2_time

        return {
            "query": query_text,
            "final_results": stage2_hits,
            "stage1_results": stage1_hits,
            "total_time": total,
            "stage1": {"model": m1, "dataset": d1, "target": t1, "engine": e1,
                       "encode_time": enc1, "search_time": s1_time, "candidates_found": len(candidate_images)},
            "stage2": {"model": m2, "dataset": d2, "target": t2, "engine": e2,
                       "encode_time": enc2, "search_time": s2_time, "results_returned": len(stage2_hits)}
        }

    # pretty print (same vibe as your Solr script)
    @staticmethod
    def print_results(p: Dict):
        if not p.get("final_results"):
            print("No final results.")
            return
        print("\n" + "="*50)
        print(f"QUERY: '{p['query']}'")
        print("="*50)
        s1, s2 = p["stage1"], p["stage2"]
        print(f"STAGE 1: {s1['model']}:{s1['target']}:{s1['dataset']}:{s1['engine']}")
        print(f"  Encode: {s1['encode_time']:.4f}s | Search: {s1['search_time']:.4f}s | Candidates: {s1['candidates_found']}")
        print(f"STAGE 2: {s2['model']}:{s2['target']}:{s2['dataset']}:{s2['engine']}")
        print(f"  Encode: {s2['encode_time']:.4f}s | Search: {s2['search_time']:.4f}s | Results: {s2['results_returned']}")
        print(f"TOTAL TIME: {p['total_time']:.4f}s")
        print("\nFINAL RESULTS (STAGE 2):")
        for i, doc in enumerate(p["final_results"], 1):
            cap = doc.get("caption", "")
            cap = (cap if len(cap) < 120 else cap[:117] + "…")
            print(f"  {i:2d}. {doc.get('image_path','N/A')}  [score={doc.get('score',0):.6f}]  {cap}")
