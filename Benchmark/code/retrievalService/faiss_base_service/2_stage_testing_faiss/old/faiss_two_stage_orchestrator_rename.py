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
        for i, row in enumerate(self.meta):
            key = _canon_name(row["image_name"])
            self.ids_by_image.setdefault(key, []).append(i)

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

    def score_subset_by_image_names(self, qvec: np.ndarray, image_names: List[str], top_k: int) -> List[Dict]:
        """
        Re-rank ONLY the provided image_names against this index, by exact IP.
        Uses faiss.reconstruct(id) to fetch vectors (works with HNSWFlat/FlatIP).
        For images/captions with multiple rows, takes the best score per image.
        """
        q = qvec.astype("float32")
        best_by_img: Dict[str, Tuple[float, int]] = {}

        for name in image_names:
            key = _canon_name(name)
            ids = self.ids_by_image.get(key)
            if not ids:
                continue
            best_score = -1e9
            best_id = -1
            for rid in ids:
                try:
                    v = faiss.vector_float_to_array(self.index.reconstruct(rid))
                except Exception:
                    # if reconstruct isn't supported for this index/id, skip
                    continue
                # cosine/IP parity: normalize db vector before dot
                v = v / (np.linalg.norm(v) + 1e-12)
                s = float(np.dot(q, v))
                if s > best_score:
                    best_score, best_id = s, rid
            if best_id >= 0:
                best_by_img[name] = (best_score, best_id)

        # sort by score desc and format
        ranked = sorted(best_by_img.items(), key=lambda kv: kv[1][0], reverse=True)[:top_k]
        out = []
        for rank, (name, (score, rid)) in enumerate(ranked, start=1):
            row = self.meta[rid]
            out.append({
                "image_path": name,
                "caption": row.get("caption", ""),
                "score": float(score),
                "rank": rank,
                "target": self.target,
                "engine": self.engine_tag
            })
        return out

# ----------------- Orchestrator -----------------

class TwoStageRetrievalOrchestratorFAISS:
    """
    Mirrors your Solr orchestrator, but uses FAISS indexes on disk.
    Stage-1: search full index
    Stage-2: re-rank only Stage-1 candidates (by exact IP on subset)
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

        # Collect candidate image_paths from Stage-1
        candidates = [h["image_path"] for h in stage1_hits if h.get("image_path")]

        # ---- Stage 2: encode + re-rank ONLY candidates
        emb2 = self._embedder(m2)
        t2 = get_time()
        q2 = emb2(query_text)
        enc2 = get_time() - t2
        q2 = _normalize(q2)

        s2 = self._searcher(m2, d2, t2, e2)
        t2s = get_time()
        stage2_hits = s2.score_subset_by_image_names(q2, candidates, top_k=stage2_k)
        s2_time = get_time() - t2s

        total = enc1 + s1_time + enc2 + s2_time

        return {
            "query": query_text,
            "final_results": stage2_hits,
            "stage1_results": stage1_hits,
            "total_time": total,
            "stage1": {"model": m1, "dataset": d1, "target": t1, "engine": e1,
                       "encode_time": enc1, "search_time": s1_time, "candidates_found": len(candidates)},
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

# ----------------- CLI -----------------

def get_user_choice(prompt, options, default=None):
    print(f"\n{prompt}")
    for i, option in enumerate(options, 1):
        print(f"  {i}. {option}")
    while True:
        choice = input(f"Enter your choice (1-{len(options)}): ").strip()
        if not choice and default is not None:
            return default
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        print(f"Invalid choice. Please enter a number between 1 and {len(options)}.")

def main():
    cfg = load_config()

    datasets = list(cfg["paths"]["dataset"].keys())  # ["coco", "flickr", ...]
    models   = list(cfg["models"].keys())           # ["clip","minilm","uniir","flava", ...]
    targets  = ["image", "caption", "joint-image-text"]
    engines  = ["faiss", "faiss-hnsw"]

    print("="*60)
    print("Two-Stage Retrieval System (FAISS)")
    print("="*60)

    orchestrator = TwoStageRetrievalOrchestratorFAISS(ef_search_default=None)

    while True:
        try:
            dataset = get_user_choice("Select a dataset:", datasets)

            print("\n--- First Stage Configuration (Top 100) ---")
            s1_model  = get_user_choice("Select first stage model:" , models)
            s1_target = get_user_choice("Select first stage target:", targets, "caption")
            s1_engine = get_user_choice("Select first stage engine:", engines, "faiss")

            print("\n--- Second Stage Configuration (Top 10) ---")
            s2_model  = get_user_choice("Select second stage model:" , models)
            s2_target = get_user_choice("Select second stage target:", targets, "caption")
            s2_engine = get_user_choice("Select second stage engine:", engines, "faiss")

            print("\nSelected Pipeline:")
            print(f"  Stage 1: {s1_model}_{dataset}_{s1_target}_{s1_engine}")
            print(f"  Stage 2: {s2_model}_{dataset}_{s2_target}_{s2_engine}")

            query = input("\nEnter your search query (or 'back' to reconfigure, 'exit' to quit): ").strip()
            if query.lower() == "exit":
                break
            if query.lower() == "back":
                continue

            result = orchestrator.execute_two_stage_pipeline(
                query_text=query,
                stage1=(s1_model, dataset, s1_target, s1_engine),
                stage2=(s2_model, dataset, s2_target, s2_engine),
                stage1_k=100,
                stage2_k=10
            )
            orchestrator.print_results(result)

            another = input("\nTry another query with same config? (y/n): ").strip().lower()
            if another != 'y':
                print("Returning to configuration...")
        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
