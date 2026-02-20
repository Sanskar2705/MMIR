#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#faiss_two_stage_orchestrator.py
"""
FAISS Two-Stage Retrieval — Orchestrator + CLI (brute-force only)
Replicates the Solr two-stage pipeline semantics:

Stage-1: encode(query with stage1_model) -> search full FAISS index (topK)
Stage-2: encode(query with stage2_model) -> re-rank ONLY Stage-1 candidates (subset scoring) and return final top-k

Requires:
- Config via Benchmark.config.config_utils.load_config()
- FAISS index files created by your builder script:
    <model>_<dataset>_faiss_img.index               + *_img_names.json
    <model>_<dataset>_faiss_txt.index               + *_txt_meta.json
    <model>_<dataset>_faiss_joint-image-text.index  + *_joint-image-text_meta.json
- Embedders via Benchmark.code.retrievalService.faiss_base_service.testing.embed_utils.get_embedder

Usage (interactive):
    python faiss_two_stage_orchestrator.py
"""

import os, sys, json, warnings, logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import faiss
import torch

# project paths
sys.path.append('/mnt/storage/RSystemsBenchmarking/gitProject')

from Benchmark.config.config_utils import load_config
from Benchmark.code.evaluation.time_util import get_time
from Benchmark.code.retrievalService.faiss_base_service.testing.embed_utils import get_embedder

# ---------- misc ----------
os.environ["HF_HUB_DISABLE_XET"] = "1"
warnings.filterwarnings("ignore")

def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def _l2norm(x: np.ndarray) -> np.ndarray:
    x = x.astype("float32", copy=False)
    n = np.linalg.norm(x) + 1e-12
    return (x / n).astype("float32", copy=False)

def _engine_tag(engine: Optional[str]) -> str:
    # brute-force only
    return "faiss"

def _target_from_core_type(ct: str) -> str:
    ct = ct.lower()
    if ct == "text":
        return "caption"
    if ct in ("image", "joint-image-text"):
        return ct
    return "caption"

def _paths(index_dir: Path, model: str, dataset: str, target: str, engine_tag: str):
    if target == "image":
        idx = index_dir / f"{model}_{dataset}_{engine_tag}_img.index"
        meta = index_dir / f"{model}_{dataset}_img_names.json"
    elif target == "caption":
        idx = index_dir / f"{model}_{dataset}_{engine_tag}_txt.index"
        meta = index_dir / f"{model}_{dataset}_txt_meta.json"
    else:  # joint-image-text
        idx = index_dir / f"{model}_{dataset}_{engine_tag}_joint-image-text.index"
        meta = index_dir / f"{model}_{dataset}_joint-image-text_meta.json"
    return idx, meta

def _load_meta(meta_path: Path, target: str):
    data = json.loads(meta_path.read_text())
    if target == "image":
        return [{"image_name": n, "caption": ""} for n in data]
    for r in data:
        r.setdefault("image_name", r.get("image_name", ""))
        r.setdefault("caption", r.get("caption", ""))
    return data

def _reconstruct_vec(index: faiss.Index, rid: int, dim: int) -> np.ndarray:
    try:
        v = index.reconstruct(rid)
        v = np.asarray(v, dtype="float32", order="C")
        if v.ndim == 0:
            raise TypeError
        return v
    except Exception:
        try:
            from faiss import vector_to_array
            return vector_to_array(index.reconstruct(rid)).astype("float32", copy=False)
        except Exception:
            try:
                from faiss import vector_float_to_array
                return vector_float_to_array(index.reconstruct(rid)).astype("float32", copy=False)
            except Exception:
                return np.zeros((dim,), dtype="float32")

# ---------- caches ----------
CFG = load_config()
INDEX_DIR = Path(CFG["vector_store"]["faiss"]["index_dir"])

EMBED_CACHE: Dict[str, callable] = {}
SEARCHER_CACHE: Dict[Tuple[str, str, str, str], tuple] = {}

def _get_embedder(model: str):
    fn = EMBED_CACHE.get(model)
    if fn is None:
        fn = get_embedder(model, CFG, _device())
        EMBED_CACHE[model] = fn
    return fn

def _get_searcher(model: str, dataset: str, target: str, engine_tag: str):
    key = (model, dataset, target, engine_tag)
    val = SEARCHER_CACHE.get(key)
    if val is None:
        idx_path, meta_path = _paths(INDEX_DIR, model, dataset, target, engine_tag)
        if not idx_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"[FAISS] Missing index/meta for {key} in {INDEX_DIR}")
        index = faiss.read_index(str(idx_path))
        meta = _load_meta(meta_path, target)
        ids_by_image: Dict[str, List[int]] = {}
        for i, row in enumerate(meta):
            ids_by_image.setdefault(row["image_name"], []).append(i)
        val = (index, meta, ids_by_image)
        SEARCHER_CACHE[key] = val
    return val

# ---------- Orchestrator ----------
class TwoStageFAISSOrchestrator:
    """
    Mirrors your Solr TwoStageRetrievalOrchestrator API, but uses FAISS brute-force.
    """
    def __init__(self):
        self.base_paths = {
            "coco":   CFG["paths"]["dataset"]["coco"]["base_image_path"],
            "flickr": CFG["paths"]["dataset"]["flickr"]["base_image_path"]
        }
        print(f"FAISS Orchestrator init. Device: {_device()}")

    def encode_text(self, model_name: str, text: str) -> Optional[Dict]:
        if not text.strip():
            return None
        embed = _get_embedder(model_name)
        t0 = get_time()
        vec = embed(text)
        enc = get_time() - t0
        return {"embedding": _l2norm(vec), "encoding_time": enc}

    def _search_full(self, model: str, dataset: str, target: str, engine_tag: str, q: np.ndarray, k: int):
        index, meta, _ = _get_searcher(model, dataset, target, engine_tag)
        t1 = get_time()
        D, I = index.search(q[None, :], k)
        dt = get_time() - t1
        out = []
        for rank, (rid, score) in enumerate(zip(I[0], D[0]), start=1):
            if rid < 0:
                continue
            row = meta[rid]
            out.append({
                "image_path": row["image_name"],
                "caption": row.get("caption", ""),
                "score": float(score),
                "rank": rank
            })
        return out, dt

    def _score_subset(self, model, dataset, target, engine_tag, q, image_names, k):
        index, meta, ids_by_image = _get_searcher(model, dataset, target, engine_tag)
        dim = index.d

        t1 = get_time()
        rows = []
        for name in image_names:
            ids = ids_by_image.get(name, [])
            for rid in ids:
                v = _reconstruct_vec(index, rid, dim)
                s = float(np.dot(q, v))
                rows.append({
                    "image_path": name,
                    "caption": meta[rid].get("caption", ""),
                    "score": s
                })
        rows.sort(key=lambda r: r["score"], reverse=True)
        dt = get_time() - t1

        out = []
        for rank, r in enumerate(rows[:k], start=1):
            out.append({
                "image_path": r["image_path"],
                "caption": r["caption"],
                "score": float(r["score"]),
                "rank": rank
            })
        return out, dt


    def execute_two_stage_pipeline(
        self,
        query_text: str,
        dataset: str,
        stage1_config: Tuple[str, str, str],  # (model_name, core_type, engine_tag['faiss'])
        stage2_config: Tuple[str, str, str],  # (model_name, core_type, engine_tag['faiss'])
        stage1_k: int = 100,
        stage2_k: int = 10,
        save_stage1_to: Optional[str] = None,
    ) -> Dict:
        """
        core_type in {'text','image','joint-image-text'}
        engine_tag only 'faiss' (brute-force)
        """
        print(f"\n=== FAISS Pipeline: {stage1_config} -> {stage2_config} | dataset={dataset} ===")
        print(f"Query: '{query_text}'")

        # ---- stage 1: encode + search full index
        s1_model, s1_core_type, _ = stage1_config
        t1 = _target_from_core_type(s1_core_type)
        s1_engine_tag = "faiss"

        enc1 = self.encode_text(s1_model, query_text)
        if enc1 is None:
            return {"error": "Stage 1 encoding failed", "results": []}
        q1 = enc1["embedding"]

        hits1, time1 = self._search_full(s1_model, dataset, t1, s1_engine_tag, q1, stage1_k)
        if not hits1:
            print("Stage 1 returned no results.")
            return {
                "results": [],
                "stage1": {"encode_time": enc1["encoding_time"], "search_time": time1, "candidates_found": 0},
                "stage2": {}
            }

        candidate_image_paths = sorted(set(h["image_path"] for h in hits1))
        print(f"Stage 2: Filtering on {len(candidate_image_paths)} candidate images.")

        # optionally persist stage1
        if save_stage1_to:
            try:
                Path(save_stage1_to).parent.mkdir(parents=True, exist_ok=True)
                with open(save_stage1_to, "w") as f:
                    json.dump({"query": query_text, "stage1_hits": hits1}, f, indent=2)
                print(f"Saved Stage-1 hits to {save_stage1_to}")
            except Exception as e:
                print(f"Warning: failed to save Stage-1 hits to file: {e}")

        # ---- stage 2: encode + re-score subset
        s2_model, s2_core_type, _ = stage2_config
        t2 = _target_from_core_type(s2_core_type)
        s2_engine_tag = "faiss"

        enc2 = self.encode_text(s2_model, query_text)
        if enc2 is None:
            return {"error": "Stage 2 encoding failed", "results": []}
        q2 = enc2["embedding"]

        hits2, time2 = self._score_subset(s2_model, dataset, t2, s2_engine_tag, q2, candidate_image_paths, stage2_k)

        total_time = enc1["encoding_time"] + time1 + enc2["encoding_time"] + time2

        return {
            "query": query_text,
            "final_results": hits2,
            "stage1_results": hits1,
            "total_time": total_time,
            "stage1": {
                "model": s1_model, "core": f"{s1_model}_{dataset}_{s1_core_type}",
                "encode_time": enc1["encoding_time"], "search_time": time1,
                "candidates_found": len(candidate_image_paths)
            },
            "stage2": {
                "model": s2_model, "core": f"{s2_model}_{dataset}_{s2_core_type}",
                "encode_time": enc2["encoding_time"], "search_time": time2,
                "results_returned": len(hits2)
            }
        }

    def print_results(self, pipeline_result: Dict):
        if "error" in pipeline_result:
            print(f"Error: {pipeline_result['error']}")
            return
        if not pipeline_result.get("final_results"):
            print("No final results to display.")
            return
        s2 = pipeline_result["stage2"]
        print(f"\n{'='*50}")
        print(f"QUERY: '{pipeline_result['query']}'")
        print(f"{'='*50}")
        print(f"STAGE 2: {s2['model']} -> {s2['core']}")
        print(f"  Encode: {s2['encode_time']:.4f}s | Search: {s2['search_time']:.4f}s | Results: {s2['results_returned']}")
        print(f"TOTAL TIME: {pipeline_result['total_time']:.4f}s")
        print(f"\nFINAL RESULTS (STAGE 2):")
        for i, doc in enumerate(pipeline_result["final_results"], 1):
            print(f"  {i}. {doc.get('image_path','N/A')} (Score: {doc.get('score','N/A'):.6f})")

# ---------- CLI ----------
def _get_user_choice(prompt: str, options: List[str], default: Optional[str] = None) -> str:
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
    orchestrator = TwoStageFAISSOrchestrator()

    datasets = ["coco", "flickr"]
    models = ["clip", "minilm", "uniir", "flava"]
    core_types = ["text", "image", "joint-image-text"]
    engines = ["faiss"]  # brute-force only

    print("="*60)
    print("Two-Stage Retrieval System (FAISS)")
    print("="*60)

    while True:
        try:
            dataset = _get_user_choice("Select a dataset:", datasets)
            print(f"\n--- First Stage Configuration (Top 100) ---")
            stage1_model = _get_user_choice("Select first stage model:", models)
            stage1_core_type = _get_user_choice("Select first stage core type:", core_types, "text")
            stage1_engine = _get_user_choice("FAISS engine:", engines, "faiss")

            print(f"\n--- Second Stage Configuration (Top 10) ---")
            stage2_model = _get_user_choice("Select second stage model:", models)
            stage2_core_type = _get_user_choice("Select second stage core type:", core_types, "text")
            stage2_engine = _get_user_choice("FAISS engine:", engines, "faiss")

            print(f"\nSelected Pipeline:")
            print(f"  Stage 1: {stage1_model} -> {stage1_model}_{dataset}_{stage1_core_type} [{stage1_engine}]")
            print(f"  Stage 2: {stage2_model} -> {stage2_model}_{dataset}_{stage2_core_type} [{stage2_engine}]")

            query = input("\nEnter your search query (or 'back' to reconfigure, 'exit' to quit): ").strip()
            if query.lower() == 'exit':
                break
            if query.lower() == 'back':
                continue

            save_path = input("Path to save Stage-1 hits (optional, blank to skip): ").strip() or None

            result = orchestrator.execute_two_stage_pipeline(
                query_text=query,
                dataset=dataset,
                stage1_config=(stage1_model, stage1_core_type, stage1_engine),
                stage2_config=(stage2_model, stage2_core_type, stage2_engine),
                stage1_k=100,
                stage2_k=10,
                save_stage1_to=save_path
            )

            orchestrator.print_results(result)

            another = input("\nWould you like to try another query with the same configuration? (y/n): ").strip().lower()
            if another != 'y':
                print("Returning to configuration...")
        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
