#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, json, warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import faiss
import torch
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel

sys.path.append("/mnt/storage/RSystemsBenchmarking/gitProject")
from Benchmark.config.config_utils import load_config
from Benchmark.code.evaluation.time_util import get_time
from Benchmark.code.retrievalService.faiss_base_service.testing.embed_utils import get_embedder

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

def _dev() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def _norm(v: np.ndarray) -> np.ndarray:
    v = v.astype("float32", copy=False)
    n = np.linalg.norm(v) + 1e-12
    return (v / n).astype("float32", copy=False)

def _engine_tag(e: Optional[str]) -> str:
    if not e:
        return "faiss"
    e = e.lower()
    return "faiss-hnsw" if e in ("faiss-hnsw", "hnsw") else "faiss"

def _target_from_core_type(ct: str) -> str:
    ct = ct.lower()
    if ct == "text": return "caption"
    if ct in ("image", "joint-image-text"): return ct
    return "caption"

def _paths(index_dir: Path, model: str, dataset: str, target: str, engine_tag: str):
    if target == "image":
        idx = index_dir / f"{model}_{dataset}_{engine_tag}_img.index"
        meta = index_dir / f"{model}_{dataset}_img_names.json"
    elif target == "caption":
        idx = index_dir / f"{model}_{dataset}_{engine_tag}_txt.index"
        meta = index_dir / f"{model}_{dataset}_txt_meta.json"
    else:
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

# robust reconstruct -> float32
def _reconstruct_vec(index: faiss.Index, rid: int, dim: int) -> np.ndarray:
    try:
        v = index.reconstruct(rid)
        v = np.asarray(v, dtype="float32", order="C")
        if v.ndim == 0: raise TypeError
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

CFG = load_config()
INDEX_DIR = Path(CFG["vector_store"]["faiss"]["index_dir"])
BASE_IMAGE_PATH_COCO   = CFG["paths"]["dataset"]["coco"]["base_image_path"]
BASE_IMAGE_PATH_FLICKR = CFG["paths"]["dataset"]["flickr"]["base_image_path"]

EMBED_CACHE: Dict[str, callable] = {}
SEARCHER_CACHE: Dict[Tuple[str, str, str, str], tuple] = {}

def get_cached_embedder(model: str):
    fn = EMBED_CACHE.get(model)
    if fn is None:
        fn = get_embedder(model, CFG, _dev())
        EMBED_CACHE[model] = fn
    return fn

def get_cached_searcher(model: str, dataset: str, target: str, engine_tag: str, ef_search: Optional[int]):
    key = (model, dataset, target, engine_tag)
    val = SEARCHER_CACHE.get(key)
    if val is None:
        idx_path, meta_path = _paths(INDEX_DIR, model, dataset, target, engine_tag)
        if not idx_path.exists() or not meta_path.exists():
            raise HTTPException(status_code=404, detail=f"Missing index/meta for {key} under {INDEX_DIR}")
        index = faiss.read_index(str(idx_path))
        if engine_tag == "faiss-hnsw" and ef_search is not None:
            try: index.hnsw.efSearch = int(ef_search)
            except Exception: pass
        meta = _load_meta(meta_path, target)
        ids_by_image: Dict[str, List[int]] = {}
        for i, row in enumerate(meta):
            ids_by_image.setdefault(row["image_name"], []).append(i)
        val = (index, meta, ids_by_image)
        SEARCHER_CACHE[key] = val
    return val

def _add_abs_paths(dataset: str, docs: List[Dict]):
    # Keep your trailing-slash behavior (may yield double '//', which is fine)
    base = BASE_IMAGE_PATH_COCO if dataset == "coco" else BASE_IMAGE_PATH_FLICKR
    for d in docs:
        img = d.get("image_path")
        if img:
            d["image_abs_path"] = f"{base}/{os.path.basename(img)}"

app = FastAPI(title="FAISS Two-Stage Retrieval Service", version="1.2")

@app.get("/ping")
def ping():
    return {"ok": True}

@app.get("/available")
def available():
    combos = []
    for idx_file in INDEX_DIR.glob("*_faiss*.index"):
        parts = idx_file.stem.split("_")
        if len(parts) < 4: continue
        model, dataset, engine, kind = parts
        target = "image" if kind == "img" else ("caption" if kind == "txt" else "joint-image-text")
        combos.append(f"{model}_{dataset}_{target}_{engine}")
    return {"available": sorted(set(combos))}

@app.get("/two_stage_retrieval")
def two_stage_retrieval(
    query: str,
    dataset: str = Query(..., description="coco|flickr"),
    stage1_model: str = Query(...),
    stage1_core_type: str = Query(..., description="text|image|joint-image-text"),
    stage2_model: str = Query(...),
    stage2_core_type: str = Query(..., description="text|image|joint-image-text"),
    stage1_k: int = Query(100, ge=1, le=1000),
    stage2_k: int = Query(10, ge=1, le=100),
    stage1_engine: str = Query("faiss"),
    stage2_engine: str = Query("faiss"),
    ef: Optional[int] = Query(None, description="efSearch for HNSW (optional)"),
    save_stage1_to: Optional[str] = Query(None, description="Optional path to dump Stage-1 hits JSON")
):
    if dataset not in ["coco","flickr"]:
        raise HTTPException(status_code=400, detail="Dataset must be 'coco' or 'flickr'")

    # ---- Stage 1 (full-index search)
    t1 = _target_from_core_type(stage1_core_type)
    e1 = _engine_tag(stage1_engine)
    embed1 = get_cached_embedder(stage1_model)

    t0 = get_time(); q1 = _norm(embed1(query)); enc1 = get_time() - t0
    index1, meta1, _ = get_cached_searcher(stage1_model, dataset, t1, e1, ef)

    s0 = get_time(); D, I = index1.search(q1[None,:], stage1_k); time1 = get_time() - s0

    hits1_full = []
    for rid, score in zip(I[0], D[0]):
        if rid < 0: continue
        row = meta1[rid]
        # keep stage-1 as returned (can include repeats)
        hits1_full.append({"image_path": row["image_name"], "score": float(score)})

    if not hits1_full:
        return {
            "list_of_top_k": [],
            "query_time": time1,
            "encoding_time": {"stage1": enc1, "stage2": 0.0},
            "retrieval_time": {"stage1": time1, "stage2": 0.0},
            "stage1_results": []
        }

    if save_stage1_to:
        try:
            Path(save_stage1_to).parent.mkdir(parents=True, exist_ok=True)
            Path(save_stage1_to).write_text(json.dumps({"query": query, "stage1_hits": hits1_full}, indent=2))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to write Stage-1 file: {e}")

    # We enumerate all caption rows for each candidate image (duplicates in final come from multiple captions)
    # Using a SET here mirrors Solr’s filter semantics (duplicates in filter do not multiply results).
    candidate_images = sorted({h["image_path"] for h in hits1_full})

    # ---- Stage 2 (subset scoring, per-caption — allows duplicate image_path in final top-k)
    t2 = _target_from_core_type(stage2_core_type)
    e2 = _engine_tag(stage2_engine)
    embed2 = get_cached_embedder(stage2_model)

    t0 = get_time(); q2 = _norm(embed2(query)); enc2 = get_time() - t0
    index2, meta2, ids_by_image2 = get_cached_searcher(stage2_model, dataset, t2, e2, ef)
    dim2 = index2.d

    rows: List[Dict] = []
    s1 = get_time()
    for name in candidate_images:
        ids = ids_by_image2.get(name, [])
        for rid in ids:
            v = _reconstruct_vec(index2, rid, dim2)
            s = float(np.dot(q2, v))
            rows.append({"image_path": name, "score": s})
    time2 = get_time() - s1

    rows.sort(key=lambda r: r["score"], reverse=True)
    hits2 = [{"image_path": r["image_path"], "score": float(r["score"])} for r in rows[:stage2_k]]

    # decorate abs paths (and stage1 for convenience)
    _add_abs_paths(dataset, hits1_full)
    _add_abs_paths(dataset, hits2)

    return {
        "list_of_top_k": hits2,                                # image_path, score, image_abs_path (duplicates possible)
        "query_time": time1 + time2,
        "encoding_time": {"stage1": enc1, "stage2": enc2},
        "retrieval_time": {"stage1": time1, "stage2": time2},
        "stage1_results": hits1_full
    }

# --- staged workflow
class Stage1Body(BaseModel):
    query: str
    dataset: str
    stage1_model: str
    stage1_core_type: str
    stage1_engine: str = "faiss"
    stage1_k: int = 100
    ef: Optional[int] = None
    save_to: Optional[str] = None

@app.post("/stage1_only")
def stage1_only(body: Stage1Body):
    if body.dataset not in ["coco","flickr"]:
        raise HTTPException(status_code=400, detail="dataset must be 'coco' or 'flickr'")
    t1 = _target_from_core_type(body.stage1_core_type)
    e1 = _engine_tag(body.stage1_engine)

    embed1 = get_cached_embedder(body.stage1_model)
    t0 = get_time(); q1 = _norm(embed1(body.query)); enc1 = get_time() - t0
    index1, meta1, _ = get_cached_searcher(body.stage1_model, body.dataset, t1, e1, body.ef)

    s0 = get_time(); D, I = index1.search(q1[None,:], body.stage1_k); time1 = get_time() - s0

    hits1 = []
    for rid, score in zip(I[0], D[0]):
        if rid < 0: continue
        row = meta1[rid]
        hits1.append({"image_path": row["image_name"], "score": float(score)})

    if body.save_to:
        try:
            Path(body.save_to).parent.mkdir(parents=True, exist_ok=True)
            Path(body.save_to).write_text(json.dumps({"query": body.query, "stage1_hits": hits1}, indent=2))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save: {e}")

    _add_abs_paths(body.dataset, hits1)
    return {"stage1_results": hits1, "encoding_time": {"stage1": enc1}, "retrieval_time": {"stage1": time1}}

class Stage2Body(BaseModel):
    query: str
    dataset: str
    stage2_model: str
    stage2_core_type: str
    stage2_engine: str = "faiss"
    stage2_k: int = 10
    ef: Optional[int] = None
    stage1_file: str

@app.post("/stage2_from_file")
@app.post("/stage2_from_file")
def stage2_from_file(body: Stage2Body):
    p = Path(body.stage1_file)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"stage1_file not found: {p}")
    blob = json.loads(p.read_text())
    hits1 = blob.get("stage1_hits", [])
    if not hits1:
        raise HTTPException(status_code=400, detail="stage1_file has no 'stage1_hits'")

    # Mirror Solr: filter candidates by unique image names; duplicates in final come from multiple caption rows
    candidate_images = sorted({h["image_path"] for h in hits1})

    t2 = _target_from_core_type(body.stage2_core_type)
    e2 = _engine_tag(body.stage2_engine)
    embed2 = get_cached_embedder(body.stage2_model)

    t0 = get_time(); q2 = _norm(embed2(body.query)); enc2 = get_time() - t0
    index2, meta2, ids_by_image2 = get_cached_searcher(body.stage2_model, body.dataset, t2, e2, body.ef)
    dim2 = index2.d

    rows: List[Dict] = []
    s1 = get_time()
    for name in candidate_images:
        ids = ids_by_image2.get(name, [])
        for rid in ids:
            v = _reconstruct_vec(index2, rid, dim2)
            s = float(np.dot(q2, v))
            rows.append({"image_path": name, "score": s})
    time2 = get_time() - s1

    rows.sort(key=lambda r: r["score"], reverse=True)
    hits2 = [{"image_path": r["image_path"], "score": float(r["score"])} for r in rows[:body.stage2_k]]
    _add_abs_paths(body.dataset, hits2)

    return {"list_of_top_k": hits2, "encoding_time": {"stage2": enc2}, "retrieval_time": {"stage2": time2}}
