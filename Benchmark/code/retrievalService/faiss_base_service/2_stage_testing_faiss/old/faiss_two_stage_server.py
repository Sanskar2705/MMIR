#!/usr/bin/env python3
"""
FAISS Two-Stage Retrieval Server (uses NEW orchestrator logic)

Endpoints
---------
GET /health
GET /options
GET /two_stage_retrieval
  Supports both new and legacy param names.

New params:
  q                (str)   : query text
  dataset          (str)   : dataset key from config
  s1_model         (str)
  s1_target        (str)   : image | caption | text | joint-image-text
  s1_engine        (str)   : faiss | faiss-hnsw
  s2_model         (str)
  s2_target        (str)
  s2_engine        (str)
  s1_k             (int)   : default 100
  s2_k             (int)   : default 10
  ef               (int)   : optional HNSW efSearch override for both stages

Legacy params (will be mapped automatically):
  query            -> q
  stage1_model     -> s1_model
  stage1_core_type -> s1_target
  stage2_model     -> s2_model
  stage2_core_type -> s2_target
  stage1_k         -> s1_k
  stage2_k         -> s2_k

Notes
-----
- Stage-2 is restricted to Stage-1's candidate subset using the new logic in:
  /mnt/storage/RSystemsBenchmarking/gitProject/Benchmark/code/retrievalService/faiss_base_service/2_stage_testing_faiss/faiss_two_stage_orchestrator.py

Run
---
NO_PROXY=localhost,127.0.0.1 uvicorn faiss_two_stage_server:app --host 0.0.0.0 --port 5055 --reload
"""

import os
import sys
import json
import importlib.util
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ---- Env hygiene (helpful for local hosts behind proxies) ----
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ["HF_HUB_DISABLE_XET"] = "1"

# ---- Project root on path for config + embed utils used by orchestrator ----
sys.path.append("/mnt/storage/RSystemsBenchmarking/gitProject")

# You can import load_config directly from the project
from Benchmark.config.config_utils import load_config  # type: ignore

# ---- Dynamically load the orchestrator file (folder name starts with a digit) ----
ORCH_FILE = "/mnt/storage/RSystemsBenchmarking/gitProject/Benchmark/code/retrievalService/faiss_base_service/2_stage_testing_faiss/faiss_two_stage_orchestrator.py"
if not Path(ORCH_FILE).exists():
    raise RuntimeError(f"Orchestrator file not found at {ORCH_FILE}")

_spec = importlib.util.spec_from_file_location("faiss_two_stage_orchestrator", ORCH_FILE)
_orch_mod = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_orch_mod)

TwoStageRetrievalOrchestratorFAISS = _orch_mod.TwoStageRetrievalOrchestratorFAISS  # type: ignore

# ---- Build app ----
app = FastAPI(title="FAISS Two-Stage Retrieval (New Logic)", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock down if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Preload config and options to help validate inputs ----
try:
    _CFG = load_config()
    _DATASETS = list(_CFG["paths"]["dataset"].keys())
    _MODELS = list(_CFG["models"].keys())
    _TARGETS = ["image", "caption", "text", "joint-image-text"]
    _ENGINES = ["faiss", "faiss-hnsw"]
except Exception as e:
    raise RuntimeError(f"Failed to load config via load_config(): {e}")

# Create a single orchestrator; efSearch can be overridden per-request
_BASE_ORCH = TwoStageRetrievalOrchestratorFAISS(ef_search_default=None)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True}


@app.get("/options")
def options() -> Dict[str, Any]:
    return {
        "datasets": _DATASETS,
        "models": _MODELS,
        "targets": _TARGETS,
        "engines": _ENGINES,
        "defaults": {"stage1_k": 100, "stage2_k": 10},
        "notes": "For caption-subset Stage-2, set both targets to 'caption' (or 'text').",
    }


@app.get("/two_stage_retrieval")
def two_stage_retrieval(
    # query text (new and legacy)
    q: Optional[str] = Query(None),
    query: Optional[str] = Query(None),

    # dataset
    dataset: str = Query(..., description="Dataset key from config"),

    # stage-1 (new + legacy)
    s1_model: Optional[str] = Query(None),
    stage1_model: Optional[str] = Query(None),
    s1_target: Optional[str] = Query(None),
    stage1_core_type: Optional[str] = Query(None),
    s1_engine: Optional[str] = Query(None),

    # stage-2 (new + legacy)
    s2_model: Optional[str] = Query(None),
    stage2_model: Optional[str] = Query(None),
    s2_target: Optional[str] = Query(None),
    stage2_core_type: Optional[str] = Query(None),
    s2_engine: Optional[str] = Query(None),

    # top-k (new + legacy)
    s1_k: Optional[int] = Query(None, ge=1, le=1000),
    stage1_k: Optional[int] = Query(None, ge=1, le=1000),
    s2_k: Optional[int] = Query(None, ge=1, le=100),
    stage2_k: Optional[int] = Query(None, ge=1, le=100),

    # hnsw efSearch override (optional)
    ef: Optional[int] = Query(None),
) -> Dict[str, Any]:
    """
    Two-stage retrieval using NEW orchestrator logic:
    - Stage 1: full-index search (Model-1) → top-K candidates
    - Stage 2: restricted to Stage-1 candidates only (Model-2)
        * caption-subset if both targets are caption/text
        * else image-subset
    """
    # --- resolve aliases ---
    qtext = q or query
    if not qtext:
        raise HTTPException(400, "Missing query text. Use 'q=' or 'query='.")

    s1m = s1_model or stage1_model
    s2m = s2_model or stage2_model
    if not s1m or not s2m:
        raise HTTPException(400, "Missing model(s). Provide 's1_model'/'stage1_model' and 's2_model'/'stage2_model'.")

    # targets: prefer new, fall back to legacy '*_core_type'
    t1 = (s1_target or stage1_core_type or "caption").lower()
    t2 = (s2_target or stage2_core_type or "caption").lower()
    # normalize 'text' → 'caption'
    t1 = "caption" if t1 == "text" else t1
    t2 = "caption" if t2 == "text" else t2

    # engines default to faiss if omitted
    e1 = (s1_engine or "faiss").lower()
    e2 = (s2_engine or "faiss").lower()

    # top-k: prefer explicit stage1_k/stage2_k if present
    k1 = stage1_k or s1_k or 100
    k2 = stage2_k or s2_k or 10

    # --- validate against config-driven options ---
    if dataset not in _DATASETS:
        raise HTTPException(400, f"Unknown dataset '{dataset}'. See /options.")
    if s1m not in _MODELS or s2m not in _MODELS:
        raise HTTPException(400, f"Unknown model (s1='{s1m}', s2='{s2m}'). See /options.")
    if t1 not in _TARGETS or t2 not in _TARGETS:
        raise HTTPException(400, f"Bad target (s1='{t1}', s2='{t2}'). Must be one of {_TARGETS}.")
    if e1 not in _ENGINES or e2 not in _ENGINES:
        raise HTTPException(400, f"Bad engine (s1='{e1}', s2='{e2}'). Must be one of {_ENGINES}.")

    # new orchestrator instance if ef provided (so efSearch is applied inside searchers)
    orch = TwoStageRetrievalOrchestratorFAISS(ef_search_default=ef) if ef is not None else _BASE_ORCH

    try:
        result = orch.execute_two_stage_pipeline(
            query_text=qtext,
            stage1=(s1m, dataset, t1, e1),
            stage2=(s2m, dataset, t2, e2),
            stage1_k=k1,
            stage2_k=k2,
        )
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Pipeline error: {e}")

    # ---- BEGIN: legacy compatibility wrapper for batch scripts ----
    # (Keep original result fields, add Solr-shaped aliases expected by your generator)
    legacy = dict(result)  # shallow copy
    final = result.get("final_results", [])

    # ensure dataset present on each hit; expose as list_of_top_k for legacy readers
    list_of_top_k = []
    for hit in final:
        h = dict(hit)
        h.setdefault("dataset", dataset)
        list_of_top_k.append(h)

    s1 = result.get("stage1", {})
    s2 = result.get("stage2", {})

    legacy.update({
        # primary list expected by generate_metadata_two_stage.py
        "list_of_top_k": list_of_top_k,

        # timing aliases
        "running_time": float(result.get("total_time", 0.0)),
        "running_time_without_reranking": float(s1.get("encode_time", 0.0)) + float(s1.get("search_time", 0.0)),
        "query_time_service": 0.0,  # not measured here
        "stage1_encoding_time": float(s1.get("encode_time", 0.0)),
        "stage2_encoding_time": float(s2.get("encode_time", 0.0)),
        "stage1_time_service": float(s1.get("search_time", 0.0)),
        "stage2_time_service": float(s2.get("search_time", 0.0)),

        # energy/rerank placeholders (0 unless you wire real meters)
        "energy_util": 0,
        "reranking_time": 0.0,
        "reranking_energy": 0,
    })
    # ---- END: legacy compatibility wrapper ----

    return legacy


# Optional short alias for parity with previous server
@app.get("/two_stage")
def two_stage_alias(**kwargs):
    return two_stage_retrieval(**kwargs)




# # #!/usr/bin/env python3
# # """
# # FAISS Two-Stage Retrieval Server (Flat-IP by default)

# # Endpoints
# # ---------
# # GET /two_stage_retrieval
# #   query, dataset, stage1_model, stage1_core_type, stage2_model, stage2_core_type,
# #   stage1_k=100, stage2_k=10, stage1_engine=faiss, stage2_engine=faiss

# # GET /two_stage    (compat alias)
# #   same params as /two_stage_retrieval

# # GET /rerank/{model}
# #   query, topk_list (JSON-encoded list from first call), K=10  [returns minimal list with scores]
# # """

# # import os
# # import sys
# # import json
# # from pathlib import Path
# # from typing import Dict, List, Optional, Tuple

# # import numpy as np
# # import faiss
# # import torch
# # from fastapi import FastAPI, Query, HTTPException
# # from fastapi.middleware.cors import CORSMiddleware
# # from pydantic import BaseModel

# # # --- project imports ---
# # sys.path.append("/mnt/storage/RSystemsBenchmarking/gitProject")
# # from Benchmark.config.config_utils import load_config
# # from Benchmark.code.evaluation.time_util import get_time
# # from Benchmark.code.retrievalService.faiss_base_service.testing.embed_utils import get_embedder

# # os.environ["HF_HUB_DISABLE_XET"] = "1"


# # # ----------------- helpers -----------------
# # def _dev() -> str:
# #     return "cuda" if torch.cuda.is_available() else "cpu"


# # def _engine_tag(engine: str) -> str:
# #     e = (engine or "").lower()
# #     if e == "faiss":
# #         return "faiss"
# #     if e in ("faiss-hnsw", "hnsw"):
# #         return "faiss-hnsw"
# #     raise ValueError("engine must be 'faiss' or 'faiss-hnsw'")


# # def _normalize(vec: np.ndarray) -> np.ndarray:
# #     if vec.ndim == 1:
# #         v = vec / (np.linalg.norm(vec) + 1e-12)
# #         return v.astype("float32")
# #     v = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12)
# #     return v.astype("float32")


# # def _kind_meta_files(
# #     index_dir: Path, model: str, dataset: str, engine_tag: str, target: str
# # ) -> Tuple[Path, Path, str]:
# #     """
# #     target ∈ {"image","caption","text","joint-image-text"}; "text" aliases to "caption".
# #     Returns (index_path, meta_path, normalized_target).
# #     """
# #     t = (target or "").lower()
# #     if t == "text":
# #         t = "caption"

# #     if t == "image":
# #         kind = "img"
# #         idx = index_dir / f"{model}_{dataset}_{engine_tag}_{kind}.index"
# #         meta = index_dir / f"{model}_{dataset}_img_names.json"
# #     elif t == "caption":
# #         kind = "txt"
# #         idx = index_dir / f"{model}_{dataset}_{engine_tag}_{kind}.index"
# #         meta = index_dir / f"{model}_{dataset}_txt_meta.json"
# #     elif t == "joint-image-text":
# #         kind = "joint-image-text"
# #         idx = index_dir / f"{model}_{dataset}_{engine_tag}_{kind}.index"
# #         meta = index_dir / f"{model}_{dataset}_joint-image-text_meta.json"
# #     else:
# #         raise ValueError("target must be image | caption | text | joint-image-text")

# #     if not idx.exists() or not meta.exists():
# #         raise FileNotFoundError(
# #             f"Missing index/meta at {index_dir} for model={model} dataset={dataset} "
# #             f"engine={engine_tag} target={t} (expected: {idx.name}, {meta.name})"
# #         )
# #     return idx, meta, t


# # def _load_meta(meta_path: Path, target_norm: str):
# #     data = json.loads(meta_path.read_text())
# #     if target_norm == "image":
# #         # img_names.json is a list[str]
# #         return [{"image_name": n, "caption": ""} for n in data]
# #     # txt_meta / joint-image-text_meta are lists of dicts
# #     for r in data:
# #         r.setdefault("caption", "")
# #         r.setdefault("image_name", r.get("image_name", ""))
# #     return data


# # def _abs_image_path(cfg: Dict, dataset: str, image_path: str) -> str:
# #     base = cfg["paths"]["dataset"][dataset]["base_image_path"]
# #     # Keep path semantics similar to your outputs
# #     return str(Path(base) / image_path)


# # def _canon_name(name: str) -> str:
# #     # basename + lowercase so '/abs/path/IMG.JPG' == 'img.jpg'
# #     return os.path.basename(str(name)).lower()


# # # ----------------- FAISS wrappers -----------------
# # class FaissSearcher:
# #     """
# #     Wrap a FAISS index + aligned metadata.
# #     """
# #     def __init__(
# #         self,
# #         index_path: Path,
# #         meta_path: Path,
# #         target_norm: str,
# #         engine_tag: str,
# #         ef_search: Optional[int] = None,
# #     ):
# #         self.index = faiss.read_index(str(index_path))
# #         self.meta = _load_meta(meta_path, target_norm)
# #         self.target_norm = target_norm
# #         self.engine_tag = engine_tag
# #         self.dim = self.index.d

# #         # sanity check: index rows match meta
# #         assert self.index.ntotal == len(self.meta), \
# #             f"Index rows ({self.index.ntotal}) != meta rows ({len(self.meta)}) for {index_path}"

# #         if engine_tag == "faiss-hnsw" and ef_search is not None:
# #             try:
# #                 self.index.hnsw.efSearch = int(ef_search)
# #             except Exception:
# #                 pass

# #         # Map CANON(image_name) -> list[row ids] for subset rescoring
# #         self.ids_by_image: Dict[str, List[int]] = {}
# #         for i, row in enumerate(self.meta):
# #             key = _canon_name(row["image_name"])
# #             self.ids_by_image.setdefault(key, []).append(i)

# #     def search(self, qvec: np.ndarray, k: int) -> List[Dict]:
# #         """Return top-k rows (as dicts) from the index search directly."""
# #         q = qvec[None, :].astype("float32")
# #         D, I = self.index.search(q, k)
# #         out = []
# #         for rank, (idx, score) in enumerate(zip(I[0], D[0]), start=1):
# #             if idx < 0:
# #                 continue
# #             row = self.meta[idx]
# #             out.append(
# #                 {
# #                     "image_path": row["image_name"],
# #                     "caption": row.get("caption", ""),
# #                     "score": float(score),
# #                     "rank": rank,
# #                     "row_id": int(idx),
# #                 }
# #             )
# #         return out

# #     def _reconstruct_row(self, rid: int) -> Optional[np.ndarray]:
# #         """
# #         Robustly reconstruct a row vector as float32 NumPy on all FAISS builds.
# #         """
# #         try:
# #             v = self.index.reconstruct(rid)
# #             if not isinstance(v, np.ndarray):
# #                 v = faiss.vector_float_to_array(v)
# #             v = v.astype("float32", copy=False)
# #             return v
# #         except Exception:
# #             try:
# #                 buf = np.empty((self.dim,), dtype="float32")
# #                 self.index.reconstruct_into(rid, faiss.swig_ptr(buf))
# #                 return buf
# #             except Exception:
# #                 return None

# #     def score_subset_by_image_names(self, qvec: np.ndarray, image_names: List[str], top_k: int) -> List[Dict]:
# #         """
# #         Per-image collapse: for each candidate image, take the BEST row (exact IP).
# #         Returns at most top_k UNIQUE images ranked by their best row score.
# #         """
# #         q = qvec.astype("float32")
# #         best_by_img: Dict[str, Tuple[float, int, str]] = {}  # canon -> (score, rid, original_name)

# #         for name in image_names:
# #             ckey = _canon_name(name)
# #             ids = self.ids_by_image.get(ckey)
# #             if not ids:
# #                 continue

# #             best_score = -1e9
# #             best_id = -1
# #             for rid in ids:
# #                 v = self._reconstruct_row(rid)
# #                 if v is None:
# #                     continue
# #                 v = v / (np.linalg.norm(v) + 1e-12)  # cosine/IP parity
# #                 s = float(np.dot(q, v))
# #                 if s > best_score:
# #                     best_score, best_id = s, rid

# #             if best_id >= 0:
# #                 best_by_img[ckey] = (best_score, best_id, name)

# #         ranked = sorted(best_by_img.items(), key=lambda kv: kv[1][0], reverse=True)[:top_k]
# #         out: List[Dict] = []
# #         for rank, (ckey, (score, rid, orig_name)) in enumerate(ranked, start=1):
# #             row = self.meta[rid]
# #             out.append(
# #                 {
# #                     "image_path": orig_name,
# #                     "caption": row.get("caption", ""),
# #                     "score": float(score),
# #                     "rank": rank,
# #                     "row_id": int(rid),
# #                 }
# #             )
# #         return out

# #     def score_all_rows_for_images(
# #         self, qvec: np.ndarray, image_names: List[str], top_k: int
# #     ) -> List[Dict]:
# #         """
# #         Row-level: score ALL rows for those images and take global top_k rows (duplicates allowed).
# #         Useful for diagnostics; image-level recall usually prefers per-image collapse.
# #         """
# #         q = qvec.astype("float32")
# #         scored_rows: List[Tuple[float, int, str]] = []  # (score, row_id, original_name)

# #         for name in image_names:
# #             ckey = _canon_name(name)
# #             ids = self.ids_by_image.get(ckey)
# #             if not ids:
# #                 continue
# #             for rid in ids:
# #                 v = self._reconstruct_row(rid)
# #                 if v is None:
# #                     continue
# #                 v = v / (np.linalg.norm(v) + 1e-12)
# #                 s = float(np.dot(q, v))
# #                 scored_rows.append((s, rid, name))

# #         scored_rows.sort(key=lambda x: x[0], reverse=True)
# #         scored_rows = scored_rows[:top_k]

# #         out: List[Dict] = []
# #         for rank, (score, rid, orig_name) in enumerate(scored_rows, start=1):
# #             row = self.meta[rid]
# #             out.append(
# #                 {
# #                     "image_path": orig_name,
# #                     "caption": row.get("caption", ""),
# #                     "score": float(score),
# #                     "rank": rank,
# #                     "row_id": int(rid),
# #                 }
# #             )
# #         return out


# # # ----------------- Orchestrator -----------------
# # class TwoStageRetrievalOrchestratorFAISS:
# #     """
# #     Stage-1: default Flat/IP ('faiss') unless explicitly asked for HNSW
# #     Stage-2: default Flat/IP ('faiss') unless explicitly asked for HNSW
# #     """
# #     def __init__(self, ef_search_default: Optional[int] = 64):
# #         self.cfg = load_config()
# #         self.dev = _dev()
# #         self.embed_cache: Dict[str, callable] = {}
# #         self.ef_search_default = ef_search_default
# #         self.index_dir = Path(self.cfg["vector_store"]["faiss"]["index_dir"])
# #         if not self.index_dir.exists():
# #             raise FileNotFoundError(f"FAISS index_dir not found: {self.index_dir}")

# #     def _embedder(self, model_tag: str):
# #         if model_tag not in self.embed_cache:
# #             self.embed_cache[model_tag] = get_embedder(model_tag, self.cfg, self.dev)
# #         return self.embed_cache[model_tag]

# #     def _searcher(
# #         self, model: str, dataset: str, target: str, engine: str, ef: Optional[int] = None
# #     ) -> Tuple[FaissSearcher, str]:
# #         eng_tag = _engine_tag(engine)
# #         idx_path, meta_path, tnorm = _kind_meta_files(self.index_dir, model, dataset, eng_tag, target)
# #         searcher = FaissSearcher(
# #             idx_path,
# #             meta_path,
# #             tnorm,
# #             eng_tag,
# #             ef_search=(ef if eng_tag == "faiss-hnsw" else None),
# #         )
# #         return searcher, tnorm

# #     def execute_two_stage(
# #         self,
# #         query_text: str,
# #         dataset: str,
# #         stage1_model: str,
# #         stage1_target: str,
# #         stage1_k: int,
# #         stage2_model: str,
# #         stage2_target: str,
# #         stage2_k: int,
# #         stage1_engine: str = "faiss",
# #         stage2_engine: str = "faiss",
# #     ) -> Dict:
# #         # ---- Stage 1: encode + search
# #         emb1 = self._embedder(stage1_model)
# #         t0 = get_time()
# #         q1 = emb1(query_text)
# #         enc1 = get_time() - t0
# #         q1 = _normalize(q1)

# #         s1, _ = self._searcher(
# #             stage1_model, dataset, stage1_target, stage1_engine, ef=self.ef_search_default
# #         )
# #         t1s = get_time()
# #         stage1_hits = s1.search(q1, stage1_k)
# #         s1_time = get_time() - t1s

# #         # ---- collect & dedupe candidate images (canonical) ----
# #         seen = set()
# #         candidates: List[str] = []
# #         for h in stage1_hits:
# #             im = h.get("image_path")
# #             if not im:
# #                 continue
# #             key = _canon_name(im)
# #             if key not in seen:
# #                 seen.add(key)
# #                 candidates.append(im)

# #         # ---- Stage 2: encode + re-rank BEST PER IMAGE ----
# #         emb2 = self._embedder(stage2_model)
# #         t2 = get_time()
# #         q2 = emb2(query_text)
# #         enc2 = get_time() - t2
# #         q2 = _normalize(q2)

# #         s2, _ = self._searcher(stage2_model, dataset, stage2_target, stage2_engine)
# #         t2s = get_time()
# #         stage2_rows = s2.score_subset_by_image_names(q2, candidates, top_k=stage2_k)
# #         s2_time = get_time() - t2s

# #         # Build the minimal final payload
# #         list_of_top_k: List[Dict] = []
# #         for r in stage2_rows:
# #             list_of_top_k.append(
# #                 {
# #                     "image_path": r["image_path"],
# #                     "score": float(r["score"]),
# #                     "image_abs_path": _abs_image_path(self.cfg, dataset, r["image_path"]),
# #                 }
# #             )

# #         return {
# #             "list_of_top_k": list_of_top_k,
# #             "retrieval_time": {"stage1": float(s1_time), "stage2": float(s2_time)},
# #             "encoding_time": {"stage1": float(enc1), "stage2": float(enc2)},
# #             "query_time": float(s1_time + s2_time),
# #         }


# # # ----------------- FastAPI app -----------------
# # app = FastAPI(title="FAISS Two-Stage Retrieval Server", version="1.3")

# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=["*"],
# #     allow_credentials=True,
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )

# # ORCH = TwoStageRetrievalOrchestratorFAISS(ef_search_default=64)


# # class RerankResponse(BaseModel):
# #     list_of_top_k: List[Dict]
# #     reranking_time: float


# # @app.get("/")
# # def root():
# #     return {
# #         "ok": True,
# #         "service": "faiss-two-stage",
# #         "endpoints": ["/two_stage_retrieval", "/two_stage", "/rerank/{model}"],
# #         "defaults": {"stage1_engine": "faiss", "stage2_engine": "faiss"},
# #     }


# # @app.get("/two_stage_retrieval")
# # def two_stage_retrieval(
# #     query: str = Query(..., description="Natural language query"),
# #     dataset: str = Query(..., description="Dataset tag (e.g., coco|flickr)"),
# #     stage1_model: str = Query(...),
# #     stage1_core_type: str = Query(..., description="image | caption | joint-image-text"),
# #     stage2_model: str = Query(...),
# #     stage2_core_type: str = Query(..., description="image | caption | joint-image-text"),
# #     stage1_k: int = Query(100, ge=1),
# #     stage2_k: int = Query(10, ge=1),
# #     stage1_engine: str = Query("faiss", description="faiss | faiss-hnsw"),
# #     stage2_engine: str = Query("faiss", description="faiss | faiss-hnsw"),
# # ):
# #     try:
# #         return ORCH.execute_two_stage(
# #             query_text=query,
# #             dataset=dataset,
# #             stage1_model=stage1_model,
# #             stage1_target=stage1_core_type,
# #             stage1_k=stage1_k,
# #             stage2_model=stage2_model,
# #             stage2_target=stage2_core_type,
# #             stage2_k=stage2_k,
# #             stage1_engine=stage1_engine,
# #             stage2_engine=stage2_engine,
# #         )
# #     except FileNotFoundError as e:
# #         raise HTTPException(status_code=404, detail=str(e))
# #     except ValueError as e:
# #         raise HTTPException(status_code=400, detail=str(e))
# #     except Exception as e:
# #         raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


# # # Compat alias for older clients that hit /two_stage
# # @app.get("/two_stage")
# # def two_stage_compat(
# #     query: str = Query(...),
# #     dataset: str = Query(...),
# #     stage1_model: str = Query(...),
# #     stage1_core_type: str = Query(...),
# #     stage2_model: str = Query(...),
# #     stage2_core_type: str = Query(...),
# #     stage1_k: int = Query(100, ge=1),
# #     stage2_k: int = Query(10, ge=1),
# #     stage1_engine: str = Query("faiss"),
# #     stage2_engine: str = Query("faiss"),
# # ):
# #     try:
# #         return ORCH.execute_two_stage(
# #             query_text=query,
# #             dataset=dataset,
# #             stage1_model=stage1_model,
# #             stage1_target=stage1_core_type,
# #             stage1_k=stage1_k,
# #             stage2_model=stage2_model,
# #             stage2_target=stage2_core_type,
# #             stage2_k=stage2_k,
# #             stage1_engine=stage1_engine,
# #             stage2_engine=stage2_engine,
# #         )
# #     except FileNotFoundError as e:
# #         raise HTTPException(status_code=404, detail=str(e))
# #     except ValueError as e:
# #         raise HTTPException(status_code=400, detail=str(e))
# #     except Exception as e:
# #         raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


# # @app.get("/rerank/{model}", response_model=RerankResponse)
# # def neural_reranking(
# #     model: str,
# #     query: str = Query(...),
# #     topk_list: str = Query(..., description="JSON-encoded list from first call"),
# #     K: int = Query(10, ge=1),
# # ):
# #     """
# #     Re-rank the provided candidates by scoring only those items using `model`.
# #     Returns one best row per image (per-image collapse) with scores and absolute paths.
# #     """
# #     try:
# #         items = json.loads(topk_list)
# #         if not isinstance(items, list) or not items:
# #             return {"list_of_top_k": [], "reranking_time": 0.0}

# #         cfg = ORCH.cfg
# #         dataset = None
# #         for ds in cfg["paths"]["dataset"].keys():
# #             base = cfg["paths"]["dataset"][ds]["base_image_path"]
# #             if "image_abs_path" in items[0] and items[0]["image_abs_path"].startswith(base):
# #                 dataset = ds
# #                 break
# #         if dataset is None and "selected_config" in cfg and "dataset" in cfg["selected_config"]:
# #             dataset = cfg["selected_config"]["dataset"]
# #         if dataset is None:
# #             raise ValueError("Cannot infer dataset for reranking (no dataset in items and no selected_config.dataset).")

# #         # dedupe & preserve input order
# #         seen = set()
# #         image_names: List[str] = []
# #         for it in items:
# #             name = it.get("image_path")
# #             if not name:
# #                 continue
# #             key = _canon_name(name)
# #             if key not in seen:
# #                 seen.add(key)
# #                 image_names.append(name)

# #         if not image_names:
# #             return {"list_of_top_k": [], "reranking_time": 0.0}

# #         emb = ORCH._embedder(model)
# #         t0 = get_time()
# #         q = emb(query)
# #         q = _normalize(q)

# #         # Use Flat/IP index for exact rescoring; assume target "caption" for text-based re-ranking
# #         s, _ = ORCH._searcher(model, dataset, "caption", "faiss")
# #         rows = s.score_subset_by_image_names(q, image_names, top_k=K)
# #         dt = get_time() - t0

# #         out = [
# #             {
# #                 "image_path": r["image_path"],
# #                 "score": float(r["score"]),
# #                 "image_abs_path": _abs_image_path(cfg, dataset, r["image_path"]),
# #             }
# #             for r in rows
# #         ]
# #         return {"list_of_top_k": out, "reranking_time": float(dt)}
# #     except ValueError as e:
# #         raise HTTPException(status_code=400, detail=str(e))
# #     except Exception as e:
# #         raise HTTPException(status_code=400, detail=f"Bad topk_list or config: {e}")


# # if __name__ == "__main__":
# #     import uvicorn

# #     # Example:
# #     # uvicorn faiss_two_stage_server:app --host 0.0.0.0 --port 5059 --reload
# #     uvicorn.run("faiss_two_stage_server:app", host="0.0.0.0", port=5059, reload=True)

# #!/usr/bin/env python3
# """
# FAISS Two-Stage Retrieval Server (Flat-IP by default)

# Endpoints
# ---------
# GET /two_stage_retrieval
#   query, dataset, stage1_model, stage1_core_type, stage2_model, stage2_core_type,
#   stage1_k=100, stage2_k=10, stage1_engine=faiss, stage2_engine=faiss

# GET /two_stage  (compat alias)
#   same params as /two_stage_retrieval

# GET /rerank/{model}
#   query, topk_list (JSON-encoded list from first call), K=10
#   [returns minimal list with scores]
# """

# import json
# import os
# import sys
# from pathlib import Path
# from typing import Dict, List, Optional, Tuple

# import faiss
# import numpy as np
# import torch
# from fastapi import FastAPI, HTTPException, Query
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel

# # --- project imports ---
# sys.path.append("/mnt/storage/RSystemsBenchmarking/gitProject")
# from Benchmark.code.evaluation.time_util import get_time  # noqa: E402
# from Benchmark.config.config_utils import load_config  # noqa: E402
# from Benchmark.code.retrievalService.faiss_base_service.testing.embed_utils import (  # noqa: E402,E501
#     get_embedder,
# )

# os.environ["HF_HUB_DISABLE_XET"] = "1"


# # ----------------- helpers -----------------
# def _dev() -> str:
#     return "cuda" if torch.cuda.is_available() else "cpu"


# def _engine_tag(engine: str) -> str:
#     e = (engine or "").lower()
#     if e == "faiss":
#         return "faiss"
#     if e in ("faiss-hnsw", "hnsw"):
#         return "faiss-hnsw"
#     raise ValueError("engine must be 'faiss' or 'faiss-hnsw'")


# def _normalize(vec: np.ndarray) -> np.ndarray:
#     if vec.ndim == 1:
#         v = vec / (np.linalg.norm(vec) + 1e-12)
#         return v.astype("float32")
#     v = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12)
#     return v.astype("float32")


# def _kind_meta_files(
#     index_dir: Path,
#     model: str,
#     dataset: str,
#     engine_tag: str,
#     target: str,
# ) -> Tuple[Path, Path, str]:
#     """
#     target ∈ {"image","caption","text","joint-image-text"}; "text" aliases to "caption".
#     Returns (index_path, meta_path, normalized_target).
#     """
#     t = (target or "").lower()
#     if t == "text":
#         t = "caption"

#     if t == "image":
#         kind = "img"
#         idx = index_dir / f"{model}_{dataset}_{engine_tag}_{kind}.index"
#         meta = index_dir / f"{model}_{dataset}_img_names.json"
#     elif t == "caption":
#         kind = "txt"
#         idx = index_dir / f"{model}_{dataset}_{engine_tag}_{kind}.index"
#         meta = index_dir / f"{model}_{dataset}_txt_meta.json"
#     elif t == "joint-image-text":
#         kind = "joint-image-text"
#         idx = index_dir / f"{model}_{dataset}_{engine_tag}_{kind}.index"
#         meta = index_dir / f"{model}_{dataset}_joint-image-text_meta.json"
#     else:
#         raise ValueError("target must be image | caption | text | joint-image-text")

#     if not idx.exists() or not meta.exists():
#         raise FileNotFoundError(
#             f"Missing index/meta at {index_dir} for model={model} dataset={dataset} "
#             f"engine={engine_tag} target={t} (expected: {idx.name}, {meta.name})"
#         )

#     return idx, meta, t


# def _load_meta(meta_path: Path, target_norm: str):
#     data = json.loads(meta_path.read_text())
#     if target_norm == "image":
#         # img_names.json is a list[str]
#         return [{"image_name": n, "caption": ""} for n in data]

#     # txt_meta / joint-image-text_meta are lists of dicts
#     for r in data:
#         r.setdefault("caption", "")
#         r.setdefault("image_name", r.get("image_name", ""))
#     return data


# def _abs_image_path(cfg: Dict, dataset: str, image_path: str) -> str:
#     base = cfg["paths"]["dataset"][dataset]["base_image_path"]
#     # Join base dir with (possibly nested) image_path, without normalizing away double slashes
#     # to stay close to your sample output.
#     return str(Path(base) / image_path)


# # ----------------- FAISS wrappers -----------------
# class FaissSearcher:
#     """Wrap a FAISS index + aligned metadata."""

#     def __init__(
#         self,
#         index_path: Path,
#         meta_path: Path,
#         target_norm: str,
#         engine_tag: str,
#         ef_search: Optional[int] = None,
#     ):
#         self.index = faiss.read_index(str(index_path))
#         self.meta = _load_meta(meta_path, target_norm)
#         self.target_norm = target_norm
#         self.engine_tag = engine_tag
#         self.dim = self.index.d

#         if engine_tag == "faiss-hnsw" and ef_search is not None:
#             try:
#                 self.index.hnsw.efSearch = int(ef_search)
#             except Exception:
#                 pass

#         # Map image_name -> list[row ids] for subset rescoring (caption rows per image)
#         self.ids_by_image: Dict[str, List[int]] = {}
#         for i, row in enumerate(self.meta):
#             self.ids_by_image.setdefault(row["image_name"], []).append(i)

#     def search(self, qvec: np.ndarray, k: int) -> List[Dict]:
#         """Return top-k rows (as dicts) from the index search directly."""
#         q = qvec[None, :].astype("float32")
#         D, I = self.index.search(q, k)

#         out: List[Dict] = []
#         for rank, (idx, score) in enumerate(zip(I[0], D[0]), start=1):
#             if idx < 0:
#                 continue
#             row = self.meta[idx]
#             out.append(
#                 {
#                     "image_path": row["image_name"],
#                     "caption": row.get("caption", ""),
#                     "score": float(score),
#                     "rank": rank,
#                     "row_id": int(idx),
#                 }
#             )
#         return out

#     def score_all_rows_for_images(
#         self, qvec: np.ndarray, image_names: List[str], top_k: int
#     ) -> List[Dict]:
#         """
#         Re-rank ALL rows associated with the provided image_names (no per-image collapse).
#         Returns top_k rows globally by exact inner-product on reconstructed vectors.
#         """
#         q = qvec.astype("float32")
#         scored_rows: List[Tuple[float, int, str]] = []  # (score, row_id, image_name)

#         for name in image_names:
#             ids = self.ids_by_image.get(name)
#             if not ids:
#                 continue
#             for rid in ids:
#                 try:
#                     v = self.index.reconstruct(rid)
#                 except Exception:
#                     # if reconstruct isn't supported for this index/id, skip this row
#                     continue
#                 # A) normalize reconstructed vector for cosine/IP parity
#                 v = v / (np.linalg.norm(v) + 1e-12)
#                 s = float(np.dot(q, v))
#                 scored_rows.append((s, rid, name))

#         # Sort by score desc, take global top_k rows
#         scored_rows.sort(key=lambda x: x[0], reverse=True)
#         scored_rows = scored_rows[:top_k]

#         out: List[Dict] = []
#         for rank, (score, rid, name) in enumerate(scored_rows, start=1):
#             row = self.meta[rid]
#             out.append(
#                 {
#                     "image_path": name,
#                     "caption": row.get("caption", ""),
#                     "score": float(score),
#                     "rank": rank,
#                     "row_id": int(rid),
#                 }
#             )
#         return out


# # ----------------- Orchestrator -----------------
# class TwoStageRetrievalOrchestratorFAISS:
#     """
#     Stage-1: default Flat/IP ('faiss') unless explicitly asked for HNSW
#     Stage-2: default Flat/IP ('faiss') unless explicitly asked for HNSW
#     """

#     def __init__(self, ef_search_default: Optional[int] = 64):
#         self.cfg = load_config()
#         self.dev = _dev()
#         self.embed_cache: Dict[str, callable] = {}
#         self.ef_search_default = ef_search_default
#         self.index_dir = Path(self.cfg["vector_store"]["faiss"]["index_dir"])
#         if not self.index_dir.exists():
#             raise FileNotFoundError(f"FAISS index_dir not found: {self.index_dir}")

#     def _embedder(self, model_tag: str):
#         if model_tag not in self.embed_cache:
#             self.embed_cache[model_tag] = get_embedder(model_tag, self.cfg, self.dev)
#         return self.embed_cache[model_tag]

#     def _searcher(
#         self,
#         model: str,
#         dataset: str,
#         target: str,
#         engine: str,
#         ef: Optional[int] = None,
#     ) -> Tuple[FaissSearcher, str]:
#         eng_tag = _engine_tag(engine)
#         idx_path, meta_path, tnorm = _kind_meta_files(
#             self.index_dir, model, dataset, eng_tag, target
#         )
#         searcher = FaissSearcher(
#             idx_path,
#             meta_path,
#             tnorm,
#             eng_tag,
#             ef_search=(ef if eng_tag == "faiss-hnsw" else None),
#         )
#         return searcher, tnorm

#     def execute_two_stage(
#         self,
#         query_text: str,
#         dataset: str,
#         stage1_model: str,
#         stage1_target: str,
#         stage1_k: int,
#         stage2_model: str,
#         stage2_target: str,
#         stage2_k: int,
#         stage1_engine: str = "faiss",
#         stage2_engine: str = "faiss",
#     ) -> Dict:
#         # Stage-1: encode + search
#         emb1 = self._embedder(stage1_model)
#         t0 = get_time()
#         q1 = emb1(query_text)
#         enc1 = get_time() - t0
#         q1 = _normalize(q1)

#         s1, _ = self._searcher(
#             stage1_model, dataset, stage1_target, stage1_engine, ef=self.ef_search_default
#         )
#         t1s = get_time()
#         stage1_hits = s1.search(q1, stage1_k)
#         s1_time = get_time() - t1s
#         candidates = [h["image_path"] for h in stage1_hits]

#         # Stage-2: encode + re-rank ALL rows from caption/joint target for those images
#         emb2 = self._embedder(stage2_model)
#         t2 = get_time()
#         q2 = emb2(query_text)
#         enc2 = get_time() - t2
#         q2 = _normalize(q2)

#         s2, _ = self._searcher(stage2_model, dataset, stage2_target, stage2_engine)
#         t2s = get_time()
#         # critical: do NOT collapse per-image; score all rows and take global top_k
#         stage2_rows = s2.score_all_rows_for_images(q2, candidates, top_k=stage2_k)
#         s2_time = get_time() - t2s

#         # Build the final response payload EXACTLY like you want:
#         # list_of_top_k: [{image_path, score, image_abs_path}, ...]
#         list_of_top_k: List[Dict] = []
#         for r in stage2_rows:
#             list_of_top_k.append(
#                 {
#                     "image_path": r["image_path"],
#                     "score": float(r["score"]),
#                     "image_abs_path": _abs_image_path(
#                         self.cfg, dataset, r["image_path"]
#                     ),
#                 }
#             )

#         return {
#             "list_of_top_k": list_of_top_k,
#             "retrieval_time": {"stage1": float(s1_time), "stage2": float(s2_time)},
#             "encoding_time": {"stage1": float(enc1), "stage2": float(enc2)},
#             "query_time": float(s1_time + s2_time),
#         }


# # ----------------- FastAPI app -----------------
# app = FastAPI(title="FAISS Two-Stage Retrieval Server", version="1.2")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# ORCH = TwoStageRetrievalOrchestratorFAISS(ef_search_default=64)


# class RerankResponse(BaseModel):
#     list_of_top_k: List[Dict]
#     reranking_time: float


# @app.get("/")
# def root():
#     return {
#         "ok": True,
#         "service": "faiss-two-stage",
#         "endpoints": ["/two_stage_retrieval", "/two_stage", "/rerank/{model}"],
#         "defaults": {"stage1_engine": "faiss", "stage2_engine": "faiss"},
#     }


# @app.get("/two_stage_retrieval")
# def two_stage_retrieval(
#     query: str = Query(..., description="Natural language query"),
#     dataset: str = Query(..., description="Dataset tag (e.g., coco|flickr)"),
#     stage1_model: str = Query(...),
#     stage1_core_type: str = Query(..., description="image | caption | joint-image-text"),
#     stage2_model: str = Query(...),
#     stage2_core_type: str = Query(..., description="image | caption | joint-image-text"),
#     stage1_k: int = Query(100, ge=1),
#     stage2_k: int = Query(10, ge=1),
#     stage1_engine: str = Query("faiss", description="faiss | faiss-hnsw"),
#     stage2_engine: str = Query("faiss", description="faiss | faiss-hnsw"),
# ):
#     try:
#         return ORCH.execute_two_stage(
#             query_text=query,
#             dataset=dataset,
#             stage1_model=stage1_model,
#             stage1_target=stage1_core_type,
#             stage1_k=stage1_k,
#             stage2_model=stage2_model,
#             stage2_target=stage2_core_type,
#             stage2_k=stage2_k,
#             stage1_engine=stage1_engine,
#             stage2_engine=stage2_engine,
#         )
#     except FileNotFoundError as e:
#         raise HTTPException(status_code=404, detail=str(e))
#     except ValueError as e:
#         raise HTTPException(status_code=400, detail=str(e))
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


# # Compat alias for older clients that hit /two_stage
# @app.get("/two_stage")
# def two_stage_compat(
#     query: str = Query(...),
#     dataset: str = Query(...),
#     stage1_model: str = Query(...),
#     stage1_core_type: str = Query(...),
#     stage2_model: str = Query(...),
#     stage2_core_type: str = Query(...),
#     stage1_k: int = Query(100, ge=1),
#     stage2_k: int = Query(10, ge=1),
#     stage1_engine: str = Query("faiss"),
#     stage2_engine: str = Query("faiss"),
# ):
#     try:
#         return ORCH.execute_two_stage(
#             query_text=query,
#             dataset=dataset,
#             stage1_model=stage1_model,
#             stage1_target=stage1_core_type,
#             stage1_k=stage1_k,
#             stage2_model=stage2_model,
#             stage2_target=stage2_core_type,
#             stage2_k=stage2_k,
#             stage1_engine=stage1_engine,
#             stage2_engine=stage2_engine,
#         )
#     except FileNotFoundError as e:
#         raise HTTPException(status_code=404, detail=str(e))
#     except ValueError as e:
#         raise HTTPException(status_code=400, detail=str(e))
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


# @app.get("/rerank/{model}", response_model=RerankResponse)
# def neural_reranking(
#     model: str,
#     query: str = Query(...),
#     topk_list: str = Query(..., description="JSON-encoded list from first call"),
#     K: int = Query(10, ge=1),
# ):
#     """
#     Re-rank the provided candidates by scoring only those items using model.
#     For compatibility with your new minimal output, we accept items without dataset/target.
#     This endpoint will only return (image_path, score, image_abs_path).
#     """
#     try:
#         items = json.loads(topk_list)
#         if not isinstance(items, list) or not items:
#             return {"list_of_top_k": [], "reranking_time": 0.0}

#         # Heuristic: try to infer dataset from absolute path if present, otherwise require it in items.
#         # Since your client likely won't call this when using the new minimal payload, this is best-effort.
#         # If unavailable, raise a helpful error.
#         # We fallback to selected_config.dataset if present.
#         cfg = ORCH.cfg
#         dataset = None
#         for ds in cfg["paths"]["dataset"].keys():
#             base = cfg["paths"]["dataset"][ds]["base_image_path"]
#             if "image_abs_path" in items[0] and items[0]["image_abs_path"].startswith(
#                 base
#             ):
#                 dataset = ds
#                 break

#         if dataset is None and "selected_config" in cfg and "dataset" in cfg["selected_config"]:
#             dataset = cfg["selected_config"]["dataset"]

#         if dataset is None:
#             raise ValueError(
#                 "Cannot infer dataset for reranking (no dataset in items and no "
#                 "selected_config.dataset)."
#             )

#         image_names = [it.get("image_path") for it in items if it.get("image_path")]
#         if not image_names:
#             return {"list_of_top_k": [], "reranking_time": 0.0}

#         emb = ORCH._embedder(model)
#         t0 = get_time()
#         q = emb(query)
#         q = _normalize(q)

#         # Use Flat/IP index for exact rescoring; assume target "caption" for text-based re-ranking
#         s, _ = ORCH._searcher(model, dataset, "caption", "faiss")
#         rows = s.score_all_rows_for_images(q, image_names, top_k=K)
#         dt = get_time() - t0

#         out = [
#             {
#                 "image_path": r["image_path"],
#                 "score": float(r["score"]),
#                 "image_abs_path": _abs_image_path(cfg, dataset, r["image_path"]),
#             }
#             for r in rows
#         ]
#         return {"list_of_top_k": out, "reranking_time": float(dt)}

#     except ValueError as e:
#         raise HTTPException(status_code=400, detail=str(e))
#     except Exception as e:  # noqa: BLE001
#         raise HTTPException(status_code=400, detail=f"Bad topk_list or config: {e}")


# if __name__ == "__main__":
#     import uvicorn

#     # Example:
#     # uvicorn faiss_two_stage_server:app --host 0.0.0.0 --port 5059 --reload
#     uvicorn.run("faiss_two_stage_server:app", host="0.0.0.0", port=5059, reload=True)
