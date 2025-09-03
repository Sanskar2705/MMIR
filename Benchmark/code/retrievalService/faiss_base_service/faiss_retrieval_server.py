#!/usr/bin/env python3
"""
faiss_retrieval_server.py
─────────────────────────
FastAPI server that does cross-modal retrieval **without any model-specific
code in this file** – all model logic lives in embed_utils.py
"""
#curl --noproxy '*' -X POST http://localhost:5052/refresh

import sys, json, os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging, warnings

import faiss
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query

os.environ["HF_HUB_DISABLE_XET"] = "1"

# ─── suppress noisy libs when debug is off ───
def _configure_runtime_logging(cfg: Dict):
    debug = bool(cfg.get("debug_logs", False))
    level = logging.DEBUG if debug else logging.WARNING

    # Root + common libraries
    logging.getLogger().setLevel(level)
    for name in [
        "uvicorn", "uvicorn.error", "uvicorn.access",
        "fastapi", "asyncio", "httpx", "urllib3", "PIL",
    ]:
        logging.getLogger(name).setLevel(level)

    # HuggingFace / transformers (if used anywhere down the stack)
    try:
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_debug() if debug else hf_logging.set_verbosity_error()
    except Exception:
        pass

    # Env + warnings to keep the console clean
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTHONASYNCIODEBUG"] = "1" if debug else "0"
    if not debug:
        warnings.filterwarnings("ignore")


# ───── project imports ───── #
sys.path.append("/mnt/storage/RSystemsBenchmarking/gitProject")  # project root
from Benchmark.config.config_utils import load_config
from Benchmark.code.evaluation.time_util import get_time      # noqa: E402
from embed_utils import get_embedder     # NEW!

# ───── generic helpers ───── #
def l2norm(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x, dim=-1)


# ───── FAISS wrapper ───── #
class FaissSearcher:
    """Wrap FAISS index + metadata + embedding fn."""
    def __init__(
        self,
        idx_path: Path,
        meta_path: Path,
        embed_fn,
        base_img_dir: Path,
        caption_lookup: Optional[Dict[str, str]] = None,
        target: str = "image",  # "image" | "caption" | "joint"
    ):
        self.index          = faiss.read_index(str(idx_path))
        self.meta           = json.loads(meta_path.read_text())
        self.embed          = embed_fn
        self.base_img_dir   = base_img_dir
        self.caption_lookup = caption_lookup or {}
        self.target         = target

    def search(self, query: str, k: int):
        # embedding
        t0     = get_time()
        vec    = self.embed(query)[None, :]          # (1, D)
        emb_ms = get_time() - t0

        # retrieval
        t1     = get_time()
        D, I   = self.index.search(vec, k)          # IP == cosine (vectors are L2-normed)
        ret_ms = get_time() - t1

        hits = []
        for rank, (idx, score) in enumerate(zip(I[0], D[0]), start=1):
            meta_row = self.meta[idx]
            if isinstance(meta_row, dict):
                img_name = meta_row["image_name"]
                caption  = meta_row.get("caption", "")
            else:
                # image index stores just names (list[str]); grab caption if we have it
                img_name = meta_row
                caption  = self.caption_lookup.get(img_name, "")
            hits.append({
                "image_path"    : img_name,
                "image_abs_path": str((self.base_img_dir / os.path.basename(img_name)).resolve()),
                "score"         : float(score),
                "rank"          : rank,
                "caption"       : caption,
                "target"        : self.target,
            })
        return hits, emb_ms, ret_ms


# ───── registry builder ───── #
def build_registry(cfg: Dict, faiss_dir: Path):
    """Dynamically discover indexes & wire them to the correct embedder."""
    dev        = "cuda" if torch.cuda.is_available() else "cpu"
    registry: Dict[Tuple[str, str, str], FaissSearcher] = {}

    # 1) Scan index directory to know what exists
    for idx_file in faiss_dir.glob("*_faiss_*.index"):
        # expected stem: <model>_<dataset>_faiss_<img|txt|joint>
        parts = idx_file.stem.split("_")
        if len(parts) < 4:
            continue
        model_tag, dataset, _, kind = parts

        if kind == "img":
            target = "image"
            meta_file = faiss_dir / f"{model_tag}_{dataset}_img_names.json"
        elif kind == "txt":
            target = "caption"
            meta_file = faiss_dir / f"{model_tag}_{dataset}_txt_meta.json"
        elif kind == "joint-image-text":
            target = "joint-image-text"
            meta_file = faiss_dir / f"{model_tag}_{dataset}_joint-image-text_meta.json"
        else:
            continue

        if not meta_file.exists():
            # skip incomplete pairs
            continue

        # 2) Get / cache embedder for this model (text-only queries)
        try:
            embed_fn = EMBEDDERS[model_tag]
        except KeyError:
            embed_fn = get_embedder(model_tag, cfg, dev)
            EMBEDDERS[model_tag] = embed_fn

        # 3) Caption lookup only needed for image index to enrich results
        caption_lookup = {}
        if target == "image":
            txt_meta = faiss_dir / f"{model_tag}_{dataset}_txt_meta.json"
            if txt_meta.exists():
                try:
                    caption_lookup = {
                        m["image_name"]: m.get("caption", "")
                        for m in json.loads(txt_meta.read_text())
                    }
                except Exception:
                    caption_lookup = {}

        base_dir = Path(cfg["paths"]["dataset"][dataset]["base_image_path"])
        registry[(model_tag, dataset, target)] = FaissSearcher(
            idx_path=idx_file,
            meta_path=meta_file,
            embed_fn=embed_fn,
            base_img_dir=base_dir,
            caption_lookup=caption_lookup if target == "image" else None,
            target=target,
        )

    return registry


# ───── FastAPI ───── #
cfg        = load_config()
_configure_runtime_logging(cfg)  # ← apply logging policy based on debug_logs
faiss_dir  = Path(cfg["vector_store"]["faiss"]["index_dir"])
EMBEDDERS  = {}                                # lazy-loaded cache
SEARCHERS  = build_registry(cfg, faiss_dir)
TOP_K      = 10

app = FastAPI(title="FAISS Retrieval Server", version="2.2")

@app.get("/available")
def available():
    return {"available": [f"{m}_{t}_{d}_faiss" for (m, d, t) in SEARCHERS.keys()]}

# Route: <model>_<target>_<dataset>_faiss
#   target ∈ {"image", "caption", "joint"}
@app.get("/{combo}")
def search(
    combo: str,
    q: str = Query(..., description="Text query"),
    k: int = Query(TOP_K, ge=1, le=100, description="Top-K to return"),
):
    """
    Expected combo pattern:  <model>_<target>_<dataset>_faiss
      e.g. uniir_image_coco_faiss
            uniir_caption_coco_faiss
            uniir_joint_coco_faiss
    """
    parts = combo.lower().split("_")
    if len(parts) != 4 or parts[3] != "faiss":
        raise HTTPException(
            status_code=400,
            detail="Path must look like <model>_<target>_<dataset>_faiss "
                   "with target in {image, caption,text, joint-image-text}",
        )

    model, target, dataset, _ = parts
    if target not in {"image", "caption", "text", "joint-image-text"}:
        raise HTTPException(status_code=400, detail="target must be one of {image, caption, text, joint-image-text}")
    
    # normalize aliases
    if target == "text":
        target = "caption"

    key = (model, dataset, target)
    if key not in SEARCHERS:
        raise HTTPException(
            status_code=404,
            detail=f"No FAISS index for combo {key}. "
                   f"Available={list(SEARCHERS.keys())}",
        )

    hits, enc_ms, ret_ms = SEARCHERS[key].search(q, k)
    return {
        "query_time"    : enc_ms + ret_ms,
        "encoding_time" : enc_ms,
        "retrieval_time": ret_ms,
        "list_of_top_k" : hits,
    }

@app.post("/refresh")
def refresh():
    global cfg, faiss_dir, EMBEDDERS, SEARCHERS
    cfg = load_config()                 # reload config.yaml
    _configure_runtime_logging(cfg)  
    faiss_dir = Path(cfg["vector_store"]["faiss"]["index_dir"])
    EMBEDDERS.clear()                   # clear cached embedders -> prompt reloads
    SEARCHERS = build_registry(cfg, faiss_dir)
    return {"ok": True, "available": [f"{m}_{t}_{d}_faiss" for (m,d,t) in SEARCHERS.keys()]}
