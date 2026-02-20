# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# preflmr_colbert_server_all3.py

# Serve OFFICIAL ColBERT indices built by `build_preflmr_colbert_indices_all3.py`:

#   /available
#   /flmr_caption_<dataset>_colbert?q=...&k=10
#   /flmr_joint_<dataset>_colbert?q=...&k=10
#   /flmr_image_<dataset>_colbert?q=...&k=10

# Environment knobs:
#   - FLMR_INDEX_PREFIX: prefer indices named "<prefix>_{caption|joint|image}"
#   - FLMR_NBITS: nbits used at build time (default: 8)

# Requires:
#   - FAISS (GPU or CPU)
#   - ColBERT repo root on sys.path (bootstrapped below)
#   - Your project `config.yaml` loader (load_config)
# """

# # --- BOOTSTRAP: ensure ColBERT repo root (has 'utility' & 'colbert') is on sys.path ---
# import sys, os, json, time
# from pathlib import Path
# ROOT = Path(__file__).resolve().parents[1]  # e.g. /mnt/storage/ziv/FLMR_CURRENT
# CAND = ROOT / "FLMR" / "third_party" / "ColBERT"
# if (CAND / "utility").exists() and str(CAND) not in sys.path:
#     sys.path.insert(0, str(CAND))
# # ---------------------------------------------------------------------------------------

# # Offline-friendly defaults
# os.environ.setdefault("HF_HUB_OFFLINE", "1")
# os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# os.environ["HF_HUB_DISABLE_XET"] = "1"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# # FAISS is needed by ColBERT Searcher
# try:
#     import faiss  # noqa: F401
# except Exception as e:
#     raise RuntimeError("FAISS is required for ColBERT search. Install faiss-gpu or faiss-cpu.") from e

# from typing import Dict, Tuple, List

# import torch
# from fastapi import FastAPI, HTTPException, Query

# from flmr import (
#     create_searcher,
#     search_custom_collection,
#     FLMRModelForRetrieval,
#     FLMRQueryEncoderTokenizer,
#     FLMRContextEncoderTokenizer,
#     FLMRConfig,
# )

# # Your project config loader
# sys.path.append("/mnt/storage/RSystemsBenchmarking/gitProject")
# from Benchmark.config.config_utils import load_config  # noqa: E402


# # ---------------- FLMR query stack ---------------- #

# def _device() -> torch.device:
#     return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def load_preflmr_query_stack(model_ckpt: str, dev: torch.device):
#     flmr_config = FLMRConfig.from_pretrained(model_ckpt)
#     qtok = FLMRQueryEncoderTokenizer.from_pretrained(
#         model_ckpt, text_config=flmr_config.text_config, subfolder="query_tokenizer"
#     )
#     dtok = FLMRContextEncoderTokenizer.from_pretrained(  # noqa: F841 (kept to mirror model init)
#         model_ckpt, text_config=flmr_config.text_config, subfolder="context_tokenizer"
#     )
#     model = FLMRModelForRetrieval.from_pretrained(
#         model_ckpt, query_tokenizer=qtok, context_tokenizer=dtok
#     ).to(dev).eval()
#     return qtok, model

# @torch.no_grad()
# def encode_queries(qtok, model, texts: List[str], dev: torch.device, max_len: int = 128):
#     enc = qtok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
#     enc = {k: v.to(dev) for k, v in enc.items()}
#     out = model.query(
#         input_ids=enc["input_ids"],
#         attention_mask=enc["attention_mask"],
#         pixel_values=None, image_features=None,
#         concat_output_from_vision_encoder=False,
#         concat_output_from_text_encoder=True,
#     )
#     return out.late_interaction_output.detach().cpu()  # (B, Lq, D)


# # ---------------- registry of indices ---------------- #

# def load_meta(index_root: Path, experiment: str, index_name: str) -> List[dict]:
#     """
#     Builder drops collection_meta.json next to <experiment>/<index_name>/
#     """
#     p = index_root / experiment / index_name / "collection_meta.json"
#     return json.loads(p.read_text()) if p.exists() else []

# def _pick_index_name(base: Path, suffix: str, prefix: str | None) -> str | None:
#     """
#     Given experiment dir (<root>/<experiment>), pick an index dir whose name ends with '_<suffix>'.
#     If FLMR_INDEX_PREFIX is set, prefer '<prefix>_<suffix>'; else pick lexicographically last.
#     """
#     cand = [d.name for d in base.iterdir() if d.is_dir() and d.name.endswith(f"_{suffix}")]
#     if not cand:
#         return None
#     cand.sort()
#     if prefix:
#         exact = f"{prefix}_{suffix}"
#         if exact in cand:
#             return exact
#     return cand[-1]

# def build_registry(cfg, nbits: int) -> Dict[Tuple[str, str], dict]:
#     """
#     Returns mapping: (dataset, target) -> {searcher, meta, index_name, base_img_dir}
#       target ∈ {'caption','joint','image'}
#     """
#     reg = {}
#     ds_map = cfg["vector_store"]["flmr"]["experiment_name_map"]
#     root   = Path(cfg["vector_store"]["flmr"]["root_dir"])
#     prefix = os.environ.get("FLMR_INDEX_PREFIX", "").strip() or None
#     use_gpu = torch.cuda.is_available()  # searcher scoring on GPU if available

#     for dataset, experiment in ds_map.items():
#         exp_dir = root / experiment
#         if not exp_dir.exists():
#             continue
#         for target in ("caption","joint","image"):
#             index_name = _pick_index_name(exp_dir, target, prefix)
#             if not index_name:
#                 continue
#             try:
#                 searcher = create_searcher(
#                     index_root_path=str(root),
#                     index_experiment_name=experiment,
#                     index_name=index_name,
#                     nbits=nbits,
#                     use_gpu=use_gpu,
#                 )
#                 meta = load_meta(root, experiment, index_name)
#                 base_img_dir = Path(cfg["paths"]["dataset"][dataset]["base_image_path"])
#                 reg[(dataset, target)] = {
#                     "searcher": searcher,
#                     "meta": meta,
#                     "index_name": index_name,
#                     "base_img_dir": base_img_dir,
#                     "experiment": experiment,
#                 }
#             except Exception as e:
#                 print(f"[warn] failed loading {experiment}/{index_name}: {e}")
#     return reg


# # ---------------- FastAPI ---------------- #

# cfg         = load_config()
# DEV         = _device()
# MODEL_CK    = cfg["models"]["flmr"]["checkpoint"]
# NBITS       = int(os.environ.get("FLMR_NBITS", "8"))

# QTOK, QMODEL = load_preflmr_query_stack(MODEL_CK, DEV)
# REGISTRY     = build_registry(cfg, NBITS)
# TOP_K        = int(cfg.get("k", 10))

# app = FastAPI(title="PreFLMR ColBERT Server (caption/joint/image)", version="1.2")

# @app.get("/available")
# def available():
#     combos, details = [], {}
#     for (dataset, target), info in REGISTRY.items():
#         combos.append(f"flmr_{target}_{dataset}_colbert")
#         details[f"{dataset}:{target}"] = {
#             "index_name": info["index_name"],
#             "experiment": info["experiment"],
#         }
#     return {"available": combos, "loaded_indices": details, "nbits": NBITS}

# @app.get("/{combo}")
# def search(combo: str, q: str = Query(...), k: int = Query(TOP_K, ge=1, le=100)):
#     parts = combo.lower().split("_")
#     if len(parts) != 4 or parts[0] != "flmr" or parts[3] != "colbert":
#         raise HTTPException(status_code=400, detail="Path must be flmr_{caption|joint|image}_{dataset}_colbert")
#     _, target, dataset, _ = parts
#     if target not in {"caption","joint","image"}:
#         raise HTTPException(status_code=400, detail="target must be one of {caption, joint, image}")
#     key = (dataset, target)
#     if key not in REGISTRY:
#         raise HTTPException(status_code=404, detail=f"No index loaded for {key}. Check /available or FLMR_INDEX_PREFIX/FLMR_NBITS.")

#     # encode query
#     t0 = time.time()
#     Q = encode_queries(QTOK, QMODEL, [q], DEV, max_len=128)  # (1, Lq, D)
#     enc_ms = (time.time() - t0) * 1000.0

#     # search (official)
#     searcher = REGISTRY[key]["searcher"]
#     t1 = time.time()
#     ranking = search_custom_collection(
#         searcher=searcher,
#         queries={0: q},
#         query_embeddings=Q,
#         num_document_to_retrieve=k,
#         remove_zero_tensors=True,   # recommended w/ PreFLMR
#     )
#     ret_ms = (time.time() - t1) * 1000.0

#     # unpack
#     ranking_dict = ranking.todict()
#     docs = ranking_dict[0]
#     meta = REGISTRY[key]["meta"]
#     base_img_dir = REGISTRY[key]["base_img_dir"]

#     hits = []
#     for rank, (doc_idx, _docid, score) in enumerate(docs, start=1):
#         md = meta[doc_idx] if 0 <= doc_idx < len(meta) else {}
#         rel = md.get("image_name", "")
#         abs_img = (base_img_dir / Path(rel).name).resolve()
#         hits.append({
#             "rank": rank,
#             "score": float(score),
#             "image_path": rel,
#             "image_abs_path": str(abs_img),
#             "caption": md.get("caption", ""),
#             "target": target
#         })

#     return {
#         "query_time": enc_ms + ret_ms,
#         "encoding_time": enc_ms,
#         "retrieval_time": ret_ms,
#         "list_of_top_k": hits
#     }

# @app.post("/refresh")
# def refresh():
#     global REGISTRY, QTOK, QMODEL, DEV
#     DEV = _device()
#     QTOK, QMODEL = load_preflmr_query_stack(MODEL_CK, DEV)
#     REGISTRY = build_registry(cfg, NBITS)
#     return {"ok": True, "available": [f"flmr_{t}_{d}_colbert" for (d,t) in REGISTRY.keys()], "nbits": NBITS}


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("preflmr_colbert_server_all3:app", host="0.0.0.0", port=5052, reload=False)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PreFLMR ColBERT Server — instruction+query *pair* encoding (no 'blank' mode).

Endpoints:
  /available
  /flmr_{caption|joint|image}_{dataset}_colbert?q=...&k=10

Behavior:
  - Encodes as a sentence-pair: [CLS] INSTR [SEP] QUERY [SEP]
  - Zero-masks tokens up to first [SEP] (instruction masking)
  - Text-only path (no vision), paper-faithful Q2T setup
  - Instruction prefix read from YAML: models.flmr.query_prefix
"""

# --- BOOTSTRAP ColBERT path ---
import sys, os, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
CAND = ROOT / "FLMR" / "third_party" / "ColBERT"
if (CAND / "utility").exists() and str(CAND) not in sys.path:
    sys.path.insert(0, str(CAND))
# --------------------------------

# Offline-friendly defaults
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# FAISS needed by ColBERT Searcher
try:
    import faiss  # noqa: F401
except Exception as e:
    raise RuntimeError("FAISS is required for ColBERT search. Install faiss-gpu or faiss-cpu.") from e

from typing import Dict, Tuple, List
import torch
from fastapi import FastAPI, HTTPException, Query

from flmr import (
    create_searcher,
    search_custom_collection,
    FLMRModelForRetrieval,
    FLMRQueryEncoderTokenizer,
    FLMRContextEncoderTokenizer,
    FLMRConfig,
)

# Your project config loader
sys.path.append("/mnt/storage/RSystemsBenchmarking/gitProject")
from Benchmark.config.config_utils import load_config  # noqa: E402


# ---------------- device & stack ---------------- #
def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_preflmr_query_stack(model_ckpt: str, dev: torch.device):
    cfg = FLMRConfig.from_pretrained(model_ckpt)
    qtok = FLMRQueryEncoderTokenizer.from_pretrained(
        model_ckpt, text_config=cfg.text_config, subfolder="query_tokenizer"
    )
    dtok = FLMRContextEncoderTokenizer.from_pretrained(  # noqa: F841
        model_ckpt, text_config=cfg.text_config, subfolder="context_tokenizer"
    )
    model = FLMRModelForRetrieval.from_pretrained(
        model_ckpt, query_tokenizer=qtok, context_tokenizer=dtok
    ).to(dev).eval()
    return cfg, qtok, model


# ---------------- helpers: masking ---------------- #
def _zero_up_to_first_sep(q_emb: torch.Tensor, input_ids: torch.Tensor, sep_id: int) -> torch.Tensor:
    """
    Zero out token rows up to and including the first [SEP] for each example.
    q_emb: (B, Lq, D)  input_ids: (B, Lq)
    """
    q = q_emb.clone()
    bsz, _ = input_ids.shape
    for i in range(bsz):
        ids = input_ids[i].tolist()
        try:
            j = ids.index(sep_id)  # first [SEP]
            q[i, : j + 1, :] = 0.0
        except ValueError:
            pass
    return q


# ---------------- encoder (instruction + query pair) ---------------- #
@torch.no_grad()
def encode_pair(qtok, model, cfg: FLMRConfig, text: str, instr: str, dev: torch.device, max_len: int = 128):
    """
    Encode as sentence-pair: instr (segment A) + text (segment B), no pixels.
    Produces [CLS] instr [SEP] text [SEP], then masks instruction tokens.
    """
    enc = qtok(
        text=[instr], text_pair=[text],
        padding=True, truncation=True, max_length=max_len, return_tensors="pt"
    )
    enc = {k: v.to(dev) for k, v in enc.items()}
    kwargs = {}
    if "token_type_ids" in enc:
        kwargs["token_type_ids"] = enc["token_type_ids"]

    out = model.query(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        pixel_values=None, image_features=None,
        concat_output_from_vision_encoder=False,
        concat_output_from_text_encoder=True,
        **kwargs
    )
    Q = out.late_interaction_output.detach().cpu()  # (1, Lq, D)
    sep_id = getattr(qtok, "sep_token_id", 102)
    return _zero_up_to_first_sep(Q, enc["input_ids"].detach().cpu(), sep_id)


# ---------------- registry of indices ---------------- #
def load_meta(index_root: Path, experiment: str, index_name: str) -> List[dict]:
    p = index_root / experiment / index_name / "collection_meta.json"
    return json.loads(p.read_text()) if p.exists() else []

def _pick_index_name(base: Path, suffix: str, prefix: str | None) -> str | None:
    cand = [d.name for d in base.iterdir() if d.is_dir() and d.name.endswith(f"_{suffix}")]
    if not cand:
        return None
    cand.sort()
    if prefix:
        exact = f"{prefix}_{suffix}"
        if exact in cand:
            return exact
    return cand[-1]

def build_registry(cfg, nbits: int) -> Dict[Tuple[str, str], dict]:
    reg = {}
    ds_map = cfg["vector_store"]["flmr"]["experiment_name_map"]
    root   = Path(cfg["vector_store"]["flmr"]["root_dir"])
    prefix = os.environ.get("FLMR_INDEX_PREFIX", "").strip() or None
    use_gpu = torch.cuda.is_available()

    for dataset, experiment in ds_map.items():
        exp_dir = root / experiment
        if not exp_dir.exists():
            continue
        for target in ("caption","joint","image"):
            index_name = _pick_index_name(exp_dir, target, prefix)
            if not index_name:
                continue
            try:
                searcher = create_searcher(
                    index_root_path=str(root),
                    index_experiment_name=experiment,
                    index_name=index_name,
                    nbits=nbits,
                    use_gpu=use_gpu,
                )

                try:
                    idx = getattr(searcher, "faiss_index", None)
                    if idx is None:
                        idx = getattr(getattr(searcher, "index", None), "faiss_index", None)
                    print("[FAISS] type:", type(idx))
                    if idx is not None:
                        import faiss
                        try:
                            ivf = faiss.extract_index_ivf(idx)
                            print(f"[FAISS] IVF detected: nlist={ivf.nlist}")
                        except Exception:
                            print("[FAISS] Not an IVF index (likely Flat or PQ-only).")
                except Exception as e:
                    print(f"[FAISS] Could not introspect index: {e}")
                # -----------------------------------




                meta = load_meta(root, experiment, index_name)
                base_img_dir = Path(cfg["paths"]["dataset"][dataset]["base_image_path"])
                reg[(dataset, target)] = {
                    "searcher": searcher,
                    "meta": meta,
                    "index_name": index_name,
                    "base_img_dir": base_img_dir,
                    "experiment": experiment,
                }
            except Exception as e:
                print(f"[warn] failed loading {experiment}/{index_name}: {e}")
    return reg


# ---------------- FastAPI ---------------- #
cfg         = load_config()
DEV         = _device()
MODEL_CK    = cfg["models"]["flmr"]["checkpoint"]
NBITS       = int(os.environ.get("FLMR_NBITS", "8"))

FLMR_CFG, QTOK, QMODEL = load_preflmr_query_stack(MODEL_CK, DEV)
REGISTRY     = build_registry(cfg, NBITS)
TOP_K        = int(cfg.get("k", 10))

# Instruction prefix from YAML (can be empty string)
QUERY_PREFIX = (cfg.get("models", {}).get("flmr", {}).get("query_prefix") or "").strip()

app = FastAPI(title="PreFLMR ColBERT Server (pair-encoding only)", version="2.1")

@app.get("/available")
def available():
    combos, details = [], {}
    for (dataset, target), info in REGISTRY.items():
        combos.append(f"flmr_{target}_{dataset}_colbert")
        details[f"{dataset}:{target}"] = {
            "index_name": info["index_name"],
            "experiment": info["experiment"],
        }
    return {"available": combos, "loaded_indices": details, "nbits": NBITS, "query_prefix": QUERY_PREFIX}

@app.get("/{combo}")
def search(
    combo: str,
    q: str = Query(..., description="User text query"),
    k: int = Query(TOP_K, ge=1, le=100),
):
    """
    Path must be: flmr_{caption|joint|image}_{dataset}_colbert
    (No mode segment; always pair-encoding text-only.)
    """
    parts = combo.lower().split("_")
    if len(parts) != 4 or parts[0] != "flmr" or parts[3] != "colbert":
        raise HTTPException(status_code=400, detail="Path must be flmr_{caption|joint|image}_{dataset}_colbert")
    _, target, dataset, _ = parts
    if target not in {"caption","joint","image"}:
        raise HTTPException(status_code=400, detail="target must be one of {caption, joint, image}")

    key = (dataset, target)
    if key not in REGISTRY:
        raise HTTPException(status_code=404, detail=f"No index loaded for {key}. Check /available or FLMR_INDEX_PREFIX/FLMR_NBITS.")

    # --- encode query (instruction+query pair, masked) ---
    t0 = time.time()
    Q = encode_pair(QTOK, QMODEL, FLMR_CFG, text=q, instr=QUERY_PREFIX, dev=DEV, max_len=128)
    enc_ms^ = (time.time() - t0) * 1000.0

    # --- search (official) ---
    searcher = REGISTRY[key]["searcher"]
    t1 = time.time()
    ranking = search_custom_collection(
        searcher=searcher,
        queries={0: q},
        query_embeddings=Q,
        num_document_to_retrieve=k,
        remove_zero_tensors=True,   # drops masked (zeroed) instruction rows
    )
    ret_ms = (time.time() - t1) * 1000.0

    # --- format results ---
    ranking_dict = ranking.todict()
    docs = ranking_dict[0]
    meta = REGISTRY[key]["meta"]
    base_img_dir = REGISTRY[key]["base_img_dir"]

    hits = []
    for rank, (doc_idx, _docid, score) in enumerate(docs, start=1):
        md = meta[doc_idx] if 0 <= doc_idx < len(meta) else {}
        rel = md.get("image_name", "")
        abs_img = (base_img_dir / Path(rel).name).resolve()
        hits.append({
            "rank": rank,
            "score": float(score),
            "image_path": rel,
            "image_abs_path": str(abs_img),
            "caption": md.get("caption", ""),
            "target": target,
        })

    return {
        "query": q,
        "query_time": enc_ms + ret_ms,
        "encoding_time": enc_ms,
        "retrieval_time": ret_ms,
        "list_of_top_k": hits
    }

@app.post("/refresh")
def refresh():
    global REGISTRY, QTOK, QMODEL, DEV, FLMR_CFG, QUERY_PREFIX
    DEV = _device()
    FLMR_CFG, QTOK, QMODEL = load_preflmr_query_stack(MODEL_CK, DEV)
    REGISTRY = build_registry(cfg, NBITS)
    QUERY_PREFIX = (cfg.get("models", {}).get("flmr", {}).get("query_prefix") or "").strip()
    return {
        "ok": True,
        "available": [f"flmr_{t}_{d}_colbert" for (d, t) in REGISTRY.keys()],
        "nbits": NBITS,
        "query_prefix": QUERY_PREFIX
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("preflmr_colbert_server_all3:app", host="0.0.0.0", port=5052, reload=False)
