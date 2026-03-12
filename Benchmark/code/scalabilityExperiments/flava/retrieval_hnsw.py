#!/usr/bin/env python3
import os
import sys
import json
import tarfile
from typing import Union, List, Dict

from fastapi import FastAPI, HTTPException

# ---------------- paths ----------------
BASE_DIR = "/mnt/storage/RSystemsBenchmarking/gitProject/Benchmark/data/testing/testing1"
FLAVA_DIR = f"{BASE_DIR}/flava"
LAION_TAR_DIR = "/mnt/storage/RSystemsBenchmarking/gitProject/Benchmark/data/laion/tar/Worker0/tars"

#  CHANGE THIS to your HNSW index filename
FAISS_INDEX_PATH = f"{FLAVA_DIR}/index/flava_image_hnsw.index"
FAISS_NAMES_PATH = f"{FLAVA_DIR}/index/image_names.json"

# import your class from testing1/flava/retrieval.py
sys.path.insert(0, FLAVA_DIR)
from retrieval import FlavaFaissSemanticSearcher  # noqa: E402

sys.path.append("/mnt/storage/RSystemsBenchmarking/gitProject")
from Benchmark.code.evaluation.time_util import get_time  # noqa: E402


# ---------------- init ----------------
app = FastAPI()

searcher = FlavaFaissSemanticSearcher(
    index_path=FAISS_INDEX_PATH,
    names_path=FAISS_NAMES_PATH,
    model_path="facebook/flava-full",
    use_hnsw=True,      #  enable HNSW efSearch
    ef_search=128,
)

ok = searcher.initialize()
print("[DEBUG] HNSW initialize() returned:", ok, flush=True)
if not ok:
    raise RuntimeError("HNSW initialize() failed")


# ---------------- helpers ----------------
def sample_captions_from_tar(tar_path: str, n: int = 20) -> List[Dict]:
    out = []
    with tarfile.open(tar_path, "r") as tf:
        members = sorted([x for x in tf.getnames() if x.endswith(".json")])
        for m in members[:n]:
            fid = os.path.basename(m).replace(".json", "")
            f = tf.extractfile(m)
            if not f:
                continue
            d = json.loads(f.read().decode("utf-8", errors="ignore"))
            cap = d.get("caption", "")
            if cap:
                out.append({"id": fid, "caption": cap})
    return out


# ---------------- endpoints ----------------
@app.get("/flava_hnsw_search")
def flava_hnsw_search(q: Union[str, None] = None, k: int = 10, ef: int = 128):
    """
    HNSW search
    ef controls speed/quality (higher = better recall, slower)
    """
    if not q:
        return {"list_of_top_k": [], "query_time": 0.0, "encoding_time": 0.0, "retrieval_time": 0.0}

    # update efSearch dynamically per request
    try:
        if hasattr(searcher.index, "hnsw"):
            searcher.index.hnsw.efSearch = int(ef)
    except Exception:
        pass

    start = get_time()
    out = searcher.search(q, top_k=k)
    total = get_time() - start

    return {
        "list_of_top_k": out["results"],
        "query_time": float(total),
        "encoding_time": float(out.get("encoding_time", 0.0)),
        "retrieval_time": float(out.get("query_time", 0.0)),
        "efSearch": int(ef),
    }


@app.get("/laion_tar_sample_hnsw")
def laion_tar_sample_hnsw(tar_name: str, n: int = 20):
    tar_path = os.path.join(LAION_TAR_DIR, tar_name)
    if not os.path.exists(tar_path):
        raise HTTPException(404, f"Tar not found: {tar_path}")

    samples = sample_captions_from_tar(tar_path, n=n)
    return {"tar": tar_name, "samples": samples, "n": len(samples)}


@app.get("/laion_tar_retrieve_hnsw")
def laion_tar_retrieve_hnsw(tar_name: str, n: int = 20, k: int = 10, ef: int = 128):
    tar_path = os.path.join(LAION_TAR_DIR, tar_name)
    if not os.path.exists(tar_path):
        raise HTTPException(404, f"Tar not found: {tar_path}")

    try:
        if hasattr(searcher.index, "hnsw"):
            searcher.index.hnsw.efSearch = int(ef)
    except Exception:
        pass

    samples = sample_captions_from_tar(tar_path, n=n)

    start = get_time()
    total_enc, total_ret = 0.0, 0.0
    items = []

    for s in samples:
        out = searcher.search(s["caption"], top_k=k)
        total_enc += float(out.get("encoding_time", 0.0))
        total_ret += float(out.get("query_time", 0.0))
        items.append({"id": s["id"], "caption": s["caption"], "results": out["results"]})

    total = get_time() - start

    return {
        "tar": tar_name,
        "n": len(samples),
        "k": k,
        "efSearch": int(ef),
        "query_time": float(total),
        "encoding_time": float(total_enc),
        "retrieval_time": float(total_ret),
        "items": items,
    }
