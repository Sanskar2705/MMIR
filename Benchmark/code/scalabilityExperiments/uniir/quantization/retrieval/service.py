#!/usr/bin/env python3
import os
import sys
import json
import tarfile
from typing import Union, List, Dict, Optional

from fastapi import FastAPI, HTTPException

# ---------------- paths ----------------
BASE_DIR = "/mnt/storage/RSystemsBenchmarking/gitProject/Benchmark/data/testing/testing1"
UNIIR_DIR = f"{BASE_DIR}/uniir"
LAION_TAR_DIR = "/mnt/storage/RSystemsBenchmarking/gitProject/Benchmark/data/laion/tar/Worker0/tars"

#  IVF-PQ index paths
FAISS_INDEX_PATH = f"{UNIIR_DIR}/quantization/index/uniir_image_ivfpq.index"
FAISS_NAMES_PATH = f"{UNIIR_DIR}/quantization/index/image_names.json"

#  UniIR local checkpoint
UNIIR_CKPT_PATH = (
    "/mnt/storage/RSystemsBenchmarking/gitProject/Benchmark/data/testing/model/uniir/"
    "ViT-L-14-quickgelu-openai-full.pt"
)

# ---------- disable proxies for local ----------
os.environ["no_proxy"] = "localhost,127.0.0.1"
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["HF_HUB_DISABLE_XET"] = "1"

# import time util
sys.path.append("/mnt/storage/RSystemsBenchmarking/gitProject")
from Benchmark.code.evaluation.time_util import get_time  # noqa: E402

import faiss
import numpy as np
import torch
import open_clip
import warnings


# ---------------- UniIR + FAISS Searcher ----------------
class UniIRFaissIVFPQSearcher:
    """
    UniIR(OpenCLIP) Text → Image FAISS Searcher (IVF-PQ index)

    Extra IVF-PQ parameter:
      - nprobe controls recall/speed tradeoff
    """

    def __init__(
        self,
        index_path: str,
        names_path: str,
        model_path: str,
        model_name: str = "ViT-L-14-quickgelu",
        device: Optional[str] = None,
        use_amp: bool = True,
        nprobe: int = 16,   #  important for IVF
    ):
        self.index_path = index_path
        self.names_path = names_path
        self.model_path = model_path
        self.model_name = model_name

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = use_amp
        self.nprobe = nprobe

        self.index = None
        self.image_names = None
        self.model = None
        self.tokenizer = None

    def _load_openclip_checkpoint(self, model, ckpt_path: str):
        """
        Loads UniIR checkpoint into OpenCLIP model.
        Supports multiple checkpoint formats.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            try:
                import omegaconf
                import torch.serialization
                torch.serialization.add_safe_globals([omegaconf.dictconfig.DictConfig])
            except Exception:
                pass

            loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        if isinstance(loaded, dict):
            state = loaded.get("model") or loaded.get("state_dict") or loaded
            if isinstance(state, dict):
                state = {k.replace("clip_model.", "", 1): v for k, v in state.items()}
                model.load_state_dict(state, strict=False)
                return model
            return state
        return loaded

    def initialize(self) -> bool:
        try:
            print("[INFO] Loading FAISS IVF-PQ index...")
            self.index = faiss.read_index(self.index_path)

            #  IVF-only search param
            try:
                self.index.nprobe = int(self.nprobe)
                print(f"[INFO] IVF nprobe set to {self.nprobe}")
            except Exception:
                print("[WARN] Could not set nprobe (index might not be IVF?)")

            print("[INFO] Loading image-name mapping...")
            with open(self.names_path, "r") as f:
                self.image_names = json.load(f)

            if self.index.ntotal != len(self.image_names):
                raise ValueError(
                    f"Index ntotal ({self.index.ntotal}) != image_names ({len(self.image_names)})"
                )

            # load UniIR(OpenCLIP) model from local checkpoint (NO HF download)
            print("[INFO] Loading UniIR(OpenCLIP) model architecture...")
            model, _, _ = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=None,
                device=self.device
            )

            print(f"[INFO] Loading UniIR checkpoint weights: {self.model_path}")
            model = self._load_openclip_checkpoint(model, self.model_path)

            self.model = model.to(self.device).eval()
            self.tokenizer = open_clip.get_tokenizer(self.model_name)

            print(f"[INFO] Ready: {self.index.ntotal:,} images indexed")
            return True

        except Exception as e:
            print(f"[ERROR] UniIR IVF-PQ initialize failed: {e}")
            return False

    def set_nprobe(self, nprobe: int):
        """Dynamically change nprobe during runtime."""
        if self.index is None:
            return
        try:
            self.index.nprobe = int(nprobe)
            self.nprobe = int(nprobe)
        except Exception:
            pass

    @torch.no_grad()
    def encode_query(self, text: str):
        if text is None or not text.strip():
            return None
        try:
            start = get_time()
            tokens = self.tokenizer([text]).to(self.device)

            if self.device.startswith("cuda") and self.use_amp:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    emb = self.model.encode_text(tokens)
            else:
                emb = self.model.encode_text(tokens)

            emb = emb.float()
            emb = emb / emb.norm(dim=-1, keepdim=True)

            encoding_time = get_time() - start
            return {
                "embedding": emb.detach().cpu().numpy().astype("float32").flatten(),
                "encoding_time": float(encoding_time),
            }
        except Exception as e:
            print(f"[ERROR] encode_query failed: {e}")
            return None

    def vector_search(self, query_embedding: np.ndarray, top_k: int = 10):
        if query_embedding is None:
            return {"results": [], "query_time": 0.0}
        try:
            start = get_time()
            q = query_embedding.reshape(1, -1).astype("float32")
            scores, idxs = self.index.search(q, top_k)
            query_time = get_time() - start

            results = []
            for fid, score in zip(idxs[0].tolist(), scores[0].tolist()):
                if fid < 0:
                    continue
                results.append({
                    "image_path": self.image_names[fid],
                    "score": float(score),
                })

            return {"results": results, "query_time": float(query_time)}
        except Exception as e:
            print(f"[ERROR] vector_search failed: {e}")
            return {"results": [], "query_time": 0.0}

    def search(self, query_text: str, top_k: int = 10):
        enc = self.encode_query(query_text)
        if enc is None:
            return {"results": [], "encoding_time": 0.0, "query_time": 0.0}

        out = self.vector_search(enc["embedding"], top_k)
        return {
            "results": out["results"],
            "encoding_time": enc["encoding_time"],
            "query_time": out["query_time"],
        }


# ---------------- init ----------------
app = FastAPI()

searcher = UniIRFaissIVFPQSearcher(
    index_path=FAISS_INDEX_PATH,
    names_path=FAISS_NAMES_PATH,
    model_path=UNIIR_CKPT_PATH,
    model_name="ViT-L-14-quickgelu",
    use_amp=True,
    nprobe=16,
)

ok = searcher.initialize()
print("[DEBUG] initialize() returned:", ok, flush=True)
if not ok:
    raise RuntimeError("UniIR IVF-PQ FAISS initialize() failed (check logs above)")


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
            cap = (d.get("caption") or "").strip()
            if cap:
                out.append({"id": fid, "caption": cap})
    return out


# ---------------- endpoints ----------------
@app.get("/uniir_faiss_ivfpq_search")
async def uniir_faiss_ivfpq_search(
    q: Union[str, None] = None,
    k: int = 10,
    nprobe: int = 16,
):
    """
    IVF-PQ search endpoint:
      - nprobe controls recall/speed
    """
    if not q:
        return {"list_of_top_k": [], "query_time": 0.0, "encoding_time": 0.0, "retrieval_time": 0.0}

    searcher.set_nprobe(nprobe)

    start = get_time()
    out = searcher.search(q, top_k=k)
    total = get_time() - start

    return {
        "list_of_top_k": out["results"],
        "query_time": float(total),
        "encoding_time": float(out.get("encoding_time", 0.0)),
        "retrieval_time": float(out.get("query_time", 0.0)),
        "nprobe": int(nprobe),
    }


@app.get("/laion_tar_sample")
async def laion_tar_sample(tar_name: str, n: int = 20):
    tar_path = os.path.join(LAION_TAR_DIR, tar_name)
    if not os.path.exists(tar_path):
        raise HTTPException(404, f"Tar not found: {tar_path}")

    samples = sample_captions_from_tar(tar_path, n=n)
    return {"tar": tar_name, "samples": samples, "n": len(samples)}
