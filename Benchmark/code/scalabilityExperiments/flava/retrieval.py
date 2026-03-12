#!/usr/bin/env python3
"""
FLAVA Semantic Search using FAISS (Text → Image)

FastAPI-ready searcher class.
Returns results in same structure as your Solr searcher:
{
  "results": [{"image_path": "...", "score": float}, ...],
  "encoding_time": float,
  "query_time": float
}
"""

import os
import json
import time
from typing import Dict, List, Optional

import faiss
import numpy as np
import torch
from transformers import FlavaProcessor, FlavaModel

os.environ['no_proxy'] = 'localhost,127.0.0.1'
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'

# If you already have this util, use it; else fallback to time.time()
try:
    import sys
    sys.path.append("/mnt/storage/RSystemsBenchmarking/gitProject")
    from Benchmark.code.evaluation.time_util import get_time
except Exception:
    def get_time():
        return time.time()

os.environ["HF_HUB_DISABLE_XET"] = "1"


class FlavaFaissSemanticSearcher:
    def __init__(
        self,
        index_path: str,
        names_path: str,
        model_path: str = "facebook/flava-full",
        device: Optional[str] = None,
        use_hnsw: bool = False,
        ef_search: int = 128,
    ):
        """
        Args:
            index_path: path to faiss index (.index)
            names_path: image_names.json mapping (faiss_id -> image_path)
            model_path: HuggingFace model or local model path
            device: "cuda" or "cpu"
            use_hnsw: if True, will try to set efSearch
            ef_search: HNSW runtime search quality/speed knob
        """
        self.index_path = index_path
        self.names_path = names_path
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = None
        self.model = None
        self.index = None
        self.image_names: List[str] = []
        self.use_hnsw = use_hnsw
        self.ef_search = ef_search

    def initialize(self) -> bool:
        """Load model + FAISS index + mapping."""
        try:
            print("[INFO] Loading FLAVA model...")
            self.processor = FlavaProcessor.from_pretrained(self.model_path)
            self.model = FlavaModel.from_pretrained(self.model_path).to(self.device).eval()

            print("[INFO] Loading FAISS index...")
            self.index = faiss.read_index(self.index_path)

            # If index is HNSW, we can optionally set efSearch
            # (safe even if not HNSW, inside try)
            if self.use_hnsw:
                try:
                    self.index.hnsw.efSearch = int(self.ef_search)
                    print(f"[INFO] HNSW efSearch set to {self.ef_search}")
                except Exception:
                    print("[WARN] This index does not support HNSW efSearch")

            print("[INFO] Loading image-name mapping...")
            with open(self.names_path, "r") as f:
                self.image_names = json.load(f)

            if self.index.ntotal != len(self.image_names):
                raise ValueError(
                    f"Index ntotal ({self.index.ntotal}) != image_names ({len(self.image_names)})"
                )

            print(f"[INFO] Ready: {self.index.ntotal:,} images indexed")
            return True

        except Exception as e:
            print(f"[ERROR] Initialization failed: {e}")
            return False

    @torch.no_grad()
    def encode_query(self, text: str) -> Optional[Dict]:
        """Encode one query text into FLAVA text embedding."""
        if text is None or not text.strip():
            return None

        try:
            start_time = get_time()

            inputs = self.processor(
                text=[text],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            out = self.model.get_text_features(**inputs)  # [1, seq, dim]
            emb = out[:, 0]                               # CLS pooling
            emb = emb / emb.norm(dim=-1, keepdim=True)    # L2 normalize

            encoding_time = get_time() - start_time

            return {
                "embedding": emb.detach().cpu().numpy().astype("float32").flatten(),
                "encoding_time": float(encoding_time),
            }

        except Exception as e:
            print(f"[ERROR] Encoding failed: {e}")
            return None

    def vector_search(self, query_embedding: np.ndarray, top_k: int = 10) -> Dict:
        """Run FAISS search on the index and return results."""
        if query_embedding is None:
            return {"results": [], "query_time": 0.0}

        try:
            start_time = get_time()

            # FAISS expects shape [nq, dim]
            q = query_embedding.reshape(1, -1).astype("float32")

            # scores: inner product similarity (cosine if normalized)
            scores, idxs = self.index.search(q, top_k)

            query_time = get_time() - start_time

            # format results like Solr output
            results = []
            for faiss_id, score in zip(idxs[0].tolist(), scores[0].tolist()):
                if faiss_id < 0:
                    continue
                results.append({
                    "image_path": self.image_names[faiss_id],
                    "score": float(score),
                })

            return {
                "results": results,
                "query_time": float(query_time),
            }

        except Exception as e:
            print(f"[ERROR] Vector search failed: {e}")
            return {"results": [], "query_time": 0.0}

    def search(self, query_text: str, top_k: int = 10) -> Dict:
        """
        Public API for FastAPI:

        Returns:
          {
            "results": [{"image_path": "...", "score": ...}, ...],
            "encoding_time": float,
            "query_time": float
          }
        """
        encoded = self.encode_query(query_text)
        if encoded is None:
            return {"results": [], "encoding_time": 0.0, "query_time": 0.0}

        embedding = encoded["embedding"]
        encoding_time = encoded["encoding_time"]

        search_out = self.vector_search(embedding, top_k)

        return {
            "results": search_out["results"],
            "encoding_time": encoding_time,
            "query_time": search_out["query_time"],
        }


# ----------------------------- quick test -----------------------------

def main():
    BASE_DIR = "/mnt/storage/RSystemsBenchmarking/gitProject/Benchmark/data/testing/testing1/flava"
    index_path = os.path.join(BASE_DIR, "index/flava_image.index")
    names_path = os.path.join(BASE_DIR, "index/image_names.json")

    searcher = FlavaFaissSemanticSearcher(
        index_path=index_path,
        names_path=names_path,
        model_path="facebook/flava-full",
        use_hnsw=False,      # set True if using HNSW index
        ef_search=128,
    )

    ok = searcher.initialize()
    if not ok:
        return

    while True:
        q = input("\nEnter query (or 'exit'): ").strip()
        if q.lower() in ("exit", "quit"):
            break

        out = searcher.search(q, top_k=10)
        print("\nEncoding time:", out["encoding_time"])
        print("Query time:", out["query_time"])
        print("Top results:")
        for i, r in enumerate(out["results"], 1):
            print(f"{i}. {r['image_path']}  | score={r['score']:.4f}")


if __name__ == "__main__":
    main()
