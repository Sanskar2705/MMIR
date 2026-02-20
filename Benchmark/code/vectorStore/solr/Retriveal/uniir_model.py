import os
import torch
import numpy as np
import sys
import pysolr
import warnings
from pathlib import Path
from typing import List, Dict, Optional, Callable
import torch.serialization
from omegaconf.dictconfig import DictConfig
torch.serialization.add_safe_globals([DictConfig])


sys.path.append("/mnt/storage/RSystemsBenchmarking/gitProject")
from Benchmark.code.evaluation.time_util import get_time
from Benchmark.config.config_utils import load_config

# Disable proxies for local Solr
os.environ['no_proxy'] = 'localhost,127.0.0.1'
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'
os.environ["HF_HUB_DISABLE_XET"] = "1"

def l2norm(t):
    return t / t.norm(dim=-1, keepdim=True)

def _load_uniir(model_cfg: Dict, dev: str) -> Callable[[str], np.ndarray]:
    import open_clip
    arch = model_cfg["arch"]
    ckpt_path = Path(model_cfg["checkpoint"])
    prompt = model_cfg.get("prompt", "").strip()

    # model, _, _ = open_clip.create_model_and_transforms(arch, pretrained="openai", device=dev)
    model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=None, device=dev)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        # explicitly allow loading full pickle objects
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = state.get("model") or state.get("state_dict") or state
    state = {k.replace("clip_model.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    tok = open_clip.get_tokenizer(arch)
    model.to(dev).eval()

    def _embed(text: str) -> np.ndarray:
        full = f"{prompt} {text}".strip() if prompt else text
        t = tok([full]).to(dev)
        with torch.no_grad():
            vec = l2norm(model.encode_text(t)).cpu().numpy()
        return vec.astype("float32")[0]
    return _embed



class UniIRSemanticSearcher:
    def __init__(self, model_cfg: Dict, solr_url="http://localhost:8983/solr", core_name="uniir_flickr_text", modality="text"):
        self.solr_url = solr_url
        self.core_name = core_name
        self.core_url = f"{solr_url}/{core_name}"
        self.model_cfg = model_cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.solr_client = pysolr.Solr(self.core_url, timeout=10)
        self.modality = modality
        self.embed_fn = None

    def initialize(self) -> bool:
        try:
            print("Loading UniIR model...")
            self.embed_fn = _load_uniir(self.model_cfg, str(self.device))
            print("UniIR model loaded successfully.")
            return self._check_solr_connection()
        except Exception as e:
            print(f"Initialization error: {e}")
            return False

    def _check_solr_connection(self) -> bool:
        try:
            ping_response = self.solr_client.ping()
            return "OK" in ping_response
        except Exception:
            return False

    def encode_query(self, text: str) -> Optional[Dict]:
        if not text.strip():
            print("Query must be a non-empty string.")
            return None
        try:
            start_time = get_time()
            features = self.embed_fn(text)
            end_time = get_time()
            encoding_time = end_time - start_time
            return {
                "embedding": features,
                "encoding_time": encoding_time
            }
        except Exception as e:
            print(f"Encoding error: {e}")
            return None

    def vector_search(self, query_embedding: np.ndarray, top_k: int = 10) -> Dict:
        try:
            embedding_str = "[" + ",".join(map(str, query_embedding.tolist())) + "]"
            knn_query = f"{{!knn f=embedding_vector topK={top_k} bruteForce=true}}{embedding_str}"
            start_time = get_time()
            if self.modality == "text":
                results = self.solr_client.search(knn_query, **{
                    "rows": top_k,
                    "fl": "image_path, caption ,score",
                    "wt": "json"
                })
            elif self.modality == "image":
                results = self.solr_client.search(knn_query, **{
                    "rows": top_k,
                    "fl": "image_path, score",
                    "wt": "json"
                })
            elif self.modality == "joint-image-text":
                results = self.solr_client.search(knn_query, **{
                    "rows": top_k,
                    "fl": "image_path, caption, score",
                    "wt": "json"
                }) 
            end_time = get_time()
            query_time = end_time - start_time

            return {
                "results": list(results),
                "query_time": query_time
            }
        except Exception as e:
            print(f"Vector search error: {e}")
            return {
                "results": [],
                "query_time": 0.0
            }

    def search(self, query_text: str, top_k: int = 10) -> Dict:
        result = self.encode_query(query_text)
        if result is None:
            return {"results": [], "encoding_time": 0.0, "query_time": 0.0}
        embedding = result["embedding"]
        encoding_time = result["encoding_time"]
        search_output = self.vector_search(embedding, top_k)

        return {
            "results": search_output["results"],
            "encoding_time": encoding_time,
            "query_time": search_output["query_time"]
        }

    def display_results(self, results: List[Dict], query: str):
        if not results:
            print("No results found.")
            return
        print(f"\nSearch results for: '{query}'")
        if self.modality == "text":
            for i, doc in enumerate(results, 1):
                print(f"{i}. Image Name: {doc.get('image_path', 'N/A')} | Caption: {doc.get('caption', 'N/A')} | Score: {doc.get('score', 'N/A')}")
        elif self.modality == "image":
            for i, doc in enumerate(results, 1):
                print(f"{i}. Image Name: {doc.get('image_path', 'N/A')} | Score: {doc.get('score', 'N/A')}")
        elif self.modality == "joint-image-text":
            for i, doc in enumerate(results, 1):
                print(f"{i}. Image Name: {doc.get('image_path', 'N/A')} | Caption: {doc.get('caption', 'N/A')} | Score: {doc.get('score', 'N/A')}")

    def interactive_search(self):
        print("\nInteractive UniIR Semantic Search")
        print("Type 'exit' to quit.")
        while True:
            try:
                query = input("\nEnter your search query: ").strip()
                if query.lower() in ['exit', 'quit']:
                    break
                if not query:
                    continue

                result_obj = self.search(query, top_k=10)
                results = result_obj["results"]
                self.display_results(results, query)

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    config = load_config()
    model_cfg = config["models"]["uniir"]  # adjust key if different
    searcher = UniIRSemanticSearcher(model_cfg)
    if not searcher.initialize():
        print("Failed to initialize UniIR search system.")
        sys.exit(1)

    searcher.interactive_search()


if __name__ == "__main__":
    main()
