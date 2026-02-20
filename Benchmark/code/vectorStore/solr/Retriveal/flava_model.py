"""
Interactive Semantic Search using CLIP (ViT-L/14@336px) Embeddings in Solr
"""

import os
import torch
import numpy as np
import sys
import pysolr
from typing import List, Dict, Optional
from transformers import FlavaProcessor, FlavaModel

sys.path.append("/mnt/storage/RSystemsBenchmarking/gitProject")

from Benchmark.code.evaluation.time_util import get_time


# Disable proxies for local Solr
os.environ['no_proxy'] = 'localhost,127.0.0.1'
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'

os.environ["HF_HUB_DISABLE_XET"] = "1"


class FlavaSemanticSearcher:
    def __init__(self, solr_url="http://localhost:8983/solr", core_name="flava_coco_text",modality="text"):
        self.solr_url = solr_url
        self.core_name = core_name
        self.core_url = f"{solr_url}/{core_name}"
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.solr_client = pysolr.Solr(self.core_url, timeout=10)
        self.modality = modality

    def initialize(self) -> bool:
        try:
            print("Loading FLAVA model...")
            self.processor = FlavaProcessor.from_pretrained("/mnt/storage/RSystemsBenchmarking/gitProject/Benchmark/Models/flava_full_model")
            self.model = FlavaModel.from_pretrained("/mnt/storage/RSystemsBenchmarking/gitProject/Benchmark/Models/flava_full_model").to(self.device).eval()

            self.model.eval()
            print("FLAVA model loaded successfully.")
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
            inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                features = self.model.get_text_features(**inputs)[:, 0]  # 768-D [CLS] token
                features = features / features.norm(dim=-1, keepdim=True)  # L2 normalize
            end_time = get_time()
            encoding_time = end_time - start_time
            return {
                "embedding": features.cpu().numpy().flatten(),
                "encoding_time": encoding_time
            }
        except Exception as e:
            print(f"Encoding error: {e}")
            return None

    def vector_search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Dict]:
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
            # elif self.modality=="image"
            else:
                results = self.solr_client.search(knn_query, **{
                    "rows": top_k,
                    "fl": "image_path, score",
                    "wt": "json"
                })
            end_time = get_time()
            query_time = end_time - start_time

            # return list(results)
            return {
                "results": list(results),
                "query_time": query_time
            }
        except Exception as e:
            print(f"Vector search error: {e}")
            # return []
            return {
                "results": [],
                "query_time": 0.0
            }

    # def search(self, query_text: str, top_k: int = 10) -> List[Dict]:
    #     embedding = self.encode_query(query_text)
    #     if embedding is None:
    #         return []
    #     return self.vector_search(embedding, top_k)

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
        else:
            for i, doc in enumerate(results, 1):
                print(f"{i}. Image Name: {doc.get('image_path', 'N/A')} | Score: {doc.get('score', 'N/A')}")

    def interactive_search(self):
        print("\nInteractive CLIP Semantic Search (ViT-L/14@336px)")
        print("Type 'exit' to quit.")
        while True:
            try:
                query = input("\nEnter your search query: ").strip()
                if query.lower() in ['exit', 'quit']:
                    break
                if not query:
                    continue

                # results = self.search(query, top_k=10)
                # self.display_results(results, query)
                result_obj = self.search(query, top_k=10)
                results = result_obj["results"]
                encoding_time = result_obj["encoding_time"]
                self.display_results(results, query)

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    searcher = FlavaSemanticSearcher()
    if not searcher.initialize():
        print("Failed to initialize CLIP search system.")
        sys.exit(1)

    searcher.interactive_search()


if __name__ == "__main__":
    main()
