#!/usr/bin/env python3

"""
Interactive Semantic Search using MiniLM (384-dim) Embeddings in Solr
"""

import os
import sys
import numpy as np
import pysolr
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer

# Disable proxies for local Solr
os.environ['no_proxy'] = 'localhost,127.0.0.1'
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'


class MiniLMSemanticSearcher:
    def __init__(self, solr_url="http://localhost:8983/solr", core_name="Coco_minilm_caption_test"):
        self.solr_url = solr_url
        self.core_name = core_name
        self.core_url = f"{solr_url}/{core_name}"
        self.model = None
        self.solr_client = pysolr.Solr(self.core_url, timeout=10)

    def initialize(self) -> bool:
        try:
            print("Loading MiniLM model (sentence-transformers)...")
            self.model = SentenceTransformer("/mnt/storage/sanskar/samy/Benchmark/Models/MiniLm")  # 384-dim
            print("MiniLM model loaded successfully.")
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

    def encode_query(self, text: str) -> Optional[np.ndarray]:
        if not text.strip():
            print("Query must be a non-empty string.")
            return None
        try:
            embedding = self.model.encode(text, normalize_embeddings=True)
            return embedding.astype(np.float32)
        except Exception as e:
            print(f"Encoding error: {e}")
            return None

    def vector_search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Dict]:
        try:
            embedding_str = "[" + ",".join(map(str, query_embedding.tolist())) + "]"
            knn_query = f"{{!knn f=embedding_vector topK={top_k} bruteForce=true}}{embedding_str}"
            results = self.solr_client.search(knn_query, **{
                "rows": top_k,
                "fl": "image_path, caption, score",
                "wt": "json"
            })
            return list(results)
        except Exception as e:
            print(f"Vector search error: {e}")
            return []

    def search(self, query_text: str, top_k: int = 10) -> List[Dict]:
        embedding = self.encode_query(query_text)
        if embedding is None:
            return []
        return self.vector_search(embedding, top_k)

    def display_results(self, results: List[Dict], query: str):
        if not results:
            print("No results found.")
            return
        print(f"\nSearch results for: '{query}'")
        for i, doc in enumerate(results, 1):
            print(f"{i}. Image Name: {doc.get('image_path', 'N/A')} | Caption: {doc.get('caption', 'N/A')} | Score: {doc.get('score', 'N/A')}")

    def interactive_search(self):
        print("\nInteractive Semantic Search using MiniLM (384-dim)")
        print("Type 'exit' to quit.")
        while True:
            try:
                query = input("\nEnter your search query: ").strip()
                if query.lower() in ['exit', 'quit']:
                    break
                if not query:
                    continue
                results = self.search(query, top_k=10)
                self.display_results(results, query)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    searcher = MiniLMSemanticSearcher()
    if not searcher.initialize():
        print("Failed to initialize MiniLM search system.")
        sys.exit(1)

    searcher.interactive_search()


if __name__ == "__main__":
    main()
