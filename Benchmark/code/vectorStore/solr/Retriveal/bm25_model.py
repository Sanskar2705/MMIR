#!/usr/bin/env python3

"""
Interactive BM25 Search on COCO Captions in Solr
"""

import os
import sys
import pysolr
from typing import List, Dict
sys.path.append("/mnt/storage/RSystemsBenchmarking/gitProject")

from Benchmark.code.evaluation.time_util import get_time

# Disable proxies for local Solr communication
os.environ['no_proxy'] = 'localhost,127.0.0.1'
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'

class BM25CaptionSearcher:
    def __init__(self, solr_url="http://localhost:8983/solr", core_name="Coco_bm25"):
        self.solr_url = solr_url
        self.core_name = core_name
        self.core_url = f"{solr_url}/{core_name}"
        self.solr_client = pysolr.Solr(self.core_url, timeout=10)

    def _check_solr_connection(self) -> bool:
        try:
            ping_response = self.solr_client.ping()
            return "OK" in ping_response
        except Exception:
            return False

    def search(self, query_text: str, top_k: int = 10) -> List[Dict]:
        try:
            start_time = get_time()
            results = self.solr_client.search(query_text, **{
                "qf": "caption",
                "rows": top_k,
                "fl": "image_path,caption,score",
                "defType": "edismax",
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
            print(f"Search error: {e}")
            # return []
            return {
                "results": [],
                "query_time": 0.0
            }


    def display_results(self, results: List[Dict], query: str):
        if not results:
            print("No results found.")
            return
        print(f"\nTop results for: '{query}'\n")
        for i, doc in enumerate(results, 1):
            caption = doc.get("caption", "N/A")
            # Captions may be returned as list if stored that way
            if isinstance(caption, list):
                caption = caption[0]
            print(f"{i}. Image: {doc.get('image_path', 'N/A')}")
            print(f"   Caption: {caption}")
            print(f"   Score: {doc.get('score', 'N/A')}\n")

    def interactive_search(self):
        if not self._check_solr_connection():
            print("Failed to connect to Solr.")
            sys.exit(1)

        print("\nInteractive BM25 Search on COCO Captions (via Solr)")
        print("Type 'exit' to quit.\n")

        while True:
            try:
                query = input("Enter your search query: ").strip()
                if query.lower() in ["exit", "quit"]:
                    break
                if not query:
                    continue
                results = self.search(query, top_k=10)
                self.display_results(results, query)
            except KeyboardInterrupt:
                print("\nExiting.")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    searcher = BM25CaptionSearcher()
    searcher.interactive_search()


if __name__ == "__main__":
    main()
