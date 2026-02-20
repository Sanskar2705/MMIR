"""
Standalone RRF fusion service for Solr retrievers.
"""

import sys
import os
from typing import List
from fastapi import FastAPI, Query

sys.path.append("/mnt/storage/RSystemsBenchmarking/gitProject")
from Benchmark.code.evaluation.time_util import get_time
from Benchmark.code.retrievalService.reranking.rrf import run_rrf_fusion

# Disable proxies
os.environ['no_proxy'] = 'localhost,127.0.0.1'
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'

SOLR_SERVICE_URL = "http://localhost:5055"

app = FastAPI(title="Solr RRF Service")

# Build endpoint map
endpoint_map = {}
models = ["clip", "flava", "minilm", "uniir", "bm25"]
datasets = ["coco", "flickr"]
modalities = {
    "clip": ["image", "text"],
    "flava": ["image", "text"],
    "minilm": ["text"],
    "uniir": ["image", "text", "joint-image-text"],
    "bm25": ["text"]
}

for model in models:
    for dataset in datasets:
        for modality in modalities[model]:
            key = f"{model}_{modality}_{dataset}_solr"
            endpoint_map[key] = f"{SOLR_SERVICE_URL}/{key}"


@app.get("/rrf2_fusion")
async def rrf_fusion_endpoint(q: str, methods: List[str] = Query(...)):
    """RRF fusion - matches your existing endpoint exactly"""
    urls = []
    start_time = get_time()
    
    for method in methods:
        if method not in endpoint_map:
            return {"error": f"Invalid method name: {method}"}
        urls.append(endpoint_map[method])
    
    try:
        fused_result, rrf_time = await run_rrf_fusion(q, urls)
        end_time = get_time()
        duration = end_time - start_time
        
        return {
            "list_of_top_k": fused_result,
            "query_time": duration,
            "rrf_time": rrf_time
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/health")
def health():
    return {"status": "ok", "methods": len(endpoint_map)}