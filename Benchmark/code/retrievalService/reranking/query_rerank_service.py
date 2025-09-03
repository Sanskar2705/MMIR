from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

from fastapi.responses import JSONResponse

from fastapi import FastAPI, Request
from fastapi import FastAPI, Query

import json
from typing import List, Dict, Any
from fastapi import FastAPI, Query
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi import Request


import torch
from PIL import Image

from lavis.models import load_model_and_preprocess
from lavis.processors import load_processor

import httpx
import requests

from fastapi.responses import JSONResponse

import inspect

import sys
sys.path.append("/mnt/storage/RSystemsBenchmarking/gitProject")
from Benchmark.config.config_utils import load_config
from Benchmark.code.evaluation.time_util import get_time

from Benchmark.code.retrievalService.reranking.rrf import run_rrf_fusion




app = FastAPI()

proxies = {
            'http': '',
            'https': ''
        }

# Load config
config = load_config()
endpoint_map = config["endpoints"]
inverted_endpoint_map = {v:k for k,v in endpoint_map.items()}
k = config["k"] 
debug = config["debug_logs"]
blip2 = config["models"]["blip2_itm"]


device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
model, vis_processors, text_processors = load_model_and_preprocess(blip2, "pretrain", device=device, is_eval=True)


@app.get("/rerank_with_blip2")
async def rerank_with_blip2(query: str , topk_list: str = Query(..., description="List of top k with item details"), K:int =10):

    try:
        # Parse topk_list from JSON string to Python list
        topk_list = json.loads(topk_list)
        if debug:
            print("Top-k list:", topk_list)

    except json.JSONDecodeError:
        return {"error": "Invalid JSON format in topk_list"}

    encode_time_list = []
    reranking_start_time = get_time()

    for result in topk_list:
       
        raw_image = Image.open(result["image_abs_path"]).convert("RGB")
        
        caption = query
        
        encode_start_time = get_time()

        img = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        txt = text_processors["eval"](caption)

        itm_output = model({"image": img, "text_input": txt}, match_head="itm")
        itm_scores = torch.nn.functional.softmax(itm_output, dim=1)
        neural_match_itm_prob = itm_scores[:, 1].item()

        encode_end_time = get_time()
        encode_time_list.append(encode_end_time-encode_start_time)

        result["reranking_score"] = neural_match_itm_prob

    reranked_items = sorted(topk_list, key=lambda x: x["reranking_score"], reverse=True)
    reranked_items = reranked_items[:K]

    reranking_end_time = get_time()
    reranking_duration = reranking_end_time - reranking_start_time
    results = {}
    results["list_of_top_k"] = reranked_items
    results["reranking_time"] = reranking_duration
    results["reranking_encoding_time_list"] = encode_time_list

    # return JSONResponse(content={"list_of_top_k":reranked_items}, status_code=200, headers={"Content-Type": "application/json"})
    # return {"list_of_top_k":reranked_items}
    return results

@app.get("/rerank_neural_blip2_endpoints")
async def rerank_mscoco_clip_only(q: str = Query(..., description="Search query"), methods:List[str] = Query(...), k:int = 10):
    query_endpoint_list = methods
    start_time = get_time()    

    results_by_model = []
    fetch_timing_by_model = []
    
    for endpoint in query_endpoint_list:
        if endpoint not in inverted_endpoint_map:
            return {"error": f"Invalid endpoint name: {endpoint}"}
    
        try:
            fetch_time_start = get_time()
            response = requests.get(endpoint, params={"q": q,"k":k}, timeout=100,proxies=proxies).json()
            fetch_time_duration =  get_time() - fetch_time_start
            
            fetch_timing_by_model.append(response["query_time"] or fetch_time_duration)
            results_by_model.extend(response.get("list_of_top_k", []))
        except Exception as e:
            return {"error": f"Failed to fetch from {endpoint}: {str(e)}"}    

    reranking_start_time = get_time()

    reranked_response = await rerank_with_blip2(query=q, topk_list=json.dumps(results_by_model),K=k)
    # reranked_response.raise_for_status()
    # reranked_results = reranked_response.json()
    reranked_results = reranked_response
    reranking_end_time = get_time()
    reranking_duration = reranking_end_time - reranking_start_time

    end_time = get_time()
    duration = end_time - start_time
    results = {}
    results["list_of_top_k"] = reranked_results["list_of_top_k"]
    results["query_time"] = duration
    results["reranking_time"] = reranking_duration
    results["endpoint_list"] = query_endpoint_list
    results["fetch_time_list"] = fetch_timing_by_model

    return results
    # return JSONResponse(content={"list_of_top_k":reranked_items}, status_code=200, headers={"Content-Type": "application/json"})


@app.get("/rrf2_fusion")
async def rrf_fusion_endpoint(q: str, methods: List[str] = Query(...), K:int =10):
    if debug:
        caller = inspect.currentframe().f_code.co_name
        print(caller," function args : ",locals())
    urls = []
    for method in methods:
        if method not in endpoint_map:
            return {"error": f"Invalid method name: {method}"}
        urls.append(endpoint_map[method])
    try:
        if debug:
            print("Before run_rrf_fusion : ")
            print(locals())
        fused = run_rrf_fusion(q, urls)
        return {"list_of_top_k": fused}
    except Exception as e:
        return {"error": str(e)}