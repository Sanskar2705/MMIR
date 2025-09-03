# import torch,time


from contextlib import asynccontextmanager

import os
import sys
sys.path.append("/mnt/storage/RSystemsBenchmarking/gitProject")


from typing import Union,List
from fastapi import FastAPI ,Query
from Benchmark.code.vectorStore.solr.Retriveal.clip_model import CLIPSemanticSearcher
from Benchmark.code.vectorStore.solr.Retriveal.bm25_model import BM25CaptionSearcher
from Benchmark.code.vectorStore.solr.Retriveal.minilm_model import MiniLMSemanticSearcher 
from Benchmark.code.vectorStore.solr.Retriveal.flava_model import FlavaSemanticSearcher

from Benchmark.code.retrievalService.reranking.rrf import run_rrf_fusion

from Benchmark.config.config_utils import load_config
from Benchmark.code.evaluation.time_util import get_time

BASE_IMAGE_PATH_COCO = None
BASE_IMAGE_PATH_FLICKR = None
endpoint_map = None



# @asynccontextmanager
# async def lifespan(app: FastAPI):
    # Load config
config = load_config()
debug = config["debug_logs"]
BASE_IMAGE_PATH_COCO = config["paths"]["dataset"]["coco"]["base_image_path"]
BASE_IMAGE_PATH_FLICKR = config["paths"]["dataset"]["flickr"]["base_image_path"]
endpoint_map = config["endpoints"]

# Initialize CLIP model (mscoco index)
core_name="clip_coco_image"
Clipsearcher_mscoco = CLIPSemanticSearcher(core_name = core_name,modality="image")
Clipsearcher_mscoco.initialize()

# Initialize CLIP model for captions (mscoco index)
core_name="clip_coco_text"
Clipsearcher_caption_mscoco = CLIPSemanticSearcher(core_name = core_name,modality="text")
Clipsearcher_caption_mscoco.initialize()

# Initialize CLIP model for Flickr images
core_name="clip_flickr_image"
Clipsearcher_flickr = CLIPSemanticSearcher(core_name = core_name,modality="image")
Clipsearcher_flickr.initialize()        

# Initialize CLIP model for Flickr captions 
core_name="clip_flickr_text"
Clipsearcher_caption_flickr = CLIPSemanticSearcher( core_name = core_name,modality="text")
Clipsearcher_caption_flickr.initialize()

# Initialize BM25 model for captions (mscoco index)
core_name="bm25_coco_text"
BM25CaptionSearcher_mscoco = BM25CaptionSearcher(core_name = core_name)


# Initialize BM25 model for Flickr captions
core_name="bm25_flickr_text"
BM25CaptionSearcher_flickr = BM25CaptionSearcher(core_name = core_name)


# Initialize MiniLM model for semantic search (mscoco index)
core_name="minilm_coco_text"
MiniLMSemanticSearcher_mscoco = MiniLMSemanticSearcher(core_name = core_name)
MiniLMSemanticSearcher_mscoco.initialize()

# Initialize MiniLM model for semantic search (flickr index)    
core_name="minilm_flickr_text"
MiniLMSemanticSearcher_flickr = MiniLMSemanticSearcher(core_name = core_name)
MiniLMSemanticSearcher_flickr.initialize()

# Initialize FLAVA model for semantic search (mscoco index)
core_name="flava_coco_text"
FlavaSearcher_Caption_mscoco = FlavaSemanticSearcher(core_name = core_name, modality="text")
FlavaSearcher_Caption_mscoco.initialize()

# Initialize FLAVA model for semantic search (flickr index)
core_name="flava_flickr_text"
FlavaSearcher_Caption_flickr = FlavaSemanticSearcher(core_name = core_name, modality="text")
FlavaSearcher_Caption_flickr.initialize()

# Initialize FLAVA model for image search (mscoco index)
core_name="flava_coco_image"
FlavaSearcher_mscoco = FlavaSemanticSearcher(core_name = core_name, modality="image")
FlavaSearcher_mscoco.initialize()

# Initialize FLAVA model for image search (flickr index)
core_name="flava_flickr_image"
FlavaSearcher_flickr = FlavaSemanticSearcher(core_name = core_name, modality="image")
FlavaSearcher_flickr.initialize()
    # yield


    # Clean up the ML models and release the resources
    # Add potential cleanup code here

# app = FastAPI(lifespan=lifespan)
app = FastAPI()


#Image_abs_path  
# @app.get("/ClipImage_mscoco")
@app.get("/clip_image_coco_solr")
async def search_clip_mscoco(q: Union[str, None] = None):
    if(debug): print(f"Received CLIP search query: {q}")
    start_time = get_time()

    # result = Clipsearcher_mscoco.search(q, top_k=10)
    result_obj = Clipsearcher_mscoco.search(q, top_k=10)
    result = result_obj["results"]
    encoding_time = result_obj["encoding_time"]
    retrieval_time = result_obj["query_time"]

    for item in result:
        item["image_abs_path"] = f"{BASE_IMAGE_PATH_COCO}/{os.path.basename(item['image_path'])}"
    
    end_time = get_time()
    duration = end_time - start_time

    if(debug): print(f"CLIP results: {result}")
    return {"list_of_top_k": result, "query_time":duration, "encoding_time": encoding_time, "retrieval_time": retrieval_time}

# @app.get("/ClipCaption_mscoco")
@app.get("/clip_text_coco_solr")
async def search_clip_caption_mscoco(q: Union[str, None] = None):
    if(debug): print(f"Received CLIP caption search query: {q}")
 
    start_time = get_time()
    result_obj = Clipsearcher_caption_mscoco.search(q, top_k=10)
    result = result_obj["results"]
    encoding_time = result_obj["encoding_time"]
    retrieval_time = result_obj["query_time"]

    for item in result:
        item["image_abs_path"] = f"{BASE_IMAGE_PATH_COCO}/{os.path.basename(item['image_path'])}"
    
    end_time = get_time()
    duration = end_time - start_time

    if(debug): print(f"CLIP caption results: {result }")
    return {"list_of_top_k": result, "query_time":duration, "encoding_time": encoding_time, "retrieval_time": retrieval_time}


# @app.get("/ClipImage_flickr")
@app.get("/clip_image_flickr_solr")
async def search_clip_flickr(q: Union[str, None] = None):
    if(debug): print(f"Received CLIP search query for Flickr: {q}")

    start_time = get_time()
    result_obj = Clipsearcher_flickr.search(q, top_k=10)
    result = result_obj["results"]
    encoding_time = result_obj["encoding_time"]
    retrieval_time = result_obj["query_time"]


    for item in result:
        item["image_abs_path"] = f"{BASE_IMAGE_PATH_FLICKR}/{os.path.basename(item['image_path'])}"
    end_time = get_time()
    duration = end_time - start_time

    if(debug): print(f"CLIP results for Flickr: {result}")
    return {"list_of_top_k": result, "query_time":duration, "encoding_time": encoding_time, "retrieval_time": retrieval_time}

# @app.get("/ClipCaption_flickr") 
@app.get("/clip_text_flickr_solr")
async def search_clip_caption_flickr(q: Union[str, None] = None):
    if(debug): print(f"Received CLIP caption search query for Flickr: {q}")

    start_time = get_time()
    result_obj = Clipsearcher_caption_flickr.search(q, top_k=10)
    result = result_obj["results"]
    encoding_time = result_obj["encoding_time"]
    retrieval_time = result_obj["query_time"]

    for item in result:
        item["image_abs_path"] = f"{BASE_IMAGE_PATH_FLICKR}/{os.path.basename(item['image_path'])}"
    end_time = get_time()
    duration = end_time - start_time

    if(debug): print(f"CLIP caption results for Flickr: {result}")
    return {"list_of_top_k": result, "query_time":duration, "encoding_time": encoding_time, "retrieval_time": retrieval_time}


# @app.get("/BM25Caption_mscoco")
@app.get("/bm25_text_coco_solr")
async def search_bm25_caption_mscoco(q: Union[str, None] = None): 
    if(debug): print(f"Received BM25 caption search query: {q}")

    start_time = get_time()
    result_obj = BM25CaptionSearcher_mscoco.search(q, top_k=10)
    result = result_obj["results"]
    retrieval_time = result_obj["query_time"]

    for item in result:
        item["image_abs_path"] = f"{BASE_IMAGE_PATH_COCO}/{os.path.basename(item['image_path'])}"
    end_time = get_time()
    duration = end_time - start_time

    if(debug): print(f"BM25 caption results: {result}")
    return {"list_of_top_k": result, "query_time":duration, "retrieval_time": retrieval_time}

# @app.get("/BM25Caption_flickr")
@app.get("/bm25_text_flickr_solr")
async def search_bm25_caption_flickr(q: Union[str, None] = None):     
    if(debug): print(f"Received BM25 caption search query for Flickr: {q}")
    
    start_time = get_time()
    result_obj = BM25CaptionSearcher_flickr.search(q, top_k=10)
    result = result_obj["results"]
    retrieval_time = result_obj["query_time"]

    for item in result:
        item["image_abs_path"] = f"{BASE_IMAGE_PATH_FLICKR}/{os.path.basename(item['image_path'])}"
    end_time = get_time()
    duration = end_time - start_time

    if(debug): print(f"BM25 caption results for Flickr: {result}")
    return {"list_of_top_k": result, "query_time":duration, "retrieval_time": retrieval_time}

# @app.get("/MiniLmCaption_mscoco")
@app.get("/minilm_text_coco_solr")
async def search_minilm_caption_mscoco(q: Union[str, None] = None):
    if(debug): print(f"Received MiniLM caption search query: {q}")

    start_time = get_time()
    result_obj = MiniLMSemanticSearcher_mscoco.search(q, top_k=10)
    result = result_obj["results"]
    encoding_time = result_obj["encoding_time"]
    retrieval_time = result_obj["query_time"]

    for item in result:
        item["image_abs_path"] = f"{BASE_IMAGE_PATH_COCO}/{os.path.basename(item['image_path'])}"
    end_time = get_time()
    duration = end_time - start_time

    if(debug): print(f"MiniLM caption results: {result}")
    return {"list_of_top_k": result, "query_time":duration, "encoding_time": encoding_time, "retrieval_time": retrieval_time}


# @app.get("/MiniLmCaption_flickr")
@app.get("/minilm_text_flickr_solr")
async def search_minilm_caption_flickr(q: Union[str, None] = None):
    if(debug): print(f"Received MiniLM caption search query for Flickr: {q}")
    
    start_time = get_time()
    result_obj = MiniLMSemanticSearcher_flickr.search(q, top_k=10)
    result = result_obj["results"]
    encoding_time = result_obj["encoding_time"]
    retrieval_time = result_obj["query_time"]

    for item in result:
        item["image_abs_path"] = f"{BASE_IMAGE_PATH_FLICKR}/{os.path.basename(item['image_path'])}"
    end_time = get_time()
    duration = end_time - start_time

    if(debug): print(f"MiniLM caption results for Flickr: {result}")
    return {"list_of_top_k": result, "query_time":duration, "encoding_time": encoding_time, "retrieval_time": retrieval_time}

@app.get("/flava_text_coco_solr")
async def search_flava_caption_mscoco(q: Union[str, None] = None):
    if(debug): print(f"Received FLAVA caption search query: {q}")
  
    start_time = get_time()
    result_obj = FlavaSearcher_Caption_mscoco.search(q, top_k=10)
    result = result_obj["results"]
    encoding_time = result_obj["encoding_time"]
    retrieval_time = result_obj["query_time"]

    for item in result:
        item["image_abs_path"] = f"{BASE_IMAGE_PATH_COCO}/{os.path.basename(item['image_path'])}"
    end_time = get_time()
    duration = end_time - start_time

    if(debug): print(f"FLAVA caption results: {result}")
    return {"list_of_top_k": result, "query_time":duration, "encoding_time": encoding_time, "retrieval_time": retrieval_time}

@app.get("/flava_text_flickr_solr")
async def search_flava_caption_flickr(q: Union[str, None] = None):  
    if(debug): print(f"Received FLAVA caption search query for Flickr: {q}")
    
    start_time = get_time()
    result_obj = FlavaSearcher_Caption_flickr.search(q, top_k=10)
    result = result_obj["results"]
    encoding_time = result_obj["encoding_time"]
    retrieval_time = result_obj["query_time"]

    for item in result:
        item["image_abs_path"] = f"{BASE_IMAGE_PATH_FLICKR}/{os.path.basename(item['image_path'])}"
    end_time = get_time()
    duration = end_time - start_time

    if(debug): print(f"FLAVA caption results for Flickr: {result}")
    return {"list_of_top_k": result, "query_time":duration, "encoding_time": encoding_time, "retrieval_time": retrieval_time}

@app.get("/flava_image_coco_solr")
async def search_flava_image_mscoco(q: Union[str, None] = None):    
    if(debug): print(f"Received FLAVA image search query: {q}")

    start_time = get_time()
    result_obj = FlavaSearcher_mscoco.search(q, top_k=10)
    result = result_obj["results"]
    encoding_time = result_obj["encoding_time"]
    retrieval_time = result_obj["query_time"]

    for item in result:
        item["image_abs_path"] = f"{BASE_IMAGE_PATH_COCO}/{os.path.basename(item['image_path'])}"
    end_time = get_time()
    duration = end_time - start_time

    if(debug): print(f"FLAVA image results: {result}")
    return {"list_of_top_k": result, "query_time":duration, "encoding_time": encoding_time, "retrieval_time": retrieval_time}

@app.get("/flava_image_flickr_solr")
async def search_flava_image_flickr(q: Union[str, None] = None):    
    if(debug): print(f"Received FLAVA image search query for Flickr: {q}")

    start_time = get_time()
    result_obj = FlavaSearcher_flickr.search(q, top_k=10)
    result = result_obj["results"]
    encoding_time = result_obj["encoding_time"]
    retrieval_time = result_obj["query_time"]

    for item in result:
        item["image_abs_path"] = f"{BASE_IMAGE_PATH_FLICKR}/{os.path.basename(item['image_path'])}"
    end_time = get_time()
    duration = end_time - start_time

    if(debug): print(f"FLAVA image results for Flickr: {result}")
    return {"list_of_top_k": result, "query_time":duration, "encoding_time": encoding_time, "retrieval_time": retrieval_time}

# RRF Fusion Endpoints 
# @app.get("/rrf_fusion")
# def rrf_fusion_endpoint(q: str, method1: str, method2: str):
#     if method1 not in endpoint_map or method2 not in endpoint_map:
#         return {"error": f"Invalid method names: {method1}, {method2}"}

#     url1 = endpoint_map[method1]
#     url2 = endpoint_map[method2]

#     try:
#         fused = run_rrf_fusion(q, url1, url2)
#         return {"list_of_top_k": fused}
#     except Exception as e:
#         return {"error": str(e)}

@app.get("/rrf2_fusion")
def rrf_fusion_endpoint(q: str, methods: List[str] = Query(...)):
    urls = []
    for method in methods:
        if method not in endpoint_map:
            return {"error": f"Invalid method name: {method}"}
        urls.append(endpoint_map[method])
    try:
        fused = run_rrf_fusion(q, urls)
        return {"list_of_top_k": fused}
    except Exception as e:
        return {"error": str(e)}