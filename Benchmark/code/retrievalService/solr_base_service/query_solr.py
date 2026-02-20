# import torch,time


from contextlib import asynccontextmanager
from functools import lru_cache

import os
import sys 
import requests
sys.path.append("/mnt/storage/RSystemsBenchmarking/gitProject")

from fastapi import FastAPI, Query, HTTPException
from typing import Union,List , Dict
from fastapi import FastAPI ,Query
from Benchmark.code.vectorStore.solr.Retriveal.clip_model import CLIPSemanticSearcher
from Benchmark.code.vectorStore.solr.Retriveal.bm25_model import BM25CaptionSearcher
from Benchmark.code.vectorStore.solr.Retriveal.minilm_model import MiniLMSemanticSearcher 
from Benchmark.code.vectorStore.solr.Retriveal.flava_model import FlavaSemanticSearcher
from Benchmark.code.vectorStore.solr.Retriveal.uniir_model import UniIRSemanticSearcher

from Benchmark.code.vectorStore.solr.Retriveal.retrieval_orchestrator import TwoStageRetrievalOrchestrator

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
model_cfg = config["models"]["uniir"]


# # Initialize CLIP model (mscoco index)
# core_name="clip_coco_image"
# Clipsearcher_mscoco = CLIPSemanticSearcher(core_name = core_name,modality="image")
# Clipsearcher_mscoco.initialize()

# # Initialize CLIP model for captions (mscoco index)
# core_name="clip_coco_text"
# Clipsearcher_caption_mscoco = CLIPSemanticSearcher(core_name = core_name,modality="text")
# Clipsearcher_caption_mscoco.initialize()
 
# # Initialize CLIP model for Flickr images
# core_name="clip_flickr_image"
# Clipsearcher_flickr = CLIPSemanticSearcher(core_name = core_name,modality="image")
# Clipsearcher_flickr.initialize()        
 
# # Initialize CLIP model for Flickr captions 
# core_name="clip_flickr_text"
# Clipsearcher_caption_flickr = CLIPSemanticSearcher( core_name = core_name,modality="text")
# Clipsearcher_caption_flickr.initialize()

# # Initialize BM25 model for captions (mscoco index)
# core_name="bm25_coco_text"
# BM25CaptionSearcher_mscoco = BM25CaptionSearcher(core_name = core_name)


# # Initialize BM25 model for Flickr captions
# core_name="bm25_flickr_text"
# BM25CaptionSearcher_flickr = BM25CaptionSearcher(core_name = core_name)


# # Initialize MiniLM model for semantic search (mscoco index)
# core_name="minilm_coco_text"
# MiniLMSemanticSearcher_mscoco = MiniLMSemanticSearcher(core_name = core_name)
# MiniLMSemanticSearcher_mscoco.initialize()

# # Initialize MiniLM model for semantic search (flickr index)    
# core_name="minilm_flickr_text"
# MiniLMSemanticSearcher_flickr = MiniLMSemanticSearcher(core_name = core_name)
# MiniLMSemanticSearcher_flickr.initialize()  

# # Initialize FLAVA model for semantic search (mscoco index)
# core_name="flava_coco_text"
# FlavaSearcher_Caption_mscoco = FlavaSemanticSearcher(core_name = core_name, modality="text")
# FlavaSearcher_Caption_mscoco.initialize()

# # Initialize FLAVA model for semantic search (flickr index)
# core_name="flava_flickr_text"
# FlavaSearcher_Caption_flickr = FlavaSemanticSearcher(core_name = core_name, modality="text")
# FlavaSearcher_Caption_flickr.initialize()

# # Initialize FLAVA model for image search (mscoco index)
# core_name="flava_coco_image"
# FlavaSearcher_mscoco = FlavaSemanticSearcher(core_name = core_name, modality="image")
# FlavaSearcher_mscoco.initialize()

# # Initialize FLAVA model for image search (flickr index)
# core_name="flava_flickr_image"
# FlavaSearcher_flickr = FlavaSemanticSearcher(core_name = core_name, modality="image")
# FlavaSearcher_flickr.initialize()


# core_name="uniir_coco_text"
# UniIRSemanticSearcher_Caption_mscoco = UniIRSemanticSearcher( model_cfg=model_cfg, core_name=core_name, modality="text")
# UniIRSemanticSearcher_Caption_mscoco.initialize()

 
# core_name="uniir_flickr_text"
# UniIRSemanticSearcher_Caption_flickr = UniIRSemanticSearcher(model_cfg=model_cfg, core_name=core_name, modality="text")
# UniIRSemanticSearcher_Caption_flickr.initialize()   
   

# core_name="uniir_coco_image"
# UniIRSemanticSearcher_mscoco_image = UniIRSemanticSearcher(model_cfg=model_cfg, core_name=core_name, modality="image")
# UniIRSemanticSearcher_mscoco_image.initialize()

# core_name="uniir_flickr_image"
# UniIRSemanticSearcher_flickr_image = UniIRSemanticSearcher(model_cfg=model_cfg, core_name=core_name, modality="image")
# UniIRSemanticSearcher_flickr_image.initialize()

# core_name="uniir_coco_joint-image-text"
# UniIRSemanticSearcher_mscoco_joint = UniIRSemanticSearcher(model_cfg=model_cfg, core_name=core_name, modality="joint-image-text")
# UniIRSemanticSearcher_mscoco_joint.initialize()

# core_name="uniir_flickr_joint-image-text"
# UniIRSemanticSearcher_flickr_joint = UniIRSemanticSearcher(model_cfg = model_cfg, core_name=core_name, modality="joint-image-text")
# UniIRSemanticSearcher_flickr_joint.initialize()


 
two_stage_orchestrator = TwoStageRetrievalOrchestrator("http://localhost:8983/solr")

    # yield


    # Clean up the ML models and release the resources
    # Add potential cleanup code here

# app = FastAPI(lifespan=lifespan)
app = FastAPI()


#Image_abs_path  

# @app.get("/clip_image_coco_solr")
# async def search_clip_mscoco(q: Union[str, None] = None,k:int = 10):
#     if(debug): print(f"Received CLIP search query: {q}")
#     start_time = get_time()

#     # result = Clipsearcher_mscoco.search(q, top_k=10)
#     result_obj = Clipsearcher_mscoco.search(q, top_k=k)
#     # searcher = get_searcher("clip", "coco", "image")
#     # result_obj = searcher.search(q, top_k=k)
    
#     result = result_obj["results"]
#     encoding_time = result_obj["encoding_time"]
#     retrieval_time = result_obj["query_time"]

#     for item in result:
#         item["image_abs_path"] = f"{BASE_IMAGE_PATH_COCO}/{os.path.basename(item['image_path'])}"
    
#     end_time = get_time()
#     duration = end_time - start_time

#     if(debug): print(f"CLIP results: {result}")
#     return {"list_of_top_k": result, "query_time":duration, "encoding_time": encoding_time, "retrieval_time": retrieval_time}


# @app.get("/clip_text_coco_solr")
# async def search_clip_caption_mscoco(q: Union[str, None] = None,k:int = 10):
#     if(debug): print(f"Received CLIP caption search query: {q}")
 
#     start_time = get_time()
#     result_obj = Clipsearcher_caption_mscoco.search(q, top_k=k)
#     # searcher = get_searcher("clip", "coco", "text")
#     # result_obj = searcher.search(q, top_k=k)

#     result = result_obj["results"]
#     encoding_time = result_obj["encoding_time"]
#     retrieval_time = result_obj["query_time"]

#     for item in result:
#         item["image_abs_path"] = f"{BASE_IMAGE_PATH_COCO}/{os.path.basename(item['image_path'])}"
    
#     end_time = get_time()
#     duration = end_time - start_time

#     if(debug): print(f"CLIP caption results: {result }")
#     return {"list_of_top_k": result, "query_time":duration, "encoding_time": encoding_time, "retrieval_time": retrieval_time}
 

# @app.get("/clip_image_flickr_solr")
# async def search_clip_flickr(q: Union[str, None] = None,k:int = 10):
#     if(debug): print(f"Received CLIP search query for Flickr: {q}")

#     start_time = get_time()
#     result_obj = Clipsearcher_flickr.search(q, top_k=k)
#     # searcher = get_searcher("clip", "flickr", "image")
#     # result_obj = searcher.search(q, top_k=k)
#     result = result_obj["results"]
#     encoding_time = result_obj["encoding_time"]
#     retrieval_time = result_obj["query_time"]


#     for item in result:
#         item["image_abs_path"] = f"{BASE_IMAGE_PATH_FLICKR}/{os.path.basename(item['image_path'])}"
#     end_time = get_time()
#     duration = end_time - start_time

#     if(debug): print(f"CLIP results for Flickr: {result}")
#     return {"list_of_top_k": result, "query_time":duration, "encoding_time": encoding_time, "retrieval_time": retrieval_time}

# @app.get("/clip_text_flickr_solr")
# async def search_clip_caption_flickr(q: Union[str, None] = None,k:int = 10):
#     if(debug): print(f"Received CLIP caption search query for Flickr: {q}")

#     start_time = get_time()
#     result_obj = Clipsearcher_caption_flickr.search(q, top_k=k)
#     # searcher = get_searcher("clip", "flickr", "text")
#     # result_obj = searcher.search(q, top_k=k)

#     result = result_obj["results"]
#     encoding_time = result_obj["encoding_time"]
#     retrieval_time = result_obj["query_time"]

#     for item in result:
#         item["image_abs_path"] = f"{BASE_IMAGE_PATH_FLICKR}/{os.path.basename(item['image_path'])}"
#     end_time = get_time()
#     duration = end_time - start_time

#     if(debug): print(f"CLIP caption results for Flickr: {result}")
#     return {"list_of_top_k": result, "query_time":duration, "encoding_time": encoding_time, "retrieval_time": retrieval_time}


# @app.get("/bm25_text_coco_solr")
# async def search_bm25_caption_mscoco(q: Union[str, None] = None,k:int = 10): 
#     if(debug): print(f"Received BM25 caption search query: {q}")

#     start_time = get_time()
#     result_obj = BM25CaptionSearcher_mscoco.search(q, top_k=k)
#     # searcher = get_searcher("bm25", "coco", "text")
#     # result_obj = searcher.search(q, top_k=k)

#     result = result_obj["results"]
#     retrieval_time = result_obj["query_time"]


#     for item in result:
#         item["image_abs_path"] = f"{BASE_IMAGE_PATH_COCO}/{os.path.basename(item['image_path'])}"
#     end_time = get_time()
#     duration = end_time - start_time

#     if(debug): print(f"BM25 caption results: {result}")
#     return {"list_of_top_k": result, "query_time":duration, "retrieval_time": retrieval_time}


# @app.get("/bm25_text_flickr_solr")
# async def search_bm25_caption_flickr(q: Union[str, None] = None,k:int = 10):     
#     if(debug): print(f"Received BM25 caption search query for Flickr: {q}")
    
#     start_time = get_time()
#     result_obj = BM25CaptionSearcher_flickr.search(q, top_k=k)
#     # searcher = get_searcher("bm25", "flickr", "text")
#     # result_obj = searcher.search(q, top_k=k)

#     result = result_obj["results"]
#     retrieval_time = result_obj["query_time"]

#     for item in result:
#         item["image_abs_path"] = f"{BASE_IMAGE_PATH_FLICKR}/{os.path.basename(item['image_path'])}"
#     end_time = get_time()
#     duration = end_time - start_time

#     if(debug): print(f"BM25 caption results for Flickr: {result}")
#     return {"list_of_top_k": result, "query_time":duration, "retrieval_time": retrieval_time}


# @app.get("/minilm_text_coco_solr")
# async def search_minilm_caption_mscoco(q: Union[str, None] = None,k:int = 10):
#     if(debug): print(f"Received MiniLM caption search query: {q}")

#     start_time = get_time()
#     result_obj = MiniLMSemanticSearcher_mscoco.search(q, top_k=k)
#     # searcher = get_searcher("minilm", "coco", "text")
#     # result_obj = searcher.search(q, top_k=k)

#     result = result_obj["results"]
#     encoding_time = result_obj["encoding_time"]
#     retrieval_time = result_obj["query_time"]

#     for item in result:
#         item["image_abs_path"] = f"{BASE_IMAGE_PATH_COCO}/{os.path.basename(item['image_path'])}"
#     end_time = get_time()
#     duration = end_time - start_time

#     if(debug): print(f"MiniLM caption results: {result}")
#     return {"list_of_top_k": result, "query_time":duration, "encoding_time": encoding_time, "retrieval_time": retrieval_time}


# @app.get("/minilm_text_flickr_solr")
# async def search_minilm_caption_flickr(q: Union[str, None] = None,k:int = 10):
#     if(debug): print(f"Received MiniLM caption search query for Flickr: {q}")
    
#     start_time = get_time()
#     result_obj = MiniLMSemanticSearcher_flickr.search(q, top_k=k)
#     # searcher = get_searcher("minilm", "flickr", "text")
#     # result_obj = searcher.search(q, top_k=k)

#     result = result_obj["results"]
#     encoding_time = result_obj["encoding_time"]
#     retrieval_time = result_obj["query_time"]

#     for item in result:
#         item["image_abs_path"] = f"{BASE_IMAGE_PATH_FLICKR}/{os.path.basename(item['image_path'])}"
#     end_time = get_time()
#     duration = end_time - start_time

#     if(debug): print(f"MiniLM caption results for Flickr: {result}")
#     return {"list_of_top_k": result, "query_time":duration, "encoding_time": encoding_time, "retrieval_time": retrieval_time}


 
# @app.get("/flava_text_coco_solr")
# async def search_flava_caption_mscoco(q: Union[str, None] = None,k:int = 10):
#     if(debug): print(f"Received FLAVA caption search query: {q}")
  
#     start_time = get_time()
#     result_obj = FlavaSearcher_Caption_mscoco.search(q, top_k=k)
#     # searcher = get_searcher("flava", "coco", "text")
#     # result_obj = searcher.search(q, top_k=k)

#     result = result_obj["results"]
#     encoding_time = result_obj["encoding_time"]
#     retrieval_time = result_obj["query_time"]

#     for item in result:
#         item["image_abs_path"] = f"{BASE_IMAGE_PATH_COCO}/{os.path.basename(item['image_path'])}"
#     end_time = get_time()
#     duration = end_time - start_time

#     if(debug): print(f"FLAVA caption results: {result}")
#     return {"list_of_top_k": result, "query_time":duration, "encoding_time": encoding_time, "retrieval_time": retrieval_time}

# @app.get("/flava_text_flickr_solr")
# async def search_flava_caption_flickr(q: Union[str, None] = None,k:int = 10):  
#     if(debug): print(f"Received FLAVA caption search query for Flickr: {q}")
    
#     start_time = get_time()
#     result_obj = FlavaSearcher_Caption_flickr.search(q, top_k=k)
#     # searcher = get_searcher("flava", "flickr", "text")
#     # result_obj = searcher.search(q, top_k=k)

#     result = result_obj["results"]
#     encoding_time = result_obj["encoding_time"]
#     retrieval_time = result_obj["query_time"]

#     for item in result:
#         item["image_abs_path"] = f"{BASE_IMAGE_PATH_FLICKR}/{os.path.basename(item['image_path'])}"
#     end_time = get_time()
#     duration = end_time - start_time

#     if(debug): print(f"FLAVA caption results for Flickr: {result}")
#     return {"list_of_top_k": result, "query_time":duration, "encoding_time": encoding_time, "retrieval_time": retrieval_time}

# @app.get("/flava_image_coco_solr")
# async def search_flava_image_mscoco(q: Union[str, None] = None,k:int = 10):    
#     if(debug): print(f"Received FLAVA image search query: {q}")

#     start_time = get_time()
#     result_obj = FlavaSearcher_mscoco.search(q, top_k=k)
#     # searcher = get_searcher("flava", "coco", "image")
#     # result_obj = searcher.search(q, top_k=k)

#     result = result_obj["results"]
#     encoding_time = result_obj["encoding_time"]
#     retrieval_time = result_obj["query_time"]

#     for item in result:
#         item["image_abs_path"] = f"{BASE_IMAGE_PATH_COCO}/{os.path.basename(item['image_path'])}"
#     end_time = get_time()
#     duration = end_time - start_time

#     if(debug): print(f"FLAVA image results: {result}")
#     return {"list_of_top_k": result, "query_time":duration, "encoding_time": encoding_time, "retrieval_time": retrieval_time}

# @app.get("/flava_image_flickr_solr")
# async def search_flava_image_flickr(q: Union[str, None] = None,k:int = 10):    
#     if(debug): print(f"Received FLAVA image search query for Flickr: {q}")

#     start_time = get_time()
#     result_obj = FlavaSearcher_flickr.search(q, top_k=k)
#     # searcher = get_searcher("flava", "flickr", "image")
#     # result_obj = searcher.search(q, top_k=k)

#     result = result_obj["results"]
#     encoding_time = result_obj["encoding_time"]
#     retrieval_time = result_obj["query_time"]

#     for item in result:
#         item["image_abs_path"] = f"{BASE_IMAGE_PATH_FLICKR}/{os.path.basename(item['image_path'])}"
#     end_time = get_time()
#     duration = end_time - start_time

#     if(debug): print(f"FLAVA image results for Flickr: {result}")
#     return {"list_of_top_k": result, "query_time":duration, "encoding_time": encoding_time, "retrieval_time": retrieval_time}


# @app.get("/uniir_text_coco_solr")
# async def search_uniir_caption_mscoco(q: Union[str, None] = None,k:int = 10):
#     if(debug): print(f"Received UniIR caption search query: {q}")

#     start_time = get_time()
#     result_obj = UniIRSemanticSearcher_Caption_mscoco.search(q, top_k=k)
#     # searcher = get_searcher("uniir", "coco", "text")
#     # result_obj = searcher.search(q, top_k=k)

#     result = result_obj["results"]
#     encoding_time = result_obj["encoding_time"]
#     retrieval_time = result_obj["query_time"] 

#     for item in result:
#         item["image_abs_path"] = f"{BASE_IMAGE_PATH_COCO}/{os.path.basename(item['image_path'])}"
#     end_time = get_time()
#     duration = end_time - start_time
 
#     if(debug): print(f"UniIR caption results: {result}")
#     return {"list_of_top_k": result, "query_time":duration, "encoding_time": encoding_time, "retrieval_time": retrieval_time}

# @app.get("/uniir_text_flickr_solr")
# async def search_uniir_caption_flickr(q: Union[str, None] = None,k:int = 10):
#     if(debug): print(f"Received UniIR caption search query for Flickr: {q}")

#     start_time = get_time()
#     result_obj = UniIRSemanticSearcher_Caption_flickr.search(q, top_k=k)
#     # searcher = get_searcher("uniir", "flickr", "text")
#     # result_obj = searcher.search(q, top_k=k)

#     result = result_obj["results"]
#     encoding_time = result_obj["encoding_time"]
#     retrieval_time = result_obj["query_time"]

#     for item in result:
#         item["image_abs_path"] = f"{BASE_IMAGE_PATH_FLICKR}/{os.path.basename(item['image_path'])}"
#     end_time = get_time()
#     duration = end_time - start_time

#     if(debug): print(f"UniIR caption results for Flickr: {result}")
#     return {"list_of_top_k": result, "query_time":duration, "encoding_time": encoding_time, "retrieval_time": retrieval_time}

# @app.get("/uniir_image_coco_solr")
# async def search_uniir_image_mscoco(q: Union[str, None] = None,k:int = 10):
#     if(debug): print(f"Received UniIR image search query: {q}")

#     start_time = get_time()
#     result_obj = UniIRSemanticSearcher_mscoco_image.search(q, top_k=k)
#     # searcher = get_searcher("uniir", "coco", "image")
#     # result_obj = searcher.search(q, top_k=k)

#     result = result_obj["results"]
#     encoding_time = result_obj["encoding_time"]
#     retrieval_time = result_obj["query_time"]

#     for item in result:
#         item["image_abs_path"] = f"{BASE_IMAGE_PATH_COCO}/{os.path.basename(item['image_path'])}"
#     end_time = get_time()
#     duration = end_time - start_time

#     if(debug): print(f"UniIR image results: {result}")
#     return {"list_of_top_k": result, "query_time":duration, "encoding_time": encoding_time, "retrieval_time": retrieval_time}

# @app.get("/uniir_image_flickr_solr")
# async def search_uniir_image_flickr(q: Union[str, None] = None,k:int = 10):
#     if(debug): print(f"Received UniIR image search query for Flickr: {q}")

#     start_time = get_time()
#     result_obj = UniIRSemanticSearcher_flickr_image.search(q, top_k=k)
#     # searcher = get_searcher("uniir", "flickr", "image")
#     # result_obj = searcher.search(q, top_k=k)

#     result = result_obj["results"]
#     encoding_time = result_obj["encoding_time"]
#     retrieval_time = result_obj["query_time"]

#     for item in result:
#         item["image_abs_path"] = f"{BASE_IMAGE_PATH_FLICKR}/{os.path.basename(item['image_path'])}"
#     end_time = get_time()
#     duration = end_time - start_time

#     if(debug): print(f"UniIR image results for Flickr: {result}")
#     return {"list_of_top_k": result, "query_time":duration, "encoding_time": encoding_time, "retrieval_time": retrieval_time}

# @app.get("/uniir_joint-image-text_coco_solr")
# async def search_uniir_joint_mscoco(q: Union[str, None] = None,k:int = 10):
#     if(debug): print(f"Received UniIR joint image-text search query: {q}")

#     start_time = get_time()
#     result_obj = UniIRSemanticSearcher_mscoco_joint.search(q, top_k=k)
#     # searcher = get_searcher("uniir", "coco", "joint-image-text")
#     # result_obj = searcher.search(q, top_k=k)

#     result = result_obj["results"]
#     encoding_time = result_obj["encoding_time"]
#     retrieval_time = result_obj["query_time"]

#     for item in result:
#         item["image_abs_path"] = f"{BASE_IMAGE_PATH_COCO}/{os.path.basename(item['image_path'])}"
#     end_time = get_time()
#     duration = end_time - start_time

#     if(debug): print(f"UniIR joint image-text results: {result}")
#     return {"list_of_top_k": result, "query_time":duration, "encoding_time": encoding_time, "retrieval_time": retrieval_time}

# @app.get("/uniir_joint-image-text_flickr_solr")
# async def search_uniir_joint_flickr(q: Union[str, None] = None,k:int = 10):    
#     if(debug): print(f"Received UniIR joint image-text search query for Flickr: {q}")
 
#     start_time = get_time()
#     result_obj = UniIRSemanticSearcher_flickr_joint.search(q, top_k=k)
#     # searcher = get_searcher("uniir", "flickr", "joint-image-text")
#     # result_obj = searcher.search(q, top_k=k)

#     result = result_obj["results"]
#     encoding_time = result_obj["encoding_time"]
#     retrieval_time = result_obj["query_time"]

#     for item in result:
#         item["image_abs_path"] = f"{BASE_IMAGE_PATH_FLICKR}/{os.path.basename(item['image_path'])}"
#     end_time = get_time()
#     duration = end_time - start_time

#     if(debug): print(f"UniIR joint image-text results for Flickr: {result}")
#     return {"list_of_top_k": result, "query_time":duration, "encoding_time": encoding_time, "retrieval_time": retrieval_time}

 
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
async def rrf_fusion_endpoint(
    q: str,
    methods: List[str] = Query(...),
    k: int = 10,              # final top-k after fusion
    candidate_k: int = 30,    # per-model fetch size
    rrf_k: int = 60
):
    urls = []
    start_time = get_time() 

    for method in methods:
        if method not in endpoint_map:
            return {"error": f"Invalid method name: {method}"}
        urls.append(endpoint_map[method])

    try:
        fused_result, rrf_time = await run_rrf_fusion(
            q,
            urls,
            k=k,
            candidate_k=candidate_k,
            rrf_k=rrf_k,
        )
        duration = get_time() - start_time
        return {
            "list_of_top_k": fused_result,
            "query_time": duration,
            "rrf_time": rrf_time,
            "k": k,
            "candidate_k": candidate_k,
            "rrf_k": rrf_k,
        }

    except Exception as e:
        return {"error": str(e)}

# @app.get("/rrf2_fusion")
# async def rrf_fusion_endpoint(q: str, methods: List[str] = Query(...)):
#         urls = []
#         start_time = get_time()
#         for method in methods:
#             if method not in endpoint_map:
#                 return {"error": f"Invalid method name: {method}"}
#             urls.append(endpoint_map[method])
#         try:
#             fused_result, rrf_time = await run_rrf_fusion(q, urls)
#             end_time = get_time()
#             duration = end_time - start_time
#             return {"list_of_top_k": fused_result, "query_time":duration, "rrf_time": rrf_time}
            
#         except Exception as e:
#             return {"error": str(e)}   


# Assuming orchestrator and paths are initialized somewhere
# two_stage_orchestrator = TwoStageRetrievalOrchestrator("http://localhost:8983/solr")
@app.get("/two_stage_retrieval")
async def two_stage_retrieval(
    query: str,
    dataset: str = Query(..., description="Dataset: coco or flickr"),
    stage1_model: str = Query(..., description="First stage model: clip, minilm, uniir, flava"),
    stage1_core_type: str = Query(..., description="First stage core type: text or image"),
    stage2_model: str = Query(..., description="Second stage model: clip, minilm, uniir, flava"),
    stage2_core_type: str = Query(..., description="Second stage core type: text or image"),
    stage1_k: int = Query(50, description="Number of results from first stage"),
    stage2_k: int = Query(10, description="Number of final results from second stage")
):
    # Validate dataset and models
    if dataset not in ["coco", "flickr"]:
        raise HTTPException(status_code=400, detail="Dataset must be 'coco' or 'flickr'")
    
    models = ["clip", "minilm", "uniir", "flava", "bm25"]
    if stage1_model not in models or stage2_model not in models:
        raise HTTPException(status_code=400, detail="Model must be one of: clip, minilm, uniir, flava")
    
    core_types = ["text", "image", "joint-image-text"]
    if stage1_core_type not in core_types or stage2_core_type not in core_types:
        raise HTTPException(status_code=400, detail="Core type must be 'text' or 'image' or 'joint-image-text'")
    
    stage1_core = f"{stage1_model}_{dataset}_{stage1_core_type}"
    stage2_core = f"{stage2_model}_{dataset}_{stage2_core_type}"
      
    # Execute pipeline
    start_total = get_time()
    result_summary = two_stage_orchestrator.execute_two_stage_pipeline(
        query_text=query,
        stage1_config=(stage1_model, stage1_core),
        stage2_config=(stage2_model, stage2_core),
        stage1_k=stage1_k,
        stage2_k=stage2_k
    )
    end_total = get_time()
    total_time = end_total - start_total
 
    # Add absolute image paths
    base_image_path = BASE_IMAGE_PATH_COCO if dataset == "coco" else BASE_IMAGE_PATH_FLICKR
    for doc in result_summary.get("final_results", []):
        doc["image_abs_path"] = f"{base_image_path}/{os.path.basename(doc['image_path'])}"
    for doc in result_summary.get("stage1_results", []):
        doc["image_abs_path"] = f"{base_image_path}/{os.path.basename(doc['image_path'])}"

    # Build response with encoding and retrieval times for both stages
    response = {
        "list_of_top_k": result_summary.get("final_results", []),
        "query_time": total_time,
        "encoding_time": {
            "stage1": result_summary.get("stage1", {}).get("encode_time", 0.0),
            "stage2": result_summary.get("stage2", {}).get("encode_time", 0.0)
        },
        "retrieval_time": {
            "stage1": result_summary.get("stage1", {}).get("search_time", 0.0),
            "stage2": result_summary.get("stage2", {}).get("search_time", 0.0)
        }
    } 

    return response

 