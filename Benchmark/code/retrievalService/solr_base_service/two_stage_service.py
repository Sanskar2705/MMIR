"""
solr_two_stage_service.py
─────────────────────────
Standalone two-stage retrieval service.
Uses the TwoStageRetrievalOrchestrator directly.
"""

import sys
import os
from fastapi import FastAPI, HTTPException, Query

sys.path.append("/mnt/storage/RSystemsBenchmarking/gitProject")
from Benchmark.code.evaluation.time_util import get_time
from Benchmark.config.config_utils import load_config
from Benchmark.code.vectorStore.solr.Retriveal.retrieval_orchestrator import TwoStageRetrievalOrchestrator

# Disable proxies
os.environ['no_proxy'] = 'localhost,127.0.0.1'
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'
os.environ["HF_HUB_DISABLE_XET"] = "1"

cfg = load_config()
orchestrator = TwoStageRetrievalOrchestrator("http://localhost:8983/solr")

app = FastAPI(title="Solr Two-Stage Retrieval Service", version="1.0")


@app.get("/two_stage_retrieval")  # ← Changed endpoint path to match client
async def two_stage_retrieval(
    query: str = Query(..., description="Text query"),
    dataset: str = Query(..., description="Dataset: coco or flickr"),
    stage1_model: str = Query(..., description="First stage model: clip, minilm, uniir, flava, bm25"),
    stage1_core_type: str = Query(..., description="First stage core type: text, image, joint-image-text"),  # ← Changed parameter name
    stage2_model: str = Query(..., description="Second stage model: clip, minilm, uniir, flava, bm25"),
    stage2_core_type: str = Query(..., description="Second stage core type: text, image, joint-image-text"),  # ← Changed parameter name
    stage1_k: int = Query(50, ge=1, le=1000, description="Number of results from first stage"),
    stage2_k: int = Query(10, ge=1, le=100, description="Number of final results from second stage")
):
    """
    Two-stage retrieval pipeline.
    
    Example:
        GET /two_stage_retrieval?query=dog&dataset=coco&stage1_model=uniir&stage1_core_type=text&stage2_model=clip&stage2_core_type=image&stage1_k=50&stage2_k=10
    
    Stage 1: Broad retrieval with first model
    Stage 2: Re-ranking with second model on Stage 1 results
    """
    
    # Validate inputs
    if dataset not in ["coco", "flickr"]:
        raise HTTPException(
            status_code=400,
            detail="Dataset must be 'coco' or 'flickr'"
        )
    
    valid_models = ["clip", "minilm", "uniir", "flava", "bm25"]
    if stage1_model not in valid_models:
        raise HTTPException(
            status_code=400,
            detail=f"stage1_model must be one of: {valid_models}"
        )
    if stage2_model not in valid_models:
        raise HTTPException(
            status_code=400,
            detail=f"stage2_model must be one of: {valid_models}"
        )
    
    valid_core_types = ["text", "image", "joint-image-text"]
    if stage1_core_type not in valid_core_types:
        raise HTTPException(
            status_code=400,
            detail=f"stage1_core_type must be one of: {valid_core_types}"
        )
    if stage2_core_type not in valid_core_types:
        raise HTTPException(
            status_code=400,
            detail=f"stage2_core_type must be one of: {valid_core_types}"
        )
    
    # Build core names (matching your Solr core naming convention)
    stage1_core = f"{stage1_model}_{dataset}_{stage1_core_type}"
    stage2_core = f"{stage2_model}_{dataset}_{stage2_core_type}"
    
    try:
        start_total = get_time()
        
        # Execute pipeline using your orchestrator
        result_summary = orchestrator.execute_two_stage_pipeline(
            query_text=query,
            stage1_config=(stage1_model, stage1_core),
            stage2_config=(stage2_model, stage2_core),
            stage1_k=stage1_k,
            stage2_k=stage2_k
        )
        
        end_total = get_time()
        
        # Add absolute image paths
        base_img_path = cfg["paths"]["dataset"][dataset]["base_image_path"]
        
        for doc in result_summary.get("final_results", []):
            img_name = os.path.basename(doc.get('image_path', ''))
            doc["image_abs_path"] = f"{base_img_path}/{img_name}"
        
        for doc in result_summary.get("stage1_results", []):
            img_name = os.path.basename(doc.get('image_path', ''))
            doc["image_abs_path"] = f"{base_img_path}/{img_name}"
        
        return {
            "list_of_top_k": result_summary.get("final_results", []),
            "query_time": end_total - start_total,
            "encoding_time": {
                "stage1": result_summary.get("stage1", {}).get("encode_time", 0.0),
                "stage2": result_summary.get("stage2", {}).get("encode_time", 0.0),
                "total": (
                    result_summary.get("stage1", {}).get("encode_time", 0.0) +
                    result_summary.get("stage2", {}).get("encode_time", 0.0)
                )
            },
            "retrieval_time": {
                "stage1": result_summary.get("stage1", {}).get("search_time", 0.0),
                "stage2": result_summary.get("stage2", {}).get("search_time", 0.0),
                "total": (
                    result_summary.get("stage1", {}).get("search_time", 0.0) +
                    result_summary.get("stage2", {}).get("search_time", 0.0)
                )
            },
            "stage1": {
                "model": stage1_model,
                "core_type": stage1_core_type,  # ← Changed to match client expectations
                "core": stage1_core,
                "k": stage1_k,
                "candidates_found": result_summary.get("stage1", {}).get("candidates_found", 0)
            },
            "stage2": {
                "model": stage2_model,
                "core_type": stage2_core_type,  # ← Changed to match client expectations
                "core": stage2_core,
                "k": stage2_k,
                "results_returned": result_summary.get("stage2", {}).get("results_returned", 0)
            }
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Two-stage retrieval error: {str(e)}"
        )


@app.get("/health")
def health():
    """Check if service is running"""
    return {
        "status": "ok",
        "service": "Two-Stage Retrieval",
        "solr_url": orchestrator.solr_base_url
    }