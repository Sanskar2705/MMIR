"""
solr_retrieval_server.py
────────────────────────
FastAPI server for Solr-based retrieval with lazy model loading.
All model logic is abstracted away, searchers are built dynamically.
"""

import sys
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
import logging
import warnings

import pysolr
import torch
from fastapi import FastAPI, HTTPException, Query

sys.path.append("/mnt/storage/RSystemsBenchmarking/gitProject")

from Benchmark.config.config_utils import load_config
from Benchmark.code.evaluation.time_util import get_time

# Disable proxies for local Solr
os.environ['no_proxy'] = 'localhost,127.0.0.1'
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'
os.environ["HF_HUB_DISABLE_XET"] = "1"


# ═══════════════════════════════════════════════════════════════════════════
# LOGGING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

def _configure_runtime_logging(cfg: Dict):
    """Suppress noisy logs when debug is off"""
    debug = bool(cfg.get("debug_logs", False))
    level = logging.DEBUG if debug else logging.WARNING

    logging.getLogger().setLevel(level)
    for name in ["uvicorn", "uvicorn.error", "uvicorn.access", "fastapi", "asyncio"]:
        logging.getLogger(name).setLevel(level)

    try:
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_debug() if debug else hf_logging.set_verbosity_error()
    except Exception:
        pass

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if not debug:
        warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════
# EMBEDDER FACTORY (Model-agnostic)
# ═══════════════════════════════════════════════════════════════════════════

def get_embedder(model_name: str, cfg: Dict, device: str) -> Callable[[str], Dict]:
    """
    Return a function that takes text and returns {embedding: np.ndarray, encoding_time: float}
    This is where ALL model-specific logic lives.
    """
    
    if model_name == "clip":
        import clip
        print(f"Loading CLIP model (ViT-L/14@336px)...")
        model, _ = clip.load("ViT-L/14@336px", device=device)
        model.eval()
        print(f" CLIP model loaded on {device}")
        
        def _embed_clip(text: str) -> Dict:
            t0 = get_time()
            tokenized = clip.tokenize([text]).to(device)
            with torch.no_grad():
                features = model.encode_text(tokenized)
                features = features / features.norm(dim=-1, keepdim=True)
            embedding = features.cpu().numpy().flatten()
            return {"embedding": embedding, "encoding_time": get_time() - t0}
        
        return _embed_clip
    
    elif model_name == "flava":
        from transformers import FlavaProcessor, FlavaModel
        print(f"Loading FLAVA model...")
        processor = FlavaProcessor.from_pretrained("facebook/flava-full")
        model = FlavaModel.from_pretrained("facebook/flava-full").to(device)
        model.eval()
        print(f" FLAVA model loaded on {device}")
        
        def _embed_flava(text: str) -> Dict:
            t0 = get_time()
            inputs = processor(text=text, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                features = model.get_text_features(**inputs)[:, 0]
                features = features / features.norm(dim=-1, keepdim=True)
            embedding = features.cpu().numpy().flatten()
            return {"embedding": embedding, "encoding_time": get_time() - t0}
        
        return _embed_flava
    
    elif model_name == "minilm":
        from sentence_transformers import SentenceTransformer
        model_path = cfg["models"]["minilm"]
        print(f"Loading MiniLM model from {model_path}...")
        model = SentenceTransformer(model_path)
        print(f" MiniLM model loaded")
        
        def _embed_minilm(text: str) -> Dict:
            t0 = get_time()
            embedding = model.encode(text, normalize_embeddings=True)
            return {"embedding": embedding, "encoding_time": get_time() - t0}
        
        return _embed_minilm
    
    elif model_name == "uniir":
        import open_clip
        import numpy as np
        
        arch = cfg["models"]["uniir"]["arch"]
        ckpt_path = Path(cfg["models"]["uniir"]["checkpoint"])
        prompt = cfg["models"]["uniir"].get("prompt", "").strip()
        
        print(f"Loading UniIR model from {ckpt_path}...")
        model, _, _ = open_clip.create_model_and_transforms(arch, pretrained="openai", device=device)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        
        state = state.get("model") or state.get("state_dict") or state
        state = {k.replace("clip_model.", "", 1): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        model.to(device).eval()
        
        tok = open_clip.get_tokenizer(arch)
        print(f" UniIR model loaded on {device}")
        
        def _embed_uniir(text: str) -> Dict:
            t0 = get_time()
            full_text = f"{prompt} {text}".strip() if prompt else text
            tokens = tok([full_text]).to(device)
            with torch.no_grad():
                vec = model.encode_text(tokens)
                vec = vec / vec.norm(dim=-1, keepdim=True)
            embedding = vec.cpu().numpy().astype("float32")[0]
            return {"embedding": embedding, "encoding_time": get_time() - t0}
        
        return _embed_uniir
    
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ═══════════════════════════════════════════════════════════════════════════
# SOLR SEARCHER (Generic wrapper)
# ═══════════════════════════════════════════════════════════════════════════

class SolrSearcher:
    """Generic Solr searcher - works with any model embedder"""
    
    def __init__(
        self,
        core_name: str,
        embed_fn: Optional[Callable] = None,  # None for BM25
        modality: str = "text",
        base_img_path: str = "",
    ):
        self.core_name = core_name
        self.core_url = f"http://localhost:8983/solr/{core_name}"
        self.solr_client = pysolr.Solr(self.core_url, timeout=10)
        self.embed_fn = embed_fn
        self.modality = modality
        self.base_img_path = base_img_path
    
    def search(self, query: str, top_k: int = 10) -> Dict:
        """Execute search and return results with timing"""
        
        # BM25 (no embedding needed)
        if self.embed_fn is None:
            return self._bm25_search(query, top_k)
        
        # Vector search (CLIP, FLAVA, MiniLM, UniIR)
        return self._vector_search(query, top_k)
    
    def _bm25_search(self, query: str, top_k: int) -> Dict:
        """BM25 text search (no embeddings)"""
        try:
            t0 = get_time()
            results = self.solr_client.search(query, **{
                "qf": "caption",
                "rows": top_k,
                "fl": "image_path,caption,score",
                "defType": "edismax",
                "wt": "json"
            })
            query_time = get_time() - t0
            
            hits = []
            for doc in results:
                hits.append({
                    "image_path": doc.get("image_path", ""),
                    "image_abs_path": f"{self.base_img_path}/{os.path.basename(doc.get('image_path', ''))}",
                    "caption": doc.get("caption", []),
                    "score": doc.get("score", 0.0)
                })
            
            return {
                "results": hits,
                "encoding_time": 0.0,
                "query_time": query_time
            }
        
        except Exception as e:
            print(f"BM25 search error: {e}")
            return {"results": [], "encoding_time": 0.0, "query_time": 0.0}
    
    def _vector_search(self, query: str, top_k: int) -> Dict:
        """Vector similarity search"""
        try:
            # Encode query
            embed_result = self.embed_fn(query)
            embedding = embed_result["embedding"]
            encoding_time = embed_result["encoding_time"]
            
            # Search Solr
            t0 = get_time()
            embedding_str = "[" + ",".join(map(str, embedding.tolist())) + "]"
            knn_query = f"{{!knn f=embedding_vector topK={top_k} bruteForce=true}}{embedding_str}"
            
            # Field selection based on modality
            if self.modality == "text":
                fl = "image_path,caption,score"
            elif self.modality in ["image", "joint-image-text"]:
                fl = "image_path,score"
            else:
                fl = "image_path,caption,score"
            
            results = self.solr_client.search(knn_query, **{
                "rows": top_k,
                "fl": fl,
                "wt": "json"
            })
            query_time = get_time() - t0
            
            hits = []
            for doc in results:
                hit = {
                    "image_path": doc.get("image_path", ""),
                    "image_abs_path": f"{self.base_img_path}/{os.path.basename(doc.get('image_path', ''))}",
                    "score": doc.get("score", 0.0)
                }
                if "caption" in doc:
                    hit["caption"] = doc.get("caption", [])
                hits.append(hit)
            
            return {
                "results": hits,
                "encoding_time": encoding_time,
                "query_time": query_time
            }
        
        except Exception as e:
            print(f"Vector search error: {e}")
            import traceback
            traceback.print_exc()
            return {"results": [], "encoding_time": 0.0, "query_time": 0.0}


# ═══════════════════════════════════════════════════════════════════════════
# REGISTRY BUILDER (Lazy loading + caching)
# ═══════════════════════════════════════════════════════════════════════════

def build_registry(cfg: Dict):
    """
    Build registry of available searchers.
    Models are loaded ONLY when first accessed (lazy loading).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    registry: Dict[Tuple[str, str, str], SolrSearcher] = {}
    
    # Define all possible combinations
    models = ["clip", "flava", "minilm", "uniir", "bm25"]
    datasets = ["coco", "flickr"]
    modalities_map = {
        "clip": ["image", "text"],
        "flava": ["image", "text"],
        "minilm": ["text"],
        "uniir": ["image", "text", "joint-image-text"],
        "bm25": ["text"]
    }
    
    for model in models:
        for dataset in datasets:
            for modality in modalities_map[model]:
                # Normalize modality names
                modality_normalized = modality
                
                core_name = f"{model}_{dataset}_{modality_normalized}"
                base_img_path = cfg["paths"]["dataset"][dataset]["base_image_path"]
                
                # BM25 doesn't need embedder
                if model == "bm25":
                    searcher = SolrSearcher(
                        core_name=core_name,
                        embed_fn=None,
                        modality=modality_normalized,
                        base_img_path=base_img_path
                    )
                else:
                    # Create a lazy embedder loader
                    def make_lazy_embedder(model_name):
                        embedder = None
                        def lazy_embed(text: str) -> Dict:
                            nonlocal embedder
                            if embedder is None:
                                print(f"🔄 Lazy-loading {model_name} embedder...")
                                embedder = get_embedder(model_name, cfg, device)
                            return embedder(text)
                        return lazy_embed
                    
                    searcher = SolrSearcher(
                        core_name=core_name,
                        embed_fn=make_lazy_embedder(model),
                        modality=modality_normalized,
                        base_img_path=base_img_path
                    )
                
                registry[(model, dataset, modality_normalized)] = searcher
    
    return registry


# ═══════════════════════════════════════════════════════════════════════════
# FASTAPI APPLICATION
# ═══════════════════════════════════════════════════════════════════════════

cfg = load_config()
_configure_runtime_logging(cfg)
SEARCHERS = build_registry(cfg)
TOP_K = 10

app = FastAPI(title="Solr Retrieval Server (Optimized)", version="2.0")


@app.get("/available")
def available():
    """List all available endpoints"""
    return {
        "available": [
            f"{model}_{modality}_{dataset}_solr"
            for (model, dataset, modality) in SEARCHERS.keys()
        ]
    }


@app.get("/{combo}")
def search(
    combo: str,
    q: str = Query(..., description="Text query"),
    k: int = Query(TOP_K, ge=1, le=100, description="Top-K to return"),
):
    """
    Universal search endpoint
    Pattern: <model>_<modality>_<dataset>_solr
    Examples:
        - clip_image_coco_solr
        - flava_text_flickr_solr
        - uniir_joint-image-text_coco_solr
    """
    parts = combo.lower().split("_")
    
    # Handle joint-image-text special case
    if len(parts) >= 5 and parts[-4] == "joint" and parts[-3] == "image":
        model = parts[0]
        modality = "joint-image-text"
        dataset = parts[-2]
        service = parts[-1]
    elif len(parts) == 4:
        model, modality, dataset, service = parts
    else:
        raise HTTPException(
            status_code=400,
            detail="Path must be <model>_<modality>_<dataset>_solr"
        )
    
    if service != "solr":
        raise HTTPException(status_code=400, detail="Service must be 'solr'")
    
    # Normalize aliases
    if modality == "caption":
        modality = "text"
    
    key = (model, dataset, modality)
    if key not in SEARCHERS:
        raise HTTPException(
            status_code=404,
            detail=f"No Solr core for {key}. Available: {list(SEARCHERS.keys())}"
        )
    
    result = SEARCHERS[key].search(q, k)
    
    return {
        "list_of_top_k": result["results"],
        "query_time": result["encoding_time"] + result["query_time"],
        "encoding_time": result["encoding_time"],
        "retrieval_time": result["query_time"]
    }


@app.post("/refresh")
def refresh():
    """Reload config and rebuild registry"""
    global cfg, SEARCHERS
    cfg = load_config()
    _configure_runtime_logging(cfg)
    SEARCHERS = build_registry(cfg)
    return {
        "ok": True,
        "available": [
            f"{model}_{modality}_{dataset}_solr"
            for (model, dataset, modality) in SEARCHERS.keys()
        ]
    }


@app.get("/cache_status")
def cache_status():
    """Show which models are loaded in memory"""
    loaded_models = set()
    for (model, _, _), searcher in SEARCHERS.items():
        if searcher.embed_fn is not None:
            # Check if lazy embedder has been triggered
            loaded_models.add(model)
    
    return {
        "total_endpoints": len(SEARCHERS),
        "loaded_models": list(loaded_models),
        "memory_gb": f"{torch.cuda.memory_allocated() / 1e9:.2f}" if torch.cuda.is_available() else "N/A"
    }


# ═══════════════════════════════════════════════════════════════════════════
# ADDITIONAL ENDPOINTS (RRF, Two-Stage)
# ═══════════════════════════════════════════════════════════════════════════

# from Benchmark.code.retrievalService.reranking.rrf import run_rrf_fusion
# from Benchmark.code.vectorStore.solr.Retriveal.retrieval_orchestrator import TwoStageRetrievalOrchestrator

# # Build endpoint map dynamically based on registry
# def build_endpoint_map(base_url: str = "http://localhost:5055"):
#     """Generate endpoint URLs for all available searchers"""
#     endpoint_map = {}
#     for (model, dataset, modality) in SEARCHERS.keys():
#         # Build the key as expected by your existing code
#         key = f"{model}_{modality}_{dataset}_solr"
#         endpoint_map[key] = f"{base_url}/{key}"
#     return endpoint_map



@app.post("/refresh")
def refresh():
    """Reload config and rebuild registry"""
    global cfg, SEARCHERS, endpoint_map
    cfg = load_config()
    _configure_runtime_logging(cfg)
    SEARCHERS = build_registry(cfg)
    endpoint_map = build_endpoint_map()
    
    return {
        "ok": True,
        "available_endpoints": len(endpoint_map),
        "sample_endpoints": list(endpoint_map.keys())[:10]
    }


@app.get("/cache_status")
def cache_status():
    """Show which models are loaded in memory"""
    return {
        "total_endpoints": len(SEARCHERS),
        "available_methods": list(endpoint_map.keys()),
        "memory_gb": f"{torch.cuda.memory_allocated() / 1e9:.2f}" if torch.cuda.is_available() else "N/A"
    }