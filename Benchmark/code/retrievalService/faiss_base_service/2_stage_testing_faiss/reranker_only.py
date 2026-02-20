#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FAISS Reranking-Only Pipeline (Batch Processing Support)

Performs only the Stage 2 reranking from the two-stage pipeline.
Reads Stage 1 results from a JSON file and applies the same reranking logic
as in faiss_two_stage_orchestrator.py. Supports both single query and batch processing.

Expected JSON format for Stage 1 results (batch format):
{
    "queries": [
        {
            "query": "first text query string",
            "stage1_hits": [
                {
                    "image_path": "path/to/image1.jpg",
                    "score": 0.85,
                    "rank": 1
                },
                {
                    "image_path": "path/to/image2.jpg", 
                    "score": 0.78,
                    "rank": 2
                }
                // ... more results
            ]
        },
        {
            "query": "second text query string",
            "stage1_hits": [
                // ... stage1 results for second query
            ]
        }
        // ... more queries
    ]
}

Alternative single query format (backward compatibility):
{
    "query": "text query string",
    "stage1_hits": [...]
}

Usage:
    python reranker_only.py stage1_results.json --model clip --core-type image --k 10
    python reranker_only.py batch_results.json --model minilm --core-type text --k 5 --dataset coco
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

# Import the same dependencies as the original orchestrator
import sys
import os
# Add the Benchmark root directory to Python path
current_dir = Path(__file__).resolve().parent
benchmark_root = current_dir.parent.parent.parent.parent.parent
sys.path.insert(0, str(benchmark_root))

from Benchmark.config.config_utils import load_config
from Benchmark.code.retrievalService.faiss_base_service.testing.embed_utils import get_embedder
from Benchmark.code.evaluation.time_util import get_time

# Import helper functions from the original orchestrator
def _l2norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)

def _target_from_core_type(ct: str) -> str:
    mapping = {
        "text": "caption", 
        "image": "image",
        "joint-image-text": "joint-image-text"
    }
    return mapping.get(ct, ct)

def _paths(index_dir: Path, model: str, dataset: str, target: str, engine_tag: str):
    """Generate file paths for index and metadata"""
    if target == "caption":
        core_suffix = "txt"
    elif target == "image": 
        core_suffix = "img"
    elif target == "joint-image-text":
        core_suffix = "joint-image-text"
    else:
        core_suffix = target
    
    # Index files have faiss in the name
    index_path = index_dir / f"{model}_{dataset}_{engine_tag}_{core_suffix}.index"
    
    # Metadata files follow different naming patterns
    if target == "image":
        meta_path = index_dir / f"{model}_{dataset}_img_names.json"
    else:
        meta_path = index_dir / f"{model}_{dataset}_{core_suffix}_meta.json"
    
    return index_path, meta_path

def _load_meta(meta_path: Path, target: str):
    """Load metadata JSON"""
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta file not found: {meta_path}")
    
    with open(meta_path, "r") as f:
        meta = json.load(f)
    
    # Create image name mapping
    ids_by_image = {}
    for i, row in enumerate(meta):
        image_name = row.get("image_name", "")
        if image_name:
            key = image_name.lower().strip()  # canonical form
            ids_by_image.setdefault(key, []).append(i)
    
    return meta, ids_by_image

def _reconstruct_vec(index, rid: int, dim: int) -> np.ndarray:
    """Reconstruct vector from FAISS index"""
    try:
        return index.reconstruct(rid)
    except Exception:
        try:
            buf = np.empty((dim,), dtype="float32")
            import faiss
            index.reconstruct_into(rid, faiss.swig_ptr(buf))
            return buf
        except Exception:
            return np.zeros((dim,), dtype="float32")

# Cache for loaded components
EMBED_CACHE = {}
SEARCHER_CACHE = {}

def _get_embedder(model: str):
    """Get cached embedder"""
    if model not in EMBED_CACHE:
        cfg = load_config()
        def _device():
            if torch.cuda.is_available():
                return "cuda"
            return "cpu"
        EMBED_CACHE[model] = get_embedder(model, cfg, _device())
    return EMBED_CACHE[model]

def _get_searcher(model: str, dataset: str, target: str, engine_tag: str):
    """Get cached FAISS searcher components"""
    key = (model, dataset, target, engine_tag)
    if key not in SEARCHER_CACHE:
        cfg = load_config()
        index_dir = Path(cfg["vector_store"]["faiss"]["index_dir"])
        
        index_path, meta_path = _paths(index_dir, model, dataset, target, engine_tag)
        
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        
        import faiss
        index = faiss.read_index(str(index_path))
        meta, ids_by_image = _load_meta(meta_path, target)
        
        SEARCHER_CACHE[key] = (index, meta, ids_by_image)
    
    return SEARCHER_CACHE[key]

class FAISSReranker:
    """
    Reranking-only pipeline that mirrors the Stage 2 logic from TwoStageFAISSOrchestrator
    Supports both single query and batch processing
    """
    
    def __init__(self):
        pass
    
    def encode_text(self, model_name: str, text: str) -> Optional[Dict]:
        """Encode text using the specified model"""
        try:
            embedder = _get_embedder(model_name)
            t1 = get_time()
            embedding = embedder(text)
            encoding_time = get_time() - t1
            
            # Normalize embedding
            if isinstance(embedding, np.ndarray):
                embedding = _l2norm(embedding)
            else:
                embedding = _l2norm(np.array(embedding))
            
            return {
                "embedding": embedding,
                "encoding_time": encoding_time
            }
        except Exception as e:
            print(f"Error encoding text with model {model_name}: {e}")
            return None
    
    def _score_subset(self, model: str, dataset: str, target: str, engine_tag: str, 
                      q: np.ndarray, image_names: List[str], k: int) -> Tuple[List[Dict], float]:
        """
        Score subset of candidates using the same logic as the original orchestrator
        """
        index, meta, ids_by_image = _get_searcher(model, dataset, target, engine_tag)
        dim = index.d
        
        t1 = get_time()
        rows = []
        
        for name in image_names:
            # Use canonical form for lookup (lowercase, stripped)
            canonical_name = name.lower().strip()
            ids = ids_by_image.get(canonical_name, [])
            
            for rid in ids:
                v = _reconstruct_vec(index, rid, dim)
                v = _l2norm(v)  # Normalize reconstructed vector
                s = float(np.dot(q, v))
                rows.append({
                    "image_path": name,  # Use original name in output
                    "caption": meta[rid].get("caption", ""),
                    "score": s
                })
        
        # Sort by score descending
        rows.sort(key=lambda r: r["score"], reverse=True)
        dt = get_time() - t1
        
        # Format output with ranks
        out = []
        for rank, r in enumerate(rows[:k], start=1):
            out.append({
                "image_path": r["image_path"],
                "caption": r["caption"],
                "score": float(r["score"]),
                "rank": rank
            })
        
        return out, dt
    
    def rerank_single_query(
        self,
        query_text: str,
        stage1_hits: List[Dict],
        model: str,
        core_type: str,
        dataset: str = "coco",
        k: int = 10,
        engine_tag: str = "faiss"
    ) -> Dict:
        """
        Perform reranking for a single query
        """
        if not stage1_hits:
            return {
                "query": query_text,
                "final_results": [],
                "stage1_results": stage1_hits,
                "reranking_time": 0.0,
                "total_candidates": 0,
                "error": "No Stage 1 results provided"
            }
        
        # Extract unique candidate image paths (same logic as original)
        candidate_image_paths = sorted(set(h["image_path"] for h in stage1_hits if h.get("image_path")))
        
        if not candidate_image_paths:
            return {
                "query": query_text,
                "final_results": [],
                "stage1_results": stage1_hits,
                "reranking_time": 0.0,
                "total_candidates": 0,
                "error": "No valid image paths in Stage 1 results"
            }
        
        # Encode query with reranking model
        target = _target_from_core_type(core_type)
        enc_result = self.encode_text(model, query_text)
        
        if enc_result is None:
            return {
                "query": query_text,
                "final_results": [],
                "stage1_results": stage1_hits,
                "reranking_time": 0.0,
                "total_candidates": len(candidate_image_paths),
                "error": "Failed to encode query"
            }
        
        q = enc_result["embedding"]
        
        # Perform reranking
        reranked_results, rerank_time = self._score_subset(
            model, dataset, target, engine_tag, q, candidate_image_paths, k
        )
        
        total_time = enc_result["encoding_time"] + rerank_time
        
        return {
            "query": query_text,
            "final_results": reranked_results,
            "stage1_results": stage1_hits,
            "reranking_time": rerank_time,
            "encoding_time": enc_result["encoding_time"],
            "total_time": total_time,
            "total_candidates": len(candidate_image_paths),
            "model": model,
            "core_type": core_type,
            "dataset": dataset,
            "k": k
        }

    def rerank_from_file(
        self,
        stage1_file: str,
        model: str,
        core_type: str,
        dataset: str = "coco",
        k: int = 10,
        engine_tag: str = "faiss"
    ) -> Dict:
        """
        Perform reranking using Stage 1 results from a JSON file (supports batch processing)
        
        Args:
            stage1_file: Path to JSON file containing Stage 1 results (single query or batch)
            model: Model name for Stage 2 (clip, minilm, uniir, flava)
            core_type: Core type for Stage 2 (text, image, joint-image-text)
            dataset: Dataset name (coco, flickr, etc.)
            k: Number of results to return
            engine_tag: Engine tag (faiss)
        
        Returns:
            Dictionary with reranked results and timing info (batch or single)
        """
        print(f"\n=== FAISS Reranking Pipeline ===")
        print(f"Stage 1 file: {stage1_file}")
        print(f"Model: {model}, Core: {core_type}, Dataset: {dataset}")
        
        # Load Stage 1 results
        stage1_path = Path(stage1_file)
        if not stage1_path.exists():
            raise FileNotFoundError(f"Stage 1 file not found: {stage1_file}")
        
        with open(stage1_path, "r") as f:
            stage1_data = json.load(f)
        
        # Check if it's batch format (has "queries" key) or single query format
        if "queries" in stage1_data:
            # Batch processing
            queries = stage1_data["queries"]
            print(f"Processing {len(queries)} queries in batch mode")
            
            batch_results = []
            total_batch_time = 0.0
            successful_queries = 0
            
            for i, query_data in enumerate(queries, 1):
                query_text = query_data.get("query", "")
                stage1_hits = query_data.get("stage1_hits", [])
                
                print(f"\n--- Query {i}/{len(queries)}: '{query_text[:50]}{'...' if len(query_text) > 50 else ''}' ---")
                print(f"Stage 1 candidates: {len(stage1_hits)}")
                
                # Process single query
                result = self.rerank_single_query(
                    query_text, stage1_hits, model, core_type, dataset, k, engine_tag
                )
                
                if "error" not in result:
                    successful_queries += 1
                    candidate_count = len(set(h["image_path"] for h in stage1_hits if h.get("image_path")))
                    print(f"Unique images for reranking: {candidate_count}")
                    print(f"Reranking completed in {result.get('reranking_time', 0):.3f}s")
                    print(f"Returned {len(result.get('final_results', []))} results")
                else:
                    print(f"Error: {result['error']}")
                
                total_batch_time += result.get("total_time", 0)
                batch_results.append(result)
            
            print(f"\n=== Batch Summary ===")
            print(f"Total queries processed: {len(queries)}")
            print(f"Successful queries: {successful_queries}")
            print(f"Total batch time: {total_batch_time:.3f}s")
            print(f"Average time per query: {total_batch_time/len(queries):.3f}s")
            
            return {
                "batch_results": batch_results,
                "batch_summary": {
                    "total_queries": len(queries),
                    "successful_queries": successful_queries,
                    "total_batch_time": total_batch_time,
                    "average_time_per_query": total_batch_time / len(queries) if queries else 0
                },
                "model": model,
                "core_type": core_type,
                "dataset": dataset,
                "k": k
            }
        
        else:
            # Single query processing (backward compatibility)
            query_text = stage1_data.get("query", "")
            stage1_hits = stage1_data.get("stage1_hits", [])
            
            print(f"Processing single query: '{query_text}'")
            print(f"Stage 1 candidates: {len(stage1_hits)}")
            
            result = self.rerank_single_query(
                query_text, stage1_hits, model, core_type, dataset, k, engine_tag
            )
            
            if "error" not in result:
                candidate_count = len(set(h["image_path"] for h in stage1_hits if h.get("image_path")))
                print(f"Unique images for reranking: {candidate_count}")
                print(f"Reranking completed in {result.get('reranking_time', 0):.3f}s (total: {result.get('total_time', 0):.3f}s)")
                print(f"Returned {len(result.get('final_results', []))} results")
            else:
                print(f"Error: {result['error']}")
            
            return result
    
    def print_results(self, result: Dict):
        """Print reranking results in a formatted way (handles both single and batch)"""
        if "batch_results" in result:
            # Batch results
            print(f"\n=== Batch Reranking Results ===")
            batch_summary = result.get("batch_summary", {})
            print(f"Model: {result.get('model', 'N/A')}")
            print(f"Total queries: {batch_summary.get('total_queries', 0)}")
            print(f"Successful queries: {batch_summary.get('successful_queries', 0)}")
            print(f"Total batch time: {batch_summary.get('total_batch_time', 0):.3f}s")
            print(f"Average time per query: {batch_summary.get('average_time_per_query', 0):.3f}s")
            
            batch_results = result.get("batch_results", [])
            for i, query_result in enumerate(batch_results, 1):
                print(f"\n--- Query {i} Results ---")
                self._print_single_result(query_result)
        else:
            # Single query result
            print(f"\n=== Reranking Results ===")
            self._print_single_result(result)
    
    def _print_single_result(self, result: Dict):
        """Print results for a single query"""
        print(f"Query: {result.get('query', 'N/A')}")
        print(f"Model: {result.get('model', 'N/A')}")
        print(f"Total candidates: {result.get('total_candidates', 0)}")
        print(f"Encoding time: {result.get('encoding_time', 0):.3f}s")
        print(f"Reranking time: {result.get('reranking_time', 0):.3f}s")
        print(f"Total time: {result.get('total_time', 0):.3f}s")
        
        if result.get('error'):
            print(f"Error: {result['error']}")
            return
        
        final_results = result.get('final_results', [])
        print(f"Top {len(final_results)} reranked results:")
        
        for i, hit in enumerate(final_results, 1):
            image_path = hit.get('image_path', 'N/A')
            score = hit.get('score', 0.0)
            caption = hit.get('caption', '')
            
            print(f"{i:2d}. {image_path} (score: {score:.4f})")
            if caption:
                print(f"    Caption: {caption[:100]}{'...' if len(caption) > 100 else ''}")


def main():
    parser = argparse.ArgumentParser(description="FAISS Reranking-Only Pipeline (Batch Support)")
    parser.add_argument("stage1_file", help="JSON file containing Stage 1 results (single or batch)")
    parser.add_argument("--model", required=True, choices=["clip", "minilm", "uniir", "flava"],
                        help="Model for reranking")
    parser.add_argument("--core-type", required=True, choices=["text", "image", "joint-image-text"],
                        help="Core type for reranking")
    parser.add_argument("--dataset", default="coco", choices=["coco", "flickr"],
                        help="Dataset name")
    parser.add_argument("--k", type=int, default=10, help="Number of results to return per query")
    parser.add_argument("--engine", default="faiss", help="Engine tag")
    parser.add_argument("--output", help="Optional JSON file to save results")
    
    args = parser.parse_args()
    
    reranker = FAISSReranker()
    
    try:
        result = reranker.rerank_from_file(
            stage1_file=args.stage1_file,
            model=args.model,
            core_type=args.core_type,
            dataset=args.dataset,
            k=args.k,
            engine_tag=args.engine
        )
        
        # Print results
        reranker.print_results(result)
        
        # Save to file if specified
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\nResults saved to: {args.output}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()