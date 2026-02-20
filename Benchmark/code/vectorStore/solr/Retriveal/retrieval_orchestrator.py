import os
import sys
import torch
import numpy as np
import pysolr
from typing import List, Dict, Optional, Tuple
import clip
from transformers import AutoModel, AutoTokenizer, FlavaProcessor, FlavaModel
import warnings
from pathlib import Path
from omegaconf.dictconfig import DictConfig
import torch.serialization

sys.path.append('/mnt/storage/RSystemsBenchmarking/gitProject')
from Benchmark.code.evaluation.time_util import get_time
from Benchmark.config.config_utils import load_config

# Disable proxies for local Solr
os.environ['no_proxy'] = 'localhost,127.0.0.1'
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'
os.environ["HF_HUB_DISABLE_XET"] = "1"

# Add safe globals for torch serialization
torch.serialization.add_safe_globals([DictConfig])

def l2norm(t):
    return t / t.norm(dim=-1, keepdim=True)

def _load_uniir(model_cfg: Dict, dev: str):
    import open_clip
    arch = model_cfg["arch"]
    ckpt_path = Path(model_cfg["checkpoint"])
    prompt = model_cfg.get("prompt", "").strip()

    # model, _, _ = open_clip.create_model_and_transforms(arch, pretrained="openai", device=dev)
    model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=None, device=dev)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        # explicitly allow loading full pickle objects
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = state.get("model") or state.get("state_dict") or state
    state = {k.replace("clip_model.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    tok = open_clip.get_tokenizer(arch)
    model.to(dev).eval()

    def _embed(text: str) -> np.ndarray:
        full = f"{prompt} {text}".strip() if prompt else text
        t = tok([full]).to(dev)
        with torch.no_grad():
            vec = l2norm(model.encode_text(t)).cpu().numpy()
        return vec.astype("float32")[0]
    return _embed


class TwoStageRetrievalOrchestrator:
    """
    A flexible orchestrator for two-stage retrieval using different embedding models
    and Solr cores. Assumes all cores for a dataset use 'image_path' as a common identifier.
    """
    def __init__(self, solr_base_url="http://localhost:8983/solr"):
        self.solr_base_url = solr_base_url
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loaded_models = {}  # Cache for loaded models: {model_name: (model, tokenizer, model_type)}
        self.solr_clients = {}   # Cache for Solr clients: {core_name: pysolr.Solr}
        
        # Load configuration for UniIR
        self.config = load_config()
        
        print(f"Orchestrator initialized on device: {self.device}")

    def load_embedding_model(self, model_name: str):
        """Dynamically load an embedding model by name."""
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]

        model, tokenizer = None, None
        model_type = "huggingface"  # default type

        try:
            if model_name.lower() == "clip":
                model, preprocess = clip.load("ViT-L/14@336px", device=self.device)
                model.eval()
                # For CLIP, we treat the preprocess function as the tokenizer
                self.loaded_models[model_name] = (model, preprocess, "clip")
                print(f"Loaded model: {model_name}")
                return self.loaded_models[model_name]

            elif "minilm" in model_name.lower():
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer("/mnt/storage/sanskar/samy/Benchmark/Models/MiniLm")
                # Here we store the encode function directly (like UniIR)
                self.loaded_models[model_name] = (model, None, "minilm")
                return self.loaded_models[model_name]

            elif "uniir" in model_name.lower():
                # Use the custom loader for UniIR
                model_cfg = self.config["models"]["uniir"]
                embed_fn = _load_uniir(model_cfg, str(self.device))
                # For UniIR, we store the embedding function directly
                self.loaded_models[model_name] = (embed_fn, None, "uniir")
                print(f"Loaded model: {model_name}")
                return self.loaded_models[model_name]

            elif "flava" in model_name.lower():
                # Use FlavaProcessor instead of FlavaTokenizer
                model_id = "/mnt/storage/RSystemsBenchmarking/gitProject/Benchmark/Models/flava_full_model"
                processor = FlavaProcessor.from_pretrained(model_id)
                model = FlavaModel.from_pretrained(model_id).to(self.device)
                # For FLAVA, we store the processor as the tokenizer
                self.loaded_models[model_name] = (model, processor, "flava")
                print(f"Loaded model: {model_name}")
                return self.loaded_models[model_name]

            else:
                raise ValueError(f"Unsupported model_name '{model_name}'. Add it explicitly.")


            model.eval()
            self.loaded_models[model_name] = (model, tokenizer, model_type)
            print(f"Loaded model: {model_name}")
            return self.loaded_models[model_name]

        except Exception as e:
            raise ValueError(f"Failed to load model '{model_name}'. Error: {e}")

    def encode_text(self, model_name: str, text: str) -> Optional[Dict]:
        """Encode a text query using the specified model."""
        if not text.strip():
            return None

        model_data = self.load_embedding_model(model_name)
        model, tokenizer, model_type = model_data

        start_time = get_time()
        try:
            with torch.no_grad():
                if model_type == "clip":
                    # CLIP uses its own tokenization
                    tokenized = clip.tokenize([text]).to(self.device)
                    features = model.encode_text(tokenized)
                    embedding = features.cpu().numpy().astype(np.float32).flatten()
                elif model_type == "uniir":
                    # UniIR uses the custom embedding function
                    embedding = model(text)  # model is actually the embed_fn
                elif model_type == "flava":
                    # FLAVA uses the processor for tokenization
                    inputs = tokenizer(text=text, return_tensors="pt", padding=True, truncation=True).to(self.device)
                    features = model.get_text_features(**inputs)[:, 0]  # 768-D [CLS] token
                    features = features / features.norm(dim=-1, keepdim=True)  # L2 normalize
                    embedding = features.cpu().numpy().flatten()

                elif model_type == "minilm":
                    # model here is a SentenceTransformer instance
                    embedding = model.encode(text, normalize_embeddings=True).astype(np.float32).flatten()
                
                else:    
                    raise ValueError(f"Unsupported model type '{model_type}'.")

        except Exception as e:
            print(f"Encoding failed for model '{model_name}': {e}")
            return None

        end_time = get_time()
        encoding_time = end_time - start_time

        return {"embedding": embedding, "encoding_time": encoding_time}


    # The rest of your methods remain the same...
    # get_solr_client, search_core, execute_two_stage_pipeline, print_results
    def get_solr_client(self, core_name: str) -> pysolr.Solr:
        """Get a Solr client for a specific core, creating it if necessary."""
        if core_name not in self.solr_clients:
            core_url = f"{self.solr_base_url}/{core_name}"
            self.solr_clients[core_name] = pysolr.Solr(core_url, timeout=60) # Increased timeout for large KNN
            print(f"Connected to Solr core: {core_name}")
        return self.solr_clients[core_name]

    def search_core_bm25(self, core_name: str, text_query: str, top_k: int, filter_query: str = None) -> Dict:
        solr_client = self.get_solr_client(core_name)
        
        fl = "image_path,caption,score"
        params = {
            "rows": top_k,
            "fl": fl,
            "wt": "json",
            "qf": "caption",          # search only caption field
            "defType": "edismax"      # use EDisMax parser
        }
        if filter_query:
            params["fq"] = filter_query

        try:
            start_time = get_time()
            response = solr_client.search(text_query, **params)
            end_time = get_time()
            search_time = end_time - start_time
            results = list(response)
            return {"results": results, "search_time": search_time, "num_found": len(results)}
        except Exception as e:
            print(f"BM25 search failed on core '{core_name}': {e}")
            return {"results": [], "search_time": 0.0, "num_found": 0}



    def search_core(self, core_name: str, query_vector: np.ndarray, top_k: int, filter_query: str = None) -> Dict:
        """Execute a KNN search on a specified Solr core."""
        solr_client = self.get_solr_client(core_name)
        query_vector_list = query_vector.tolist()
        embedding_str = "[" + ",".join(map(str, query_vector_list)) + "]"

        # The !knn query parser for Solr
        knn_query = f"{{!knn f=embedding_vector topK={top_k}}}{embedding_str}"

        # Fields to return. Ensure 'image_path' is always included.
        fl = "image_path, score" # Add other fields like 'id', 'caption' if needed

        params = {"rows": top_k, "fl": fl, "wt": "json"}
        if filter_query:
            params["fq"] = filter_query

        try:
            start_time = get_time()
            response = solr_client.search(knn_query, **params)
            end_time = get_time()
            search_time = end_time - start_time

            results = list(response)
            return {"results": results, "search_time": search_time, "num_found": len(results)}

        except Exception as e:
            print(f"Search failed on core '{core_name}': {e}")
            return {"results": [], "search_time": 0.0, "num_found": 0}

    def execute_two_stage_pipeline(
            self,
            query_text: str,
            stage1_config: Tuple[str, str], # (model_name, core_name)
            stage2_config: Tuple[str, str], # (model_name, core_name)
            stage1_k: int = 50,
            stage2_k: int = 10
        ) -> Dict:
        """
        Execute a two-stage retrieval pipeline.
        Example:
            stage1_config = ("clip", "clip_coco_text")
            stage2_config = ("minilm", "minilm_coco_text")
        """
        print(f"\n=== Starting Pipeline: {stage1_config} -> {stage2_config} ===")
        print(f"Query: '{query_text}'")

        # --- STAGE 1: Broad Retrieval ---
        # print(f"\nStage 1: Encoding with '{stage1_config[0]}'...")
        # enc_result = self.encode_text(stage1_config[0], query_text)
        # if enc_result is None:
        #     return {"error": "Stage 1 encoding failed", "results": []}
        # stage1_vector = enc_result["embedding"]
        # stage1_encode_time = enc_result["encoding_time"]

        # print(f"Stage 1: Searching core '{stage1_config[1]}' for top-{stage1_k}...")
        # stage1_search = self.search_core(stage1_config[1], stage1_vector, top_k=stage1_k)
        # stage1_results = stage1_search["results"]
        # stage1_time = stage1_search["search_time"]
        # --- STAGE 1: Broad Retrieval ---
        if stage1_config[0].lower() == "bm25":
            print(f"\nStage 1: BM25 search on core '{stage1_config[1]}' for top-{stage1_k}...")
            stage1_search = self.search_core_bm25(stage1_config[1], query_text, top_k=stage1_k)
            stage1_results = stage1_search["results"]
            stage1_time = stage1_search["search_time"]
            stage1_encode_time = 0.0  # no embedding step
        else:
            print(f"\nStage 1: Encoding with '{stage1_config[0]}'...")
            enc_result = self.encode_text(stage1_config[0], query_text)
            if enc_result is None:
                return {"error": "Stage 1 encoding failed", "results": []}
            stage1_vector = enc_result["embedding"]
            stage1_encode_time = enc_result["encoding_time"]

            print(f"Stage 1: Searching core '{stage1_config[1]}' for top-{stage1_k}...")
            stage1_search = self.search_core(stage1_config[1], stage1_vector, top_k=stage1_k)
            stage1_results = stage1_search["results"]
            stage1_time = stage1_search["search_time"]


        if not stage1_results:
            print("Stage 1 returned no results. Aborting pipeline.")
            return {"results": [], "stage1_encode_time": stage1_encode_time, "stage1_search_time": stage1_time}

        # Extract the image_paths from Stage 1 results to create the filter for Stage 2
        candidate_image_paths = [doc["image_path"] for doc in stage1_results if "image_path" in doc]
        if not candidate_image_paths:
            print("No 'image_path' found in Stage 1 results. Aborting pipeline.")
            return {"results": [], "stage1_encode_time": stage1_encode_time, "stage1_search_time": stage1_time}

        # Create the Filter Query for Stage 2
        # Escape paths and wrap in quotes for the Solr query
        escaped_paths = [f'"{path}"' for path in candidate_image_paths]
        stage2_filter_query = f"image_path:({' '.join(escaped_paths)})"
        print(f"Stage 2: Filtering on {len(candidate_image_paths)} candidate images.")

        # --- STAGE 2: Focused Re-Ranking ---
        # print(f"\nStage 2: Encoding with '{stage2_config[0]}'...")
        # enc_result_2 = self.encode_text(stage2_config[0], query_text)
        # if enc_result_2 is None:
        #     return {"error": "Stage 2 encoding failed", "results": []}
        # stage2_vector = enc_result_2["embedding"]
        # stage2_encode_time = enc_result_2["encoding_time"]

        # print(f"Stage 2: Re-ranking with '{stage2_config[1]}' for top-{stage2_k}...")
        # stage2_search = self.search_core(stage2_config[1], stage2_vector, top_k=stage2_k, filter_query=stage2_filter_query)
        # stage2_results = stage2_search["results"]
        # stage2_time = stage2_search["search_time"]
        # --- STAGE 2: Focused Re-Ranking ---
        if stage2_config[0].lower() == "bm25":
            print(f"\nStage 2: BM25 search on core '{stage2_config[1]}' for top-{stage2_k}...")
            stage2_search = self.search_core_bm25(stage2_config[1], query_text, top_k=stage2_k, filter_query=stage2_filter_query)
            stage2_results = stage2_search["results"]
            stage2_time = stage2_search["search_time"]
            stage2_encode_time = 0.0
        else:
            print(f"\nStage 2: Encoding with '{stage2_config[0]}'...")
            enc_result_2 = self.encode_text(stage2_config[0], query_text)
            if enc_result_2 is None:
                return {"error": "Stage 2 encoding failed", "results": []}
            stage2_vector = enc_result_2["embedding"]
            stage2_encode_time = enc_result_2["encoding_time"]

            print(f"Stage 2: Re-ranking with '{stage2_config[1]}' for top-{stage2_k}...")
            stage2_search = self.search_core(stage2_config[1], stage2_vector, top_k=stage2_k, filter_query=stage2_filter_query)
            stage2_results = stage2_search["results"]
            stage2_time = stage2_search["search_time"]

        # --- Compile Results ---
        total_time = stage1_encode_time + stage1_time + stage2_encode_time + stage2_time

        result_summary = {
            "query": query_text,
            "final_results": stage2_results,
            "stage1_results": stage1_results,  # Add stage1 results to the output
            "total_time": total_time,
            "stage1": {
                "model": stage1_config[0],
                "core": stage1_config[1],
                "encode_time": stage1_encode_time,
                "search_time": stage1_time,
                "candidates_found": len(candidate_image_paths)
            },
            "stage2": {
                "model": stage2_config[0],
                "core": stage2_config[1],
                "encode_time": stage2_encode_time,
                "search_time": stage2_time,
                "results_returned": len(stage2_results)
            }
        }
        print(f"Pipeline complete. Total time: {total_time:.4f}s")
        return result_summary

    def print_results(self, pipeline_result: Dict):
        """Print the results of a pipeline run in a readable format."""
        if "error" in pipeline_result:
            print(f"Error: {pipeline_result['error']}")
            return
        if not pipeline_result["final_results"]:
            print("No final results to display.")
            return

        # s1 = pipeline_result["stage1"]
        s2 = pipeline_result["stage2"]

        print(f"\n{'='*50}")
        print(f"QUERY: '{pipeline_result['query']}'")
        print(f"{'='*50}")
        # print(f"STAGE 1: {s1['model']} -> {s1['core']}")
        # print(f"  Encode: {s1['encode_time']:.4f}s | Search: {s1['search_time']:.4f}s | Candidates: {s1['candidates_found']}")
        print(f"STAGE 2: {s2['model']} -> {s2['core']}")
        print(f"  Encode: {s2['encode_time']:.4f}s | Search: {s2['search_time']:.4f}s | Results: {s2['results_returned']}")
        print(f"TOTAL TIME: {pipeline_result['total_time']:.4f}s")
        
        # Print top results from Stage 1
        # print(f"\nTOP 10 RESULTS FROM STAGE 1:")
        # for i, doc in enumerate(pipeline_result["stage1_results"][:10], 1):
        #     print(f"  {i}. {doc.get('image_path', 'N/A')} (Score: {doc.get('score', 'N/A'):.6f})")
        
        # Print final results from Stage 2
        print(f"\nFINAL RESULTS (STAGE 2):")
        for i, doc in enumerate(pipeline_result["final_results"], 1):
            print(f"  {i}. {doc.get('image_path', 'N/A')} (Score: {doc.get('score', 'N/A'):.6f})")