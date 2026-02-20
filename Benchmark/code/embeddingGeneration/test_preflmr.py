#!/usr/bin/env python3
"""
PreFLMR ColBERT-style indexer for your COCO/Flickr mappings.

- Reads dataset + mapping from your config.yaml (ingest_annotations_path / base_image_path)
- Builds a custom_collection with ONE doc per image:
    doc_text = concat(all captions for the image)  (truncated by doc_maxlen at indexing time)
    doc_image = absolute path to the image
- Runs FLMR indexer (no FAISS)
- Persists meta.json (doc_id -> {image_name, captions, text})

Usage:
  python test_preflmr.py \
      --dataset flickr \
      --checkpoint LinWeizheDragon/PreFLMR_ViT-L \
      --image-processor laion/CLIP-ViT-bigG-14-laion2B-39B-b160k \
      --index-root /mnt/storage/RSystemsBenchmarking/gitProject/Benchmark/data/preflmr_index \
      --experiment FLICKR \
      --index-name preflmr_vitl_nb8 \
      --nbits 8 \
      --doc-maxlen 512 \
      --use-gpu
"""
import os, sys, json, argparse
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.append("/mnt/storage/RSystemsBenchmarking/gitProject")  # project root
from Benchmark.config.config_utils import load_config  # noqa: E402

os.environ["HF_HUB_DISABLE_XET"] = "1"

# put this near the top of test_preflmr.py, before calling index_custom_collection
from transformers.modeling_utils import PreTrainedModel

__orig_save_pretrained = PreTrainedModel.save_pretrained

def _save_pretrained_no_safetensors(self, *args, **kwargs):
    # disable safetensors so shared/ tied weights don't crash saving
    kwargs.setdefault("safe_serialization", False)
    return __orig_save_pretrained(self, *args, **kwargs)

PreTrainedModel.save_pretrained = _save_pretrained_no_safetensors


def _abs_img(base: Path, rel: str) -> Path:
    p = (base / rel).resolve()
    if p.exists():
        return p
    # try basename fallback (your datasets often need this)
    q = (base / Path(rel).name).resolve()
    return q if q.exists() else p

def _build_collection(mapping_path: Path, base_img_dir: Path) -> Tuple[List[Tuple[str, None, str]], List[Dict]]:
    """
    Returns:
      custom_collection: list of (passage_text, None, image_abs_path)
      meta: list of {image_name, captions, text}
    """
    raw = json.loads(mapping_path.read_text())
    # Expect list[ { "image": "...", "caption": [..] } ]
    docs = []
    meta = []
    for item in raw:
        img_rel = item.get("image")
        caps    = item.get("caption", [])
        caps = [c for c in caps if isinstance(c, str)]
        doc_text = " [SEP] ".join(caps) if caps else ""
        img_abs  = str(_abs_img(base_img_dir, img_rel))
        docs.append((doc_text, None, img_abs))
        meta.append({"image_name": img_rel, "captions": caps, "text": doc_text})
    return docs, meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None, help="coco | flickr (defaults to selected_config.dataset)")
    ap.add_argument("--checkpoint", default="LinWeizheDragon/PreFLMR_ViT-L")
    ap.add_argument("--image-processor", default="laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
    ap.add_argument("--index-root", default=None, help="Index root dir (defaults to <project_root>/data/preflmr_index)")
    ap.add_argument("--experiment", default=None, help="Experiment name (defaults to DATASET uppercased)")
    ap.add_argument("--index-name", default="preflmr_vitl_nb8")
    ap.add_argument("--nbits", type=int, default=8)
    ap.add_argument("--doc-maxlen", type=int, default=512)
    ap.add_argument("--indexing-batch-size", type=int, default=64)
    ap.add_argument("--nranks", type=int, default=1)
    ap.add_argument("--use-gpu", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    dataset = args.dataset or cfg["selected_config"]["dataset"]
    ds_cfg = cfg["paths"]["dataset"][dataset]
    mapping_path = Path(ds_cfg["ingest_annotations_path"])
    base_img_dir = Path(ds_cfg["base_image_path"])

    # Where to store index files
    default_root = Path(cfg["paths"]["project_root"]) / "data" / "preflmr_index"
    index_root = Path(args.index_root) if args.index_root else default_root
    index_root.mkdir(parents=True, exist_ok=True)

    experiment = args.experiment or dataset.upper()
    index_name = args.index_name

    # Lazy import PreFLMR after flags parsed
    from transformers import AutoImageProcessor
    from flmr import (
        index_custom_collection, FLMRConfig,
        FLMRQueryEncoderTokenizer, FLMRContextEncoderTokenizer,
        FLMRModelForRetrieval,
    )

    print(f"[PreFLMR] Loading mapping: {mapping_path}")
    custom_collection, meta = _build_collection(mapping_path, base_img_dir)
    print(f"[PreFLMR] Documents: {len(custom_collection)}")

    # Load model + tokenizers exactly as documented by FLMR
    print(f"[PreFLMR] Loading model: {args.checkpoint}")
    flmr_config = FLMRConfig.from_pretrained(args.checkpoint)
    query_tokenizer = FLMRQueryEncoderTokenizer.from_pretrained(
        args.checkpoint, text_config=flmr_config.text_config, subfolder="query_tokenizer"
    )
    context_tokenizer = FLMRContextEncoderTokenizer.from_pretrained(
        args.checkpoint, text_config=flmr_config.text_config, subfolder="context_tokenizer"
    )
    model = FLMRModelForRetrieval.from_pretrained(
        args.checkpoint,
        query_tokenizer=query_tokenizer,
        context_tokenizer=context_tokenizer,
    )
    _ = AutoImageProcessor.from_pretrained(args.image_processor)  # ensures dependency cached

    # Build index
    print(f"[PreFLMR] Indexing → {index_root} / {experiment} / {index_name}")
    index_custom_collection(
        custom_collection=custom_collection,
        model=model,
        index_root_path=str(index_root),
        index_experiment_name=experiment,
        index_name=index_name,
        nbits=args.nbits,
        doc_maxlen=args.doc_maxlen,
        overwrite=args.overwrite,
        use_gpu=bool(args.use_gpu),
        indexing_batch_size=args.indexing_batch_size,
        model_temp_folder=str(index_root / "tmp"),
        nranks=args.nranks,
    )

    # Save meta file alongside index for serving
    index_dir = index_root / experiment / index_name
    index_dir.mkdir(parents=True, exist_ok=True)
    meta_path = index_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[PreFLMR] ✓ Wrote meta → {meta_path}")

if __name__ == "__main__":
    main()
