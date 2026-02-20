#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build OFFICIAL ColBERT-format indices for PreFLMR:
  • caption : 1 doc per caption (text-only)
  • joint   : 1 doc per (image, caption) pair (text + image)
  • image   : 1 doc per image (vision-only approximation via empty text + image)

Outputs (ColBERT layout incl. doclens.json, plan.json, shards) under:
  <cfg.vector_store.flmr.root_dir>/<experiment>/<prefix>_{caption|joint|image}/

Also writes a docID-aligned collection_meta.json so a server can map results.
"""

# --- BOOTSTRAP: ensure ColBERT repo root (has 'utility' & 'colbert') is on sys.path ---
import sys, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # e.g. /mnt/storage/ziv/FLMR_CURRENT
CAND = ROOT / "FLMR" / "third_party" / "ColBERT"
if (CAND / "utility").exists() and str(CAND) not in sys.path:
    sys.path.insert(0, str(CAND))
# ---------------------------------------------------------------------------------------

# Offline + clean tokenizer logs
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Ensure FAISS is present (ColBERT indexing requires it)
try:
    import faiss  # noqa: F401
except Exception as e:
    raise RuntimeError(
        "FAISS is required for ColBERT indexing. Install faiss-gpu (preferred) or faiss-cpu, then re-run."
    ) from e

import json
import argparse
from typing import List, Dict, Tuple
from tqdm import tqdm

# Official FLMR API
from flmr import index_custom_collection

# Your project config (adjust path if your loader lives elsewhere)
sys.path.append("/mnt/storage/RSystemsBenchmarking/gitProject")
from Benchmark.config.config_utils import load_config  # noqa: E402


# ---------------- helpers ---------------- #

def has_faiss_gpu() -> bool:
    """Return True if the FAISS build exposes GPU symbols."""
    return hasattr(faiss, "StandardGpuResources")

def load_mapping(p: Path) -> List[Dict]:
    return json.loads(p.read_text())

def resolve_image_path(base_dir: Path, rel_path: str) -> Path:
    """Try full relative path, then basename under base_dir."""
    p1 = (base_dir / rel_path).resolve()
    if p1.exists():
        return p1
    p2 = (base_dir / Path(rel_path).name).resolve()
    return p2 if p2.exists() else p1

def wrap_bok_eok(text: str) -> str:
    text = (text or "").strip()
    return f"<BOK> {text} <EOK>"

def make_caption_collection(mapping: List[Dict]) -> Tuple[list, list]:
    """
    Returns:
      collection = List[str] captions (text-only docs)
      meta       = List[dict] docId-aligned metadata
    """
    coll, meta = [], []
    for row in mapping:
        rel = row["image"]
        caps = [c if isinstance(c, str) else "" for c in row.get("caption", [])[:4]]
        for c in caps:
            coll.append(wrap_bok_eok(c))
            meta.append({"image_name": rel, "caption": c})
    return coll, meta

def make_joint_collection(mapping: List[Dict], base_img_dir: Path) -> Tuple[list, list]:
    """
    Returns:
      collection = List[(text, None, image_path)]
      meta       = List[dict]
    """
    coll, meta = [], []
    for row in mapping:
        rel = row["image"]
        img = resolve_image_path(base_img_dir, rel)
        if not img.exists():
            continue
        caps = [c if isinstance(c, str) else "" for c in row.get("caption", [])[:4]]
        for c in caps:
            coll.append((wrap_bok_eok(c), None, str(img)))
            meta.append({"image_name": rel, "caption": c})
    return coll, meta

def make_image_collection(mapping: List[Dict], base_img_dir: Path) -> Tuple[list, list]:
    """
    OFFICIAL-API APPROXIMATION of image-only:
      collection = List[(wrap_bok_eok(""), None, image_path)]
    """
    coll, meta = [], []
    for row in mapping:
        rel = row["image"]
        img = resolve_image_path(base_img_dir, rel)
        if not img.exists():
            continue
        coll.append((wrap_bok_eok(""), None, str(img)))
        meta.append({"image_name": rel, "caption": ""})
    return coll, meta

def dump_meta(index_root: Path, experiment: str, index_name: str, meta: list):
    outdir = index_root / experiment / index_name
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "collection_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"✓ wrote meta → {outdir/'collection_meta.json'}  (N={len(meta)})")


# ---------------- main ---------------- #

def main():
    ap = argparse.ArgumentParser("Build official PreFLMR ColBERT indices (caption, joint, image)")
    ap.add_argument("--dataset", choices=["coco","flickr"], default=None)
    ap.add_argument("--index-name-prefix", default="preflmr_vitl")   # suffixes _caption/_joint/_image will be added
    ap.add_argument("--nbits", type=int, default=8)
    ap.add_argument("--doc-maxlen", type=int, default=128)
    ap.add_argument("--indexing-batch-size", type=int, default=64)
    ap.add_argument("--nranks", type=int, default=1)                 # #GPUs for indexing
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--use-gpu", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    dataset = args.dataset or cfg["selected_config"]["dataset"]

    ds_cfg      = cfg["paths"]["dataset"][dataset]
    base_img    = Path(ds_cfg["base_image_path"])
    ingest_path = Path(ds_cfg["ingest_annotations_path"])

    flmr_cfg    = cfg["vector_store"]["flmr"]
    index_root  = Path(flmr_cfg["root_dir"])
    experiment  = flmr_cfg["experiment_name_map"][dataset]

    model_ckpt  = cfg["models"]["flmr"]["checkpoint"]      # local path to PreFLMR_ViT-*

    print(f"[info] dataset={dataset}  experiment={experiment}")
    mapping = load_mapping(ingest_path)
    print(f"[info] mapping rows={len(mapping)}")

    # Prepare three collections
    caption_collection, caption_meta = make_caption_collection(mapping)
    joint_collection,   joint_meta   = make_joint_collection(mapping, base_img)
    image_collection,   image_meta   = make_image_collection(mapping, base_img)

    # Decide GPU usage based on FAISS build
    local_use_gpu = bool(args.use_gpu and has_faiss_gpu())
    if args.use_gpu and not local_use_gpu:
        print("[warn] --use-gpu requested but FAISS GPU not available; falling back to CPU.")

    # CAPTION
    cap_index = f"{args.index_name_prefix}_caption"
    print(f"\n[CAPTION] Building → {cap_index}")
    _ = index_custom_collection(
        custom_collection=caption_collection,   # List[str]
        model=model_ckpt,
        index_root_path=str(index_root),
        index_experiment_name=experiment,
        index_name=cap_index,
        nbits=args.nbits,
        doc_maxlen=args.doc_maxlen,
        overwrite=args.overwrite,
        use_gpu=local_use_gpu,
        indexing_batch_size=args.indexing_batch_size,
        model_temp_folder="tmp",
        nranks=args.nranks,
    )
    dump_meta(index_root, experiment, cap_index, caption_meta)

    # JOINT
    joint_index = f"{args.index_name_prefix}_joint"
    print(f"\n[JOINT] Building → {joint_index}")
    _ = index_custom_collection(
        custom_collection=joint_collection,     # List[(text, None, image_path)]
        model=model_ckpt,
        index_root_path=str(index_root),
        index_experiment_name=experiment,
        index_name=joint_index,
        nbits=args.nbits,
        doc_maxlen=args.doc_maxlen,
        overwrite=args.overwrite,
        use_gpu=local_use_gpu,
        indexing_batch_size=args.indexing_batch_size,
        model_temp_folder="tmp",
        nranks=args.nranks,
    )
    dump_meta(index_root, experiment, joint_index, joint_meta)

    # IMAGE (approx)
    image_index = f"{args.index_name_prefix}_image"
    print(f"\n[IMAGE] Building (approx vision-only) → {image_index}")
    _ = index_custom_collection(
        custom_collection=image_collection,     # List[(wrap_bok_eok(""), None, image_path)]
        model=model_ckpt,
        index_root_path=str(index_root),
        index_experiment_name=experiment,
        index_name=image_index,
        nbits=args.nbits,
        doc_maxlen=args.doc_maxlen,
        overwrite=args.overwrite,
        use_gpu=local_use_gpu,
        indexing_batch_size=args.indexing_batch_size,
        model_temp_folder="tmp",
        nranks=args.nranks,
    )
    dump_meta(index_root, experiment, image_index, image_meta)

    print("\n✓ Done. Built indices:", cap_index, joint_index, image_index)


if __name__ == "__main__":
    main()
