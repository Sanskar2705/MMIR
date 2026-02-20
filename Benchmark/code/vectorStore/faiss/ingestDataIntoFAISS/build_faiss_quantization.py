#!/usr/bin/env python3
"""
Build FAISS SQ8 (Scalar Quantization 8-bit) inner-product indexes from JSON embeddings.

Inputs:
    <in-dir>/<model>_<dataset>_image_embeddings.json      (optional)
    <in-dir>/<model>_<dataset>_text_embeddings.json       (legacy, optional)
    <in-dir>/<model>_<dataset>_caption_embeddings.json    (new, optional)
    <in-dir>/<model>_<dataset>_joint-image-text_embeddings.json (new, optional)

Outputs (default: /mnt/storage/RSystemsBenchmarking/gitProject/Benchmark/data/faiss_index/quantization):
    <model>_<dataset>_faiss_img.index       (+ _img_names.json)
    <model>_<dataset>_faiss_txt.index       (+ _txt_meta.json)
    <model>_<dataset>_faiss_joint-image-text.index (+ _joint-image-text_meta.json)

This version builds SQ8 indexes (compressed, simple, great for 1K–5K vectors).
"""

import sys, json, argparse, warnings
from pathlib import Path

import faiss
import numpy as np

# project import
sys.path.append("/mnt/storage/RSystemsBenchmarking/gitProject")
from Benchmark.config.config_utils import load_config


# ---------- helpers ----------

def load_vecs_generic(path: Path):
    """Return (meta, vectors) from a standard embedding JSON list."""
    data = json.loads(path.read_text())
    meta = [{"image_name": o.get("image_name"), "caption": o.get("caption", "")} for o in data]
    vecs = np.asarray([o["embedding"] for o in data], dtype="float32")
    return meta, vecs

def load_image_names_only(path: Path):
    """Return (names, vectors) for image embeddings file."""
    data  = json.loads(path.read_text())
    names = [o["image_name"] for o in data]
    vecs  = np.asarray([o["embedding"] for o in data], dtype="float32")
    return names, vecs

def maybe_normalize(x: np.ndarray, normalize: bool):
    """
    If you want cosine similarity with inner-product search,
    normalize vectors to unit length.
    """
    if normalize:
        faiss.normalize_L2(x)
    return x

def build_sq8_index(x: np.ndarray, normalize: bool = True) -> faiss.Index:
    """
    Build a Scalar Quantization (8-bit) index for inner product search.
    - No training required
    - Good for small/medium datasets (1K–5K vectors)
    """
    if x.ndim != 2:
        raise ValueError("Embeddings must be a 2-D array.")
    d = int(x.shape[1])

    x = maybe_normalize(x, normalize)

    # SQ8 compresses vectors (1 byte per dimension)
    index = faiss.IndexScalarQuantizer(
        d,
        faiss.ScalarQuantizer.QT_8bit,
        faiss.METRIC_INNER_PRODUCT
    )
    # index.add(x)
    index.train(x)
    index.add(x)

    return index


# ---------- main ----------

def main():
    cfg = load_config()

    p = argparse.ArgumentParser(
        description="Build FAISS SQ8 indexes from JSON embeddings (image, caption/text, joint)."
    )
    p.add_argument("--dataset", "-d", default=None,
                   help="Dataset key (e.g. coco, flickr). Defaults to selected_config.dataset.")
    p.add_argument("--model", default="uniir",
                   help="Model tag in file names (e.g. flava, uniir, minilm).")
    p.add_argument("--in-dir", default=str(Path(cfg["paths"]["project_root"]) / "data" / "embeddings"),
                   help="Directory containing the JSON embeddings.")
    p.add_argument("--out-dir", default="/mnt/storage/RSystemsBenchmarking/gitProject/Benchmark/data/faiss_index/quantization",
                   help="Directory to save FAISS indexes + metadata.")
    p.add_argument("--no-normalize", action="store_true",
                   help="Disable L2 normalization. Keep enabled for cosine similarity with IP.")
    args = p.parse_args()

    dataset = args.dataset or cfg["selected_config"]["dataset"]
    in_dir  = Path(args.in_dir)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    normalize = not args.no_normalize

    # ----- input files -----
    img_json     = in_dir / f"{args.model}_{dataset}_image_embeddings.json"
    text_legacy  = in_dir / f"{args.model}_{dataset}_text_embeddings.json"       # old
    caption_json = in_dir / f"{args.model}_{dataset}_caption_embeddings.json"    # new
    joint_json   = in_dir / f"{args.model}_{dataset}_joint-image-text_embeddings.json"  # new

    have_img     = img_json.exists()
    have_txt     = text_legacy.exists()
    have_cap     = caption_json.exists()
    have_joint   = joint_json.exists()

    if not any([have_img, have_txt, have_cap, have_joint]):
        raise FileNotFoundError(
            f"No embeddings found for model '{args.model}' and dataset '{dataset}' in {in_dir}"
        )

    print(f"[SQ8 config] normalize={normalize}")
    print(f"Input dir : {in_dir}")
    print(f"Output dir: {out_dir}")

    # ===== IMAGE index =====
    if have_img:
        names_img, x_img = load_image_names_only(img_json)
        idx_img = build_sq8_index(x_img, normalize=normalize)

        img_idx_path = out_dir / f"{args.model}_{dataset}_faiss_img.index"
        faiss.write_index(idx_img, str(img_idx_path))

        (out_dir / f"{args.model}_{dataset}_img_names.json").write_text(
            json.dumps(names_img, indent=2)
        )

        print(f"✓ Built SQ8 IMAGE index → {img_idx_path}  "
              f"({len(names_img)} vecs, dim={x_img.shape[1]})")
    else:
        warnings.warn("No image embedding file found – skipping image index.")

    # ===== CAPTION/TEXT index (old 'text' or new 'caption') =====
    cap_src = None
    if have_cap:
        cap_src = caption_json
    elif have_txt:
        cap_src = text_legacy

    if cap_src:
        meta_cap, x_cap = load_vecs_generic(cap_src)
        idx_txt = build_sq8_index(x_cap, normalize=normalize)

        txt_idx_path = out_dir / f"{args.model}_{dataset}_faiss_txt.index"
        faiss.write_index(idx_txt, str(txt_idx_path))

        (out_dir / f"{args.model}_{dataset}_txt_meta.json").write_text(
            json.dumps(meta_cap, indent=2)
        )

        print(f"✓ Built SQ8 TEXT/CAPTION index → {txt_idx_path}  "
              f"({len(meta_cap)} vecs, dim={x_cap.shape[1]}) "
              f"[source: {cap_src.name}]")
    else:
        warnings.warn("No caption/text embedding file found – skipping caption index.")

    # ===== JOINT index =====
    if have_joint:
        meta_joint, x_joint = load_vecs_generic(joint_json)
        idx_joint = build_sq8_index(x_joint, normalize=normalize)

        joint_idx_path = out_dir / f"{args.model}_{dataset}_faiss_joint-image-text.index"
        faiss.write_index(idx_joint, str(joint_idx_path))

        (out_dir / f"{args.model}_{dataset}_joint-image-text_meta.json").write_text(
            json.dumps(meta_joint, indent=2)
        )

        print(f"✓ Built SQ8 JOINT index → {joint_idx_path}  "
              f"({len(meta_joint)} vecs, dim={x_joint.shape[1]})")
    else:
        warnings.warn("No joint embedding file found – skipping joint index.")

if __name__ == "__main__":
    main()
