#!/usr/bin/env python3
"""
Build FAISS inner-product indexes from JSON embeddings.

Backwards-compatible I/O:
- Inputs (will accept either legacy *text* or new *caption/joint* files):
    <in-dir>/<model>_<dataset>_image_embeddings.json      (optional)
    <in-dir>/<model>_<dataset>_text_embeddings.json       (legacy, optional)
    <in-dir>/<model>_<dataset>_caption_embeddings.json    (new, optional)
    <in-dir>/<model>_<dataset>_joint_embeddings.json      (new, optional)

- Outputs (default: cfg['vector_store']['faiss']['index_dir']):
    <model>_<dataset>_faiss_img.index       (+ _img_names.json)
    <model>_<dataset>_faiss_txt.index       (+ _txt_meta.json)   [built from CAPTION/TEXT]
    <model>_<dataset>_faiss_joint.index     (+ _joint_meta.json) [if joint available]
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

def build_ip_index(x: np.ndarray) -> faiss.Index:
    if x.ndim != 2:
        raise ValueError("Embeddings must be a 2-D array.")
    idx = faiss.IndexFlatIP(int(x.shape[1]))
    idx.add(x)
    return idx

# ---------- main ----------

def main():
    cfg = load_config()

    p = argparse.ArgumentParser(
        description="Build FAISS IP indexes from JSON embeddings (image, caption/text, joint)."
    )
    p.add_argument("--dataset", "-d", default=None,
                   help="Dataset key (e.g. coco, flickr). Defaults to selected_config.dataset.")
    p.add_argument("--model", default="uniir",
                   help="Model tag in file names (e.g. flava, uniir, minilm).")
    p.add_argument("--in-dir", default=str(Path(cfg["paths"]["project_root"]) / "data" / "embeddings"),
                   help="Directory containing the JSON embeddings.")
    p.add_argument("--out-dir", default=None,
                   help="Override output directory (defaults to faiss.index_dir in config).")
    args = p.parse_args()

    dataset = args.dataset or cfg["selected_config"]["dataset"]
    in_dir  = Path(args.in_dir)

    # Defaults to config value unless user overrides
    default_out = Path(cfg["vector_store"]["faiss"]["index_dir"])
    out_dir = Path(args.out_dir) if args.out_dir else default_out
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- input files (accept both legacy and new) -----
    img_json     = in_dir / f"{args.model}_{dataset}_image_embeddings.json"
    text_legacy  = in_dir / f"{args.model}_{dataset}_text_embeddings.json"       # old
    caption_json = in_dir / f"{args.model}_{dataset}_caption_embeddings.json"    # new
    joint_json   = in_dir / f"{args.model}_{dataset}_joint-image-text_embeddings.json"      # new

    have_img     = img_json.exists()
    have_txt     = text_legacy.exists()
    have_cap     = caption_json.exists()
    have_joint   = joint_json.exists()

    if not any([have_img, have_txt, have_cap, have_joint]):
        raise FileNotFoundError(
            f"No embeddings found for model '{args.model}' and dataset '{dataset}' in {in_dir}"
        )

    # ===== IMAGE index =====
    if have_img:
        names_img, x_img = load_image_names_only(img_json)
        idx_img = build_ip_index(x_img)
        img_idx_path = out_dir / f"{args.model}_{dataset}_faiss_img.index"
        faiss.write_index(idx_img, str(img_idx_path))
        (out_dir / f"{args.model}_{dataset}_img_names.json").write_text(
            json.dumps(names_img, indent=2)
        )
        print(f"✓ Built IMAGE index → {img_idx_path}  "
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
        idx_txt = build_ip_index(x_cap)
        txt_idx_path = out_dir / f"{args.model}_{dataset}_faiss_txt.index"
        faiss.write_index(idx_txt, str(txt_idx_path))
        (out_dir / f"{args.model}_{dataset}_txt_meta.json").write_text(
            json.dumps(meta_cap, indent=2)
        )
        print(f"✓ Built TEXT/CAPTION index → {txt_idx_path}  "
              f"({len(meta_cap)} vecs, dim={x_cap.shape[1]}) "
              f"[source: {cap_src.name}]")
    else:
        warnings.warn("No caption/text embedding file found – skipping caption index.")

    # ===== JOINT index (new) =====
    if have_joint:
        meta_joint, x_joint = load_vecs_generic(joint_json)
        idx_joint = build_ip_index(x_joint)
        joint_idx_path = out_dir / f"{args.model}_{dataset}_faiss_joint-image-text.index"
        faiss.write_index(idx_joint, str(joint_idx_path))
        (out_dir / f"{args.model}_{dataset}_joint-image-text_meta.json").write_text(
            json.dumps(meta_joint, indent=2)
        )
        print(f"✓ Built JOINT index → {joint_idx_path}  "
              f"({len(meta_joint)} vecs, dim={x_joint.shape[1]})")
    else:
        warnings.warn("No joint embedding file found – skipping joint index.")

if __name__ == "__main__":
    main()
