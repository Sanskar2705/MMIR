"""
Generate OpenCLIP (ViT-L/14) with LAION2B weights for MSCOCO and Flickr30k datasets.

Usage:
    # Text embeddings for COCO
    python code/embeddingGeneration/openCLIP_model.py --dataset coco --modality text

    # Image embeddings for Flickr30k
    python code/embeddingGeneration/openCLIP_model.py --dataset flickr --modality image

    # Both modalities
    python code/embeddingGeneration/openCLIP_model.py --dataset coco --modality both --batch-img 32 --batch-text 128
"""

#!/usr/bin/env python3
import os
import sys
import json
import torch
from pathlib import Path
from typing import List, Dict, Tuple
from PIL import Image
from tqdm import tqdm
import open_clip
import argparse

# Project imports
sys.path.append("/mnt/storage/RSystemsBenchmarking/gitProject")
from Benchmark.config.config_utils import load_config


def l2norm(x: torch.Tensor) -> torch.Tensor:
    """L2 normalization of embeddings."""
    return x / x.norm(dim=-1, keepdim=True)


def load_openclip_vitl14_laion2b(device: str):
    """
    Load OpenCLIP ViT-L/14 model trained on LAION2B.
    Using weights from the open-source OpenCLIP training (not OpenAI's proprietary training).
    """
    print(f"Loading OpenCLIP ViT-L/14 (LAION2B trained) model on {device}")
    model, _, transform = open_clip.create_model_and_transforms(
        "ViT-L-14",
        pretrained="laion2b_s32b_b82k",  # LAION2B trained weights
        device=device
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    return model, transform, tokenizer, device


def resolve_image_path(base_dir: Path, rel_path: str) -> Path:
    """Resolve image path with fallback to filename."""
    p1 = (base_dir / rel_path).resolve()
    if p1.exists():
        return p1
    p2 = (base_dir / Path(rel_path).name).resolve()
    if p2.exists():
        return p2
    return p1


def batch(iterable, n):
    """Batch an iterable into chunks of size n."""
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]


def generate_image_embeddings(mapping: List[Dict],
                              base_image_path: str,
                              model,
                              transform,
                              device: str,
                              batch_size: int = 32) -> List[Dict]:
    """
    Generate OpenCLIP image embeddings in batches.
    """
    base_dir = Path(base_image_path)
    results = []
    entries: List[Tuple[str, Path]] = []

    for it in mapping:
        rel = it["image"]
        pth = resolve_image_path(base_dir, rel)
        entries.append((rel, pth))

    for chunk in tqdm(list(batch(entries, batch_size)), desc="OpenCLIP image embeddings"):
        imgs, rel_names = [], []
        for rel, pth in chunk:
            if not pth.exists():
                print(f"[warn] missing image: {rel} (looked for: {pth})")
                continue
            try:
                imgs.append(Image.open(pth).convert("RGB"))
                rel_names.append(rel)
            except Exception as e:
                print(f"[warn] failed to open {pth}: {e}")

        if not imgs:
            continue

        # Preprocess images
        img_tensors = torch.stack([transform(img) for img in imgs]).to(device)

        with torch.no_grad():
            img_features = model.encode_image(img_tensors)
            img_features = l2norm(img_features).cpu().tolist()

        for name, emb in zip(rel_names, img_features):
            results.append({"image_name": name, "embedding": emb})

    return results


def generate_text_embeddings(mapping: List[Dict],
                             model,
                             tokenizer,
                             device: str,
                             batch_size: int = 128) -> List[Dict]:
    """
    Generate OpenCLIP text embeddings in batches.
    """
    # Create (image_name, caption) pairs
    pairs: List[Tuple[str, str]] = []
    for it in mapping:
        img = it["image"]
        for c in it.get("caption", []):
            pairs.append((img, c if isinstance(c, str) else ""))

    results = []
    for chunk in tqdm(list(batch(pairs, batch_size)), desc="OpenCLIP text embeddings"):
        captions = [c for _, c in chunk]

        # Tokenize captions
        text_tokens = tokenizer(captions).to(device)

        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
            text_features = l2norm(text_features).cpu().tolist()

        for (img_name, cap), emb in zip(chunk, text_features):
            results.append({
                "image_name": img_name,
                "caption": cap,
                "embedding": emb
            })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate OpenCLIP (ViT-L/14 LAION2B) embeddings for MSCOCO and Flickr30k."
    )
    parser.add_argument(
        "--dataset", "-d",
        default=None,
        help="Dataset key in config (e.g., coco, flickr). Defaults to selected_config.dataset."
    )
    parser.add_argument(
        "--modality", "-m",
        default="both",
        choices=["image", "text", "both"],
        help="Which embeddings to generate."
    )
    parser.add_argument(
        "--batch-img",
        type=int,
        default=32,
        help="Batch size for image embedding."
    )
    parser.add_argument(
        "--batch-text",
        type=int,
        default=128,
        help="Batch size for text embedding."
    )
    args = parser.parse_args()

    # Load config
    config = load_config()
    dataset = args.dataset or config["selected_config"]["dataset"]

    ds_cfg = config["paths"]["dataset"][dataset]
    base_image_path = ds_cfg["base_image_path"]
    mapping_file = ds_cfg["ingest_annotations_path"]

    # Create output directory
    out_dir = Path(config["paths"]["project_root"]) / "data" / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)

    img_out_path = out_dir / f"openclip_{dataset}_image_embeddings.json"
    txt_out_path = out_dir / f"openclip_{dataset}_text_embeddings.json"

    # Load mapping/annotations
    with open(mapping_file, 'r') as f:
        mapping = json.load(f)

    print(f"\nProcessing dataset: {dataset}")
    print(f"Found {len(mapping)} items in mapping")

    # Initialize model with LAION2B weights
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, transform, tokenizer, device = load_openclip_vitl14_laion2b(device)

    # Generate image embeddings
    if args.modality in ("image", "both"):
        print("\n" + "=" * 50)
        print("Generating image embeddings...")
        print("=" * 50)
        img_results = generate_image_embeddings(
            mapping,
            base_image_path,
            model,
            transform,
            device,
            args.batch_img
        )
        with open(img_out_path, 'w') as f:
            json.dump(img_results, f, indent=2)
        if img_results:
            print(f"✓ Wrote {len(img_results)} image embeddings → {img_out_path}")
            print(f"  Embedding dimension: {len(img_results[0]['embedding'])}")

    # Generate text embeddings
    if args.modality in ("text", "both"):
        print("\n" + "=" * 50)
        print("Generating text embeddings...")
        print("=" * 50)
        txt_results = generate_text_embeddings(
            mapping,
            model,
            tokenizer,
            device,
            args.batch_text
        )
        with open(txt_out_path, 'w') as f:
            json.dump(txt_results, f, indent=2)
        if txt_results:
            print(f"✓ Wrote {len(txt_results)} text embeddings → {txt_out_path}")
            print(f"  Embedding dimension: {len(txt_results[0]['embedding'])}")

    print("\n" + "=" * 50)
    print("All done!")
    print("=" * 50)


if __name__ == "__main__":
    main()