"""

# Both image + text
python code/embeddingGeneration/flava_embeddings.py \
  --dataset flickr --modality both --batch-img 32 --batch-text 128

python code/embeddingGeneration/flava_embeddings.py \
  --dataset coco --modality both --batch-img 32 --batch-text 128

"""

#!/usr/bin/env python3
import os, sys, json, torch
from pathlib import Path
from typing import List, Dict, Tuple
from PIL import Image
from tqdm import tqdm
from transformers import FlavaProcessor, FlavaModel
import argparse

# project import
sys.path.append("/mnt/storage/RSystemsBenchmarking/gitProject")
from Benchmark.config.config_utils import load_config

os.environ["HF_HUB_DISABLE_XET"] = "1"

# ───────────────────────── helpers ───────────────────────── #

def l2norm(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True)

def resolve_image_path(base_dir: Path, rel_path: str) -> Path:
    p1 = (base_dir / rel_path).resolve()
    if p1.exists():
        return p1
    p2 = (base_dir / Path(rel_path).name).resolve()
    if p2.exists():
        return p2
    return p1  # fallback

def load_flava(device: str):
    processor = FlavaProcessor.from_pretrained("facebook/flava-full")
    model = FlavaModel.from_pretrained("facebook/flava-full").to(device).eval()
    return processor, model

def batch(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]

# ───────────────────── embedding generators ─────────────────── #

def generate_image_embeddings(mapping: List[Dict], base_image_path: str,
                              processor, model, device: str,
                              batch_size: int) -> List[Dict]:
    base_dir = Path(base_image_path)
    results = []
    entries: List[Tuple[str, Path]] = []
    for it in mapping:
        rel = it["image"]
        pth = resolve_image_path(base_dir, rel)
        entries.append((rel, pth))

    for chunk in tqdm(list(batch(entries, batch_size)), desc="FLAVA image embeddings"):
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

        inputs = processor(images=imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            vec = l2norm(model.get_image_features(**inputs)[:, 0]).cpu().tolist()

        for name, emb in zip(rel_names, vec):
            results.append({"image_name": name, "embedding": emb})
    return results

def generate_text_embeddings(mapping: List[Dict],
                             processor, model, device: str,
                             batch_size: int) -> List[Dict]:
    pairs: List[Tuple[str, str]] = []
    for it in mapping:
        img = it["image"]
        for c in it.get("caption", []):
            pairs.append((img, c if isinstance(c, str) else ""))

    results = []
    for chunk in tqdm(list(batch(pairs, batch_size)), desc="FLAVA text embeddings"):
        caps = [c for _, c in chunk]
        inputs = processor(
            text=caps,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        ).to(device)
        with torch.no_grad():
            vec = l2norm(model.get_text_features(**inputs)[:, 0]).cpu().tolist()

        for (img_name, cap), emb in zip(chunk, vec):
            results.append({"image_name": img_name, "caption": cap, "embedding": emb})
    return results

# ─────────────────────────── main ─────────────────────────── #

def main():
    parser = argparse.ArgumentParser(description="Generate FLAVA embeddings in required JSON format.")
    parser.add_argument("--dataset", "-d", default=None,
                        help="Dataset key in config (e.g., coco, flickr). Defaults to selected_config.dataset.")
    parser.add_argument("--modality", "-m", default="both",
                        choices=["image", "text", "both"],
                        help="Which embeddings to generate.")
    parser.add_argument("--batch-img", type=int, default=32,
                        help="Batch size for image embedding.")
    parser.add_argument("--batch-text", type=int, default=128,
                        help="Batch size for text embedding.")
    args = parser.parse_args()

    config = load_config()
    dataset = args.dataset or config["selected_config"]["dataset"]

    ds_cfg = config["paths"]["dataset"][dataset]
    base_image_path = ds_cfg["base_image_path"]
    mapping_file = ds_cfg["ingest_annotations_path"]

    out_dir = Path(config["paths"]["project_root"])/"data"/"embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)

    img_out_path = out_dir / f"flava_{dataset}_image_embeddings.json"
    txt_out_path = out_dir / f"flava_{dataset}_text_embeddings.json"

    mapping = json.loads(Path(mapping_file).read_text())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, model = load_flava(device)

    if args.modality in ("image", "both"):
        img_results = generate_image_embeddings(
            mapping, base_image_path, processor, model, device, args.batch_img
        )
        img_out_path.write_text(json.dumps(img_results, indent=2))
        if img_results:
            print(f"✓ Wrote {len(img_results)} image embeddings → {img_out_path}")
            print(f"  dim = {len(img_results[0]['embedding'])}")

    if args.modality in ("text", "both"):
        txt_results = generate_text_embeddings(
            mapping, processor, model, device, args.batch_text
        )
        txt_out_path.write_text(json.dumps(txt_results, indent=2))
        if txt_results:
            print(f"✓ Wrote {len(txt_results)} text embeddings  → {txt_out_path}")
            print(f"  dim = {len(txt_results[0]['embedding'])}")

if __name__ == "__main__":
    main()
