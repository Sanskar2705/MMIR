#!/usr/bin/env python3
"""
UNIIR exporter (candidate-side only)
────────────────────────────────────
Exports:
  1) image-only embeddings
  2) caption-only (no prompt) embeddings
  3) joint (image+text) embeddings

Variants:
  - CLIP-SF : score-level fusion  → w3·f_I(img) + w4·f_T(txt)
  - BLIP-FF : feature-level fusion→ BLIP image-grounded text encoder output

Usage examples:
  # CLIP-SF (existing behavior)
  python exporter.py -d coco --variant clip_sf --ckpt /path/to/clip_sf_large.pth

  # BLIP-FF (new)
  python exporter.py -d coco --variant blip_ff --ckpt /mnt/storage/ziv/COCO_UNI/checkpoint/BLIP_FF/blip_ff_large.pth
"""

# ─── stdlib ────────────────────────────────────────────────────────────
import os, sys, json, argparse, warnings
from pathlib import Path
from typing import List, Dict, Tuple

# ─── third-party ───────────────────────────────────────────────────────
import torch, numpy as np
from PIL import Image
from tqdm import tqdm

# CLIP deps stay as-is
import open_clip

# ─── project config ────────────────────────────────────────────────────
sys.path.append("/mnt/storage/RSystemsBenchmarking/gitProject")
from Benchmark.config.config_utils import load_config   # noqa: E402

os.environ["HF_HUB_DISABLE_XET"] = "1"

# ─── helpers ───────────────────────────────────────────────────────────
def l2norm(x: torch.Tensor) -> torch.Tensor:
    """L2-normalise last dim (matches CLIP/BLIP retrieval practice)."""
    return torch.nn.functional.normalize(x, dim=-1)

def resolve_image_path(base_dir: Path, rel_path: str) -> Path:
    """Try <base>/<rel> then <base>/<basename(rel)> (COCO quirks)."""
    p1 = (base_dir / rel_path).resolve()
    if p1.exists(): return p1
    p2 = (base_dir / Path(rel_path).name).resolve()
    return p2 if p2.exists() else p1

def batch(it, n):
    for i in range(0, len(it), n):
        yield it[i : i + n]

# ───────────────────────────────────────────────────────────────────────
# CLIP-SF (existing) loader + encoders
# ───────────────────────────────────────────────────────────────────────
def load_uniir_clip_sf(model_arch: str, ckpt_path: Path, device: str, fp16: bool = False):
    """Load OpenAI-CLIP backbone + UniIR-SF checkpoint."""
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_arch, pretrained="openai", device=device
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        ckpt = torch.load(ckpt_path, map_location="cpu")

    state = ckpt.get("model") or ckpt.get("state_dict") or {}
    state = {k.replace("clip_model.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)

    if fp16 and device == "cuda":
        model = model.half()
    model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(model_arch)
    return model, preprocess, tokenizer

def fuse_target_scores(img_emb: np.ndarray, txt_emb: np.ndarray, w3: float, w4: float) -> np.ndarray:
    """Score-level fusion (CLIP-SF): w3*image + w4*text."""
    return w3 * img_emb + w4 * txt_emb

def embed_images_clip(mapping: List[Dict], img_root: str, model, preprocess, device: str, bsz: int, fp16: bool) -> Dict[str, np.ndarray]:
    """Return dict { image_rel_path → f_I(img) }."""
    base = Path(img_root)
    cache: Dict[str, np.ndarray] = {}
    todo = [(it["image"], resolve_image_path(base, it["image"])) for it in mapping]

    for chunk in tqdm(list(batch(todo, bsz)), desc="[CLIP] encode image"):
        names, tensors = [], []
        for rel, p in chunk:
            if p.exists():
                names.append(rel)
                tensors.append(preprocess(Image.open(p).convert("RGB")))
        if not tensors:
            continue
        t = torch.stack(tensors).to(device)
        if fp16: t = t.half()
        with torch.no_grad():
            vecs = l2norm(model.encode_image(t)).cpu().numpy()
        cache.update({n: v for n, v in zip(names, vecs)})
    return cache

def embed_caption_and_joint_clip(
    mapping: List[Dict],
    model,
    tokenizer,
    device: str,
    img_cache: Dict[str, np.ndarray],
    bsz: int,
    w3: float,
    w4: float,
) -> Tuple[List[Dict], List[Dict]]:
    caption_res, joint_res = [], []
    pairs = [(it["image"], cap) for it in mapping for cap in it.get("caption", [])]

    for chunk in tqdm(list(batch(pairs, bsz)), desc="[CLIP] encode text(+joint)"):
        texts = [txt for _, txt in chunk]
        toks = tokenizer(texts).to(device)
        with torch.no_grad():
            txt_vecs = l2norm(model.encode_text(toks)).cpu().numpy()

        for (img_rel, raw_caption), e_txt in zip(chunk, txt_vecs):
            caption_res.append({"image_name": img_rel, "caption": raw_caption, "embedding": e_txt.tolist()})
            e_img = img_cache.get(img_rel)
            if e_img is None:  # skip broken images
                continue
            t_vec = fuse_target_scores(e_img, e_txt, w3, w4)
            joint_res.append({"image_name": img_rel, "caption": raw_caption, "embedding": t_vec.tolist()})

    return caption_res, joint_res

# ───────────────────────────────────────────────────────────────────────
# BLIP-FF (new) loader + encoders
#   - true feature-level fusion via BLIP image-grounded text encoder
# ───────────────────────────────────────────────────────────────────────
class BlipFFWrapper:
    """Thin wrapper over LAVIS BLIP feature extractor to expose encode_* APIs."""

    def __init__(self, model, vis_proc, txt_proc, device: str):
        self.model = model
        self.vis = vis_proc   # torchvision-like transform
        self.txt = txt_proc   # callable(string) -> normalized string
        self.device = device
        self.model.to(device).eval()

    def _images_to_tensor(self, imgs: List[Image.Image]) -> torch.Tensor:
        tensors = [self.vis(img) for img in imgs]  # each is CHW, already tensor
        return torch.stack(tensors).to(self.device)

    def encode_image(self, pixel_batch: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            feats = self.model.extract_features({"image": pixel_batch}, mode="image")
        # LAVIS returns a SimpleNamespace or dict with image_embeds [B, L, D]
        image_embeds = getattr(feats, "image_embeds", None) or feats["image_embeds"]
        pooled = image_embeds[:, 0, :]  # CLS
        return l2norm(pooled).cpu().numpy()

    def encode_text(self, texts: List[str]) -> np.ndarray:
        proc = [self.txt(t) for t in texts]
        with torch.no_grad():
            feats = self.model.extract_features({"text_input": proc}, mode="text")
        text_embeds = getattr(feats, "text_embeds", None) or feats["text_embeds"]
        pooled = text_embeds[:, 0, :]
        return l2norm(pooled).cpu().numpy()

    def encode_fused(self, pixel_batch: torch.Tensor, texts: List[str]) -> np.ndarray:
        proc = [self.txt(t) for t in texts]
        with torch.no_grad():
            feats = self.model.extract_features({"image": pixel_batch, "text_input": proc}, mode="multimodal")
        mm = getattr(feats, "multimodal_embeds", None) or feats["multimodal_embeds"]
        pooled = mm[:, 0, :]
        return l2norm(pooled).cpu().numpy()

def load_blip_ff(ckpt_path: Path, device: str):
    """
    Loads BLIP feature extractor from LAVIS and then (optionally) overlays your UniIR BLIP-FF checkpoint.
    Requires: pip install salesforce-lavis
    """
    try:
        from lavis.models import load_model_and_preprocess
    except Exception as e:
        raise RuntimeError(
            "BLIP-FF path requires 'salesforce-lavis'. Install via: pip install salesforce-lavis"
        ) from e

    # Large variant is what UniIR uses; this pulls default BLIP weights
    model, vis_processors, txt_processors = load_model_and_preprocess(
        name="blip_feature_extractor", model_type="large", is_eval=True, device=device
    )

    # Overlay your finetuned UniIR BLIP-FF weights (strict=False)
    if ckpt_path and ckpt_path.is_file():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("model") or ckpt.get("state_dict") or ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[BLIP-FF] note: {len(missing)} missing keys (ok): e.g. {missing[:3]}")
        if unexpected:
            print(f"[BLIP-FF] note: {len(unexpected)} unexpected keys (ok): e.g. {unexpected[:3]}")
    else:
        print(f"[BLIP-FF] WARNING: checkpoint not found at {ckpt_path}. Using base BLIP weights.")

    return BlipFFWrapper(model, vis_processors["eval"], txt_processors["eval"], device)

def embed_images_blip(mapping: List[Dict], img_root: str, blip: BlipFFWrapper, bsz: int) -> Dict[str, np.ndarray]:
    base = Path(img_root)
    cache: Dict[str, np.ndarray] = {}
    todo = [(it["image"], resolve_image_path(base, it["image"])) for it in mapping]

    for chunk in tqdm(list(batch(todo, bsz)), desc="[BLIP] encode image"):
        names, imgs = [], []
        for rel, p in chunk:
            if p.exists():
                names.append(rel)
                imgs.append(Image.open(p).convert("RGB"))
        if not imgs:
            continue
        pixels = blip._images_to_tensor(imgs)
        vecs = blip.encode_image(pixels)
        cache.update({n: v for n, v in zip(names, vecs)})
    return cache

def embed_caption_and_joint_blip(
    mapping: List[Dict],
    blip: BlipFFWrapper,
    img_root: str,
    bsz: int,
) -> Tuple[List[Dict], List[Dict]]:
    caption_res, joint_res = [], []
    base = Path(img_root)

    pairs = [(it["image"], cap) for it in mapping for cap in it.get("caption", [])]

    for chunk in tqdm(list(batch(pairs, bsz)), desc="[BLIP] encode text(+fused)"):
        # prepare images + texts
        rels = [rel for rel, _ in chunk]
        texts = [cap for _, cap in chunk]
        imgs = []
        for rel in rels:
            p = resolve_image_path(base, rel)
            if not p.exists():
                imgs.append(None)
            else:
                imgs.append(Image.open(p).convert("RGB"))

        # tokenize/encode text-only
        txt_vecs = blip.encode_text(texts)

        # fused (skip any missing image)
        valid_idx = [i for i, im in enumerate(imgs) if im is not None]
        if valid_idx:
            pixels = blip._images_to_tensor([imgs[i] for i in valid_idx])
            fused_vecs = blip.encode_fused(pixels, [texts[i] for i in valid_idx])
        else:
            fused_vecs = np.zeros((0, txt_vecs.shape[1]), dtype=np.float32)

        # emit results
        j = 0
        for i, (img_rel, raw_caption) in enumerate(chunk):
            # caption-only
            caption_res.append({"image_name": img_rel, "caption": raw_caption, "embedding": txt_vecs[i].tolist()})
            # joint (only if image exists)
            if i in valid_idx:
                joint_res.append({"image_name": img_rel, "caption": raw_caption, "embedding": fused_vecs[j].tolist()})
                j += 1

    return caption_res, joint_res

# ─── main ──────────────────────────────────────────────────────────────
def main():
    cfg = load_config()
    uni = (cfg.get("models") or {}).get("uniir", {})

    p = argparse.ArgumentParser("UNIIR exporter (CLIP-SF or BLIP-FF)")
    p.add_argument("--dataset", "-d", default=None)

    # Variant selection
    p.add_argument("--variant", choices=["clip_sf", "blip_ff"], default=uni.get("variant", "clip_sf"))

    # CLIP args
    p.add_argument("--arch", default=uni.get("arch_clip", "ViT-L-14-quickgelu"))
    p.add_argument("--ckpt_clip", default=uni.get("checkpoint_clip", "/mnt/storage/ziv/final_uniir/checkpoint/CLIP_SF/clip_sf_large.pth"))

    # BLIP args
    p.add_argument("--ckpt", default=uni.get("checkpoint_blip_ff", "/mnt/storage/ziv/COCO_UNI/checkpoint/BLIP_FF/blip_ff_large.pth"))

    # modality selection
    p.add_argument("--modality", choices=["all", "image", "text", "joint"], default="all",
                   help="Which embeddings to export: image, text (caption-only), joint, or all.")

    # batching / precision
    p.add_argument("--batch-img", type=int, default=32)
    p.add_argument("--batch-text", type=int, default=128)
    p.add_argument("--fp16", action="store_true", help="FP16 for CLIP only.")

    # score-level fusion weights (CLIP-SF only)
    p.add_argument("--w3", type=float, default=1.0, help="vision weight for targets (w₃) [CLIP-SF]")
    p.add_argument("--w4", type=float, default=1.0, help="text   weight for targets (w₄) [CLIP-SF]")

    # output suffix to avoid file overwrite when running both variants
    p.add_argument("--suffix", default=uni.get("out_suffix", ""), help="Optional filename suffix (e.g., _clipsf, _blipff).")

    args = p.parse_args()

    ds_key = args.dataset or cfg["selected_config"]["dataset"]
    ds_cfg = cfg["paths"]["dataset"][ds_key]
    mapping = json.loads(Path(ds_cfg["ingest_annotations_path"]).read_text())

    # outputs
    out_dir = Path(cfg["paths"]["project_root"]) / "data" / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)
    suf = args.suffix

    img_json   = out_dir / f"uniir_{ds_key}_image_embeddings{suf}.json"
    text_json  = out_dir / f"uniir_{ds_key}_text_embeddings{suf}.json"
    joint_json = out_dir / f"uniir_{ds_key}_joint-image-text_embeddings{suf}.json"

    # decide device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # gates
    need_image_cache = args.modality in ("all", "image", "joint")
    write_image_file = args.modality in ("all", "image")
    need_text_branch = args.modality in ("all", "text", "joint")
    write_text_file  = args.modality in ("all", "text")
    write_joint_file = args.modality in ("all", "joint")

    # ── CLIP-SF path ───────────────────────────────────────────────────
    if args.variant == "clip_sf":
        model, preprocess, tokenizer = load_uniir_clip_sf(args.arch, Path(args.ckpt_clip), device, args.fp16)

        img_cache = {}
        if need_image_cache:
            img_cache = embed_images_clip(mapping, ds_cfg["base_image_path"], model, preprocess, device, args.batch_img, args.fp16)
            if write_image_file:
                img_json.write_text(json.dumps(
                    [{"image_name": k, "embedding": v.tolist()} for k, v in img_cache.items()], indent=2))
                print(f"✓ wrote {len(img_cache):,} image vectors  → {img_json}")
            else:
                print(f"✓ computed {len(img_cache):,} image vectors (not written; used for joint)")

        if need_text_branch:
            local_img_cache = img_cache if args.modality in ("all", "joint") else {}
            caption_res, joint_res = embed_caption_and_joint_clip(
                mapping=mapping,
                model=model,
                tokenizer=tokenizer,
                device=device,
                img_cache=local_img_cache,
                bsz=args.batch_text,
                w3=args.w3,
                w4=args.w4,
            )

            if write_text_file:
                text_json.write_text(json.dumps(caption_res, indent=2))
                print(f"✓ wrote {len(caption_res):,} caption vectors → {text_json}")
            else:
                print(f"✓ computed {len(caption_res):,} caption vectors (not written)")

            if write_joint_file:
                joint_json.write_text(json.dumps(joint_res, indent=2))
                print(f"✓ wrote {len(joint_res):,} joint vectors   → {joint_json}")
                if joint_res:
                    print(f"  vector dim = {len(joint_res[0]['embedding'])}")

    # ── BLIP-FF path ───────────────────────────────────────────────────
    else:
        blip = load_blip_ff(Path(args.ckpt), device)

        if need_image_cache:
            img_cache = embed_images_blip(mapping, ds_cfg["base_image_path"], blip, args.batch_img)
            if write_image_file:
                img_json.write_text(json.dumps(
                    [{"image_name": k, "embedding": v.tolist()} for k, v in img_cache.items()], indent=2))
                print(f"✓ wrote {len(img_cache):,} image vectors  → {img_json}")
            else:
                print(f"✓ computed {len(img_cache):,} image vectors (not written)")

        if need_text_branch:
            caption_res, joint_res = embed_caption_and_joint_blip(
                mapping=mapping,
                blip=blip,
                img_root=ds_cfg["base_image_path"],
                bsz=args.batch_text,
            )

            if write_text_file:
                text_json.write_text(json.dumps(caption_res, indent=2))
                print(f"✓ wrote {len(caption_res):,} caption vectors → {text_json}")
            else:
                print(f"✓ computed {len(caption_res):,} caption vectors (not written)")

            if write_joint_file:
                joint_json.write_text(json.dumps(joint_res, indent=2))
                print(f"✓ wrote {len(joint_res):,} joint vectors   → {joint_json}")
                if joint_res:
                    print(f"  vector dim = {len(joint_res[0]['embedding'])}")

if __name__ == "__main__":
    main()
