#!/usr/bin/env python3
"""
UNIIR CLIP-SF exporter (candidate-side only)
────────────────────────────────────────────
Creates three embedding files per dataset:

1) image-only                  →  uniir_<ds>_image_embeddings.json
2) caption-only (no prompt)    →  uniir_<ds>_caption_embeddings.json
3) joint (image + caption)     →  uniir_<ds>_joint_embeddings.json

Fusion (paper Eq. 1/2, candidate side):
    target_vec = w3·f_I(c_img) + w4·f_T(c_txt)

Queries (instruction + held-out caption) are built in eval, not here.
"""

# ─── stdlib ────────────────────────────────────────────────────────────
import os, sys, json, argparse, warnings
from pathlib import Path
from typing import List, Dict, Tuple

# ─── third-party ───────────────────────────────────────────────────────
import torch, numpy as np
from PIL import Image
from tqdm import tqdm
import open_clip

# ─── project config ────────────────────────────────────────────────────
sys.path.append("/mnt/storage/RSystemsBenchmarking/gitProject")
from Benchmark.config.config_utils import load_config   # noqa: E402

os.environ["HF_HUB_DISABLE_XET"] = "1"

# ─── helpers ───────────────────────────────────────────────────────────
def l2norm(x: torch.Tensor) -> torch.Tensor:
    """L2-normalise last dim (matches CLIP training)."""
    return torch.nn.functional.normalize(x, dim=-1)

def resolve_image_path(base_dir: Path, rel_path: str) -> Path:
    """Try <base>/<rel> then <base>/<basename(rel)> (COCO quirks)."""
    p1 = (base_dir / rel_path).resolve()
    if p1.exists(): return p1
    p2 = (base_dir / Path(rel_path).name).resolve()
    return p2 if p2.exists() else p1

def load_uniir(model_arch: str, ckpt_path: Path, device: str, fp16: bool = False):
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

def batch(it, n):
    for i in range(0, len(it), n):
        yield it[i : i + n]

# ─── fusion helper (paper-exact, candidate side) ───────────────────────
def fuse_target(img_emb: np.ndarray, txt_emb: np.ndarray, w3: float, w4: float) -> np.ndarray:
    return w3 * img_emb + w4 * txt_emb

# ─── embedding generators ─────────────────────────────────────────────
def embed_images(
    mapping: List[Dict],
    img_root: str,
    model,
    preprocess,
    device: str,
    bsz: int,
    fp16: bool,
) -> Dict[str, np.ndarray]:
    """Return dict  { image_rel_path → f_I(img) } (L2-normalised)."""
    base = Path(img_root)
    cache: Dict[str, np.ndarray] = {}
    todo = [(it["image"], resolve_image_path(base, it["image"])) for it in mapping]

    for chunk in tqdm(list(batch(todo, bsz)), desc="encode image"):
        names, tensors = [], []
        for rel, p in chunk:
            if p.exists():
                names.append(rel)
                tensors.append(preprocess(Image.open(p).convert("RGB")))
        if not tensors:
            continue
        t = torch.stack(tensors).to(device)
        if fp16:
            t = t.half()
        with torch.no_grad():
            vecs = l2norm(model.encode_image(t)).cpu().numpy()
        cache.update({n: v for n, v in zip(names, vecs)})
    return cache


def embed_caption_and_joint(
    mapping: List[Dict],
    model,
    tokenizer,
    device: str,
    img_cache: Dict[str, np.ndarray],
    bsz: int,
    w3: float,
    w4: float,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Returns:
        caption_res – candidate caption-only embeddings  (f_T(caption))
        joint_res   – candidate joint embeddings         (w3*f_I(img)+w4*f_T(caption))
    """
    caption_res, joint_res = [], []

    # NOTE: no prompt here — candidate captions are raw (paper-consistent)
    pairs = [(it["image"], cap) for it in mapping for cap in it.get("caption", [])]

    for chunk in tqdm(list(batch(pairs, bsz)), desc="encode caption(+joint)"):
        texts = [txt for _, txt in chunk]
        toks = tokenizer(texts).to(device)
        with torch.no_grad():
            txt_vecs = l2norm(model.encode_text(toks)).cpu().numpy()  # f_T(caption)

        for (img_rel, raw_caption), e_txt in zip(chunk, txt_vecs):
            # 1) candidate caption-only vector
            caption_res.append(
                {"image_name": img_rel, "caption": raw_caption, "embedding": e_txt.tolist()}
            )

            # 2) candidate joint vector = image + caption (score-level fusion)
            e_img = img_cache.get(img_rel)
            if e_img is None:
                # Option A: skip broken image
                # Option B: use zeros_like(e_txt) to keep text-only candidate
                # Here we skip to avoid indexing invalid items
                continue
            t_vec = fuse_target(e_img, e_txt, w3, w4)
            joint_res.append(
                {"image_name": img_rel, "caption": raw_caption, "embedding": t_vec.tolist()}
            )

    return caption_res, joint_res


# ─── main ──────────────────────────────────────────────────────────────
def main():
    cfg = load_config()
    uni = (cfg.get("models") or {}).get("uniir", {})

    p = argparse.ArgumentParser("UNIIR CLIP-SF (image / caption / joint) exporter")
    p.add_argument("--dataset", "-d", default=None)
    p.add_argument("--arch", default=uni.get("arch", "ViT-L-14-quickgelu"))
    p.add_argument(
        "--ckpt",
        default=uni.get(
            "checkpoint",
            "/mnt/storage/ziv/final_uniir/checkpoint/CLIP_SF/clip_sf_large.pth",
        ),
    )

    # NEW: modality selection (use 'text' to mean caption-only)
    p.add_argument(
        "--modality",
        choices=["all", "image", "text", "joint"],
        default="all",
        help="Which embeddings to export: image, text (caption-only), joint, or all.",
    )

    # batching / precision
    p.add_argument("--batch-img", type=int, default=32)
    p.add_argument("--batch-text", type=int, default=128)
    p.add_argument("--fp16", action="store_true")

    # score-level fusion weights (candidate-side only)
    p.add_argument("--w3", type=float, default=1.0, help="vision weight for targets (w₃)")
    p.add_argument("--w4", type=float, default=1.0, help="text   weight for targets (w₄)")
    args = p.parse_args()

    ds_key = args.dataset or cfg["selected_config"]["dataset"]
    ds_cfg = cfg["paths"]["dataset"][ds_key]
    mapping = json.loads(Path(ds_cfg["ingest_annotations_path"]).read_text())

    # output dirs
    out_dir = Path(cfg["paths"]["project_root"]) / "data" / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_json   = out_dir / f"uniir_{ds_key}_image_embeddings.json"
    text_json  = out_dir / f"uniir_{ds_key}_text_embeddings.json"
    joint_json = out_dir / f"uniir_{ds_key}_joint-image-text_embeddings.json"

    # keep your later reference name intact
    caption_json = text_json

    # load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess, tokenizer = load_uniir(args.arch, Path(args.ckpt), device, args.fp16)

    # ── modality gating ────────────────────────────────────────────────
    need_image_cache = args.modality in ("all", "image", "joint")
    write_image_file = args.modality in ("all", "image")
    need_text_branch = args.modality in ("all", "text", "joint")
    write_text_file  = args.modality in ("all", "text")
    write_joint_file = args.modality in ("all", "joint")

    img_cache = {}
    if need_image_cache:
        # 1) image branch (f_I)
        img_cache = embed_images(
            mapping,
            ds_cfg["base_image_path"],
            model,
            preprocess,
            device,
            args.batch_img,
            args.fp16,
        )
        if write_image_file:
            img_json.write_text(
                json.dumps(
                    [{"image_name": k, "embedding": v.tolist()} for k, v in img_cache.items()],
                    indent=2,
                )
            )
            print(f"✓ wrote {len(img_cache):,} image vectors  → {img_json}")
        else:
            print(f"✓ computed {len(img_cache):,} image vectors (not written; used for joint)")

    if need_text_branch:
        # 2) caption-only (candidates) & joint (candidates)
        #    For pure 'text' mode, we can pass an empty img_cache to skip joint creation.
        local_img_cache = img_cache if args.modality in ("all", "joint") else {}
        caption_res, joint_res = embed_caption_and_joint(
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
            caption_json.write_text(json.dumps(caption_res, indent=2))
            print(f"✓ wrote {len(caption_res):,} caption vectors → {caption_json}")
        else:
            print(f"✓ computed {len(caption_res):,} caption vectors (not written)")

        if write_joint_file:
            joint_json.write_text(json.dumps(joint_res, indent=2))
            print(f"✓ wrote {len(joint_res):,} joint vectors   → {joint_json}")
            if joint_res:
                print(f"  vector dim = {len(joint_res[0]['embedding'])}")
        else:
            if args.modality in ("all", "text"):
                # joint not requested; ignore
                pass
            elif args.modality == "joint":
                # Already handled above by write_joint_file True
                pass

if __name__ == "__main__":
    main()
