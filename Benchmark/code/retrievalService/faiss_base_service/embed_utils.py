#!/usr/bin/env python3
"""
embed_utils.py
──────────────
Central registry for all text-embedding models used by the retrieval server.
"""

from __future__ import annotations

import warnings, torch, numpy as np
from pathlib import Path
from typing import Callable, Dict, Union

def l2norm(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x, dim=-1)

def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────── MODEL LOADERS ─────────────────── #

def _load_flava(model_cfg: str | Dict, dev: str) -> Callable[[str], np.ndarray]:
    from transformers import FlavaProcessor, FlavaModel
    proc  = FlavaProcessor.from_pretrained(model_cfg)
    model = FlavaModel.from_pretrained(model_cfg).to(dev).eval()

    def _embed(text: str) -> np.ndarray:
        toks = proc(text=[text], return_tensors="pt", padding=True,
                    truncation=True, max_length=77).to(dev)
        with torch.no_grad():
            vec = l2norm(model.get_text_features(**toks)[:, 0]).cpu().numpy()
        return vec.astype("float32")[0]
    return _embed

def _load_uniir(model_cfg: Dict, dev: str) -> Callable[[str], np.ndarray]:
    """
    model_cfg:
      arch: "ViT-L-14-quickgelu"
      checkpoint: "/path/clip_sf_large.pth"
      prompt: ""  # optional prompt prefix
    """
    import open_clip
    arch      = model_cfg["arch"]
    ckpt_path = Path(model_cfg["checkpoint"])
    prompt    = model_cfg.get("prompt", "").strip()

    #model, _, _ = open_clip.create_model_and_transforms(arch, pretrained="openai", device=dev)
    model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=None, device=dev)


    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        import omegaconf
        import torch.serialization

        torch.serialization.add_safe_globals([omegaconf.dictconfig.DictConfig])

        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = state.get("model") or state.get("state_dict") or state
    state = {k.replace("clip_model.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    tok = open_clip.get_tokenizer(arch)
    model.to(dev).eval()

    def _embed(text: str) -> np.ndarray:
        # instruction on query side only
        full = f"{prompt} {text}".strip() if prompt else text
        t = tok([full]).to(dev)
        with torch.no_grad():
            vec = l2norm(model.encode_text(t)).cpu().numpy()
        return vec.astype("float32")[0]
    return _embed

def _load_minilm(model_cfg: str | Dict, dev: str) -> Callable[[str], np.ndarray]:
    from sentence_transformers import SentenceTransformer
    mdl_path = model_cfg if isinstance(model_cfg, str) else model_cfg["model_name"]
    model = SentenceTransformer(mdl_path, device=dev)
    def _embed(text: str) -> np.ndarray:
        vec = model.encode(text, device=dev, convert_to_numpy=True, normalize_embeddings=True)
        return vec.astype("float32")
    return _embed

def _load_clip(model_cfg: str | Dict, dev: str) -> Callable[[str], np.ndarray]:
    import importlib
    clip = importlib.import_module("clip")
    model_variant = (model_cfg if isinstance(model_cfg, str)
                     else model_cfg.get("model_name") or model_cfg.get("arch")) or "ViT-L/14@336px"
    device = dev or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = clip.load(model_variant, device=device)
    model.eval()
    def _embed(text: str) -> np.ndarray:
        tokens = clip.tokenize([text], truncate=True).to(device)
        with torch.no_grad():
            vec = l2norm(model.encode_text(tokens)).cpu().numpy()[0]
        return vec.astype("float32")
    return _embed

MODEL_LOADERS: Dict[str, Callable[[Union[str, Dict], str], Callable]] = {
    "flava" : _load_flava,
    "uniir" : _load_uniir,
    "minilm": _load_minilm,
    "clip"  : _load_clip,
}

def get_embedder(model_tag: str, full_cfg: Dict, dev: str | None = None) -> Callable[[str], np.ndarray]:
    if dev is None:
        dev = _device()
    if model_tag not in MODEL_LOADERS:
        raise ValueError(f"[embed_utils] Unknown model tag '{model_tag}'.")
    model_cfg = full_cfg["models"][model_tag]
    return MODEL_LOADERS[model_tag](model_cfg, dev)
