"""GPU-batched query encoders for CLIP, OpenCLIP, and UniIR."""

from __future__ import annotations

import sys
import warnings
from typing import List, Sequence

import numpy as np
import torch

sys.path.append("/mnt/storage/RSystemsBenchmarking/gitProject")
from Benchmark.config.config_utils import load_config


def _l2norm_np(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (x / norms).astype("float32")


class BatchQueryEncoder:
    """Encode caption batches on GPU (one model load per encoder)."""

    def __init__(self, encoder_name: str, device: str | None = None):
        self.encoder_name = encoder_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._load()

    def _load(self) -> None:
        cfg = load_config()
        name = self.encoder_name

        if name == "clip_image":
            import clip

            model_name = cfg["models"]["clip"]
            self.model, _ = clip.load(model_name, device=self.device)
            self.model.eval()
            self._encode_batch = self._encode_clip
            return

        if name == "openclip_image":
            import open_clip

            oc = cfg["models"]["openclip"]
            arch = oc.get("arch", "ViT-L-14")
            pretrained = oc.get("pretrained", "laion2b_s32b_b82k")
            self.model, _, _ = open_clip.create_model_and_transforms(
                arch, pretrained=pretrained, device=self.device
            )
            self.tokenizer = open_clip.get_tokenizer(arch)
            self.model.eval()
            self._encode_batch = self._encode_openclip
            return

        if name == "uniir_joint":
            import open_clip
            from pathlib import Path

            uniir = cfg["models"]["uniir"]
            arch = uniir["arch"]
            ckpt = Path(uniir["checkpoint"])
            self.model, _, _ = open_clip.create_model_and_transforms(
                arch, pretrained=None, device=self.device
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                state = torch.load(ckpt, map_location="cpu", weights_only=False)
            state = state.get("model") or state.get("state_dict") or state
            state = {k.replace("clip_model.", "", 1): v for k, v in state.items()}
            self.model.load_state_dict(state, strict=False)
            self.tokenizer = open_clip.get_tokenizer(arch)
            self.model.eval()
            self._encode_batch = self._encode_uniir
            return

        raise ValueError(f"Unknown encoder: {name}")

    def _encode_clip(self, captions: Sequence[str]) -> np.ndarray:
        import clip

        with torch.no_grad():
            tokens = clip.tokenize(list(captions), truncate=True).to(self.device)
            emb = self.model.encode_text(tokens).float()
            emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb.cpu().numpy().astype("float32")

    def _encode_openclip(self, captions: Sequence[str]) -> np.ndarray:
        with torch.no_grad():
            tokens = self.tokenizer(list(captions)).to(self.device)
            emb = self.model.encode_text(tokens).float()
            emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb.cpu().numpy().astype("float32")

    def _encode_uniir(self, captions: Sequence[str]) -> np.ndarray:
        with torch.no_grad():
            tokens = self.tokenizer(list(captions)).to(self.device)
            emb = self.model.encode_text(tokens).float()
            emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb.cpu().numpy().astype("float32")

    def encode_all(self, captions: Sequence[str], batch_size: int = 128) -> np.ndarray:
        chunks: List[np.ndarray] = []
        for start in range(0, len(captions), batch_size):
            batch = captions[start : start + batch_size]
            chunks.append(self._encode_batch(batch))
        return _l2norm_np(np.vstack(chunks))


def get_batch_encoder(encoder_name: str, device: str | None = None) -> BatchQueryEncoder:
    return BatchQueryEncoder(encoder_name, device)
