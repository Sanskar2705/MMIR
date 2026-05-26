"""Load BLIP2-ITM in-process (minimal LAVIS import, no FastAPI)."""

from __future__ import annotations

import os
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import importlib.util
import logging
import os
import sys
import types
from pathlib import Path
from typing import Any, Tuple

LAVIS_ROOT = os.environ.get(
    "LAVIS_ROOT",
    "/mnt/storage/bharati/Projects/RSystems/NeuralReranker/BLIP_family/LAVIS",
)

# BLIP2 reranker screen uses this Python (not 39myenv)
BLIP2_PYTHON = os.environ.get(
    "BLIP2_PYTHON",
    "/home/bharati/anaconda3/envs/py3_9_neuralReranker1_0/bin/python",
)


def _patch_transformers() -> None:
    import transformers.modeling_utils as modeling_utils
    from transformers.pytorch_utils import (
        apply_chunking_to_forward,
        find_pruneable_heads_and_indices,
        prune_linear_layer,
    )

    modeling_utils.apply_chunking_to_forward = apply_chunking_to_forward
    modeling_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices
    modeling_utils.prune_linear_layer = prune_linear_layer

    try:
        import huggingface_hub as hfh
        if not hasattr(hfh, "cached_download"):
            hfh.cached_download = hfh.hf_hub_download
    except Exception:
        pass


def _ensure_pkg(name: str, path: str | None = None) -> types.ModuleType:
    if name not in sys.modules:
        mod = types.ModuleType(name)
        if path:
            mod.__path__ = [path]  # type: ignore[attr-defined]
        sys.modules[name] = mod
    return sys.modules[name]


def _load_module(name: str, path: str) -> types.ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_checkpoint_flexible(model: Any, ckpt_path: str) -> None:
    import torch
    from lavis.common.utils import is_url
    from lavis.common.dist_utils import download_cached_file

    if is_url(ckpt_path):
        ckpt_path = download_cached_file(ckpt_path, check_hash=False, progress=True)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("model") or checkpoint
    current = model.state_dict()
    filtered = {
        k: v for k, v in state_dict.items()
        if k in current and current[k].shape == v.shape
    }
    msg = model.load_state_dict(filtered, strict=False)
    logging.info("Loaded BLIP2 weights from %s (matched %d keys)", ckpt_path, len(filtered))
    if msg.missing_keys:
        logging.info("Missing keys: %d", len(msg.missing_keys))


def load_blip2_itm(device: str = "cuda") -> Tuple[Any, Any, Any]:
    """Load BLIP2-ITM + eval processors without lavis/__init__.py."""
    _patch_transformers()
    os.environ.setdefault("PYTHONNOUSERSITE", "1")

    root = Path(LAVIS_ROOT)
    lavis_root = root / "lavis"
    if str(root) not in sys.path:
        sys.path.insert(0, str(LAVIS_ROOT))

    _ensure_pkg("lavis", str(lavis_root))
    _ensure_pkg("lavis.common", str(lavis_root / "common"))
    _ensure_pkg("lavis.models", str(lavis_root / "models"))
    _ensure_pkg("lavis.models.blip2_models", str(lavis_root / "models" / "blip2_models"))
    _ensure_pkg("lavis.processors", str(lavis_root / "processors"))

    _load_module("lavis.common.registry", str(lavis_root / "common" / "registry.py"))
    _load_module("lavis.common.utils", str(lavis_root / "common" / "utils.py"))
    bm_mod = _load_module("lavis.models.base_model", str(lavis_root / "models" / "base_model.py"))
    models_pkg = sys.modules["lavis.models"]
    models_pkg.BaseModel = bm_mod.BaseModel

    bp_mod = _load_module("lavis.processors.base_processor", str(lavis_root / "processors" / "base_processor.py"))
    proc_pkg = sys.modules["lavis.processors"]
    proc_pkg.BaseProcessor = bp_mod.BaseProcessor
    _load_module("lavis.processors.blip_processors", str(lavis_root / "processors" / "blip_processors.py"))

    _load_module("lavis.models.blip2_models.blip2", str(lavis_root / "models" / "blip2_models" / "blip2.py"))
    _load_module("lavis.models.blip2_models.blip2_qformer", str(lavis_root / "models" / "blip2_models" / "blip2_qformer.py"))
    itm_mod = _load_module(
        "lavis.models.blip2_models.blip2_image_text_matching",
        str(lavis_root / "models" / "blip2_models" / "blip2_image_text_matching.py"),
    )

    from omegaconf import OmegaConf
    from lavis.common.registry import registry

    default_cfg = OmegaConf.load(str(lavis_root / "configs" / "default.yaml"))
    registry.register_path("library_root", str(lavis_root))
    registry.register_path("repo_root", str(root))
    registry.register_path("cache_root", str(root / default_cfg.env.cache_root))

    if "blip2_image_text_matching" not in registry.mapping["model_name_mapping"]:
        registry.register_model("blip2_image_text_matching")(itm_mod.Blip2ITM)

    model_cls = registry.get_model_class("blip2_image_text_matching")
    model_cfg = OmegaConf.load(model_cls.default_config_path("pretrain")).model
    model_cfg.load_pretrained = False
    model_cfg.load_finetuned = False
    model = model_cls.from_config(model_cfg)

    pretrain_path = model_cfg.get("pretrained")
    if pretrain_path:
        _load_checkpoint_flexible(model, pretrain_path)
    finetune_path = model_cfg.get("finetuned")
    if finetune_path:
        _load_checkpoint_flexible(model, finetune_path)

    model.eval()

    from lavis.processors.blip_processors import BlipCaptionProcessor, BlipImageEvalProcessor

    vis_processors = {"eval": BlipImageEvalProcessor(image_size=224)}
    txt_processors = {"eval": BlipCaptionProcessor()}

    torch = __import__("torch")
    dev = torch.device(device)
    if str(dev) == "cpu":
        model = model.float()
    return model.to(dev), vis_processors, txt_processors
