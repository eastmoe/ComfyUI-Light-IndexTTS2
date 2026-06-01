from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .paths import default_model_dir, resolve_path

REPO_ROOT = Path(__file__).resolve().parents[2]
INDEXTTS_ROOT = REPO_ROOT / "indexTTS2"

_MODEL_CACHE: dict[tuple[str, str, str | None, bool, bool | None], "IndexTTS2ModelHandle"] = {}


def ensure_indextts_import_path() -> None:
    path = str(INDEXTTS_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)


def auto_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def normalize_device(device: str) -> str | None:
    value = (device or "auto").strip()
    return None if value.lower() == "auto" else value


def normalize_cuda_kernel(value: str) -> bool | None:
    value = (value or "auto").strip().lower()
    if value == "enabled":
        return True
    if value == "disabled":
        return False
    return None


def set_seed(seed: int) -> None:
    if int(seed) < 0:
        return
    import numpy as np
    import torch

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cleanup_torch_memory() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


@dataclass
class IndexTTS2ModelHandle:
    model: Any
    model_dir: Path
    config_path: Path
    device: str
    fp16: bool
    cuda_kernel: bool | None


def load_model_handle(
    model_dir: str,
    config_path: str,
    device: str,
    fp16: bool,
    cuda_kernel: str,
    reload_model: bool = False,
) -> IndexTTS2ModelHandle:
    ensure_indextts_import_path()
    resolved_model_dir = resolve_path(model_dir, default_model_dir()).resolve()
    config_text = (config_path or "").strip()
    if not config_text or config_text.lower() == "auto":
        resolved_config = resolved_model_dir / "config.yaml"
    else:
        config_candidate = Path(config_text).expanduser()
        if not config_candidate.is_absolute():
            config_candidate = resolved_model_dir / config_candidate
        resolved_config = config_candidate.resolve()

    if not resolved_model_dir.is_dir():
        raise FileNotFoundError(f"IndexTTS2 model directory not found: {resolved_model_dir}")
    if not resolved_config.is_file():
        raise FileNotFoundError(f"IndexTTS2 config.yaml not found: {resolved_config}")

    selected_device = normalize_device(device)
    actual_device = selected_device or auto_device()
    actual_fp16 = bool(fp16) and str(actual_device).startswith("cuda")
    actual_cuda_kernel = normalize_cuda_kernel(cuda_kernel)
    key = (str(resolved_model_dir), str(resolved_config), selected_device, actual_fp16, actual_cuda_kernel)
    if not reload_model and key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    os.environ["HF_HUB_CACHE"] = str(resolved_model_dir / "hf_cache")
    from indextts.infer_indextts2 import IndexTTS2

    os.environ["HF_HUB_CACHE"] = str(resolved_model_dir / "hf_cache")
    model = IndexTTS2(
        cfg_path=str(resolved_config),
        model_dir=str(resolved_model_dir),
        is_fp16=actual_fp16,
        device=selected_device,
        use_cuda_kernel=actual_cuda_kernel,
    )
    handle = IndexTTS2ModelHandle(
        model=model,
        model_dir=resolved_model_dir,
        config_path=resolved_config,
        device=model.device,
        fp16=model.is_fp16,
        cuda_kernel=model.use_cuda_kernel,
    )
    _MODEL_CACHE[key] = handle
    return handle


def clear_model_cache() -> None:
    _MODEL_CACHE.clear()
    cleanup_torch_memory()
