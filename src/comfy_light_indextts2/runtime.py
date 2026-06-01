from __future__ import annotations

import gc
import inspect
import os
import random
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .paths import default_model_dir, resolve_path

REPO_ROOT = Path(__file__).resolve().parents[2]
INDEXTTS_ROOT = REPO_ROOT / "indexTTS2"

_MODEL_CACHE: dict[tuple[str, str, str | None, str, bool | None], "IndexTTS2ModelHandle"] = {}


def ensure_indextts_import_path() -> None:
    path = str(INDEXTTS_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)


def auto_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu:0"
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


def normalize_precision(value: str) -> str:
    precision = (value or "FP16").strip().lower()
    aliases = {
        "float8": "fp8",
        "torch.float8": "fp8",
        "float16": "fp16",
        "half": "fp16",
        "torch.float16": "fp16",
        "bfloat16": "bf16",
        "torch.bfloat16": "bf16",
        "float32": "fp32",
        "full": "fp32",
        "torch.float32": "fp32",
    }
    precision = aliases.get(precision, precision)
    if precision not in {"fp8", "fp16", "bf16", "fp32"}:
        raise ValueError(f"Unsupported compute precision: {value}")
    return precision


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
    if hasattr(torch, "xpu") and torch.xpu.is_available() and hasattr(torch.xpu, "manual_seed_all"):
        torch.xpu.manual_seed_all(seed)


def cleanup_torch_memory() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "xpu") and torch.xpu.is_available() and hasattr(torch.xpu, "empty_cache"):
        torch.xpu.empty_cache()
    if (
        hasattr(torch, "mps")
        and hasattr(torch.mps, "empty_cache")
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        torch.mps.empty_cache()
    gc.collect()


def resolve_hf_endpoint(source: str, custom_endpoint: str = "") -> str:
    value = (source or "huggingface").strip().lower()
    if value == "huggingface":
        return "https://huggingface.co"
    if value == "hf-mirror":
        return "https://hf-mirror.com"
    if value != "custom":
        raise ValueError(f"Unsupported Hugging Face source: {source}")

    endpoint = (custom_endpoint or "").strip()
    if not endpoint:
        raise ValueError("custom_endpoint is required when source is custom.")
    if "://" not in endpoint:
        endpoint = f"https://{endpoint}"
    parsed = urlparse(endpoint)
    if not parsed.netloc:
        raise ValueError(f"Invalid custom Hugging Face endpoint: {custom_endpoint}")
    return endpoint.rstrip("/")


def configure_hf_network(endpoint: str, disable_ssl_verification: bool) -> None:
    os.environ["HF_ENDPOINT"] = endpoint
    if disable_ssl_verification:
        os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
        os.environ["CURL_CA_BUNDLE"] = ""
        os.environ["REQUESTS_CA_BUNDLE"] = ""
    else:
        os.environ.pop("HF_HUB_DISABLE_SSL_VERIFICATION", None)
        os.environ.pop("CURL_CA_BUNDLE", None)
        os.environ.pop("REQUESTS_CA_BUNDLE", None)

    try:
        import httpx
        from huggingface_hub import set_async_client_factory, set_client_factory
        from huggingface_hub.utils import close_session
    except Exception:
        pass
    else:
        try:
            from huggingface_hub.utils._http import (
                async_hf_request_event_hook,
                async_hf_response_event_hook,
                hf_request_event_hook,
            )
        except Exception:
            hf_request_event_hook = None
            async_hf_request_event_hook = None
            async_hf_response_event_hook = None

        def client_factory():
            hooks = {"request": [hf_request_event_hook]} if hf_request_event_hook else None
            return httpx.Client(
                event_hooks=hooks,
                follow_redirects=True,
                timeout=None,
                verify=not disable_ssl_verification,
            )

        def async_client_factory():
            hooks: dict[str, list[Any]] = {}
            if async_hf_request_event_hook:
                hooks["request"] = [async_hf_request_event_hook]
            if async_hf_response_event_hook:
                hooks["response"] = [async_hf_response_event_hook]
            return httpx.AsyncClient(
                event_hooks=hooks or None,
                follow_redirects=True,
                timeout=None,
                verify=not disable_ssl_verification,
            )

        set_client_factory(client_factory)
        set_async_client_factory(async_client_factory)
        close_session()
        return

    try:
        import requests
        from huggingface_hub import configure_http_backend
    except Exception:
        return

    def backend_factory():
        session = requests.Session()
        session.verify = not disable_ssl_verification
        return session

    try:
        configure_http_backend(backend_factory=backend_factory)
    except TypeError:
        configure_http_backend(backend_factory)


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"IndexTTS2 config.yaml must contain a mapping: {path}")
    return data


def _config_value(config: dict[str, Any], dotted_key: str) -> Any:
    value: Any = config
    for key in dotted_key.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _resolve_model_reference(model_dir: Path, reference: Any) -> Path | None:
    if not isinstance(reference, str) or not reference.strip():
        return None
    path = Path(reference.strip()).expanduser()
    if not path.is_absolute():
        path = model_dir / path
    return path


def validate_model_files(model_dir: Path, config_path: Path) -> list[str]:
    if not model_dir.is_dir():
        raise FileNotFoundError(f"IndexTTS2 model directory not found: {model_dir}")
    if not config_path.is_file():
        raise FileNotFoundError(f"IndexTTS2 config.yaml not found: {config_path}")

    config = _load_yaml(config_path)
    required_files = {
        "gpt_checkpoint": "GPT checkpoint",
        "s2mel_checkpoint": "S2Mel checkpoint",
        "w2v_stat": "W2V statistics",
        "dataset.bpe_model": "BPE model",
        "emo_matrix": "emotion matrix",
    }
    required_dirs = {
        "qwen_emo_path": "Qwen emotion model",
    }

    missing: list[str] = []
    for key, label in required_files.items():
        path = _resolve_model_reference(model_dir, _config_value(config, key))
        if path is None:
            missing.append(f"{label}: missing config key `{key}`")
        elif not path.is_file():
            missing.append(f"{label}: {path}")

    for key, label in required_dirs.items():
        path = _resolve_model_reference(model_dir, _config_value(config, key))
        if path is None:
            missing.append(f"{label}: missing config key `{key}`")
        elif not path.is_dir():
            missing.append(f"{label}: {path}")

    if missing:
        joined = "\n  - ".join(missing)
        raise FileNotFoundError(f"IndexTTS2 local model validation failed:\n  - {joined}")
    return sorted(required_files) + sorted(required_dirs)


def _csv_patterns(value: str) -> list[str] | None:
    patterns = [part.strip() for part in (value or "").split(",") if part.strip()]
    return patterns or None


def download_model_snapshot(
    repo_id: str,
    local_dir: str,
    revision: str,
    source: str,
    custom_endpoint: str,
    disable_ssl_verification: bool,
    force_download: bool,
    allow_patterns: str = "",
    ignore_patterns: str = "",
) -> tuple[Path, str]:
    if not (repo_id or "").strip():
        raise ValueError("repo_id is required.")

    resolved_dir = resolve_path(local_dir, default_model_dir()).resolve()
    resolved_dir.mkdir(parents=True, exist_ok=True)
    endpoint = resolve_hf_endpoint(source, custom_endpoint)
    os.environ["HF_HUB_CACHE"] = str(resolved_dir / "hf_cache")
    configure_hf_network(endpoint, disable_ssl_verification)

    from huggingface_hub import snapshot_download

    kwargs: dict[str, Any] = {
        "repo_id": repo_id.strip(),
        "repo_type": "model",
        "revision": (revision or "main").strip() or "main",
        "local_dir": str(resolved_dir),
        "cache_dir": str(resolved_dir / "hf_cache"),
        "force_download": bool(force_download),
        "allow_patterns": _csv_patterns(allow_patterns),
        "ignore_patterns": _csv_patterns(ignore_patterns),
    }
    if "local_dir_use_symlinks" in inspect.signature(snapshot_download).parameters:
        kwargs["local_dir_use_symlinks"] = False
    snapshot_download(**{key: value for key, value in kwargs.items() if value is not None})
    return resolved_dir, endpoint


@contextmanager
def hf_network_context(
    model_dir: Path,
    source: str = "huggingface",
    custom_endpoint: str = "",
    disable_ssl_verification: bool = False,
):
    endpoint = resolve_hf_endpoint(source, custom_endpoint)
    os.environ["HF_HUB_CACHE"] = str(model_dir / "hf_cache")
    configure_hf_network(endpoint, disable_ssl_verification)
    yield endpoint


@dataclass
class IndexTTS2ModelHandle:
    model: Any
    model_dir: Path
    config_path: Path
    device: str
    precision: str
    cuda_kernel: bool | None


def load_model_handle(
    model_dir: str,
    config_path: str,
    device: str,
    precision: str,
    cuda_kernel: str,
    reload_model: bool = False,
    hf_source: str = "huggingface",
    custom_endpoint: str = "",
    disable_ssl_verification: bool = False,
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

    validate_model_files(resolved_model_dir, resolved_config)

    selected_device = normalize_device(device)
    actual_device = selected_device or auto_device()
    actual_precision = normalize_precision(precision)
    if str(actual_device).startswith("cpu") and actual_precision in {"fp8", "fp16"}:
        actual_precision = "fp32"
    if str(actual_device).startswith("mps") and actual_precision in {"fp8", "fp16", "bf16"}:
        actual_precision = "fp32"
    actual_cuda_kernel = normalize_cuda_kernel(cuda_kernel)
    key = (str(resolved_model_dir), str(resolved_config), selected_device, actual_precision, actual_cuda_kernel)
    if not reload_model and key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    with hf_network_context(resolved_model_dir, hf_source, custom_endpoint, disable_ssl_verification):
        from indextts.infer_indextts2 import IndexTTS2

        model = IndexTTS2(
            cfg_path=str(resolved_config),
            model_dir=str(resolved_model_dir),
            is_fp16=actual_precision == "fp16",
            precision=actual_precision,
            device=selected_device,
            use_cuda_kernel=actual_cuda_kernel,
        )
    handle = IndexTTS2ModelHandle(
        model=model,
        model_dir=resolved_model_dir,
        config_path=resolved_config,
        device=model.device,
        precision=getattr(model, "precision_name", actual_precision).upper(),
        cuda_kernel=model.use_cuda_kernel,
    )
    _MODEL_CACHE[key] = handle
    return handle


def unload_model_handle(handle: IndexTTS2ModelHandle) -> None:
    stale_keys = [key for key, cached in _MODEL_CACHE.items() if cached is handle]
    for key in stale_keys:
        _MODEL_CACHE.pop(key, None)
    handle.model = None
    cleanup_torch_memory()


def clear_model_cache() -> None:
    _MODEL_CACHE.clear()
    cleanup_torch_memory()
