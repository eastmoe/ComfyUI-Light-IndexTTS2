from __future__ import annotations

import gc
import inspect
import os
import random
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .paths import DEFAULT_MODEL_SUBDIR, MODEL_FOLDER_NAME, default_model_dir, default_model_root, resolve_path
from .quantization import (
    apply_index_tts2_quantization,
    default_quantization_cache_dir,
    normalize_quantization,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
INDEXTTS_ROOT = REPO_ROOT / "indexTTS2"

_MODEL_CACHE: dict[tuple[Any, ...], "IndexTTS2ModelHandle"] = {}


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
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
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
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
    _patch_hf_runtime_constants()
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
            timeout = httpx.Timeout(
                _env_float("HF_HUB_DOWNLOAD_TIMEOUT", 60),
                connect=_env_float("HF_HUB_ETAG_TIMEOUT", 30),
                read=_env_float("HF_HUB_DOWNLOAD_TIMEOUT", 60),
                write=_env_float("HF_HUB_DOWNLOAD_TIMEOUT", 60),
                pool=_env_float("HF_HUB_ETAG_TIMEOUT", 30),
            )
            return httpx.Client(
                event_hooks=hooks,
                follow_redirects=True,
                timeout=timeout,
                verify=not disable_ssl_verification,
            )

        def async_client_factory():
            hooks: dict[str, list[Any]] = {}
            if async_hf_request_event_hook:
                hooks["request"] = [async_hf_request_event_hook]
            if async_hf_response_event_hook:
                hooks["response"] = [async_hf_response_event_hook]
            timeout = httpx.Timeout(
                _env_float("HF_HUB_DOWNLOAD_TIMEOUT", 60),
                connect=_env_float("HF_HUB_ETAG_TIMEOUT", 30),
                read=_env_float("HF_HUB_DOWNLOAD_TIMEOUT", 60),
                write=_env_float("HF_HUB_DOWNLOAD_TIMEOUT", 60),
                pool=_env_float("HF_HUB_ETAG_TIMEOUT", 30),
            )
            return httpx.AsyncClient(
                event_hooks=hooks or None,
                follow_redirects=True,
                timeout=timeout,
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


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _patch_hf_runtime_constants() -> None:
    try:
        from huggingface_hub import constants
    except Exception:
        return

    constants.HF_HUB_DISABLE_XET = _env_bool("HF_HUB_DISABLE_XET", True)
    constants.HF_HUB_DOWNLOAD_TIMEOUT = _env_float("HF_HUB_DOWNLOAD_TIMEOUT", 60)
    constants.HF_HUB_ETAG_TIMEOUT = _env_float("HF_HUB_ETAG_TIMEOUT", 30)


def _hf_download_workers() -> int:
    try:
        value = int(os.environ.get("INDEXTTS2_HF_DOWNLOAD_WORKERS", "1"))
    except ValueError:
        value = 1
    return max(1, min(value, 8))


def _call_hf_download(fn, **kwargs):
    signature = inspect.signature(fn).parameters
    return fn(**{key: value for key, value in kwargs.items() if key in signature and value is not None})


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


def _repo_dir_name(repo_id: str) -> str:
    leaf = (repo_id or "").strip().rstrip("/").rsplit("/", 1)[-1] or DEFAULT_MODEL_SUBDIR
    return re.sub(r"[^A-Za-z0-9._-]+", "-", leaf).strip(".-") or DEFAULT_MODEL_SUBDIR


def _resolve_model_dir(value: str | None, repo_id: str = "IndexTeam/IndexTTS-2") -> Path:
    text = (value or "").strip()
    if not text or text.lower() == "auto":
        candidate = default_model_dir()
        legacy = default_model_root()
    else:
        candidate = Path(text).expanduser()
        legacy = candidate

    if (candidate / "config.yaml").is_file():
        return candidate.resolve()

    child = candidate / _repo_dir_name(repo_id)
    if child != candidate and (child / "config.yaml").is_file():
        return child.resolve()

    if (legacy / "config.yaml").is_file():
        return legacy.resolve()

    return candidate.resolve()


def _resolve_download_root(local_dir: str) -> Path:
    return resolve_path(local_dir, default_model_root()).resolve()


def _resolve_download_target(local_dir: str, repo_id: str) -> Path:
    root = _resolve_download_root(local_dir)
    subdir = _repo_dir_name(repo_id)
    if root.name == subdir:
        return root
    return root / subdir


def _model_root_for_dependencies(model_dir: Path) -> Path:
    if model_dir.name == DEFAULT_MODEL_SUBDIR or model_dir.parent.name == MODEL_FOLDER_NAME:
        return model_dir.parent
    return model_dir


def _local_repo_dir(root: Path, repo_id: str) -> Path:
    return root / _repo_dir_name(repo_id)


def _has_any_file(directory: Path, filenames: tuple[str, ...]) -> bool:
    return any((directory / filename).is_file() for filename in filenames)


def _dependency_status(model_dir: Path, config_path: Path) -> tuple[list[str], list[str]]:
    dependency_root = _model_root_for_dependencies(model_dir)
    config = _load_yaml(config_path)
    present: list[str] = []
    missing: list[str] = []

    w2v_dir = _local_repo_dir(dependency_root, "facebook/w2v-bert-2.0")
    if (w2v_dir / "config.json").is_file() and (w2v_dir / "preprocessor_config.json").is_file() and _has_any_file(
        w2v_dir,
        ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json"),
    ):
        present.append(str(w2v_dir))
    else:
        missing.append("facebook/w2v-bert-2.0")

    maskgct_file = _local_repo_dir(dependency_root, "amphion/MaskGCT") / "semantic_codec" / "model.safetensors"
    if maskgct_file.is_file():
        present.append(str(maskgct_file.parent.parent))
    else:
        missing.append("amphion/MaskGCT")

    campplus_file = _local_repo_dir(dependency_root, "funasr/campplus") / "campplus_cn_common.bin"
    if campplus_file.is_file():
        present.append(str(campplus_file.parent))
    else:
        missing.append("funasr/campplus")

    bigvgan_name = _config_value(config, "vocoder.name")
    if isinstance(bigvgan_name, str) and bigvgan_name.strip() and not Path(bigvgan_name).expanduser().is_dir():
        bigvgan_dir = _local_repo_dir(dependency_root, bigvgan_name.strip())
        if (bigvgan_dir / "config.json").is_file() and (bigvgan_dir / "bigvgan_generator.pt").is_file():
            present.append(str(bigvgan_dir))
        else:
            missing.append(bigvgan_name.strip())

    return present, missing


def _hf_cache_dir(model_dir: Path) -> Path:
    return _model_root_for_dependencies(model_dir) / "hf_cache"


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
    config_path: str = "auto",
) -> tuple[Path, str]:
    if not (repo_id or "").strip():
        raise ValueError("repo_id is required.")

    resolved_dir = _resolve_download_target(local_dir, repo_id)
    endpoint = resolve_hf_endpoint(source, custom_endpoint)
    if not force_download:
        config_text = (config_path or "").strip()
        resolved_config = resolved_dir / "config.yaml" if not config_text or config_text.lower() == "auto" else Path(config_text).expanduser()
        if not resolved_config.is_absolute():
            resolved_config = resolved_dir / resolved_config
        if resolved_config.is_file():
            try:
                validate_model_files(resolved_dir, resolved_config.resolve())
                print(f">> Local IndexTTS2 model found, skip Hugging Face snapshot download: {resolved_dir}", flush=True)
                return resolved_dir.resolve(), endpoint
            except FileNotFoundError:
                pass

    resolved_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = _hf_cache_dir(resolved_dir)
    os.environ["HF_HUB_CACHE"] = str(cache_dir)
    configure_hf_network(endpoint, disable_ssl_verification)

    from huggingface_hub import snapshot_download

    kwargs: dict[str, Any] = {
        "repo_id": repo_id.strip(),
        "repo_type": "model",
        "revision": (revision or "main").strip() or "main",
        "local_dir": str(resolved_dir),
        "cache_dir": str(cache_dir),
        "force_download": bool(force_download),
        "allow_patterns": _csv_patterns(allow_patterns),
        "ignore_patterns": _csv_patterns(ignore_patterns),
        "max_workers": _hf_download_workers(),
        "etag_timeout": _env_float("HF_HUB_ETAG_TIMEOUT", 30),
        "endpoint": endpoint,
    }
    if "local_dir_use_symlinks" in inspect.signature(snapshot_download).parameters:
        kwargs["local_dir_use_symlinks"] = False
    _call_hf_download(snapshot_download, **kwargs)
    return resolved_dir, endpoint


def download_runtime_dependencies(
    model_dir: Path,
    config_path: Path,
    *,
    source: str,
    custom_endpoint: str,
    disable_ssl_verification: bool,
    force_download: bool = False,
) -> list[str]:
    if not force_download:
        present, missing = _dependency_status(model_dir, config_path)
        if not missing:
            print(">> Local IndexTTS2 dependencies found, skip Hugging Face dependency downloads.", flush=True)
            return present

    endpoint = resolve_hf_endpoint(source, custom_endpoint)
    dependency_root = _model_root_for_dependencies(model_dir)
    cache_dir = _hf_cache_dir(model_dir)
    os.environ["HF_HUB_CACHE"] = str(cache_dir)
    configure_hf_network(endpoint, disable_ssl_verification)

    from huggingface_hub import hf_hub_download, snapshot_download

    config = _load_yaml(config_path)
    downloaded: list[str] = []

    w2v_dir = _local_repo_dir(dependency_root, "facebook/w2v-bert-2.0")
    if force_download or "facebook/w2v-bert-2.0" in _dependency_status(model_dir, config_path)[1]:
        snapshot_kwargs: dict[str, Any] = {
            "repo_id": "facebook/w2v-bert-2.0",
            "repo_type": "model",
            "cache_dir": str(cache_dir),
            "local_dir": str(w2v_dir),
            "force_download": bool(force_download),
            "max_workers": _hf_download_workers(),
            "etag_timeout": _env_float("HF_HUB_ETAG_TIMEOUT", 30),
            "endpoint": endpoint,
            "allow_patterns": [
                "config.json",
                "preprocessor_config.json",
                "model.safetensors",
                "model.safetensors.index.json",
                "*.safetensors",
                "pytorch_model.bin",
            ],
        }
        if "local_dir_use_symlinks" in inspect.signature(snapshot_download).parameters:
            snapshot_kwargs["local_dir_use_symlinks"] = False
        _call_hf_download(snapshot_download, **snapshot_kwargs)
    downloaded.append(str(w2v_dir))

    hub_kwargs: dict[str, Any] = {
        "cache_dir": str(cache_dir),
        "force_download": bool(force_download),
        "etag_timeout": _env_float("HF_HUB_ETAG_TIMEOUT", 30),
        "endpoint": endpoint,
    }
    maskgct_dir = _local_repo_dir(dependency_root, "amphion/MaskGCT")
    if force_download or not (maskgct_dir / "semantic_codec" / "model.safetensors").is_file():
        _call_hf_download(
            hf_hub_download,
            repo_id="amphion/MaskGCT",
            filename="semantic_codec/model.safetensors",
            local_dir=str(maskgct_dir),
            **hub_kwargs,
        )
    downloaded.append(str(maskgct_dir))

    campplus_dir = _local_repo_dir(dependency_root, "funasr/campplus")
    if force_download or not (campplus_dir / "campplus_cn_common.bin").is_file():
        _call_hf_download(
            hf_hub_download,
            repo_id="funasr/campplus",
            filename="campplus_cn_common.bin",
            local_dir=str(campplus_dir),
            **hub_kwargs,
        )
    downloaded.append(str(campplus_dir))

    bigvgan_name = _config_value(config, "vocoder.name")
    if isinstance(bigvgan_name, str) and bigvgan_name.strip() and not Path(bigvgan_name).expanduser().is_dir():
        bigvgan_dir = _local_repo_dir(dependency_root, bigvgan_name.strip())
        if force_download or not ((bigvgan_dir / "config.json").is_file() and (bigvgan_dir / "bigvgan_generator.pt").is_file()):
            _call_hf_download(
                hf_hub_download,
                repo_id=bigvgan_name.strip(),
                filename="config.json",
                local_dir=str(bigvgan_dir),
                **hub_kwargs,
            )
            _call_hf_download(
                hf_hub_download,
                repo_id=bigvgan_name.strip(),
                filename="bigvgan_generator.pt",
                local_dir=str(bigvgan_dir),
                **hub_kwargs,
            )
        downloaded.append(str(bigvgan_dir))

    return downloaded


@contextmanager
def hf_network_context(
    model_dir: Path,
    source: str = "huggingface",
    custom_endpoint: str = "",
    disable_ssl_verification: bool = False,
):
    endpoint = resolve_hf_endpoint(source, custom_endpoint)
    os.environ["HF_HUB_CACHE"] = str(_hf_cache_dir(model_dir))
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
    quantization: str
    quantization_info: dict[str, Any]
    inference_signature: tuple[Any, ...] | None = None


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
    quantization: str = "none",
    use_quantization_cache: bool = True,
    rebuild_quantization_cache: bool = False,
    quantized_linear_chunk_size: int = 1024,
) -> IndexTTS2ModelHandle:
    ensure_indextts_import_path()
    resolved_model_dir = _resolve_model_dir(model_dir)
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
    actual_quantization = normalize_quantization(quantization)
    quantization_cache_dir = (
        default_quantization_cache_dir(resolved_model_dir)
        if actual_quantization is not None and bool(use_quantization_cache)
        else None
    )
    key = (
        str(resolved_model_dir),
        str(resolved_config),
        selected_device,
        actual_precision,
        actual_cuda_kernel,
        actual_quantization,
        str(quantization_cache_dir.resolve()) if quantization_cache_dir else None,
        int(quantized_linear_chunk_size),
    )
    if not reload_model and not rebuild_quantization_cache and key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    _evict_related_model_handles(resolved_model_dir, resolved_config, keep_key=key)

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
    quantization_info = apply_index_tts2_quantization(
        model,
        model_dir=resolved_model_dir,
        config_path=resolved_config,
        mode=actual_quantization,
        cache_dir=quantization_cache_dir,
        rebuild_cache=bool(rebuild_quantization_cache),
        chunk_size=int(quantized_linear_chunk_size),
    )
    cleanup_torch_memory()
    handle = IndexTTS2ModelHandle(
        model=model,
        model_dir=resolved_model_dir,
        config_path=resolved_config,
        device=model.device,
        precision=getattr(model, "precision_name", actual_precision).upper(),
        cuda_kernel=model.use_cuda_kernel,
        quantization=quantization_info["mode"],
        quantization_info=quantization_info,
    )
    _MODEL_CACHE[key] = handle
    return handle


def _evict_related_model_handles(model_dir: Path, config_path: Path, keep_key: tuple[Any, ...] | None = None) -> None:
    stale_keys = [
        key
        for key in _MODEL_CACHE
        if key != keep_key and len(key) >= 2 and key[0] == str(model_dir) and key[1] == str(config_path)
    ]
    for key in stale_keys:
        cached = _MODEL_CACHE.pop(key, None)
        if cached is not None:
            cached.model = None
    if stale_keys:
        cleanup_torch_memory()


def clear_model_runtime_state(model: Any) -> None:
    if model is None:
        return
    for attr in (
        "cache_spk_cond",
        "cache_s2mel_style",
        "cache_s2mel_prompt",
        "cache_spk_audio_prompt",
        "cache_emo_cond",
        "cache_emo_audio_prompt",
        "cache_mel",
    ):
        if hasattr(model, attr):
            setattr(model, attr, None)

    inference_model = getattr(getattr(model, "gpt", None), "inference_model", None)
    if inference_model is not None:
        if hasattr(inference_model, "cached_mel_emb"):
            inference_model.cached_mel_emb = None
        if hasattr(inference_model, "_cache"):
            delattr(inference_model, "_cache")
    cleanup_torch_memory()


def unload_model_handle(handle: IndexTTS2ModelHandle) -> None:
    stale_keys = [key for key, cached in _MODEL_CACHE.items() if cached is handle]
    for key in stale_keys:
        _MODEL_CACHE.pop(key, None)
    handle.model = None
    cleanup_torch_memory()


def clear_model_cache() -> None:
    for handle in _MODEL_CACHE.values():
        handle.model = None
    _MODEL_CACHE.clear()
    cleanup_torch_memory()
