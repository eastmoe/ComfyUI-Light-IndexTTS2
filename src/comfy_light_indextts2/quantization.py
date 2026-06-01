from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


QUANTIZATION_CACHE_DIR_NAME = "IndexTTS2-cache"


def normalize_quantization(value: str | None) -> str | None:
    if value is None:
        return None
    aliases = {
        "": None,
        "none": None,
        "off": None,
        "false": None,
        "no": None,
        "fp8": "fp8",
        "float8": "fp8",
        "fbgemm-fp8": "fp8",
        "fp4": "fp4",
        "float4": "fp4",
    }
    key = str(value).strip().lower()
    if key not in aliases:
        raise ValueError(f"Unsupported quantization mode: {value!r}. Choose one of: none, FP8, FP4.")
    return aliases[key]


def default_quantization_cache_dir(model_dir: Path) -> Path:
    resolved = model_dir.resolve()
    root = resolved.parent if resolved.name == "IndexTTS-2" or resolved.parent.name == "IndexTTS2" else resolved
    return root / QUANTIZATION_CACHE_DIR_NAME


class WeightOnlyQuantizedLinear(torch.nn.Module):
    def __init__(
        self,
        linear: torch.nn.Linear,
        mode: str,
        *,
        codebook: torch.Tensor | None = None,
        chunk_size: int | None = 1024,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.mode = mode
        self.chunk_size = chunk_size

        target_device = torch.device(device) if device is not None else linear.weight.device
        weight = linear.weight.detach().float().cpu()
        bias = None if linear.bias is None else linear.bias.detach().clone().to(target_device)
        self.register_buffer("bias", bias)

        if mode == "fp8":
            if not hasattr(torch, "float8_e4m3fn"):
                raise RuntimeError("FP8 quantization requires PyTorch float8_e4m3fn support.")
            self.register_buffer("qweight", weight.to(device=target_device, dtype=torch.float8_e4m3fn))
            self.register_buffer("scale", None)
            return

        if mode != "fp4":
            raise ValueError(f"Unsupported quantization mode: {mode}")
        if codebook is None:
            codebook = _four_bit_codebook()

        packed, scale = _quantize_fp4_weight(weight, codebook)
        self.register_buffer("qweight", packed.to(target_device))
        self.register_buffer("scale", scale.to(target_device))
        self.register_buffer("codebook", codebook.float().to(target_device))

    @classmethod
    def empty_from_linear(
        cls,
        linear: torch.nn.Linear,
        mode: str,
        *,
        chunk_size: int | None,
        has_bias: bool,
        device: torch.device | str,
    ) -> "WeightOnlyQuantizedLinear":
        module = cls.__new__(cls)
        torch.nn.Module.__init__(module)
        module.in_features = linear.in_features
        module.out_features = linear.out_features
        module.mode = mode
        module.chunk_size = chunk_size

        target_device = torch.device(device)
        if mode == "fp8":
            fp8_dtype = getattr(torch, "float8_e4m3fn", torch.uint8)
            qweight = torch.empty(
                (linear.out_features, linear.in_features),
                dtype=fp8_dtype,
                device=target_device,
            )
            scale = None
            codebook = None
        elif mode == "fp4":
            qweight = torch.empty(
                (linear.out_features, (linear.in_features + 1) // 2),
                dtype=torch.uint8,
                device=target_device,
            )
            scale = torch.empty((linear.out_features, 1), dtype=torch.float32, device=target_device)
            codebook = torch.empty((16,), dtype=torch.float32, device=target_device)
        else:
            raise ValueError(f"Unsupported quantization mode: {mode}")

        bias = None
        if has_bias:
            bias_dtype = linear.bias.dtype if linear.bias is not None else torch.float32
            bias = torch.empty((linear.out_features,), dtype=bias_dtype, device=target_device)

        module.register_buffer("bias", bias)
        module.register_buffer("qweight", qweight)
        module.register_buffer("scale", scale)
        if codebook is not None:
            module.register_buffer("codebook", codebook)
        return module

    def _dequantize_weight(
        self,
        dtype: torch.dtype,
        device: torch.device,
        row_start: int = 0,
        row_end: int | None = None,
    ) -> torch.Tensor:
        row_end = self.out_features if row_end is None else row_end
        if self.mode == "fp8":
            return self.qweight[row_start:row_end].to(device=device, dtype=dtype)

        packed = self.qweight[row_start:row_end].to(device)
        low = packed & 0x0F
        high = (packed >> 4) & 0x0F
        indices = torch.stack((low, high), dim=-1).flatten(1)[:, : self.in_features]
        values = self.codebook.to(device=device, dtype=dtype)
        return values[indices.long()] * self.scale[row_start:row_end].to(device=device, dtype=dtype)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.chunk_size and self.chunk_size > 0 and self.out_features > 16384:
            flat_inputs = inputs.reshape(-1, self.in_features)
            chunks = []
            for row_start in range(0, self.out_features, self.chunk_size):
                row_end = min(row_start + self.chunk_size, self.out_features)
                weight = self._dequantize_weight(inputs.dtype, inputs.device, row_start, row_end)
                bias = None
                if self.bias is not None:
                    bias = self.bias[row_start:row_end].to(device=inputs.device, dtype=inputs.dtype)
                chunks.append(F.linear(flat_inputs, weight, bias))
            return torch.cat(chunks, dim=-1).reshape(*inputs.shape[:-1], self.out_features)

        weight = self._dequantize_weight(inputs.dtype, inputs.device)
        bias = None if self.bias is None else self.bias.to(device=inputs.device, dtype=inputs.dtype)
        return F.linear(inputs, weight, bias)


def apply_index_tts2_quantization(
    model: Any,
    *,
    model_dir: Path,
    config_path: Path,
    mode: str | None,
    cache_dir: Path | None,
    rebuild_cache: bool,
    chunk_size: int | None = 1024,
) -> dict[str, Any]:
    normalized_mode = normalize_quantization(mode)
    if normalized_mode is None:
        return {
            "mode": "none",
            "cache_dir": str(cache_dir) if cache_dir else "",
            "cache_path": "",
            "cache_status": "disabled",
            "linear_layers": 0,
        }

    if chunk_size is not None and int(chunk_size) <= 0:
        chunk_size = None
    device = torch.device(getattr(model, "device", "cpu"))
    cache_path = None
    metadata = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path, metadata = _cache_path(cache_dir, model_dir, config_path, normalized_mode, chunk_size)
        if cache_path.is_file() and not rebuild_cache:
            try:
                started = time.monotonic()
                payload = _torch_load(cache_path, map_location=device)
                if payload.get("metadata") != metadata:
                    raise RuntimeError("cache metadata does not match the current model files")
                applied = _load_quantized_cache(
                    model,
                    payload["components"],
                    normalized_mode,
                    chunk_size=chunk_size,
                    device=device,
                )
                return {
                    "mode": normalized_mode.upper(),
                    "cache_dir": str(cache_dir),
                    "cache_path": str(cache_path),
                    "cache_status": f"loaded in {time.monotonic() - started:.2f}s",
                    "linear_layers": applied,
                }
            except Exception as exc:
                print(f">> Could not use IndexTTS2 quantization cache ({exc}); rebuilding.", flush=True)

    started = time.monotonic()
    components = _named_components(model)
    saved_components: dict[str, Any] = {}
    total = 0
    for component_name, component in components.items():
        names = _replace_linear_modules(
            component,
            normalized_mode,
            chunk_size=chunk_size,
            device=device,
        )
        if not names:
            continue
        total += len(names)
        saved_components[component_name] = {
            "quantized_module_names": names,
            "state_dict": _quantized_state_dict(component, names),
        }

    status = f"rebuilt in {time.monotonic() - started:.2f}s"
    if cache_path is not None and metadata is not None:
        try:
            tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
            torch.save({"metadata": metadata, "components": saved_components}, tmp_path)
            os.replace(tmp_path, cache_path)
            status = f"saved in {time.monotonic() - started:.2f}s"
        except Exception as exc:
            status = f"save failed: {exc}"
            print(f">> Warning: failed to save IndexTTS2 quantization cache: {exc}", flush=True)

    return {
        "mode": normalized_mode.upper(),
        "cache_dir": str(cache_dir) if cache_dir else "",
        "cache_path": str(cache_path) if cache_path else "",
        "cache_status": status,
        "linear_layers": total,
    }


def _four_bit_codebook() -> torch.Tensor:
    return torch.linspace(-1.0, 1.0, steps=16, dtype=torch.float32)


def _quantization_chunk_rows(weight: torch.Tensor) -> int:
    if weight.ndim != 2 or weight.shape[1] == 0:
        return 1
    max_temp_bytes = int(os.environ.get("INDEXTTS2_QUANTIZE_MAX_TEMP_MB", "256")) * 1024 * 1024
    bytes_per_row = max(1, weight.shape[1] * 16 * 4)
    return max(1, min(weight.shape[0], max_temp_bytes // bytes_per_row))


def _quantize_fp4_weight(weight: torch.Tensor, codebook: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    out_features, in_features = weight.shape
    packed_width = (in_features + 1) // 2
    packed = torch.empty((out_features, packed_width), dtype=torch.uint8)
    scale = torch.empty((out_features, 1), dtype=torch.float32)
    cpu_codebook = codebook.float().cpu()

    for start in range(0, out_features, _quantization_chunk_rows(weight)):
        end = min(start + _quantization_chunk_rows(weight), out_features)
        chunk = weight[start:end].float()
        chunk_scale = chunk.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        normalized = (chunk / chunk_scale).clamp(-1, 1)
        distances = (normalized.unsqueeze(-1) - cpu_codebook.view(1, 1, -1)).abs()
        qweight = distances.argmin(dim=-1).to(torch.uint8)
        if in_features % 2:
            qweight = F.pad(qweight, (0, 1))
        packed[start:end] = qweight[:, 0::2] | (qweight[:, 1::2] << 4)
        scale[start:end] = chunk_scale
    return packed.contiguous(), scale


def _replace_linear_modules(
    module: torch.nn.Module,
    mode: str,
    *,
    chunk_size: int | None,
    device: torch.device,
    prefix: str = "",
) -> list[str]:
    replaced: list[str] = []
    codebook = _four_bit_codebook() if mode == "fp4" else None
    for name, child in list(module.named_children()):
        child_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, torch.nn.Linear):
            setattr(
                module,
                name,
                WeightOnlyQuantizedLinear(
                    child,
                    mode,
                    codebook=codebook,
                    chunk_size=chunk_size,
                    device=device,
                ),
            )
            replaced.append(child_name)
        else:
            replaced.extend(
                _replace_linear_modules(
                    child,
                    mode,
                    chunk_size=chunk_size,
                    device=device,
                    prefix=child_name,
                )
            )
    return replaced


def _load_quantized_cache(
    model: Any,
    components_payload: dict[str, Any],
    mode: str,
    *,
    chunk_size: int | None,
    device: torch.device,
) -> int:
    components = _named_components(model)
    total = 0
    for component_name, payload in components_payload.items():
        component = components.get(component_name)
        if component is None:
            raise RuntimeError(f"quantization cache component not found: {component_name}")
        names = list(payload["quantized_module_names"])
        state_dict = payload["state_dict"]
        for module_name in names:
            module = component.get_submodule(module_name)
            if not isinstance(module, torch.nn.Linear):
                raise RuntimeError(f"cache expected Linear module {component_name}.{module_name}")
            parent_name, child_name = _split_module_name(module_name)
            parent = component.get_submodule(parent_name) if parent_name else component
            setattr(
                parent,
                child_name,
                WeightOnlyQuantizedLinear.empty_from_linear(
                    module,
                    mode,
                    chunk_size=chunk_size,
                    has_bias=f"{module_name}.bias" in state_dict,
                    device=device,
                ),
            )
        incompatible = component.load_state_dict(state_dict, strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(
                f"unexpected quantization cache keys for {component_name}: "
                f"{', '.join(incompatible.unexpected_keys[:5])}"
            )
        total += len(names)
    return total


def _quantized_state_dict(module: torch.nn.Module, names: list[str]) -> dict[str, torch.Tensor]:
    prefixes = tuple(f"{name}." for name in names)
    state: dict[str, torch.Tensor] = {}
    for key, tensor in module.state_dict().items():
        if key.startswith(prefixes):
            state[key] = tensor.detach().cpu()
    return state


def _named_components(model: Any) -> dict[str, torch.nn.Module]:
    names = (
        "qwen_emo.model",
        "gpt",
        "semantic_model",
        "semantic_codec",
        "s2mel",
        "campplus_model",
        "bigvgan",
    )
    components: dict[str, torch.nn.Module] = {}
    for name in names:
        value: Any = model
        for part in name.split("."):
            value = getattr(value, part, None)
            if value is None:
                break
        if isinstance(value, torch.nn.Module):
            components[name] = value
    return components


def _cache_path(
    cache_dir: Path,
    model_dir: Path,
    config_path: Path,
    mode: str,
    chunk_size: int | None,
) -> tuple[Path, dict[str, Any]]:
    metadata = {
        "cache_version": 1,
        "format": "indextts2-weight-only-linear",
        "mode": mode,
        "chunk_size": chunk_size,
        "signature": _model_signature(model_dir, config_path),
    }
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"IndexTTS2-{mode}-{digest}.pt", metadata


def _model_signature(model_dir: Path, config_path: Path) -> dict[str, Any]:
    paths = [config_path]
    try:
        import yaml

        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        for key in ("gpt_checkpoint", "s2mel_checkpoint", "w2v_stat", "emo_matrix"):
            value = config.get(key)
            if isinstance(value, str):
                paths.append((model_dir / value).resolve())
        qwen_path = config.get("qwen_emo_path")
        if isinstance(qwen_path, str):
            qwen_dir = (model_dir / qwen_path).resolve()
            for filename in ("config.json", "model.safetensors", "model.safetensors.index.json"):
                paths.append(qwen_dir / filename)
    except Exception:
        pass

    files = []
    for path in paths:
        if path.is_file():
            stat = path.stat()
            files.append(
                {
                    "path": str(path.resolve()),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
    return {"model_dir": str(model_dir.resolve()), "files": files}


def _split_module_name(name: str) -> tuple[str, str]:
    if "." not in name:
        return "", name
    return name.rsplit(".", 1)


def _torch_load(path: Path, map_location: torch.device) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
