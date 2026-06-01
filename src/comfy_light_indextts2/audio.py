from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .paths import comfy_temp_dir


def select_audio_batch(audio: dict[str, Any], batch_index: int = 0):
    import torch

    if not isinstance(audio, dict) or "waveform" not in audio:
        raise TypeError("Expected ComfyUI AUDIO input with waveform and sample_rate.")

    waveform = audio["waveform"]
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.as_tensor(waveform)
    waveform = waveform.detach().float().cpu()

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0).unsqueeze(0)
    elif waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim != 3:
        raise ValueError(f"Expected AUDIO waveform [batch, channels, samples], got {tuple(waveform.shape)}.")

    index = max(0, min(int(batch_index), waveform.shape[0] - 1))
    sample_rate = int(audio.get("sample_rate") or 44100)
    return waveform[index], sample_rate


def audio_to_temp_wav(audio: dict[str, Any], prefix: str, batch_index: int = 0) -> Path:
    import numpy as np
    import torchaudio

    waveform, sample_rate = select_audio_batch(audio, batch_index)
    if waveform.numel() == 0:
        raise ValueError("Input AUDIO is empty.")
    if not torch.isfinite(waveform).all():
        raise ValueError("Input AUDIO contains NaN or Inf values.")

    mono = waveform.mean(dim=0, keepdim=True).clamp(-1.0, 1.0)
    digest = hashlib.sha256()
    digest.update(mono.contiguous().numpy().view(np.uint8))
    digest.update(str(sample_rate).encode("utf-8"))

    temp_dir = comfy_temp_dir() / "Comfy-Light-IndexTTS2"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / f"{prefix}_{digest.hexdigest()[:24]}.wav"
    if not path.is_file():
        torchaudio.save(str(path), mono, sample_rate)
    return path


def comfy_audio_from_infer_result(result: tuple[int, Any]) -> dict[str, Any]:
    import numpy as np
    import torch

    sample_rate, data = result
    array = np.asarray(data)
    if array.ndim == 1:
        array = array[:, None]
    if array.dtype.kind in {"i", "u"}:
        array = array.astype(np.float32) / 32768.0
    else:
        array = array.astype(np.float32, copy=False)
    waveform = torch.from_numpy(np.ascontiguousarray(array.T)).unsqueeze(0).clamp(-1.0, 1.0)
    return {"waveform": waveform, "sample_rate": int(sample_rate)}


def comfy_audio_from_wav(path: Path) -> dict[str, Any]:
    import torchaudio

    waveform, sample_rate = torchaudio.load(str(path))
    return {"waveform": waveform.float().unsqueeze(0).clamp(-1.0, 1.0), "sample_rate": int(sample_rate)}
