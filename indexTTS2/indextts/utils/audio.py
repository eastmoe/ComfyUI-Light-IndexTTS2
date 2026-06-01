from pathlib import Path

import numpy as np
import torch
import torchaudio
from scipy.io import wavfile


def load_audio_mono(path, sr=22050):
    try:
        audio, sample_rate = torchaudio.load(path)
    except ImportError:
        if Path(path).suffix.lower() != ".wav":
            raise
        sample_rate, data = wavfile.read(path)
        if np.issubdtype(data.dtype, np.integer):
            data = data.astype(np.float32) / max(abs(np.iinfo(data.dtype).min), np.iinfo(data.dtype).max)
        else:
            data = data.astype(np.float32)
        if data.ndim == 1:
            data = data[None, :]
        else:
            data = data.T
        audio = torch.from_numpy(data)

    if audio.size(0) > 1:
        audio = audio.mean(dim=0, keepdim=True)

    if sr is not None and sample_rate != sr:
        audio = torchaudio.functional.resample(audio, sample_rate, sr)
        sample_rate = sr

    return audio.squeeze(0).to(torch.float32), sample_rate
