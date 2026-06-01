from __future__ import annotations

CATEGORY = "eastmoe/Comfy-Light-IndexTTS2"
MODEL_TYPE = "COMFY_LIGHT_INDEXTTS2_MODEL"

EMOTION_NAMES = ("happy", "angry", "sad", "fear", "hate", "low", "surprise", "neutral")
EMOTION_MODES = ("speaker_audio", "emotion_audio", "emotion_vector", "emotion_text")
DEVICE_OPTIONS = ("auto", "cuda:0", "cuda:1", "cpu", "mps")
CUDA_KERNEL_OPTIONS = ("auto", "enabled", "disabled")


def ui(name: str, tooltip: str, **kwargs):
    data = {"tooltip": tooltip}
    data.update(kwargs)
    return data
