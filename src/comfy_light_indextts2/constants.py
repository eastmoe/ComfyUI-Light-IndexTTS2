from __future__ import annotations

CATEGORY = "eastmoe/Comfy-Light-IndexTTS2"
MODEL_TYPE = "COMFY_LIGHT_INDEXTTS2_MODEL"

EMOTION_NAMES = ("happy", "angry", "sad", "fear", "hate", "low", "surprise", "neutral")
DEVICE_OPTIONS = ("auto", "cuda:0", "cuda:1", "xpu:0", "xpu:1", "cpu", "mps")
PRECISION_OPTIONS = ("FP16", "BF16", "FP32", "FP8")
CUDA_KERNEL_OPTIONS = ("auto", "enabled", "disabled")
HF_SOURCE_OPTIONS = ("huggingface", "hf-mirror", "custom")


def ui(name: str, tooltip: str, **kwargs):
    data = {"tooltip": tooltip}
    data.update(kwargs)
    return data
