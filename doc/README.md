# Comfy-Light-IndexTTS2 Documentation

This plugin wraps the trimmed IndexTTS2 project as native ComfyUI nodes.

## Model Layout

Place the IndexTTS2 files under:

```text
ComfyUI/models/IndexTTS2/
```

The loader expects `config.yaml` in that folder when `config_path` is set to `auto`.

## Nodes

- `加载 IndexTTS2 模型`: validates local IndexTTS2 files, then creates and caches a model handle with selectable device and compute precision.
- `在线加载 IndexTTS2 模型`: downloads the model from Hugging Face, hf-mirror, or a custom Hugging Face reverse proxy, validates the local files, then loads the model.
- `IndexTTS2 语音合成（说话人音频）`: uses one speaker reference `AUDIO` for both timbre and emotion.
- `IndexTTS2 语音合成（情感音频）`: uses a speaker reference `AUDIO` for timbre and a separate emotion reference `AUDIO` for expression. The speaker reference is optional; when omitted, the emotion reference is also used as the timbre reference.
- `IndexTTS2 语音合成（情感向量）`: exposes the eight IndexTTS2 emotion-vector controls directly.
- `IndexTTS2 语音合成（情感文本）`: derives emotion from text with the Qwen emotion model.

All synthesis nodes return native ComfyUI `AUDIO`, optional wav output path text, and JSON metadata. They can unload the model and clear accelerator memory after generation.
