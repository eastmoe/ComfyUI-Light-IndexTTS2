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
- `IndexTTS2 语音合成`: accepts native ComfyUI `AUDIO` as the speaker reference and returns native ComfyUI `AUDIO`. It can unload the model and clear accelerator memory after generation.
