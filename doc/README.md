# Comfy-Light-IndexTTS2 Documentation

This plugin wraps the trimmed IndexTTS2 project as native ComfyUI nodes.

## Model Layout

Place the IndexTTS2 files under:

```text
ComfyUI/models/IndexTTS2/
```

The loader expects `config.yaml` in that folder when `config_path` is set to `auto`.

## Nodes

- `加载 IndexTTS2 模型`: creates and caches an IndexTTS2 model handle.
- `IndexTTS2 语音合成`: accepts native ComfyUI `AUDIO` as the speaker reference and returns native ComfyUI `AUDIO`.
