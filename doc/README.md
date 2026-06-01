# Comfy-Light-IndexTTS2 Documentation

This plugin wraps the trimmed IndexTTS2 project as native ComfyUI nodes.

## Model Layout

The recommended layout keeps the model root as `ComfyUI/models/IndexTTS2` and puts each downloadable model in its own subdirectory:

```text
ComfyUI/models/IndexTTS2/
├── IndexTTS-2/
│   ├── config.yaml
│   ├── gpt.pth
│   ├── s2mel.pth
│   ├── wav2vec2bert_stats.pt
│   ├── bpe.model
│   └── qwen0.6bemo4-merge/
├── w2v-bert-2.0/
├── MaskGCT/
├── campplus/
├── bigvgan_v2_22khz_80band_256x/
├── IndexTTS2-cache/
└── hf_cache/
```

With `model_dir=auto`, the local loader reads `ComfyUI/models/IndexTTS2/IndexTTS-2/config.yaml`. For compatibility, it falls back to the old flat layout `ComfyUI/models/IndexTTS2/config.yaml` when the new subdirectory does not exist.

## Stalled Downloads

The online loader disables Hugging Face Xet downloads by default and uses one `snapshot_download` worker to avoid large-file transfers stalling midway. These defaults can be overridden before starting ComfyUI:

```bash
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=60
export HF_HUB_ETAG_TIMEOUT=30
export INDEXTTS2_HF_DOWNLOAD_WORKERS=1
```

If direct Hugging Face downloads are still unstable, set the node `download_source` to `hf-mirror` and rerun. Existing `.incomplete` files are resumable in normal cases.

## Nodes

- `加载 IndexTTS2 模型`: validates local IndexTTS2 files, then creates and caches a model handle with selectable device, compute precision, and optional FP8/FP4 Linear quantization.
- `在线加载 IndexTTS2 模型`: downloads the main model and runtime dependencies from Hugging Face, hf-mirror, or a custom Hugging Face reverse proxy into separate subdirectories, validates the local files, then loads the model.
- `IndexTTS2 语音合成（说话人音频）`: uses one speaker reference `AUDIO` for both timbre and emotion.
- `IndexTTS2 语音合成（情感音频）`: uses a speaker reference `AUDIO` for timbre and a separate emotion reference `AUDIO` for expression. The speaker reference is optional; when omitted, the emotion reference is also used as the timbre reference.
- `IndexTTS2 语音合成（情感向量）`: exposes the eight IndexTTS2 emotion-vector controls directly.
- `IndexTTS2 语音合成（情感文本）`: derives emotion from text with the Qwen emotion model.

All synthesis nodes return native ComfyUI `AUDIO`, optional wav output path text, and JSON metadata. They can unload the model and clear accelerator memory after generation.
