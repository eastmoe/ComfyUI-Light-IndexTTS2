# IndexTTS2 Core Inference

This copy keeps only the core IndexTTS2 inference path:

- CLI synthesis from text and a speaker reference audio file
- optional emotion reference audio
- optional text-guided emotion
- optional target-duration inference
- Python API through `indextts.infer_indextts2.IndexTTS2`

Training, evaluation, test scripts, WebUI/GUI code, old IndexTTS1 inference code, and their dependencies have been removed.

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

Download the model files separately and place them in `checkpoints/`, or pass `--model-dir` and `--config`.

## CLI

```bash
indextts "欢迎大家来体验 IndexTTS2。" \
  --voice reference.wav \
  --output-path gen.wav \
  --model-dir checkpoints \
  --config checkpoints/config.yaml \
  --force
```

Emotion reference audio:

```bash
indextts "今天的心情不错。" \
  --voice speaker.wav \
  --emo-audio emotion.wav \
  --output-path gen.wav \
  --force
```

Text-guided emotion:

```bash
indextts "这一句带一点开心。" \
  --voice speaker.wav \
  --emo-text "开心、轻松" \
  --output-path gen.wav \
  --force
```

Target duration:

```bash
indextts "请合成到指定时长。" \
  --voice speaker.wav \
  --target-dur 7.5 \
  --output-path gen.wav \
  --force
```

## Python API

```python
from indextts.infer_indextts2 import IndexTTS2

tts = IndexTTS2(
    cfg_path="checkpoints/config.yaml",
    model_dir="checkpoints",
    is_fp16=True,
)

tts.infer(
    spk_audio_prompt="reference.wav",
    text="欢迎大家来体验 IndexTTS2。",
    output_path="gen.wav",
    verbose=True,
)
```
