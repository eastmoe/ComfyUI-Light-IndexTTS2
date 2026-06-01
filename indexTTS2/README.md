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
python infer.py "欢迎大家来体验 IndexTTS2。" \
  --voice reference.wav \
  --output-path gen.wav \
  --model-dir checkpoints \
  --config checkpoints/config.yaml \
  --force
```

Emotion reference audio:

```bash
python infer.py "今天的心情不错。" \
  --voice speaker.wav \
  --emo-audio emotion.wav \
  --emo-alpha 1.0 \
  --output-path gen.wav \
  --force
```

Text-guided emotion:

```bash
python infer.py "这一句带一点开心。" \
  --voice speaker.wav \
  --emo-text "开心、轻松" \
  --output-path gen.wav \
  --force
```

Target duration:

```bash
python infer.py "请合成到指定时长。" \
  --voice speaker.wav \
  --target-dur 7.5 \
  --output-path gen.wav \
  --force
```

Custom generation:

```bash
python infer.py --text-file input.txt \
  --voice speaker.wav \
  --output-path gen.wav \
  --top-p 0.8 \
  --top-k 30 \
  --temperature 0.8 \
  --num-beams 3 \
  --max-mel-tokens 1500 \
  --diffusion-steps 25 \
  --cfg-rate 0.7 \
  --cfm-temperature 1.0 \
  --seed 1234 \
  --force
```

Run `python infer.py --help` to see all options, including `--emo-vector`,
`--use-emo-text`, `--length-scale`, `--do-sample/--no-do-sample`,
`--typical-sampling`, `--device`, `--fp16/--no-fp16`, and
`--cuda-kernel/--no-cuda-kernel`.

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
