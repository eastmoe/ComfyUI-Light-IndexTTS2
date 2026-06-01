#!/usr/bin/env python3
import argparse
import os
import random
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


EMOTION_NAMES = ("happy", "angry", "sad", "fear", "hate", "low", "surprise", "neutral")


def auto_device():
    import torch

    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_emo_vector(value):
    if value is None or value.strip() == "":
        return None

    parts = [item.strip() for item in value.replace(";", ",").split(",") if item.strip()]
    if len(parts) == 1:
        parts = [item.strip() for item in value.split() if item.strip()]

    try:
        vector = [float(item) for item in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--emo-vector must contain only numbers") from exc

    if len(vector) != 8:
        raise argparse.ArgumentTypeError(
            "--emo-vector must contain 8 numbers ordered as "
            + ",".join(EMOTION_NAMES)
        )
    return vector


def positive_float(value):
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a number, got {value!r}") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return number


def nonnegative_float(value):
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a number, got {value!r}") from exc
    if number < 0:
        raise argparse.ArgumentTypeError("value must be greater than or equal to 0")
    return number


def positive_int(value):
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return number


def read_text(args, parser):
    sources = [
        args.text is not None,
        args.text_file is not None,
        args.stdin,
    ]
    if sum(sources) != 1:
        parser.error("provide exactly one text source: positional TEXT, --text-file, or --stdin")

    if args.text_file:
        with open(args.text_file, "r", encoding="utf-8") as file:
            text = file.read()
    elif args.stdin:
        text = sys.stdin.read()
    else:
        text = args.text

    text = text.strip()
    if not text:
        parser.error("text is empty")
    return text


def ensure_file(path, label, parser):
    if path and not os.path.isfile(path):
        parser.error(f"{label} does not exist: {path}")


def set_seed(seed):
    if seed is None:
        return

    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_parser():
    parser = argparse.ArgumentParser(
        description="IndexTTS2 high-customization CLI inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    text_group = parser.add_argument_group("Text input")
    text_group.add_argument("text", nargs="?", help="Text to synthesize")
    text_group.add_argument("--text-file", help="Read synthesis text from a UTF-8 text file")
    text_group.add_argument("--stdin", action="store_true", help="Read synthesis text from standard input")

    io_group = parser.add_argument_group("I/O")
    io_group.add_argument("-v", "--voice", required=True, help="Speaker reference audio path")
    io_group.add_argument("-o", "--output-path", default="gen.wav", help="Output wav path")
    io_group.add_argument("-f", "--force", action="store_true", help="Overwrite output when it exists")

    model_group = parser.add_argument_group("Model")
    model_group.add_argument("-c", "--config", default="checkpoints/config.yaml", help="Path to config.yaml")
    model_group.add_argument("--model-dir", default="checkpoints", help="Path to model directory")
    model_group.add_argument("--device", default=None, help="Device, for example cpu, cuda:0, or mps")
    model_group.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True, help="Use FP16 on CUDA")
    model_group.add_argument(
        "--cuda-kernel",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use BigVGAN custom CUDA kernel when available",
    )

    emotion_group = parser.add_argument_group("Emotion control")
    emotion_group.add_argument("--emo-audio", help="Emotion reference audio path")
    emotion_group.add_argument("--emo-alpha", type=float, default=1.0, help="Blend weight for emotion reference audio")
    emotion_group.add_argument(
        "--emo-vector",
        type=parse_emo_vector,
        default=None,
        help="8 emotion weights ordered as happy,angry,sad,fear,hate,low,surprise,neutral",
    )
    emotion_group.add_argument(
        "--use-emo-text",
        action="store_true",
        help="Infer emotion from text; uses --emo-text when provided, otherwise synthesis text",
    )
    emotion_group.add_argument("--emo-text", default=None, help="Text used by Qwen emotion scorer")
    emotion_group.add_argument("--emo-text-weight", type=float, default=1.0, help="Scale for text-guided emotion")

    duration_group = parser.add_argument_group("Duration and text splitting")
    duration_group.add_argument("--target-dur", type=positive_float, default=None, help="Target duration in seconds")
    duration_group.add_argument(
        "--max-text-tokens-per-sentence",
        type=positive_int,
        default=120,
        help="Max text tokens per sentence chunk",
    )
    duration_group.add_argument(
        "--length-scale",
        type=positive_float,
        default=1.72,
        help="Scale from generated semantic token length to acoustic target length",
    )

    generation_group = parser.add_argument_group("GPT generation")
    generation_group.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True, help="Enable sampling")
    generation_group.add_argument("--top-p", type=float, default=0.8, help="Nucleus sampling top_p")
    generation_group.add_argument("--top-k", type=int, default=30, help="Top-k sampling")
    generation_group.add_argument("--temperature", type=positive_float, default=0.8, help="GPT sampling temperature")
    generation_group.add_argument("--num-beams", type=positive_int, default=3, help="Beam count")
    generation_group.add_argument("--length-penalty", type=float, default=0.0, help="Generation length penalty")
    generation_group.add_argument("--repetition-penalty", type=float, default=10.0, help="Generation repetition penalty")
    generation_group.add_argument("--max-mel-tokens", type=positive_int, default=1500, help="Max generated mel tokens")
    generation_group.add_argument("--typical-sampling", action="store_true", help="Enable custom typical sampling")
    generation_group.add_argument("--typical-mass", type=float, default=0.9, help="Typical sampling mass")

    diffusion_group = parser.add_argument_group("S2Mel diffusion")
    diffusion_group.add_argument("--diffusion-steps", type=positive_int, default=25, help="CFM diffusion steps")
    diffusion_group.add_argument("--cfg-rate", type=nonnegative_float, default=0.7, help="Classifier-free guidance rate")
    diffusion_group.add_argument("--cfm-temperature", type=positive_float, default=1.0, help="CFM initial noise temperature")

    runtime_group = parser.add_argument_group("Runtime")
    runtime_group.add_argument("--seed", type=int, default=None, help="Random seed")
    runtime_group.add_argument("--verbose", action="store_true", help="Print detailed inference diagnostics")

    return parser


def validate_args(args, parser):
    ensure_file(args.voice, "speaker reference audio", parser)
    ensure_file(args.config, "config file", parser)
    ensure_file(args.emo_audio, "emotion reference audio", parser)
    ensure_file(args.text_file, "text file", parser)

    if args.emo_text and not args.use_emo_text:
        args.use_emo_text = True

    emotion_modes = [
        args.emo_audio is not None,
        args.emo_vector is not None,
        args.use_emo_text,
    ]
    if sum(emotion_modes) > 1:
        parser.error("--emo-audio, --emo-vector, and --use-emo-text/--emo-text are mutually exclusive")

    if args.emo_alpha != 1.0 and not args.emo_audio:
        parser.error("--emo-alpha can only be used with --emo-audio")

    if not (0.0 < args.top_p <= 1.0):
        parser.error("--top-p must be in (0, 1]")
    if args.top_k < 0:
        parser.error("--top-k must be greater than or equal to 0")
    if not (0.0 < args.typical_mass < 1.0):
        parser.error("--typical-mass must be in (0, 1)")

    if os.path.exists(args.output_path):
        if not args.force:
            parser.error(f"output file exists: {args.output_path}. Use --force to overwrite")
        os.remove(args.output_path)


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)
    text = read_text(args, parser)

    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch is not installed. Please install requirements first.", file=sys.stderr)
        sys.exit(1)

    set_seed(args.seed)

    device = args.device or auto_device()
    fp16 = args.fp16 and str(device).startswith("cuda")
    if device == "cpu":
        print("WARNING: Running on CPU may be slow.", file=sys.stderr)
    elif args.fp16 and not fp16:
        print(f"INFO: FP16 disabled on device {device}.", file=sys.stderr)

    from indextts.infer_indextts2 import IndexTTS2

    tts = IndexTTS2(
        cfg_path=args.config,
        model_dir=args.model_dir,
        is_fp16=fp16,
        device=device,
        use_cuda_kernel=args.cuda_kernel,
    )
    tts.infer(
        spk_audio_prompt=args.voice,
        text=text,
        output_path=args.output_path,
        emo_audio_prompt=args.emo_audio,
        emo_alpha=args.emo_alpha,
        emo_vector=args.emo_vector,
        use_emo_text=args.use_emo_text,
        emo_text=args.emo_text,
        emo_text_weight=args.emo_text_weight,
        use_speed=args.target_dur is not None,
        target_dur=args.target_dur,
        verbose=args.verbose,
        max_text_tokens_per_sentence=args.max_text_tokens_per_sentence,
        do_sample=args.do_sample,
        top_p=args.top_p,
        top_k=args.top_k,
        temperature=args.temperature,
        num_beams=args.num_beams,
        length_penalty=args.length_penalty,
        repetition_penalty=args.repetition_penalty,
        max_mel_tokens=args.max_mel_tokens,
        typical_sampling=args.typical_sampling,
        typical_mass=args.typical_mass,
        diffusion_steps=args.diffusion_steps,
        inference_cfg_rate=args.cfg_rate,
        cfm_temperature=args.cfm_temperature,
        length_scale=args.length_scale,
    )


if __name__ == "__main__":
    main()
