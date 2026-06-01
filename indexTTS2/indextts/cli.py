import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def _auto_device():
    import torch

    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _parse_emo_vector(value):
    if not value:
        return None
    vector = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(vector) != 8:
        raise argparse.ArgumentTypeError("--emo-vector must contain 8 comma-separated numbers")
    return vector


def main():
    parser = argparse.ArgumentParser(description="IndexTTS2 command line inference")
    parser.add_argument("text", type=str, help="Text to synthesize")
    parser.add_argument("-v", "--voice", type=str, required=True, help="Path to the speaker reference audio")
    parser.add_argument("-o", "--output-path", type=str, default="gen.wav", help="Path to the output wav file")
    parser.add_argument("-c", "--config", type=str, default="checkpoints/config.yaml", help="Path to config.yaml")
    parser.add_argument("--model-dir", type=str, default="checkpoints", help="Path to the model directory")
    parser.add_argument("--device", type=str, default=None, help="Device to run on, for example cpu, cuda:0, or mps")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True, help="Use FP16 outside CPU/MPS")
    parser.add_argument("-f", "--force", action="store_true", default=False, help="Overwrite the output file")

    parser.add_argument("--emo-audio", type=str, default=None, help="Optional emotion reference audio")
    parser.add_argument("--emo-alpha", type=float, default=1.0, help="Blend weight for emotion audio")
    parser.add_argument("--emo-vector", type=_parse_emo_vector, default=None, help="8 emotion weights, comma-separated")
    parser.add_argument("--emo-text", type=str, default=None, help="Use text-guided emotion; defaults to input text")
    parser.add_argument("--emo-text-weight", type=float, default=1.0, help="Scale for text-guided emotion")
    parser.add_argument("--target-dur", type=float, default=None, help="Target duration in seconds")
    parser.add_argument("--max-text-tokens-per-sentence", type=int, default=120)
    parser.add_argument("--verbose", action="store_true", default=False)
    args = parser.parse_args()

    text = args.text.strip()
    if not text:
        print("ERROR: Text is empty.")
        parser.print_help()
        sys.exit(1)

    for path, label in ((args.voice, "Speaker reference audio"), (args.config, "Config file")):
        if not os.path.exists(path):
            print(f"ERROR: {label} does not exist: {path}")
            parser.print_help()
            sys.exit(1)

    if args.emo_audio and not os.path.exists(args.emo_audio):
        print(f"ERROR: Emotion reference audio does not exist: {args.emo_audio}")
        parser.print_help()
        sys.exit(1)

    if os.path.exists(args.output_path):
        if not args.force:
            print(f"ERROR: Output file exists: {args.output_path}. Use --force to overwrite.")
            parser.print_help()
            sys.exit(1)
        os.remove(args.output_path)

    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch is not installed. Please install it first.")
        sys.exit(1)

    device = args.device or _auto_device()
    fp16 = False if device == "cpu" or device == "mps" else args.fp16
    if device == "cpu":
        print("WARNING: Running on CPU may be slow.")

    from indextts.infer_indextts2 import IndexTTS2

    tts = IndexTTS2(cfg_path=args.config, model_dir=args.model_dir, is_fp16=fp16, device=device)
    tts.infer(
        spk_audio_prompt=args.voice,
        text=text,
        output_path=args.output_path,
        emo_audio_prompt=args.emo_audio,
        emo_alpha=args.emo_alpha,
        emo_vector=args.emo_vector,
        use_emo_text=args.emo_text is not None,
        emo_text=args.emo_text,
        emo_text_weight=args.emo_text_weight,
        use_speed=args.target_dur is not None,
        target_dur=args.target_dur,
        verbose=args.verbose,
        max_text_tokens_per_sentence=args.max_text_tokens_per_sentence,
    )


if __name__ == "__main__":
    main()
