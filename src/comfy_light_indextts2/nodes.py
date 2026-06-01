from __future__ import annotations

import json
import time
from pathlib import Path

from .audio import audio_to_temp_wav, comfy_audio_from_infer_result, comfy_audio_from_wav
from .constants import CATEGORY, CUDA_KERNEL_OPTIONS, DEVICE_OPTIONS, EMOTION_MODES, MODEL_TYPE, ui
from .paths import default_model_dir, resolve_output_path
from .runtime import IndexTTS2ModelHandle, cleanup_torch_memory, load_model_handle, set_seed


class ComfyLightIndexTTS2LoadModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_dir": (
                    "STRING",
                    ui(
                        "模型目录",
                        "IndexTTS2 权重目录。auto 使用 ComfyUI/models/IndexTTS2。",
                        default="auto",
                    ),
                ),
                "config_path": (
                    "STRING",
                    ui(
                        "配置文件",
                        "config.yaml 路径。auto 使用模型目录下的 config.yaml。",
                        default="auto",
                    ),
                ),
                "device": (
                    DEVICE_OPTIONS,
                    ui("设备", "auto 会优先使用 CUDA，其次 MPS，最后 CPU。"),
                ),
                "fp16": (
                    "BOOLEAN",
                    ui("FP16", "在 CUDA 上使用半精度推理；CPU/MPS 会自动关闭。", default=True),
                ),
                "cuda_kernel": (
                    CUDA_KERNEL_OPTIONS,
                    ui("BigVGAN CUDA 内核", "auto 使用 IndexTTS2 默认策略；仅 CUDA 可用。"),
                ),
                "reload_model": (
                    "BOOLEAN",
                    ui("重新加载", "忽略缓存并重新实例化模型。", default=False),
                ),
            },
        }

    RETURN_TYPES = (MODEL_TYPE, "STRING")
    RETURN_NAMES = ("模型", "信息")
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = "加载 IndexTTS2 模型，默认从 ComfyUI/models/IndexTTS2 读取权重。"

    def load(
        self,
        model_dir: str = "auto",
        config_path: str = "auto",
        device: str = "auto",
        fp16: bool = True,
        cuda_kernel: str = "auto",
        reload_model: bool = False,
    ):
        handle = load_model_handle(
            model_dir=model_dir,
            config_path=config_path,
            device=device,
            fp16=bool(fp16),
            cuda_kernel=cuda_kernel,
            reload_model=bool(reload_model),
        )
        info = {
            "model_dir": str(handle.model_dir),
            "config_path": str(handle.config_path),
            "device": handle.device,
            "fp16": handle.fp16,
            "cuda_kernel": handle.cuda_kernel,
            "default_model_dir": str(default_model_dir()),
        }
        return (handle, json.dumps(info, ensure_ascii=False, indent=2))


class ComfyLightIndexTTS2Infer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "indextts2_model": (MODEL_TYPE, ui("IndexTTS2 模型", "模型加载节点输出的模型句柄。")),
                "speaker_audio": ("AUDIO", ui("说话人参考音频", "用于克隆音色的 ComfyUI AUDIO。")),
                "text": (
                    "STRING",
                    ui(
                        "合成文本",
                        "需要合成的文本。text_file_path 为空时使用这里的内容。",
                        default="你好，这是一段 IndexTTS2 语音合成测试。",
                        multiline=True,
                    ),
                ),
                "text_file_path": (
                    "STRING",
                    ui("文本文件", "可选 UTF-8 文本文件路径；非空时覆盖合成文本。", default=""),
                ),
                "emotion_mode": (
                    EMOTION_MODES,
                    ui(
                        "情感模式",
                        "speaker_audio 使用说话人参考音频；emotion_audio 使用情感参考音频；emotion_vector 使用八维情感向量；emotion_text 使用文本情感分析。",
                    ),
                ),
                "emo_alpha": (
                    "FLOAT",
                    ui("情感音频权重", "仅 emotion_audio 模式有效，对情感参考音频进行混合。", default=1.0, min=0.0, max=2.0, step=0.05),
                ),
                "emo_text": (
                    "STRING",
                    ui("情感文本", "emotion_text 模式使用的情感文本；为空时使用合成文本。", default="", multiline=True),
                ),
                "emo_text_weight": (
                    "FLOAT",
                    ui("文本情感权重", "对 Qwen 情感评分结果进行缩放。", default=1.0, min=0.0, max=2.0, step=0.05),
                ),
                "emo_happy": ("FLOAT", ui("喜", "emotion_vector 模式：happy 权重。", default=0.0, min=0.0, max=1.2, step=0.01)),
                "emo_angry": ("FLOAT", ui("怒", "emotion_vector 模式：angry 权重。", default=0.0, min=0.0, max=1.2, step=0.01)),
                "emo_sad": ("FLOAT", ui("哀", "emotion_vector 模式：sad 权重。", default=0.0, min=0.0, max=1.2, step=0.01)),
                "emo_fear": ("FLOAT", ui("惧", "emotion_vector 模式：fear 权重。", default=0.0, min=0.0, max=1.2, step=0.01)),
                "emo_hate": ("FLOAT", ui("厌恶", "emotion_vector 模式：hate 权重。", default=0.0, min=0.0, max=1.2, step=0.01)),
                "emo_low": ("FLOAT", ui("低落", "emotion_vector 模式：low 权重。", default=0.0, min=0.0, max=1.2, step=0.01)),
                "emo_surprise": ("FLOAT", ui("惊喜", "emotion_vector 模式：surprise 权重。", default=0.0, min=0.0, max=1.2, step=0.01)),
                "emo_neutral": ("FLOAT", ui("自然", "emotion_vector 模式：neutral 权重。", default=1.0, min=0.0, max=1.2, step=0.01)),
                "target_duration_seconds": (
                    "FLOAT",
                    ui("目标时长", "大于 0 时启用 infer.py 的 target_dur/use_speed 控制。", default=0.0, min=0.0, max=3600.0, step=0.1),
                ),
                "max_text_tokens_per_sentence": (
                    "INT",
                    ui("单句最大文本 Token", "长文本按此上限切分。", default=120, min=1, max=4096, step=1),
                ),
                "length_scale": (
                    "FLOAT",
                    ui("长度比例", "语义 token 长度到声学目标长度的比例。", default=1.72, min=0.01, max=10.0, step=0.01),
                ),
                "do_sample": ("BOOLEAN", ui("启用采样", "开启 GPT 采样；关闭时更偏确定性。", default=True)),
                "top_p": ("FLOAT", ui("Top P", "核采样 top_p，范围 (0, 1]。", default=0.8, min=0.01, max=1.0, step=0.01)),
                "top_k": ("INT", ui("Top K", "Top-k 采样；0 表示不限制。", default=30, min=0, max=4096, step=1)),
                "temperature": ("FLOAT", ui("温度", "GPT 采样温度。", default=0.8, min=0.01, max=5.0, step=0.01)),
                "num_beams": ("INT", ui("Beam 数", "束搜索数量。", default=3, min=1, max=64, step=1)),
                "length_penalty": ("FLOAT", ui("长度惩罚", "生成长度惩罚。", default=0.0, min=-10.0, max=10.0, step=0.1)),
                "repetition_penalty": (
                    "FLOAT",
                    ui("重复惩罚", "生成重复惩罚。", default=10.0, min=0.0, max=50.0, step=0.1),
                ),
                "max_mel_tokens": ("INT", ui("最大 Mel Token", "生成 mel token 的最大数量。", default=1500, min=1, max=8192, step=1)),
                "typical_sampling": ("BOOLEAN", ui("Typical Sampling", "启用 infer.py 的 typical sampling 参数。", default=False)),
                "typical_mass": ("FLOAT", ui("Typical Mass", "Typical sampling mass，范围 (0, 1)。", default=0.9, min=0.01, max=0.99, step=0.01)),
                "diffusion_steps": ("INT", ui("扩散步数", "S2Mel CFM 扩散步数。", default=25, min=1, max=256, step=1)),
                "cfg_rate": (
                    "FLOAT",
                    ui("CFG Rate", "Classifier-free guidance rate。", default=0.7, min=0.0, max=10.0, step=0.05),
                ),
                "cfm_temperature": (
                    "FLOAT",
                    ui("CFM 温度", "CFM 初始噪声温度。", default=1.0, min=0.01, max=5.0, step=0.01),
                ),
                "seed": ("INT", ui("随机种子", "-1 表示不固定随机种子。", default=-1, min=-1, max=0x7FFFFFFF, step=1)),
                "speaker_batch_index": (
                    "INT",
                    ui("说话人音频批次", "当 AUDIO 含多个 batch 时选择参考音频批次。", default=0, min=0, max=4096, step=1),
                ),
                "emotion_batch_index": (
                    "INT",
                    ui("情感音频批次", "当情感 AUDIO 含多个 batch 时选择参考音频批次。", default=0, min=0, max=4096, step=1),
                ),
                "output_path": (
                    "STRING",
                    ui("输出路径", "可选 wav 输出路径。相对路径写入 ComfyUI 输出目录；为空则只返回 AUDIO。", default=""),
                ),
                "overwrite": ("BOOLEAN", ui("覆盖输出", "输出路径存在时是否覆盖。", default=True)),
                "verbose": ("BOOLEAN", ui("详细日志", "打印 infer.py 的详细推理诊断。", default=False)),
            },
            "optional": {
                "emotion_audio": ("AUDIO", ui("情感参考音频", "emotion_audio 模式使用的 ComfyUI AUDIO。")),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("音频", "输出路径", "信息")
    FUNCTION = "infer"
    CATEGORY = CATEGORY
    DESCRIPTION = "使用 IndexTTS2 从文本和 ComfyUI AUDIO 参考音频生成语音，覆盖 infer.py 的推理参数。"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        if int(kwargs.get("seed", -1)) < 0:
            return time.time()
        return None

    def infer(
        self,
        indextts2_model,
        speaker_audio,
        text: str,
        text_file_path: str = "",
        emotion_mode: str = "speaker_audio",
        emo_alpha: float = 1.0,
        emo_text: str = "",
        emo_text_weight: float = 1.0,
        emo_happy: float = 0.0,
        emo_angry: float = 0.0,
        emo_sad: float = 0.0,
        emo_fear: float = 0.0,
        emo_hate: float = 0.0,
        emo_low: float = 0.0,
        emo_surprise: float = 0.0,
        emo_neutral: float = 1.0,
        target_duration_seconds: float = 0.0,
        max_text_tokens_per_sentence: int = 120,
        length_scale: float = 1.72,
        do_sample: bool = True,
        top_p: float = 0.8,
        top_k: int = 30,
        temperature: float = 0.8,
        num_beams: int = 3,
        length_penalty: float = 0.0,
        repetition_penalty: float = 10.0,
        max_mel_tokens: int = 1500,
        typical_sampling: bool = False,
        typical_mass: float = 0.9,
        diffusion_steps: int = 25,
        cfg_rate: float = 0.7,
        cfm_temperature: float = 1.0,
        seed: int = -1,
        speaker_batch_index: int = 0,
        emotion_batch_index: int = 0,
        output_path: str = "",
        overwrite: bool = True,
        verbose: bool = False,
        emotion_audio=None,
    ):
        if not isinstance(indextts2_model, IndexTTS2ModelHandle):
            raise TypeError("IndexTTS2 model handle is invalid. Please connect the model loader node.")

        synth_text = self._read_text(text, text_file_path)
        self._validate_generation(top_p, top_k, temperature, typical_mass, diffusion_steps, cfg_rate, cfm_temperature, length_scale)
        set_seed(int(seed))
        cleanup_torch_memory()

        speaker_path = audio_to_temp_wav(speaker_audio, "speaker", int(speaker_batch_index))
        emo_audio_prompt = None
        emo_vector = None
        use_emo_text = False
        effective_emo_alpha = 1.0
        effective_emo_text = None

        if emotion_mode == "emotion_audio":
            if emotion_audio is None:
                raise ValueError("emotion_audio mode requires the optional emotion_audio input.")
            emo_audio_prompt = audio_to_temp_wav(emotion_audio, "emotion", int(emotion_batch_index))
            effective_emo_alpha = float(emo_alpha)
        elif emotion_mode == "emotion_vector":
            emo_vector = [
                float(emo_happy),
                float(emo_angry),
                float(emo_sad),
                float(emo_fear),
                float(emo_hate),
                float(emo_low),
                float(emo_surprise),
                float(emo_neutral),
            ]
        elif emotion_mode == "emotion_text":
            use_emo_text = True
            effective_emo_text = (emo_text or "").strip() or None
        elif emotion_mode != "speaker_audio":
            raise ValueError(f"Unsupported emotion_mode: {emotion_mode}")

        resolved_output = resolve_output_path(output_path)
        if resolved_output is not None and resolved_output.is_file() and not bool(overwrite):
            raise FileExistsError(f"Output file exists and overwrite is disabled: {resolved_output}")

        use_speed = float(target_duration_seconds) > 0.0
        result = indextts2_model.model.infer(
            spk_audio_prompt=str(speaker_path),
            text=synth_text,
            output_path=str(resolved_output) if resolved_output is not None else None,
            emo_audio_prompt=str(emo_audio_prompt) if emo_audio_prompt is not None else None,
            emo_alpha=effective_emo_alpha,
            emo_vector=emo_vector,
            use_emo_text=use_emo_text,
            emo_text=effective_emo_text,
            emo_text_weight=float(emo_text_weight),
            use_speed=use_speed,
            target_dur=float(target_duration_seconds) if use_speed else None,
            verbose=bool(verbose),
            max_text_tokens_per_sentence=int(max_text_tokens_per_sentence),
            do_sample=bool(do_sample),
            top_p=float(top_p),
            top_k=int(top_k),
            temperature=float(temperature),
            num_beams=int(num_beams),
            length_penalty=float(length_penalty),
            repetition_penalty=float(repetition_penalty),
            max_mel_tokens=int(max_mel_tokens),
            typical_sampling=bool(typical_sampling),
            typical_mass=float(typical_mass),
            diffusion_steps=int(diffusion_steps),
            inference_cfg_rate=float(cfg_rate),
            cfm_temperature=float(cfm_temperature),
            length_scale=float(length_scale),
        )

        if resolved_output is not None:
            audio = comfy_audio_from_wav(Path(result))
            output_text = str(result)
        else:
            audio = comfy_audio_from_infer_result(result)
            output_text = ""

        info = {
            "text_length": len(synth_text),
            "speaker_audio_path": str(speaker_path),
            "emotion_mode": emotion_mode,
            "emotion_audio_path": str(emo_audio_prompt) if emo_audio_prompt is not None else "",
            "output_path": output_text,
            "sample_rate": audio["sample_rate"],
            "samples": int(audio["waveform"].shape[-1]),
            "duration_seconds": int(audio["waveform"].shape[-1]) / float(audio["sample_rate"]),
            "seed": int(seed),
        }
        cleanup_torch_memory()
        return (audio, output_text, json.dumps(info, ensure_ascii=False, indent=2))

    @staticmethod
    def _read_text(text: str, text_file_path: str) -> str:
        path_text = (text_file_path or "").strip()
        if path_text:
            path = Path(path_text).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"Text file not found: {path}")
            value = path.read_text(encoding="utf-8")
        else:
            value = text or ""
        value = value.strip()
        if not value:
            raise ValueError("Synthesis text is empty.")
        return value

    @staticmethod
    def _validate_generation(
        top_p: float,
        top_k: int,
        temperature: float,
        typical_mass: float,
        diffusion_steps: int,
        cfg_rate: float,
        cfm_temperature: float,
        length_scale: float,
    ) -> None:
        if not (0.0 < float(top_p) <= 1.0):
            raise ValueError("top_p must be in (0, 1].")
        if int(top_k) < 0:
            raise ValueError("top_k must be greater than or equal to 0.")
        if float(temperature) <= 0.0:
            raise ValueError("temperature must be greater than 0.")
        if not (0.0 < float(typical_mass) < 1.0):
            raise ValueError("typical_mass must be in (0, 1).")
        if int(diffusion_steps) <= 0:
            raise ValueError("diffusion_steps must be greater than 0.")
        if float(cfg_rate) < 0.0:
            raise ValueError("cfg_rate must be greater than or equal to 0.")
        if float(cfm_temperature) <= 0.0:
            raise ValueError("cfm_temperature must be greater than 0.")
        if float(length_scale) <= 0.0:
            raise ValueError("length_scale must be greater than 0.")


NODE_CLASS_MAPPINGS = {
    "ComfyLightIndexTTS2LoadModel": ComfyLightIndexTTS2LoadModel,
    "ComfyLightIndexTTS2Infer": ComfyLightIndexTTS2Infer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyLightIndexTTS2LoadModel": "加载 IndexTTS2 模型",
    "ComfyLightIndexTTS2Infer": "IndexTTS2 语音合成",
}
