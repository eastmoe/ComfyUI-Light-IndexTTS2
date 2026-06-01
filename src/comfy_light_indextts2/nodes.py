from __future__ import annotations

import json
import time
from pathlib import Path

from .audio import audio_to_temp_wav, comfy_audio_from_infer_result, comfy_audio_from_wav
from .constants import (
    CATEGORY,
    CUDA_KERNEL_OPTIONS,
    DEVICE_OPTIONS,
    HF_SOURCE_OPTIONS,
    MODEL_TYPE,
    PRECISION_OPTIONS,
    QUANTIZATION_OPTIONS,
    ui,
)
from .paths import default_model_dir, default_model_root, resolve_output_path
from .runtime import (
    IndexTTS2ModelHandle,
    clear_model_runtime_state,
    cleanup_torch_memory,
    download_model_snapshot,
    download_runtime_dependencies,
    load_model_handle,
    set_seed,
    unload_model_handle,
    validate_model_files,
)


class ComfyLightIndexTTS2LoadModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_dir": (
                    "STRING",
                    ui(
                        "模型目录",
                        "IndexTTS2 主模型目录。auto 使用 ComfyUI/models/IndexTTS2/IndexTTS-2，并兼容旧的 ComfyUI/models/IndexTTS2 平铺目录。",
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
                    ui("计算设备", "auto 会优先使用 CUDA，其次 Intel XPU、MPS，最后 CPU。"),
                ),
                "precision": (
                    PRECISION_OPTIONS,
                    ui("计算精度", "选择模型推理计算精度。FP8 依赖当前 PyTorch 和设备支持。"),
                ),
                "cuda_kernel": (
                    CUDA_KERNEL_OPTIONS,
                    ui("BigVGAN CUDA 内核", "auto 使用 IndexTTS2 默认策略；仅 CUDA 可用。"),
                ),
                "quantization": (
                    QUANTIZATION_OPTIONS,
                    ui("线性层量化", "对模型中的 Linear 层启用 weight-only 量化以降低显存占用。"),
                ),
                "use_quantization_cache": (
                    "BOOLEAN",
                    ui("量化缓存", "将 FP8/FP4 量化后的 Linear 权重保存到 IndexTTS2-cache 并复用。", default=True),
                ),
                "rebuild_quantization_cache": (
                    "BOOLEAN",
                    ui("重建量化缓存", "忽略已有 IndexTTS2-cache 量化文件并重新生成。", default=False),
                ),
                "quantized_linear_chunk_size": (
                    "INT",
                    ui("量化分块行数", "大型量化 Linear 层每块反量化的行数。设为 0 表示禁用分块。", default=1024, min=0, max=65536, step=64),
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
    DESCRIPTION = "加载 IndexTTS2 模型，默认从 ComfyUI/models/IndexTTS2/IndexTTS-2 读取权重。"

    def load(
        self,
        model_dir: str = "auto",
        config_path: str = "auto",
        device: str = "auto",
        precision: str = "FP16",
        cuda_kernel: str = "auto",
        quantization: str = "none",
        use_quantization_cache: bool = True,
        rebuild_quantization_cache: bool = False,
        quantized_linear_chunk_size: int = 1024,
        reload_model: bool = False,
        fp16: bool | None = None,
    ):
        if fp16 is not None:
            precision = "FP16" if bool(fp16) else "FP32"
        handle = load_model_handle(
            model_dir=model_dir,
            config_path=config_path,
            device=device,
            precision=precision,
            cuda_kernel=cuda_kernel,
            reload_model=bool(reload_model),
            quantization=quantization,
            use_quantization_cache=bool(use_quantization_cache),
            rebuild_quantization_cache=bool(rebuild_quantization_cache),
            quantized_linear_chunk_size=int(quantized_linear_chunk_size),
        )
        info = {
            "model_dir": str(handle.model_dir),
            "config_path": str(handle.config_path),
            "device": handle.device,
            "precision": handle.precision,
            "cuda_kernel": handle.cuda_kernel,
            "quantization": handle.quantization,
            "quantization_info": handle.quantization_info,
            "default_model_root": str(default_model_root()),
            "default_model_dir": str(default_model_dir()),
        }
        return (handle, json.dumps(info, ensure_ascii=False, indent=2))


class ComfyLightIndexTTS2OnlineLoadModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "repo_id": (
                    "STRING",
                    ui("模型仓库", "Hugging Face 模型仓库 ID。默认下载 IndexTeam/IndexTTS-2。", default="IndexTeam/IndexTTS-2"),
                ),
                "revision": (
                    "STRING",
                    ui("版本", "分支、标签或提交哈希。", default="main"),
                ),
                "local_dir": (
                    "STRING",
                    ui("本地模型根目录", "模型下载根目录。auto 使用 ComfyUI/models/IndexTTS2，并在其中为主模型和依赖模型创建子目录。", default="auto"),
                ),
                "config_path": (
                    "STRING",
                    ui("配置文件", "config.yaml 路径。auto 使用主模型子目录下的 config.yaml。", default="auto"),
                ),
                "download_source": (
                    HF_SOURCE_OPTIONS,
                    ui("下载源", "选择 huggingface、hf-mirror 或自定义 Hugging Face 反向代理。"),
                ),
                "custom_endpoint": (
                    "STRING",
                    ui("自定义端点", "自定义反向代理地址，可填写 host name、IP address 或完整 URL。", default=""),
                ),
                "disable_ssl_verification": (
                    "BOOLEAN",
                    ui("关闭 SSL 证书验证", "跳过 HTTPS 证书校验；仅在自签名代理或受控内网中使用。", default=False),
                ),
                "device": (
                    DEVICE_OPTIONS,
                    ui("计算设备", "auto 会优先使用 CUDA，其次 Intel XPU、MPS，最后 CPU。"),
                ),
                "precision": (
                    PRECISION_OPTIONS,
                    ui("计算精度", "选择模型推理计算精度。FP8 依赖当前 PyTorch 和设备支持。"),
                ),
                "cuda_kernel": (
                    CUDA_KERNEL_OPTIONS,
                    ui("BigVGAN CUDA 内核", "auto 使用 IndexTTS2 默认策略；仅 CUDA 可用。"),
                ),
                "quantization": (
                    QUANTIZATION_OPTIONS,
                    ui("线性层量化", "对模型中的 Linear 层启用 weight-only 量化以降低显存占用。"),
                ),
                "use_quantization_cache": (
                    "BOOLEAN",
                    ui("量化缓存", "将 FP8/FP4 量化后的 Linear 权重保存到 IndexTTS2-cache 并复用。", default=True),
                ),
                "rebuild_quantization_cache": (
                    "BOOLEAN",
                    ui("重建量化缓存", "忽略已有 IndexTTS2-cache 量化文件并重新生成。", default=False),
                ),
                "quantized_linear_chunk_size": (
                    "INT",
                    ui("量化分块行数", "大型量化 Linear 层每块反量化的行数。设为 0 表示禁用分块。", default=1024, min=0, max=65536, step=64),
                ),
                "force_download": (
                    "BOOLEAN",
                    ui("强制重新下载", "忽略已有快照文件并重新从远端下载。", default=False),
                ),
                "reload_model": (
                    "BOOLEAN",
                    ui("重新加载", "忽略模型缓存并重新实例化。", default=False),
                ),
            },
            "optional": {
                "allow_patterns": (
                    "STRING",
                    ui("下载白名单", "逗号分隔的文件匹配规则；为空表示下载仓库默认内容。", default=""),
                ),
                "ignore_patterns": (
                    "STRING",
                    ui("下载黑名单", "逗号分隔的排除匹配规则。", default=""),
                ),
            },
        }

    RETURN_TYPES = (MODEL_TYPE, "STRING")
    RETURN_NAMES = ("模型", "信息")
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = "从 Hugging Face、hf-mirror 或自定义反向代理下载 IndexTTS2 主模型与依赖模型到独立子目录并加载。"

    def load(
        self,
        repo_id: str = "IndexTeam/IndexTTS-2",
        revision: str = "main",
        local_dir: str = "auto",
        config_path: str = "auto",
        download_source: str = "huggingface",
        custom_endpoint: str = "",
        disable_ssl_verification: bool = False,
        device: str = "auto",
        precision: str = "FP16",
        cuda_kernel: str = "auto",
        quantization: str = "none",
        use_quantization_cache: bool = True,
        rebuild_quantization_cache: bool = False,
        quantized_linear_chunk_size: int = 1024,
        force_download: bool = False,
        reload_model: bool = False,
        allow_patterns: str = "",
        ignore_patterns: str = "",
    ):
        model_dir, endpoint = download_model_snapshot(
            repo_id=repo_id,
            local_dir=local_dir,
            revision=revision,
            source=download_source,
            custom_endpoint=custom_endpoint,
            disable_ssl_verification=bool(disable_ssl_verification),
            force_download=bool(force_download),
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            config_path=config_path,
        )
        resolved_config = model_dir / "config.yaml" if (config_path or "auto").strip().lower() == "auto" else Path(config_path).expanduser()
        if not resolved_config.is_absolute():
            resolved_config = model_dir / resolved_config
        checked = validate_model_files(model_dir, resolved_config.resolve())
        dependencies = download_runtime_dependencies(
            model_dir,
            resolved_config.resolve(),
            source=download_source,
            custom_endpoint=custom_endpoint,
            disable_ssl_verification=bool(disable_ssl_verification),
            force_download=bool(force_download),
        )
        handle = load_model_handle(
            model_dir=str(model_dir),
            config_path=str(resolved_config),
            device=device,
            precision=precision,
            cuda_kernel=cuda_kernel,
            reload_model=bool(reload_model),
            hf_source=download_source,
            custom_endpoint=custom_endpoint,
            disable_ssl_verification=bool(disable_ssl_verification),
            quantization=quantization,
            use_quantization_cache=bool(use_quantization_cache),
            rebuild_quantization_cache=bool(rebuild_quantization_cache),
            quantized_linear_chunk_size=int(quantized_linear_chunk_size),
        )
        info = {
            "repo_id": repo_id,
            "revision": revision,
            "endpoint": endpoint,
            "model_dir": str(handle.model_dir),
            "config_path": str(handle.config_path),
            "validated_items": checked,
            "downloaded_dependencies": dependencies,
            "device": handle.device,
            "precision": handle.precision,
            "cuda_kernel": handle.cuda_kernel,
            "quantization": handle.quantization,
            "quantization_info": handle.quantization_info,
            "ssl_verification": not bool(disable_ssl_verification),
        }
        return (handle, json.dumps(info, ensure_ascii=False, indent=2))


def common_infer_inputs() -> dict:
    return {
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
        "num_beams": ("INT", ui("束搜索数量", "Beam search 的候选束数量。", default=3, min=1, max=64, step=1)),
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
            ui("CFG 强度", "Classifier-free guidance 强度。", default=0.7, min=0.0, max=10.0, step=0.05),
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
        "output_path": (
            "STRING",
            ui("输出路径", "可选 wav 输出路径。相对路径写入 ComfyUI 输出目录；为空则只返回 AUDIO。", default=""),
        ),
        "overwrite": ("BOOLEAN", ui("覆盖输出", "输出路径存在时是否覆盖。", default=True)),
        "verbose": ("BOOLEAN", ui("详细日志", "打印 infer.py 的详细推理诊断。", default=False)),
        "unload_model_after_generation": (
            "BOOLEAN",
            ui("生成后卸载模型", "生成结束后从插件缓存中卸载模型并清理显存缓存。", default=False),
        ),
    }


class _ComfyLightIndexTTS2InferBase:
    RETURN_TYPES = ("AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("音频", "输出路径", "信息")
    FUNCTION = "infer"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        if int(kwargs.get("seed", -1)) < 0:
            return time.time()
        return None

    @classmethod
    def base_input_types(cls):
        return {
            "required": common_infer_inputs(),
        }

    def _infer(
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
        unload_model_after_generation: bool = False,
        emotion_audio=None,
    ):
        if not isinstance(indextts2_model, IndexTTS2ModelHandle):
            raise TypeError("IndexTTS2 model handle is invalid. Please connect the model loader node.")
        if indextts2_model.model is None:
            raise RuntimeError("IndexTTS2 model has been unloaded. Please run the model loader node again.")

        synth_text = self._read_text(text, text_file_path)
        self._validate_generation(top_p, top_k, temperature, typical_mass, diffusion_steps, cfg_rate, cfm_temperature, length_scale)
        set_seed(int(seed))
        cleanup_torch_memory()

        emo_audio_prompt = None
        emo_vector = None
        use_emo_text = False
        effective_emo_alpha = 1.0
        effective_emo_text = None

        if emotion_mode == "emotion_audio":
            if emotion_audio is None:
                raise ValueError("emotion_audio mode requires the emotion_audio input.")
            emo_audio_prompt = audio_to_temp_wav(emotion_audio, "emotion", int(emotion_batch_index))
            speaker_path = (
                audio_to_temp_wav(speaker_audio, "speaker", int(speaker_batch_index))
                if speaker_audio is not None
                else emo_audio_prompt
            )
            effective_emo_alpha = float(emo_alpha)
        elif emotion_mode == "emotion_vector":
            if speaker_audio is None:
                raise ValueError("emotion_vector mode requires the speaker_audio input.")
            speaker_path = audio_to_temp_wav(speaker_audio, "speaker", int(speaker_batch_index))
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
            if speaker_audio is None:
                raise ValueError("emotion_text mode requires the speaker_audio input.")
            speaker_path = audio_to_temp_wav(speaker_audio, "speaker", int(speaker_batch_index))
            use_emo_text = True
            effective_emo_text = (emo_text or "").strip() or None
        elif emotion_mode != "speaker_audio":
            raise ValueError(f"Unsupported emotion_mode: {emotion_mode}")
        else:
            if speaker_audio is None:
                raise ValueError("speaker_audio mode requires the speaker_audio input.")
            speaker_path = audio_to_temp_wav(speaker_audio, "speaker", int(speaker_batch_index))

        resolved_output = resolve_output_path(output_path)
        if resolved_output is not None and resolved_output.is_file() and not bool(overwrite):
            raise FileExistsError(f"Output file exists and overwrite is disabled: {resolved_output}")

        use_speed = float(target_duration_seconds) > 0.0
        inference_signature = (
            synth_text,
            str(speaker_path),
            str(emo_audio_prompt) if emo_audio_prompt is not None else "",
            emotion_mode,
            float(emo_alpha),
            tuple(float(value) for value in emo_vector) if emo_vector is not None else None,
            bool(use_emo_text),
            effective_emo_text,
            float(emo_text_weight),
            use_speed,
            float(target_duration_seconds) if use_speed else None,
            int(max_text_tokens_per_sentence),
            float(length_scale),
            bool(do_sample),
            float(top_p),
            int(top_k),
            float(temperature),
            int(num_beams),
            float(length_penalty),
            float(repetition_penalty),
            int(max_mel_tokens),
            bool(typical_sampling),
            float(typical_mass),
            int(diffusion_steps),
            float(cfg_rate),
            float(cfm_temperature),
            int(seed),
        )
        if indextts2_model.inference_signature != inference_signature:
            clear_model_runtime_state(indextts2_model.model)
            indextts2_model.inference_signature = inference_signature

        try:
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
                "model_unloaded": bool(unload_model_after_generation),
            }
            return (audio, output_text, json.dumps(info, ensure_ascii=False, indent=2))
        finally:
            if bool(unload_model_after_generation):
                unload_model_handle(indextts2_model)
            else:
                cleanup_torch_memory()

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


class ComfyLightIndexTTS2InferSpeaker(_ComfyLightIndexTTS2InferBase):
    DESCRIPTION = "使用说话人参考音频同时控制音色和情感，生成语音。"

    @classmethod
    def INPUT_TYPES(cls):
        return cls.base_input_types()

    def infer(self, **kwargs):
        return self._infer(emotion_mode="speaker_audio", **kwargs)


class ComfyLightIndexTTS2InferEmotionAudio(_ComfyLightIndexTTS2InferBase):
    DESCRIPTION = "使用独立情感参考音频控制情感表达，生成语音。"

    @classmethod
    def INPUT_TYPES(cls):
        inputs = cls.base_input_types()
        required = inputs["required"]
        speaker_audio = required.pop("speaker_audio")
        emotion_inputs = {
            "emotion_audio": ("AUDIO", ui("情感参考音频", "用于控制情感表达的 ComfyUI AUDIO。")),
            "emo_alpha": (
                "FLOAT",
                ui("情感音频权重", "对情感参考音频进行混合。", default=1.0, min=0.0, max=2.0, step=0.05),
            ),
            "emotion_batch_index": (
                "INT",
                ui("情感音频批次", "当情感 AUDIO 含多个 batch 时选择参考音频批次。", default=0, min=0, max=4096, step=1),
            ),
        }
        for key, value in reversed(emotion_inputs.items()):
            required = dict(list(required.items())[:1] + [(key, value)] + list(required.items())[1:])
        inputs["required"] = required
        inputs["optional"] = {
            "speaker_audio": (
                speaker_audio[0],
                ui("说话人参考音频", "可选；连接后用于控制音色，未连接时使用情感参考音频兼作音色参考。"),
            )
        }
        return inputs

    def infer(self, emotion_audio, emo_alpha: float = 1.0, emotion_batch_index: int = 0, **kwargs):
        return self._infer(
            emotion_mode="emotion_audio",
            emotion_audio=emotion_audio,
            emo_alpha=emo_alpha,
            emotion_batch_index=emotion_batch_index,
            **kwargs,
        )


class ComfyLightIndexTTS2InferEmotionVector(_ComfyLightIndexTTS2InferBase):
    DESCRIPTION = "使用八维情感向量控制情感表达，生成语音。"

    @classmethod
    def INPUT_TYPES(cls):
        inputs = cls.base_input_types()
        inputs["required"].update(
            {
                "emo_happy": ("FLOAT", ui("喜悦权重", "happy 情感权重。", default=0.0, min=0.0, max=1.2, step=0.01)),
                "emo_angry": ("FLOAT", ui("愤怒权重", "angry 情感权重。", default=0.0, min=0.0, max=1.2, step=0.01)),
                "emo_sad": ("FLOAT", ui("悲伤权重", "sad 情感权重。", default=0.0, min=0.0, max=1.2, step=0.01)),
                "emo_fear": ("FLOAT", ui("恐惧权重", "fear 情感权重。", default=0.0, min=0.0, max=1.2, step=0.01)),
                "emo_hate": ("FLOAT", ui("厌恶权重", "hate 情感权重。", default=0.0, min=0.0, max=1.2, step=0.01)),
                "emo_low": ("FLOAT", ui("低落权重", "low 情感权重。", default=0.0, min=0.0, max=1.2, step=0.01)),
                "emo_surprise": ("FLOAT", ui("惊喜权重", "surprise 情感权重。", default=0.0, min=0.0, max=1.2, step=0.01)),
                "emo_neutral": ("FLOAT", ui("自然权重", "neutral 情感权重。", default=1.0, min=0.0, max=1.2, step=0.01)),
            }
        )
        return inputs

    def infer(
        self,
        emo_happy: float = 0.0,
        emo_angry: float = 0.0,
        emo_sad: float = 0.0,
        emo_fear: float = 0.0,
        emo_hate: float = 0.0,
        emo_low: float = 0.0,
        emo_surprise: float = 0.0,
        emo_neutral: float = 1.0,
        **kwargs,
    ):
        return self._infer(
            emotion_mode="emotion_vector",
            emo_happy=emo_happy,
            emo_angry=emo_angry,
            emo_sad=emo_sad,
            emo_fear=emo_fear,
            emo_hate=emo_hate,
            emo_low=emo_low,
            emo_surprise=emo_surprise,
            emo_neutral=emo_neutral,
            **kwargs,
        )


class ComfyLightIndexTTS2InferEmotionText(_ComfyLightIndexTTS2InferBase):
    DESCRIPTION = "使用文本情感分析控制情感表达，生成语音。"

    @classmethod
    def INPUT_TYPES(cls):
        inputs = cls.base_input_types()
        inputs["required"].update(
            {
                "emo_text": (
                    "STRING",
                    ui("情感文本", "用于情感分析的文本；为空时使用合成文本。", default="", multiline=True),
                ),
                "emo_text_weight": (
                    "FLOAT",
                    ui("文本情感权重", "对 Qwen 情感评分结果进行缩放。", default=1.0, min=0.0, max=2.0, step=0.05),
                ),
            }
        )
        return inputs

    def infer(self, emo_text: str = "", emo_text_weight: float = 1.0, **kwargs):
        return self._infer(
            emotion_mode="emotion_text",
            emo_text=emo_text,
            emo_text_weight=emo_text_weight,
            **kwargs,
        )


NODE_CLASS_MAPPINGS = {
    "ComfyLightIndexTTS2LoadModel": ComfyLightIndexTTS2LoadModel,
    "ComfyLightIndexTTS2OnlineLoadModel": ComfyLightIndexTTS2OnlineLoadModel,
    "ComfyLightIndexTTS2InferSpeaker": ComfyLightIndexTTS2InferSpeaker,
    "ComfyLightIndexTTS2InferEmotionAudio": ComfyLightIndexTTS2InferEmotionAudio,
    "ComfyLightIndexTTS2InferEmotionVector": ComfyLightIndexTTS2InferEmotionVector,
    "ComfyLightIndexTTS2InferEmotionText": ComfyLightIndexTTS2InferEmotionText,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyLightIndexTTS2LoadModel": "加载 IndexTTS2 模型",
    "ComfyLightIndexTTS2OnlineLoadModel": "在线加载 IndexTTS2 模型",
    "ComfyLightIndexTTS2InferSpeaker": "IndexTTS2 语音合成（说话人音频）",
    "ComfyLightIndexTTS2InferEmotionAudio": "IndexTTS2 语音合成（情感音频）",
    "ComfyLightIndexTTS2InferEmotionVector": "IndexTTS2 语音合成（情感向量）",
    "ComfyLightIndexTTS2InferEmotionText": "IndexTTS2 语音合成（情感文本）",
}
