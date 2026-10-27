# ComfyUI-Light-IndexTTS2

在 ComfyUI 中使用 IndexTTS2 合成 22050 Hz 语音。支持参考音色克隆，以及参考音频、八维情感权重和情感文本控制。长文本按段生成后拼接。

## 安装与模型

将仓库放入 `ComfyUI/custom_nodes/`，使用 ComfyUI 的 Python 环境安装依赖：

```shell
python -m pip install -r requirements.txt
```

从 [eastmoe/IndexTTS-Comfy](https://huggingface.co/eastmoe/IndexTTS-Comfy) 下载 IndexTTS2 `.safetensors`，放入 `ComfyUI/models/checkpoints/`，重启 ComfyUI。

## 用法

导入示例工作流，选择 checkpoint，载入参考音频，填写文本后运行：

- [参考音色与情感](examples/speaker.json)：同一段参考音频控制音色和情感。
- [独立情感音频](examples/emotion_audio.json)：分别设置音色参考和情感参考；音色参考留空时使用情感音频。
- [八维情感权重](examples/emotion_vector.json)：调整高兴、愤怒、悲伤、恐惧、厌恶、低落、惊讶、自然的权重。
- [情感文本](examples/emotion_text.json)：输入情感描述；留空时分析待合成文本。

自定义节点位于 `eastmoe → IndexTTS`。加载器输出 `MODEL`、`CLIP`、`VAE`。条件编码器与引导器提供语音条件，AR / CFM 采样器、随机源、CFM 采样计划和空 Mel Latent 接入原生 `SamplerCustomAdvanced`。输出经原生 `VAE Decode Audio` 解码，再由音频分段拼接节点合并，连接 `Save Audio` 保存。

`max_text_tokens` 控制每段文本的 token 上限，`max_tokens` 控制每段语义 token 上限。空 Mel Latent 的 `seconds` 设置每段最大容量，`target_duration` 提供 GPT 时长条件。`steps` 设置 CFM Euler 积分步数，`cfm_cfg` 默认 0.7，对应常规 CFG 1.7。各参数的含义可在节点悬停提示中查看。

## 上游项目

- [IndexTTS](https://github.com/index-tts/index-tts)
- [ComfyUI](https://github.com/Comfy-Org/ComfyUI)
- [BigVGAN](https://github.com/NVIDIA/BigVGAN)
- [Amphion / MaskGCT](https://github.com/open-mmlab/Amphion)
