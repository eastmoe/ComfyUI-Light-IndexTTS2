# ComfyUI-Light-IndexTTS2

`ComfyUI-Light-IndexTTS2` 是基于当前裁剪版 IndexTTS2 项目的 ComfyUI 插件封装。节点注册在 `eastmoe/Comfy-Light-IndexTTS2` 菜单下，并复用 ComfyUI 原生 `AUDIO` 类型，方便接入加载音频、剪辑音频、保存音频等工作流。

## 节点

- `加载 IndexTTS2 模型`
  - 默认模型路径：`ComfyUI/models/IndexTTS2`
  - 默认配置文件：模型目录下的 `config.yaml`
  - 支持 `auto/cuda/cpu/mps` 设备选择、CUDA FP16、BigVGAN CUDA kernel 开关和模型缓存重载。

- `IndexTTS2 语音合成`
  - 输入：模型句柄、说话人参考 `AUDIO`、合成文本。
  - 可选输入：情感参考 `AUDIO`。
  - 输出：ComfyUI 原生 `AUDIO`、可选 wav 输出路径、JSON 信息。
  - 覆盖 `infer.py` 的主要功能：文本文件、情感音频、八维情感向量、文本情感评分、目标时长、文本切分、GPT 采样参数、typical sampling、S2Mel 扩散参数、随机种子、详细日志和 wav 输出。

## 模型目录

将 IndexTTS2 权重放到：

```text
ComfyUI/models/IndexTTS2/
```

目录中应包含 `config.yaml` 以及该配置引用的 GPT、S2Mel、BPE、情感矩阵等文件。依赖的 Hugging Face 模型会沿用 IndexTTS2 原逻辑下载，缓存目录默认指向模型目录下的 `hf_cache`。

## 项目结构

```text
ComfyUI-Light-IndexTTS2/
├── __init__.py
├── src/comfy_light_indextts2/
├── local/zh-cn/nodes.json
├── loacl/zh-cn/nodes.json
├── doc/
├── temp/
├── static/
└── workflow/
```

`local/zh-cn/nodes.json` 是标准中文翻译路径；`loacl/zh-cn/nodes.json` 也保留了一份，以兼容早期文档或拼写误用。

## 使用

1. 重启 ComfyUI。
2. 在右键菜单中选择 `eastmoe -> Comfy-Light-IndexTTS2`。
3. 添加 `加载 IndexTTS2 模型` 节点。
4. 添加 `IndexTTS2 语音合成` 节点，把模型输出和参考音频接入。
5. 运行工作流，合成结果会作为 ComfyUI `AUDIO` 返回；填写 `output_path` 时也会保存 wav。
