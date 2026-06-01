# ComfyUI-Light-IndexTTS2

`ComfyUI-Light-IndexTTS2` 是基于当前裁剪版 IndexTTS2 项目的 ComfyUI 插件封装。节点注册在 `eastmoe/Comfy-Light-IndexTTS2` 菜单下，并复用 ComfyUI 原生 `AUDIO` 类型，方便接入加载音频、剪辑音频、保存音频等工作流。

## 节点

- `加载 IndexTTS2 模型`
  - 默认模型路径：`ComfyUI/models/IndexTTS2`
  - 默认配置文件：模型目录下的 `config.yaml`
  - 加载前会校验本地 `config.yaml` 及其引用的关键权重文件。
  - 支持 `auto/cuda/xpu/cpu/mps` 设备选择、FP8/FP16/BF16/FP32 计算精度、BigVGAN CUDA kernel 开关和模型缓存重载。

- `在线加载 IndexTTS2 模型`
  - 默认模型仓库：`IndexTeam/IndexTTS-2`
  - 支持从 Hugging Face、`hf-mirror.com` 或自定义 Hugging Face 反向代理下载模型。
  - 自定义端点可填写 host name、IP address 或完整 URL，并可关闭 SSL 证书验证以适配自签名代理或受控内网。
  - 下载后会先校验本地模型，再按所选设备与计算精度加载。

- `IndexTTS2 语音合成`
  - 输入：模型句柄、说话人参考 `AUDIO`、合成文本。
  - 可选输入：情感参考 `AUDIO`。
  - 输出：ComfyUI 原生 `AUDIO`、可选 wav 输出路径、JSON 信息。
  - 覆盖 `infer.py` 的主要功能：文本文件、情感音频、八维情感向量、文本情感评分、目标时长、文本切分、GPT 采样参数、typical sampling、S2Mel 扩散参数、随机种子、详细日志和 wav 输出。
  - 可选择生成后自动卸载模型并清理 CUDA/Intel XPU/MPS 显存缓存。

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
├── locales/zh-CN/nodeDefs.json
├── locales/zh-cn/nodeDefs.json
├── locales/zh/nodeDefs.json
├── doc/
├── temp/
├── static/
└── workflow/
```

`locales/*/nodeDefs.json` 是当前 ComfyUI 前端实际读取的节点翻译文件。

## 使用

1. 重启 ComfyUI。
2. 在右键菜单中选择 `eastmoe -> Comfy-Light-IndexTTS2`。
3. 添加 `加载 IndexTTS2 模型` 节点；如果需要自动下载，改用 `在线加载 IndexTTS2 模型` 节点。
4. 添加 `IndexTTS2 语音合成` 节点，把模型输出和参考音频接入。
5. 运行工作流，合成结果会作为 ComfyUI `AUDIO` 返回；填写 `output_path` 时也会保存 wav。
