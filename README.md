# ComfyUI-Light-IndexTTS2

`ComfyUI-Light-IndexTTS2` 是基于当前裁剪版 IndexTTS2 项目的 ComfyUI 插件封装。节点注册在 `eastmoe/Comfy-Light-IndexTTS2` 菜单下，并复用 ComfyUI 原生 `AUDIO` 类型，方便接入加载音频、剪辑音频、保存音频等工作流。

## 节点

- `加载 IndexTTS2 模型`
  - 默认模型路径：`ComfyUI/models/IndexTTS2/IndexTTS-2`
  - 默认配置文件：模型目录下的 `config.yaml`
  - `auto` 会优先读取新的子目录布局，并兼容旧的 `ComfyUI/models/IndexTTS2` 平铺目录。
  - 加载前会校验本地 `config.yaml` 及其引用的关键权重文件。
  - 支持 `auto/cuda/xpu/cpu/mps` 设备选择、FP8/FP16/BF16/FP32 计算精度、BigVGAN CUDA kernel 开关和模型缓存重载。

- `在线加载 IndexTTS2 模型`
  - 默认模型仓库：`IndexTeam/IndexTTS-2`
  - 支持从 Hugging Face、`hf-mirror.com` 或自定义 Hugging Face 反向代理下载模型。
  - 自定义端点可填写 host name、IP address 或完整 URL，并可关闭 SSL 证书验证以适配自签名代理或受控内网。
  - `local_dir=auto` 时会把主模型和依赖模型分别下载到 `ComfyUI/models/IndexTTS2` 下的独立子目录，先补齐依赖再按所选设备与计算精度加载。

- `IndexTTS2 语音合成（说话人音频）`
  - 使用同一段说话人参考 `AUDIO` 同时控制音色和情感。

- `IndexTTS2 语音合成（情感音频）`
  - 使用说话人参考 `AUDIO` 控制音色，并使用独立情感参考 `AUDIO` 控制情感表达。
  - 说话人参考 `AUDIO` 是可选输入；未连接时会使用情感参考 `AUDIO` 兼作音色参考。

- `IndexTTS2 语音合成（情感向量）`
  - 使用八维情感向量控制情感表达，适合精确调节喜悦、愤怒、悲伤、恐惧、厌恶、低落、惊喜和自然程度。

- `IndexTTS2 语音合成（情感文本）`
  - 使用情感文本经 Qwen 情感分析得到情感向量；情感文本为空时使用合成文本。

四个语音合成节点共享：
  - 输入：模型句柄、说话人参考 `AUDIO`、合成文本。
  - 输出：ComfyUI 原生 `AUDIO`、可选 wav 输出路径、JSON 信息。
  - 覆盖 `infer.py` 的主要功能：文本文件、目标时长、文本切分、GPT 采样参数、typical sampling、S2Mel 扩散参数、随机种子、详细日志和 wav 输出。
  - 可选择生成后自动卸载模型并清理 CUDA/Intel XPU/MPS 显存缓存。

## 模型目录

推荐使用新的子目录布局管理 IndexTTS2 主模型和依赖模型：

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

`IndexTTS-2/` 中应包含 `config.yaml` 以及该配置引用的 GPT、S2Mel、BPE、情感矩阵等文件。在线加载节点会在根目录下为 `w2v-bert-2.0`、`MaskGCT`、`campplus` 和 BigVGAN 创建独立目录；Hugging Face 缓存统一放在 `hf_cache/`，量化缓存放在 `IndexTTS2-cache/`。

旧版平铺布局仍可使用：

```text
ComfyUI/models/IndexTTS2/config.yaml
```

当 `model_dir` 为 `auto` 且不存在 `IndexTTS-2/config.yaml` 时，加载器会自动回退到这个旧目录。

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
4. 按情感控制方式添加对应的 `IndexTTS2 语音合成（...）` 节点，把模型输出和参考音频接入。
5. 运行工作流，合成结果会作为 ComfyUI `AUDIO` 返回；填写 `output_path` 时也会保存 wav。
