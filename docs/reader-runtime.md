# Readest 音色包与轻量运行时

本分支把 IndexTTS-2.5 的“参考音频编码”和“语音生成”拆成两个阶段。开发者机器使用完整模型把参考 WAV 转为 `.ivp`；阅读器运行时只读取 `.ivp` 和裁剪模型，不会读取或重新编码参考音频。

## 制作音色包

命令行制包：

```powershell
indextts-voicepack build `
  --reference .\voice.wav `
  --voice-id reader-female-01 `
  --name "阅读女声" `
  --gender female `
  --model-dir .\checkpoints `
  --output .\voices\reader-female-01.ivp `
  --fp16
```

IndexTTS-2.5 WebUI 将使用和制作分成两个页签。“音频生成”页顶部只显示当前引用音色包，不提供重复的选择列表；官方示例从页面下方“快速设置”表选择，自定义包则通过拖入 `.ivp` 应用。选中后直接使用预计算张量，不会读取或编码参考 WAV。“音色包制作”页单独提供参考音频、音色 ID、名称、性别和生成下载入口；制作完成后会自动安装并应用。拖入或制作的包都必须先通过 schema、哈希和源模型指纹校验。

WebUI 首次启动时会提前为全部官方示例音色生成并校验内置 `.ivp`。页面下方的“快速设置”行只切换对应的现成音色包、文本、语言和情感参数，点击时不再执行参考音频编码。

内置音色使用“年龄-性别-风格”三段式显示名，并使用能体现相同特征的稳定 ASCII ID（例如 `young-female-clear-lively`），不再使用测试性质的 `example-voice_*` ID。WebUI 从上传参考音频制作的新包默认生成到 Readest 父目录的 `index-tts-package/SoundPackage`；可通过 `INDEXTTS_VOICEPACK_EXPORT_DIR` 覆盖服务端输出目录。浏览器下载按钮以该文件为下载源，最终客户端下载目录仍由浏览器设置决定。

`.ivp` 是确定性 ZIP，只允许以下文件：

- `manifest.json`
- `conditioning.safetensors`
- `LICENSE`、`LICENSE_ZH.txt` 和 `DERIVATIVE_DISCLAIMER.txt`

它不包含 WAV、原始路径、试听文件、pickle 或 PyTorch checkpoint。loader 会验证 ZIP 路径、schema、文件哈希、张量白名单、shape、dtype、conditioning ABI 和源模型指纹。

检查音色包：

```powershell
indextts-voicepack inspect .\voices\reader-female-01.ivp
indextts-voicepack verify .\voices\reader-female-01.ivp --model-dir .\checkpoints
```

## 导出裁剪模型

```powershell
indextts-runtime export-model `
  --source-model-dir .\checkpoints `
  --output-dir ..\index-tts-package\ModulePackage\model

indextts-runtime doctor --model-dir ..\index-tts-package\ModulePackage\model
```

质量优先导出结果保留 FP32 GPT 生成部分、FP32 s2mel、decoder-only semantic codec 和 FP32 BigVGAN，并强制关闭 TF32 矩阵计算。Wav2Vec-BERT、CAMPPlus、GPT speaker/emotion conditioner、`text_head` 和 codec encoder 不会进入运行时模型。核心文件上限为 3.2GiB，并写入 `runtime_model.json` 哈希与精度清单。音色包生成器默认也使用 FP32；`--fp16` 仅作为显式选择保留。

## Python API

```python
from indextts.runtime import ReaderRuntime

runtime = ReaderRuntime(
    r".\reader-model",
    [r".\voices"],
    None,
    "cuda:0",
    cache_dir=r".\cache",
)
result = runtime.synthesize("第一章，故事开始了。", "reader-female-01", emotion="base")
```

`base` 直接使用音色包从原参考音频提取的基础情感；显式向量顺序固定为 `[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]`，归一化后总强度不超过 `0.8`，剩余权重继续使用基础情感。全零向量等价于 `base`。Qwen 是可选插件，本资源包不包含或启用它。

## JSONL sidecar

sidecar 通过 stdin/stdout 逐行收发 JSON，不监听端口。所有日志写 stderr，stdout 只包含协议响应：

```powershell
indextts-runtime serve `
  --model-dir .\reader-model `
  --voice-dir .\voices `
  --cache-dir .\cache `
  --emotion-backend none
```

请求与响应示例：

```json
{"id":"1","method":"synthesize","params":{"text":"你好。","voiceId":"reader-female-01","emotion":"base","durationFactor":1.0}}
{"id":"1","ok":true,"result":{"audioPath":"C:\\...\\cache\\tts-....wav","sampleRate":22050,"durationMs":1200,"emotionMode":"base","emotionVector":null,"timings":{},"warnings":[]}}
```

固定方法为 `health`、`voices.list`、`voices.reload`、`synthesize`、`cancel` 和 `shutdown`。`cancel` 参数为 `{"requestId":"待取消的 synthesize id"}`。GPU 合成并发固定为 1；排队任务立即取消，运行中任务在情感、分段、GPT 和声学阶段边界取消。`synthesize` 不接受输出路径，所有临时文件和 WAV 都由 sidecar 的 cache 目录管理。

## 当前边界和许可

- Windows、NVIDIA CUDA、中文、GPU TTS；不支持 CPU TTS、DirectML 或 TensorRT。
- 第一版不包含自动情感模型；TTS 常驻 GPU，低显存文本按 40 个字符附近分段。
- 本功能不包含便携 Python/PyTorch 分发环境，也不修改 Readest 的 TypeScript、Rust 或 Edge TTS 流程。
- 模型、裁剪权重和音色包继续受 bilibili Model Use License 与原始版权声明约束。预制音色的授权、同意和声音权利由音色提供者另行保证。

## 同步上游

此分支以 fork 的 `origin` 为推送目标，以官方仓库 `upstream` 为同步来源：

```powershell
git fetch upstream
git switch readest-integration
git rebase upstream/main
git push --force-with-lease origin readest-integration
```

如果分支已经多人协作，可用 `git merge upstream/main` 并普通 push，避免改写历史。父仓库继续固定 submodule commit，不配置自动跟踪分支。
