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

WebUI 的“保存为预设”仍保存参考 WAV 和界面参数；它不是音色包。IndexTTS-2.5 页面的“导出音色包”会生成可下载的 `.ivp`，并自动安装、选中到“当前应用音色包”列表。也可以把已有 `.ivp` 直接拖入该区域；包通过 schema、哈希和源模型指纹校验后才会安装。选中音色包进行生成时，完整 WebUI 也直接使用预计算张量，不再读取或编码参考 WAV；选择“使用上方参考音频（不使用音色包）”即可切回原始推理方式。

点击页面下方的官方示例行时，WebUI 会在首次使用时自动为对应参考音频生成 `example-*.ivp`，随后选中该音色包；再次点击会复用已校验的包。

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
  --output-dir .\reader-model

indextts-runtime doctor --model-dir .\reader-model
```

导出结果保留 BF16 GPT 生成部分、FP32 s2mel、decoder-only semantic codec 和 FP32 BigVGAN。Wav2Vec-BERT、CAMPPlus、GPT speaker/emotion conditioner、`text_head` 和 codec encoder 不会进入运行时模型。导出器强制核心文件合计不超过 2.2GiB，并写入 `runtime_model.json` 哈希清单。

## Python API

```python
from indextts.runtime import QwenEmotionProvider, ReaderRuntime

emotion = QwenEmotionProvider(r".\checkpoints\qwen0.6bemo4-merge")
runtime = ReaderRuntime(
    r".\reader-model",
    [r".\voices"],
    emotion,
    "cuda:0",
    cache_dir=r".\cache",
)
result = runtime.synthesize("第一章，故事开始了。", "reader-female-01")
```

Qwen 情感模型在首次自动情感请求时懒加载，并强制放在 CPU。情感顺序固定为 `[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]`，归一化后总强度不超过 `0.8`。无效 JSON、超时或异常会回退 calm，并在 `warnings` 返回原因。

## JSONL sidecar

sidecar 通过 stdin/stdout 逐行收发 JSON，不监听端口。所有日志写 stderr，stdout 只包含协议响应：

```powershell
indextts-runtime serve `
  --model-dir .\reader-model `
  --voice-dir .\voices `
  --cache-dir .\cache `
  --emotion-backend qwen `
  --qwen-model-dir .\checkpoints\qwen0.6bemo4-merge
```

请求与响应示例：

```json
{"id":"1","method":"synthesize","params":{"text":"你好。","voiceId":"reader-female-01","emotion":"auto","durationFactor":1.0}}
{"id":"1","ok":true,"result":{"audioPath":"C:\\...\\cache\\tts-....wav","sampleRate":22050,"durationMs":1200,"emotionVector":[0,0,0,0,0,0,0,0.8],"timings":{},"warnings":[]}}
```

固定方法为 `health`、`voices.list`、`voices.reload`、`synthesize`、`cancel` 和 `shutdown`。`cancel` 参数为 `{"requestId":"待取消的 synthesize id"}`。GPU 合成并发固定为 1；排队任务立即取消，运行中任务在情感、分段、GPT 和声学阶段边界取消。`synthesize` 不接受输出路径，所有临时文件和 WAV 都由 sidecar 的 cache 目录管理。

## 当前边界和许可

- Windows、NVIDIA CUDA、中文、GPU TTS；不支持 CPU TTS、DirectML 或 TensorRT。
- Qwen 在 CPU，TTS 常驻 GPU；低显存文本按 40 个字符附近分段。
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
