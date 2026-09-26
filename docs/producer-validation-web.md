# 独立音色制包与情感验证

新增入口 `python -m indextts.validation_web`。两个页签分别验证参考音频 → `.ivp` 和文本 → 八维情感向量；本轮不提供 TTS 合成试听。参考音频可以播放，音色包能独立产出，但这不代表完成了克隆音色的主观听感验证。

## 跨设备启动

在 IndexTTS 子模块目录执行。Python 使用 3.10 或 3.11，推荐与验证机器相同的 3.11；不需要复制本机 `.venv`、缓存或父仓库的忽略文件。

```powershell
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new()
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [Console]::OutputEncoding
$env:PYTHONUTF8 = '1'
uv sync --locked --extra validation-web --extra test

# 制包所需源模型与固定参考音频：使用已跟踪的 revision / SHA-256 锁文件。
.venv/Scripts/python.exe -m indextts.runtime.assets fetch
.venv/Scripts/python.exe -m indextts.runtime.assets verify

# 将情感模型目录改为本设备的实际路径，也可省略参数后在页面填写。
.venv/Scripts/python.exe -m indextts.validation_web `
  --source-model-dir checkpoints `
  --emotion-model-dir D:/Models/readest-emotion/onnx `
  --output-dir outputs/validation-web `
  --port 7861 --cpu-threads 4
```

Linux 使用 `.venv/bin/python`，并将模型路径改为本机路径。安装范围沿用项目的基础依赖；新增 `validation-web` extra 锁定 Gradio 5.45.0、CPU ONNX Runtime 1.23.2，完整解析结果在 `uv.lock` 中。无需 DeepSpeed、flash-attn 或 Qwen extra。

默认地址为 `http://127.0.0.1:7861`，`share=False`。参数 `--host` 可改变监听地址，默认仅供本机使用。情感模型不包含在 Git 中：需指定已有、通过原有发布校验的 v3 ONNX 目录，包含 `emotion_model.json`、`emotion.onnx`、tokenizer 和 manifest 列出的资产。不存在或损坏会显示错误，绝不返回假成功结果。

两个页签按需加载，缺少任一类权重不阻止另一类操作。仅测试情感时无需下载 IndexTTS 源模型。全部模型、上传音频和输出位于忽略目录，跨设备需要重新下载源模型或自行复制情感模型；本次测试的资产指纹见 [验证记录](validation/producer-web-2026-09-26/README.md)。

## 音色包生产

上传参考音频后显示原始时长、采样率、声道数和截断标志，填写音色 ID、名称、性别，选择 FP32 或 BF16 和设备后生成。每次任务使用独立会话目录和随机任务目录；同名音色再次生成也不覆盖旧成果。成功后可下载 `.ivp` 并查看 manifest、provenance 和校验结果。

默认 FP32，设备自动选择 CUDA，否则 CPU。BF16 沿用现有混合精度路径：情感/说话人投影采用 BF16，其余参考组件保留原精度。显式选择 BF16 而设备不支持时抛错；可执行 BF16 不代表具备原生加速。BF16 包必须匹配对应 BF16 运行时；不能通过转换 FP32 包的 dtype 替代重新制包。

“检查已有音色包”验证结构、哈希、张量以及源模型指纹；源模型目录留空可仅检查包自身完整性。模型指纹检查仍需读取 codec 和声码器源文件的哈希，但不会初始化这些网络。

`VoicePackBuilder` 默认使用 `ReferenceEncoder`，只构建 Wav2Vec2-BERT、CAMPPlus、情感 Conformer/perceiver 及投影、说话人投影和长度调节器，读取必要统计量与情感矩阵。GPT、s2mel checkpoint 先在 CPU 上读取，严格提取目标子模块权重；不构建 GPT 自回归主干、语义 codec、CFM/DiT、BigVGAN、文本前端或 Qwen。保留历史 librosa 默认 22.05kHz 单声道加载、前 15 秒截取、resample、fbank、计算顺序和六种输出张量，不改变预处理指纹。

组件分阶段使用 GPU，其他组件留在 CPU。因此仍需足够的系统内存读取源 checkpoint，不能将显存峰值误当作总内存开销。制包请求串行执行，设备、精度或源目录改变时卸载旧实例；“卸载制包模型”主动释放缓存。失败后清理实例，下次任务可以重试。

制包 CLI 和旧 WebUI 保持调用兼容；显式传入完整 TTS 实例仍使用该实例。建议 CLI 显式指定 profile，以生成包含 provenance 的 schema v2 包：

```powershell
.venv/Scripts/python.exe -m indextts.voicepack.cli build --reference examples/voice_01.wav --voice-id reader-one --name reader-one --gender unknown --model-dir checkpoints --profile compatible-fp32 --output outputs/reader-one.ivp
```

## 情感验证

单句模式默认对白，也可选旁白。上下文表格包含句子 ID、章节、段落、类型和正文，目标句通过 ID 选择。沿用现有 reader 上下文策略：一次请求仅允许同一章节、按原文顺序排列，跨章节输入明确拒绝；保留同段后句和前句选择规则，总预算 512 tokens，超长目标按现有策略截断。

`OnnxEmotionProvider.analyze_window_details` 共用现有的上下文选择、一次 ONNX 推理和后处理；`analyze` / `analyze_window` 继续返回八维列表。详情包含原始输出、独立总强度、阈值、基础情感回退标志、最终向量、实际目标标记文本、token 数、选中句子、截断状态和耗时。总强度来自模型自己的输出，不是八维分数之和。

默认阈值读取当前模型 manifest；取消“使用模型默认阈值”后才能应用临时值，恢复按钮重新取回 manifest 默认值。不会修改模型文件。低于阈值时最终向量为全零，表示使用参考音频的基础情感，不等于强制平静。这里展示 provider 输出，阅读器后续的情感偏置/总和缩放不在此页面重复执行。

JSON 下载包含输入句子、目标 ID、模型版本和指纹，便于复现。模型使用 CPUExecutionProvider；可独立卸载。两种模型缓存按服务进程共享，制包和情感队列独立，上传与结果不在不同会话间展示或复用；下载链接由 Gradio 管理。本工具是本机验证工具，没有多用户认证，不应把本机会话隔离理解为公网租户权限系统。输出文件保留以供复查，确认不再需要时自行清理输出目录。

## 验证命令

```powershell
.venv/Scripts/python.exe -m pytest tests/test_voicepack.py tests/test_reader_runtime.py tests/test_reader_optimizations.py tests/test_emotion_context_policy.py tests/test_validation_modules.py tests/test_webui_syntax.py -q

# GPU 实测：四个独立进程，两种精度各比较完整路径和独立路径；输出目录必须未存在。
.venv/Scripts/python.exe tools/validate_voicepack_producer.py --output outputs/producer-parity-new
# 只复查已生成文件，不运行模型：追加 --compare-only。

.venv/Scripts/python.exe tools/validate_emotion_provider.py --model-dir D:/Models/readest-emotion/onnx --output outputs/emotion-validation-new.json
```

完整路径对比需要更多内存/显存；硬件不足时应换用可运行完整基线的设备，不通过降低精度伪造同精度对比。GPU 型号不作为验收标准。本轮以独立产出与数值兼容为验收目标，实际数据见验证记录。
