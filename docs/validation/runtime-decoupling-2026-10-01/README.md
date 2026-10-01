# 第一轮解耦验证（2026-10-01）

基于 `4a9f814`，只重构服务、页面及按需启动。未修改推理算法、ONNX/tokenizer、音色包格式、JSONL sidecar 协议或安装依赖；本地资产不迁移。

## 自动化回归

141 项通过，8 条第三方弃用/未来变更警告，完整输出见 `pytest.txt`。包含六种页面组合的独立进程导入检查、未加载模型检查、缺失跨页连接、向量副本、GPU 串行切换、旧公开接口、包及试听原子替换、排队取消、子进程异常恢复和原推理回归。

最终增加“已经取消的排队任务不卸载另一用途模型”的保护，并针对 GPU 协调、排队取消、失败恢复和子进程取消补跑 6 项，见 `cancel-regression.txt`。

```powershell
.venv/Scripts/python.exe -m pytest tests/test_validation_decoupling.py tests/test_audition_worker.py tests/test_voice_workbench.py tests/test_reader_runtime.py tests/test_reader_optimizations.py tests/test_validation_modules.py tests/test_voicepack.py tests/test_emotion_context_policy.py tests/test_webui_syntax.py -q
.venv/Scripts/python.exe tools/measure_validation_startup.py --serve --output outputs/startup-measurement.json
.venv/Scripts/python.exe tools/measure_validation_startup.py --serve --baseline --output outputs/startup-baseline.json
```

## 本机启动测量

使用同一 Python 3.11.13 / Gradio 5.45.0 环境，每个模式启动一个新 Python 进程，使用空临时工作区和不存在的模型目录。时间从工具函数开始到 Gradio 本地服务可访问，不含解释器启动和首次模型加载。RSS 是该 Python 进程当前工作集，不是显存、峰值或整台机器内存。单次测量受系统负载和文件缓存影响，不作为跨设备速度标准。

| 模式 | 服务就绪秒 | RSS MiB |
|---|---:|---:|
| 原组合页（4a9f814） | 11.852 | 557.4 |
| 全功能 | 12.395 | 553.4 |
| 制包 | 10.329 | 534.8 |
| 情感 | 7.286 | 189.8 |
| 试听 | 11.399 | 534.6 |
| 制包＋试听 | 10.833 | 535.8 |
| 情感＋试听 | 11.374 | 551.6 |

纯情感模式相对原组合页减少约 368 MiB（约 66%）启动工作集；全功能模式占用基本不变。制包/试听模式仍需音色包的 PyTorch 张量校验。所有模式页面构造均未导入参考编码器、ReaderRuntime 生成主干或 ONNX Runtime；纯情感模式亦未导入 torch、音色包归档模块或其他功能页面。

`startup.json` / `baseline-startup.json` 保存服务就绪测量；`construction.json` / `baseline-construction.json` 保存另一次仅构造页面的测量及导入明细。

## 真实模型与浏览器

- 使用项目参考 `examples/voice_01.wav` 生成 FP32 音色包，与已有同参考包逐张量比较，shape、dtype、数值和 provenance 全部一致。
- 使用本机已批准的情感 ONNX，验证上下文详细结果与原 provider 的向量、强度、阈值及选中上下文一致。
- 使用已有 FP32 裁剪模型、基础情感、种子 17、文本“今天阳光很好。”，新应用门面输出的 WAV 哈希与之前提交的固定种子回归完全一致。未重新导出任何模型。
- 浏览器完成真实情感分析、选择缺失 BF16 版本并跳转同一音色的制包页、返回 FP32 采用情感页向量试听、播放及下载 WAV。`browser-audition.json` 的输入向量与 `browser-emotion.json` 的最终向量一致。
- 本地测试条目为 `decoupling-check-20261001`，参考、包及试听只保存在忽略的工作区。截图见 `browser-proof.png`；完整数值、资源口径与资产指纹见 `real-validation.json`。

本轮真实试听以 FP32 验证应用重构；BF16 参数与协议由已有回归覆盖。功能和数值通过不代表主观音色质量达标。

跨设备复现可用以下工具；基线包须在相同模型、参考、设备、精度和依赖下生成。工具从基线包读取精度，使用新音色 ID 保存结果，已有条目不会被覆盖；可额外传 `--expected-wav-sha256` 验证同设备的历史固定种子 WAV。

```powershell
.venv/Scripts/python.exe tools/validate_service_decoupling.py --reference examples/voice_01.wav --baseline-pack D:/Models/baseline.ivp --source-model checkpoints --emotion-model D:/Models/emotion-onnx --reader-model D:/Models/reader-fp32 --workspace outputs/decoupling-check --output outputs/decoupling-check/report.json
```

浏览器仍出现此前已记录的 Gradio 5.45 波形 `ResizeObserver / offsetWidth` 错误，发生于页签切换组件销毁；本轮播放、下载和跨页任务完成正常，未修改第三方组件。
