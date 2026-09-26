# 独立制包与情感验证记录

2026-09-26 开始验证，09-27 完成页面收尾。基于 `dd35171`，工作分支 `codex/indextts-producer-validation-web`。改动全部位于 IndexTTS 子模块，父仓库保持原分支。本记录保留配置、资产指纹和文本结果，大型模型、运行缓存、上传音频与本次生成包不提交。

## 环境和复现

本机 NVIDIA GeForce RTX 2060 6GB，Windows，Python 3.11.13，Torch 2.8.0+cu128，Gradio 5.45.0，ONNX Runtime 1.23.2，详见 [environment.json](environment.json)。已实际执行 `uv sync --locked --extra validation-web --extra test`。这是在现有环境中按锁文件同步并启动的验证，不宣称另建了完全空白环境。

启动与测试命令见 [使用说明](../../producer-validation-web.md)。源模型、参考音频来自已跟踪的 `tests/fixtures/reader-assets.lock.json`，制包报告同时记录模型、编码器、预处理、参考音频指纹及生产库版本。情感报告记录实际模型 manifest、ONNX 文件 SHA-256、tokenizer 资产、推理输入与 CPUExecutionProvider；模型目录必须由使用者指定。

## 制包数值与资源

[producer-parity.json](producer-parity.json) 记录 FP32/BF16 × `voice_01.wav`/`voice_02.wav` 四组比较。每组的六个张量均满足 shape、dtype、数值完全相等，最大绝对误差 0；完整 manifest（包括 provenance）相同。比较以解包后的内容为准，不依赖 ZIP 字节一致。

每种路径在独立进程内连续处理两段音频，固定 4 个 PyTorch CPU 线程。独立进程检查禁止导入完整 TTS、GPT 主干、codec、CFM、BigVGAN 模块。源 checkpoint 仍会在 CPU 读取，因此并不意味着源模型文件可以全部删除。

| 精度 | 路径 | 两段总耗时（含初始化） | PyTorch peak allocated | peak reserved |
|---|---|---:|---:|---:|
| FP32 | 原完整路径 | 162.72 s | 6.49 GiB | 6.55 GiB |
| FP32 | 独立参考路径 | 88.90 s | 2.20 GiB | 2.22 GiB |
| BF16 | 原完整路径 | 107.64 s | 4.95 GiB | 5.03 GiB |
| BF16 | 独立参考路径 | 89.76 s | 2.20 GiB | 2.22 GiB |

这些是单次运行的诊断数据，耗时包含加载和哈希，且验证期间存在其他 CPU 工作；不是稳定吞吐/RTF 基准。PyTorch 分配统计不等于 Windows WDDM dedicated 显存或物理显存上限；完整 FP32 路径的分配值可超过本机物理显存。不据此声称严格显存上限，也不推算其他显卡速度。独立路径两档峰值相同，是因为最大的参考编码器仍为 FP32。

另外用真实已导出的 FP32/BF16 runtime manifest 调用现有 `ReaderRuntime.reload_voices` 的包加载与兼容检查，四个新包全部接受，见 [runtime-compatibility.json](runtime-compatibility.json)。此检查没有初始化生成网络或运行 TTS。

## 情感模型和回归

[emotion-real.json](emotion-real.json) 保存真实 CPU ONNX 推理。单句、前后上下文、长目标三组详细接口输出均与原 `analyze_window` 相等，临时阈值 1.0 正确回退零向量，模型默认阈值仍为 0.355。实际输入分别为 17、42、512 tokens；上下文包含同段后句，长目标的截断标志为 true。记录的单次纯 ONNX 时间约为 19、31、376 ms，只代表本次输入和设备。

[pytest.txt](pytest.txt)：100 项通过。覆盖原音色包/运行时回归、立体声、采样率转换、超过 15 秒截取、损坏音频、非法元数据、错误模型目录、损坏包、任务失败后恢复、输出不覆盖、会话路径隔离、BF16 CPU 拒绝、异常 ONNX 输出、阈值校验、单句/上下文、章节边界、同段后句和 512-token 规则。警告来自音频/第三方依赖，没有测试失败。

## 浏览器验收

在 `http://127.0.0.1:7861` 使用真实 Gradio 页面完成：

- 上传官方 `voice_01.wav`，识别为 48kHz 单声道、2.438708 秒，无截断，显示可播放的参考音频控件。
- FP32/自动设备生成 `browser-voice-01.ivp`，40.74 秒（含本次初始化），返回成功与 manifest，见 [browser-producer.json](browser-producer.json)。
- 点击实际下载链接并收到浏览器下载事件；重新上传该 `.ivp`，包检查返回 `ok=true` 和 `sourceCompatibilityChecked=true`。
- 单句“太好了，我们终于成功了！”输出总强度 0.8718，默认阈值 0.355，成功下载 JSON。
- 切换上下文模式，双击并编辑前句为“暴风雨过去，他终于平安回来了。”，目标选择 s2；实际模型输入包含该前句和 `[TGT][对白]…[/TGT]`，32 tokens，总强度 0.6681。
- 将临时阈值改为 1.0，显示基础情感零向量；恢复按钮重新读取模型默认阈值。
- 修正图表默认堆叠造成的误读后，重新启动最终页面，验证原始输出和阈值后输出以两张独立条形图展示；详情 JSON 可展开，下载入口可直接访问。浏览器控制台无 error。

本轮验收证明独立产出、包兼容和情感接口可操作，不包含合成试听或克隆音色主观听感评价。
