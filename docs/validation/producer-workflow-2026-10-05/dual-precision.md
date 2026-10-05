# 双精度音色包验收（2026-10-05）

- 简化制作页固定生成 FP32 + BF16，设备选择保留高级设置；精度选项移入文本试听区。
- 新 `.ivp` schema v4 含 7 个文件：根 manifest、FP32 conditioning、BF16 manifest/conditioning 和 3 个许可文件。无原音频、工作区或试听文件。
- 两档真实预计算共用同一份裁剪文件快照，防止 WAV PEAK 时间戳导致来源哈希不同。运行时按模型精度读取对应张量，读取任一档都会校验整个包。
- 最终 102 项读包、工作区、运行时、旧行为及 UI 测试通过；包含共享参考文件快照的回归验证。
- 真实 GPU 验收在 `outputs/producer-dual-acceptance/`：两个 5 秒片段一次制包；同一版本 `6ad39895b8b1458ba85a2123eaa584ad` 分别完成 FP32、BF16 文本合成；切换精度没有新包。
- 修改名称/性别后导出，两档张量字节均保持原样，两档读包、来源匹配和工作区重启恢复通过。导出包为 2,000,231 字节；完整报告为该目录 `results.json`。
- 原 schema v1～v3 继续读取；旧客户端使用 v4 需要更新读包代码，或通过 `extract_voicepack` 无损拆出对应单精度文件。音色库导入 v4 自动拆为原有两个精度文件。
- 浏览器当前工作区补齐 BF16，复用已有 FP32，生成版本 `09275d5f`；原成功试听未更新。切换试听 BF16 后下载仍为同一版本，最后恢复 FP32。浏览器实际下载到 `C:/Users/ADMIN/Downloads/音色-小琳 Test (1).ivp`，2,912,548 字节；重新读取 FP32/BF16 均通过，speaker_latent 分别为 float32/bfloat16。界面证据为 `outputs/producer-dual-precision-20261005.png`。

验证命令：

```powershell
.venv/Scripts/python.exe -m pytest tests/test_producer_drafts.py tests/test_voicepack.py tests/test_voice_workbench.py tests/test_reader_runtime.py tests/test_audition_worker.py tests/test_validation_decoupling.py -q
.venv/Scripts/python.exe tools/verify_producer_workflow.py --source-id 6a93e3e9179e454a96cecda15fceedce
```
