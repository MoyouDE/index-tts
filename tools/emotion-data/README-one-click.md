# 情感模型一键候选训练（2026-09-25）

双击 `index-tts/train-emotion.bat`：按 `one-click-profile.json` 固定的新版数据和损失路线运行预检、完整 train 训练、dev 阈值校准、冻结 dev/test 对照，并将全部产物写入 `E:/Projects/Readest-temp/emotion-full-retrain-20260925-reviewed-balanced-v1-progress`。**不会自动覆盖正式 A 或阅读器交接包，也不会提交/推送。**

当前默认方法是已有正式 A 使用、且作为数据增量 D0 对照的 `balanced-regression-v1 + uniform`：8 epoch、batch 6、梯度累积 4、FP32、512 token、neutral loss weight 3。P1 正例感知回归、N2 旁白重放、R3 更高正例权重在先前实验中没有通过完整替换验收，故没有偷偷加入默认 BAT。新版快照仅对新增 train 的一条高置信 calm 误标做修正；dev/test 与 2026-09-22 版字节一致。

启动前会核对 train/dev/test 行数、完整文件 SHA-256、基础 MacBERT 权重与文件、正式 A 哈希，以及训练/评估实现 SHA-256，并在输出目录写 `run-manifest.json`。第二次运行若指纹相同，会由训练器自动恢复或跳过已完成训练；任何输入、方法或代码变化都会拒绝在旧目录续训。控制台沿用 speaker-id 的 `[1/3] 预检 → [2/3] 训练 → [3/3] 评估` 阶段标题和配置摘要；预检单次扫描同时核对分词长度与完整上下文，训练与验证输入准备、逐轮训练／验证和最终评估均显示进度条。训练进度刷新限频，但损失、步数、学习率和显存信息保留。训练和验证上下文编码只在本次进程内准备一次，组批时仍动态补齐，不改变模型输入或生成持久缓存。完整预检报告仍保存在 `preflight.json`。按用户要求，默认 BAT 不设置空闲显存门槛，也不等待显存；仍要求 CUDA 可用及至少 10 GiB 磁盘空间。实际显存不足时 PyTorch 可能直接报 CUDA OOM，**不会自动切换到 CPU 或仅仅变慢**；不会关闭其它 GPU 应用。调整配置必须使用**新的输出目录**。

输出结构：

- `preflight.json`：数据、上下文与运行环境预检。
- `run/`：训练配置、逐轮报告、final/best、最近两个恢复 checkpoint 和完成标记。
- `evaluation/`：正式 A 与候选在旧 dev、冻结旧 test、新增 test/对白/旁白上的详细指标、校准报告、带 ID／作品的逐条预测、新增 test 固定改善／退化案例、预测缓存、逐项冻结质量门和中文结论。
- `status.json`：`preflight`、`training`、`evaluation`、`completed`、`interrupted` 或 `failed`。

运行中的模型选择只用 dev；test 是冻结的最终描述与预先约定门槛核验，不因 test 表现回调训练参数。旧 dev 的 neutral 仅 7 条，相关误激活率本身不稳定；必须同时阅读新增集切片。即使全部门槛通过，也仍需用户明确确认才发布；未通过时保留候选，不替换正式 A。

无 GPU 的只读 profile/哈希检查：

```powershell
.venv/Scripts/python.exe -X utf8 tools/emotion-data/run_one_click.py --validate-only
```
