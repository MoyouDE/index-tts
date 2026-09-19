# 情感不平衡初筛实验复现

使用 `index-tts/.venv/Scripts/python.exe tools/emotion-data/run_imbalance_experiment.py` 启动或恢复。
默认集中输出到 `E:/Projects/Readest-temp/emotion-imbalance-experiment-20260919`。2026-09-19 的正式实验已完成并清理；重新运行会从头创建该目录。

执行顺序：A/B 实际训练与恢复冒烟 → A 八轮 → B 八轮 → 三组 dev 推理和冻结结论 → 三组 test 推理和描述性报告。
同一时间只运行一个 GPU 子进程。实验入口需要至少 8 GiB 空闲显存、20 GiB 空闲磁盘；失败停止后续阶段。
原普通训练入口的 9 GiB 门槛不变。实验冒烟额外覆盖 batch 6 × 512 的 FP32 训练和 batch 32 × 512 推理峰值。
GPU 连续训练/断点恢复采用 atol=1e-7、rtol=1e-6 的逐参数容差校验；CPU 单元测试仍要求逐位一致。

- `status.json`：当前阶段、协调进程及子进程 PID、日志、失败详情。
- `experiment.json`：输入文件、基础模型、基线和实现文件 SHA-256、环境版本、固定参数。
- `runs/`：A/B 的 best、final、最近两个恢复 checkpoint、权重、曝光、训练曲线和预测。
- `baseline/`：原报告与重新推理结果，不复制基线大模型。
- `comparison/`：中文 dev/test 报告、固定排序改善/退化案例、完整指标、训练曲线、清理清单。
- `smoke/`、`unit-tests/`、`logs/`、`stages/`：验证和执行记录。

从命令行重启会验证冻结文件与已完成阶段哈希，仅继续尚未完成的阶段。
运行过程中不要编辑被 experiment.json 冻结的实现或输入，或并行启动其他训练。
若需中断，应先根据 status.json 确认子进程；正常 Ctrl+Break 会让训练器在 optimizer 边界保存。
强行结束会丢失最近 checkpoint 后的进度，但不会改动源数据。禁止通过改 manifest 来绕过指纹。

A/B 的主比较固定 final（第 8 轮），best 仅作补充；neutral 阈值只用 dev 校准，test 不重选方案。该轮最终选定 A：维度加权组成 BCE＋八维 Smooth L1、均匀抽样。它现在是普通训练入口的默认正式目标；B 的混合采样仍仅用于显式实验。

方案确定后，先用 `promote_balanced_model.py` 迁移选中的模型和报告，再用 `publish_emotion_handoff.py` 发布交接包；核对哈希后可清理整个实验根目录。不要清理源数据、基础模型缓存或正式模型目录。
