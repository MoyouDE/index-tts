# Readest 中文自动情感模型

> 当前阶段正式训练数据（2026-09-17）：`data/emotion/dialogue-stage-20260917-deepseek-v3-final-41763/`，共 **41,763 条**上下文校正对白记录。正式目录只保留 train/dev/test 三份 JSONL；旧版本、逐条审计与生成中间产物不参与训练。训练入口 `_train_emotion_inner.bat` 已指向该版本。`humanSemanticReview=false`，本版本不称为全量人工语义认证。

该模块使用 `hfl/chinese-macbert-base` 初始化中文编码器，只随机初始化八维情感头和总强度头。正式发布模型保持 FP32，不使用动态 INT8。

## 数据约束

训练 loader 同时接受历史 JSONL schema v1、连续 v3，以及正式的 `readest-emotion-target-context-v1`。目标上下文 schema 不再使用 `previousText`，而是保存同一 section 内按原文顺序排列的 `sentences[]` 和 `targetSentenceId`；唯一磁盘标签仍是八维 `emotions`。历史 20,518 条小说训练快照位于 `outputs/emotion-data/training-ready-v3-novel-tgt-context512/`，由 `training-ready-v3-r2` 标签和原文重新分句后严格映射生成，ID、split、作品归属与八维标签逐值不变。

模型输入是带原子标记的单序列：目标句渲染为 `[TGT][对白|旁白]目标正文[/TGT]`，上下文句渲染为 `[对白|旁白]正文`，跨段落插入 `[NL]`。512-token 预算按“目标 → 最近前一句 → 同 section 且同段的紧邻后一句 → 继续向前”选择，最终恢复原文顺序。上下文句必须完整；前句放不下时仍尝试同段后句但停止加入更早前文，后句放不下时可继续向前填充。任何上下文都不跨 section，只有目标自身超过上限时才对目标正文确定性右截断。

```powershell
uv run indextts-emotion prepare-emotion-target-context512 `
  --input outputs/emotion-data/training-ready-v3-r2 `
  --project-root .. `
  --output outputs/emotion-data/training-ready-v3-novel-tgt-context512
```

BRIGHTER 中文强度数据仅覆盖 `happy/angry/sad/afraid/disgusted/surprised`，因此转换器会把 `melancholic/calm` 的掩码设为 0，而不是错误地把它们标成 0。

```powershell
uv sync --extra emotion_train
uv run --project tools/emotion-data -- python -m indextts.emotion.cli prepare-brighter --output outputs/emotion-data/brighter
uv run indextts-emotion validate-data --train outputs/emotion-data/training-ready-v2/train.jsonl --dev outputs/emotion-data/training-ready-v2/dev.jsonl --test outputs/emotion-data/training-ready-v2/test.jsonl
```

当前测试训练快照位于 `outputs/emotion-data/training-ready-v2/`，状态为 `test-only`。v1 的小说 dev 只有 1 部作品、test 只有 2 部，且 test 没有 `melancholic` 正例，因此不能作为可靠的八维路线判断。v2 将全部既有标签按作品重新无泄漏切分为 10 / 2 / 3 部作品，小说 train/dev/test 为 13,522 / 2,897 / 4,135；合并 BRIGHTER 后为 16,164 / 3,097 / 6,777。dev 每维至少 100 个正例，test 每维至少 90 个正例。`resplit-report.json` 固定记录作品名单、覆盖、输入和输出 SHA-256。

重分割命令可复现当前快照；它会拒绝重复 ID、dev/test 作品重叠、未知作品及正例覆盖不足：

```powershell
uv run indextts-emotion resplit-by-work `
  --input outputs/emotion-data/training-ready-v1/train.jsonl `
  --input outputs/emotion-data/training-ready-v1/dev.jsonl `
  --input outputs/emotion-data/training-ready-v1/test.jsonl `
  --output outputs/emotion-data/training-ready-v2 `
  --dev-work CSI --dev-work 至尊仙道 `
  --test-work 人生模拟：让女剑仙抱憾终身 --test-work 攻略系统 --test-work 蛊真人 `
  --minimum-dev-positive 100 --minimum-test-positive 90
```

补标覆盖集还完成了独立 Qwen 审计：2,481 条接受样本中，语义校正后的八维向量有 2,233 条不同；其中 1,707 条严重分歧经过第二次决断，保留 agent、Qwen 和最终决断三层记录。该审计不会反向覆盖双标训练真值，详细统计见 `readest-emotion-data-pipeline.md`。

## 训练与导出

Windows 本机训练入口会调用 `_train_emotion_inner.bat`。它读取 `data/emotion/dialogue-stage-20260917-deepseek-v3-final-41763/` 的对白 train/dev/test（41,763 条），不读取 BRIGHTER；固定 512 token、batch 6、梯度累积 4（有效 batch 24），依次执行完整上下文预检、训练、dev-only neutral 阈值校准和小说 test 评估。默认目标已固定为温和维度加权组成 BCE、`0.7 ×` 总强度损失和八维 Smooth L1；默认均匀抽样，混合采样只保留为显式实验选项。启动前要求 CUDA 空闲显存不少于 9GiB、输出盘空间不少于 10GiB，条件不足时只报错，不会结束其他进程。

训练终端实时显示 epoch、batch、optimizer step、ETA、组成/强度/回归三项 loss、学习率和 CUDA 显存。每 200 个 optimizer step 以及每轮结束保存原子 checkpoint，最多保留最近两个；按一次 Ctrl+C 会在安全更新边界保存，重新双击 BAT 会按数据哈希和完整训练配置自动续训。配置或数据不匹配时拒绝恢复。固定参数为 8 epochs、强度损失权重 0.7、neutral 样本权重 3.0；正式模型固定取第 8 轮 `final/`，而非根据 test 或单项 dev 指标回挑。输出位于 `outputs/emotion-data/macbert-training-dialogue-stage-20260917-deepseek-v3-final-41763-balanced-v1/`。

阈值校准不接触 test。默认要求主动情感召回率至少 0.8，并在该约束下选择 neutral/base 误触发最低的 dev 阈值。当前正式模型的 dev 阈值为 `0.355`；BAT 每次仍会根据对应第 8 轮 checkpoint 重新计算，而不是硬编码该数值。

领域继续预训练只接受许可证白名单中的文本。其输出必须与未经继续预训练的 MacBERT 在同一小说验证集和 BRIGHTER dev 上做受控对照，再决定是否用于监督训练。

```powershell
uv run indextts-emotion pretrain-domain --corpus data/novels.txt --license PROPRIETARY-AUTHORIZED --output outputs/macbert-novel
uv run indextts-emotion train `
  --train outputs/emotion-data/training-ready-v2/train.jsonl `
  --train outputs/emotion-data/brighter/brighter-chn-train.jsonl `
  --dev outputs/emotion-data/training-ready-v2/dev.jsonl `
  --dev outputs/emotion-data/brighter/brighter-chn-dev.jsonl `
  --output outputs/emotion-data/macbert-training-v2 `
  --epochs 8 --batch-size 12 --gradient-accumulation 2 `
  --learning-rate 2e-5 --head-learning-rate 1e-4 `
  --intensity-loss-weight 0.7 --neutral-loss-weight 3.0 `
  --max-length 256 --seed 20260829 --device cuda
uv run indextts-emotion evaluate `
  --checkpoint outputs/emotion-data/macbert-training-v2/best `
  --data outputs/emotion-data/training-ready-v2/test.jsonl `
  --data outputs/emotion-data/brighter/brighter-chn-test.jsonl `
  --neutral-threshold 0.16 --max-length 256
```

以上命令与指标仅用于复现已经完成的 256-token v2 历史实验，不代表新的 512-token 路线。最佳验证 checkpoint 位于第 8 轮；在 4,135 条小说 test 上 Macro F1 约 0.593、Macro Spearman 约 0.398、强度 MAE 约 0.187、neutral 误触发约 0.234。在 6,777 条小说+BRIGHTER test 上相应为 0.534 / 0.439 / 0.172 / 0.233。使用本地 BRIGHTER JSONL 可以固定数据哈希并避免训练时联网。

同一 4,135 条小说 test 的既有 Qwen 预标注经过文本哈希校验，并把 IndexTTS 的 `natural` 默认输出语义对齐为 base 后，Macro F1 约 0.155、Macro Spearman 约 0.220、强度 MAE 约 0.363、neutral 误触发约 0.173。MacBERT 在八个维度全部领先。CPU 实测 MacBERT PyTorch batch=1 热推理平均约 37ms、p95 约 51ms，Qwen 平均约 6.8s、p95 约 7.2s；目录体积约 391MiB 对 1,152MiB。因此当前路线保留专用 MacBERT，不采用现有 Qwen 作为最终后端。

41,763 条数据上的两组不平衡实验选定 A-loss 为正式方案。相对旧基线，固定 test 的 Macro F1 从 `0.6175` 提高到 `0.6489`，Macro Spearman 从 `0.5857` 提高到 `0.5884`，八维向量 MAE 从 `0.0659` 降到 `0.0557`，零标签维度误激活率从 `0.0356` 降到 `0.0172`；总强度 MAE 基本持平（`0.0914 → 0.0917`）。B 组增加混合采样后没有进一步收益，因此不进入默认训练路径。

ONNX 发布状态只依赖可复现的自动质量证据：训练报告、dev 指标、test 指标和实验最终验收报告。`release-check` 只校验这些文件的结构、有限数值、最终轮信息与 SHA-256，不要求额外人工审批材料。

```powershell
$env:PYTHONPATH = (Resolve-Path .)
uv run indextts-emotion release-check --training-report <training-report.json> --dev-metrics <dev-metrics.json> --test-metrics <test-metrics.json> --experiment-audit <final-audit.json> --output <release-approval.json>
uv run --project tools/emotion-onnx python -m indextts.emotion.cli export-onnx --checkpoint <final> --output <onnx> --release-approval <release-approval.json>
uv run --project tools/emotion-onnx python -m indextts.emotion.cli verify-onnx --checkpoint <final> --model-dir <onnx>
```

ONNX 使用独立工具环境，是因为 IndexTTS 上游音频依赖要求旧版 protobuf，而现代 ONNX 要求新版 protobuf；两个环境必须隔离，不能通过升级主环境 protobuf 绕过。

导出的情感 manifest 使用 `conditioningAbi=readest-emotion-v3`，并固定 `contextPolicy=target-prev-next-same-line-backfill-prev-section-v1`、`contextEncoding=tgt-type-marked-single-sequence-v1`、五个 special token 及其 ID 与 `maxLength=512`。tokenizer 与模型 embedding 必须包含同一套新增词；导出时不匹配会直接失败。

阅读器向 Tauri 传入目标所在 section 的完整句序列和目标 ID。Rust 使用模型 tokenizer 重现与训练完全相同的选择、标记渲染和预算规则，返回采用句 ID、是否采用同段后句、实际 token 数、预算受限及目标截断诊断。情感管线仍复用 `SentencesCache`，当前句在音频合成前等待分析，后续最多五句按各自 section 后台预计算；不会跨章节取上下文，也不会让未来句进入非紧邻或跨段目标。跳转、文本修订和销毁会取消旧 generation，失败则挂载 base fallback，不阻塞播放。本阶段仍不把向量传给具体 TTS 供应商。

质量文件不手写，由 `release-check` 组合四份固定产物并记录 SHA-256。导出后必须运行 `verify-onnx`，同时校验 PyTorch/ONNX 数值误差、ABI、tokenizer 原子 special token 和包内文件哈希。
