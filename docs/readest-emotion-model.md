# Readest 中文自动情感模型

该模块使用 `hfl/chinese-macbert-base` 初始化中文编码器，只随机初始化八维情感头和总强度头。正式发布模型保持 FP32，不使用动态 INT8，也不使用 Qwen 输出作为伪标签。

## 数据约束

训练 loader 同时接受历史 JSONL schema v1 和仅含 `emotions` 标签字段的连续 v3 schema。连续 v3 在内存中派生全维监督掩码及 `max(emotions)` 辅助强度，不向磁盘快照回写 `labelMask` 或 `intensity`。`previousText` 的当前语义是目标句前最多三个完整句子，按最早到最近以换行连接。训练与推理上限固定为 256 token；上文逐句试加，遇到超限立即停止，只有目标自身超长时才允许 tokenizer 右截断。

BRIGHTER 中文强度数据仅覆盖 `happy/angry/sad/afraid/disgusted/surprised`，因此转换器会把 `melancholic/calm` 的掩码设为 0，而不是错误地把它们标成 0。正式发布还要求至少 12,000 条完整八维、权利明确的中文小说上下文标注。

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

Windows 本机测试训练优先双击仓库根目录的 `train-emotion.bat`。脚本固定使用当前 `.venv`、FP32 MacBERT、v2 小说数据和已落盘 BRIGHTER，依次执行预检、训练、dev-only neutral 阈值校准、小说 test 评估和小说+BRIGHTER 综合 test；不会安装依赖、下载模型、启动 TensorBoard 或启用 `--release`。启动前要求 CUDA 空闲显存不少于 9GiB、输出盘空间不少于 10GiB，条件不足时只报错，不会结束其他进程。

训练终端实时显示 epoch、batch、optimizer step、ETA、三项 loss、学习率和 CUDA 显存。每 200 个 optimizer step 以及每轮结束保存原子 checkpoint，最多保留最近两个；按一次 Ctrl+C 会在安全更新边界保存，重新双击 BAT 会按数据哈希和完整训练配置自动续训。配置或数据不匹配时拒绝恢复，训练完成后再次启动只跳过优化并重新验证最佳模型。固定参数为 8 epochs、强度损失权重 0.7、neutral 样本权重 3.0；输出位于 `outputs/emotion-data/macbert-training-v2/`，并包含 `best/`、`training-report.json`、`threshold-calibration.json`、`neutral-threshold.txt`、`novel-test-metrics.json` 和 `test-metrics.json`。

阈值校准不接触 test。默认要求主动情感召回率至少 0.8，并在该约束下选择 neutral/base 误触发最低的 dev 阈值。当前受控实验得到 0.16；BAT 每次仍会根据对应最佳 checkpoint 重新计算，而不是硬编码该数值。

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

以上路线已经在 RTX 4070 SUPER 完整执行。最佳验证 checkpoint 位于第 8 轮；在 4,135 条小说 test 上 Macro F1 约 0.593、Macro Spearman 约 0.398、强度 MAE 约 0.187、neutral 误触发约 0.234。在 6,777 条小说+BRIGHTER test 上相应为 0.534 / 0.439 / 0.172 / 0.233。它刻意不带 `--release`；使用本地 BRIGHTER JSONL 可以固定数据哈希并避免训练时联网。

同一 4,135 条小说 test 的既有 Qwen 预标注经过文本哈希校验，并把 IndexTTS 的 `natural` 默认输出语义对齐为 base 后，Macro F1 约 0.155、Macro Spearman 约 0.220、强度 MAE 约 0.363、neutral 误触发约 0.173。MacBERT 在八个维度全部领先。CPU 实测 MacBERT PyTorch batch=1 热推理平均约 37ms、p95 约 51ms，Qwen 平均约 6.8s、p95 约 7.2s；目录体积约 391MiB 对 1,152MiB。因此当前路线保留专用 MacBERT，不采用现有 Qwen 作为最终后端。

当前第 4 至第 8 轮的 dev Macro F1 已基本停留在约 0.54 至 0.55，继续增加 epoch 不再是默认路线。下一轮质量优化的优先级是：先补充训练集仍缺口最大的 `disgusted/melancholic` 真值及主动 `calm` 困难样本，再试零膨胀的 activation/strength 双门控；只有在固定 v2 test 和 TTS 盲听上确有提升时才更换结构。更大中文编码器需要用相同门槛做受控对照，不能仅凭参数量替换当前模型。

路线采用固定停损规则：连续三个完整 epoch 的最佳 Macro F1 提升不足 0.005 时，不再仅增加轮数或进行无目标学习率搜索；先检查类别覆盖、上下文结构和标签分歧。一次数据或结构改造只有在固定 test 上带来至少 0.015 Macro F1，或在不少于 100 条的 TTS 盲听中带来至少 5 个百分点偏好提升且关键指标不退化，才进入下一轮。连续两次有实质差异的改造仍未过线时，冻结自研训练并与更大编码器及 Qwen 做同集对照。只有当候选 Qwen 的八维指标和盲听均不低于 MacBERT，同时 neutral 误触发、CPU 延迟和分发体积满足阅读器预算时，才切换为 Qwen；当前 Qwen 明显未达到这一门槛。

候选 ONNX 可以直接导出，但 manifest 会标记为 `candidate-unvalidated`。只有数据审计、模型指标和 TTS 人工盲测均通过的批准 JSON 才会生成 `approved` manifest。

```powershell
$env:PYTHONPATH = (Resolve-Path .)
uv run --project tools/emotion-onnx python -m indextts.emotion.cli export-onnx --checkpoint outputs/emotion-v1/best --output outputs/emotion-v1/onnx
uv run --project tools/emotion-onnx python -m indextts.emotion.cli verify-onnx --checkpoint outputs/emotion-v1/best --model-dir outputs/emotion-v1/onnx
```

ONNX 使用独立工具环境，是因为 IndexTTS 上游音频依赖要求旧版 protobuf，而现代 ONNX 要求新版 protobuf；两个环境必须隔离，不能通过升级主环境 protobuf 绕过。

导出的情感 manifest 使用 `conditioningAbi=readest-emotion-v2`，并固定 `contextPolicy=nearest-three-complete-preceding-v1`、`contextSentenceLimit=3`、`contextDelimiter="\n"` 与 `maxLength=256`。阅读器向 Tauri 传入按最早到最近排列的 `previousSentences`；Rust 使用模型 tokenizer 重现完整句预算，并在结果中返回采用句数、实际输入 token 数和是否因预算受限。

阅读器侧情感管线复用全书 `SentencesCache`：当前句在音频合成前等待分析完成，后续最多五句在后台预计算。每个目标仅回溯自身之前的三句，可跨 section；跳转、文本修订和销毁会取消旧 generation，缓存则同时绑定书籍、文本修订、目标指纹、模型版本和上下文策略。分析失败只给当前句挂载 base fallback，不阻塞播放。本阶段仍不把向量传给具体 TTS 供应商。

发布批准文件必须将 `qualityGatePassed`、`humanBlindTestPassed`、`commercialDataAuditPassed`、`noQwenPseudoLabels` 四项设为 `true`，并携带训练报告、MacBERT/Qwen 指标、盲测和数据审计五份输入的 SHA-256。发布包需要保留 MacBERT Apache-2.0 声明、BRIGHTER CC BY 4.0 署名以及自建语料授权记录。

批准文件不手写。既有 Qwen 输出可用 `benchmark-qwen-annotations` 做 ID、文本哈希和完整覆盖校验，避免重新执行数千次生成；新 Qwen 后端仍应使用 `benchmark-qwen` 实际运行。随后由 `release-check` 组合训练报告、MacBERT/Qwen 指标、至少 100 项 TTS 盲测和商业数据审计；任何一项未通过都会拒绝生成批准文件。
