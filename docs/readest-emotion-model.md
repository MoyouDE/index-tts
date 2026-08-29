# Readest 中文自动情感模型

该模块使用 `hfl/chinese-macbert-base` 初始化中文编码器，只随机初始化八维情感头和总强度头。正式发布模型保持 FP32，不使用动态 INT8，也不使用 Qwen 输出作为伪标签。

## 数据约束

训练记录使用 JSONL schema v1。每条记录必须提供作品 ID、上一句、当前句、旁白/对白类型、八维 0–1 强度、标签掩码和许可证元数据。切分必须以作品为单位；校验器会拒绝完整上下文跨集合泄漏。质量优先的训练和推理上限固定为 256 token；当前合并 BRIGHTER 后 train/test 超长率仅 0.197%/0.192%。

BRIGHTER 中文强度数据仅覆盖 `happy/angry/sad/afraid/disgusted/surprised`，因此转换器会把 `melancholic/calm` 的掩码设为 0，而不是错误地把它们标成 0。正式发布还要求至少 12,000 条完整八维、权利明确的中文小说上下文标注。

```powershell
uv sync --extra emotion_train
uv run --project tools/emotion-data -- python -m indextts.emotion.cli prepare-brighter --output outputs/emotion-data/brighter
uv run indextts-emotion validate-data --train outputs/emotion-data/training-ready-v1/train.jsonl --dev outputs/emotion-data/training-ready-v1/dev.jsonl --test outputs/emotion-data/training-ready-v1/test.jsonl
```

当前测试训练快照位于 `outputs/emotion-data/training-ready-v1/`，状态为 `test-only`，`testTrainingReady=true`、`trainingStarted=false`。小说 train/dev/test 为 16,643 / 1,339 / 2,572；合并 BRIGHTER 后为 19,285 / 1,539 / 5,214。八个情感维度的 train 正例均不少于 1,000，数据、MacBERT FP32 权重和 RTX 4070 SUPER CUDA 环境均已通过只读预检。授权审计及历史 Qwen 对照来源仍会阻断 `--release`，但不阻断用户要求的内部测试训练。

补标覆盖集还完成了独立 Qwen 审计：2,481 条接受样本中，语义校正后的八维向量有 2,233 条不同；其中 1,707 条严重分歧经过第二次决断，保留 agent、Qwen 和最终决断三层记录。该审计不会反向覆盖双标训练真值，详细统计见 `readest-emotion-data-pipeline.md`。

## 训练与导出

领域继续预训练只接受许可证白名单中的文本。其输出必须与未经继续预训练的 MacBERT 在同一小说验证集和 BRIGHTER dev 上做受控对照，再决定是否用于监督训练。

```powershell
uv run indextts-emotion pretrain-domain --corpus data/novels.txt --license PROPRIETARY-AUTHORIZED --output outputs/macbert-novel
uv run indextts-emotion train `
  --train outputs/emotion-data/training-ready-v1/train.jsonl `
  --train outputs/emotion-data/brighter/brighter-chn-train.jsonl `
  --dev outputs/emotion-data/training-ready-v1/dev.jsonl `
  --dev outputs/emotion-data/brighter/brighter-chn-dev.jsonl `
  --output outputs/emotion-data/macbert-training-v1 `
  --epochs 5 --batch-size 12 --gradient-accumulation 2 `
  --learning-rate 2e-5 --head-learning-rate 1e-4 `
  --max-length 256 --seed 20260829 --device cuda
uv run indextts-emotion evaluate `
  --checkpoint outputs/emotion-data/macbert-training-v1/best `
  --data outputs/emotion-data/training-ready-v1/test.jsonl `
  --data outputs/emotion-data/brighter/brighter-chn-test.jsonl `
  --max-length 256
```

以上训练命令是当前内部测试训练的准确启动方式，但尚未执行。它刻意不带 `--release`；使用本地 BRIGHTER JSONL 而不是 `--include-brighter`，可以固定数据哈希并避免训练时重新访问 Hugging Face。

候选 ONNX 可以直接导出，但 manifest 会标记为 `candidate-unvalidated`。只有数据审计、模型指标和 TTS 人工盲测均通过的批准 JSON 才会生成 `approved` manifest。

```powershell
$env:PYTHONPATH = (Resolve-Path .)
uv run --project tools/emotion-onnx python -m indextts.emotion.cli export-onnx --checkpoint outputs/emotion-v1/best --output outputs/emotion-v1/onnx
uv run --project tools/emotion-onnx python -m indextts.emotion.cli verify-onnx --checkpoint outputs/emotion-v1/best --model-dir outputs/emotion-v1/onnx
```

ONNX 使用独立工具环境，是因为 IndexTTS 上游音频依赖要求旧版 protobuf，而现代 ONNX 要求新版 protobuf；两个环境必须隔离，不能通过升级主环境 protobuf 绕过。

发布批准文件必须将 `qualityGatePassed`、`humanBlindTestPassed`、`commercialDataAuditPassed`、`noQwenPseudoLabels` 四项设为 `true`，并携带训练报告、MacBERT/Qwen 指标、盲测和数据审计五份输入的 SHA-256。发布包需要保留 MacBERT Apache-2.0 声明、BRIGHTER CC BY 4.0 署名以及自建语料授权记录。

批准文件不手写。先在同一人工测试集运行 `benchmark-qwen`，再由 `release-check` 组合训练报告、MacBERT/Qwen 指标、至少 100 项 TTS 盲测和商业数据审计；任何一项未通过都会拒绝生成批准文件。
