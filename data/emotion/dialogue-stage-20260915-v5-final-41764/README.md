# Readest 对白情感训练数据 v5

这是当前唯一的 Readest 对白情感正式训练资源，共 41,764 条目标上下文记录：

- `train.jsonl`：27,627 条
- `dev.jsonl`：6,234 条
- `test.jsonl`：7,903 条
- `manifest.json`：schema、字段清单、条数与 SHA-256

每行严格使用 `readest-emotion-target-context-v1`，只保留训练加载器需要的上下文、目标句、八维情感、来源和许可字段。八维顺序为 `happy`、`angry`、`sad`、`afraid`、`disgusted`、`melancholic`、`surprised`、`calm`。

校验命令：

```powershell
uv run indextts-emotion validate-data `
  --train data/emotion/dialogue-stage-20260915-v5-final-41764/train.jsonl `
  --dev data/emotion/dialogue-stage-20260915-v5-final-41764/dev.jsonl `
  --test data/emotion/dialogue-stage-20260915-v5-final-41764/test.jsonl
```

全部记录的 `licenseStatus` 为 `test-only`，可用于当前内部训练和评测，但不表示数据或模型可对外发布。`humanSemanticReview=false`；本版本经过机器结构校验和上下文模型标注，不称为全量人工语义认证。
