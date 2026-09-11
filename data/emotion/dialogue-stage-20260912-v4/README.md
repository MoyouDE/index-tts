# 对白情感训练数据 v4（冻结版）

本目录是"对白情感训练数据扩充"任务的**冻结发布版**：以 `dialogue-stage-20260907-v1` 的 8,116 条全量复核数据为基础，加上已完成八维审核写回的 **33,648 条**新增对白，合计 **41,764 条**。

## 构成

| 项 | 条数 |
|---|---:|
| 基线（来自 v1 全量复核） | 8,116 |
| 本轮已完成审核的新增对白 | 33,648 |
| **合计** | **41,764** |

原计划的 50,000 条对应"基线 8,116 + 新增候选 41,884"。本轮决定不再继续标注剩余的 **8,236 条**未标注候选，因此本版本冻结在 41,764 条。未标注候选池保留在源工作目录，未纳入本仓库：

- 归档路径：`E:\Projects\Readest\.codex\artifacts\emotion-data-archive-20260912\dialogue-stage-20260909-v3\expansion-candidates.jsonl`
- 条数：41,884，SHA-256：`98637EDE8430B17BBB3906ED6F1D61D3EB22774815CE8CD2DA44E22F3FD82F2C`

后续如恢复标注，从该候选池继续即可；本目录的 `expansion-review.jsonl` 是已写回台账，可直接对照。

## 数据规格

- schema：`readest-emotion-target-context-v1`（每行含同段落 `sentences[]` 与 `targetSentenceId`，可直接作为模型上下文；唯一标签字段是八维 `emotions`）
- 八维顺序：happy、angry、sad、afraid、disgusted、melancholic、surprised、calm，每维独立取值 0–1，全零合法
- **目标句 100% 为对白**
- 作品划分：共 15 部，train 10 部 / dev 2 部 / test 3 部，dev、test 与 train **无作品重叠**
- 上下文规模：每行 p50 451 字 / p95 482 字 / 最大 790 字

## 分片

| split | 条数 | 作品数 | SHA-256 |
|---|---:|---:|---|
| train | 27,627 | 10 | `A4E93E52BCEE7EF655C98367210D69486374D3EF1DE7D770932629C9F119C414` |
| dev | 6,234 | 2 | `18A5262080F7316598DB45468F95BFE12A1FA631CDF5F8A35D0C0116D9D78D79` |
| test | 7,903 | 3 | `FD1B0E2BA3A11E00C6731E0A196E3574FAE4DE44F93977BD0A068A3A9BAE5C4C` |

## 标签分布

全零记录 968 条（2.32%）。

| 维度 | 非零条数 | 非零占比 | 均值 |
|---|---:|---:|---:|
| calm | 31,365 | 75.1% | 0.167 |
| surprised | 22,116 | 53.0% | 0.096 |
| angry | 20,709 | 49.6% | 0.103 |
| happy | 20,666 | 49.5% | 0.074 |
| afraid | 19,995 | 47.9% | 0.077 |
| sad | 16,776 | 40.2% | 0.044 |
| disgusted | 16,365 | 39.2% | 0.040 |
| melancholic | 14,215 | 34.0% | 0.026 |

## 文件

| 文件 | 说明 |
|---|---|
| `train.jsonl` / `dev.jsonl` / `test.jsonl` | 训练/验证/测试分片，共 41,764 条 |
| `expansion-review.jsonl` | 新增对白的逐条审核台账（33,648 条），含理由与置信度，用于逐条回退核验 |
| `corrections.jsonl` | v1 阶段的接受修订记录（255 条） |
| `manifest.json` | 本版本的统计、哈希与校验信息 |
| `expansion-candidate-manifest.json` | 候选池的生成清单（抽样、分层与 split 目标） |
| `qwen-initial-reference/` | 本地 Qwen 初始参考标签，仅用于分组与复核触发，**不是人工真值**，从未写入分片 |

## 许可

全部记录 `licenseStatus` 为 `test-only`。本地训练与评测可用；如需对外发布模型或数据，须另行确认授权范围。

## 使用与校验

训练入口 `_train_emotion_inner.bat` 已指向本目录，输出目录为 `outputs/emotion-data/macbert-training-dialogue-stage-20260912-v4/`。schema 与训练 loader 兼容，`training_preflight` 会报出超长截断数与每维正例覆盖。

核验关系式（与 `dialogue-stage-20260909-v3` 完全一致）：

```
train + dev + test == 8,116(基线) + expansion-review 行数
41,764            == 8,116        + 33,648
```
