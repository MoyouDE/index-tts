# Readest Qwen 原始连续八维情感数据 v3

## 目标与边界

`readest-emotion-continuous-v3` 是一份内部测试用的连续八维伪标签数据。固定输入全集由 `training-ready-v2` 的 20,554 条小说样本和本地 BRIGHTER 的 5,484 条中文样本组成，共 26,038 条。本轮只生成、复核、审计和验证数据，不训练模型，也不修改 MacBERT、ONNX、阅读器或 IndexTTS 运行时。

v3 的唯一训练标签是 `emotions`：

```json
{
  "happy": 0.0,
  "angry": 0.0,
  "sad": 0.0,
  "afraid": 0.0,
  "disgusted": 0.0,
  "melancholic": 0.0,
  "surprised": 0.0,
  "calm": 0.0
}
```

每维都是 `[0,1]` 内的有限连续浮点数，同时表示方向和强度；不量化、不归一化，也不再生成 `intensity`、`primaryEmotion` 或 `labelMask`。八维全零是合法的 base/neutral 标签，表示目标句没有主动情感。未来模型应直接输出八维 sigmoid；全零输入 TTS 时由音色包的 `base_emotion` 自然接管，但这项模型和运行时改造不属于本轮。

## Qwen 原始标注协议

本地模型固定使用 `qwen0.6bemo4-merge` 和它微调时的原始消息形状：system 消息只有 `文本情感分类`，user 消息只有目标句。生成时关闭 thinking，设置 `do_sample=False`，不向模型提供上文、`dialogue`/`narration` 句型、旧标签或期望情感方向。

三种输入方式的 10 条真实对照表明，加入“上文/目标句/句型”会使部分样本显著漂移，而要求模型只在指定方向内评分时 10/10 违反约束。因此 v3 不再使用 Qwen 做方向约束重评分；上文和句型只供后续 Codex Agent 理解目标句。

原始响应必须是仅含八个中文键的 JSON 对象：`高兴/愤怒/悲伤/恐惧/反感/低落/惊讶/自然`。缺键、多键、非 JSON、布尔值、非数值、NaN、Infinity 或越界值均判为失败，每条最多生成三次。英文存储维度按固定顺序映射为 `happy/angry/sad/afraid/disgusted/melancholic/surprised/calm`。

本次真实运行中有 48 条在三次确定性生成后仍输出非法 JSON 或非法数值。经人工明确授权，这 48 条不做补键、截断、归一化或 Agent 代填，而是写入 `excluded-invalid.jsonl` 后从 v3 有效全集排除。排除审计完整保存 ID、split、来源、输入/模型/提示哈希及三次原始响应；其余 25,990 条才进入方向冲突检测和 Agent 复核。

这里必须绕过 `QwenEmotion.convert()`。该 Python 包装器是在 Qwen JSON 已生成并解析后才把全零向量改成 `calm=1`，还包含低落关键词触发的 `sad/melancholic` 交换；两种后处理都不得进入 v3。Qwen 原始全零必须原样保存，只有原始响应中的 `自然` 数值才映射到 `calm`。

Qwen 阶段保存原始响应、解析向量、每次尝试及错误、模型指纹、提示指纹、候选文件哈希、`inputSha256` 和 `textSha256`。结果按 ID/分片原子落盘；续跑时如候选内容、顺序、输入哈希、模型、提示或参数与既有 manifest 不一致，立即拒绝复用。

## 候选与旧方向证据

小说样本的旧标签只作为离散方向证据，旧数值强度不得复制到 v3：

- 普通记录取 v2 `emotions` 中全部非零维，同时作为 `oldDirections` 和 `oldDominantDirections`；全零代表 base。
- 954 条历史 `qwen`/`revised` 记录不能使用已经裁决后的向量，必须从 `training-prepared-v1/adjudication-audit.jsonl` 的 `originalEmotions` 恢复原 Agent 方向。恢复缺失、ID 重复或哈希不一致都会阻止准备阶段完成。

BRIGHTER 只观察 `happy/angry/sad/afraid/disgusted/surprised` 六维。六维中的非零项形成旧方向证据；未标注的 `melancholic/calm` 是 unknown，不得按零、负例或 base 证据处理。

BRIGHTER 原数据没有句型。v3 使用可审计的保守规则：去除首尾空白后，只有文本整体被一对 `“”`、`「」`、`『』`、`‘’`、`""` 或 `''` 包围时才标为 `dialogue`，其余均为 `narration`。规则版本记录在候选审计和 manifest 中。

候选 schema 为 `readest-emotion-continuous-candidate-v3`，至少保存 `id`、原 split、`workId`、`previousText`、`text`、`sentenceType`、`sourceKind`、来源/许可证元数据、观察维度、旧方向证据、规范化输入哈希和目标文本哈希。v1/v2、旧 Agent 结果和旧 Qwen 对照数据始终只读。

## 方向冲突与 Codex Agent 单轮校正

Qwen 值 `>=0.10` 的维度构成显著方向集合。若没有显著维度，主导集合为空；否则主导集合包含所有“与八维最大值之差不超过 `0.10`”的显著维度。

Qwen 显著方向集合与旧方向集合完全一致时直接保留 Qwen，不因连续强弱或主导集合差异单独进入复核。其余样本在 base/主动状态冲突、旧方向未进入 Qwen 显著集合或方向集合不一致时进入复核。BRIGHTER 只在六个已观察维度上比较；`melancholic/calm` 不制造冲突，也不能据此判定 BRIGHTER 为 base。

本次有效数据中有 7,977 条方向一致记录直接保留 Qwen，另外 18,013 条进入单轮 Agent 校正。校正队列按每批 100 条切分，并优先把旧方向、Qwen 显著/主导方向相同或接近的记录放在同批，再按方向上的 Qwen 强度线索排序。每个 Agent 必须先通读整批，在语义相近记录之间比较轻微、明确、强烈和极端表达的相对强弱，然后逐条输出。批内比较不能变成强制排名、归一化或固定档位。

每条记录只由一个 Codex Agent 判断一次。可以并发处理互不重叠的批次。Agent 能看到目标句、仅用于消歧的上文、句型、Qwen 原始连续向量、旧离散方向及冲突原因，但看不到旧强度。每条输出必须严格包含：

```json
{
  "decision": "keep",
  "emotions": {
    "happy": 0.0,
    "angry": 0.0,
    "sad": 0.0,
    "afraid": 0.0,
    "disgusted": 0.0,
    "melancholic": 0.0,
    "surprised": 0.0,
    "calm": 0.0
  },
  "confidence": "high",
  "reason": "只说明目标句的情感判断依据"
}
```

`decision` 只能是 `keep` 或 `revise`。`keep` 的八维向量必须与首次合法 Qwen 向量完全一致；`revise` 可以给出全零、主动 `calm` 或其他合法连续值，不受旧四档取值限制。

单轮文件严格覆盖全部 18,013 条冲突记录后，直接用于最终化。缺行、多行、重复 ID、乱序、哈希不匹配、非法字段/值域或 `keep` 改写 Qwen 向量时，汇总和最终化必须失败。系统不得补零、截断、回退旧离散强度或静默丢弃样本。

## 工具流程与输出

数据工具按以下阶段运行；各命令均写入 `outputs/emotion-data/` 下的新目录，不覆盖旧快照：

```powershell
uv run indextts-emotion prepare-continuous-v3 ...
uv run indextts-emotion annotate-qwen-continuous-v3 ...
uv run indextts-emotion materialize-qwen-continuous-v3-valid ...
uv run indextts-emotion prepare-continuous-v3-reviews ...
uv run indextts-emotion prepare-continuous-v3-comparison-batches ...
uv run indextts-emotion validate-continuous-v3-reviews ...
uv run indextts-emotion assemble-continuous-v3-reviews ...
uv run indextts-emotion finalize-continuous-v3 --reviews agent-reviews.jsonl ...
```

最终目录固定为 `outputs/emotion-data/training-ready-v3/`：

| split | 小说 | BRIGHTER | 合计 |
| --- | ---: | ---: | ---: |
| train | 13,496 | 2,636 | 16,132 |
| dev | 2,896 | 200 | 3,096 |
| test | 4,126 | 2,636 | 6,762 |
| 总计 | 20,518 | 5,472 | 25,990 |

`train.jsonl`、`dev.jsonl` 和 `test.jsonl` 的每条最终记录使用 `schema: readest-emotion-continuous-v3` 与 `schemaVersion: 3`，保留 ID、split、作品、目标句、上文、句型、来源和许可证元数据。唯一标签字段是顺序固定、键完整的八维 `emotions`；不得出现 `intensity`、`primaryEmotion`、`labelMask` 或其他标签字段。

`manifest.json` 和 `report.json` 记录输入输出哈希、split/来源/句型计数、每维均值/分位数/严格非零率、全零率、并列主导分布，以及冲突原因、复核队列、Qwen 保留和 Agent 修订数量。`audit/` 保存候选、Qwen 原始响应、冲突清单、单轮 Agent 结果及每条最终向量来源。

最终化必须确认 25,990 个有效 ID 全部且只出现一次、48 个排除 ID 全部且只出现在排除审计中，调整后的 split 计数和小说作品级切分不变，所有向量键序和值域合法，并允许严格全零。任何数量、作品归属、哈希、复核、排除审计或 schema 校验失败都会阻止生成训练就绪快照。

## 历史三句完整上文快照

当前小说标签从 `training-ready-v3-r2` 派生到 `outputs/emotion-data/training-ready-v3-novel-context3/`。派生过程只替换 `previousText`，全部 20,518 条小说记录的 ID、split、句型、来源字段和 `emotions` 逐值不变；train/dev/test 分别为 13,496 / 2,896 / 4,126。

`previousText` 语义固定为目标句前最多三个完整句子，按最早到最近用换行连接。工具从目标 ID 指定的原始标注窗口和 `sentenceId` 精确取句，从最近前句开始用本地 MacBERT tokenizer 试加；一旦完整 pair 超过 256 token 就停止，不截取该句，也不跳过该句去选更远文本。目标句自身超长时上文为空，由 tokenizer 对目标做确定性右截断，并进入异常审计。

```powershell
uv run indextts-emotion expand-continuous-v3-context `
  --input outputs/emotion-data/training-ready-v3-r2 `
  --source-root ../speaker-id/typescript/data `
  --output outputs/emotion-data/training-ready-v3-novel-context3 `
  --base-model hfl/chinese-macbert-base --max-length 256
```

本次固定分布为 0/1/2/3 个前句 522 / 790 / 973 / 18,233 条，10 条目标句自身超长。`audit/context-expansion.jsonl` 为每条记录保存前句 ID、原始与扩充上文、实际 token 数、停止原因和输入输出哈希。

## 当前 TGT 关联窗口快照

当前正式输入快照为 `outputs/emotion-data/training-ready-v3-novel-tgt-context512/`，schema 是 `readest-emotion-target-context-v1`。它从 `training-ready-v3-r2` 只读取 20,518 条小说标签，再从普通小说原文或 CSI/JY/WP2021 clean 原文及 manifest 重新建立 section、段落和完整句序列。旧标注窗口只作为目标严格映射证据，不再作为模型上下文。

每条记录保存按原文顺序排列的 `sentences[]` 与 `targetSentenceId`，不保存 `previousText`。512-token 选择顺序是目标、最近前一句、同 section 且同 `lineIndex` 的紧邻后一句、继续向前；最终渲染为带 `[TGT]`、`[/TGT]`、`[旁白]`、`[对白]`、`[NL]` 原子标记的单序列。任何上下文都不能跨 section，紧邻后句不能跨段落。

```powershell
uv run indextts-emotion prepare-emotion-target-context512 `
  --input outputs/emotion-data/training-ready-v3-r2 `
  --project-root .. `
  --output outputs/emotion-data/training-ready-v3-novel-tgt-context512
```

输出继续固定为 train/dev/test 13,496 / 2,896 / 4,126，且每条 `emotions`、ID、split 和作品归属逐值不变。`audit/source-sections.jsonl` 保存重采集句序列，`audit/context-selection.jsonl` 保存目标映射方法、采用句、实际 token 数、停止原因和输入输出哈希；任何缺失、歧义或非法映射都会阻止最终化。
