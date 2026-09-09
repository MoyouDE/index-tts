# 当前阶段对白训练数据

全量上下文语义审核已完成。train/dev/test 共 8,116 条记录全部完成审核；remaining-review.jsonl 含 7,864 条逐批人工审核台账，corrections.jsonl 中 252 条已接纳记录继续保留，3 条排除记录不进入分片。

审核后标签已直接同步写回 train/dev/test。原文、上下文、ID、split 和其他 emotions 字段保持不变；每条台账记录保留原始行哈希、审核前标签、审核后标签、理由和置信度，可用于逐条回退核验。

训练数据沿用 readest-emotion-target-context-v1，八维依次为 happy、angry、sad、afraid、disgusted、melancholic、surprised、calm，各自独立取值 0–1，全零合法。manifest.json 已更新为全量审核完成状态及最终哈希、修改统计、分布和校验结果。
