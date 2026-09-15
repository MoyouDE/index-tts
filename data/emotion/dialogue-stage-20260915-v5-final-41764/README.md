# 最终情感训练资源

共 41,763 条：按原 split 及保留记录的原始顺序组织。文件只含正式训练字段；审核、API、初判、日志和排除审计均在仓库外。

全量一审及残余 2,082 条风险复核已完成；其中 1 条多说话者混杂目标经用户批准排除。复核后完成 240 条风险分层语义抽检及 54 条定向上下文复查，另修正最终结构核对发现的 2 条旧记录多维异常，并通过官方 validate-data。抽检不是全量人工认证，humanSemanticReview=false。

目录名中的 41764 保留以兼容现有资源路径；实际条数以 manifest.json 为准。正文、目标句和保留记录的 split、相对顺序未改，仅更新情感标签并排除上述 1 条。

许可仍为 test-only，仅供内部测试、训练和评估；releaseEligible=false，不声明可对外发布。

文件：train.jsonl、dev.jsonl、test.jsonl、manifest.json、README.md、.gitattributes。
