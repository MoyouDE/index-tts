# BV1RYEc65EYg 新素材准备

## 当前状态

2026-10-02 已提取公开完整 AAC 音轨，导入现有素材工作区并用固定 Silero VAD v6.0 CPU 分段。尚未编码、制包或合成；目标人物一致性、BGM／音效污染和主参考适用性等待用户试听确认。本素材不沿用上一条视频的人工确认，也不与旧素材混合身份向量。

- 视频：[BV1RYEc65EYg](https://www.bilibili.com/video/BV1RYEc65EYg/)，单 P。
- 实际音轨 561.152 秒；公开接口提供的最高码率 AAC 约 110 kbps，原文件 7,714,506 字节。
- VAD 检测语音区间总计 175.168 秒，133 段。区间包含正常短停顿；此数值不等于有效制包素材量。
- 工作区默认选中的全部分段总计约 215.07 秒，包含边界余量。未经人工筛选，不代表全部可用。
- 保留原 AAC；解码为现有工具的 22,050 Hz 单声道浮点音轨及 16,000 Hz VAD 音轨，无降噪、人声分离或自动说话人识别。

## 候选与试听

`candidates.json` 只是待确认的候选范围，不是已经通过验收的实验配方。除主参考候选 R01 外，其他训练候选按相邻 VAD 区间组合，保留中间停顿；每个片段不超过 15 秒。没有改动素材库原有分段与选择记录。

|编号|源音轨区间（秒）|用途|
|---|---|---|
|R01|66.186–79.606|拟用主参考|
|R02|231.498–242.678|身份融合候选|
|R03|251.050–261.238|身份融合候选|
|R04|354.058–366.774|身份融合候选|
|R05|461.322–470.966|身份融合候选|
|H01|15.978–25.174|留出评价，不参与编码|
|H02|541.354–546.230|留出评价，不参与编码|

本机试听产物在忽略 Git 的 `outputs/reference-review-BV1RYEc65EYg-20261002/`：逐段浮点 WAV、`training-review.wav`（R01→R05，61.148 秒）、`held-out-review.wav`（H01→H02，15.072 秒）及 `review.json`。拼接仅在片段之间加入 1 秒零值间隔，不调整片段内部停顿或音量。`candidate-review.json` 记录各段哈希及试听时间轴。

准备校验逐样本比较每段 WAV 与解码音轨相应区间，全部一致；拼接与原片段加间隔逐样本一致。校验进程未导入 PyTorch、未加载制包或试听模型。该校验不能判断人物一致性或声音污染。

## 复现

在 IndexTTS 根目录使用现有 `validation-web` 环境：

```powershell
.venv/Scripts/python.exe tools/prepare_bilibili_audio.py BV1RYEc65EYg --output-dir <本地素材目录> --workspace outputs/voice-workbench
.venv/Scripts/python.exe tools/preview_reference_candidates.py --material-id <导入返回的sourceId> --manifest docs/validation/new-material-2026-10-02/candidates.json --output outputs/reference-review-new-run
```

两条命令都不覆盖已有产物目录。下载器只使用公开接口，没有登录、保存 cookie 或绕过访问限制；没有完整音轨时直接报错。视频仅支持单 P。FFmpeg 使用当前工具既有实现，VAD 依赖既有固定资产锁。

导入 UUID 因本机而异，应使用实际返回的 ID，不能直接复用记录中的本机 ID。上游音轨可能变更，应比较 `source.json` 的原音轨 SHA256；不同 FFmpeg 版本解码可能产生差异，不要求跨设备 WAV 字节相同。

## 后续对照

用户确认后才固定训练／留出范围和主参考。沿用 `tools/reference_fusion_experiment.py` 的 FP32、基础情感、语速 1.0、四段文本和种子 17／29，独立比较单段、原向量等权平均、归一化后平均再恢复平均范数。保持主参考的四个非身份张量不变，重新投影 `speaker_latent`；编码片段串行且每段最多 15 秒。

空白对照需要另外固定并确认含内部长空白的完整区间。未经确认不运行该组；不默认复用 0–15 秒或提前判定缩短空白更好。多段方法继续隔离于正式音色库，是否采用仍由跨文本／种子试听决定。

原始媒体、解码音轨、参考片段与运行缓存仅在本地。Git 保存脚本、范围、源资产指纹、VAD 记录和准备校验结果；主仓库与交接目录不改。
