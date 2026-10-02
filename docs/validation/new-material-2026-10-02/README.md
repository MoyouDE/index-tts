# BV1RYEc65EYg 新素材准备

## 当前状态

2026-10-02 已提取公开完整 AAC 音轨，导入现有素材工作区并用固定 Silero VAD v6.0 CPU 分段。用户试听候选后指出“R1排除”；据此排除 R01，保留其余候选开展隔离试验，并改用 R02 作固定主参考。本素材不沿用上一条视频的人工确认，也不与旧素材混合身份向量。此反馈不代表整条视频没有污染，多段方案仍待输出试听评价。

- 视频：[BV1RYEc65EYg](https://www.bilibili.com/video/BV1RYEc65EYg/)，单 P。
- 实际音轨 561.152 秒；公开接口提供的最高码率 AAC 约 110 kbps，原文件 7,714,506 字节。
- VAD 检测语音区间总计 175.168 秒，133 段。区间包含正常短停顿；此数值不等于有效制包素材量。
- 工作区默认选中的全部分段总计约 215.07 秒，包含边界余量。未经人工筛选，不代表全部可用。
- 保留原 AAC；解码为现有工具的 22,050 Hz 单声道浮点音轨及 16,000 Hz VAD 音轨，无降噪、人声分离或自动说话人识别。

## 候选与试听

`candidates.json` 和 `candidate-review.json` 保留首次待确认范围和试听时间轴。除原主参考候选 R01 外，其他训练候选按相邻 VAD 区间组合，保留中间停顿；每个片段不超过 15 秒。`accepted-selection.json` 另记用户筛选反馈及更新后的素材选择，不覆盖初次 VAD 记录。

|编号|源音轨区间（秒）|用途|
|---|---|---|
|R01|66.186–79.606|用户要求排除，不参与编码|
|R02|231.498–242.678|固定主参考|
|R03|251.050–261.238|身份融合候选|
|R04|354.058–366.774|身份融合候选|
|R05|461.322–470.966|身份融合候选|
|H01|15.978–25.174|留出评价，不参与编码|
|H02|541.354–546.230|留出评价，不参与编码|

本机试听产物在忽略 Git 的 `outputs/reference-review-BV1RYEc65EYg-20261002/`：逐段浮点 WAV、`training-review.wav`（R01→R05，61.148 秒）、`held-out-review.wav`（H01→H02，15.072 秒）及 `review.json`。拼接仅在片段之间加入 1 秒零值间隔，不调整片段内部停顿或音量。`candidate-review.json` 记录各段哈希及试听时间轴。

原候选试听保留用于追溯，因此仍含已排除的 R01；实验编码目录不包含 R01。当前素材页面保留七个候选范围，仅 R02～R05 选中，总计 43.728 秒；H01／H02 保留为不选中的留出片段。素材尚未用于正式制包，因此更新此选择不会改变已有音色条目的来源。

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

根据本次筛选固定 R02／R03／R04／R05 为四个训练片段、H01／H02 为留出片段、R02 为主参考。沿用 `tools/reference_fusion_experiment.py` 的 FP32、基础情感、语速 1.0、四段文本和种子 17／29，独立比较单段、原向量等权平均（3／4 段）、归一化后平均再恢复平均范数（3／4 段），不重复片段凑五段。保持主参考的四个非身份张量不变，重新投影 `speaker_latent`；编码片段串行且每段最多 15 秒。

空白对照采用本次保留的 R02 完整区间（231.498～242.678 秒），检查到了内部超过 500ms 的 VAD 间隔，保持对白样本不变，仅将这些间隔缩短至 200ms。比较“保留／缩短”两组，不复用 0–15 秒，不提前判定缩短更好。多段方法继续隔离于正式音色库，是否采用仍由跨文本／种子试听决定。

本机执行：

```powershell
.venv/Scripts/python.exe tools/reference_fusion_experiment.py prepare --material-id dafbe08290bf4a729e46919fafb01d2f --segments r02,r03,r04,r05 --held-out h01,h02 --primary r02 --confirmed-target --confirmed-pause-target --pause-start 231.498 --pause-end 242.678 --source-model-dir voice-producer/models/checkpoints --reader-model-dir outputs/reader-opt/fp32-rebuilt --output outputs/reference-fusion-BV1RYEc65EYg-20261002
.venv/Scripts/python.exe tools/reference_fusion_experiment.py run --output outputs/reference-fusion-BV1RYEc65EYg-20261002
```

隔离实验已按用户“先暂时暂停等我指示”停止，所有实验进程已退出。已保留 25 条有效输出：基准、三段等权、四段等权各 8 条，三段归一化 1 条；剩余组没有完成，不将部分输出写为完整技术验证成功。`pause-state.json` 记录断点，`partial-results.json` 保存现有输出哈希和逐条资源记录。等待用户明确指示才继续，恢复前检查断点并保留已有成果，不直接重跑全部实验。

后续用户明确回复“继续”，已启动断点续跑；原暂停记录保留为历史证据，恢复状态见 `resume-state.json`。为实验脚本增加可选 `--resume`：校验参考、文本、生成参数、编码张量及 reader manifest 指纹，逐条验证已有 WAV 的哈希、采样率和有限值；跳过完成组及完成样本，不重新编码、不覆盖既有音频。旧调用不加此参数时保持原执行方式。恢复时新增模型加载耗时另记于 `resumeModelLoadSeconds`，不覆盖首次加载记录。六项断点保护测试通过，记录见 `resume-tests.txt`。

```powershell
.venv/Scripts/python.exe tools/reference_fusion_experiment.py run --output outputs/reference-fusion-BV1RYEc65EYg-20261002 --resume
```

编码用时 77.29 秒（CPU 串行）；`encoding-audit.json` 验证了七组均符合六个有限 FP32 张量的 ABI、融合组的四个非身份张量逐元素完全不变、身份平均公式及原 checkpoint 的说话人投影一致。R01 没有进入裁剪／编码目录，留出片段与训练片段不相交。

空白对照参考从 11.180 秒缩短为 8.228 秒，保留组与 R02 基准六张量逐元素一致；缩短组的音频仅删除了记录的间隔，保留对白样本逐元素不变。这是输入控制检查，不是输出停顿改善结论。所有实验 safetensors 均被正式 IVP 加载器拒绝，未安装进正式音色库。

原始媒体、解码音轨、参考片段与运行缓存仅在本地。Git 保存脚本、范围、源资产指纹、VAD 记录和准备校验结果；主仓库与交接目录不改。
