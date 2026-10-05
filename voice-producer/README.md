# 本地音色制作与试听

同页流程：**上传素材 → 调整自动选取参数 → 选择、微调参考 → 制作临时音色 → 输入文本试听 → 调整 → 下载 `.ivp`**。

独立入口启用 producer/audition 服务，只显示简化页面。旧验证页和 CLI 保留音色库安装行为。音色信息只有名称、性别；音色包固定包含 FP32、BF16 两套数据。高级设置只保留设备，默认自动；文本试听区选择试听精度，默认 FP32。单段使用单段参考，多段自动等权融合身份，主参考提供声学提示。

## 准备与启动

在 IndexTTS 根目录执行；PowerShell 使用项目 UTF-8 环境预检，Linux 替换 Python 路径。

```powershell
uv sync --locked --extra validation-web
.venv/Scripts/python.exe -m indextts.runtime.assets fetch --root voice-producer/models --lock voice-producer/assets.lock.json
.venv/Scripts/python.exe -m indextts.runtime.assets verify --root voice-producer/models --lock voice-producer/assets.lock.json
.venv/Scripts/python.exe voice-producer/prepare_runtime.py
.venv/Scripts/python.exe voice-producer/start.py
```

地址 `http://127.0.0.1:7862/`，支持 `--port`、`--cpu-threads`。入口从自身位置定位文件，支持其他工作目录启动。打开页面不会下载或导出模型。

在 Readest 开发工作区中，依赖准备完成后可直接双击上层根目录的 `start-voice-producer.bat`；已运行时复用服务。满意的最终 `.ivp` 下载后，由用户手动放入上层的 `index-tts-package/SoundPackage/official/`，作为之后阅读器交接的正式音色来源；制作工具不自动加入该清单。

资产锁包含正式制包权重、Silero VAD、tokenizer 和 BigVGAN 配置。独立准备命令先核验正式资产，再原子导出并校验 FP32 runtime ABI v2 和 BF16 ABI v3。已有运行模型只校验，不覆盖。旧 ABI v1 目录保留，新流程使用独立目录。文本试听需要 CUDA；缺失或不兼容时，参考编辑、恢复和已有包下载仍可进行。

| 内容 | 相对路径 | Git |
| --- | --- | --- |
| 启动与准备入口 | `voice-producer/start.py`、`prepare_runtime.py` | 跟踪 |
| 资产锁 | `voice-producer/assets.lock.json` | 跟踪 |
| 正式权重 | `voice-producer/models/checkpoints/` | 忽略 |
| 两档试听模型 | `voice-producer/models/runtime/<profile>/` | 忽略 |
| VAD | `voice-producer/models/vad/` | 忽略 |
| 制作工作区 | `outputs/voice-producer-drafts/<workspaceId>/` | 忽略 |
| 应用回收区 | `outputs/voice-producer-drafts/.trash/` | 忽略 |
| 旧音色库 | `outputs/voice-workbench/` | 忽略 |

统一入口仍为 `python -m indextts.validation_web`，默认三页。简化入口使用 `--producer-workflow --modules producer,audition`。现有 GPU 调度只协调单个服务进程，同时运行两个服务时避免并行 GPU 计算。

## 工作区与版本规则

- 顶部可新建、显式切换和恢复上次工作区。刷新按浏览器当前工作区恢复；新浏览器默认最近有效工作区。列表显示名称或创建时间。
- 有效编辑防抖 500ms 自动保存，操作前立即提交最新浏览器快照。JSON 原子替换；显示保存失败，保存失败阻止后续操作。
- 每个工作区保留原素材、解码音轨、VAD/波形缓存、片段、主参考、参数、逐片段试听起点、名称/性别/精度、文本、临时包及最近两次成功试听。版本绑定参考选择和裁剪参考数据。
- 旧素材通过“使用已有素材”复制进独立工作区。旧素材与已安装音色不删除；不批量迁移。浏览器片段缓存键包含工作区 ID。
- 清空需页面确认，取消任务并等待释放后，仅把目标工作区移入回收区并新建空工作区。回收入口支持 7 天内恢复，服务启动清理过期内容。模型、其他工作区、旧音色库和已下载文件不受影响。
- 每个工作区音色 ID 稳定。新流程不安装音色库；`WorkbenchService.build(..., temporary=True)` 为临时出口，默认出口维持旧行为。
- 参考、边界、主参考变化只保存草稿，在制作/试听/导出时分别编码 FP32、BF16 并合并；一档失败不提交半成品。切换试听精度和修改文本复用同一包；仅改名称/性别同时更新两档 manifest，张量、许可和来源记录保持不变。
- 任务提交时固定快照，运行期间可以编辑。历史结果不回写新的草稿；修订一致才更新当前包指针。所有提交检查工作区及取消状态，取消不提交结果。制包计算阶段需等待结束后停止。
- 试听默认基础情感、正常语速、种子 17，其余沿用精度默认值。文本下的“情感向量（手动设置）”提供高兴、愤怒、悲伤、恐惧、厌恶、低落、惊讶、平静八个 0～1.2 滑块，可混合；全部归零沿用基础情感。向量随工作区保存，改变向量只影响合成、不重新制包；试听记录绑定提交时的向量。页面只显示本次成功试听，失败保留该结果；内部仍保留最近两次成功文件用于缓存恢复。文本、向量或音色修改后，已有结果标记“修改前的试听”，显示文本、情感与音色版本。
- 下载先准备并验证最新双精度包，再提供 `.ivp` 文件链接。无需先试听；更新失败不提供旧包。不附带原音频、草稿或试听文件。
- “生成音色包”只更新临时包，不合成文本，也不重新加载已有试听播放器；“下载音色包”同样保留播放器。只有点击“生成试听”且合成成功时才替换试听音频，制包后已有试听可标为“修改前的试听”。

实现入口：`indextts/producer_drafts.py`（保存、回收、任务、包和试听），`producer_workflow_web.py`/`producer_workflow.js`（页面与保存），`producer_workflow.css`（简化页面布局），`material_player.py`/`material_waveform.js`（参考编辑）。

宽窗口采用左侧参考编辑、右侧制作／试听／下载的布局，顶部集中工作区操作；1080px 以下按流程上下排列，640px 以下参考列表也改为单列。布局样式只作用于简化制作页，片段选择、微调、自动保存和任务规则保持一致。

## 双精度音色包

新制作流程生成 schema v4 的单个 `.ivp`：根目录 `conditioning.safetensors` 是 FP32，`bf16/conditioning.safetensors` 是 BF16；两档分别保留真实参考预计算结果和来源记录，BF16 不由 FP32 强制转换伪造。许可文件共享；根 manifest 和 `bf16/manifest.json` 校验两档的哈希、张量、身份及参考一致性。

`load_voicepack(path, profile=...)` 按运行精度选取数据，并同时验证两档完整性；未指定时默认 FP32。运行服务自动传入自己的模型精度。单精度 schema v1～v3 仍按原规则读取；旧客户端需要更新读包支持才能直接使用 v4。`extract_voicepack(source, destination, profile)` 可无损拆出旧格式单精度包，导入音色库时自动拆为两档，后续保持原有库行为。CLI inspect 展示两档清单。

新流程首次使用旧工作区包时，复用参考一致且校验通过的已有精度缓存，只计算缺失的版本。两档都齐全后切换试听精度不会重新制包，播放器标出本次结果实际使用的精度；切换后旧音频仍保留并标为修改前的试听。

## 参考定位

停顿 0.1～3 秒、最低音量 -80～0 dBFS（默认关闭）、最短片段 0.1～15 秒实时更新候选，不删除已选片段。浏览器复用 VAD 概率和原始 RMS 缓存。VAD 不识别说话人或移除背景音乐。

点候选“加入”，整张选中卡片点击进入微调；试听、星标、移除各自独立。左右把手调边界，中间拖动整段，每段最多 15 秒且不重叠。精确秒数及键盘方向键可用，Shift 精调 0.01 秒；有效修改可撤销。

选中卡片的“启用”勾选框控制是否参与音色制作。取消勾选后片段变灰，仍保留边界、试听起点并可继续试听和微调；再次勾选即可恢复参与。列表显示启用数量与时长。主参考停用时自动使用最长的启用片段，重新启用后恢复原主参考；全部停用仍可保存，但制作、试听和下载要求至少启用一段。启用状态随工作区保存，可撤销；停用片段的微调不触发重新编码。

点击/拖动蓝色标记指定试听起点；普通试听每次从该点开始，“从头播放”单独从片段起点开始。局部导航拖动紫色框查看附近音轨，按钮平移 0.5 秒，“回到片段”恢复合适视野。位置随工作区保存。

## 辅助模块以后如何恢复

简化页面不构建“诊断与模型”“音色库与下载”、旧试听和情感页。以后可在 `producer_workflow_web.build_page()` 中复用 `library_web.library_controls` 或 `producer_web` 的诊断回调，按需要提供局部入口。

旧制包页保留 `indextts/producer_web.py` 的 `SHOW_PRODUCER_TOOLS = False`。改为 `True` 并重启，仅恢复旧制包页两个模块，不会自动暴露到新流程。

## 验证

```powershell
.venv/Scripts/python.exe -m pytest tests/test_producer_drafts.py tests/test_materials.py tests/test_material_segmentation_settings.py tests/test_validation_decoupling.py -q
.venv/Scripts/python.exe tools/verify_producer_workflow.py --source-id <已有素材ID>
```

真实 GPU 验收使用隔离目录 `outputs/producer-dual-acceptance/`，一次生成两档音色，在同一包上分别做 FP32/BF16 文本合成，再验证元数据更新、两档张量保持、下载读包和重启恢复，不写旧音色库。素材需有至少两段 4 秒以上的选中片段。结果与日志留在隔离目录。

本机与浏览器验收见 [2026-10-05 验证记录](../docs/validation/producer-workflow-2026-10-05/README.md)。500ms 保存窗口内的快速刷新由按工作区保存的浏览器恢复记录补齐；服务启动和工作区恢复使用原子保存的磁盘记录。
