# 本地音色生成工具

参考音频 → `.ivp`，提供制包、音色库管理、两档精度补生成、包检查及下载。独立入口只启用 producer，不构建情感或试听页，不加载它们的服务与模型。实现继续复用 `indextts` 的制包核心、音色库和页面；本轮先独立目录及入口，不拆 Python 环境或新增 Git 子模块。

## 固定目录

| 内容 | 路径（相对 IndexTTS 根目录） | Git |
| --- | --- | --- |
| 启动入口 | `voice-producer/start.py` | 跟踪 |
| 下载地址及资产校验 | `voice-producer/assets.lock.json` | 跟踪 |
| 正式制包权重 | `voice-producer/models/checkpoints/` | 忽略 |
| Silero VAD v6.0 ONNX | `voice-producer/models/vad/silero-v6.0.onnx` | 忽略；哈希进入资产锁 |
| 本机音色库 | `outputs/voice-workbench/` | 忽略 |
| 会话输出 | `outputs/validation-web/` | 忽略 |

权重默认路径固定在工具目录内，页面只读显示绝对路径。音色库与统一页面共用，现有资产不需要迁移。旧的工程根 `checkpoints/` 仍可供历史 CLI 和完整模型使用；新工具不依赖那个目录。目录尚未准备好时，制包及兼容检查明确报缺少文件，不回退到旧位置。

## 跨设备准备与启动

在 IndexTTS 根目录安装现有环境后执行（Windows Python 3.11；Linux 换用 `.venv/bin/python`）：

```powershell
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new()
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [Console]::OutputEncoding
$env:PYTHONUTF8 = '1'
uv sync --locked --extra validation-web
.venv/Scripts/python.exe -m indextts.runtime.assets fetch --root voice-producer/models --lock voice-producer/assets.lock.json
.venv/Scripts/python.exe -m indextts.runtime.assets verify --root voice-producer/models --lock voice-producer/assets.lock.json
.venv/Scripts/python.exe voice-producer/start.py
```

默认本地地址 `http://127.0.0.1:7862/`，不开公开分享。可用 `--port 7863 --cpu-threads 4` 修改端口与线程数。入口从自身位置定位模型和成果目录，可从其他工作目录使用绝对脚本路径启动。

统一页面和独立入口共用音色库及 GPU，实际操作时选择一个服务运行；本轮没有增加跨服务的 GPU 调度或音色库并发写入协调。

统一页面仍使用 `python -m indextts.validation_web`（默认端口 7861），同样默认使用这个固定模型目录。部署统一页面时可用 `--source-model-dir` 配置其他目录，页面仍不可编辑。

## 当前独立边界

素材整理复用同一个入口的“长视频／录音素材整理”区域。上传音频或视频后，FFmpeg 提取音轨，Silero VAD 在 CPU 按需加载；它们不加载制包模型。安装 `validation-web` extra 会带入固定版本的 `imageio-ffmpeg`，没有系统 FFmpeg 时使用其随包二进制。现有 PATH 中的 FFmpeg 优先；缺失或解码失败明确报错。

页面提供完整音轨、分段表、片段回放、边界修改、拆分、合并、选择／排除和主参考指定。先保存编辑，再确认选中片段属于同一人物，填写新的音色 ID 并“使用主参考生成音色包”。正式制包只用主参考，其他片段不参与融合；选中片段超过 15 秒必须拆分，不默默截取。VAD 不识别说话人，不能替代人工确认。

素材保存在工作区 `materials/<sourceId>/`，含原文件、22.05kHz 回放音轨、16kHz VAD 输入和分段记录。刷新及重启后从“已保存素材”重新选择即可恢复。制包保存固定主参考及选择快照；补生成另一精度沿用该快照。修改选择后需要新的音色条目。详情和隔离融合实验说明见 [长素材验证记录](../docs/validation/long-material-2026-10-01/README.md)。

锁文件只列制包计算及兼容校验需要的资产，不包含参考示例音频、文本 tokenizer 或 Qwen。保留源 GPT、s2mel checkpoint，运行时严格抽取所需权重；codec/BigVGAN 文件仅用于保持已有包的兼容指纹。当前是资产与入口独立，尚未成为可单独安装的 Python 项目，也没有裁剪专用权重格式。后续若进一步裁剪，应单独验证数值及指纹兼容，不能仅删除这些源文件。

## 本机验证（2026-10-01）

- 12 个资产共 7,081,573,999 字节，从原源目录复制到正式目录，逐个校验字节数及 SHA-256；旧目录保留。模型与音色库确认被 Git 忽略。
- 调用原有指纹函数，对两个目录分别计算，结果一致：源模型 `30fbd7beb7da3dc24f62b0ddcca820992ca5ac6dc487fa9b1e65c8f9af9d3253`；参考编码器 `169f78d8c812370ccccd4aec0047081d8c4f07f8f3c273d2f0e7548f29b4cd4c`。
- 只读目录修改后，制包、模块隔离和语法回归 45 项通过；加入独立目录和入口后，相关专项 11 项通过（包含六种模块模式）。这些测试集合有重叠，不相加为独立测试数。
- 独立入口从 IndexTTS 以外的工作目录真实启动；浏览器确认只有制包页、源目录不可编辑、沿用原音色库。统一入口仍保留三个页签，固定目录一致。
- 本轮没有改编码计算、权重内容或包格式，没有重新运行 GPU 制包或主观试听，也没有重新创建此前清理的测试音色。


## 可选多段身份融合

素材整理区域可选择 `speaker-mean-v1` 等权融合；默认单段主参考不变。所有选中片段最多 15 秒且属于同一目标人物，声学提示和基础情感固定为主参考。音色库保存裁剪快照，两档精度分别制包；修改素材选择后需新音色 ID。融合包使用 schema v3，新加载器支持，旧加载器拒绝；无需重新导出生成权重。[技术与全流程记录](../docs/validation/formal-fusion-2026-10-02/README.md)。该可选方法已进行技术检查，成品听感由使用者判断能否接受。
