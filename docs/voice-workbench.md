# 本地音色库与合成试听

`python -m indextts.validation_web` 现在包含“音色包生成”“情感推理”“合成试听”三个页签。此工具不会扫描、删除或写入 `index-tts-package` 交接目录；定稿交接仍由使用者手动处理。

## 启动与目录

在 IndexTTS 子模块中执行：

```powershell
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new()
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [Console]::OutputEncoding
$env:PYTHONUTF8 = '1'
uv sync --locked --extra validation-web --extra test
.venv/Scripts/python.exe -m indextts.validation_web --workspace-dir outputs/voice-workbench --source-model-dir checkpoints --port 7861
```

默认仅监听 `127.0.0.1`，不开启分享。情感模型仍通过 `--emotion-model-dir` 或情感页指定，缺少情感模型不影响基础情感试听。Linux 将解释器路径改为 `.venv/bin/python`。

### 按功能启动

不传 `--modules` 时保留三个页签。参数接受逗号分隔的 `producer`（制包及音色管理）、`emotion`（情感验证）、`audition`（合成试听），页签始终按此顺序显示；空列表、未知名称和重复名称会报错。

```powershell
.venv/Scripts/python.exe -m indextts.validation_web --modules producer
.venv/Scripts/python.exe -m indextts.validation_web --modules emotion --emotion-model-dir D:/Models/emotion-onnx
.venv/Scripts/python.exe -m indextts.validation_web --modules audition
.venv/Scripts/python.exe -m indextts.validation_web --modules producer,audition
.venv/Scripts/python.exe -m indextts.validation_web --modules emotion,audition --emotion-model-dir D:/Models/emotion-onnx
```

每次选择一种模式启动服务；切换模式需重启。纯情感模式不创建或读取音色工作区，不导入制包、音色包校验和试听服务。首次分析仍使用原 ONNX provider 和 Transformers tokenizer，tokenizer 会引入 PyTorch；本轮保持原安装环境和 tokenizer 行为。

只启用试听时，缺失精度会给出制包启动指引；未启用情感时不显示“采用情感页结果”。同时启用对应功能时保留跨页导航和会话向量传递。关闭其他模块不要求其模型目录有效，模型在执行操作时才加载。

### 模块边界

制包、情感、试听服务分别管理自己的模型和卸载；制包只产出包，试听只接收包快照并产出 WAV/报告。音色库保存成果及本机设置；应用层负责快照、安装和最近成果替换。GPU 协调器通过公开卸载方法串行切换制包与试听，情感分析独立运行。

`ValidationService`、`WorkbenchService` 的原公开调用保持兼容，改为组合委托；`create_app` 新增可选 `modules`、`cpu_threads` 参数。嵌入调用者可使用返回应用的 `app.workbench.close()` 释放模型。页面间连接由入口组装，情感桥接只保存本会话向量副本，不从试听页调用情感服务。模型及 GPU 切换日志写入服务日志，JSONL 协议保持原样。

启动测量与本轮验证见 [第一轮解耦记录](validation/runtime-decoupling-2026-10-01/README.md)。

本地音色库默认为 `outputs/voice-workbench/`，已由子模块的 `/outputs/` 规则忽略。自定义工作区应选择本机数据目录，不放在需要提交的源码目录中。`--output-dir` 继续控制临时会话及情感 JSON 输出，与持久音色库不同。

工作区中 `voices/<voiceId>/voice.json` 保存音色身份、备注和参考音频信息；该目录保留原始参考音频，以及 FP32/BF16 各一份 `.ivp`。每个精度的 preview 目录只保留最近成功的 WAV、报告和 `latest.json` 指针。`settings.json` 保存本机模型目录及设备偏好，`runtime/` 保存试听进程日志，`.jobs/` 保存请求期间的快照。模型、原始音频、合成音频、上传缓存和本机设置均不提交 Git。

音色库跨页面刷新和服务重启保留，同一服务的本机会话共享音色库。进行中的请求、取消操作、输入文本与情感页结果按会话区分；本工具没有公网多用户身份系统。

## 制包与管理

首次上传参考音频，填写新的稳定音色 ID 后生成。音频仍沿用原预处理规则：单声道加载并只使用前 15 秒，完整参考文件在本地保留。更换参考音频需要新音色 ID。

在“本地音色库”选择已有音色后，可播放参考、保存备注、下载包、补生成另一精度或重新生成所选精度。缺少的版本显示“尚未生成”。新包通过校验后才替换旧包，并清除该精度旧试听，避免将旧声音误认为新结果；生成失败保留旧成果。

导入只接受通过结构、哈希及张量校验的 `.ivp`，不会覆盖同 ID 条目。导入条目没有参考文件，只能试听，不能补生成另一精度。旧 schema v1 包仅在全部张量都是 FP32 时按 FP32 导入；缺少 provenance 的低精度旧包须从参考重新制包。

删除入口显示具体音色和删除范围，并要求勾选确认。只删除所选音色在工作区中的参考、两档包、试听及备注。已有会话输出和工程示例包不会自动导入本地库。

## 试听流程

1. 选择库中音色及 FP32/BF16 版本。
2. 填写已经导出的同精度裁剪模型目录，必须包含 `runtime_model.json`。两档模型目录分别记忆。页面不自动导出模型，也不会从交接目录自动寻找模型。
3. 输入文本，选择基础情感、手动八维向量或本会话情感页最近成功结果，生成后播放或下载。默认基础情感；情感页向量作为显式向量进入 reader，报告保留原始与有效值。

FP32/BF16 是模型计算精度，不是音色身份。当前包中的说话人、基础情感投影与计算精度相关；两档必须分别从相同参考音频产生，不能把一个包强转 dtype 作为另一档。包、模型指纹及 BF16 provenance 不匹配会明确报错。

合成使用现有 `ReaderRuntime`，仅支持 NVIDIA CUDA。语速 0.5～2.0 在接口层换算成 `durationFactor=1/语速`。默认种子 17，也可选择每次随机，实际种子始终记录。

高级设置开放原有 GPT 采样参数、beam、惩罚、最大生成 token 数，以及声学步数和 CFG。最大 token 数同时受当前模型容量约束；默认声学 25 步、CFG 0.7。FP32 默认关闭随机采样，BF16 默认使用原有采样配置，可恢复当前精度默认值。比较精度效果时还需注意两档默认采样策略不同。

优化项支持全部启用、全部关闭或逐项选择；CPU 线程默认 4，分配器默认 native，可选 cudaMallocAsync。这些选择不承诺在每张显卡上提速。

试听在独立 JSONL 子进程运行。设备、模型、精度、实例级优化、线程或分配器改变时重建进程；普通文本和生成参数变化可以复用模型。制包与试听共用 GPU 锁，切换用途时卸载另一侧模型；CPU 情感分析独立运行。“取消”作用于当前会话，按生成阶段边界停止，超过取消等待时间则回收进程。“卸载”等待当前 GPU 操作结束后释放模型。

每次请求先保存包和参数快照。重新制包或删除不会改变正在运行的模型输入；如果完成时原音色已经删除或版本已变更，不将旧任务结果写回新条目，并显示错误。失败及取消不替换上一次成功试听。

报告分别记录模型加载和合成时间、实际音频时长、RTF、完整生成参数、包哈希、模型 manifest 哈希、情感、种子和设备。显存字段是 PyTorch 当前 allocated/reserved 及该进程生命周期峰值，包含模型加载，不是严格单次峰值、WDDM dedicated 或系统总内存。

## 接口与验证

Python `ReaderRuntime.synthesize` 新增可选 `generation_settings`；JSONL `synthesize` 新增 `generationSettings`。省略时保留原默认参数。允许字段为 `do_sample`、`temperature`、`top_p`、`top_k`、`num_beams`、`repetition_penalty`、`length_penalty`、`max_generate_length`、`acoustic_steps`、`cfg`，未知字段和非有限数值被拒绝。

JSONL 新增 `voices.load`，参数为 `{"path":"包的绝对路径"}`，执行现有源模型、精度和 provenance 校验，再替换当前装载音色；已入队请求继续持有原包快照。阅读器应用无需修改。

```powershell
.venv/Scripts/python.exe -m pytest tests/test_audition_worker.py tests/test_voice_workbench.py tests/test_reader_runtime.py tests/test_reader_optimizations.py tests/test_validation_modules.py tests/test_voicepack.py tests/test_emotion_context_policy.py tests/test_webui_syntax.py -q
.venv/Scripts/python.exe tools/validate_workbench.py --fp32-model D:/Models/reader-fp32 --bf16-model D:/Models/reader-bf16 --workspace outputs/voice-workbench --voice-id your-voice --output outputs/workbench-regression.json
```

真实对比工具要求两档音色和两档模型都已存在，输出文件不得存在。它比较同精度、相同输入与种子下旧请求和显式默认参数的 WAV 哈希，再验证调整声学参数确实改变输出。功能验证不等于主观音色质量验收，试听品质由使用者判断。
