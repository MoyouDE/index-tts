# 固定音色推理优化复刻

本实现以 [外部设备研究报告](fixed-voice-optimization-research.md) 为参考，基于子模块基线 `c7e670e7000ffd438a07cf92367bc90650cb836b` 复刻。研究中的 480 条成绩不是本机成绩。实现与测试全部位于 IndexTTS 子模块，主仓库无需迁移。

## 配置与资产

| 配置 | GPT / 静态投影 | 其他生成网络 | 默认生成 |
|---|---|---|---|
| `compatible-fp32` | FP32 | FP32 | 原有确定性 beam=3 |
| `fixed-voice-bf16` | BF16 | FP32 | 文档采样参数、beam=3 |

两档均保持声学 25 步、CFG=0.7、40 字符附近分段、IEEE FP32 运算以及 BigVGAN 原生 PyTorch 路径。BF16 包必须从参考音频重新预计算；禁止把旧 FP32 包转换 dtype 后当作新包。PyTorch 支持 BF16 但 GPU 没有原生 BF16 时保持真实 BF16 运算，`health.nativeBf16=false`，性能不能外推；不支持 BF16 时明确报错。

旧模型 ABI v2、音色包 schema v1 继续可用。BF16 模型使用 runtime ABI v3，音色包显式 `--profile` 使用 schema v2；两种音色包的 conditioning ABI 不变。新包保存参考音频 SHA-256、参考编码器/统计量/情感矩阵指纹、预处理指纹和生产库版本。兼容性检查不需要运行端访问参考 WAV 或编码器。文件哈希用于完整性和兼容校验，不是数字签名。

在仓库根目录执行（PowerShell 先设置 UTF-8，按项目 AGENTS.md）：

```powershell
# 使用项目锁文件创建环境；pytest 版本以 uv.lock 为准。
uv sync --locked --extra test

# 两套资产保存在不同目录；导出器拒绝覆盖非空目录。
.venv/Scripts/python.exe -m indextts.runtime.cli export-model --source-model-dir checkpoints --output-dir outputs/reader-fp32 --profile compatible-fp32
.venv/Scripts/python.exe -m indextts.runtime.cli export-model --source-model-dir checkpoints --output-dir outputs/reader-bf16 --profile fixed-voice-bf16

.venv/Scripts/python.exe -m indextts.voicepack.cli build --reference examples/voice_01.wav --voice-id reader-one --name reader-one --gender unknown --model-dir checkpoints --profile fixed-voice-bf16 --output outputs/voices-bf16/reader-one.ivp --device cuda:0

.venv/Scripts/python.exe -m indextts.runtime.cli serve --model-dir outputs/reader-bf16 --voice-dir outputs/voices-bf16 --cache-dir outputs/reader-cache --emotion-backend none
```

制包使用原有完整开发环境；运行端仅加载裁剪模型和音色包。通用 CLI 的历史 Qwen 可选后端保留，阅读器启动使用 `none`，不导入/实例化任何情感模型。生产音色的原 WAV 不进入运行包。

## 优化映射与回退

`serve --optimizations all` 默认启用全部复刻优化；`none` 关闭；也可传逗号分隔的名字，仅开启指定项。设置按运行实例生效。

| 开关 | 实现与边界 |
|---|---|
| `compact_mask` | 非因果 DiT、无 KV cache 时预分配长度 8；实际 forward 提供广播掩码，位置表及实际序列不缩短 |
| `shared_gpt_mask` | 检查各层一致后仅传输一份 GPT bias，GPU 上共享存储 |
| `broadcast` | style 及非因果 mask 的 repeat 改只读 expand |
| `dynamic_cache` | 原生 DynamicCache 贯通 Transformers 4.52.1 与 vendored generation wrapper |
| `cpu_beam` | 实例工厂选择 CPU scorer，成组传输候选数组，process/finalize 沿用原实现 |
| `incremental_position` | 阅读器单文本及同长度 beam 副本的位置增量，不用于异长混合 batch |
| `euler_invariants` | 单次声学生成的 CFG prompt/style/mu 拼接移出循环 |
| `discard_euler_history` | 只保留最后状态，保持求解与提示区归零顺序 |
| `inference_mode` | 整次独立生成使用 inference_mode，输出不承诺用于训练 |
| `voice_cache` | GPU 仅保留当前音色静态条件，不缓存跨文本 KV |
| `release_intermediates` | 声学/声码器及下一分段之间释放无用张量引用 |

旧通用模型默认不启用这些 reader 实例开关。模型裁剪已有 speaker/emotion conditioner、text_head 和 codec encoder 分离；本轮继续保留完整 codec 解码器，加载使用严格键校验。每次请求清理 GPT 前缀引用，无需逐请求 empty_cache。

`--cpu-threads 4` 为独立 sidecar 默认。`--performance-cores` 只在 Windows 存在不同效率等级时约束当前工作线程，遵守已有 CPU set 限制并在正常结束或异常时恢复；同构 CPU 无操作。`--allocator cudaMallocAsync` 必须在新进程、导入 torch 前选择；不默认启用，回退需重启。

## 请求与隔离

JSONL 方法保持不变，合成参数新增可选 `seed`（整数 0 到 2^32−1）：

```json
{"id":"example","method":"synthesize","params":{"voiceId":"reader-one","text":"清晨的阳光照进了安静的书房。","emotion":[0.8,0,0,0,0,0,0,0],"seed":17}}
```

入队复制请求参数并绑定当时的音色包对象，重载音色不会更改排队请求。GPU 单工作线程串行执行。显式种子作用于整次生成，并在结束/异常后恢复 RNG；调用者不应在同一进程其他线程同时改变全局 RNG。未提供种子时保持原行为。

八维顺序为 happy/angry/sad/afraid/disgusted/melancholic/surprised/calm。沿用现有接口允许的 [0,1.2] 范围（文档测试输入为 [0,1]），偏置后总和封顶 0.8，只处理一次。零向量与 base 等价，calm 不等于零向量。响应增加原始情感、有效向量、seed、profile、有效语音时长及 RTF。

## 验证与测量

```powershell
.venv/Scripts/python.exe -m pytest tests/test_reader_runtime.py tests/test_voicepack.py tests/test_reader_optimizations.py -q

# 正确性配对：两次独立进程，配置、音色、输入和种子必须相同。
.venv/Scripts/python.exe -m indextts.runtime.benchmark --model-dir outputs/reader-bf16 --voice-dir outputs/voices-bf16 --voice-id reader-one --optimizations none --trace --output outputs/bench/baseline.jsonl
.venv/Scripts/python.exe -m indextts.runtime.benchmark --model-dir outputs/reader-bf16 --voice-dir outputs/voices-bf16 --voice-id reader-one --optimizations all --trace --compare outputs/bench/baseline.jsonl --output outputs/bench/optimized.jsonl
```

不带 `--trace` 单独测 RTF；trace 会回传张量计算哈希，有额外同步开销。输出拒绝覆盖。`--suite full` 使用已跟踪的 20 文本×指定音色×4 情感×3 种子；传两个 `--voice-id` 即 480 条。`--suite voices` 检查指定音色，`--suite alternating` 做 100 次切换；`--rounds 5` 检查持续运行。输入、配置、资产哈希、设备、库版本和逐条结果记录在 JSONL，旁边生成 summary。

RTF 从音色静态条件加载后开始计时，至 CPU 波形可用，不计文件写入；分母排除段间附加静音。切换音色耗时、首次请求和加载单独记录。阶段 timings 保留原有计时方式，含 GPU 异步提交的影响，不能据此声称精确阶段占比。

显存同时记录 PyTorch allocated/reserved 与 Windows WDDM 进程 dedicated/shared 100ms 采样。采样不可用时字段为 null 并保存原因；采样峰值永远不宣称严格 ≤3GiB 认证。cudaMallocAsync 的 PyTorch 部分统计不能与 native 等同解释。

本机结果与未验证边界见同目录交接报告。测试通过证明已测优化没有改变对应精度档输出，不证明原模型发音与情感质量问题已经解决。
