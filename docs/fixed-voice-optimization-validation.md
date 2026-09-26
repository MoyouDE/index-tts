# 固定音色优化：本机验证与交接

日期：2026-09-26。子模块分支：`codex/indextts-reader-inference-opt`。主仓库保持 `master`，未修改主仓库文件，也未提交其子模块指针。所有实现、复现入口、音色测试包和下述报告均由子模块 Git 保存；大型权重通过固定 revision 与 SHA-256 的资产锁重建。

## 实现范围

按[研究文档](fixed-voice-optimization-research.md)复刻固定音色、外部八维情感输入的专用推理路径。运行端不依赖 Wav2Vec2-BERT、CAMPPlus、参考音频或情感识别模型。保留原 FP32 兼容档，新增 GPT BF16、其余生成网络 FP32 的生产档；显式精度档不同的音色包拒绝混用。音色包必须离线实际预计算，不以 dtype 转换冒充 BF16 生产。

掩码共享/广播、非因果掩码分配缩小、DynamicCache、CPU beam scorer、增量位置、Euler 循环不变量、最终状态保留、inference_mode、单音色 GPU 条件缓存和临时张量释放均可按实例开关。未降低 beam、声学步数或实际位置表长度。可选线程 CPU set 与 cudaMallocAsync 不默认开启。使用及回退命令见[实施说明](fixed-voice-optimization.md)。

## 环境与测试

Windows 11 build 26200，RTX 2060 6GB，i7-9750H；Python 3.11.13、PyTorch 2.8.0+cu128、Transformers 4.52.1、pytest 9.0.3，CPU 线程数 4。本卡没有原生 BF16，PyTorch 兼容 BF16 可以执行，但速度不能外推到研究报告的 RTX 5060。

| 验证 | 本次结果 |
|---|---|
| reader、voicepack、优化单元/小模型测试 | 62 passed；包含 CPU/CUDA、FP32/BF16、采样/确定性 beam、EOS、缓存/位置、Euler、队列快照、种子恢复和精度隔离 |
| 真实 BF16 模型离线隔离测试 | 1 passed；禁止参考文件/编码器与网络访问，检查音色切换、零向量与 base 一致、取消/失败后恢复 |
| FP32 原路径与全部优化配对 | 3 条：conditioning、语义 token、PCM SHA-256 全部一致 |
| BF16 原路径与全部优化配对 | 两音色 × 三长度共 6 条：上述三类结果全部一致 |
| BF16 情感配对 | base、高兴、悲伤、混合、全零共 5 条：上述三类结果全部一致 |
| 两音色连续五轮 | 10 条，与对应首轮三类结果完全一致 |
| cudaMallocAsync + performance-cores | 真实模型完成合成，与 native 的同输入/种子结果完全一致；本机同构 CPU，未覆盖异构核心约束/恢复 |
| FP32 权重重建 | 从锁定源资产重新导出，manifest 与原有模型包完全相等，包含全部文件哈希；核心权重 3,258,878,844 bytes |
| 源资产锁校验 | 本地文件按锁校验；10 个小文件与固定远端下载字节相同，6 个大型权重与远端 LFS SHA-256 相同 |

测试原始结果及逐请求数据保存在 [validation/2026-09-26](validation/2026-09-26)。完整 480 条矩阵与 100 次交替切换入口已提供，本轮没有执行该完整矩阵。

## 显存与计时的实际结果

以下取对应 smoke 配对中请求峰值的最大值；GiB = 2^30 bytes。进程显存是 WDDM PDH 每 100ms 采样，不能代替严格峰值测量。

| 配置 | PyTorch allocated 峰值：关闭 → 开启优化 | WDDM dedicated 采样峰值：关闭 → 开启优化 |
|---|---|---|
| FP32，3 条 | 4.023 → 3.822 GiB | 4.421 → 4.638 GiB |
| BF16，6 条 | 2.910 → 2.712 GiB | 3.858 → 3.581 GiB |

因此，本机已验证降低张量占用，但**没有证明进程峰值 ≤3GiB**。FP32 的进程采样峰值甚至更高，不能只报告 allocated 的改善。BF16 情感小集的 optimized dedicated 采样峰值约 3.009GiB，也不构成达标。

五轮切换中，请求结束后的 allocated 在 2,258,158,080～2,258,470,400 bytes 范围内变化；reserved 前九条为 3,116,367,872 bytes，第十条增至 3,728,736,256 bytes。因此没有发现持续增长的活跃张量引用，但不能宣称分配器预留池已达到稳定平台。

这些配对开启 trace，会引入哈希计算、CPU 回传和同步；部分运行与环境安装、测试或下载重叠。RTF 保留用于检查与诊断，不作为正式提速比例或 P95 达标证明。RTX 2060 的 BF16 兼容路径本轮未达到 RTF P95<1。研究报告的性能结论没有冒充本机成绩。

正式性能验收应在目标设备上关闭 trace、隔离并发负载，分别运行 baseline/optimized，使用同精度、同资产、同种子与同 CPU 策略；按五个文本长度组分别报告 P50/P95，并同时观察 WDDM dedicated/shared 与 PyTorch allocated/reserved。RTF 分母排除人为插入的段间静音，队列等待及文件写入独立记录。

## 可迁移成果与记录边界

- `tests/fixtures/reader-assets.lock.json` 固定全部参考权重/配置和两个演示参考 WAV 的来源及哈希。
- `tests/fixtures/voices-bf16` 是实际 BF16 预计算的两个 schema-v2 音色包；`voices-fp32` 保存两个历史 schema-v1 兼容包。许可证和来源说明见 fixtures README 与包内文件。
- 两种导出模型的 manifest 已保存到验证目录，生成权重本身不入 Git。
- 最早配对报告的 `revision` 为 `c7e670e…`，因为运行时实现尚未提交；对应实现随后保存为 `8757c36`。后续报告记录 `8757c36`，本次交接提交另含零向量精度修正、精度档混用拒绝、队列计时及隔离测试。原始 revision 不倒填。
- BF16 与 FP32 档的采样规则不同，不把跨档音频差异当作优化回归；正确性比较始终在同一精度档内进行。
- 本轮没有主观听感盲评，没有重训音色或情感模型，也没有接入/改动主仓库阅读器。

全新环境复现未完成：`uv sync --locked --extra test` 的 PyTorch wheel 下载先后遇到 TLS EOF 和下载超时。其余依赖已从锁文件装入隔离环境；大型 wheel 的分片续传尚未完成，已停止该额外下载，不把它记作全新环境测试通过。上述 62+1 项测试均运行在已有环境，测试前 pytest 已更新至锁定的 9.0.3。错误日志随报告保存；在网络可用的设备继续执行实施说明中的 `uv sync --locked --extra test` 即可重新验证。
