# 6000D 部署实测报告的借鉴分析

本文对照两份 RTX PRO 6000D（SM120，无 NVLink，PCIe-only）部署实测报告，说明其中哪些优化值得
ForgeKernel 的 serve 子系统与 `hw_kb/sm120` 吸收、哪些已覆盖、哪些方向不符不必做。

- 来源一：《AllReduce 后端选型与 GLM 5.2 P/D 分离优化总结》
  （8×RTX 6000D，vLLM 0.25.1→0.28，消息大小决定 AllReduce 选型；P/D 配比 1P:2D→1P1D）
- 来源二：《8×RTX 6000D 部署 DeepSeek-V4.1-Flash 实测总结》
  （vLLM 0.34.0，TP8+EP8+DSpark，单流 296.5 tok/s；DeepInfer Cake→NCCL 快 6.6–8×）

> 范围：报告是**整机部署**视角，ForgeKernel serve 是**单 kernel 换入 + 门禁**视角。因此报告里
> 大量「整机调度/配比/硬件选型」不吸收；只吸收能防 6000D 评测重复踩坑的**硬件事实**与
> **评测口径正确性**。

---

## 一、已覆盖（仓库已有，无需重复造，但可交叉引用）

| 报告事实 | 仓库已有条目 | 说明 |
|---|---|---|
| 6000D 无 multicast → FlashInfer PCIe IPC all-reduce 不可用 | `hw_kb/tilelang/sm120/clusters_free_multicast_broken.md` | 该文中 TMA multicast 在 sm120 上 ~60,000× 慢；报告二 §1.5 的「无 multicast，不可用」与此一致 |
| Decode 是唯一**带宽受限** case，单流慢是纯硬件 | `hw_kb/tilelang/sm120/decode_is_the_one_bandwidth_bound_case.md` | `dram_pct≈90%`，实测读 BW 880 / 峰值 1145 GB/s；报告二 §5.4「唯一硬短板是单流解码」同源 |
| SM120 ≠ SM100（无 tcgen05、无 NVLink） | `hw_kb` 多处 + arch gating | `arch.feature_available` 按 arch 卡住 tcgen05/blockscaled |
| sglang 配置 flag 是真的、不是装饰 | `hw_kb/tilelang/sm120/sglang_config_flags_are_real.md` | 569 个 argparse 选项已核，flag 选择真实内核对 |

**结论：报告的硬件事实层仓库已基本对齐，不需要再写重复条目。**

---

## 二、值得吸收（当前仓库空白，6 项）

### 借 1. `NCCL_P2P_LEVEL=SYS` 是 6000D 小消息单流的关键杠杆（+18%）

报告二 §1.4：单流 158→187 tok/s 来自这一条环境变量（降 PCIe 小消息延迟）。当前 serve 的
`harness.serve_command` **不设**该 env。

- **做法**：作为 sm120/PCIe 部署的**默认 env** 写进 hw_kb，并在 `ServeConfig` 文档里点名；
  operator 起服务时应能一键带上。
- **落地点**：`hw_kb/*/sm120/`（一条「PCIe 小消息延迟杠杆」配方）+ `serve/harness.py` 文档注释。
- **可验证**：纯 env，无需 GPU 即可生成命令；实测需 6000D。

### 借 2. 「默认后端 ≠ 最优后端」：Cake→NCCL 在纯 PCIe 快 6.6–8×

报告二 §3.3：DeepInfer 默认 Cake 后端在纯 PCIe 上 prefill 265.6 → 换 NCCL 2139 tok/s/GPU
（8.1×），decode 6.6×；根因是 per-transfer 往返延迟（m64 分块 128 块 × 7 对端 ≈ 900 次往返
≈ 34 ms/层，4KB 往返 9.5 µs），**不是带宽、不是拓扑**。

- **价值**：这是**评测方法论**级教训——「先确认后端是不是真的在用最优路径，再谈调参」。
- **落地点**：一条 KB「PCIe-only 机器的 MoE/通信默认后端可能被往返延迟拖垮；朴素 NCCL 反更快」。
- **可验证**：结论可写成判据；实测需 6000D。

### 借 3. P2P 微基准：区分「延迟受限 vs 带宽受限」的判据

报告二 §3.2：4 KB 往返 9.5 µs（延迟）× 16 MB 52 GB/s（带宽）两档即可判定瓶颈归属。这应作为
serve 的 parity/decode gate **前置诊断探针**。

- **做法**：`hw_kb/sm120` 加一条 P2P probe 配方（4 KB / 16 MB 两档），judge 前先跑，用结果决定
  「加并发 vs 降报文」。
- **落地点**：`hw_kb/*/sm120/`（PROBE 配方，标 PARTIAL 待 6000D 实测）。

### 借 4. AllReduce 三段式选路（按报文大小切后端）

报告一核心：4–16 MiB（decode）→ FlashInfer PCIe IPC 1.59×；64–256 MiB（prefill）→ B12X DMA
FP8 ring 1.76×（FP8 线字节 −48.4%）；32 / 512 MiB → NCCL。

- **6000D 特殊性**：因**无 multicast**，FlashInfer IPC **不可用** → 6000D 实际只剩 **NCCL vs
  B12X**；B12X FP8 ring 仍可用于 prefill 大报文。
- **落地点**：一张「报文大小→后端」选路表入 KB，并显式标注 6000D 的可用子集。
- **可验证**：选路表可静态写；实测需 6000D + B12X。

### 借 5. 前缀缓存把长上下文 TTFT 提升 38–54×，是生产必开项（**最值得做**）

报告二 §5.3：32K TTFT 27.98→0.73 s（38×）、128K 52×、512K 54×。对 serve 的正确性影响：
`ServingReading.ttft_s` 在不同上下文长度、不同前缀缓存状态下会差 50×，若非同口径比较会得出
荒谬结论——与当初「拒跨并发比」是同一类陷阱。

- **做法（纯代码，可单测）**：
  1. `e2e_record.ServingReading` 增加 `prefix_cache: bool` 与 `context_len: int` 字段；
  2. `harness.compare_to_baseline` 拒跨**前缀缓存状态**（不同 `prefix_cache`）与跨上下文长度的比较，
     诊断信息写明原因。
- **落地点**：`src/forgekernel/serve/e2e_record.py`、`src/forgekernel/serve/harness.py` + 单测。
- **可验证**：无 GPU，fake backend 单元测试即可覆盖拒绝逻辑。

### 借 6. CUDA Graph 对 decode 是 10.97→29.23 steps/s（2.7×）

报告二 §4.2：`--enforce-eager` 让 decode 掉到 1/3。这印证 `checks.graph_capturable` 门禁的价值：
**decode 必须能 capture，否则服务静默回退 eager 丢 2.7×**。

- **做法**：KB 加一句动机证据（vLLM 实测），指向 `graph` gate。
- **落地点**：`hw_kb/*/sm120/` 一句话引用 + `checks/__init__.py` 文档已述（补硬件数字）。

---

## 三、不吸收（方向不符，避免过度工程）

| 报告内容 | 为何不吸收 |
|---|---|
| P/D 配比 1P:2D→1P1D、P|D 节点推算 | 整机部署编排；serve 定位是单 kernel 换入 + 门禁，不做全栈 scheduler |
| DSpark 投机 k=5、TP8/EP8 调参、关 thinking | 模型级/采样级调优，非 kernel 层 |
| 1M×20×200 达标评估、H200/B200 选型估算 | 硬件选型决策，非 kernel 可借内容 |
| B12X `a2a`/`i8_ring`/`mx_ring` 编码细节 | 属 b12x 库内部；只借「按报文大小选后端」的结论，不借具体编码实现 |

---

## 四、落地优先级

| 优先级 | 借鉴项 | 成本 | 前置 | 推荐 | 落地点 |
|---|---|---|---|---|---|
| **P0** | 借 5 前缀缓存口径（`ServingReading` + `compare_to_baseline` 拒跨比） | 低 | 无 GPU，纯代码 + 单测 | ✅ | `serve/e2e_record.py`、`serve/harness.py` |
| **P1** | 借 1 `NCCL_P2P_LEVEL=SYS` 默认 env | 低 | 无 GPU 可生成命令 | ✅ | `hw_kb/*/sm120/` + `serve/harness.py` |
| **P1** | 借 3 P2P 延迟/带宽判据 probe 配方 | 中 | PROBE，需 6000D 实测 | ✅ | `hw_kb/*/sm120/` |
| **P2** | 借 2 默认后端 ≠ 最优后端（Cake→NCCL） | 低 | 需 6000D 实测 | ✅ | `hw_kb/*/sm120/` |
| **P2** | 借 4 AllReduce 三段式选路表（标 6000D 子集） | 中 | 需 6000D + B12X | ✅ | `hw_kb/*/sm120/` |
| **P2** | 借 6 CUDA Graph decode 动机数字 | 低 | 引报告即可 | ✅ | `hw_kb/*/sm120/` + `checks` 文档 |

---

## 五、验收标准（可机械检验）

- **借 5**：`ServingReading` 含 `prefix_cache`/`context_len`；`compare_to_baseline` 对跨前缀缓存状态
  或跨上下文长度的比较返回拒绝 + 原因；新增单测断言两种拒绝与同口径放行（无 GPU 可测）。
- **借 1**：`ServeConfig` 生成的 `serve_command`/文档能带出 `NCCL_P2P_LEVEL=SYS`（无 GPU 可测文本）。
- **借 3/4**：KB 条目含选路表/微基准配方，`kb_lint` 通过，标 `State: PARTIAL`（待 6000D 实测）。
- **借 2/6**：KB 条目引用报告实测数字，`kb_lint` 通过。

---

## 一句话总结

**报告的硬件事实仓库已基本覆盖**（无 multicast、decode 带宽受限、SM120≠SM100、config flag 真实）；
真正值得吸收的是 **6 条**——其中**唯一纯代码、当下即可做且可单测的是「借 5：前缀缓存/上下文长度
纳入 `ServingReading` 并让 `compare_to_baseline` 拒绝跨口径比较」**（与「拒跨并发比」同类正确性修复）；
其余 5 条沉淀为 `hw_kb/sm120` 配方（`NCCL_P2P_LEVEL=SYS`、P2P 延迟/带宽判据、默认后端≠最优、
AllReduce 三段式选路、CUDA Graph decode 动机）。整机 P/D 配比、投机调参、硬件选型**不吸收**。
