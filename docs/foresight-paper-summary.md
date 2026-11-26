# ForeSight 论文摘要 & 可借鉴性评估

> 论文：*Exploiting What NVIDIA Profilers Hide: Fast and Accurate Analytical GPU Performance
> Modeling*（Ziwen Wang, Qi Luo, Kunlin Li, Changxi Liu, Dongsheng Wang, Yun Chen），
> alphaXiv 独家，2026-11-10。
> 来源：https://www.alphaxiv.org/zh/abs/2611.analytical-gpu-performance-modeling
>
> 本文分两部分：前半是对论文内容的提炼（仅摘要与 AI 概述所述，未补外部信息）；后半是把它的
> **固定延迟 / 排队停顿二分**思路对照到 ForgeKernel 现有代码，评估能否、在哪里借鉴。

---

## 第一部分 · 论文摘要

### 1. 问题

GPU 架构设计空间大，周期级模拟器（如 Accel-Sim）**太慢**，无法探索；现有分析模型（Habitat 等
基于缩放、GCoM / AMALI 等基于区间/公式）在**换配置后不准**——因为它们把某 warp 的**每一项停顿
一起缩放或重估**，混合了本应稳定与本应随压力变化的两种量。

### 2. 核心洞察：停顿分两类

把 warp 发射间隔里的停顿拆成两种：

| 类别 | 含义 | 跨配置行为 |
|---|---|---|
| **固定延迟停顿** | 等一个结果就绪（数据依赖的固有延迟） | **基本不变** |
| **排队停顿** | 在繁忙硬件单元的队列里等待 | **随各单元压力变化** |

论文用 H100 PCIe 上的 SiLU kernel 说明：活跃 SM 从 114 降到 16，固定延迟停顿几乎不变
（3.5→3.7 周期/指令），而内存排队停顿大幅下降（11.8→3.8）。

**方法论**：复用真机测得的**固定延迟停顿**，只按各硬件单元在目标配置上承受的**压力**重新计算
**排队停顿**。

### 3. 两个组件

- **UnitSight**（分析器）：NVIDIA 的 NCU / CUPTI 默认把逐单元（per-SM / per-L1 / per-L2-slice /
  per-DRAM-channel）计数器**聚合**掉，看不到不均衡；但 CUPTI 的 `CounterData` 缓冲区里其实保留了
  逐单元数据，UnitSight 解码它（也支持 kernel 内读取选定计数器）。在 H20 上以约 44000 个
  SM/SMSP 指标、约 29000 个 L1 指标验证：逐单元数据聚合后与 CUPTI 结果完全一致。
- **ForeSight**（排队分析模型）：
  1. 源 GPU 上**剖一次**：UnitSight 取逐单元计数器 + 静态 SASS 分析（指令数、流水、依赖）+
     NVBit 记录内存地址；
  2. 在 CTA barrier / 外层循环边界处**切成阶段**；
  3. 把源端测量**适配到目标配置**：SM 计数器按"每指令均值"重分配，内存单元按请求率重分配
     （不同 L2 分区分别算分片均值），缓存流量用记录的地址做**缓存模拟**；
  4. 按目标 GPU 的**块调度规则与占用率限制**重放 CTA 调度，在每个事件（CTA 开始/结束/进下一阶段）
     估算活跃 CTA 的指令速率并推进到下一事件；
  5. 逐时刻：复用测得的固定延迟停顿，用**均值分析（mean-value analysis，排队论方法）**算 SM
     调度器/流水/L1 的排队，以及各 L2 分片/DRAM 通道的内存排队；SM 侧与内存侧**联合迭代到收敛**。

### 4. 结果

| 指标 | ForeSight | 最佳基线 |
|---|---|---|
| 内核执行时间 MAPE（九款 NVIDIA GPU） | **8.3%** | Habitat 14.1% |
| vLLM + Llama-3.1-8B 端到端推理 MAPE（H20） | **2.0%**（最大 3.0%） | — |
| 速度（单核，每工作负载/配置） | 中位 **26s** | Accel-Sim 1272s / GCoM 40s / AMALI 108s |
| 设计搜索（H20 派生 180 配置，选 104 SM / 10MB L2 / 4 HBM 堆栈） | 预测**单位成本性能 1.78×** | — |

> 限定（论文自己声明）：8.3% 是汇总值，非每次预测上限；RTX 4090 上 Habitat 反而更好
> （9.3% vs 10.1%）；端到端 2.0% 是**预测耗时**，不代表推理加速；设计搜索的 1.78× 是
> **预测**吞吐 + 面积/堆栈成本模型，未实际流片。

### 5. 一句话

> 固定延迟停顿照抄真机、排队停顿按单元压力重算——用一个能"看见"逐单元计数器的分析器
> （UnitSight）喂数据，用一个排队论模型（ForeSight）联合迭代，做到又快（比模拟器快 49×）
> 又准（MAPE 8.3%）的 GPU 性能分析建模。

---

## 第二部分 · 对 ForgeKernel 的可借鉴性评估

### 0. 先看现状：ForgeKernel 已在做"停顿时清单"，但没做"停顿时建模"

ForgeKernel 已有两套与"停顿 / 杠杆定价"直接相关的机制：

| 现有机制 | 位置 | 做什么 |
|---|---|---|
| **STALL LEDGER**（逐 SASS 指令的动态停顿时清单） | `skills/stall/stall_tools.py` | 读 NCU PC-sampling，按 SASS 指令归属统计各 stall 指标的样本数，输出 `STALL-OK/FAIL` 账本 |
| **stall-pricing run** | `skills/stall/stall_price_run.py` | 在 plateau 诊断时对当前最优 kernel 跑一次，产出实测停顿时清单 + 可选阶段时间线 |
| **lever-pricing**（杠杆消融定价） | `skills/lever-pricing/lever_pricing_run.py` | 对候选杠杆做单加/单删/全开/全减 的**实测消融**，读出每个杠杆的真实价格 |
| **precision roofline** | `hw_kb/_shared/precision_roofline.md` | 各精度吞吐/占峰比/正确性（SM120 实测），属静态 roofline |

关键发现：`stall_tools.py` 的 `STALL_METRICS` **已经天然区分了论文的两类停顿**（按 NVIDIA
stall-reason 语义归类）：

- **固定延迟类**：`long_scoreboard`、`short_scoreboard`（等内存/结果）、`wait`（等固定指令
  延迟，如 MUFU/特殊函数）、`barrier` 与 `membar`（等屏障/内存序——本质是同步延迟，不是某单元
  饱和导致的排队）；
- **排队/争用类**：`math_pipe_throttle`、`mio_throttle`、`lg_throttle`、`tex_throttle`、
  `dispatch_stall`（某单元饱和、请求在队列里等，随占用/压力变化）；
- **其他**：`not_selected`（被 warp 调度器选走，非真停顿）、`no_instructions`、
  `branch_resolving`、`drain`、`sleeping`、`misc`。

也就是说，**定性二分 ForgeKernel 已经有了**；缺的是论文的**定量用途**：把"固定延迟"当常量、
把"排队"当压力函数，从而**外推到另一个配置**或**预测某杠杆的能力上限**。

> 注：`barrier` / `membar` 归固定延迟是按其"等同步事件"语义；若屏障等待主要来自 warp 间
> 负载不均（一 warp 早到等另一 warp），其等待里含可随调度的成分——这对"归桶"是个已知边界，
> 但对下面要做的"占比呈现"足以，不必为此新采数据。

### 1. 直接可借鉴的三处（按契合度排序）

#### ① lever-pricing：用"固定/排队"二分来解释**为什么某杠杆无用**（契合度最高）

现状：`lever_pricing_run.py` 做实测消融，给出每个杠杆的**价格**（加了变快/变慢多少），但它是
**纯实测、单配置**的——它不回答"这个杠杆为什么在别的压力下才有用"。而论文的二分恰好给出解释框架：

- 一个针对**固定延迟**的杠杆（如提前预取、缩短依赖链、改流水线深度）——它的收益**跨配置稳定**，
  且**上限**就是账本里 `long_scoreboard + wait` 那部分的周期数；实测价格若远低于此上限，说明
  被排队掩盖了。
- 一个针对**排队**的杠杆（如加 warp/占用率、错开 bank、减少 barrier 争用、调 tile 降 L2 压力）——
  它的收益**依赖当前压力**；在低占用/单 CTA 场景实测可能近零，却在满占用时关键。

→ **可落地的低成本改进**：在 `stall_price_run.py` 输出的账本里，把每个 stall 指标**归入
固定/排队/其他**三桶（复用现有 `STALL_METRICS` 名称规则即可，无需新数据），并把每桶的
**周期占比**写进 `STALL-OK` 行（如 `fixed=38% queued=51% other=11%`）。这样 lever-pricing 或
plateau 诊断在引用"停顿时清单"时，能直接对标论文的二分，判断一个杠杆该攻固定延迟还是排队，
并给出该方向的**理论上限**（桶内周期数）。这几乎不改采集，只改账本的分组呈现 + 诊断文案。

#### ② plateau 诊断 / 杠杆候选生成：把"固定 vs 排队"作为**分类器**指导下一个杠杆

现状：teacher 在 plateau 时读账本 + lever 价格，自己写 cause_class 与下一个杠杆。
可借鉴：论文的二分给出**机械化的选杠杆规则**——

| 账本信号 | 该攻的方向 | 典型杠杆 |
|---|---|---|
| 固定延迟桶占比高（scoreboard/wait） | 缩短依赖 / 加深流水 / 预取 | `ring_per_stage`、软件流水、`cp.async` 预取 |
| 排队桶占比高（mio/lg/tex/math throttle） | 降压力 / 提占用 / 错开 | 加 warp、调 tile、错 bank、降 L2 冲突 |
| barrier 占比高 | 同步开销 | 减少 stage 数、放宽 barrier、`mbarrier` 化 |

这可以直接写进 teacher 的诊断提示或 `lever-pricing` 的 `expect` 模板，让候选杠杆的产生
**有据可依**（对齐论文"按单元压力选干预"）。

#### ③ precision roofline / 资源建模：给排队桶一个**压力—排队**的反向读数

论文的排队论（mean-value analysis）本质是把"单元压力 → 排队时间"定量化。ForgeKernel 当前的
`precision_roofline.md` 是**静态占峰比**（吞吐 / peak），没有"压力→排队"的动态项。
可借鉴（中长期）：若某 kernel 的排队桶占比高，可用类似"驻留 warp 数 × 每 warp 请求率 / 单元
吞吐"的粗排队估计，**反推当前单元压力**，作为 lever（加 warp 还是降请求）的选择依据。
但这条路要逐单元计数器（论文的 UnitSight 所解）——**NCU 默认聚合，拿不到逐单元**，所以只能停留
在**整 GPU 聚合的粗估**，精度远不及论文，且**当前环境无 GPU**，无法验证。

### 2. 不适合直接借鉴 / 有硬约束的地方

- **逐单元压力**：论文的价值一半来自 UnitSight 解出的 per-SM/per-L2-slice/per-DRAM-channel
  计数器。ForgeKernel 走 NCU/CUPTI 的**聚合**读数，拿不到逐单元不均衡。要复刻需自己解码 CUPTI
  `CounterData`（论文的 UnitSight 做法）——工程量大，且**必须有 GPU 验证**，当前环境（无 torch、
  GPU 机失联）做不了。
- **跨配置外推**：ForeSight 的全部卖点是"在一个配置测、预测另一个配置"（绿色上下文、锁频、MIG
  等）。ForgeKernel 是**同配置单次运行内优化**，没有"目标配置"这个概念——除非将来引入
  "给定候选占用率/tile 预测性能"的需求，否则外推动机不足。
- **周期级重放**：ForeSight 的 CTA 调度重放 + 均值分析迭代收敛是完整的模拟器级建模，远超
  ForgeKernel 需要的"给杠杆定价"粒度。引入它是**过度工程**。
- **性能模型精度**：论文自己也承认 RTX 4090 上不如 Habitat、8.3% 是汇总非上限。ForgeKernel
  当下用**实测消融**（lever-pricing）拿"真价格"，本就比任何分析模型更准——用分析模型去替代
  实测是**退步**。分析模型的价值只在"实测太贵/不可测"时（如预测未造出的配置），而 ForgeKernel
  不做设计空间搜索。

### 3. 结论：借"二分"的**思想与呈现**，不借"建模"

| 借鉴项 | 契合度 | 成本 | 是否推荐 |
|---|---|---|---|
| ① 账本按固定/排队/其他三桶归类 + 占比 | 高 | 低（只改账本分组与文案） | **推荐**，无 GPU 也能改，直接增强 lever-pricing 与 plateau 诊断 |
| ② 用二分作分类器指导下一个杠杆 | 高 | 低（诊断提示模板） | **推荐**，提升候选杠杆的"有据可依" |
| ③ 粗排队估计反推单元压力 | 中 | 中（且需 GPU 验证） | 观望，当前环境无法验证 |
| 逐单元计数器（UnitSight 式） | 低（硬约束） | 高（需自解码 CUPTI + GPU） | 不做 |
| 跨配置外推 / CTA 重放模拟 | 很低（动机不足） | 很高 | 不做 |

一句话：**把论文的"固定延迟 vs 排队"二分当成 ForgeKernel 停顿时清单的一个分析维度与杠杆选择
规则来用——这是零 GPU、低成本、立刻可落地的增强；而论文那套"逐单元 + 重放 + 外推"的建模本体，
与 ForgeKernel 的"同配置实测消融"定位不符，不应引入。**
