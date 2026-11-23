# Cohere megakernel 调度器精读：能否用 TileLang 移植到 Qwen3.8-Flash / sm120

> 状态：**源码精读 + 静态分析，未在本机运行**。cohere-ai/cohere-megakernel 未在本仓库 clone，
> 读的是 GitHub 上的 `main`（2026-09-08 最后 push）对应文件。凡标「源码」的结论都可回查行号；
> 凡标「推断」的是我基于源码的推论，没有硬件证据。

## TL;DR

**不能移植。** 不是工作量问题，是三处硬性不兼容：调度器的加速前提在 Qwen3.8 上不成立、
指令流 ABI 绑定 ThunderKittens 的 warpgroup 模型、目标架构是 `sm_90a` 而目标是 `sm_120a`。
TileLang 连调度器最基本的承载体（持久化 + 运行时改 task 流 + 设备侧 spin barrier）都还没有。

但它的**三个机制设计**值得抄进 TileLang 的路线图，只有其中一个在 sm120 上有正收益的可能性。

## 1. 调度器怎么工作（源码事实）

### 1.1 数据结构

一条指令 = **32 个 int32**（`decode/schedule.py:30` `NMC_INSTRUCTION_WORDS = 32`），
每个 SM 有自己的一条指令流：

```c
// decode/abi.h:41-88  NmcLaunchDesc（LAYOUT-LOCKED，跨 dlsym 边界）
uint64_t inst_buf;          // 指令流基址
uint64_t num_inst_per_sm;   // 每 SM 指令条数
int32_t  max_inst;
// 14 个 barrier 数组
uint64_t bar_upgate, bar_silu, bar_ffn_down, bar_qkv, bar_combine, bar_attn,
         bar_oproj, bar_layer, bar_router, bar_topk, bar_route, bar_gather,
         bar_moe_upgate, bar_moe_down;
```

启动就是一次普通 launch（`decode/launch.cuh:165`）：

```c
mk_nmc_kernel<Cfg><<<dim3(num_sms), dim3(Cfg::NUM_THREADS), smem_size, stream>>>(g);
```

`num_sms` 个 block，每 block 384 线程（`schedule.py:624` `num_threads=384, num_warps=12`），
跑完自己的指令流才退出。**整个 decode step 一次 launch。**

### 1.2 依赖如何编码

每条指令自带三个依赖字段（`schedule.py:703-714`，`gemm_field`）：

| 字段 | 含义 |
|---|---|
| `WAIT_BAR_IDX` | 要等的 barrier 编号 |
| `WAIT_TARGET` | 该 barrier 的计数值到达多少才放行 |
| `PRODUCE_BAR_IDX` | 完成后递增哪个 barrier |

设备侧是**全局内存里的计数器**：spin 直到 `bar[i] >= target`，完成后 `atomicAdd(bar[j], 1)`。
这是 classic counter-based synchronization，没有 `__syncthreads` 之外的硬件网格同步。

17 个 opcode（`schedule.py:631-636`）：

```
0 NOP            1 FFN_DOWN       2 QKV_PROJ      3 ATTN_DECODE   4 ATTN_COMBINE
5 O_PROJ         6 LM_HEAD        7 FFN_UPGATE_ACT 8 ATTN_DRAIN   9 ROUTER_GEMM
10 ROUTER_TOPK  11 ROUTE_FINALIZE 12 MOE_GATHER   13 MOE_UPGATE_ACT_DRAIN
14 MOE_DOWN_DRAIN 15 MOE_COMBINE  16 ADD_RMSNORM  17 GRID_SYNC（仅消融用）
```

### 1.3 并行度从哪来

`schedule.py:1846-1859` —— 这是全篇最关键的一段：

```python
# Attention and FFN run in parallel and rejoin at rmsnorm.
if not is_moe:
    waves += reorder_waves(layer, named,
        "qkv", "ffn_upgate", "attn", "attn_combine",
        "ffn_down", "oproj", "rmsnorm")
else:
    waves += reorder_waves(layer, named,
        "qkv", "router", "router_topk", "route_finalize", "moe_gather",
        "attn", "attn_combine",
        "moe_up_drain", "moe_down_drain", "moe_combine",
        "oproj", "rmsnorm")
```

Blog 明说（`Where the speedup comes from` 第 1 条）：

> North Mini Code gains more from this than most architectures because it uses
> **parallel transformer layers**: attention and the MoE feed-forward are computed from the
> same normalized input and are rejoined only by a fused residual add + RMSNorm at the end
> of the layer, so neither attention nor the MoE needs each other's output.

**即：加速的主要来源，建立在 North Mini Code 的 parallel layer 结构上。**

### 1.4 动态部分：两个共享工作队列

静态指令流表达不了两种长度未知的工作，它们走设备侧队列：
`attn_queue_words` / `attn_queue_heads` / `moe_up_task_words` / `moe_down_task_words`。
谁先空谁去抢，用 `ATTN_DYNAMIC_SENTINEL = 0` 让长度在**每次 launch 前由 host 刷新**
（`schedule.py:1539-1545`）。这是它能适应任意 context 长度和 ragged batch 的机制。

### 1.5 放置策略

`_dependency_affinity_round_robin_waves`（`schedule.py:1287`）：在保持 wave 顺序与
每 SM 配额的前提下，让消费者尽量落到生产者的 SM 上（`attn_combine`、`router_topk`、
`route_finalize`、`moe_gather` 四类有 affinity）。注释里写明：**barrier 才是正确性来源，
放置只影响性能**。

## 2. 为什么不能移植到 Qwen3.8-Flash

### 2.1 加速前提不成立（决定性）

| | North Mini Code（它优化的对象） | Qwen3.8-Flash（你的对象） |
|---|---|---|
| 层结构 | **parallel layer**：attn 与 FFN 从同一 norm 输入出发，末尾才汇合 | 48 层 = 36 linear_attn + 12 full_attn，**串行**（`full_attention_interval=4`） |
| 规模 | hidden 2048、128 experts、top-8、expert_dff 768（`schedule.py:33-36`） | hidden 2560、**512 experts、top-10**、moe_inter 640（部署方案 `:19-23`） |
| 权重 | BF16，单卡 6.6 GB/step | **NVFP4，~130 GB，~75 GiB/卡** |
| 卡数 | 1×H100 | **2×RTX 6000D，无 NVLink** |

Blog 的收益分解里，第 1 条（消除 wave quantization / backfill）**明确依赖 parallel layer**。
Qwen3.8 是串行的：attn 必须做完才能进 FFN，没有可 backfill 的空档。
剩下第 2 条（去掉假依赖）和第 3 条（权重预取）在串行结构上仍然成立，但量级小得多。

它的自身数据也印证了这一点：end-to-end 只有 1.25–1.41×，而 decode-only 是 1.32–1.58×。
Blog 说差异来自「prefill 仍是独立 kernel」**和**「expert routing 随 batch 变化，加速比不恒定」。
512 experts top-10 的 batch=8 decode 只有 80 条 route，**绝大多数 expert 被激活 0 次**——
它那套「按 expert 切 row-block 灌满 SM」的调度在这种极稀疏激活下会大面积空转。

### 2.2 指令流 ABI 绑死 ThunderKittens warpgroup 模型

`launch.cuh:70-77` —— 权重是按 **layer × expert 的 3D GL（global layout）** 组织的：

```c
make_nmc_w_layered_gl(W_moe_upgate, L * NMC_NUM_EXPERTS, 2u * NMC_EXPERT_DFF, D)
```

tiling 也是硬编码的常量表（`schedule.py:603-621`）：`bm=16`、`qkv.stages=10`、
`oproj.split_k=4`、`down.split_k=2`，`num_cwg=1~2`，`num_warps=12`。
32-int 指令 ABI 的每个字段（`TILE_M`/`K_TILES`/`split`）都是喂给这些**固定的 TK 原语**的。
换成 512 experts、hidden 2560、NVFP4 的 4-bit 打包，指令语义本身就要重定义，
不存在「把 TK 换成 TileLang 而指令流不动」的路径。

### 2.3 目标架构不同

README 与 BUILD_AND_RUN 都写死 `sm_90a`，`CMakeLists.txt` 不构建其他架构。
它依赖 WGMMA（`gemm-n8-wgmma.cuh` 专门补了 TK 没有的 `m64n8k16`）。
你自己的实测：RTX 6000D 是 `HMMA`，无 wgmma、无 tcgen05（`hw_kb/cutedsl/README.md:39`）。
ThunderKittens 的 sm90 warpgroup 原语在 sm120 上不存在，移植不等于改 `-arch`。

### 2.4 多卡缺位

它的调度器**没有跨卡概念**：`NmcLaunchDesc` 里没有 rank、没有 peer 指针，
`k_cache`/`v_pool` 都是本卡地址。Qwen3.8 走 EP2，MoE 必须跨卡 all-to-all，
而该机器的物理上限是 **45 GB/s**（部署方案 `:153-167`）。megakernel 省的是**卡内**间隙。

## 3. TileLang 侧缺什么（逐项对照）

| 调度器需要 | TileLang 0.1.15 sm120 现状 | 证据 |
|---|---|---|
| 一次 launch 跑完，block 常驻 | `T.Kernel(threads=)` 可以，**但**没有「跑完整个 step」的循环载体 | 需手写 while + 指令指针，可行但全是胶水 |
| 指令流放全局内存、host 侧重建 | **无**。`T.prim_func` 是静态 AST，编译期定死控制流 | 静态不支持 |
| 设备侧 spin on 全局计数器 | **无原语**。`T.atomic_add` 有，spin-wait 要 raw PTX | 需 `T.import_source` + `call_extern` 逃逸 |
| per-SM 静态共享内存 + 384 线程 warp specialization | 有 `T.Pipelined`/warpgroup，但**不可控** | KB 已记：stages 在 2 处平台化 |
| 网格级 barrier（消融用的 GRID_SYNC） | **CLC 在 sm120 编译成空** | `hw_kb/tilelang/sm120/clc_compiles_to_nothing.md` |
| 细粒度 task queue（attn drain / MoE drain） | **无** | — |
| GDN 线性注意力（36/48 层） | **无任何 sm120 TileLang kernel** | `kernels/` 只有 9 个单算子 kernel |
| 512-expert grouped MoE | **无** | 同上 |

结论：即使不谈模型差异，**TileLang 现在不具备写这个调度器的表达能力**。
它的「静态编译」与「运行时改指令流」在根本上冲突——这正是 cohere 选单文件 CUDA 的原因
（Blog: "no compiler, no new programming paradigm"）。

## 4. 值得抄的三个机制（按可移植性排序）

| 机制 | cohere 的做法 | 能否进 TileLang | 预期收益 |
|---|---|---|---|
| **权重预取** | router/QKV 在上层 O-proj 尾部就开始拉权重，不等 RMSNorm | **能**，`T.Pipelined` 的 prefetch 已部分具备；关键是「跨算子边界预取」需手工拼接 | 小但确定 |
| **去掉假依赖** | O-proj 按 KV group 分别放行；MoE down 按 expert 分别放行 | 部分能：TileLang 无法表达设备侧 spin，但可以在一个 kernel 内把 2 个算子拼起来省掉中间 barrier | 中，限于算子内融合 |
| **消除 wave quantization** | 空闲 SM 去抢别的算子的 tile | **不能**：依赖 parallel layer，Qwen3.8 没有 | 0 |

第 3 条是它在 North Mini Code 上的最大收益来源，也是**唯一完全无法迁移**的一条。

## 4bis. 按 ForgeKernel 自己的诊断框架定性

ForgeKernel 的 meta-milestone 把「学生卡住」分成两类（G = 引导失败，R = 触达失败）。
把本问题代入，答案会更清楚它属于哪一类、该由谁修。

| 诊断 | 定义 | 本问题是吗 | 理由 |
|---|---|---|---|
| **G1 欠定** | 缺目标/触发条件 | 否 | 「用持久化 kernel 融合 decode」这个决策写得很清楚，不缺方向 |
| **G2 过粗** | 一步太大，实施不了 | **是（部分）** | 「写一个 megakernel」是数百行的单一巨步，必须拆；但拆完也解决不了下面的 G3/R1 |
| **G3 接口未知** | 知道做什么，写不出硬件接口 | **是（核心）** | TileLang 没有持久化调度载体、设备侧 spin barrier、CLC（sm120 编译成空） |
| **G4 学生噪声** | 方法够了，实现偶发错 | 否 | 不是实现抖动问题 |
| **R1 地基天花板** | 当前架构限制了可达收益 | **是（决定性）** | Qwen3.8 是串行 48 层，backfill 这条最大收益路径物理上不存在 |
| **R2 缺决策** | 路径走完还漏一个方向 | 否 | — |

按 meta-milestone 的「地基优先」原则，**修 G3 贵、修 R1 更贵，且 R1 不是补丁能解决的**：

- G3 的正确处理是**补 `hw_kb/` 配方**（这是 meta-milestone 明确指定的目的地），不是改架构；
- R1 的正确处理是「**定价后重开地基**」，而在本例中，定价的结论是**天花板不可破**——
  因为限制来自模型结构（串行）与卡间带宽（45 GB/s），不是来自 kernel 写法。

因此本问题的正确定性是：**G3 + R1 的叠加，且 R1 属不可移除类型**。
按这套方法论，它不该进入 milestone 课程，而应记为一条 **Known dead end**。

## 4ter. 这份判断在 ForgeKernel 的产物体系里该落在哪

| 产物 | 是否该记录 | 内容 |
|---|---|---|
| 单算子 milestone | **否** | 按 A.6 的 INFEASIBLE 规则记入死路，不生成课程 |
| `hw_kb/tilelang/sm120/` | **是（部分）** | 「TileLang 缺持久化调度 / 自旋 barrier / CLC 空编译」是真实 G3 缺口，值得成条目 |
| `meta_milestone.md` | 否 | 这是方法层，本问题不暴露方法缺陷 |

「权重预取」「同层去假依赖」两条可迁移机制属于**机制记忆**，符合 KB 的第一性原理
（同一接口问题在多个算子重复出现），应作为候选 KB 条目。

## 5. 该先做什么（不需要 megakernel）

在投入任何移植工作前，先测一个数：

> **现有 9 个 sm120 TileLang kernel 串行跑一个 decode step，kernel 间间隙占总时间多少？**

- 若间隙占比 < 5%：megakernel 在这台机器上不值做，直接放弃。
- 若间隙占比 > 15%：才值得讨论融合，且优先做**同层内 2–3 个算子的手工融合**
  （如 RMSNorm + QKV、O-proj + residual），而不是全局 megakernel。

第二件（可选）：把 cohere 的 `megakernel.cuh` 里 **task interpreter 主循环**
（controller warp 读 descriptor → 分派到 op 函数）单独读一遍，看能否用 TileLang 的
`T.import_source` + `call_extern` 把那层「分派」封成 C++ 而当算子仍用 TileLang 写。
这是唯一有可能保留 TileLang 生产力又得到调度灵活性的路径，但仍是研究项目，不是改动。

## 6. 一句话结论

**「用 TileLang 修改 cohere 的 megakernel 来支持 Qwen3.8-Flash」不存在可执行路径**：
它的调度器为「单卡 H100 + BF16 + parallel layer + 小 MoE」专门设计，
而 Qwen3.8-Flash 是「双卡 sm120 + NVFP4 + 串行混合注意力 + 512-expert MoE」。
两条路上唯一的交集是「权重预取」和「算子内去假依赖」，这两个不需要 megakernel。

按 ForgeKernel 自己的 G/R 分类，这是 **G3（TileLang 缺机制）+ R1（模型结构天花板）的叠加**，
且 R1 属于不可移除类型（串行层结构 + 45 GB/s 卡间带宽，均非 kernel 写法所能改变）。
按方法论，它应收为 **Known dead end**，而不是变成一条里程碑决策。

## 附：源码位置

- `decode/schedule.py:30` 指令宽度、`:631-636` opcode、`:703-800` 指令字段布局
- `decode/schedule.py:1287` 亲和性放置、`:1507` wave 构造、`:1846-1859` 并行排序（关键）
- `decode/schedule.py:33-36` North Mini Code 的硬编码几何
- `decode/abi.h:41-88` launch descriptor（LAYOUT-LOCKED）
- `decode/launch.cuh:165` 单次 launch、`:70-77` 分层权重布局
- Blog「Where the speedup comes from」：三条收益、parallel layer 依赖
