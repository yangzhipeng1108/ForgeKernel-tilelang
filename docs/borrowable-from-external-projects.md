# 三个外部项目的可借鉴内容全集（PipeLink / CUDA-L2 / ForeSight）

> 目的：把 [PipeLink](https://github.com/alumkal/PipeLink)、
> [CUDA-L2](https://github.com/ornith-ai/CUDA-L2)、
> [ForeSight](https://www.alphaxiv.org/zh/abs/2611.analytical-gpu-performance-modeling)
> 三个项目的机制**完整拆解**，逐条对照 ForgeKernel 当前实现，标出「已实现 / 可借鉴 / 不该借鉴」，
> 并给出落地点（文件、接口、诊断码、验收）。这是「能借鉴的所有内容」的全集，不删项——
> 不推荐的项也写清为什么不做。
>
> 相关既有文档：`docs/pipelink-feature-status.md`（PipeLink 实现状态）、
> `docs/foresight-paper-summary.md`（ForeSight 评估）、`docs/borrowable-designs.md`
> （ForgeKernel 自身可迁移设计）。本文是把三者合并成「借鉴总表」，并补齐此前缺失的 CUDA-L2。

---

## 0. 三个项目一句话 + 与 ForgeKernel 的关系

| 项目 | 一句话 | 与 ForgeKernel 的关系 |
|---|---|---|
| **PipeLink** | 按**契约**把别人写好的 kernel 阶段链接成融合 kernel，求解流水线环深与资源，自动生成张量交接 | ForgeKernel **已用 `src/forgekernel/link.py` 实现**了一个 TileLang 版链接器；剩下的是执行层/跨 GPU 通信 |
| **CUDA-L2** | 用 **LLM + 强化学习**自动优化 HGEMM（半精度 GEMM）CUDA kernel，系统性地超过 cuBLAS / cuBLASLt（含 **RTX 3090/sm86**） | ForgeKernel **没有 RL/群体搜索**；CUDA-L2 补的是「广度搜索 + 奖励驱动」这条轴 |
| **ForeSight** | 把 warp 停顿分**固定延迟**（真机照抄）与**排队**（按单元压力重算），跨配置预测 GPU 执行时间 | ForgeKernel 的 `stall_tools.py` **已能定性二分**停顿；缺的是「用二分做定量判断/选杠杆」 |

ForgeKernel 的定位（务必记住，决定「该不该借鉴」）：**同配置、单次运行内、用强模型 Teacher 蒸馏
Student 把 kernel 写快**，并靠实测消融（lever-pricing）拿杠杆的**真价格**。它**不做** GPU 设计空间
搜索、**不跨配置外推**、**默认用实测而非分析模型**。任何借鉴项若违背这三条定位，就是「不该借鉴」。

---

## 1. PipeLink —— 全部可借鉴内容

PipeLink 的完整能力清单 + ForgeKernel 现状（详见 `docs/pipelink-feature-status.md`）。
链接的**决策层/控制层已基本做完**，下表「状态」列：✅ 已实现，⬜ 可借鉴，✗ 方向不适用。

### 1.1 能力逐条对照

| # | PipeLink 能力 | ForgeKernel 现状 | 状态 | 落地点 |
|---|---|---|---|---|
| 1 | 组合别人写的阶段（组件契约） | `Component` / `read_component` / `read_contract` / `list_components`；`contract.json` 声明 `axes`/`inputs`/`outputs`/`semantics`/`invariants` | ✅ | `src/forgekernel/component.py` |
| 2 | 交付前静态检查 | `check_tree`（结构 + 接口 + 资源/角色 + 端口 handle/generation） | ✅ | `component.py` |
| 3 | 接缝检查（两阶段 dtype/shape 匹配） | `_seam_for_port` / `_same_port` / `_shape_canon`，失败报 `L-PROTO` | ✅ | `link.py` |
| 4 | 桥接建议（接缝不匹配时点名插入某组件） | `_bridge_hint` → `L-PROTO … insert X` | ✅ | `link.py` |
| 5 | tensor handle（tmem/reg/smem 物理位置） | `_handle_for_edge`；不同 handle 由生成拷贝桥接，未知报 `L-HANDLE` | ✅ | `link.py` |
| 6 | 阶段代际检查（多代在途） | `generation` 字段；不一致报 `L-PHASE` 并给可行环深 `|Δ|+1` | ✅ | `link.py` |
| 7 | 流水线环 + 链接时选环深 | `_ring_limits`/`_ring_throughput`/`_ring_reason` → `T.Pipelined(num_stages=…)` | ✅ | `link.py` |
| 8 | 资源与角色求解（峰值并发，非简单求和） | `_solve` + `_peak_residency`；`L-RES`/`L-ROLE` 点名峰值持有阶段 | ✅ | `link.py` |
| 9 | 自动生成张量交接代码（TMEM/reg/SMEM） | `_handoff_fn` → TileLang `T.copy` + `T.alloc_shared`/`T.alloc_fragment` | ✅ | `link.py` |
| 10 | 混用语言（源留在手里可替换） | 多语言槽位 + 按契约替换（单语言融合） | ✅（单语言） | `link.py` |
| 11 | 前端工具链（CUTLASS/CuTe/TK/TileLang） | 统一为 TileLang 一等目标 | ✅（以 TileLang 替代） | `link.py` |
| 12 | 只检查声明、不检查不透明代码 | 定位相同 | ✅ | `link.py` |
| 13 | CPU 语义一致性证明 | `chained_golden` + `run_parity`（无需 GPU） | ✅ | `link.py` |
| 14 | 生成可 lower 的真实 kernel | `_render_fused_tilelang` → 真实 `@T.prim_func` | ✅ | `link.py` |
| 15 | 规划 / 可视化 / 机读 / 干跑 | `forge component plan`、`render_graph`(ascii/mermaid)、`--format json`、`--from-run` | ✅ | `component_cli.py` |
| 16 | 换阶段重链、报告差异 | `diff_plans`（`forge component diff`） | ✅ | `link.py` |
| 17 | 诊断码体系 | `L-STAGE/L-PROTO/L-WIRE/L-HANDLE/L-PHASE/L-RES/L-ROLE/L-RING/L-ORDER` | ✅ | `link.py` |

### 1.2 PipeLink 尚未做 / 可继续借鉴（⬜）

| 项 | 为什么现在没做 | 可借鉴的落地方案 | 前置 |
|---|---|---|---|
| **a. `.plink` 二进制包（LTO IR + manifest）** | 走 TileLang「源码 + JIT」路线，阶段是 `kernel_tl.py`+`verify.json`，天然可读可替换，不需要二进制封装 | 若将来要「不带源码分发组件」：把 manifest 用 `contract.json` 已覆盖的部分 + 一个 `bundle` 命令打包 `kernel_tl.py`+`verify.json`+golden 为 zip；本域非必需 | 低 |
| **b. 真实 nvcc/CuTe/nvJitLink 编译与 profile 选择** | 由 `@tilelang.jit` 接管 | 无需做（TileLang 管）；若要评 TileLang vs 手写，可加 `compile_profiles` 字段记录 `sm_86/sm_90a/...` 供审计 | 无 |
| **c. 主机端 ABI（provider 库/C 头/persistent 会话/CUDA graph）** | persistent/CUDA graph 要主机侧代码（`torch.cuda.CUDAGraph`），需 GPU 验证 | 加一个「launch 模式」抽象：`single_launch` 现在有；`cuda_graph` / `persistent` 作为声明字段 + 生成占位 launcher，留 GPU 验证钩子 | 需 GPU |
| **d. 跨 GPU 通信重叠（通信链进计算 kernel，NVLink 重叠）** | TileLang 面向单 kernel 单 CTA 内计算，不覆盖跨 GPU 通信 | **无 GPU 也能做的部分**：在 `contract.json` 声明 `comm_domain`/`ready_counter`/`multicast`；链接时校验跨阶段就绪是否匹配（静态检查，不实跑 NVLink）。真运行留 GPU | 静态可做 |
| **e. 多槽环运行时（每代一槽的真实行为）** | `L-PHASE` 已建模所需环深并映射 `T.Pipelined`，但真运行时多槽语义需 GPU 验证 | 已经静态算出环深；补一个 `ring_slot_map` 声明（哪个 generation→哪个槽）供生成器用，运行验证留 GPU | 需 GPU 验证 |
| **f. 真实资源分配器（分配/复用图，而非峰值上界）** | TileLang 自动分配 SMEM/寄存器 | 保留峰值上界做静态拒绝；精确读数（每阶段实际 SMEM/TMEM）从编译产物解析，留 GPU | 需 GPU |
| **g. 多前端交叉链接 / 打过补丁的 TileLang 后端** | 单语言路线下非必需 | 不推荐（背离「以 TileLang 为一等目标」） | — |
| **h. 硬件 evaluator（测吞吐/memcheck/racecheck/synccheck 出证据）** | 只做 CPU parity，硬件测速需 GPU | 接入 `TILELANG do_bench/Profiler` 作为可选 evaluator；产出「证据 JSON」进 KB | 需 GPU |

### 1.3 PipeLink 特有的、值得移植的**思想**（非具体功能）

- **「按契约链接，而非按名字」**：ForgeKernel 的多 kernel/多原子拼接也应只信 `contract.json` 的声明，
  绝不靠文件名匹配。这是已落地的原则。
- **「链接期把所有能查的错都查了」**：把本应在运行期 hang 的错（dtype/shape/handle/generation/资源）
  全部前移到链接期用编号诊断拒绝——这条应推广到 ForgeKernel 的**里程碑交付前检查**（`check_tree` 已做）。
- **「峰值并发而非求和」**：资源求解按「阶段从启动驻留到最后一个消费者启动」求峰值，而非累加。
  ForgeKernel 的 KB/里程碑若描述 CTA 常驻，应采用同一口径（已实现）。

---

## 2. CUDA-L2 —— 全部可借鉴内容（ForgeKernel 完全缺失的一条轴）

CUDA-L2 是 LLM + **强化学习**优化 HGEMM 的系统。要借鉴的是它的**范式**，不是它的某一组 kernel。

### 2.1 CUDA-L2 的核心机制拆解

1. **奖励驱动搜索**：把「kernel 变体在某 MNK 配置上的实测加速（相对 cuBLAS/torch）」作为 reward，
   用 RL 让 LLM 朝高 reward 方向变异 kernel。不是单轨试错，是**群体 + 选择/变异**。
2. **双评测模式**：
   - `offline`（批处理吞吐，warmup + benchmark 稳态计时）；
   - `server`（按 `--target_qps` 的**延迟敏感**评测，模拟请求式场景）。
   两者口径不同，同一 kernel 表现可能不同，必须分开报。
3. **配置矩阵**：按 `--mnk M_N_K` 组织问题规模（如 `64_4096_64`），每配置单独优化。
4. **精度累加器开关**：`--acc_precise fp16|fp32`（F16F16F16F16 / F32F16F16F32），影响数值与速度。
5. **设备矩阵**：A100 / **RTX 3090（sm86）** / H100，按设备分别训练/发布 kernel——**跨设备的
   kernel 不能通用**（FAQ 明确：A100 训的不保证在 3090 快）。
6. **基线族**：torch.matmul / cuBLAS / cuBLASLt-heuristic / cuBLASLt-AutoTuning，逐一报加速比。
7. **发布成果**：每个设备 × 精度 × 配置释出**一批**（如 1000 个 A100 kernel）可选 kernel。

### 2.2 能借鉴进 ForgeKernel 的功能（⬜，全部）

| # | CUDA-L2 机制 | 改造成 ForgeKernel 的什么 | 落地点 | 前置 |
|---|---|---|---|---|
| A1 | RL/进化式变体搜索 | **新增「广度搜索环」**：在现有 `run_inner_loop`（单轨试错）之外，加一个**候选群体 + reward 选择**层。reward = 候选相对 reference 的实测加速（复用 `trajectory.IterationRecord.verdict/metric_after`） | 新模块 `src/forgekernel/loop/search.py`（或 `skills/kernel-search/`），由 driver 调度 | 需 GPU 验证 |
| A2 | offline / server 双评测 | GPU job 评测加 `mode=offline\|server` + `target_qps`；server 模式按 QPS 计延迟敏感分。接入现有 GPU submit 流程 | `src/forgekernel/loop/job_env.py` / GPU job 定义 | 需 GPU |
| A3 | MNK 配置矩阵 | 把 GEMM 的 `M_N_K` 做成 `tasks/` 里的显式任务集（对齐 `shapes.json`），每配置一个 target | `tasks/gemm-*/shapes.json` | 低 |
| A4 | acc_precise 精度开关 | task.toml 增加累加器精度字段；评测按精度分开报，避免混比 | `tasks/*/task.toml` + precision_gate | 低 |
| A5 | 设备分离训练/发布 | 与本地刚做的 **sm86 支持**天然契合：把 CUDA-L2 的 **RTX 3090 HGEMM kernel 作为 hw_kb/sm86 的真实范例** + 参考基线 | `hw_kb/*/sm86/` | 低（kernel 现成） |
| A6 | 基线族（cuBLAS/cuBLASLt/Triton/torch） | 给 TASK 加「多基线对比」：学习 CUDA-L2 同时报多个基线的做法，避免只报 torch 显得虚高 | leaderboard / report 生成 | 中 |
| A7 | 一批 kernel 可选 | 借鉴「每配置释出一批 kernel 供选最优」：run 结束时除 `best/kernel.py` 外，存一个 `candidates/` 池 + 每候选的 demangled 指标 | run 产物结构 | 低 |

### 2.3 CUDA-L2 与 ForgeKernel 定位的**根本差异**（决定借鉴方式）

| 维度 | CUDA-L2 | ForgeKernel |
|---|---|---|
| 搜索方式 | RL 群体搜索，广度优先 | 单轨试错 + Teacher 蒸馏，深度优先 |
| 优化对象 | 纯 GEMM（HGEMM）单一算子族 | 任意算子（attention/gemm/conv/...） |
| 知识沉淀 | 每组配置一批 kernel | 跨算子 KB（hw_kb）+ 里程碑课程 |
| 驱动信号 | reward=加速比 | 双环：内环改产物、外环改策略 |

→ **借鉴方式**：不替换 ForgeKernel 的「蒸馏 + 双环」，而是**在它内部加一个可选的广度搜索子环**（A1），
并复用它的评测/基线/设备分离做法（A2/A5/A6）。这是「补一条正交的轴」，不是「换掉主干」。

### 2.4 CUDA-L2 不该直接照搬的

- **纯 GEMM 局限**：CUDA-L2 只做 HGEMM，ForgeKernel 要覆盖多算子族，RL 搜索环必须**通用化**成
  「任意 kernel 的候选群体」，不能写死 GEMM 逻辑。
- **预训练好的一批 kernel**：那是 CUDA-L2 的产物，不是机制；借鉴机制即可，产物（3090 HGEMM）
  只作为 sm86 的 KB 范例/基线（A5）。
- **CUTLASS v4.2.1 依赖**：那是它生成 kernel 的工具链，ForgeKernel 走 TileLang，不需要照搬。

---

## 3. ForeSight —— 全部可借鉴内容

详见 `docs/foresight-paper-summary.md`。这里给全集表。

### 3.1 ForeSight 的核心机制

- **停顿二分**：固定延迟停顿（等结果就绪，跨配置基本不变）vs 排队停顿（在某单元队列等，随压力变）。
- **UnitSight**：解 CUPTI `CounterData` 里被聚合掉的**逐单元计数器**（per-SM/per-L1/per-L2-slice/
  per-DRAM-channel）。
- **ForeSight 模型**：源 GPU 剖一次 → 切阶段 → 适配到目标配置（SM 计数器按每指令均值重分、
  内存按请求率重分、缓存用地址重放）→ 按目标块调度规则重放 CTA → 逐时刻复用固定延迟、用**排队论
  mean-value analysis** 算 SM/L1/L2/DRAM 排队，联合迭代到收敛。
- **结果**：内核 MAPE 8.3%（基线 14.1%）、vLLM 端到端 2.0%、比 Accel-Sim 快 49×。

### 3.2 能借鉴进 ForgeKernel 的功能（⬜，全集）

| # | ForeSight 机制 | 改造成 ForgeKernel 的什么 | 落地点 | 契合度/成本/前置 |
|---|---|---|---|---|
| F1 | 固定/排队/其他**三桶归类 + 占比** | 停顿时清单每个 stall 指标归入固定/排队/其他，每桶**周期占比**写进 `STALL-OK`（如 `fixed=38% queued=51% other=11%`）。`STALL_METRICS` 已天然可分（见 3.3） | `skills/stall/stall_tools.py`（`short_stall`/账本输出） | 高/低/无 GPU ✅推荐 |
| F2 | 用二分作**选杠杆分类器** | teacher 诊断模板里加规则表：固定桶高→攻依赖/流水/预取；排队桶高→提占用/错 tile/错 bank/降 L2；barrier 高→减 stage/mbarrier 化 | teacher 诊断 prompt、lever-pricing 的 `expect` 模板 | 高/低/无 GPU ✅推荐 |
| F3 | 桶内周期数 = 杠杆**理论上限** | lever 实测价格若远低于桶内周期数 → 说明被另一类掩盖，诊断里标注「该方向上限=X 周期，实测仅拿到 Y」 | plateau 诊断、`docs` 文案 | 高/低/无 GPU ✅推荐 |
| F4 | 排队论反推单元压力 | 粗排队估计（驻留 warp × 每 warp 请求率 / 单元吞吐）反推当前压力，作为「加 warp 还是降请求」的依据 | 需要逐单元/聚合计数器的排队估计模块 | 中/中/**需 GPU 验证**；且只能整 GPU 聚合、精度远低 |
| F5 | 逐单元计数器（UnitSight 式） | 自解码 CUPTI `CounterData` 拿 per-SM/per-L2-slice 数据 | 新分析器（工程量大） | 低/高/**需 GPU**；与「实测消融」定位不符 → 不做 |
| F6 | 跨配置外推（绿色上下文/锁频/MIG 预测） | 给定候选占用率/tile 预测性能 | — | 很低/高/动机不足 → 不做 |
| F7 | CTA 调度重放 + 均值分析联合迭代 | 完整模拟器级建模 | — | 很低/很高/过度工程 → 不做 |
| F8 | 端到端预测（把各 kernel 预测相加 → 推理耗时） | 若将来要预测整条推理链路耗时 | — | 低/高/超出当前定位 → 观望 |

### 3.3 F1 的可信依据：`STALL_METRICS` 已天然二分

`skills/stall/stall_tools.py` 的 `STALL_METRICS`（16 个）按 NVIDIA stall-reason 语义可归三桶，
**无需新采数据**，只在账本输出处加分组：

| 桶 | 指标（去前缀后） | 归类理由 |
|---|---|---|
| **固定延迟** | `long_scoreboard`、`short_scoreboard`、`wait`、`barrier`、`membar` | 等内存/结果/固定指令延迟/同步事件——跨配置基本稳定 |
| **排队/争用** | `math_pipe_throttle`、`mio_throttle`、`lg_throttle`、`tex_throttle`、`dispatch_stall` | 某单元饱和、请求在队列等——随占用/压力变 |
| **其他** | `not_selected`、`no_instructions`、`branch_resolving`、`drain`、`sleeping`、`misc` | 非真停顿/被调度器选走/分支解析等 |

> 已知边界：`barrier`/`membar` 若主要来自 warp 间负载不均（而非纯同步延迟），其等待含可随调度
> 变的成分；对「占比呈现」够用，不必为此新采数据（同 `docs/foresight-paper-summary.md`）。

### 3.4 为什么「二分思想 + 呈现」可借，「建模本体」不可借

- ForgeKernel **同配置单次运行**，没有「目标配置」概念 → 外推（F6）、CTA 重放（F7）无动机。
- ForgeKernel 用**实测消融**（lever-pricing）拿真价格，**比任何分析模型更准** → 用分析模型替代实测
  是退步；分析模型只在「实测太贵/不可测」时才有价值（如未造出的配置），ForgeKernel 不做设计搜索。
- 逐单元计数器（F5）要自解码 CUPTI + 必须 GPU 验证，与「无 GPU 也能改」的目标冲突。

---

## 4. 跨项目合并后的落地优先级总表

| 优先级 | 借鉴项 | 来源 | 成本 | 前置 | 推荐 | 落地状态 |
|---|---|---|---|---|---|---|
| **P0** | F1 停顿时清单三桶归类 + 占比 | ForeSight | 低 | 无 GPU，纯改 `stall_tools.py` | ✅ | ✅ 已完成（`stall_tools.py` + 单测） |
| **P0** | F2 二分作选杠杆分类器（诊断模板） | ForeSight | 低 | 无 GPU，改 prompt | ✅ | ✅ 已完成（`diagnose_prompt.md` + drift-guard） |
| **P0** | F3 桶内周期 = 理论上限，标注杠杆差距 | ForeSight | 低 | 无 GPU | ✅ | ✅ 已完成 | 
| **P1** | A1 RL/进化式候选群体搜索环（reward=实测加速） | CUDA-L2 | 中高 | 需 GPU | ✅（补广度轴） | ✅ 已完成（`search_loop.evolve` + 14 单测，fake backend） |
| **P1** | A2 offline/server 双评测口径 | CUDA-L2 | 中 | 需 GPU | ✅ | ✅ 已完成（`timing_audit` + `server_bench`，3090 真跑烟测通过） |
| **P1** | A5 sm86/3090 HGEMM 入 hw_kb + 当基线 | CUDA-L2 | 低 | kernel 现成 | ✅ | ✅ 机制完成；基线已 3090 实测（69.6 TF=98% 峰值 @8192³，PROBE PASS 计时），仍标 PARTIAL 待 NCU 计数器 |
| **P2** | A3 MNK 配置矩阵 → tasks/ | CUDA-L2 | 低 | — | ✅ | ✅ 已完成（`tasks/gemm-mnk/` 6 config） |
| **P2** | A4 acc_precise 精度开关 + 分报 | CUDA-L2 | 低 | — | ✅ | ✅ 已完成（`task.acc_precise` + 单测） |
| **P2** | A6 多基线对比（cuBLAS/cuBLASLt/Triton/torch） | CUDA-L2 | 中 | — | ✅ | ✅ 已完成（`launch.save_comparison_baseline`/`render_comparison_table` + 单测） |
| **P2** | A7 每 run 存 candidates/ 池 + 每候选指标 | CUDA-L2 | 低 | — | ✅ | ✅ 已完成（`goal_ref` `candidates/NNN`） |
| **P2** | 链接器产出沉淀为 KB recipe | PipeLink | 中 | — | ⬜ | ✅ 已完成（`link.render_kb_recipe`/`render_kb_recipe_for` + 单测） |
| **P2** | 跨 GPU 通信声明级建模（静态校验） | PipeLink | 中 | 仅静态可做，运行需 GPU | ⬜ | ✅ 已完成（`link.L-XGPU`/`cross_gpu` + `component.TRANSPORTS` + 单测） |
| **P3** | PipeLink `.plink` 包 / CUDA graph / persistent | PipeLink | 中 | 需 GPU / 方向非必需 | 观望 | 观望（非必需） |
| **P3** | F4 粗排队估计反推压力 | ForeSight | 中 | 需 GPU | 观望 | ✅ 已完成（`timing_audit.estimate_queue` Little 定律 + 单测） |
| **✗** | F5 逐单元计数器 / F6 外推 / F7 CTA 重放 | ForeSight | 高 | 需 GPU + 定位不符 | 不做 | 不做 |
| **✗** | PipeLink 多前端交叉链接 / CUDA-L2 纯 GEMM 写死 / 照搬 CUTLASS 依赖 | 三者 | — | — | 不做 | 不做 |

---

## 5. 各借鉴项的验收标准（可机械检验）

- **F1/F3**：`stall_tools.py` 账本输出每行含 `fixed=..% queued=..% other=..%`，三桶之和 = 100%，
  且每桶都给出周期数；新增单测用固定输入断言分组正确（无 GPU 可测）。
- **F2**：teacher 诊断模板里含「桶→方向→杠杆」映射表；给定一份 `fixed=..%/queued=..%` 账本，
  prompt 能机械推出建议方向（可用快照测试）。
- **A1**：搜索环在无 GPU 下用 fake backend 跑「生成 N 候选→按 reward 排序→选 top→变异」的单元测试；
  reward 读取自 `IterationRecord.metric_after` 与 reference 比较。
- **A2**：`mode=server --target_qps` 时 GPU job 按期/QPS 计时，报告独立于 offline 口径。
- **A5**：`hw_kb/*/sm86/` 增加 3090 HGEMM 范例 + 基线数字，kb_lint 通过。
- **PipeLink 跨 GPU 建模**：`contract.json` 加 `comm_domain/ready_counter` 字段后，`check_tree`
  能对「就绪计数器不匹配」报新诊断码（如 `L-COMM`），纯静态、无 GPU 可测。

---

## 6. 一句话总结

- **PipeLink**：决策/控制层已基本做完（契约/接缝/句柄/代际/环深/资源/交接/诊断/CPU parity）；
  还能借鉴的是**跨 GPU 通信的声明级静态建模**与**把链接产出沉淀进 KB**，其余执行层需 GPU 或非必需。
- **CUDA-L2**：ForgeKernel **完全缺失**的「RL/进化广度搜索 + reward 驱动 + offline/server 双评测 +
  设备分离（含 sm86 3090）+ 多基线」这一整条轴，应作为**正交子环**补进来，不替换蒸馏主干。
- **ForeSight**：只借「固定延迟 vs 排队」的**二分思想与三桶呈现/选杠杆规则**（零 GPU、低成本、立即可做）；
  其逐单元解码、跨配置外推、CTA 重放建模本体与 ForgeKernel 定位不符，不做。

> 落地建议顺序：先做 **P0（ForeSight 三桶 + 选杠杆规则，无 GPU 可立即测试）**，再做 **P2 低成本的
> CUDA-L2 评测/任务/基线改造（A3/A4/A5/A6/A7）**，最后才是需 GPU 的 **A1 搜索环 / A2 双评测 /
> PipeLink 跨 GPU 运行验证**。
