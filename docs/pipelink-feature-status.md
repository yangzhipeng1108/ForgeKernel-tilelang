# PipeLink 功能实现状态（已完成 / 未完成）

本文档对照 [PipeLink](https://github.com/alumkal/PipeLink)（*Contract-Guided Linking of GPU
Kernel Components*）的能力清单，说明 ForgeKernel 目前**已经实现了哪些**、**还缺少哪些**，
以及每个未完成项为什么没做（需 GPU / 方向不适用 / TileLang 管不到）。

本实现的定位：**一个模拟的 PipeLink 式链接器 + 以 TileLang 为一等融合目标的代码生成**。
它做的是不需要 GPU 的那部分——链接期的所有决策与控制——并把这些决策落成 TileLang 能直接
lower 运行的真实 kernel。

- 代码：`src/forgekernel/link.py`（链接器）、`src/forgekernel/component.py`（组件契约）、
  `src/forgekernel/component_cli.py`（`forge component` 子命令）
- 文档：`components/README.md`（用法）
- 测试：`tests/test_link.py`（50 项 + 5 项 torch 跳过）、`tests/test_component.py`（29 项）

最新提交：`7885ef4`。

---

## 一、已实现

### 1. 组件契约体系（组合别人写的阶段）

每个阶段是一个组件目录，`contract.json` 声明接口（`axes` / `inputs` / `outputs` /
`semantics` / `invariants`），`<language>/verify.json` 自带证明（shape、tolerance、golden）。
可被导入、替换、复用——**按契约而非按名字**。

- `Component` / `read_component` / `read_contract` / `list_components`
- `check_tree`：交付前的静态检查（结构 + 接口 + 资源/角色 + 端口 handle/generation）

### 2. 接缝检查（contract checking）

两阶段只有在 dtype 与「按轴写的 shape 结构」都匹配时才链接；否则以带编号的 `L-*` 诊断拒绝，
而不是生成一个会挂起的 kernel。

- `_seam_for_port` / `_same_port` / `_shape_canon`
- **桥接建议**：接缝不匹配时，若 root 下有组件能夹在中间，`L-PROTO` 直接说「insert X」
  （`_bridge_hint`）

### 3. 句柄检查（tensor handle）

端口可声明 `handle`（`tmem` / `reg` / `smem`）表示张量物理位置。

- 不同 handle **由生成的拷贝桥接**（即 tensor import），不算接缝失败；
- **未知** handle 无法搬运 → `L-HANDLE`（`_handle_for_edge`）

### 4. 阶段代际检查（phase / generation）

端口可声明 `generation`（默认 0）。生产者与消费者代际不一致 = 多代数据同时在途：

- `L-PHASE`，并给出可行环深 `|Δ| + 1`（`_handle_for_edge`）

### 5. 流水线环 + 链接时选环深

- 每阶段可声明 `ring_min` / `ring_max` 与 `producer_rate` / `consumer_rate`；
- 边上的速率不匹配需要 `ceil(prod/cons)` 个在途槽位；
- 求解出的环深 → 直接变成 TileLang 的 `T.Pipelined(num_stages=...)`
  （`_ring_limits` / `_ring_throughput` / `_ring_reason`）

### 6. 资源与角色求解

`[resources]`：`smem_bytes`、`tmem_cols`、`regs_per_thread`、`barrier_slots`；
`[roles]`：`warps`、`ring_min`、`ring_max`、`producer_rate`、`consumer_rate`、`expected_ms`。

- **峰值并发求解**（非简单求和）：阶段从启动驻留到最后一个消费者启动；链式 8×64KB 峰值仅
  128KB，可装 227KB CTA（`_peak_residency`）；
- warp 峰值同法求解；
- 超 caps 时 `L-RES` / `L-ROLE` 诊断**点名峰值持有阶段**（`smem_bytes peak 262144 at {a, b}`）；
- `expected_ms` 汇总为声明预算（标注非实测）（`_expected_ms`）。

### 7. 自动张量交接代码生成（tensor import）

融合源按边的两端 handle 生成**真实搬运**：

- 通用骨架：`acc_tmem_to_reg` / `acc_reg_to_smem` / `acc_identity`（`_handoff_fn`）；
- TileLang：`T.copy` 在 `T.alloc_shared`（smem）与 `T.alloc_fragment`（reg）之间搬运，
  handle 相同则不拷贝。

### 8. 以 TileLang 为一等融合目标（编译到可运行 kernel）

`render_fused` 对 `language == "tilelang"` 生成**真正的 `@T.prim_func`**：

- 流水线的外部 I/O（源阶段未接线的输入、汇阶段未被消费的输出）→ `T.Tensor` 参数，尺寸取自
  各阶段 `verify.json` 的具体整数（无法钉死尺寸报 `L-PROTO`）；
- 阶段按拓扑序运行于 `with T.Kernel(...)`；
- 环深 → `with T.Pipelined(num_stages=ring_depth, order=[...])`；
- 每个接缝 → `T.copy`；
- 菱形 DAG（fan-out + 多输入 join）正确渲染，外部参数只含真正的源与汇。

→ 这条路线下，**代码生成、编译（TileLang JIT）、流水线环、单语言前端、launch** 都由 TileLang
接管（`_render_fused_tilelang` / `_tl_extent` / `_port_handles`）。

### 9. 替换阶段 / 重新链接

`forge component diff <before...> --then <after...>` 重链两条 pipeline，报告阶段/边/资源/环深
变化（「换一个阶段，其余不动」的检查）（`diff_plans`）。

### 10. 纯声明校验 + CPU 语义一致性

- 链接只检查**声明的接口**，不检查声明背后的不透明代码；
- `chained_golden` 把各阶段 `verify.json` 的 golden 按同一接线串成一段 CPU-torch 程序；
- `run_parity` 在 CPU torch 上跑它并与参考比对，证明「A then B = 声明的组合」——无需 GPU；
  缺 golden 报 `parity: "incomplete"`，每阶段读数带输出 shape 便于定位接缝。

### 11. 工具便利（规划 / 可视化 / 机读 / 干跑）

- `forge component plan`：只读规划（含每资源峰值、环深理由、`--graph ascii|mermaid`）；
- `forge component link --emit DIR --format json`：机读输出 + 退出码语义；
- `forge component new --from-run <run>`：从跑完的 run 直接建组件；
- `render_graph`（ASCII / Mermaid）、`summarize`。

### 12. 诊断码

| 码 | 含义 |
|---|---|
| `L-STAGE` | 阶段缺失 / 不可读 / 不是组件 |
| `L-PROTO` | 两阶段接口不一致（dtype/shape/name）；有桥接时点名 |
| `L-WIRE` | 边指向不存在的阶段或端口 |
| `L-HANDLE` | 端口 handle 未知，无法生成搬运 |
| `L-PHASE` | producer/consumer 代际不一致（多代在途需环） |
| `L-RES` | 某资源峰值超 caps（点名持有阶段） |
| `L-ROLE` | warp 峰值超 block，或角色非法 |
| `L-RING` | 请求环深超出阶段允许窗口 |
| `L-ORDER` | 空 pipeline / 有环 / 阶段重复 |

重复诊断会合并（`L-PROTO [2 sites] …`）。

### 对照 PipeLink 能力表

| PipeLink 能力 | ForgeKernel 实现 | 状态 |
|---|---|---|
| 组合别人写的阶段 | `Component` + `link()` | ✅ |
| 契约检查 + 编号诊断 | `_seam_for_port` / `L-PROTO` / 桥接建议 | ✅ |
| tensor import 的句柄语义 | `_handle_for_edge` / `L-HANDLE` | ✅ |
| 多代在途 = 需多槽环 | `L-PHASE`（含所需环深） | ✅ |
| 流水线（mbarrier 环），链接时选环深 | `_ring_limits` → `T.Pipelined(num_stages)` | ✅ |
| 求解 warp / SMEM / TMEM / reg / barrier | `_solve` + `_peak_residency`（峰值） | ✅ |
| 自动生成张量交接（TMEM/reg/SMEM） | `T.copy` + alloc_shared/fragment | ✅ |
| 混用语言（源留在手里，可替换实现） | 多语言槽位 + 按契约替换 | ✅（单语言融合）|
| 前端工具链（CUTLASS/CuTe/TK/TileLang）| 统一为 TileLang 一等目标 | ✅（以 TileLang 替代）|
| 只检查声明、不检查不透明代码 | 定位相同（模拟器） | ✅ |
| CPU 语义一致性证明 | `chained_golden` + `run_parity` | ✅ |
| 生成可 lower 的真实 kernel | TileLang `@T.prim_func` | ✅ |
| 规划 / 可视化 / 机读 / 干跑 | `plan` / `render_graph` / `--format json` / `--from-run` | ✅ |

---

## 二、未实现（及原因）

下表每一项都注明**为什么没做**：需 GPU 才能验证、TileLang 管不到、或走 TileLang 路线后本就不
适用。

### 1. `.plink` 二进制包格式（LTO IR + manifest 的打包 / 发布 / 加载）— 方向不适用

PipeLink 把阶段打包成 `.plink`（LTO IR + manifest），使用者看不到源码也能组合。本实现走的是
**TileLang 源码 + JIT** 路线：阶段是 `kernel_tl.py` + `verify.json`，天然可读、可替换，不需要
二进制封装。因此该格式**在当前路线下不适用**，未做。

### 2. 真实 nvcc / CuTeDSL / nvJitLink 编译与 profile 选择 — 由 TileLang 接管

手写 CUDA / CUTLASS / CuTe 阶段需要 `nvcc` + `nvJitLink`，并选 `sm_90a/100a/103a/120a/121a`
等 profile。本实现的融合目标是 TileLang，由 `@tilelang.jit` 内部完成编译与 profile，**这一段
不再需要手写**——但真正编译并运行仍需一次 GPU + TileLang 环境，当前环境无。

### 3. 主机端 ABI（provider 库 / C 头 / persistent 会话 / CUDA graph 捕获）— 部分，需 GPU

T5 的「不带源码发布 + 主机侧初始化 + persistent + CUDA graph」：TileLang JIT 已接管普通
launch（省去手写 trampoline），但 **persistent 常驻会话、CUDA graph 捕获**仍要主机侧代码
（如 `torch.cuda.CUDAGraph`），且**需 GPU 才能验证**，故未做。

### 4. 跨 GPU 通信重叠（T7）— TileLang 管不到

把别的团队的通信阶段链进计算 kernel，使 NVLink 传输与计算在单 kernel 内重叠：需要就绪计数器、
按通信域划分的对端内存 / 多播内存、NCCL 等。**TileLang 面向单 kernel 单 CTA 内计算，不覆盖
跨 GPU 通信**，这块仍属手写 CUDA/NCCL 领域，未做。

### 5. 多代数据同时在途的真实多槽环运行时 — 部分，需 GPU

`L-PHASE` 已建模「代际不一致 = 需要多槽环」并算出所需环深、映射到 `T.Pipelined`；但**真正的
多槽环形缓冲语义**（每个 generation 一个槽位的运行时行为）需 GPU 运行验证，当前只能静态生成。

### 6. 真实资源分配器（分配 / 复用图，而非峰值上界）— 由 TileLang 接管

本实现用「峰值上界比较」而非真正的分配器。走 TileLang 后，**SMEM/寄存器分配由 TileLang
自动完成**，不再需要手写分配器；但要精确读数（每阶段实际占用的 SMEM/TMEM）需查看编译产物，
**需 GPU**。

### 7. 前端工具链（CUTLASS/CuTe、ThunderKittens、打过补丁的 TileLang 后端）— 由单语言替代

PipeLink 支持多种前端并把它们链接进同一个 kernel。本实现**统一为一个前端（TileLang）**，
从而不需要 CUTLASS/CuTe/TK 解析与 bridge。真正的「跨语言交叉链接」与「打过补丁的 TileLang
后端」未做——但单语言路线下并非必需。

### 8. 性能实测与 parity 证据框架（evaluator）— 需 GPU

PipeLink 用 evaluator 在硬件上测吞吐、跑 memcheck/racecheck/synccheck 并生成证据。本实现只做
**CPU 语义模拟**（证明组合正确），**不在硬件上测速**。TileLang 的 `do_bench`/`Profiler` 能做，
但**必须有 GPU 运行**，当前环境无（本机无 torch、GPU 机失联）。

---

## 三、一句话总结

**链接的决策层与控制层已基本做完**（契约、接缝、句柄/代际、资源/环深求解、张量交接代码生成、
TileLang 真实 kernel 生成、CPU 语义证明、诊断与可视化）。**剩下的是执行层**：

- 需 GPU 才能验证的：性能实测、多槽环运行时、persistent/CUDA graph；
- TileLang 管不到的：跨 GPU 通信重叠；
- 走 TileLang 路线后不适用的：`.plink` 二进制包、多前端工具链、手写资源分配器（均由 TileLang
  或源码+JIT 替代）。

在当前无 GPU 环境下，唯一还能在不依赖 GPU 前提下推进的方向是**跨 GPU 通信的声明级建模**
（在契约里声明通信域/就绪计数器，链接时校验跨阶段就绪是否匹配）——但仍无法真正运行 NVLink。
