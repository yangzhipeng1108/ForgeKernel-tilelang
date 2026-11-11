# TileLang kernel 如何设计才能更好被通信重叠

> 面向「双卡 / 多卡训练与推理」场景的 kernel 设计准则。
> 所有数字来自本仓库 `hw_kb/tilelang/sm120/` 的 RTX 6000D 实测（驱动 610.43.02，CUDA 13.3，
> TileLang 0.1.15），卡型拓扑与跨卡带宽见同目录 `RTX6000D_双卡优化总纲.md`。
>
> **前提**：TileLang 不写通信 kernel、也没有多 GPU API。它只决定三件事——
> **占多少 SM、占多少 HBM、什么时候能被打断**——而这三件事决定了外部的
> NCCL 集合通信能不能挤进来并被重叠掉。

## 0. 一句话结论

**通信能不能被藏住，取决于你的 kernel 有没有给通信 kernel 留下 SM 和 HBM；
kernel 占得越死（persistent、满 SM、满带宽），重叠越难；颗粒度越细、峰值越低，重叠越容易。
本卡实测所有「调度技巧」对单 kernel 性能都是 tie，但对**可重叠性**并非中立——
persistent 省 L2 命中的同时也把 SM 占死。**

## 1. 为什么要谈这件事（物理背景）

通信与计算争抢的是同一批资源：

| 资源 | 通信侧占用 | 计算侧占用 | 冲突后果 |
|---|---|---|---|
| **SM** | 节点内 NCCL ring/tree reduce 跑在 SM 上 | GEMM/Attention 占 SM | 通信 kernel 排队，重叠失败 |
| **HBM / L2** | 消息打包、中转 staging | 权重/激活读取 | 计算有效带宽下降 |
| **寄存器** | 少 | GEMM 240 regs → 1 block/SM | 通信 kernel 无空槽可插 |

跨节点 IB/RoCE 走 GPUDirect RDMA，**不占 SM**；真正的 SM 争抢来自**节点内**那一段。
双卡机（本机）跨卡 = PCIe 45 GB/s，两段都在 GPU 上，冲突最直接。

## 2. 准则一：控制 SM 占用（最重要）

### 2.1 不要用 persistent 把 SM 占死
`block_scheduling_is_a_tie.md` 实测：persistent 把 1024 个 block 折叠到 **156 个常驻 worker**
（`tilelang.carver.arch.driver.get_num_sms()`），性能与 raster/swizzle 相同（±2%，因为
`dram_pct ≈ 13%`，L2 顺序够不到关键路径）。

| 调度 | 单 kernel 性能 | 对通信重叠 |
|---|---|---|
| raster（默认） | 同 | **好**——block 逐个完成，有空窗 |
| `T.use_swizzle` | 同（±2%）| 好 |
| `T.PersistentTileScheduler` | 同（±2%）| **差**——worker 循环到 kernel 结束才释放 SM |

**准则**：**推理/训练的并行主 kernel 不要用 persistent**，除非单 kernel 有实测收益。
本卡上 persistent 从来没有单 kernel 收益，却会独占 SM。

### 2.2 寄存器预算决定每 SM 的 block 数
`flash_attention_is_compute_bound_too.md` 实测：**240 regs/thread → 1 block/SM → warps_active 仅 15.8%
→ `stall_wait` 22.4%**。此时 SM 只剩一个 block 在跑，通信 kernel 完全没有插入机会。

| 手段 | 效果 | 注意 |
|---|---|---|
| 降 tile 尺寸 / 降 stages → 降寄存器 | 每 SM 可容纳多 block | 可能损单 kernel 吞吐 |
| `launch_bounds` 限制寄存器 | 提高 occupancy | FA 实测**强行降 reg 更慢**（240→160→128 得 268→211→179 TF）|

**准则**：**不要为重叠去强行压寄存器**——本卡 FA 实测压了就掉算力。
正确的做法是**选一个寄存器刚好不撑满 SM 的 tile**，让至少留 1 个空 block 槽。

### 2.3 控制 `threads=` 与 launch_bounds
`gemm_operand_layout_and_policy.md`：emitted `launch_bounds = (256, 1)`，`threads=128` 加了
producer warpgroup 变 256。`software_pipeline_stage_budget.md`：`stages=0` 是 128，`stages≥2` 变 256。

**准则**：`threads` 越大，单 block 占 SM 越多。**并行主 kernel 用 128–256，不要盲目上 512+**。

## 3. 准则二：压低 HBM / L2 峰值（给通信腾带宽）

### 3.1 让 Tile 读取量最小
- `tiled_copy_scope_and_swizzle.md`：纯 copy kernel 已达 `dram_pct = 93%`，**带宽就是它的上限**；
  此时任何 swizzle/布局都不起作用，因为瓶颈在 DRAM 不在 SMEM 端口。
- 结论：**当 kernel 本身 dram_pct 已 >90%，它就不可能被通信重叠**——带宽已经饱和，
  通信打包会直接拖慢它。

**准则**：通信频繁的并行阶段，**优先选 dram_pct 低的实现**（本卡 GEMM 是 13%，FA 是 8.8%），
而不是选单 kernel 最快的实现。

### 3.2 复用 shared tile，减少重复读
`gemm_operand_layout_and_policy.md`：`Square` 策略的 `smem_wavefronts_ld = 3.36e7`，
比 FullRow/FullCol **少读 25% 的 shared memory**。

**准则**：**保持 `T.gemm` 默认 `GemmWarpPolicy.Square`**——它既是最快的（133.3 TFLOP/s），
也是 shared 读取最省的，间接给 HBM 留余量。

### 3.3 减少 elementwise 往返
`memory_bound_elementwise_saturates_hbm.md`：RMSNorm 1331 GB/s、RoPE 1286 GB/s，
**已超过 1194 GB/s 的 copy roofline**（归约藏在流里）。融合的 TileLang kernel 是
**eager torch 的 10 倍**（torch 132 GB/s）。

**准则**：**把 norm/RoPE 融合进相邻 kernel**，不要单独发 elementwise kernel——
单独的 elementwise 会白白吃掉本可留给通信的带宽。

## 4. 准则三：分块粒度（决定能否被插入）

通信只能插在**同步点之间**。kernel 越粗，可插入点越少。

| 粒度 | 可重叠性 | 代价 |
|---|---|---|
| 单一大 kernel 跑完整层 | 差（只在层边界）| launch 开销低 |
| 按 tile/段拆分（多 kernel 或 persistent 外的分段）| 好 | launch 开销 |
| TP 的 GEMM 拆成子块输出 | 好——子块完成即可 all-reduce | 需框架支持（Megatron `--tp-comm-overlap`）|

**准则**：
- **训练**：让 GEMM 输出按 bucket 边界分块，梯度一算完就发——这是 DDP 分桶重叠
  （本仓库 `scripts/train_dp.py` 验证过 1.87–1.90×）。
- **推理 MoE**：`dispatch → 专家 GEMM → combine` 三步之间就是天然插入点，
  kernel 按专家分组（grouped GEMM）能让边界更整齐。
- **不要**为了重叠把 kernel 拆得过碎——本卡 launch 约 6 µs（`gpu_probe_ladder.md`），
  小 kernel 会被 host 端开销吃掉。

## 5. 准则四：layout / stride 对齐（影响通信是否零拷贝）

通信库对输入 tensor 的 stride 有要求。kernel 输出若与之对齐，NCCL 可直接发；否则要重排
（多一个 kernel + 一次 HBM 往返，正好抵消重叠收益）。

| 要求 | 说明 |
|---|---|
| 输出连续（contiguous） | 避免 `transpose_B=True` 产生非连续输出后又要 copy |
| 按 bucket/专家边界对齐 | MoE dispatch 的 token 分桶要对齐到 tile |
| 避免 `T.copy` 产生 padded 布局 | `tiled_copy_scope_and_swizzle.md`：padded tile 会被 TMA bulk 拒绝 |

**准则**：**给通信消费的 tensor 用连续、无 padding 的 layout**；swizzle 只在 SMEM 内部做，
不要泄漏到 global 输出的 layout 上。

## 6. 本卡实测的「反向清单」（别为重叠做的事）

| 做法 | 实测结论 | 出处 |
|---|---|---|
| 用 persistent 换重叠 | 单 kernel 无收益，却独占 SM | `block_scheduling_is_a_tie.md` |
| 强压寄存器提 occupancy | FA 240→160→128 得 268→211→179 TF，**更慢** | `flash_attention_is_compute_bound_too.md` |
| 加 swizzle/persistent 调度 | 24 个 cell 全部 ±2% tie | `block_scheduling_is_a_tie.md` |
| 上 warp specialization 求重叠 | `role_based` 每个 cell **慢 2–8×** | `warp_specialization_is_already_on.md` |
| 用 cluster multicast 省带宽 | TileLang 的 `T.copy_cluster` **慢约 6×10⁴**（include 缺失）| `clusters_free_multicast_broken.md` |
| 用 CLC 动态调度 | 编译成 **0 条指令**，且与活 barrier 混用会**破坏输出** | `clc_compiles_to_nothing.md` |

## 7. 判据流程（拿来即用）

```
1. 这个 kernel 会被放在通信密集的并行阶段吗？
   ├─ 否 → 只按单 kernel 性能调，不用管本文
   └─ 是 → 继续
2. 它的 dram_pct 是多少？（NCU）
   ├─ >90% → 带宽已饱和，无法被重叠；改用低带宽实现
   └─ <30% → 有重叠空间，继续
3. 每 SM 有几个 block resident？
   ├─ 1 且 warps_active <20% → 无空槽，通信插不进来；降 tile/stages
   └─ ≥2 → 继续
4. 用 persistent 了吗？
   ├─ 是 → 换回 raster（本卡无收益却有代价）
   └─ 否 → 继续
5. 输出 layout 对通信零拷贝友好吗？
   ├─ 否 → 改成连续/对齐
   └─ 是 → 完成
```

## 8. 与 CuTeDSL 的差距（能力边界）

本文全部准则在 TileLang 里都只能做「减法」——降占用、降带宽、调粒度。
若要主动优化单 GPU 内集群资源（真正给通信腾出更多余量），需要 CuTeDSL 的能力：

| 能力 | TileLang | CuTeDSL | 对通信重叠的作用 |
|---|---|---|---|
| TMA multicast（HBM 读量降 1/N）| ❌ 坏的 | ✅ | 直接省带宽给通信 |
| `.cta_group::2` 两 CTA 共享 barrier | ❌ | ✅ | 省同步资源 |
| 2CTA UMMA（tcgen05）| ❌ | ✅ | 提高单 SM 算力密度 |
| DSMEM 跨 CTA 交换 | ❌ | ✅ | 免 HBM 中转 |
| CLC 动态调度 | ❌ 0 指令 | ✅ | 动态让出 SM |

详见 `porting/6000D/RTX6000D_双卡优化总纲.md` 与 `hw_kb/cutedsl/`。

## 9. 底线

1. **跨卡通信本身 TileLang 管不了**，重叠是外部 NCCL + 框架的事。
2. **能做的只有让 kernel 更「可被插空」**：不占死 SM（别 persistent、保 occupancy）、
   不占满带宽（低 dram_pct、融合 elementwise）、边界清晰（分块粒度）、
   输出零拷贝（连续 layout）。
3. **本卡上所有「调度技巧」对单 kernel 都是 tie**，所以按本文做减法**没有单 kernel 性能代价**——
   这是唯一能放心执行的部分。
4. **主动优化需要 CuTeDSL**，TileLang 在集群协作层几乎全缺。
