# CuTeDSL cluster / CTA-pair 骨架（最小可运行）

> 三份从 `hw_kb/cutedsl/` 的 recipe 提炼的最小骨架，用于在支持 CuTeDSL 的机器上
> **直接跑通再逐项填内核**。它们的目的是**先把「静默 hang」这类陷阱暴露出来**，
> 再让你换自己的形状，而不是让你照抄成生产 kernel。

## 环境要求

| 项 | 值 |
|---|---|
| CuTeDSL | `cutlass` Python 包（本机验证版本 **4.8.0**）|
| 已验证所在 | `lmsysorg/sglang:dev` 镜像内（宿主与 `~/tlvenv` **都未安装**）|
| 架构 | multicast GEMM → **sm90a+**；CTA-pair load / 2CTA MMA → **sm100a（数据中心 Blackwell）** |

本仓库的 **RTX 6000D 是 sm120**，它的 MMA 路径是 `HMMA`（见
`hw_kb/tilelang/sm120/index.md`），**没有 tcgen05/wgmma**。因此：

| 骨架 | 能否在 6000D 上跑 |
|---|---|
| `tma_multicast_gemm.py` | ⚠️ 构建可跑，**多播行为**只在真 cluster 生效时才有意义 |
| `cta_pair_bulk_load.py` | ⚠️ 同上，构造可见；实际 pair 需要 cluster 双 CTA |
| `tcgen05_2cta_mma.py` | ❌ **不能**，脚本内 `assert_eligible` 会直接拒绝 sm120 |
| `multimem_tp_allreduce.py` | ❌ **不能**，本机无 NVLink，无 multicast memory 对象 |

## 怎么跑

```bash
docker run --rm -it --gpus all --network=host \
  -v "$PWD":/w -w /w lmsysorg/sglang:dev bash

python3 -c "import cutlass; print(cutlass.__version__)"   # 4.8.0
python3 kernels/_cutedsl_cluster/tma_multicast_gemm.py
python3 kernels/_cutedsl_cluster/cta_pair_bulk_load.py
python3 kernels/_cutedsl_cluster/tcgen05_2cta_mma.py      # 在 sm120 上会被 gate 拒绝
```

## 四个文件各自暴露的核心陷阱

### 1. `tma_multicast_gemm.py` — 多播要两个旋钮同时动
- `cluster=[N,1,1]` **和** atom 的 `num_multicast` 必须相等，缺一则**静默无效**
- 必须用 `CopyBulkTensorTileG2SMulticastOp()`，不是 `CopyBulkTensorTileG2SOp()`
- 首个多播前要 `mbarrier_init_fence()` + `cluster_arrive_relaxed()` + `cluster_wait()`

### 2. `cta_pair_bulk_load.py` — 漏写 `.cta_group::2` 是静默 hang
- `CopyBulkTensorTileG2SOp(CtaGroup.TWO)` 与默认 `CtaGroup.ONE` 的 copy 形状/描述符/字节数**完全相同**
- 所以漏写**不报错、不编译失败、数字也对**，只是 barrier 永远不填 → hang
- `.cta_group::1` 下把完成信号重定向到 peer 是 **UB**

### 3. `tcgen05_2cta_mma.py` — 形状五连约束 + 第二次启动 hang
- smem 必须是 `allocate_tensor(layout=outer, swizzle=inner)`，**不能**直接用 composite layout
- `cute.gemm` 要吃 `make_fragment_A/B` 的 `(MMA, MMA_MN, MMA_K, STAGE)`，不是 (mn,k) 面
- 累加器在 TMEM，需 `TmemAllocator(..., two_cta_tmem_dealloc_mbar_ptr=...)`
- **缺 teardown → 同一 entry 第二次启动永不返回**（`tail()` + `tmem.free`）
- 手写 pair commit 必须 `tcgen05.commit(mbar_ptr, mask_self | mask_peer, CtaGroup.TWO)`

### 4. `multimem_tp_allreduce.py` — 真正跨 GPU 的 kernel 内归约（需 NVLink）
- `cutlass.utils.distributed` 提供 15 个 `multimem.ld_reduce` + 3 个 `multimem.st`，**全在 kernel 内**
- 这意味着「GEMM epilogue 直接完成跨卡 all-reduce」，无需 NCCL kernel、无 HBM 中转
- **只有 add**：本模块没有 min/max/mul 的 multicast reduce
- **无架构守卫**：扫不到任何 `arch/capability` 判断，无 NVLink 时不会提前报错
- 前提是多播内存对象（host 侧注册），普通 cudaMalloc 指针不是

## 诚实说明

1. 这四份是**骨架**：构建 atom / 断言陷阱的部分可直接跑，**MMA 主循环留白**，
   因为原 recipe 明确只覆盖 mechanism，不含完整 operator。
2. 真正要在 6000D 上要收益的路径不是这些（6000D 无 tcgen05、TileLang 的
   cluster multicast 是坏的）；这些骨架的价值在 **sm100 数据中心卡**上。
3. 每个文件末尾的 `todo:` 行就是按 recipe 顺序排好的填空步骤。
4. `multimem_tp_allreduce.py` 只验证了 **API 表面**（26 个符号全部存在、只有 add、
   无架构守卫），**机制未在硬件上验证**。

## 对应 recipe

| 骨架 | recipe |
|---|---|
| `tma_multicast_gemm.py` | `hw_kb/cutedsl/sm90/tma_multicast_cluster.md` |
| `cta_pair_bulk_load.py` | `hw_kb/cutedsl/sm100/cta_pair_bulk_load_cta_group2.md` |
| `tcgen05_2cta_mma.py` | `hw_kb/cutedsl/sm100/tcgen05_umma_issue_cta_pair.md` |
| `multimem_tp_allreduce.py` | `hw_kb/cutedsl/sm100/multimem_ld_reduce_and_store.md` |