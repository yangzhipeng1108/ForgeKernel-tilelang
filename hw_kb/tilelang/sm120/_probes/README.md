# sm120 recipe 探针原始记录（probe archive）

> 从 GPU 机器（RTX 6000D）`/tmp` 抢救回来的原始探针脚本与测量产物。
> 打包时间 `2026-11-11 02:47 UTC`，来源 `/tmp/fk_probes_20261111_0247.tar.gz`
> （md5 `ab352bd1749ade9f6f4676ea9ffde252`，213 文件，1.3 MB）。
>
> 这些是写 23 份 recipe 时**当次在机器上临时写的内核**，此前从未入库；`/tmp` 会被系统清理，
> 故按原样归档，**不做改写**。文件名是当时随手起的过程名（`fa1..fa8`、`sk1..sk10`……），
> 与 recipe 的对应见下表。

## 统计

| 类别 | 数量 |
|---|---|
| `.py` 探针 | 178（其中 **150 引用 tilelang**，133 含 `@T.prim_func`/`T.Kernel`）|
| `.cu` 手写/导出源码 | 11 |
| `.sass` / `.cubin` | 3 / 3 |
| `.ncu-repz` NCU 报告 | 2 |
| `.txt` / `.log` | 8 / 5 |

不用 tilelang 的 28 个是双卡通信探针（NCCL / P2P / DDP / PP），对应 `porting/6000D/` 那几份文档。

## recipe ↔ probe 映射（按文件名）

| recipe | probe 脚本 |
|---|---|
| `software_pipeline_stage_budget.md` | `p0.py p0b.py p0c.py p1a.py p1b.py p1c.py p2.py p3.py` |
| `gemm_operand_layout_and_policy.md` | `g1.py–g4.py gp.py` |
| `tiled_copy_scope_and_swizzle.md` | `sw.py swz.py findsw.py k_swz8.cu` |
| `block_scheduling_is_a_tie.md` | `bs.py bs2.py pers.py pers2.py pers3.py k_pers8.cu k_raster.cu` |
| `split_k_fills_a_starved_grid.md` | `sk.py sk2.py–sk10.py sp_k.cu` |
| `stream_k_is_absent_and_loses.md` | `streamk*`（若存在）|
| `clc_compiles_to_nothing.md` | `clc.py clc1.py clc2.py clc3.py clc*.sass clc*.cubin clc_k*.cu` |
| `clusters_free_multicast_broken.md` | `cl1.py–cl6.py` |
| `sparse_gemm_two_mechanisms.md` | `sp.py sp1.py–sp6.py` |
| `flash_attention_is_compute_bound_too.md` | `fa1.py–fa8.py fa8c.py fag.py fa_prof.py fareg.py fa_sk.py fa_sp.py faws.py` |
| `causal_attention_skips_the_masked_half.md` | `ca1.py ca2.py` |
| `decode_is_the_one_bandwidth_bound_case.md` | `mha.py mha2.py–mha7.py` |
| `mla_absorbed_decode_beats_mha.md` | `mla0.py–mla5.py` |
| `fp8_qk_ports_split_k_ties.md` | `q.py q2.py qa.py fp16grad.py` |
| `head_dim_decides_the_stage_budget.md` | `dt.py dt2.py dt_sweep.py dttile.py dtprobe.py` |
| `lm_head_is_wide_not_starved.md` | `lm1.py lm2.py` |
| `memory_bound_elementwise_saturates_hbm.md` | `n1.py–n5.py` |
| `warp_specialization_is_already_on.md` | `ws.py ws1.py–ws7.py` |
| `gpu_probe_ladder.md` | `probe probe2 probe3 probe.cu probe3.cu` |
| 双卡（`porting/6000D/`）| `dual_gpu_bench.py nccl_*.py dp*.py pp_*.py p2p*.py ddp_*.py f1b1*.py tp_model*.py` |

## NCU 报告

- `rep_867.ncu-repz`、`rep_23011.ncu-repz` —— Nsight Compute 原始报告（压缩）
- `ncuout.txt`、`ncu_case.py`、`ncucase.py` —— NCU 采集脚本与输出

## 本轮复跑验证（2026-11-11）

两个环境都跑通 `scripts/measure_kb_entries.py --arch sm120`，4/4 PASS：

| 环境 | tilelang | 结果 | 耗时 |
|---|---|---|---|
| 宿主 `~/tlvenv`（**与 KB 对齐**）| **0.1.15** | 4/4 PASS, `max_diff=0` | 7 s |
| 镜像 `lmsysorg/sglang:dev` | 0.1.12 | 4/4 PASS, `max_diff=0` | 34 s |

报告：`_measure_tl015.txt`（0.1.15）、`_measure_run.txt`（0.1.12），均在本目录。

> 结论：`pipeline` / `copy_swizzle` / `gemm_policy` / `probe_harness` 四条在真机可复现。
> 其余 recipe 的探针脚本已归档于此，但**没有统一的复跑入口**，需按上表逐个手动运行。
