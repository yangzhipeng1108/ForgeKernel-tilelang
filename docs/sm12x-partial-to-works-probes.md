# SM12X PARTIAL → WORKS 探针配方（在 RTX 6000D 上把 5 份借用条目升级）

本文给出在 **真 RTX 6000D（SM120，PCIe-only，无 NVLink）** 上验证
`hw_kb/tilelang/sm120/sm120_pcie_collectives.md` 与 `sm120_serving_decode_levers.md` 里
**5 项 `State: PARTIAL`** 的确切命令、观测值、判定门槛与升级步骤。这些数字目前来自一份 8×6000D
部署报告（vLLM），**不是本仓库探针实测**；跑完本文的探针、把每项判成 WORKS / BROKEN / ABSENT
后，才算"sm12X 优化真验证完成"。

> 本文是**程序**（procedure），不是 recipe，故放 `docs/` 而非 `hw_kb/`（不进学生读集）。
> 已归档的探针脚本在 `hw_kb/tilelang/sm120/_probes/fk_probes_20261111_0247/`（`nccl_bw.py`、
> `p2p_ceiling.py`、`mha*.py` 等）。这些脚本是**无 argparse 的 torchrun 脚本**（尺寸/迭代写死），
> 下面给的是它们**原样的**启动方式；需要改尺寸时就地改常量或按下文"自包含变体"跑。

## 共同前置（每次探针都先做）

```bash
# 1) 确认是 6000D / SM120，且看到至少 2 张卡
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
#   期望：RTX 6000D / 12.0 / ~85651 MiB，≥2 行

# 2) 锁 SM 时钟（不锁则空载 ~200MHz、boost 2430，任何计时都不可信）
sudo nvidia-smi -i 0 -lgc 2000,2000
sudo nvidia-smi -i 1 -lgc 2000,2000

# 3) 工具链
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
python3 -c "import torch; print(torch.cuda.device_count(), torch.version.cuda)"
```

---

## 借 1：`NCCL_P2P_LEVEL=SYS` 小消息延迟（`sm120_pcie_collectives.md` Finding 3）

**探针（用归档脚本原样）**：`nccl_bw.py` 是 `torchrun` 脚本，内含 1/8/32/128 MB 三 op all-reduce
扫描，直接跑两次——默认 vs `SYS`：

```bash
cd hw_kb/tilelang/sm120/_probes/fk_probes_20261111_0247
torchrun --nproc_per_node=2 nccl_bw.py                      # 默认 P2P 级别
NCCL_P2P_LEVEL=SYS torchrun --nproc_per_node=2 nccl_bw.py   # 强制 SYS
```

**判据**：对比两次输出里 **1MB allreduce 的 us 值**。`SYS` 更小（延迟下降）→ **WORKS**（本机实测），
否则 **BROKEN**（记录 GPU/驱动/NCCL 版本，说明本机 `SYS` 不适用）。报告只给单流 +18%，这里的延迟
下降是同性质信号。

---

## 借 3：P2P 延迟 / 带宽判据（`sm120_pcie_collectives.md` Finding 4）

**探针（自包含，4KB 延迟 + 16MB 带宽，单进程双卡拷贝）**：4KB 看**每跳往返**，16MB 看**带宽**。
归档 `p2p_ceiling.py` 只测了 64MB 单档，这里给两档的变体（可直接跑）：

```bash
cat > /tmp/fk_p2p_probe.py <<'PY'
import torch, time
torch.cuda.set_device(0)
def probe(nbytes, iters):
    a = torch.empty(nbytes, dtype=torch.uint8, device="cuda:0")
    b = torch.empty(nbytes, dtype=torch.uint8, device="cuda:1")   # peer copy, no NCCL
    for _ in range(5): b.copy_(a)                                  # warmup
    torch.cuda.synchronize(); s = time.perf_counter()
    for _ in range(iters): b.copy_(a)
    torch.cuda.synchronize()
    us = (time.perf_counter() - s) / iters * 1e6
    gbs = nbytes / (us * 1e-6) / 1e9
    print(f"{nbytes:>10} B  {us:8.2f} us  {gbs:6.1f} GB/s")
print("== 4KB (latency) / 16MB (bandwidth) ==")
probe(4096, 2000)
probe(16 << 20, 200)
PY
python3 /tmp/fk_p2p_probe.py
```

**判据**：4KB ≈ 9–10 µs（每跳往返延迟）× 16MB ≈ 50 GB/s（带宽）都复现 → 借 3 判据 **WORKS**。
若 4KB 往返远高或 16MB 带宽远低 → 记录实测值（拓扑/NUMA 差异），仍可作为判据探针（**WORKS** 但写本机数）。
若 peer copy 直接报错（无 P2P）→ **ABSENT**（写"两卡经 host 中转，无直连 P2P"）。

---

## 借 2 + 借 4：默认后端≠最优 + 报文大小选后端（`sm120_pcie_collectives.md` Finding 1/2）

**探针 A（NCCL 报文扫描，逐档转折点）**：用归档 `nccl_bw.py` 的 1–128MB 扫描做基线（见借 1 的
命令，取默认那次输出）。

**探针 B（FP8 线/B12X 对比，若装了该库）**：`B12X_PCIE_DMA_FP8=ring` 在 64/128MB 档是否快于 NCCL：
```bash
B12X_PCIE_DMA_FP8=ring torchrun --nproc_per_node=2 nccl_bw.py   # 仅当 b12x 库可用
```

**判据**：
- 报文扫描显示"小消息某后端 > NCCL、大消息另一后端 > NCCL、128MB 附近回退/持平"的**转折形状** →
  借 4 三段式选路 **WORKS**（把本机实测转折点写进条目，替换报告数字）；
- 本机无 B12X / 无多播 IPC → Findings 2 的 FP8 列与 IPC 列标 **ABSENT**，只保留 NCCL 实测列；
- 若某"默认/花哨后端"在本机明显慢于朴素 NCCL（同报告 Cake 现象）→ 借 2 **WORKS**（写清是哪一个
  后端、慢多少、怎么切回 NCCL）。

---

## 借 6：CUDA Graph 对 decode 的倍率（`sm120_serving_decode_levers.md` Finding 2）

**探针**：一个 decode 形状 kernel 分别 eager 与 CUDA-graph replay 计时。归档里有对应形状脚本
（`mha.py` 等，q=16/KV=8192/D=128，见 `decode_is_the_one_bandwidth_bound_case.md`）。用下述自包含
骨架套在任意 `fn(*inputs)` 上测 graph 倍率：

```bash
cat > /tmp/fk_graph_probe.py <<'PY'
import torch, time
# fn(*args) = the decode kernel call you want to test (swap in mha.py's build()+inputs)
def measure(fn, args, iters=500):
    for _ in range(10): fn(*args)                       # warmup (compile outside capture)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn(*args)                 # capture
    torch.cuda.synchronize()
    s = time.perf_counter()
    for _ in range(iters): g.replay()                   # graph replay
    torch.cuda.synchronize()
    return iters / (time.perf_counter() - s)            # steps/s (graph)
# eager baseline: same loop without the graph, time it the same way
def measure_eager(fn, args, iters=500):
    for _ in range(10): fn(*args)
    torch.cuda.synchronize(); s = time.perf_counter()
    for _ in range(iters): fn(*args)
    torch.cuda.synchronize()
    return iters / (time.perf_counter() - s)
# usage: fn, args = <from mha.py>; print(measure_eager(fn,args), measure(fn,args))
PY
```

**判据**：graph 的 steps/s 明显高于 eager（报告 29.23 vs 10.97 ≈ 2.7×）→ 借 6 **WORKS**（写本机倍率）。
若 capture 报错（该 kernel 不能 capture）→ 这本身就是 `cuda_graph` 门禁的证据：条目写"该 kernel 在本机
不可 capture → 不可部署"（仍 **WORKS** 作为门禁事实，如实记 capture 异常类型）。

---

## 借 5-fact：前缀缓存 TTFT 倍率（`sm120_serving_decode_levers.md` Finding 1）

**探针（需 served-model 环境：vLLM/sglang 镜像 + 一个长上下文模型）**：同一长上下文 20 并发，
开/关前缀缓存各测 TTFT 中位。

```bash
# 关闭前缀缓存
vllm serve <model> --disable-prefix-caching --max-model-len 524288 --port 8000 ... &
python -m vllm.benchmarks.benchmark_serving --backend openai \
  --base-url http://127.0.0.1:8000 --model <model> \
  --dataset-name random --random-input-len 524288 --random-output-len 128 --num-prompts 8
# 开启前缀缓存（默认即开；显式确认后重跑同一 benchmark）
vllm serve <model> --enable-prefix-caching --max-model-len 524288 --port 8000 ... &
# 对比两次 TTFT 中位
```

**判据**：开缓存后长上下文 TTFT 中位下降数十倍（报告 ~239s → ~4.4s @512K）→ 借 5 的"记录
prefix_cache/context_len、拒跨口径比较"补全硬件数字，**WORKS**。
> 本项要**真模型 + 服务镜像**，与 kernel 探针不同。无模型可跑时保留 PARTIAL，注明"需 served-model 环境"。

---

## 升级 PARTIAL → WORKS 的机械步骤

每跑完一项：

1. 在对应 `hw_kb/tilelang/sm120/*.md` 顶部把 `> State: PARTIAL — …报告数字…` 改成
   `> State: WORKS — measured on this box (RTX 6000D, clock-locked): <本机实测关键数>`；
2. 加 `## Measured on` 小节（**改成 WORKS 后 kb_lint 要求存在**）：TileLang/框架版本、GPU 名与算力、
   `SM clock locked at 2000 MHz`、卡数、驱动/CUDA/NCCL 版本；
3. 实验日期/run 号写进 `docs/kb-provenance/<entry>.md`（**不能**写 hw_kb，narrative 规则禁）；
4. 用本机实测数字替换正文里"报告给的"数字（转折点、+18%、2.7×），保留一行"与部署报告一致/不一致"对照；
5. `python3 -m forgekernel.loop.kb_lint hw_kb` 必须 **0 findings**；
6. `index.md` 对应行 `⚠️` 改成 `✅`（与条目 State 一致，`kb_state` 会校验）。

## 判成 BROKEN / ABSENT 时

- 本机某后端不可用（无 B12X / 无多播 IPC / 无 P2P）→ **ABSENT**，正文写"本机不可用，改用 X"；
- 数字与报告严重不符（`SYS` 反而变慢等）→ **BROKEN**，写清本机条件下的正确做法。

## 一句话

**5 项探针全部有真实入口**：借 1/2/4 用归档 `nccl_bw.py`（torchrun 原样跑，加 `NCCL_P2P_LEVEL` /
`B12X_PCIE_DMA_FP8` env）；借 3/6 用文内自包含变体；借 5 用 vLLM benchmark。跑完按 6 步把 `> State:`
升到 WORKS（或据实 BROKEN/ABSENT）、补 `## Measured on`、替换报告数字、kb_lint 归零。在此之前两份
sm12X 条目诚实保持 PARTIAL——代码侧（借 5 的拒绝逻辑）已完成且单测通过，缺的只是这 5 项真硬件读数。
