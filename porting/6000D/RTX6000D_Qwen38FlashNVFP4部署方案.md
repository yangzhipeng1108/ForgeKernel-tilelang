# Qwen3.8-Flash-Next-NVFP4 双卡 RTX 6000D 最优部署方案

> 本机实测定稿。全部数字在 2× RTX 6000D（无 NVLink，GPU0↔GPU1 = SYS）上测得，
> 两卡 SM 锁 2000 MHz，`NCCL_P2P_LEVEL=SYS`。
> 镜像 `lmsysorg/sglang:dev`（sglang `0.0.0.dev1+g38d9e4a3f`，torch 2.14.1+cu130，flashinfer 0.7.2）。
> 硬件基线与跨卡带宽结论见同目录 `RTX6000D_双卡优化总纲.md`。

## 0. 一句话结论

**`tp2 + ep2 + flashinfer_cutlass` 是本模型在双卡上的唯一可行最优并行组合；
`mamba-track-interval=128`、`max-running-requests=48`、`page-size=32` 三项参数
把 C=16 输出吞吐从原始 522 tok/s 提到 730 tok/s（+40%），TTFT 从 24.4s 降到 6.4s。
再往上受限于无 NVLink 的 45 GB/s 跨卡带宽，算子层已到顶。**

## 1. 模型固有约束（决定参数，不可绕）

| 项 | 值 | 部署含义 |
|---|---|---|
| 架构 | `Qwen4ExpForConditionalGeneration`（`qwen4_exp.py`） | 需 `--model-path` 直接指向目录 |
| 层结构 | 48 层 = 36 linear_attention + 12 full_attention（`full_attention_interval=4`） | 线性注意力 kernel 是关键路径 |
| hidden / heads | 2560 / 24 q-heads，**2 KV heads**，head_dim 256 | GQA 极窄，KV 缓存省 |
| MoE | 512 专家 × 每 token 10，moe_inter 640，shared 640 | EP2 每卡 256 专家 |
| vocab | 248320 | embedding 大，需 `--ple-offload-embedding` |
| 量化 | `modelopt` NVFP4（含 `flashinfer-jit-cache-sm120f`） | **必须 `flashinfer_cutlass`** |
| 权重 | 10 safetensors，约 130 GB | 每卡模型已占约 75 GiB，显存极紧 |

## 2. 必须踩过的三个坑

| 坑 | 现象 | 解法 |
|---|---|---|
| MoE runner auto 选错 | 报 `AttributeError: 'FusedMoE' object has no attribute 'g1_scale_c'`（本模型缺 g1 权重，auto 走了 `flashinfer_trtllm` FP4 路径） | **强制 `--moe-runner-backend flashinfer_cutlass`** |
| `track` 与 `page` 不整除 | `AssertionError: view.mamba_track_interval % view.page_size == 0`（如 page64/track96 崩） | **`track % page == 0`** |
| MoE all-to-all 后端不兼容 | `AssertionError: Flashinfer MoE A2A requires --attn-dp-size equal to --tp-size` | 本模型不满足，**放弃 `--moe-a2a-backend flashinfer`** |

## 3. 最终最优部署脚本（PG32）

```bash
#!/bin/bash
# Qwen3.8-Flash-Next-NVFP4 双卡 RTX 6000D 最优部署 (PG32)
# 实测: C=16 输出吞吐 730 tok/s, TTFT 6.4s

docker rm -f sglang-qwen38-nvfp4 2>/dev/null; sleep 3

docker run -d --name sglang-qwen38-nvfp4 \
  --gpus all \
  --shm-size 32g \
  --network=host \
  --ipc=host \
  --cap-add SYS_NICE \
  -v /models:/models \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1 \
  -e NCCL_P2P_LEVEL=SYS \
  lmsysorg/sglang:dev sglang serve \
    --model-path /models/Qwen3.8-Flash-Next-NVFP4 \
    --tp 2 --ep 2 --disable-custom-all-reduce \
    --fp4-gemm-backend flashinfer_cutlass \
    --moe-runner-backend flashinfer_cutlass \
    --mem-fraction-static 0.85 \
    --chunked-prefill-size 8192 --max-prefill-tokens 16384 \
    --page-size 32 \
    --mamba-track-interval 128 \
    --context-length 262144 \
    --linear-attn-prefill-backend flashinfer \
    --linear-attn-decode-backend flashinfer \
    --linear-attn-verify-backend triton \
    --mamba-ssm-dtype bfloat16 \
    --kv-cache-dtype fp8_e4m3 \
    --mamba-radix-cache-strategy extra_buffer_lazy \
    --max-running-requests 48 \
    --max-mamba-cache-size 48 \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --reasoning-parser qwen3 \
    --host 0.0.0.0 --port 30000 \
    --ple-offload-embedding

# 等待就绪
for i in $(seq 1 100); do
  curl -sf http://127.0.0.1:30000/health >/dev/null 2>&1 && { echo "HEALTHY $((i*5))s"; break; }
  docker ps --format '{{.Names}}' | grep -q sglang-qwen38-nvfp4 || { echo "DIED"; break; }
  sleep 5
done
```

启动日志参考：`load_weight≈41s`，`cuda_graph prefill≈51s`，冷启到 HEALTHY 约 **205s**。

## 4. 压测脚本（注意三个调用坑）

```bash
#!/bin/bash
SP=/models/ShareGPT_V3_unfiltered_cleaned_split/ShareGPT_V3_unfiltered_cleaned_split.json

for C in 1 8 16; do
  echo "==== C=$C ===="
  docker run --rm --gpus all --shm-size 32g --network=host --ipc=host \
    -v /models:/models \
    lmsysorg/sglang:dev \
    python3 -m sglang.bench_serving \
      --backend vllm-chat \
      --dataset-name random \
      --dataset-path $SP \
      --model /models/Qwen3.8-Flash-Next-NVFP4 \
      --tokenizer /models/Qwen3.8-Flash-Next-NVFP4 \
      --random-input-len 10000 --random-output-len 2000 --random-range-ratio 1 \
      --num-prompts $C \
      --base-url http://127.0.0.1:30000 \
      --max-concurrency $C 2>&1 \
      | grep -iE "throughput|ttft|tpot"
done
```

### 压测调用三个坑

1. **`--base-url` 用 `127.0.0.1`，不要用 `0.0.0.0`**：`0.0.0.0` 是监听地址不是可连接地址。
2. **`--dataset-name random` 必须配 `--dataset-path`** 指向本地 sharegpt 文件：该版本 random 模式内部仍会 `download_and_cache_hf_file`，本机无外网会抛 `LocalEntryNotFoundError`。
3. **`--num-prompts=8` 样本太少**：C=8 吞吐在 330–456 tok/s 间抖动 ±30%，**C=16 才稳定可作为决策依据**。

## 5. 实测数据（in=10000 / out=2000 / range-ratio 1）

### 5.1 原始方案 → 逐步优化

| 阶段 | 关键变更 | C=8 tok/s | C=16 tok/s | C=16 TTFT |
|---|---|---|---|---|
| 原始 | run16 / track64 / page64 | 522 | 522 | 24.4s |
| run48 基线 | run48 / track128 / page64 | 456 | 631 | 6.4s |
| **PG32 定稿** | + **page32** | 448 | **730** | 6.4s |

### 5.2 参数扫描全表（A 方案：page-size / mamba-track-interval）

| 档位 | page | track | C=8 tok/s | C=16 tok/s | C=16 TTFT | C=16 TPOT |
|---|---|---|---|---|---|---|
| 基线 | 64 | 128 | 456 | 631 | 6.4s | 17.1ms |
| TR64 | 64 | 64 | 330 | 720 | 6.4s | 17.3ms |
| TR256 | 64 | 256 | 341 | 634 | 6.4s | 16.9ms |
| PG128 | 128 | 128 | 441 | 685 | 6.4s | 16.5ms |
| **PG32** | **32** | 128 | 448 | **730** | 6.4s | 16.6ms |

**读法**：C=16 为稳定维度；PG32（730）≈ TR64（720）> PG128（685）> TR256（634）≈ 基线（631）。
TTFT 全部约 6.4s，说明 `track-interval=128` 已吃满收益，64/256 无差异。

### 5.3 早期基线（含 C=1 / C=32）

| 并发 | req/s | 输出 tok/s | TPOT | TTFT |
|---|---|---|---|---|
| C=1 | 0.08 | 164 | 6.06ms | 42ms |
| C=8 | 0.26 | 522 | 12.4ms | 550ms |
| C=16 | — | 522 | — | 2.8s（早期基线最佳 TTFT） |
| C=32 | 0.40 | 809 | 17.9ms | 24.4s（拖垮服务） |

## 6. 已验证无效 / 不可行（别再试）

| 项 | 结论 |
|---|---|
| `--moe-a2a-backend flashinfer` | 需 attn-dp-size=tp-size，本模型不满足 |
| `chunked-prefill 16384` + run32/run64 | **OOM**（每卡模型占 75 GiB，`OutOfMemoryError` GPU0 82.94/83.09 GiB） |
| `max-mamba-cache 96` | OOM |
| C=32（32×12000 token） | 超出双卡能力，必超时并拖垮服务 |
| 调度轴（swizzle/persistent/cluster/CLC） | 单卡 KB 与本机实测均**从未生效** |

## 7. 天花板（为什么到此为止）

- **跨卡 45 GB/s（无 NVLink）** 是 MoE EP2 all-to-all 的物理上限，与 `RTX6000D_双卡优化总纲.md` 的 TP/通信结论同源。
- `flashinfer_cutlass`（FP4 GEMM+MoE）与 `flashinfer`（线性注意力）已是 sm120 当前最佳 kernel。
- 投机解码 accept rate 0.55–0.58、accept len 2.64–2.73，**不可关**（关了会显著掉吞吐）。

## 8. 后续可选方向（未实测，按收益/风险排）

| 方案 | 预期 | 风险 |
|---|---|---|
| `--speculative-num-steps 4`（accept 0.55 仍有空间） | 5~10% | 显存 + 延迟 |
| `--attention-backend trtllm_mha`（12 层 full-attn 未单独调） | 5~10% | 需重启验证 |
| `--enable-torch-compile` | 5~15% | 编译慢、可能 OOM |
| `--linear-attn-decode-backend triton` | ±5% | 需实测 |
| `--kv-cache-dtype nvfp4` | 省显存 | 可能不兼容崩溃 |
