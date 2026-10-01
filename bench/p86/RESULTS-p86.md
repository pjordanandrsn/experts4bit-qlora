# P86 — results: **READ**. vLLM's B=16 lead is the expert kernel. Marlin MoE runs Qwen3-30B-A3B's experts in 4.78 ms per step, e4b's int4 GEMV in 6.98 ms; quantized linear is 2.86 of the 3.32 ms gap

Registration: `bench/p86/PREREG-p86.md` (#800, `b5f18b9`), with amendment 1 (#801, `db527f9`: the CUDA 12.9 image, and
a toolkit refusal before any install). Issue: #564.

**Verdict by `p86_reduce.py`: READ**, with the stated expectation held: quantized linear is the largest B=16 gap and
routing glue the second. One RTX 5090, the same prompt token ids as P58 (digest `f67e7e4d…`), both engines' censuses
reconciled to their own step time.

| | e4b int4 stack (0.37.8 + gnf4 0.33.7) | vLLM 0.30.0 (GPTQ-Int4, Marlin) | ratio |
|---|---:|---:|---:|
| B=16 step | 12.064 ms (12.07 / 12.06) | 8.746 ms (8.737 / 8.756) | **1.379** (P58: 1.396) |
| B=1 step | 4.274 ms (4.27 / 4.27) | 3.998 ms (3.999 / 3.998) | **1.069** (P58: 1.087) |

## Runs and cost

| run | outcome | cost (ledger) |
|---|---|---:|
| `p86-prove-1` | **NOT_RUN** at the launcher's pre-flight: download bandwidth 38.0 MB/s, under its 40 MB/s floor (machine 37675, which had reappeared as the cheapest offer after the readiness check) | $0.0221 |
| `p86-prove-2` | **the proof caught a defect**, rc 23: vLLM 0.30.0 installed, then its warmup died in FlashInfer's JIT (`requires GPUs with sm75 or higher`) under the registered CUDA 12.8 image. That led to amendment 1 | $0.0935 |
| `p86-prove-3` | **PROVED** under the CUDA 12.9 image: the census arm ran on vLLM 0.30.0 in-process with full CUDA graphs and recorded 33 decode kernels, 7 of them once per layer per step | $0.1686 |
| `p86-5090-2` | **NOT_RUN**: ssh did not authenticate (machine 37675 again). That receipt is the machine's exclusion evidence from then on | $0.0549 |
| `p86-5090-3` | **the reading: READ** | $0.5170 |
| **total** | | **$0.8561** of the $2.50 ceiling |

Every teardown is proven (`vast-destroy`, HTTP 200, instance absent). The ledger prices each run as the offer's rate ×
its runtime, not from an invoice.

## The reading (`p86-5090-3`)

- **Host.** One RTX 5090 (driver 595.71.05) on an Intel Xeon E5-2698 v4 (machine 96642;
  [`forensics.txt`](receipts/p86-5090-3/forensics.txt)), image `pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel`.
  - vLLM 0.30.0 ran on torch 2.13.0+cu130 in its own venv; e4b 0.37.8 at `db527f9` with grouped-nf4-gemm 0.33.7.
  - The prompt rows reproduced P37/P54/P57/P58's digest `f67e7e4d…`.
- **Timeline (UTC).** Lane started 02:38:58. Both installs done 02:43. Fetches and bake done 03:04. The ten arms ran
  03:04–03:31, and the lane exited rc 0 at 03:31.
- **The instruments reconcile.**
  - vLLM's B=16 census sums **8.733 ms** per decode step against its slope-timed 8.746, and its B=1 census 3.990
    against 3.998. Its step is essentially all GPU kernels.
  - e4b's B=16 census sums **12.372 ms** against a 12.064 ms step, 2.6 % over (inside the registered 10 %), and B=1
    sums 4.279 against 4.274.
  - e4b's excess comes from the census's 8 profiled replays, taken after the timed window. The negative "host and
    launch" row it produces is a profiler effect, not a saving.
  - The prefill jitter the decode filter excluded from vLLM's census was −0.0009 ms per step at B=16.
  - "Other" holds 6.3 % (e4b) and 3.8 % (vLLM) of B=16 kernel time, under the 10 % that would make the reading
    NOT_READ.

### By role (the registered reading)

| role | B=16 e4b | B=16 vLLM | **B=16 gap** | B=1 e4b | B=1 vLLM | B=1 gap |
|---|---:|---:|---:|---:|---:|---:|
| **quantized linear** (experts + attention projections) | 8.605 | 5.751 | **+2.855** | 2.088 | 2.008 | +0.080 |
| routing / gather glue | 1.072 | 0.454 | **+0.619** | 0.340 | 0.441 | −0.101 |
| other kernels | 0.775 | 0.334 | +0.441 | 0.467 | 0.125 | +0.342 |
| norm / rope / activation | 0.349 | 0.258 | +0.091 | 0.316 | 0.203 | +0.113 |
| dense bf16 GEMM (output head) | 0.609 | 0.591 | +0.018 | 0.518 | 0.506 | +0.013 |
| attention + KV write | 0.961 | 1.345 | **−0.384** | 0.550 | 0.708 | −0.158 |
| host and launch (step − kernels) | −0.308 | 0.013 | −0.322 | −0.005 | 0.008 | −0.013 |

### By kernel family, B=16

| e4b family | ms/step | | vLLM family | ms/step |
|---|---:|---|---|---:|
| `_gemv_int4_b32` (experts) | **6.979** | ↔ | Marlin MoE (`marlin_moe_wna16::Marlin`) | **4.783** |
| `_reduce_partials` (split-K) | 0.390 | | | |
| `_quant_x_rows` (activation quant) | 0.345 | | | |
| `_gemm_int4_b32_smallm` (attention projections, K16) | 0.891 | ↔ | Marlin dense (+ its fused norm) | 0.968 |
| index / gather / scatter (`indexSelect` ×2, `indexFuncSmallIndex`) | 1.072 | ↔ | MoE routing (`topkGating`, `moe_align_block_size`, `moe_sum`) | 0.454 |
| `_fp8_paged_decode_split_f8dot` + `_fp8_append_bt1_side` | 0.961 | ↔ | FlashAttention split-KV + combine + `reshape_and_cache` | 1.345 |
| `_rope_norm_heads` + `_rmsnorm*` | 0.349 | ↔ | `act_and_mul` + fused rms-norm (Triton) | 0.258 |
| cutlass bf16 GEMM (output head) | 0.609 | ↔ | cutlass fp16 GEMM | 0.591 |
| other (`elementwise` ×2, `_router_epilogue`, …) | 0.775 | ↔ | other (`elementwise`, `FillFunc`, sampling) | 0.334 |

The full kernel lists are in [`verdict.json`](receipts/p86-5090-3/verdict.json) (each family's top kernels), the
e4b replay censuses (`logs/census_e4b_b{16,1}.txt`) and the vLLM censuses (`vllm_census_b{16,1}.json`).

## What it says

- **The B=16 gap is the expert kernel.** Marlin MoE does the experts' work in 4.78 ms per step. e4b's GEMV takes
  6.98 ms, plus 0.39 ms of split-K reduce and 0.35 ms of activation quantization.
  - Against #564's byte floor at the measured routing (58.7 distinct experts per layer per step, P57: 4.89 ms for
    int4-b32's 2.654 MB per expert), e4b's GEMV runs at about **70 %** here. P57 read 77 % on another box at 6.34 ms.
  - GPTQ g128 stores about 8 % fewer bytes per expert: 2.43–2.45 MB (the same 2.359 MB of nibbles, 0.074 MB of
    scales, at most 0.018 MB of zero points), against int4-b32's 2.654. Marlin MoE's floor on the same routing is
    therefore about 4.5 ms, so it runs at about **94 %**.
  - These two percentages assume vLLM's router touches as many distinct experts as e4b's on these prompts. Its
    weights are a different quantization of the same model, so that is an approximation, not a measurement.
- **The attention projections are not the gap.** K16 (0.89 ms) is at parity with Marlin dense (0.97 ms).
- **e4b's attention is faster.** Its fp8 paged decode plus KV append (0.96 ms) beats FlashAttention on a bf16 cache
  (1.35 ms).
- **Glue is the second gap.** e4b's routing gathers are three generic torch index kernels (1.07 ms), against vLLM's
  purpose-built `topkGating`, `moe_align_block_size` and `moe_sum` (0.45 ms).
- **B=1 is close (1.07×).** The largest gap there is "other" (+0.34 ms: torch elementwise kernels and the router
  epilogue), not the expert kernel (+0.08 ms).

## What follows (the registered rule)

The largest B=16 gap is quantized linear, so the next lane is a **kernel lane on the B=16 expert matmul, with Marlin
MoE's 4.78 ms as the benchmark**. Three earlier attempts on this row shape the design:
- **P60:** one row per distinct expert, instead of one per routed row, runs 0.92 ms per step faster.
- **K18:** a grouped GEMV sharing loads across an expert's rows was exact but 1.49× slower.
- **P61:** row work and expert bytes overlap rather than add.

Marlin MoE does something none of those tried: it sorts the routed rows by expert (`moe_align_block_size`) and runs
tensor-core MMAs over each expert's block of rows, with Marlin's register-level int4 dequant. The next lane's first
step is a $0 microbench on the A2000 (sm_86), where vLLM 0.11 runs: e4b's `_gemv_int4_b32` against Marlin MoE on the
same shapes and a recorded B=16 routing. It finds where e4b's 30 % goes before any kernel is written. Proposed, not
started.

## Receipts

- **In this repo** (every file byte-identical to the store's, checked file by file):
  - [`receipts/p86-5090-3/`](receipts/p86-5090-3/), 16 files: the verdict, summary, forensics, versions, the four
    e4b and four vLLM timing receipts, both vLLM censuses, both e4b replay censuses, and `SHA256SUMS`.
  - [`receipts/p86-prove-3/`](receipts/p86-prove-3/): the proof's summary, forensics, versions and its census on
    Qwen3-0.6B.
  - [`receipts/p86-prove-2/`](receipts/p86-prove-2/): the summary and forensics of the attempt that caught the image
    defect. `run_vllm_census_prove_tail.txt` is the last 30 lines of its census log, an excerpt kept for the
    traceback and not in `SHA256SUMS`.
- **In the adertha receipt store:** `receipts/experts4bit-qlora/2026-10-01/p86-{prove-1,prove-2,prove-3,5090-2,5090-3}/`
  (commits `2b4e401`, `88627b6`, `8599a7e`, `3a1e20e`, `0c757b6`).
