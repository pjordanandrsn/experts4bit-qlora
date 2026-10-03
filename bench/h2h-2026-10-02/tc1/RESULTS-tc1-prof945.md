# TC1 amendment 12 read (#945): where e4b's fused training step goes after the sync fix — P19 HELD

**Box:** `tc1-5090-41` (instance 54000851, Intel Core Ultra 9 285K, RTX 5090, driver 595.91.07, $0.27; receipts in
[`receipts/tc1-5090-41/`](receipts/tc1-5090-41/)). The code was e4b `635b66b` (with #946's single-read grouping) and grouped-nf4-gemm
`133ad9d` (the pinned index ring as the default, #439). Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md),
amendment 12.

**What was asked.** Qwen3-30B-A3B at the field recipe ran TC1's profile instrument: 3 warm steps, then 3 profiled steps, with `nvidia-smi dmon`
beside. There were three arms, one draw each:

- e4b shipped on the new path;
- e4b matched (fp32 adapters, matched init) on the new path;
- e4b matched on the legacy path (legacy grouping, pageable index copies) as the before-picture, on the same host.

Every arm is VALID, and each arm's `sync_ab` record names the path it ran.

## Readings

| arm | s/step (steps 11+) | profiled wall / step | device / step | device busy | device events / step | CPU ops / step | peak GB | held-out at N |
|---|---|---|---|---|---|---|---|---|
| shipped, new path | 2.207 | 2,467.5 ms | 2,048.8 ms | **0.830** | 133,898 | 733,073 | 24.58 | 0.8147 |
| matched, new path | 3.119 | 3,552.7 ms | 2,628.7 ms | **0.740** | 136,969 | 750,409 | 27.82 | 0.8504 |
| matched, legacy path | 3.924 | 4,425.0 ms | 2,631.9 ms | **0.595** | 140,809 | 753,087 | 27.85 | 0.8510 |

**P19 HELD.** The matched arm's busy fraction rose from 0.595 to 0.740 (+0.145, against the +0.05 predicted). The two matched arms do the
same device work, 2,629 ms against 2,632 ms per step. The legacy arm's 4,436 `cudaStreamSynchronize` calls per step (974 ms of CPU time) are
gone, and the new path's remaining wait shows up inside `cudaMemcpyAsync`.

**Where the device time goes on the shipped arm**, per step:

| kernel | calls | ms |
|---|---|---|
| `_gemm_nf4_grouped` (forward and recompute) | 768 | 811.4 |
| `_dgrad_nf4_grouped` | 384 | 293.6 |
| one bf16 scalar-times-tensor kernel | 1,152 | **94.2** |
| the attention backward | 192 | 43.3 |

The grouped GEMM is far from both of the card's limits. This fixture's alpaca rows are short: 1,521 real tokens a step (30,428 over 20 steps),
2,080 with padding. That is about 24 to 32 routed rows per expert per layer pass. At that size the forward moves 11 to 15 TFLOP and reads
up to 116 GB of NF4 weights a step (every expert's stack, if every expert is hit) in 0.81 s: at most about 19 TFLOPS and 143 GB/s, under a tenth of the 5090's
bf16 tensor rate and of its memory bandwidth. It is the largest device consumer and the kernel target once the launch work is done.

The scalar kernel is grouped-nf4-gemm's
`lora_delta_grouped` multiplying the padded `[G, widest, N]` expert delta by `scaling`, in the forward, the recompute and the backward. At the
field recipe (r16, alpha 16) the scaling is 1.0.

The matched arm's fp32 counterpart runs 5,008 calls in 228.7 ms: the shipped arm's 3,856 fp32 calls plus the same 1,152. The matched arm also
runs its fp32 LoRA `bmm`s through SIMT sgemm kernels, and each `bmm` call costs 197 µs of CPU time (607 ms per step against the shipped arm's
47 ms). That is most of the 0.9 s per step between matched and shipped.

## What follows

Amendment 12's decision rule reads the matched arm. Device events and CPU ops per step stayed near the before-picture's, and the busy
fraction rose but stayed under 0.75. That is the launch-bound branch: the next work is launch volume in the attention and LoRA paths.

The first change is grouped-nf4-gemm#440: an exact trim of the padded LoRA delta that removes the scalar pass and the delta's index sorts,
fills and copies. On an RTX A2000 it measured 1,424 → 1,344 launches and 241.0 → 227.1 ms per step. Amendment 13 (#954) registers its 5090
A/B, P20 and P21. After that, the candidates are the matched arm's fp32 `bmm` and the grouped GEMM at about 24 to 32 rows per expert.
