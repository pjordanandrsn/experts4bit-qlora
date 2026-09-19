# Results — P45: where the training step's time goes (Qwen3-30B-A3B field recipe, RTX 5090, 2026-09-19)

Pre-registration: [`P45-PREREG.md`](P45-PREREG.md) (2026-09-19, before any box). Run `p45-qwen3prof-2` (vast 51531525; run 1 was refused at the manifest guard — the driver's box-E fix had not reached main). e4b `7f9bb05` + gnf4 v0.31.0 (`24f8c9fb`), transformers 5.17.0, torch 2.8.0+cu128. Field fixture: alpaca, seq 2048, mb 2 × accum 4, r 16, adamw_8bit; 8 steps, steps 3–5 profiled (`torch.profiler`, CPU + CUDA). Reducer: `bench/p45/p45_reduce.py`; receipts `receipts/experts4bit-qlora/2026-09-19/p45-qwen3prof-2/tp4/` (`qwen3_e4b_fused_attn4_profile.json`, `logs/dmon_*.txt`).

## The e4b arm (`fused_attn4`: `enable_fast_train(dgrad=True)` + int4 attention), three profiled steps

| quantity | value |
|---|---|
| wall per profiled step | 38.6 s (unprofiled steps in the same arm: 23.6–28.1 s; T1 read 29.3 s) |
| **device busy fraction** (summed device self time / wall) | **0.108** |
| memcpy share of device time | 0.111 |
| **device events per step** (launches + copies) | **996,845** |
| CPU ops per step (profiler op records) | 6,277,948 |
| CPU op self time / wall | 0.883 |
| CPU self time by family | matmul 0.250 · other 0.234 · autograd 0.208 · norm_act 0.176 · memcpy 0.123 · fused_kernel 0.0038 · routing 0.0036 · optimizer 0.0013 |
| device time by family | matmul 0.305 · fused_kernel 0.264 · other 0.148 · norm_act 0.127 · memcpy 0.124 |
| dmon (1 s samples, whole arm) | sm 3.1 % mean · PCIe rx+tx 0.073 GB/s mean, 9.9 peak |

### Top CPU rows (self time over the 3 profiled steps)

| op | family | calls | self ms |
|---|---|---|---|
| `aten::mm` | matmul | 819,832 | 24,822 |
| `cudaLaunchKernel` | other | 1,386,158 | 9,174 |
| `cudaMemcpyAsync` | memcpy | 813,796 | 7,346 |
| `cuLaunchKernel` | other | 776,536 | 6,146 |
| `IndexPutBackward0` | autograd | 696 | 5,547 |
| `aten::copy_` | memcpy | 857,272 | 4,732 |
| `aten::mul` | norm_act | 345,480 | 4,672 |
| `MmBackward0` | autograd | 203,380 | 3,837 |
| `aten::empty` | norm_act | 551,620 | 2,203 |
| `aten::slice` | norm_act | 533,956 | 1,957 |
| `aten::permute` | norm_act | 594,552 | 1,930 |
| `autograd::engine::evaluate_function: MmBackward0` | autograd | 203,380 | 1,905 |
| `aten::as_strided` | other | 2,498,520 | 1,764 |
| `aten::fill_` | norm_act | 413,828 | 1,717 |

### Top device rows

| kernel | family | calls | self ms |
|---|---|---|---|
| `_gemm_nf4_grouped` | fused_kernel | 2,304 | 2,419 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::CU` | norm_act | 312,612 | 1,536 |
| `Memcpy DtoD (Device -> Device)` | memcpy | 802,552 | 1,375 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::Fi` | other | 409,700 | 1,026 |
| `_dgrad_nf4_grouped` | fused_kernel | 1,152 | 877 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_b` | matmul | 37,462 | 815 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_` | matmul | 149,848 | 802 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_` | matmul | 193,582 | 516 |

## Predictions, read against the rows

- **P1 (the GPU is idle, not slow) — HOLDS.** Device busy 0.108 of the wall (≤ 0.30 registered); dmon's sm-utilisation mean over the whole arm 3.1 %. The step is host time.
- **P2 (where the host time is) — NONE of the four registered families dominates, and the family the registration did not list does.** routing + autograd = 0.211 (≥ 0.50 registered); fused_kernel 0.0038, memcpy 0.123, optimizer 0.0013 — none reaches its alternative threshold. What the top rows say: **819,832 `aten::mm` calls in three steps ≈ 273,277 per optimizer step**, 813,796 `cudaMemcpyAsync`, 857,272 `aten::copy_`, and their autograd nodes (`MmBackward0` × 203,380, `IndexPutBackward0`, `SelectBackward0`, `CopySlices`). Per layer-microbatch that is ≈ 128 experts × (A, B) × (forward, recompute, backward) — **the per-expert LoRA adapter matmuls, run as a Python loop over experts, with the slice/select/permute/copy plumbing around them and their autograd graph.** The fused NF4 expert kernels (`_gemm_nf4_grouped` 2,304 calls, `_dgrad_nf4_grouped` 1,152) are 26 % of a device timeline that is 11 % of the wall. The registration's family list was incomplete; the row stands and the missing family is named here, not folded into "other".
- **P3 (dispatch-bound structure) — HOLDS.** 996,845 device events per step (≥ 20,000 registered) at ~39 µs of wall per event.
- **P4 (the comparator) — NOT READ.** The Unsloth arm was killed by its alarm (1,388 s) before its first step: the arm alarm is derived from the run's deadline, and the e4b arm's profiling — three profiled steps at 38.6 s plus ~30 min of trace processing at 0 % GPU and 94 GB RSS — had consumed the budget. Redraws `p45-unsloth` and `p45-unsloth-2` were both NOT_RUN (the box never reached the lane); under the STOP rule the lane stops — P4 stays NOT READ (amendment 2).
- **P5 (not the bus) — HOLDS.** PCIe rx+tx 0.073 GB/s mean over the arm (< 2 registered; the 9.9 GB/s peak is the checkpoint load).

## Decision

P1 ∧ P3 ∧ P5 hold and P2 names a family outside the registered four: **the adapter-side dispatch**. The lever to register (its own pre-registration, with tp1's B2/C2 parity gate before any speed is quoted): a **grouped LoRA adapter path for training** — the A and B matmuls of all routed experts in one launch each (the structure the base expert GEMM already has in `_gemm_nf4_grouped`), with the gather/scatter done once per layer instead of per expert — expected to remove most of the ≈ 273k per-step matmul dispatches and their autograd nodes. Nothing is fixed in this lane. The 29.3 s/step field-recipe number (T1) is a host-dispatch number; no GPU-side lever changes it.

## Cost

Run 1 refused at the manifest guard ($0). Run 2 ≈ 76 min of a $0.65/h box ≈ $0.82 (guard 1.5 h). The Unsloth redraw is registered at ≤ $0.65.
