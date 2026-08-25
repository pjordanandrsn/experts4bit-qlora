# RESULTS — F1 Stage A: the elementwise block, attributed

Measured 2026-08-25 under PREREG-f1-elementwise + AMENDMENT-f1-tracer.
Receipts in `receipts-f1/stageA/` (RTX 5090, driver 595.71, torch
2.13.0+cu130 / triton 3.7.1 — the package's declared floor stack;
gnf4 `1517955`, e4b `6ddbf4d`; instance destroyed, account verified
zero). Attempt 1's vacuous receipt is kept beside it in
`receipts-f1/stageA-attempt1-vacuous/`.

## Gates

| gate | bar | measured |
|---|---|---|
| census coverage | ≥ 90% of the profiler's Self CUDA total | **100.2%** |
| attribution window | 12/12 active | **12/12** |
| unclassified ops | reported, not hidden | **0.003 ms/step** |
| named call sites | ≥ 90% of attributed time | **100.0%** (47 sites) |

Device budget: **10.76 ms/step**, of which **elementwise 4.74 ms over
4006 launches** (kernel view). The tracer attributed 4.88 ms/step over
the op view; the two views agree to 3%.

## Where the block comes from

| µs/step | launches | site | ops |
|---|---|---|---|
| 587 | 6176 | `modeling_qwen3_moe.py:298` | mul, rsqrt, add |
| 539 | 1272 | `fp8_paged_kv.py:147` `_write_side` | copy_ |
| 539 | 1272 | `fp8_paged_kv.py:149` `_write_side` | copy_ |
| 429 | 4117 | `modeling_qwen3_moe.py:297` | mean, pow |
| 316 | 1016 | `nf4_grouped.py:919` `gemm_4bit_grouped` | copy_, sum |
| 281 | 4096 | `fp8_kv.py:75` `quantize_kv_fp8` | where, div, gt |
| 265 | 2048 | `modeling_qwen3_moe.py:55` `rotate_half` | cat, neg |
| 247 | 4117 | `modeling_qwen3_moe.py:299` | mul, _to_copy |
| 219 | 2048 | `fp8_kv.py:72` `quantize_kv_fp8` | amax, abs |
| 165 | 1016 | `fp8_paged_kv.py:378` `kernel_args` | index_select |
| 164 | 1536 | `modeling_qwen3_moe.py:79` `apply_rotary_pos_emb` | mul, add |
| 164 | 1536 | `modeling_qwen3_moe.py:80` `apply_rotary_pos_emb` | mul, add |
| 145 | 1024 | `modeling_qwen3_moe.py:264` | sum, div_ |
| 114 | 1016 | `fp8_paged_kv.py:243` `append_many` | mul, index_add_ |

Four mechanisms account for **3.75 of the 4.88 ms**:

1. **RMSNorm, unfused — 1.26 ms/step** (`:297`+`:298`+`:299`, the
   `pow → mean → rsqrt → mul → mul` chain over ~14.4k launches). One
   fused kernel replaces five ops per norm.
2. **fp8 KV write + quantize — 1.58 ms/step** (`_write_side` ×2 at
   539 each, `quantize_kv_fp8` 500). This is OUR code, and the two
   `_write_side` copies are symmetric halves of one write.
3. **RoPE — 0.59 ms/step** (`rotate_half` cat/neg + the two
   `apply_rotary_pos_emb` mul/add lines).
4. **The nf4 wrapper's own copy+sum — 0.32 ms/step**
   (`gemm_4bit_grouped:919`).

The PASS bar needs 2.96 ms removed. Mechanisms 1+2+3 alone total
3.43 ms, so the bar is reachable without touching the wrapper — but
none of these is free: each is a real kernel to write and a real
fidelity question to gate.

## Note on the two views

Attribution runs on the op view (kernel rows carry no python stack)
and the census on the kernel view; they agree to 3% here (4.88 vs
4.74 ms). Site-level time is apportioned per op by launch share, which
is exact in this regime — the block runs at 1.18 µs/launch, at this
GPU's minimum kernel duration, so cost tracks launch count rather than
tensor size. Launch counts in the table are exact; per-site µs carry
that apportionment.

## Stage B

Registered bars unchanged: PASS ≤ 10.5 ms/step (≥ 95.2 tok/s),
PARTIAL 10.5–12.0, REFUTED > 12.0, with greedy token identity a hard
refusal rather than a disclosure.
