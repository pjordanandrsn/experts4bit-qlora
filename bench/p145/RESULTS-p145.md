# P145 — results: **READ**. After K33, Mixtral-8x7B's one-request step on an RTX 5090 is led by its bf16 attention projections (2026-10-18)

Registration: `bench/p145/PREREG-p145.md` (#1771, merged `67ba40b7`). This is an instrument. It changes no default, gate or
speed claim, and profiled node times are never quoted as speed.

Code under test:
- e4b 0.55.0 at `67ba40b7`;
- grouped-nf4-gemm 0.47.0 at `eea5df6b`, which carries #556, so `GNF4_GEMV_BW=auto` serves Mixtral's expert GEMVs on K33
  at P141's plans with no knob set;
- torch 2.8.0+cu128, triton 3.4.0, transformers 5.17.0, bitsandbytes 0.50.2, Nsight Systems 2025.6.1.190;
- `mistralai/Mixtral-8x7B-Instruct-v0.1` at `eba92302`, on the default `serve_paged` server at 16 slots × 768 tokens,
  graphs on.

**Verdict by `p145_reduce.py`: READ.** No VOID condition fired. Both tables close every replay exactly, with 61 kept steps.
The W1 expert path ran 32 launches per step for each of gate_up, `_swiglu_rows` and down. The unprofiled self-pair is
0.9993.

**Run:** `p145-5090-1`, one RTX 5090 on a 24-CPU Intel Core Ultra 9 285K host, Vast instance 56617171. **Cost $0.816**, and
teardown was verified at 21:12:02Z.

## W1: where the 8.125 ms goes

The unprofiled arms read 8.2499 and 8.2559 ms a step. The node replay period P is 8.125 ms.

Removable time by class (ms): moe_expert 4.662, dense_gemm 2.324, attn 0.369, moe_route 0.195, norm_elem 0.161.

| line (by grid) | class | per step | removable (ms) | floor (ms) | addressable (ms) | share of P |
|---|---|---:|---:|---:|---:|---:|
| `_gemv_nf4_bw` gate_up | moe_expert | 32 | 2.749 | 3.086 | below 0 | — |
| `_gemv_nf4_bw` down | moe_expert | 32 | 1.882 | 1.950 | below 0 | — |
| `kernel` 1024×1×1 (q and o) | dense_gemm | 64 | 1.471 | 0 | **1.471** | **18.1 %** |
| `kernel` 256×1×1 (k and v) | dense_gemm | 64 | 0.573 | 0 | 0.573 | 7.1 % |
| `_fp8_paged_decode_split_f8dot` | attn | 32 | 0.250 | 0 | 0.250 | 3.1 % |
| `kernel` 8000×1×1 (the head) | dense_gemm | 1 | 0.157 | 0 | 0.157 | 1.9 % |
| `kernel` 2×1×1 | dense_gemm | 32 | 0.121 | 0 | 0.121 | 1.5 % |

- **The expert GEMVs run faster in the served graph than in P141's isolated timing:** gate_up 2.749 against 3.086 ms, down
  1.882 against 1.950 ms. Their in-situ excess is −0.405 ms together, so they have no addressable time.
- **Bandwidth floors (reporting only).** The two GEMVs' copy floors are 2.821 and 1.410 ms. The dense_gemm class moves
  2,946,498,560 B a step at bf16, a floor of 1.966 ms at P141's 1,498.9 GB/s, against its 2.324 ms removable.

## Next: proposed, not licensed

The top addressable line is dense_gemm, whose floor of 0 is not kernel headroom. So the reducer proposes a **format** lever:
a quality premise for compressing Mixtral's attention projections, before any kernel. Fewer bytes is the only way below the
bf16 copy floor. The q and o projections already move their 2.147 GB at 0.98 of the copy rate: 1.465 ms of kernel time
(the table's `dur_ms`; 1.471 ms removable) against a 1.432 ms copy floor.

A secondary kernel option is about 0.2 ms. The k and v projections move 0.537 GB at 0.63 of the copy rate: 0.568 ms of
kernel time (0.573 ms removable) against a 0.358 ms copy floor. Mixtral serves them unfused (`fuse_qkv_n` is 0), so a QKV
fusion is a candidate. Either one needs its own registration.

## W16 (reported, not proposed)

P = 53.224 ms. The grouped NF4 GEMM leads: `_gemm_nf4_grouped_smallm` reads 29.562 ms (gate_up) and 18.940 ms (down). Next
come dense_gemm at 3.095 ms and attn at 0.771 ms.

## Predictions

| prediction | outcome |
|---|---|
| W1's P is 8.0–8.8 ms | **HIT** (8.125) |
| dense_gemm removable is 1.6–2.2 ms, the top line | the top line **HIT**; the class removable **MISSED high** (2.324) |
| the two expert GEMVs' in-situ excess is 0.0–0.6 ms together | **MISSED low** (−0.405: faster in the graph than isolated) |
| the K33 split-K tail (sum + copy) is 0.05–0.25 ms | **not graded**: the table has no separate split-K line. The likely pair is `reduce_kernel` 16×1×1 (0.092) and `vectorized_elementwise_kernel` 8×1×1 (0.036), 32 a step each, filed as moe_route, which would sum to 0.128 |
| attention decode is 0.3–0.7 ms | **HIT** (attn 0.369) |

## What this cannot say

Speed: profiled node times are never quoted as one. It also says nothing about other cards, other batches than 1 and 16,
prefill, or Mixtral-8x22B. A format lever's quality is untested here.

## Evidence

- `receipts/p145-5090-1/` holds:
  - `verdict_p145.json`;
  - the per-name tables `pername_b1.json` and `pername_b16.json`, and the census records;
  - the arms `arm_S1a.json` and `arm_S1b.json`;
  - `summary.txt`, the logs and `teardown-proof.json`;
  - `SHA256SUMS`, which covers the directory.
- The maintainer re-derived `verdict_p145.json` byte-identical from the receipt store (`f3ce417a`), with the reducer and
  `p136_reduce.py` from `67ba40b7` (the staged copies) and the self-test at 17/17.
