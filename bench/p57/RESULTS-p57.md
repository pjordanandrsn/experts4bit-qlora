# Results — P57: K17's fused reduce in the consumer (refuted at both batches), P54's B=16 divergence split (not the glue — and, corrected 2026-09-22 by P59, not the K16 kernel either), the distinct-expert count (unread — amendment 1) — RTX 5090, 2026-09-22

Pre-registration: [`P57-PREREG.md`](P57-PREREG.md) (#666, merged 389d28b before any run; amendment 1 appended after this read, before any P4 data). Receipts: [`receipts/`](receipts/) (12 step receipts with token lists, 6 first-draw censuses, `summary.txt`, `forensics.txt`, `versions.txt`, `teardown-proof.json`, the empty `distinct_experts_b16.json` and its log). Every number below is read from them by [`p57_reduce.py`](p57_reduce.py) (`receipts/p57_rep.json`); the tables under "Generated read" are its output, unedited.

**Two lanes.** `p57-5090-1` (13:08Z, box 52063602, driver 580.119) died in the NF4 bake at the first shard read — an opaque CUDA error whose text the drive's fetch dropped (`--exclude work_qwen3`; fixed in #676, `k8_bake.py`'s `bake.json` now travels with the receipt). $0.09, torn down, nothing measured. **`p57-5090-2`** (13:26Z → `TP_DONE` 13:58Z, box 52066091, RTX 5090 on an AMD EPYC host, driver as in `forensics.txt`; e4b **4cab05a**, grouped-nf4-gemm **65cb104** = the K17 merge; transformers 5.16.1, torch 2.8.0+cu128, triton 3.4.0) ran all twelve timed arms clean in 32 minutes for **$0.22** against the $1.65 estimate; teardown proven.

## The read

| pair (B) | control median | fused median | Δ (control − fused) | registered band | verdict |
|---|---|---|---|---|---|
| `int4_b1` vs `int4_b1_fr` (B=1, fused q/k/v, K17 reduce OFF vs ON) | 4.236 ms (4.236 / 4.235) | 4.274 ms (4.274 / 4.274) | **−0.038 ms** | saving ≥ 0.15, refuted < 0.08 | **P1 REFUTED** |
| `int4_b16` vs `int4_b16_fr` (B=16, unfused, reduce OFF vs ON) | 11.578 ms (11.576 / 11.580) | 11.796 ms (11.791 / 11.800) | **−0.217 ms** | saving ≥ 0.10, refuted < 0.05 | **P2 REFUTED** |
| `int4_b16_nor2` vs `int4_b16_fqkv_nor2` (B=16, round-2 glue OFF on both, unfused vs fused q/k/v) | 13.266 ms | 12.929 ms | +0.337 ms (incidental) | tokens: diverge → kernel; identical → glue | **P3: DIVERGE — the kernel** |

- **P1 / P2 — K17's fused reduce costs time in the consumer.** The route engaged exactly as asked (`K17ROUTE`: `_reduce_partials` rows 0 in both `_fr` censuses, 1 in both controls; STOP-3 clean) and the tokens are identical to the control's on every sequence at both batches (K17's P1, bitwise, holds in the wild). But the step got **slower**: 0.038 ms at B=1, 0.217 ms at B=16, with A/A spreads of 0.001–0.009 ms. The census says why, and it is K17's own read restated at the step level: at B=1 the `_reduce_partials` family falls 0.318 → 0.005 ms/step (193 → 1 calls) while `_gemv_int4_b32` rises **1.415 → 1.642 ms** (+0.227) — the last-arriver epilogue lands inside the GEMV's own duration, and that kernel is on the critical path, whereas the separate reduce launches were overlapped in the graph. Net −0.038. At B=16 the reduce family falls 0.365 → 0.060 while the GEMV rises **6.340 → 6.781 ms (+7.0 %)** at R=128 rows per call — the 6–9 % R=128 slowdown K17 measured per call, times 96 calls per step. **The `_reduce_partials` row P54's census showed (0.300 ms/step at B=1) was never a launch cost recoverable by fusion; it was GPU time that the fused epilogue spends more expensively.** Decision rule: `GNF4_GEMV_FUSED_REDUCE` stays 0 at both batches; K17's opt-in ships as exact and slower.
- **CORRECTION (2026-09-22, P59 — [`../p59/RESULTS-p59.md`](../p59/RESULTS-p59.md)): the second half of this P3 reading is withdrawn.** The glue exoneration stands (fused-qkv still diverges with round-2 glue off on both legs). The attribution to the K16 GEMM does not: on an A2000 the K16 small-M kernel is **bitwise invariant** to fusing q/k/v at 2–16 rows (same plan for every N, columns independent), and P59 read KL exactly 0 between the fused and unfused stacks at 16-row decode on the 5090. The divergence lives in the >16-row path (the harness's 128-token prefill steps through cuBLAS on the cached bf16 weight, whose kernel choice depends on N). Also corrected: the first-divergence indices below are **not** the same as P54's — P54 recorded (1,59) (2,117) (3,21) (5,8) (6,2) (7,16) (8,31) (9,53) with 14/16 diverging; this lane recorded (1,41) (2,3) (3,35) (4,38) (5,8) (6,2) (7,17) (8,10) with 15/16 — a different glue state, a different trajectory. The original paragraph is kept as written:
- **P3 — the B=16 divergence is the K16 GEMM's, not the glue's.** With `E4B_FUSE_T1_GLUE_R2=0` on both legs (each 1.7 ms/step slower than its glued twin, so the knob demonstrably took effect), fused q/k/v vs unfused **still diverges on 15 of 16 sequences on both draws, at the SAME first-divergence indices P54 recorded** ((1,41) (2,3) (3,35) (4,38) (5,8) (6,2) (7,17) (8,10)…), while the control's own two draws are bit-identical. The round-2 glue path is exonerated; the divergence follows `_gemm_int4_b32_smallm` tiling over a 3× wider N (192 → 96 calls/step, the K16 family 1.296 → 0.807 ms). P54's "leading hypothesis" is now the only one standing. What this does **not** say: whether the different tokens are worse — the KL/K8 read that would license a B=16 default remains the gate, and `--fuse-qkv` stays opt-in at B=16.
- **P4 — READ (`p57d-5090-1`, amendment 3, $0.07): mean 58.7 distinct experts per layer per decode step at B=16 — HOLDS the ≤ 80 prior.** 73 decode steps counted on device (3 eager warm-up + the 70 graph-replayed steps; `steps` identical on all 48 layers), layer means **50.9 … 71.7** (layer 0 the widest at 71.7, the deepest third 50–58), against a uniform-router expectation of 82.4: real routing is ~29 % below uniform. The skew is in the per-expert touch fractions ([`receipts/p57d/distinct_experts_b16.json`](receipts/p57d/distinct_experts_b16.json) `touched_frac`): per layer **7–41 of 128 experts were never touched** in 73 steps × 16 rows while 1–12 were touched on every step. **Against #564's expert-tier byte roofline** (5.336 ms/step at 64 distinct experts, 6.670 at 80, i.e. 83.4 µs per distinct expert per layer-step): at the measured 58.7 the floor is **4.89 ms**, and the same lane's B=16 census read `_gemv_int4_b32` at **6.34 ms** — the expert GEMV runs at **77 % of its byte roofline at the routing it actually sees**, not the ~90 % the 80-expert assumption implied; ~1.4 ms/step of headroom sits in that row (K17's route was not it). The arm's own step time (13.10 ms, `E4B_FUSE_ROUTER_EPI=0` so the router module is called) is recorded and never quoted. Register row `e4b.serve.p57.qwen3.b16.distinct-experts.5090.2026-09-22`. The road here, kept for the record — **P4 — PARTIAL (3 decode steps) on `p57b`/`p57c`:** `p57b-5090-2` ($0.10) ran amendment 1's arm and the router counter's `torch.unique` broke CUDA-graph capture (`operation failed due to a previous error during capture`); it counted the harness's 3 eager warm-up decode steps: **mean 54.8 distinct experts per layer per step** (layer means 46.7–67.0; uniform expectation 82.4). Consistent with the ≤ 80 prior, **not registered** — three steps is a glimpse. Amendment 2 moved the untimed arm to the eager loop — **inert** (`--b1d-loop` governs B=1 only): `p57c-5090-2` ($0.14) repeated the 3-step outcome and the new `steps < 16` refusal fired (rc 45; [`receipts/p57c/`](receipts/p57c/)). **Amendment 3** rewrites the counter to accumulate on device (capture-safe; verified under `torch.cuda.graph` capture + replay on the A2000, 24/24 steps exact) — lane `p57d-5090`. Receipts: [`receipts/p57b/`](receipts/p57b/). The original failure, kept for the record: **P4 — UNREAD** on `p57-5090-2`: The untimed distinct-expert arm ran the model end to end (hook attached, `P57 HOOK: distinct-expert counter on 48 routers`) and then hit the harness's own guard: `--gen-tokens 64` leaves the bv3 stage a 6-step window (`AssertionError: window too small for bv3 (6)`; P54's series arm used 128 → 70 steps). The hook's atexit dump has `steps: 0`; the runner's `[ -s file ]` passed on it and the lane exited rc=0 — a green skipped path, caught by the reducer ("MISSING — P4 unread"). Amendment 1 registers the re-run (`--gen-tokens 128`, `steps: 0` → rc 45, `P57_ONLY_DISTINCT=1`, lane `p57b-5090`, ≤ $0.35) before any P4 data exists. The dry-run rule was applied to the counter, not to the harness's window arithmetic at the registered token count — recorded as the same class of gap as P54's series arm.
- **P5 — bookkeeping.** B=16 control **11.578 ms** vs P54's 11.420 on its box: **+1.4 %** (inside ±3 %). B=1: the pre-registration compared the fused `int4_b1` arm against P54's **unfused** 4.21 ms (+0.6 %, "within") — the wrong reference; against P54's **fused** 3.693 ms it is **+14.7 %**, outside STOP-1's ±8 %. This lane has no unfused B=1 arm, so it cannot say whether the fusion's 12.4 % saving shrank on this host or the whole B=1 step is slower here (B=1 is host-bound; the class carries ~8.5 % dispersion). **No B=1 number from this box is compared to P54's**; the within-box K17 read at B=1 (−0.038 ms, A/A 0.001) stands on its own.

## What changes

Nothing in the shipped defaults — and that is the finding. K17's fused reduce: **opt-in, exact, slower** (register row `gnf4.kernel.k17-fused-splitk-gemv.5090.2026-09-21` already says so at the kernel level; the consumer rows below say it at the step level). `--fuse-qkv`: **default at B=1, opt-in at B=16**, unchanged; the divergence hypothesis narrows to the kernel. The B=1 launch lever named in K17's prereg — folding `_quant_x_rows` (0.158 ms/step at B=1, 192 calls) into the GEMV — is **not** licensed by anything here and is the next candidate only if a per-call microbench shows the launch is on the critical path (K17 shows how that can fail). Register rows: `e4b.serve.p57.qwen3.{b1,b16}.{control,fr}.5090.2026-09-22` and `e4b.serve.p57.qwen3.b16.{nor2_control,nor2_fqkv}.5090.2026-09-22` (measured, same-box; no cross-box number quoted).

## Generated read (`p57_reduce.py`, unedited)

Pre-registration: [`P57-PREREG.md`](P57-PREREG.md). Every number below is read from the fetched receipts by `p57_reduce.py`.

**STOP-1 (b1_fr)** -- control first draw 4.236 ms vs P54's 4.21 ms on its box: +0.6 % (within +-8 %; ratios within this box are the position either way).

**STOP-1 (b16_fr)** -- control first draw 11.576 ms vs P54's 11.42 ms on its box: +1.4 % (within +-8 %; ratios within this box are the position either way).

## b1_fr: `int4_b1` vs `int4_b1_fr` (B=1)

| arm | draw 1 (ms) | draw 2 (ms) | median |
|---|---|---|---|
| control | 4.236 | 4.235 | 4.236 |
| fused | 4.274 | 4.274 | 4.274 |

**Saving (control median - fused median): -0.038 ms/step** -> **REFUTED (saving -0.038 < 0.08)**

| kernel family | control ms/step | fused ms/step | rel | control calls/step | fused calls/step |
|---|---|---|---|---|---|
| k16 small-M GEMM | 0.000 | 0.000 | -- | 0 | 0 |
| int4 expert GEMV | 1.415 | 1.642 | +16.1 % | 192 | 192 |
| bf16 GEMM (cutlass/cublas) | 0.493 | 0.493 | -0.1 % | 49 | 49 |
| fp8 paged decode | 0.347 | 0.347 | -0.0 % | 48 | 48 |
| index / gather / scatter | 0.300 | 0.301 | +0.1 % | 193 | 193 |
| activation quant | 0.158 | 0.159 | +0.9 % | 192 | 192 |
| split-K reduce | 0.318 | 0.005 | -98.4 % | 193 | 1 |
| rope/norm folds | 0.241 | 0.241 | +0.0 % | 193 | 193 |

P3 calls (control, fused): (192.0, 0) -> OK; other families moved > 5 %: ['int4 expert GEMV']

<details><summary>top_control (top 10 kernels)</summary>

| kernel | ms/step | calls/step |
|---|---|---|
| `_gemv_int4_b32` | 1.415 | 192 |
| `std::enable_if<!(false), void>::type internal::gemvx...` | 0.379 | 1 |
| `_fp8_paged_decode_split_f8dot` | 0.347 | 48 |
| `_reduce_partials` | 0.313 | 192 |
| `_quant_x_rows` | 0.158 | 192 |
| `void at::native::(anonymous namespace)::indexSelectS...` | 0.139 | 49 |
| `_router_epilogue` | 0.135 | 48 |
| `_fp8_append_t1_side` | 0.133 | 96 |
| `std::enable_if<!(false), void>::type internal::gemvx...` | 0.115 | 48 |
| `_rope_norm_heads` | 0.110 | 96 |

</details>

<details><summary>top_fused (top 10 kernels)</summary>

| kernel | ms/step | calls/step |
|---|---|---|
| `_gemv_int4_b32` | 1.642 | 192 |
| `std::enable_if<!(false), void>::type internal::gemvx...` | 0.378 | 1 |
| `_fp8_paged_decode_split_f8dot` | 0.347 | 48 |
| `_quant_x_rows` | 0.159 | 192 |
| `void at::native::(anonymous namespace)::indexSelectS...` | 0.140 | 49 |
| `_router_epilogue` | 0.135 | 48 |
| `_fp8_append_t1_side` | 0.133 | 96 |
| `void at::native::vectorized_elementwise_kernel<4, at...` | 0.124 | 193 |
| `std::enable_if<!(false), void>::type internal::gemvx...` | 0.115 | 48 |
| `_rope_norm_heads` | 0.110 | 96 |

</details>

**P5 (token parity)** -- generated tokens, fused vs control:

- draw 1: **identical** on all 1 sequence(s)
- draw 2: **identical** on all 1 sequence(s)
- A/A (control draw 1 vs draw 2): **identical** on all 1 sequence(s)

## b16_fr: `int4_b16` vs `int4_b16_fr` (B=16)

| arm | draw 1 (ms) | draw 2 (ms) | median |
|---|---|---|---|
| control | 11.576 | 11.580 | 11.578 |
| fused | 11.791 | 11.800 | 11.796 |

**Saving (control median - fused median): -0.217 ms/step** -> **REFUTED (saving -0.217 < 0.05)**

| kernel family | control ms/step | fused ms/step | rel | control calls/step | fused calls/step |
|---|---|---|---|---|---|
| k16 small-M GEMM | 1.296 | 1.309 | +1.0 % | 192 | 192 |
| int4 expert GEMV | 6.340 | 6.781 | +7.0 % | 96 | 96 |
| bf16 GEMM (cutlass/cublas) | 0.566 | 0.565 | -0.1 % | 97 | 97 |
| fp8 paged decode | 0.682 | 0.685 | +0.5 % | 48 | 48 |
| index / gather / scatter | 0.953 | 0.955 | +0.2 % | 241 | 241 |
| activation quant | 0.310 | 0.310 | +0.0 % | 96 | 96 |
| split-K reduce | 0.365 | 0.060 | -83.6 % | 145 | 49 |
| rope/norm folds | 0.268 | 0.268 | +0.2 % | 193 | 193 |

P3 calls (control, fused): (96.0, 0) -> OK; other families moved > 5 %: ['int4 expert GEMV']

<details><summary>top_control (top 10 kernels)</summary>

| kernel | ms/step | calls/step |
|---|---|---|
| `_gemv_int4_b32` | 6.340 | 96 |
| `_gemm_int4_b32_smallm` | 1.296 | 192 |
| `_fp8_paged_decode_split_f8dot` | 0.682 | 48 |
| `void at::native::(anonymous namespace)::indexSelectS...` | 0.464 | 48 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_...` | 0.384 | 1 |
| `_quant_x_rows` | 0.310 | 96 |
| `_reduce_partials` | 0.305 | 96 |
| `void at::native::indexFuncSmallIndex<int, long, unsi...` | 0.209 | 48 |
| `_fp8_append_bt1_side` | 0.172 | 96 |
| `void at::native::(anonymous namespace)::indexSelectS...` | 0.147 | 48 |

</details>

<details><summary>top_fused (top 10 kernels)</summary>

| kernel | ms/step | calls/step |
|---|---|---|
| `_gemv_int4_b32` | 6.781 | 96 |
| `_gemm_int4_b32_smallm` | 1.309 | 192 |
| `_fp8_paged_decode_split_f8dot` | 0.685 | 48 |
| `void at::native::(anonymous namespace)::indexSelectS...` | 0.464 | 48 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_...` | 0.383 | 1 |
| `_quant_x_rows` | 0.310 | 96 |
| `void at::native::indexFuncSmallIndex<int, long, unsi...` | 0.211 | 48 |
| `_fp8_append_bt1_side` | 0.172 | 96 |
| `void at::native::(anonymous namespace)::indexSelectS...` | 0.147 | 48 |
| `_router_epilogue` | 0.136 | 48 |

</details>

**P5 (token parity)** -- generated tokens, fused vs control:

- draw 1: **identical** on all 16 sequence(s)
- draw 2: **identical** on all 16 sequence(s)
- A/A (control draw 1 vs draw 2): **identical** on all 16 sequence(s)

## b16_nor2: `int4_b16_nor2` vs `int4_b16_fqkv_nor2` (B=16)

| arm | draw 1 (ms) | draw 2 (ms) | median |
|---|---|---|---|
| control | 13.264 | 13.269 | 13.266 |
| fused | 12.932 | 12.927 | 12.929 |

**Saving (control median - fused median): 0.337 ms/step** -> **parity read (saving 0.337 ms is incidental: the split asks about TOKENS, see P3)**

| kernel family | control ms/step | fused ms/step | rel | control calls/step | fused calls/step |
|---|---|---|---|---|---|
| k16 small-M GEMM | 1.296 | 0.807 | -37.7 % | 192 | 96 |
| int4 expert GEMV | 6.652 | 6.465 | -2.8 % | 96 | 96 |
| bf16 GEMM (cutlass/cublas) | 0.567 | 0.566 | -0.2 % | 97 | 97 |
| fp8 paged decode | 0.681 | 0.685 | +0.6 % | 48 | 48 |
| index / gather / scatter | 0.939 | 0.947 | +0.8 % | 241 | 241 |
| activation quant | 0.312 | 0.312 | +0.1 % | 96 | 96 |
| split-K reduce | 0.459 | 0.454 | -0.9 % | 193 | 193 |
| rope/norm folds | 0.194 | 0.193 | -0.3 % | 145 | 145 |

P3 calls (control, fused): (192.0, 96.0) -> OK; other families moved > 5 %: none

<details><summary>top_control (top 10 kernels)</summary>

| kernel | ms/step | calls/step |
|---|---|---|
| `_gemv_int4_b32` | 6.652 | 96 |
| `_gemm_int4_b32_smallm` | 1.296 | 192 |
| `_fp8_paged_decode_split_f8dot` | 0.681 | 48 |
| `void at::native::(anonymous namespace)::indexSelectS...` | 0.460 | 48 |
| `void at::native::elementwise_kernel<128, 4, at::nati...` | 0.410 | 240 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_...` | 0.384 | 1 |
| `_reduce_partials` | 0.316 | 96 |
| `_quant_x_rows` | 0.312 | 96 |
| `void at::native::indexFuncSmallIndex<int, long, unsi...` | 0.200 | 48 |
| `void at::native::unrolled_elementwise_kernel<at::nat...` | 0.200 | 97 |

</details>

<details><summary>top_fused (top 10 kernels)</summary>

| kernel | ms/step | calls/step |
|---|---|---|
| `_gemv_int4_b32` | 6.465 | 96 |
| `_gemm_int4_b32_smallm` | 0.807 | 96 |
| `_fp8_paged_decode_split_f8dot` | 0.685 | 48 |
| `void at::native::(anonymous namespace)::indexSelectS...` | 0.463 | 48 |
| `void at::native::elementwise_kernel<128, 4, at::nati...` | 0.413 | 240 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_...` | 0.383 | 1 |
| `_quant_x_rows` | 0.312 | 96 |
| `_reduce_partials` | 0.311 | 96 |
| `void at::native::indexFuncSmallIndex<int, long, unsi...` | 0.203 | 48 |
| `_rmsnorm_rows` | 0.193 | 145 |

</details>

**P5 (token parity)** -- generated tokens, fused vs control:

- draw 1: **15 of 16 sequences diverge**; first divergence (seq, index): [('1', 41), ('2', 3), ('3', 35), ('4', 38), ('5', 8), ('6', 2), ('7', 17), ('8', 10)]
- draw 2: **15 of 16 sequences diverge**; first divergence (seq, index): [('1', 41), ('2', 3), ('3', 35), ('4', 38), ('5', 8), ('6', 2), ('7', 17), ('8', 10)]
- A/A (control draw 1 vs draw 2): **identical** on all 16 sequence(s)

## P4: distinct experts per layer per decode step (B=16, `int4_b16_distinct`, untimed)

`distinct_experts_b16.json` MISSING -- P4 unread.
