# Results — P54: qkv fusion on the int4 attention store (RTX 5090, 2026-09-21)

Pre-registration: [`P54-PREREG.md`](P54-PREREG.md). Run `p54-fqkv-1`, receipt `receipts/experts4bit-qlora/2026-09-21/p54-fqkv-1/` (private tree), **$0.4737**, 54 min, torn down with proof. Box: one RTX 5090 (32607 MiB, driver 595.71.05) on an Intel Xeon Platinum 8347C host, 251 GB; e4b `89e7342f`, grouped-nf4-gemm 0.32.1, torch 2.8.0+cu128, triton 3.4.0, transformers 5.16.1. Code under test: [#651](https://github.com/pjordanandrsn/experts4bit-qlora/pull/651) (`Int4Linear.fuse`; `fuse_qkv` takes the int4 store).

Public evidence in this repository: [`receipts/`](receipts) — the eight timed step receipts, the four first-draw censuses, `summary.txt`, `forensics.txt`, `versions.txt`, and the series arm's failing log. Everything below is produced by [`p54_reduce.py`](p54_reduce.py) from those files.

**One sentence:** the fusion is worth **0.52 ms/step (12.4 %) at B=1 with byte-identical output**, and **0.22 ms/step (2.0 %) at B=16 where it changes the arithmetic** — so it ships as a B=1 lever and stays opt-in at B=16 until the divergence is bounded by a quality instrument.

**STOP-1** -- control first draw 11.443 ms vs K16 P5's 11.19 ms on its box: +2.3 % (within +-8 %; ratios within this box are the position either way).

## B=16: `int4_b16` vs `int4_b16_fqkv`

| arm | draw 1 (ms) | draw 2 (ms) | median |
|---|---|---|---|
| control | 11.443 | 11.396 | 11.420 |
| fused | 11.194 | 11.200 | 11.197 |

**Saving (control median - fused median): 0.223 ms/step** -> **SMALLER THAN PREDICTED, REAL (saving 0.223 in [0.1, 0.25))**

| kernel family | control ms/step | fused ms/step | rel | control calls/step | fused calls/step |
|---|---|---|---|---|---|
| k16 small-M GEMM | 1.291 | 0.821 | -36.4 % | 192 | 96 |
| int4 expert GEMV | 6.261 | 6.386 | +2.0 % | 96 | 96 |
| bf16 GEMM (cutlass/cublas) | 0.564 | 0.573 | +1.5 % | 97 | 97 |
| fp8 paged decode | 0.686 | 0.703 | +2.5 % | 48 | 48 |
| index / gather / scatter | 0.964 | 0.986 | +2.4 % | 241 | 241 |
| activation quant | 0.308 | 0.318 | +3.4 % | 96 | 96 |
| split-K reduce | 0.368 | 0.385 | +4.6 % | 145 | 145 |
| rope/norm folds | 0.268 | 0.276 | +3.2 % | 193 | 193 |

P3 calls (control, fused): (192.0, 96.0) -> OK; other families moved > 5 %: none

<details><summary>top_control (top 10 kernels)</summary>

| kernel | ms/step | calls/step |
|---|---|---|
| `_gemv_int4_b32` | 6.261 | 96 |
| `_gemm_int4_b32_smallm` | 1.291 | 192 |
| `_fp8_paged_decode_split_f8dot` | 0.686 | 48 |
| `void at::native::(anonymous namespace)::indexSelectS...` | 0.474 | 48 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_...` | 0.383 | 1 |
| `_reduce_partials` | 0.308 | 96 |
| `_quant_x_rows` | 0.308 | 96 |
| `void at::native::indexFuncSmallIndex<int, long, unsi...` | 0.207 | 48 |
| `_fp8_append_bt1_side` | 0.172 | 96 |
| `void at::native::(anonymous namespace)::indexSelectS...` | 0.150 | 48 |

</details>

<details><summary>top_fused (top 10 kernels)</summary>

| kernel | ms/step | calls/step |
|---|---|---|
| `_gemv_int4_b32` | 6.386 | 96 |
| `_gemm_int4_b32_smallm` | 0.821 | 96 |
| `_fp8_paged_decode_split_f8dot` | 0.703 | 48 |
| `void at::native::(anonymous namespace)::indexSelectS...` | 0.485 | 48 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_...` | 0.388 | 1 |
| `_reduce_partials` | 0.324 | 96 |
| `_quant_x_rows` | 0.318 | 96 |
| `void at::native::elementwise_kernel<128, 4, at::nati...` | 0.232 | 144 |
| `void at::native::indexFuncSmallIndex<int, long, unsi...` | 0.212 | 48 |
| `_fp8_append_bt1_side` | 0.176 | 96 |

</details>

**P5 (token parity)** -- generated tokens, fused vs control:

- draw 1: **14 of 16 sequences diverge**; first divergence (seq, index): [('1', 59), ('2', 117), ('3', 21), ('5', 8), ('6', 2), ('7', 16), ('8', 31), ('9', 53)]
- draw 2: **14 of 16 sequences diverge**; first divergence (seq, index): [('1', 59), ('2', 117), ('3', 21), ('5', 8), ('6', 2), ('7', 16), ('8', 31), ('9', 53)]
- A/A (control draw 1 vs draw 2): **identical** on all 16 sequence(s)

## B=1: `int4_b1` vs `int4_b1_fqkv`

| arm | draw 1 (ms) | draw 2 (ms) | median |
|---|---|---|---|
| control | 4.209 | 4.210 | 4.210 |
| fused | 3.693 | 3.694 | 3.693 |

**Saving (control median - fused median): 0.516 ms/step** -> **OVER THE BAND (0.516 > 0.45) -- read the census before believing it**

| kernel family | control ms/step | fused ms/step | rel | control calls/step | fused calls/step |
|---|---|---|---|---|---|
| k16 small-M GEMM | 0.000 | 0.000 | -- | 0 | 0 |
| int4 expert GEMV | 1.669 | 1.402 | -16.0 % | 288 | 192 |
| bf16 GEMM (cutlass/cublas) | 0.492 | 0.491 | -0.2 % | 49 | 49 |
| fp8 paged decode | 0.333 | 0.334 | +0.5 % | 48 | 48 |
| index / gather / scatter | 0.293 | 0.293 | -0.1 % | 193 | 193 |
| activation quant | 0.225 | 0.153 | -31.9 % | 288 | 192 |
| split-K reduce | 0.476 | 0.305 | -35.9 % | 289 | 193 |
| rope/norm folds | 0.235 | 0.236 | +0.1 % | 193 | 193 |

P3 calls (control, fused): (288.0, 192.0) -> OK; other families moved > 5 %: ['activation quant', 'split-K reduce']

<details><summary>top_control (top 10 kernels)</summary>

| kernel | ms/step | calls/step |
|---|---|---|
| `_gemv_int4_b32` | 1.669 | 288 |
| `_reduce_partials` | 0.471 | 288 |
| `std::enable_if<!(false), void>::type internal::gemvx...` | 0.378 | 1 |
| `_fp8_paged_decode_split_f8dot` | 0.333 | 48 |
| `_quant_x_rows` | 0.225 | 288 |
| `void at::native::(anonymous namespace)::indexSelectS...` | 0.136 | 49 |
| `_router_epilogue` | 0.131 | 48 |
| `_fp8_append_t1_side` | 0.130 | 96 |
| `std::enable_if<!(false), void>::type internal::gemvx...` | 0.113 | 48 |
| `_rope_norm_heads` | 0.108 | 96 |

</details>

<details><summary>top_fused (top 10 kernels)</summary>

| kernel | ms/step | calls/step |
|---|---|---|
| `_gemv_int4_b32` | 1.402 | 192 |
| `std::enable_if<!(false), void>::type internal::gemvx...` | 0.378 | 1 |
| `_fp8_paged_decode_split_f8dot` | 0.334 | 48 |
| `_reduce_partials` | 0.300 | 192 |
| `_quant_x_rows` | 0.153 | 192 |
| `void at::native::(anonymous namespace)::indexSelectS...` | 0.136 | 49 |
| `_router_epilogue` | 0.130 | 48 |
| `_fp8_append_t1_side` | 0.128 | 96 |
| `std::enable_if<!(false), void>::type internal::gemvx...` | 0.113 | 48 |
| `_rope_norm_heads` | 0.107 | 96 |

</details>

**P5 (token parity)** -- generated tokens, fused vs control:

- draw 1: **identical** on all 1 sequence(s)
- draw 2: **identical** on all 1 sequence(s)
- A/A (control draw 1 vs draw 2): **identical** on all 1 sequence(s)

## P4: distinct experts per layer per decode step (B=16, `int4_b16_series`, untimed)

Series file MISSING -- P4 unread.

---

# The reading, against the bands registered before the run

## P1 — B=16: **smaller than predicted, real**

Registered band 0.25–0.50 ms/step, refuted under 0.10. Measured **0.223 ms** (11.420 → 11.197, 2.0 %; 1401 → 1429 tok/s). That lands in the zone the pre-registration named in advance — "between 0.10 and 0.25 is *smaller than predicted, real*, stated as such" — so it is stated as such. The two draws of each arm are 0.047 and 0.006 ms apart, so the effect is an order of magnitude outside the A/A spread.

**Why it is smaller than the kernel arithmetic implied.** The K16 row itself fell **0.470 ms** (1.291 → 0.821 ms/step, 192 → 96 calls) — close to the 0.35 ms the microbench predicted and then some. The *step* fell half that. Under graph replay the removed launches were partly overlapped with neighbouring kernels, so at B=16 the launch count is not the wall. This is the same lesson the B=16 host-bound census taught, one level down: a kernel-row saving is an upper bound on a step saving, never the step saving.

## P2 — B=1: **over the band, and the census says why**

Registered band 0.20–0.45. Measured **0.516 ms** (4.210 → 3.693, 12.4 %; 237.5 → 270.8 tok/s). The pre-registration said an over-band reading means *"something other than launch count moved — read the census before believing it."* Read:

| family | control | fused | Δ | calls |
|---|---|---|---|---|
| int4 expert GEMV (`_gemv_int4_b32`) | 1.669 | 1.402 | **−0.267** | 288 → 192 |
| split-K reduce (`_reduce_partials`) | 0.476 | 0.305 | **−0.171** | 289 → 193 |
| activation quant (`_quant_x_rows`) | 0.225 | 0.153 | **−0.072** | 288 → 192 |
| everything else | | | ±0.001 | unchanged |

Sum of the three: **0.510 ms**, against a measured step change of 0.516. It *is* launch count — three launches per removed projection, not one. At B=1 every attention projection served by the int4 GEMV carries its own activation-quantise and its own split-K reduce; fusing q/k/v into one projection removes 96 GEMV calls **and** their 96 quantise and 96 reduce calls. Nothing else moved, and the whole step is accounted for.

## P3 — **holds at B=16, refuted as written at B=1**

P3 predicted the call counts *and* that "every other family [is] within ±5 % of its control row".

- **B=16: HOLDS.** K16 192 → 96 calls exactly; the largest other move is split-K reduce at +4.6 %, everything else ≤ +3.4 %.
- **B=1: the call-count half holds exactly** (GEMV 288 → 192, −96 as registered) **and the "every other family" half is REFUTED** — activation quant −31.9 % and split-K reduce −35.9 %, both far outside ±5 %.

**This is my error, and it is worth naming.** I wrote P3 from the B=16 K16 path, where the small-M GEMM has a fused split-K and no separate activation quantise, and carried the same "only the target family moves" sentence to B=1 without checking the B=1 route — where the companion launches are structurally inseparable from the GEMV call they serve. The families are not independent, and the code says so plainly (`hot_residency.py`: each `_mm` call quantises its rows and the GEMV reduces its partials). A prediction that a mechanism is *isolated* must be checked against the code that implements it, not inherited from a neighbouring batch size. The refutation costs nothing here — the movement is in the favourable direction and fully explains P2 — but it would have hidden a real surprise had it gone the other way.

## P5 — **holds at B=1, REFUTED at B=16, and this is the finding that decides what ships**

| pair | verdict |
|---|---|
| B=1, fused vs control, draw 1 | **identical**, 1/1 sequences |
| B=1, fused vs control, draw 2 | **identical** |
| B=16, fused vs control, draw 1 | **14 of 16 sequences diverge**; first divergence at token 2, 8, 16, 21, 31, 53, 59, 117 … |
| B=16, fused vs control, draw 2 | **identical divergence pattern** — same sequences, same first indices |
| B=16, control draw 1 vs draw 2 (A/A) | **identical**, 16/16 |

The A/A is what makes this readable: the configuration is bit-deterministic on this box, so the fused arm is computing a **different function**, reproducibly, at B=16 — and not at B=1.

**The mechanism, stated as the leading hypothesis and not as a finding.** `Int4Linear.fuse` changes no byte — the packed rows and scales are the parts' own, asserted by `tests/test_int4_attn.py`. What changes is the *shape the kernel sees*: at B=1 the int4 GEMV computes an independent dot product per output row, so concatenating rows along N cannot change any row's arithmetic (hence bit-identical). At B=16 the K16 small-M GEMM tiles and split-K-reduces over an N of 5120 instead of three launches over 4096 / 512 / 512, so the fp32 accumulation order per output column differs. That is reorder-class noise by construction, but at 16 rows × 48 layers × greedy decode it flips tokens, and **"reorder-class by construction" is an argument, not a measurement**.

A second candidate has to be named because it is not excluded by this lane: with `--fuse-qkv` the round-2 glue takes its `qkv_proj` path (`glue_r2._patch_attention`) while the control takes the unfused-attention fold — a different rotary/norm chain. Separating the two candidates needs an arm this lane did not run (fused projections with the round-2 fold forced off).

## P4 — **NOT RUN: the arm was unbuildable as I registered it**

The untimed distinct-expert arm asked for `--amort on` at B=16 with the graph loop. The harness refuses that combination, in `step_decomp.py` itself:

```
AssertionError: amort-armed runs keep the baseline dispatch path (not capturable)
```

`--series-out` requires `--amort on`; `--amort on` is incompatible with the captured B>1 stage that produces a comparable step. So the arm as written could not exist, the lane exited on my own `SERIES MISSING` guard (rc 45), and **#564's distinct-expert count is still unmeasured**. The guard did its job — the run is recorded `HARNESS_ERROR / fail` rather than reported as a success with a silent hole.

This is the second time in three lanes that I registered an instrument without building it first (P52's K8 gate on Gemma-4 was the first). The rule that follows is narrower and cheaper than "be careful": **an untimed diagnostic arm must be dry-run against the harness's own argument checks before the pre-registration is merged** — `step_decomp.py --help` plus the assertion it would hit, on CPU, costs nothing. The replacement measurement is registered in the amendment below.

## The decision rule, applied

The pre-registration said: *P1 ∧ P3 ∧ P5 hold → `--fuse-qkv` becomes the int4 lanes' default*. **P5 fails at B=16, so no default ships there.** Applying it as written:

- **B=1: the fusion is licensed as a lever.** Token-identical, 0.516 ms/step (12.4 %) on this box, the mechanism fully accounted for in the census. `Int4Linear.fuse` is what a B=1 serving stack should use.
- **B=16: no default, no quoted position.** The saving is real (0.223 ms) and the mechanism is understood, but the stack computes different tokens and this lane carries no quality instrument. `--fuse-qkv` stays opt-in on the int4 lanes until a KL-from-checkpoint or K8 two-text read bounds the divergence. A 2 % speed-up bought with an unbounded arithmetic change is exactly the trade this repository does not make silently.
- **No cross-box number is quoted.** The B=1 gap to vLLM 0.28.0 `graph_r1` (3.497 ms on P37's box) would close from ×1.20 to ×1.06 if the two boxes were comparable. They are not measured to be, so that sentence is an observation about two different hosts and not a position.

## Amendment 1 (registered after the run, before the replacement measurement)

1. **P4's replacement.** The distinct-expert count is read from a **B=1-stage run with `--amort on`** (the b1d stage takes it; only the bv3 stage refuses) at batch 16 prompts, or from `--profile-out`'s per-expert `tokens_routed` histogram, which `step_decomp` writes under the same `--amort on` requirement without needing the captured loop. Whichever is used, its step time is not quoted. To be run in the next lane that rents this model, not on a box of its own.
2. **P5's follow-up, and the gate on any B=16 default.** One arm, same box: fused projections with the round-2 glue forced to the unfused chain, to separate the kernel-shape hypothesis from the glue-path hypothesis; plus a KL-from-checkpoint read (`bench/p44/kl_serve.py`) of the fused stack against the control. The registered bar is the one already on `main` for shipped configurations: **≤ 0.10 nats and top-1 ≥ 0.93**. No B=16 default before that passes.

## Caveats a reader should carry

1. **One box, one prompt set, one model.** Both pairs are internally comparable (same box, same session, same install); nothing here transfers to another host, and the 5090 class carries ~8.5 % inter-box dispersion.
2. **The censuses are first draws only.** The timed medians use both draws; the kernel attribution uses one. A/A on the timed side says the configuration is stable, but a per-kernel A/A was not run.
3. **The B=16 arithmetic difference is unbounded by this lane.** It is reported as a token divergence, which is the crudest possible instrument — it says *different*, not *worse*. Nothing in this page licenses a claim in either direction about quality at B=16.
