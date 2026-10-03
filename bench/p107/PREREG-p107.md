# P107 — Paged prefill attention's route A/B on one RTX 5090: `math` (today: SDPA's fp32 math backend) against `flash` (the same lower-right causal mask as a bias the flash kernel takes), on one engine, gated by the calibrated K8 rule on the served-prefill NLL (registered 2026-10-03, before any run)

Issue: experts4bit-qlora#960, which claimed the lane number when it was opened. Follows P102 (#931). Code under test:
`E4B_PAGED_PREFILL_ATTN` (#963).

**Why.**
- P102 read `DEFAULT=k19` and the int4 prefill loop is gone (#937). Its descriptive profile of one 4096-token k19
  prefill (`p102-5090-6`, `ttft_routes.json` `profile_4096`) has 924 ms of device time in a 1.51 s wall. The largest
  items are the shape of prefill attention computed in fp32:

  | item | device ms | launches |
  |---|---:|---:|
  | fp32 SIMT SGEMMs (`cutlass_80_simt_sgemm_128x32`, `tn` + `nn`), one pair per layer per chunk | 231 | 768 |
  | fp32 elementwise around them (`where`, `add`, `isneginf`) | 182 | 1,152 |
  | fp32 softmax | ≥ 31 | ≥ 96 |
  | for scale: K19, the experts | 143 | 768 |

- **The cause, confirmed at $0** on the A2000 under torch 2.8.0+cu128, the build P102's 5090 box ran. For a layer
  without sinks or a sliding window, `paged_attention`'s prefill branch calls SDPA with an explicit boolean lower-right
  mask and `enable_gqa=True` (Hq 32 against Hkv 4). Flash attention refuses a non-null mask, memory-efficient attention
  refuses mismatched head counts, and cuDNN attention is runtime-disabled. So the call lands on the math backend, which
  upcasts bf16 to fp32: bf16 reduction in it is disallowed and TF32 is off, hence SIMT SGEMMs.
- **#963 added `E4B_PAGED_PREFILL_ATTN`**, read at every call, with the default unchanged (`math`). `flash` passes the
  same mask as `torch.nn.attention.bias.causal_lower_right(T, t_total)` with `enable_gqa`, which reaches
  `pytorch_flash::flash_fwd_kernel`. Layers with sinks or a sliding window keep the explicit mask under either value.
  On the A2000, at Qwen3's shapes with 512 query rows:

  | context | math | flash | error vs fp64 (math / flash) |
  |---:|---:|---:|---|
  | 512 | 2.64 ms | 0.18 ms | 0.0016 / 0.0020 |
  | 2048 | 9.27 ms | 0.68 ms | 0.0017 / 0.0023 |
  | 4096 | 19.13 ms | 1.31 ms | 0.0017 / 0.0023 |

- **The saving to expect:** the items above total about 445 ms of the 924, and flash serves each call about 14×
  faster. So `flash` should remove roughly 400 ms of device time from a 4096-token prefill whose TTFT was 1.37 s.
- **Why a new quality instrument.** Neither existing K8 reading scores a logit the changed code produces:
  - the decode-shaped K8 prefills its 512-token prompt through this branch, but scores 2,048 positions that are all
    produced by DECODE attention (T == 1);
  - P102's prefill-shaped NLL (`step_decomp --ppl-oracle eager`) runs the model's own attention, with paged attention
    never registered.

  This lane scores the SERVED prefill instead: every scored logit comes out of the paged prefill branch under test, in
  all 48 layers, over 512-token chunks with staged K/V.
- **This lane** decides the default, reading speed and quality for both routes on the same engine, with the rule fixed
  in advance.
- **Not touched:** lane SC1's registered box runs.

Rule: the owner's standing no-ask tier for a single run under $15 (2026-09-26), with the usual mechanics:
- this page merged before the launch;
- receipts and ledger rows;
- proven teardown.

The guard is 1 h, so there is no proving rental. The premise runs first on the box, before anything is fetched.

## The lane

- **Box and software.** P102's: one RTX 5090.
  - e4b at the launch commit, which carries #963 (and #937's `E4B_INT4_PREFILL=auto`, so k19 under both routes);
    grouped-nf4-gemm `34da93d`; transformers 5.17.0; the image's torch, recorded.
  - Qwen3-30B-A3B @`ad44e77`, by pinned revision.
  - The NF4 arena is baked on the box by P98's `p98_bake.py`.
  - SC1's prompt rows, refused (rc 19) unless their digests are SC1 box B's.
  - The stack is SC1's `int4_sched` (`SPEEDENV`, the three folds, SC1's four route knobs). `E4B_PAGED_PREFILL_ATTN`
    and `E4B_INT4_PREFILL` start unset.
- **The tripwire** (rc 9) asserts:
  - the commits, and transformers 5.17.0;
  - pytest is importable (P102 A1);
  - `paged_attention.PREFILL_ATTN_ROUTES == ("math", "flash")`, and the default reads `math` at the launch commit;
  - `causal_lower_right` is importable;
  - the int4 prefill route reads `k19`;
  - `fp8_paged_attn` is importable.
- **The premise** (rc 25, before the fetch): `tests/test_paged_prefill_attn_route_gpu.py` on this card, 1 passed, none
  skipped. At Qwen3's attention shapes in bf16, over three 512-token chunks through `paged_attention_forward`: `flash`
  launches the flash kernel on every chunk, `math` launches none, and the two agree within 1 % relative Frobenius.
- **Arm `ab`** (`p107_box.py ab`) runs ONE engine: `serve_paged.build_engine`, all-VRAM, `max_seqs` 1, chunk 512,
  graphs on with bucket 1, fused q/k/v. The box sets `E4B_PAGED_PREFILL_ATTN` between requests.
  1. **TTFT.** Each route is warmed once at 512 and at 4096 tokens. Then three rounds; round r runs the routes in an
     order rotated by r, each with one timed `max_new_tokens=1` request at 512 tokens and one at 4096
     (`sc1_e4b_sched.run_batch`). Each request's first token is recorded.
  2. **Engagement.** Per route, one 512-token request under `torch.profiler`: device kernels whose names contain
     `flash`, and fp32 SGEMMs.
  3. **Quality, the served-prefill NLL.**
     - **Windows:** P96's layout. Window k starts at token k × 4096, span 2600, a 512-token prompt then 2,048 scored
       tokens: c4val1 k = 9…16 (W = 8) and wikitext k = 9…12 (W = 4). They are the windows P102 scored, fresh relative
       to P95's calibration; P102 read them under the model's eager attention, so no math-against-flash reading exists
       on them.
     - **Scoring** (`served_prefill_nll`): the window's first 2,560 tokens go through the runner's model in 512-token
       chunks, the way `PagedModelRunner.run_prefill` sends them: its mode switch, the paged context in PREFILL mode,
       `use_cache=False`. The 2,048 next-token predictions after the prompt are scored. The staging is dropped after the
       window instead of flushed, so nothing is written to the KV pool.
     - Within a window the routes run in an order rotated by the window's index.
  4. **Descriptive.** Per route, one 4096-token request under `torch.profiler`: the device time by kernel.
- **The order:** install and tripwire; premise; fetch; bake; prompts; `ab`; reduce.

**The reducer** (`p107_reduce.py`, 7-case self-test).

- **VOID** if any of these holds (each reason listed):
  - a rehearsal knob is set;
  - the record is missing or off the registered shape. That covers: the build's census (48 int4 expert layers on
    `int4_b32`); chunk 512 with three rounds; SC1's prompt digests; every route in every round at both lengths; and in
    the NLL, both routes on every window, scoring the same text for 2,048 steps;
  - a route did not run as registered in its 512-token kernel census:
    - **`flash`:** at least 48 flash kernels (one per layer for the one chunk);
    - **`math`:** no flash kernel, and at least 96 fp32 SGEMMs (the QK^T and PV pair per layer, as P102 profiled).
- **QUALITY**, `flash` against `math`, by the calibrated K8 rule: P95's windowed reading. Per text, take the mean over
  its windows of ppl(flash) − ppl(math). `flash` PASSES iff |mean| ≤ 0.05 on both texts (`k8_gate.verdict`,
  uncalibrated regime, since the RTN int4 store is uncalibrated). The per-window deltas, their SD, and the windows over
  0.05 are reported. Whether every timed draw's first token matches is reported, not gated.
- **SPEED:** each route's median TTFT over the three rounds, at 4096 and at 512 tokens.
- **DEFAULT:** `DEFAULT=flash` iff `flash` PASSES and its TTFT-4096 is at most 0.9 × `math`'s (a gain of at least
  10 %); otherwise `NO_CHANGE`.

**The registered consequence.**
- **`DEFAULT=flash`:** a PR makes `flash` `E4B_PAGED_PREFILL_ATTN`'s default, with `math` kept selectable. The PR
  carries a CHANGELOG entry and a `docs/SERVING.md` note citing this read, and the flip ships in the next release.
- **`NO_CHANGE`:** `math` stays the default, and #960 records the reading.
- **VOID:** fix the defect it names and redraw once.

## Predictions (written before the data)

- **`math`:** TTFT-4096 1.25–1.55 s and TTFT-512 0.10–0.13 s, if the host is in `p102-5090-6`'s class (P102's k19:
  1.373 / 0.113 s; on another host P102 read k19 at 3.4 s).
- **`flash`:** TTFT-4096 0.85–1.10 s, so 1.25–1.6× faster than `math`. TTFT-512 0.09–0.11 s: one chunk carries 1/36
  of a 4096 prefill's attention work, so little moves.
- **Engagement:** `flash` exactly 48 flash kernels per 512-token request; `math` none, and 96 fp32 SGEMMs.
- **Quality:** PASS, with |mean Δppl| ≤ 0.01 on both texts and no window over 0.05. Both routes accumulate in fp32;
  flash rounds the probabilities to bf16 before PV, which moved the per-call error against fp64 from 0.0017 to 0.0023
  on the A2000.
- **First tokens:** identical in every timed draw.
- **The verdict: `DEFAULT=flash`.**
- **The 4096-token profile under `flash`:** the SGEMM pair and its elementwise ops are gone, and K19 is the largest
  device-time item.

## Box and cost

- **`p107-5090-<n>`:** one RTX 5090 with ≥ 150 GB of disk (the runner refuses below 110 GB) and ≥ 64 GB of host RAM.
  **Guard 1 h at ≤ $0.75/h (≤ $0.75).**
- Estimated from `p102-5090-6` (23 min wall, $0.21):
  - acquisition 1–7 min;
  - install and premise ~2 min;
  - fetch ~9 min;
  - bake ~1 min;
  - prompts ~1 min;
  - `ab` ~6 min: one build of ~1.5 min; 20 TTFT requests at ≤ 1.5 s each; 24 window-routes, each a 2,560-token
    prefill plus the window's tokenisation; two profiled requests.
- **Lane ceiling $1.50; hard stop $2.00.**

## Rehearsal

- **The engine arm cannot run on the A2000.** The int4 store needs 16.3 GB, and the fp8 paged KV needs sm_89+.
- **What ran on a 5090 in P102:** the install, fetch, bake, prompt dump, stack, `build_engine`, `run_batch`, the
  per-request route switch and the kernel profiles. P107 changes which knob is switched.
- **On the A2000 ($0), at #963's head:**
  - the backend probe above, including which SDPA backends accept today's call;
  - the premise test passed: 26 passed with three neighbouring paged-attention suites, none skipped;
  - the CPU tests passed with CUDA hidden (25);
  - two mutations failed as they should: the flash branch removed, and a top-left causal bias.
- **The new piece, the scorer:** `tests/test_p107_served_prefill_scorer.py` (CPU, 3) runs it on a tiny Qwen3 at the
  registered shape (512 + 2,048, 512-token chunks).
  - For both routes, the chunked paged prefill's NLL equals one non-paged forward's within 1e-4.
  - It leaves no staging, no bound context and the runner's mode restored, on success and when a forward raises.
  - It ran on CPU in the A2000 container: 3 passed. Two mutations of the scorer each failed it: off-by-one targets,
    and positions restarting at each chunk.
- CI covers the self-tests, the scorer test, the staged pin and the driver's dry run.

Amendments, dated, go below this line before any data is read.
