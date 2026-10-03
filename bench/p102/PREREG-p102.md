# P102 — The int4 store's prefill route A/B on one RTX 5090: `loop` (today) against `batched` (bit-identical), `k19` (the loop's operands) and `mtile` (int8 activations), on one engine, gated by the calibrated K8 rule on a prefill-shaped NLL (registered 2026-10-03, before any run)

Issue: experts4bit-qlora#916. Follows P100 (#920). Code under test: `E4B_INT4_PREFILL` (#921). The lane number is
P102 because P101 went to the hybrid-graphs lane (#919).

**Why.**
- P100 read REFUTED by its rule, but it located the time. On an RTX 5090 with a fast host (Ryzen 9 9950X3D), Qwen3-30B-A3B
  served by `serve_paged` with int4 experts at `max_seqs == 1` showed:
  - every prefill MoE call runs the per-expert host loop: 48 calls per chunk per layer;
  - the loop is 73 % of a 512-token chunk in the profile;
  - a chunk costs about 0.43 s fixed plus about 0.35 ms per token;
  - TTFT was 0.55 s at 512 tokens and 4.79 s at 4096 (chunk 512).

  SC1 box B's host read 2.08 / 16.6 s for the same configuration.
- #921 added `E4B_INT4_PREFILL`, read at every MoE call, with the default unchanged (`loop`):

  | route | what serves a host-grouped T > 1 call | numerics against loop |
  |---|---|---|
  | `loop` | one `dequant_int4_ref` per routed expert per projection, then a bf16 matmul | — |
  | `batched` | the same decode in slices of 16 experts, the same matmuls | bit-identical |
  | `k19` | device grouping, K19 at every row count (bf16 activations, int4 decoded in registers, bf16 MMA) | the same operands, a different summation order (within one bf16 ulp) |
  | `mtile` | device grouping, the grouped int4-b32 M-tile GEMM batched decode uses | int8 activations |

- **This lane** decides the default. It reads speed and quality for all four routes on the same engine and the same
  model, with the rule fixed in advance.
- **Not touched:** lane SC1's registered box runs.

Rule: the owner's standing no-ask tier for a single run under $15 (2026-09-26), with the usual mechanics:
- this page merged before the launch;
- receipts and ledger rows;
- proven teardown.

The guard is 1 h, so there is no proving rental. The premise runs first on the box, before anything is fetched.

## The lane

- **Box and software.** P100's: one RTX 5090.
  - e4b at the launch commit, which carries #921; grouped-nf4-gemm `34da93d`; transformers 5.17.0.
  - Qwen3-30B-A3B @`ad44e77`, by pinned revision.
  - The NF4 arena is baked on the box by P98's `p98_bake.py`.
  - SC1's prompt rows, refused (rc 19) unless their digests are SC1 box B's.
  - The stack is SC1's `int4_sched` (`SPEEDENV`, the three folds, SC1's four route knobs). `E4B_INT4_PREFILL` starts
    unset.
- **The tripwire** (rc 9) asserts:
  - the commits;
  - `hot_residency.INT4_PREFILL_ROUTES == ("loop", "batched", "k19", "mtile")`, and the default reads `loop` at the
    launch commit;
  - K19 is importable;
  - the loop line the trace read is present, and `DEVICE_GROUPING` starts `[False]`;
  - the user site is enabled.
- **The premise** (rc 25, before the fetch): `tests/test_int4_prefill_route_gpu.py` on this card, 2 passed, none
  skipped. At Qwen3's expert shapes and 4,096 / 16,384 rows: batched is bit-identical to loop; k19 is within 0.8 % of
  loop and mtile within 3 %; neither calls the reference decode.
- **Arm `ttft`** (`p102_box.py ttft`) runs ONE engine: `serve_paged.build_engine`, `max_seqs` 1, chunk 512, graphs on
  with bucket 1, fused q/k/v.
  - Each route is warmed once at 512 and at 4096 tokens.
  - Then three rounds. Round r runs the routes in an order rotated by r, each with one timed `max_new_tokens=1` request
    at 512 tokens and one at 4096. Each request's first token is recorded.
  - Then, per route, one 512-token request under P100's dispatch census, and one under `torch.profiler` (kernel names).
  - Then, for `k19` and `mtile`, one 4096-token request under `torch.profiler` for the device time by kernel. This is
    descriptive: where the time goes once the loop is gone.
- **Arm `nll`** (`p102_box.py nll`) runs `bench/p39/step_decomp.py` at its registered bytes with `--ppl-oracle eager`.
  - This is the prefill-shaped NLL SC1's `nll_e4b_prefill` rows use: K8's flags, the stack's levers through P42's
    hook, chunked teacher forcing through the model's own eager attention.
  - Its `_ppl_oracle_main` is replaced by a loop over every registered window and every route, on ONE model build.
  - **Windows:** fresh, P96's layout. Window k starts at token k × 4096, span 2600, a 512-token prompt then 2,048 scored
    tokens: c4val1 k = 9…16 (W = 8) and wikitext k = 9…12 (W = 4).
  - Within a window the routes run in an order rotated by the window's index.
  - **Every MoE call here has T ≥ 256** (the 512-token prompt, then 256-token chunks), so ≥ 2,048 rows, and every call
    takes the route under test. The decode-shaped K8 scores at T == 1, which no route changes, so it would not run the
    code it gates.
- **The order:** install and tripwire; premise; fetch; bake; prompts; `ttft`; `nll`; reduce.

**The reducer** (`p102_reduce.py`, 11-case self-test).

- **VOID** if any of these holds (each reason listed):
  - a rehearsal knob is set;
  - a record is missing or off the registered shape. That covers: the build's census (48 int4 expert layers on
    `int4_b32`, `device_grouping` False); chunk 512 with three rounds; SC1's prompt digests; and in the NLL, every route
    on every window, scoring the same text for 2,048 steps;
  - a route did not run as registered in its 512-token dispatch census:
    - **`loop`:** 48 grouped calls, all the loop, ≥ 9,600 reference decodes;
    - **`batched`:** 48 host-grouped calls on the int4 store, 0 reference decodes;
    - **`k19`:** 48 device-grouped calls, 0 reference decodes, ≥ 96 K19 launches and 0 M-tile launches;
    - **`mtile`:** 48 device-grouped calls, 0 reference decodes, ≥ 96 M-tile launches and 0 K19 launches.
- **QUALITY** against `loop`:
  - **`batched`** PASSES iff its `mean_nll` equals loop's exactly on all 12 windows AND its first token equals loop's in
    every timed draw. Otherwise it is QUALITY_FAIL.
  - **`k19` and `mtile`** are gated by the calibrated K8 rule: P95's windowed reading, on fresh windows sized from its
    measured spread (W ≥ 5 on c4val1, ≥ 2 on wikitext). Per text, take the mean over its windows of ppl(route) −
    ppl(loop). A route PASSES iff |mean| ≤ 0.05 on both texts (`k8_gate.verdict`, uncalibrated regime, since the RTN
    int4 store is uncalibrated). The per-window deltas, their SD, and the windows over 0.05 are reported.
- **SPEED:** each route's median TTFT over the three rounds, at 4096 and at 512 tokens.
- **DEFAULT:**
  - Among the routes that PASS, take the lowest TTFT-4096.
  - A PASSing route within 10 % of it with better numerics (batched, then k19, then mtile) is taken instead.
  - The verdict is `DEFAULT=<route>` iff that route's TTFT-4096 is at most half of loop's; otherwise `NO_CHANGE`.

**The registered consequence.**
- **`DEFAULT=<route>`:** a PR makes that route `E4B_INT4_PREFILL`'s default. The other routes stay selectable, `loop`
  included. The PR carries a CHANGELOG entry and a `docs/SERVING.md` note citing this read, and the flip ships in the
  next release.
- **`NO_CHANGE`:** `loop` stays the default, and #916 records the reading.
- **VOID:** fix the defect it names and redraw once.

## Predictions (written before the data)

- **`loop`:** TTFT-512 0.5–0.6 s and TTFT-4096 4.5–5.0 s, if the host is in P100's class (another host can differ by
  3.5×).
- **`batched`:** bit-identical, so it PASSES. TTFT-4096 within ±20 % of loop's. On a fast host the loop's fixed part
  looks bound by the bytes the decode moves (P100's estimate), and batched moves the same bytes.
- **`k19`:** PASSES, with |mean Δppl| ≤ 0.01 on both texts. TTFT-4096 0.3–1.5 s; TTFT-512 0.05–0.25 s.
- **`mtile`:** PASSES, with |mean Δppl| ≤ 0.05 (likely ≤ 0.03). TTFT-4096 0.3–2.0 s, at least 10 % slower than k19.
  The A2000 proxy was 39 vs 23 ms per layer at 16,384 rows.
- **The verdict: `DEFAULT=k19`.**
- **The 4096-token profile under k19:** the MoE GEMMs are no longer the largest device-time item. No prediction on
  which item is.

## Box and cost

- **`p102-5090-<n>`:** one RTX 5090 with ≥ 150 GB of disk (the runner refuses below 110 GB) and ≥ 64 GB of host RAM.
  **Guard 1 h at ≤ $0.75/h (≤ $0.75).**
- Estimated from P100's logs:
  - acquisition 1–7 min;
  - install and premise ~2 min;
  - fetch ~9 min;
  - bake ~1 min;
  - prompts ~1 min;
  - `ttft` ~4 min (one build of ~1.5 min, then about 40 requests, the loop's 4096 at ~4.8 s each);
  - `nll` ~5 min (one build, then 48 window-routes, the loop's at ~5 s each).
- **Lane ceiling $1.50; hard stop $2.00.**

## Rehearsal

- **The engine arms cannot run on the A2000.** The int4 store needs 16.3 GB, and `ttft`'s fp8 paged KV needs sm_89+.
- **What ran on a 5090 in P100:** the install, fetch, bake, prompt dump, stack, `build_engine`, `run_batch`, P100's
  census and step_decomp with P42's hook.
- **On the A2000, at #921's head:**
  - the premise test passed: 56 passed with five neighbouring suites, none skipped;
  - the CPU tests passed with CUDA hidden (25);
  - three mutations failed as they should.
- **The new pieces:**
  - `p102_box.py`: one engine with the route switched per request, and the `_ppl_oracle_main` loop. Its call path was
    read against step_decomp's `--ppl-oracle` flow: the model is built with the hook's lanes, paged attention is never
    registered, and `_ppl_oracle_main` is looked up by its global name at the call;
  - the reducer;
  - the arm configurations.

  CI covers the self-tests, the staged pin and the driver's dry run.

Amendments, dated, go below this line before any data is read.

### A1 (2026-10-03, before any data was read)

`p102-5090-1` (adertha-receipts `139544a`, $0.0303) stopped at the premise, rc 25, before anything was fetched:
`/opt/conda/bin/python: No module named pytest`. The runner was derived from P100's, whose install line has no pytest
(P100 ran no test on the box). P99's runner, which also runs a premise, installs it. The A2000 rehearsal of the premise
ran in a container where pytest had been installed by hand, so it could not catch this.

- **Changes:** the install line adds `pytest`, and the tripwire imports it (rc 9 before the premise, not rc 25 after).
  `tests/test_p102_staged_pin.py` adds a check that whatever the premise runs is installed and imported first.
- **Unchanged:** nothing the lane measures and nothing in its rule. No route was exercised, and no TTFT or NLL was
  recorded.
- The next attempt is `p102-5090-2`, at this amendment's merge commit.
