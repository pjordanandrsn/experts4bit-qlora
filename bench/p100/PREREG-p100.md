# P100 — Where the paged prefill's time goes on the int4 expert store: TTFT against chunk size, and the per-expert loop counted, on one RTX 5090, with no code change (registered 2026-10-03, before any run)

Issue: experts4bit-qlora#916. Follows lane SC1's box B reading (#846).

**Why.**
- SC1 box B (`sc1b-5090-1`, adertha-receipts `cabef5a`, `sc1/ttft_e4b_{512,4096}.json`, e4b `3db414e`) served
  Qwen3-30B-A3B on one RTX 5090 through `serve_paged.build_engine` and `ContinuousScheduler`. The stack was all-VRAM,
  int4 experts, int4 attention and fused q/k/v, at `E4B_PAGED_CHUNK_TOKENS=512`, `max_seqs=1`.
- The TTFT of a `max_new_tokens=1` request was **2.08 s at 512 tokens and 16.6 s at 4096 tokens**: about 2.07 s per
  512-token chunk. An upstream serving engine on the same card took 0.17 s for 4096 tokens.
- A read-only code trace (#916) names a cause. With `max_seqs == 1`, `serve_paged.py:721` leaves
  `hot_residency.DEVICE_GROUPING` off. A T > 1 call on the int4 store therefore takes host grouping
  (`hot_residency.py:556-561`, a `counts.tolist()` sync), then the else-branch at `hot_residency.py:721-752`. That branch
  loops in Python over the routed experts (about 128 per layer). Each iteration calls grouped-nf4-gemm's pure-torch
  `dequant_int4_ref`, casts the weight to bf16, runs one small matmul and copies a slice.
- That would be about 12.3k iterations and about 170k kernel launches per chunk, repeated every chunk, with nothing
  cached. The trace has not been confirmed on a GPU.
- **This lane** confirms or refutes it with no code change. Two predictions separate the readings:
  - TTFT-4096 against chunk size. A fixed per-chunk cost predicts about 4.2 s at chunk 2048. A per-token cost
    predicts no change.
  - The loop itself, counted on the box: every T > 1 call, its grouping, and the `dequant_int4_ref` calls inside it.
    This includes `step_decomp.py --cprofile-out` at `--batch 1`.
- **Not touched:** lane SC1's registered box runs, which measure the engine as it is.

Rule: the owner's standing no-ask tier for a single run under $15 (2026-09-26), with the usual mechanics:
- this page merged before the launch;
- receipts and ledger rows;
- proven teardown.

The guard is 1 h, so there is no proving rental.

## The lane

- **Box and software.** One RTX 5090.
  - e4b at the launch commit; grouped-nf4-gemm `34da93d` (SC1 box B's); transformers 5.17.0.
  - Qwen3-30B-A3B @`ad44e77`, fetched by pinned revision. The NF4 arena is baked on the box by P98's
    `p98_bake.py` (revision-pinned).
  - The prompt rows are SC1's (`sc1_prompts.py`, `step_decomp._k8_window`, wikitext-2 test). The runner refuses
    (rc 19) unless `prompts_b1.json` and `prompts_b1_4096.json` carry SC1 box B's digests (`a8e6ea1`, `cd70a14`): the
    same token ids SC1 read.
- **The stack** is SC1's `int4_sched` arm, byte for byte:
  - `E4B_SERVE_EXP_INT4=1 E4B_SERVE_ATTN_INT4=1 E4B_SERVE_ATTN_INT4_CALIB=0 E4B_CALIB_SOURCE=c4` with the three folds;
  - the route knobs `E4B_INT4_GROUPED_SMALLM=auto E4B_INT4_LEAN_GLUE=auto E4B_NF4_GROUPED_SMALLM=0
    E4B_MXFP4_GROUPED_SMALLM=auto`;
  - the engine all-VRAM, `max_seqs=1`, graphs on with bucket 1, fused q/k/v, 8 torch threads.
- **The TTFT arms** are `bench/sc1/sc1_e4b_sched.py --ttft` at its registered bytes: one warm and three timed
  `max_new_tokens=1` requests, wall per request, median recorded. They run through `p100_box.py`, which:
  - keeps the engine `build_engine` returns;
  - after the arm has written its record, installs counters and runs ONE more request on the same row (the dispatch
    census);
  - in arm `c512_t512` only, runs one more request after that under `torch.profiler` (CUDA activity) to count device
    kernels.

  The timed requests run the library's functions unwrapped.
- **The dispatch census** wraps two module attributes for the census request: `hot_residency._fused_over_stack` and
  `int4_pack_ref.dequant_int4_ref`, both looked up at call time. For every MoE call it records:
  - the routed rows R and the distinct experts;
  - the grouping flags it was given;
  - whether an int4 store rode along;
  - the `dequant_int4_ref` calls made inside it.

  The library is not edited.
- **The prof arm** runs `bench/p39/step_decomp.py` at its registered bytes, through `p100_box.py prof`, with the counters
  on and snapshotted when step_decomp's cProfile window opens and closes. The flags are:
  - `--placement-override all-vram --amort off --batch 1 --prompt-len 512 --chunk 512 --gen-tokens 32 --fuse-qkv`;
  - `--cprofile-out cprofile_b1.txt`.

  The levers come from P42's hook on `PYTHONPATH`, the way SC1's step_decomp arms get them. step_decomp's listing holds
  the top 60 functions by cumulative time, so `dequant_int4_ref` may not appear in it. The census counts the same
  window either way.
- **Five arms, each a fresh engine in its own process, in this order:**

  | arm | what | chunks |
  |---|---|---|
  | `c512_t512` | TTFT-512 at chunk 512 (SC1's arm) + census + kernel count | 1 |
  | `c2048_t4096` | TTFT-4096 at chunk 2048 + census | 2 |
  | `prof` | step_decomp `--cprofile-out`, B=1, int4 experts + census | 1 (then 31 decode steps) |
  | `c512_t4096` | TTFT-4096 at chunk 512 (SC1's arm) + census | 8 |
  | `c1024_t4096` | TTFT-4096 at chunk 1024 + census | 4 |

- **The order:** install and tripwire; fetch; bake; prompt dump (digests checked); the arms above; reduce.
- **The tripwire** (rc 9) asserts:
  - the installed e4b and grouped-nf4-gemm are the launch and pinned commits;
  - `_fused_over_stack` still holds the loop line the trace read (`w = dequant_int4_ref(st["packed"][e_],`);
  - `serve_paged._batched_graph_grouping` still requires `max_seqs > 1`, and `DEVICE_GROUPING` starts `[False]`;
  - the user site is enabled (P42's hook needs it).

**The reducer** (`p100_reduce.py`, 13-case self-test).

- **VOID** if any of these holds (each reason listed):
  - a rehearsal knob is set;
  - a TTFT record is missing, not `ok`, or off the registered shape. The shape is: model and revision, B=1, the
    arm's chunk and prompt length, SC1's prompt digest, all-VRAM, bucket 1 captured, fused q/k/v, SC1's route knobs,
    48 int4 expert layers on the `int4_b32` store, and `device_grouping` False;
  - a census is missing;
  - the prof arm did not report `INT4EXP enabled: 48 layers`, has no cProfile window, or names `dequant_int4_ref` in its
    listing with an `ncalls` that differs from the census's count for the same window.
- **SCALING.** a1 = TTFT-512 at chunk 512. T8, T4, T2 = TTFT-4096 at chunk 512, 1024, 2048. rho = T8 / T2.
  - **CONFIRMED** iff rho ≥ 3.0, T2 ≤ 3 × a1, and T2 < T4 < T8;
  - **REFUTED** iff rho < 2.0;
  - **INDETERMINATE** otherwise.
- **MECHANISM** is **CONFIRMED** iff all of these hold, and **NOT_CONFIRMED** otherwise:
  - In `c512_t512`'s census there are 48 grouped (T > 1) calls, every one the host-grouped int4 loop, none
    device-grouped. They make ≥ 9,600 `dequant_int4_ref` calls (≥ 100 distinct experts per layer), exactly two per
    distinct expert per call.
  - Every TTFT census shows 48 × n_chunks loop calls, none device-grouped: the loop is paid per chunk, per layer.
  - In the prof arm's cProfile window there are 48 loop calls (one 512-token chunk) and ≥ 9,600 `dequant_int4_ref`
    calls, and none of those calls happens inside a T == 1 (singleton) call.
- **Verdict:**
  - **CONFIRMED** iff SCALING and MECHANISM are both CONFIRMED;
  - **REFUTED** iff SCALING is REFUTED or MECHANISM is NOT_CONFIRMED;
  - **INDETERMINATE** otherwise.
- **Reported beside it:**
  - the fit T(n) = alpha × n + beta over the three 4096-token arms (alpha = the per-chunk cost);
  - T2 / (2 × a1), and this box's c512 arms against SC1's;
  - the kernel count and its 12 most frequent kernels;
  - distinct experts per layer;
  - the cProfile's own `ncalls` for `dequant_int4_ref`, or that it is not in the top 60.

**The registered consequence.**
- **CONFIRMED:** register P101, the fix as an A/B against this path, at matched numerics where possible. The
  candidates are below; P101 writes its own rule. #916's comment and docstring corrections land with the fix.
- **REFUTED:** no fix lane against the loop. #916 records the reading. The census and the scaling say where the time
  is instead, and the next lane is written against that.
- **INDETERMINATE:** #916 records the reading. One redraw, with the arms at more reps, before any fix lane.
- **VOID:** fix the defect it names and redraw once.

## Predictions (written before the data)

- **Scaling CONFIRMED.**
  - a1 ≈ 2.1 s (SC1: 2.08);
  - T8 ≈ 16.6 s (SC1: 16.6);
  - T4 ≈ 8.3 s;
  - T2 ≈ 4.2 s;
  - rho ≈ 4.0, alpha ≈ 2.07 s per chunk, beta within ±0.5 s.
- **Mechanism CONFIRMED.**
  - 48 loop calls per chunk, with 4096 rows at chunk 512, 8192 at 1024 and 16384 at 2048;
  - 125–128 distinct experts per layer, so 12,000–12,288 `dequant_int4_ref` calls per chunk;
  - about 170k device kernels in the 512-token request (at least 150k), with the dequant's elementwise kernels the
    most frequent;
  - the prof window: 48 loop calls, about 12.2k dequant calls, 0 inside the 1,488 T == 1 calls (31 decode steps × 48
    layers);
  - `dequant_int4_ref` in step_decomp's top-60 listing, with `ncalls` equal to the census's window count.

## Candidates for P101 (named here so the A/B's design is not chosen after seeing this lane's numbers)

1. **Device-grouped prefill.** T > 1 calls on the int4 store take the existing device-grouped int4 M-tile GEMM
   (`gemm_int4_b32_grouped_captured`), whatever `max_seqs` is. This is the path B=16 decode serves today. Its
   activations are int8 (`quant_x_rows`), so the numerics change: the gate is the calibrated K8 rule (P95/P96's
   windowed reading) applied to a prefill-shaped NLL (`step_decomp --ppl-oracle eager`, which scores through T = 256
   chunks). The decode-shaped K8 scores at T == 1 and would not see a prefill-only change.
2. **Matched numerics.** Keep the bf16 matmuls, but decode each layer's routed experts in one batched pass instead of
   one `dequant_int4_ref` per expert. The bf16 weights are the same values, so the output should be bit-identical; an
   on-box bitwise check holds it to that.
3. **Caching every decoded weight is not a candidate.** All 48 layers' experts in bf16 are about 58 GB, against the
   card's 32 GB.

## Box and cost

- **`p100-5090-<n>`:** one RTX 5090 with ≥ 150 GB of disk (the runner refuses below 110 GB) and ≥ 64 GB of host RAM.
  **Guard 1 h at ≤ $0.75/h (≤ $0.75).** Estimated, from SC1 box B's logs:
  - install ~2 min;
  - fetch ~5 min (61 GB);
  - bake ~2–4 min;
  - prompts ~1 min;
  - arms ~15 min (each build ~130 s; the 4096-token arms at chunk 512 run four 16.6 s requests).

  The arms check the deadline before each one starts (STOP-2). The order puts the deciding arms (`c512_t512`,
  `c2048_t4096`, `prof`) first.
- **Lane ceiling $1.50; hard stop $2.00.**

## Rehearsal

The arms cannot run on the A2000: the int4 store needs 16.3 GB, and the fp8 paged KV needs sm_89 or newer. Every
piece except the census has run on a 5090:
- SC1 box B ran the install, the fetch, the prompt dump, the stack and `sc1_e4b_sched.py --ttft`;
- P98 ran `p98_bake.py`;
- SC1's window arms ran step_decomp with P42's hook.

The new pieces are `p100_box.py` (the census and the prof wrapper), the arm configurations (one env var, the chunk) and
the reducer. CI covers the self-tests, the staged pin and the driver's dry run.

**The census on the A2000 ($0, 2026-10-03, before registration).** The test container was torch 2.8.0+cu128, e4b at
this branch, and grouped-nf4-gemm `34da93d`. `p100_box.Census` wrapped the real `hot_residency._fused_over_stack`.
The inputs were synthetic int4 stores (random weights, `pack_int4_b32`) at Qwen3-30B-A3B's expert shapes (128 experts,
gate_up 1536×2048, down 2048×768), with skewed top-8 routing.
- **The census recorded the calls exactly.** One loop call: 4,096 rows, 128 distinct experts, 256 `dequant_int4_ref`
  calls (two per distinct expert). One T == 1 call through the singleton GEMV: 0.
- **The loop's per-layer cost does not move with rows.** It took 104 ms at 4,096 rows (T = 512) and 110 ms at 16,384
  rows (T = 2048). One call ran 3,341 device kernels and 263 copies, about 173k per 48-layer chunk.
- **This is a shape check, not the reading.** The A2000 is shared with live home services, and its host is the NAS's
  CPU, so these milliseconds are not a timing receipt. The 5090's per-chunk cost is what the lane reads.
- **Design input for P101 only; P100's rule does not read it.** On the same inputs, the existing device-grouped branch
  (`device_grouping=True`) took:
  - 8.3 ms at T = 512 and 38.8 ms at T = 2048;
  - 48 device kernels per call.

  Against an fp32 reference on the same int4 weights, its relative error was 0.0122 (T = 512), against the loop's
  0.0048. That is the int8 activations, as expected.

Amendments, dated, go below this line before any data is read.
