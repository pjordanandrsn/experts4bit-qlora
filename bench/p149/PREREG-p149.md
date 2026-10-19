# P149 (DRAFT): int8 weight-only attention projections for Mixtral-8x7B on `serve_paged`: a kernel premise, then quality at decode and the served read

**Status: DRAFT, revised per the maintainer's RED rulings (2026-10-19).** This is $0 design. It is **not registered**
until the maintainer's final pass after one independent review: nothing here authorizes a box or spend.
**Mixtral-8x7B only.** Nothing carries over from Qwen3-30B-A3B, whose attention parked at 1.26× its bound under the same quality rule (the W8A16 split premise).

## Why

P147 (`bench/p147/RESULTS-p147.md`) read **I8ATTN GO**. Per-output-row symmetric int8 rounding of Mixtral's 128
attention projections keeps the W8A16 split premise's gate at both gates. At W1 the upper bound is 0.002325 against the
bound 0.011560, and argmax agreement is 0.9813. NF4 parked. P147 rules that a GO starts nothing by itself: the next step
is a speed registration whose own box re-checks the rule on `serve_paged`. This is that registration.

**The ceiling** (P145, `bench/p145/RESULTS-p145.md`, one request on an RTX 5090 at 450 W):
- q and o take 1.4648 ms a step (64 calls, cuBLAS grid 1024 × 1 × 1) and already run at 0.98 of the copy rate. k and v
  take 0.5675 ms (64 calls, grid 256 × 1 × 1) at 0.63.
- Halving the bytes (int8 plus an fp32 row scale: 1,343,488,000 B a step against 2,684,354,560) saves **at most
  0.895 ms** of the 8.125 ms node period at P141's copy rate, and q and o alone at most 0.716 ms.
- These are bounds, not targets. No served gain has been measured.

## What exists, and what this lane must build first

- The only W8A16 kernel is P148's prototype (`bench/p148/p148_kernels.py`, `gemv_i8`). It handles M = 1 only, has no
  split-K, has never run on a card, and was only ever shaped for Qwen3's head (151936 × 2048).
- There is no int8 kernel for M = 2..16, which the W16 decode bucket needs.
- No served module exists. The template is `engines/int4_attn.py`: `Int4Linear` with a GEMV at one row, a small-M kernel
  at 2..16, preallocated buffers for capture, `attention_linears()` and the `E4B_SERVE_ATTN_INT4` lever in
  `_apply_levers`.

### Phase 0, the code ($0, before each box)

Kernels land in grouped-nf4-gemm first, and this package floors on the release that ships them (AGENTS.md). The K box
times prototypes, as P148 did. The library work starts only after K_GO, so a slow kernel costs no release.

**0a. The prototypes, before box K:** `bench/p149/p149_kernels.py`.
- **Kernels:**
  - **M == 1:** a bandwidth GEMV on P148's `gemv_i8` design, shaped for 4096 × 4096 and 1024 × 4096, with an optional
    split-K under K33's split contract (fixed-order host sum).
  - **2 ≤ M ≤ 16:** a small-M kernel that converts the int8 codes to bf16 in registers (exact, since |q| ≤ 127), runs
    `tl.dot` with fp32 accumulation, and applies the fp32 scale once at the end.
- **Weights:** int8 `[N, K]` with an fp32 per-row scale from the quantizer.
  - Its codes and scales are `i8_fakequant`'s, so their bf16 product equals P147's I8ATTN round trip **bit for bit**
    (tested).
  - **Both** kernels compute s · Σ q·x: an fp32 sum of exact codes times activations, scaled once by the fp32 scale.
    Neither rounds q·s to bf16, as the fake-quant stored it. That is the less lossy direction.
- **CI tests** (the interpreter, in their own process as P148's):
  - every config at ragged shapes and at Mixtral's shapes against an fp64 reference of the dequantized weights;
  - a one-hot activation reads one column exactly;
  - split-K agrees with unsplit;
  - the quantizer is bit-identical to `i8_fakequant`.
- **The lab A2000**, $0 and correctness only: the suite compiled on a card, all passing, none skipped.

**0b. The library and the engine, after K_GO and before box Q+S:**
- **The kernels move into grouped-nf4-gemm verbatim**, with K's chosen configs, in a release. The kernel source's sha256 is
  recorded beside the prototype's, so K's timing is the shipped code's.
- **The engine:** `experts4bit_qlora/engines/int8_attn.py`.
  - `Int8Linear` floors on that release.
  - **Paths:** the GEMV at M == 1, the small-M kernel at 2..16, and above 16 (prefill) a transient dequantization to
    bf16 plus `F.linear`, with no persistent bf16 copy.
  - Buffers are preallocated at the swap, so CUDA-graph capture holds.
  - **Engagement counters:** each path counts its calls at capture and in eager prefill. Q reads them.
- **The lever.** `E4B_SERVE_ATTN_INT8=1` goes in `_apply_levers`, beside `E4B_SERVE_ATTN_INT4`. It refuses to combine
  with it, is recorded in `LEVER_ENV` and `info["levers_env"]`, and is off by default.
  - **Census refusal:** exactly the 128 attention projections on Mixtral are swapped, and the head is untouched.
  - The module keeps the `{q,k,v,o}_proj` child names, so R2's rope fold (`glue_r2._patch_attention_rope_only`) stays
    engaged. The census asserts it.
- **Tests:** the lever's census, refusals and counters on a tiny Mixtral (CPU), and the suite compiled on the A2000.
- **Review:** the kernel PR and the engine PR each get their own review and the maintainer's final pass.

## Phase 1: two boxes, each gated (the maintainer's rulings, 2026-10-19)

The kernels have never run on a card, so the kernel premise runs first on its own cheap box (P148's form, with no
checkpoint). The quality-and-speed box is created only after K_GO. The first rung that fails decides, and later rungs do
not run.

### Host floors, on both boxes (before any install; refusals, not strikes)

- The card is a 170-SM RTX 5090. Anything else refuses (18).
- **POWER:** `nvidia-smi --query-gpu=power.limit` must read **≥ 575 W**, or the box refuses with **rc 17** (the launcher's
  host-limited power-cap code, P61's pattern) before the install. An unreadable limit refuses (15). Within-host ratios
  partly cancel a cap, but nothing else would carry that confound into a FLIP, and a host refusal costs cents.
  - Every record carries `power.limit`, `clocks.max.sm` and the enforced limit.
  - At the pin, the host registry's caution machines are skipped (machine 40521 is marked for a 400 W cap).

### Box K (`p149k-5090-N`): the kernel premise, P148's form at Mixtral's shapes

One no-model process. After the install, the tripwire and the kernel suite compiled on the card (all passing, none
skipped):
- **Contract:** per shape (4096 × 4096 and 1024 × 4096) at M == 1 and M == 16, the relative RMS error against fp64 of
  the dequantized weights must be ≤ 1.05 × cuBLAS bf16's against fp64 of the bf16 weights.
- **Select and read:** the same steps as P148.
  - Select the fastest contract-passing config.
  - Fresh graphs A (cuBLAS bf16 `F.linear`) and B (the choice), timed **A1 B1 B2 A2**: 20 warm and 200 timed replays,
    medians.
  - Every launch reads its own layer's weight, above L2.
  - **NOISY** (a VOID and a strike) if a self-pair (A2/A1 or B2/B1) is outside [0.985, 1.015].
- **The harness transfers:** at M == 1, each shape's cuBLAS medians lie within [0.8, 1.25] × P145's in-situ per-call
  time (q and o 1.4648 / 64 ms, k and v 0.5675 / 64 ms), and the profiled eager cuBLAS call runs P145's grid. Otherwise
  VOID.
  - The band was set from P145's read at **450 W**, and this lane refuses below 575 W. A 575 W card moves a
    memory-bound GEMV little, and K_GO itself is a ratio (1 − B/A).
- **K_GO** iff the W1 projected in-situ saving reaches the bar.
  - The saving per shape is the smaller of P145's in-situ time × (1 − B1/A1) and × (1 − B2/A2). It is summed over
    q, k, v and o, at 32 calls each.
  - The bar is **2 % of P145's W1 node period**, i.e. ≥ 0.1625 ms.
  - M == 16 is reported, never ruled.
- Otherwise **KERNEL_NO_GAIN**: a reading, not a strike. No Q+S box is created.
- **The choice is pinned.** Each shape's chosen configs (M == 1 and M == 16) go into the engine's config table in the kit
  that the Q+S box runs. The Q+S box compiles and runs the kernel suite on its card for correctness, and never re-times K.

### Box Q+S (`p149-5090-N`, only after K_GO): quality at decode, then the served read

**Arms:** S0 is the default server. S1 is the same with `E4B_SERVE_ATTN_INT8=1`. The server is Mixtral's: 16 slots × 768
tokens, graphs on.

**Process order:** S0a, S1a, then the Q reduce, then S1b S0b S0c S1c S1d S0d. This is P143 Amendment 3's eight fresh
processes, ABBA twice.
- In S0a and S1a, the timed speed reps run first and Q runs after them in the same process. Q's work never precedes a
  timed rep **in the same process**. Across processes, S0a's 15-20 min of decode is followed only by S1a's load (about
  7 min) before S1a's timed reps, while S0a's own reps start cold.
  - **Heat is recorded:** the GPU temperature and SM clock at the start and end of every timed block, reported and never
    ruled.
  - **It is conservative for FLIP:** g1 is the **minimum** of the four S1/S0 pairs, so a slower, hotter S1a can only
    lower g1.
  - A self-pair that trips NOISY with a temperature gap of more than 10 °C between its two blocks is still NOISY. The
    record shows the gap, so the next order can add a cool-down if heat is the cause.
- A QUALITY_FAIL after S1a stops the box: the six remaining processes are not run.

#### Q, the quality rule ruled at decode (term 2: before any speed term rules)

P147's rule is re-read through `serve_paged`'s own KV path, at the kernels that make the speed:
- **The windows:** P147's exact windows, sha256-checked against P147's records.
- **The forward:** each window's first 512 tokens are prefilled. Then its tokens 512..639 are fed **teacher-forced at
  T == 1** through the served decode step with graphs on, and the NLL and argmax are read at positions 512..639, as in
  P147.
- **The gates:**
  - **W1:** 196 windows, one per pass at **B = 1**, served by the **M == 1 GEMV**;
  - **W16:** 123 windows, **16 per pass**, served by the **M == 16 small-M kernel**.
- **The rule, at both gates:** argmax agreement ≥ 0.95, and d + 1.645·SE ≤ ln(1 + 0.05 / ppl_S0). Here d is the mean
  per-window NLL shift S1 − S0 on this box and SE comes from the sample SD. A failure is **QUALITY_FAIL**: a PARK of the
  lever on this path, and no speed term rules.
- **ENGAGEMENT, asserted by the box:**
  - S1's captured graph for bucket 1 holds the M == 1 GEMV and no other int8 path, and is replayed for W1.
  - S1's captured graph for bucket 16 holds the M == 16 small-M kernel, and is replayed for W16.
  - The transient-dequant path runs only in prefill. Each path's call count is recorded at capture and in eager prefill.
  - A zero count on any required path, or a forbidden path's nonzero count, is a **VOID**.
  - S0's records show no int8 path at all.
- **Reported, never ruling:**
  - **prefill-Q**, the same rule teacher-forced through prefill (the M > 16 path), as a continuity row against P147;
  - **SANE**, P143's 12 windows at T == 1, as the function check;
  - **prefill TTFT** per arm.

#### S, the served speed read (term 1: P143 Amendment 3's form, verbatim except the arms)

- **The workloads:** P109's W16 and W1, 512-token wikitext prompts, SHORT 32 and LONG 160 new tokens, five timed reps
  after one untimed warm pass.
  - tok/s = B · (LONG − SHORT) / (min wall_LONG − min wall_SHORT), with `decode_ms_per_step` beside it.
- **The rule** (the first rung that applies):
  1. VOID;
  2. **NOISY** (a VOID and a strike): any W1 self-pair (S0b/S0a, S1b/S1a, S0d/S0c, S1d/S1c) outside [0.985, 1.015];
  3. **FUNCTION_FAIL:** an arm decodes tokens different from its own arm's first record, or its reps digest
     differently;
  4. **FLIP** iff g1 = the **minimum** of the four S1/S0 W1 pairs ≥ **1.03**, **and** g16 ≥ **0.99**, both compared
     unrounded;
  5. **NO_FLIP** otherwise, with each failed bar named.

**The verdict order:** host REFUSED (17, 18, or 15 for an unreadable power limit) → VOID → NOISY → KERNEL_NO_GAIN (box K) → QUALITY_FAIL →
FUNCTION_FAIL → FLIP / NO_FLIP.

## What a FLIP starts (never automatic)

A separate PR makes `E4B_SERVE_ATTN_INT8` default on Mixtral only, citing this read. Nothing changes for any other family.
A NO_FLIP keeps the lever opt-in.

## Predictions (graded in the read, never ruling)

- **K:** K_GO. q and o's projected W1 saving is 0.45-0.70 ms (an int8 GEMV at 0.80-0.95 of the copy rate). k and v's is
  under 0.20 ms (256 programs and small rows). The M == 16 kernel is slower than cuBLAS at 1024 × 4096.
- **Q (at decode):** GO at both gates, with upper bounds within 1.5× P147's (0.0023 at W1). Both kernels differ from the
  fake-quant's bf16-stored weights only by the unrounded fp32 scale and the summation order, the less lossy direction.
- **S:** FLIP, with g1 between 1.04 and 1.08 and g16 between 0.99 and 1.03.

## The boxes and the run

- **Host:** one RTX 5090 per box, Vast verified/secure, at most $0.85/h including the disk, under the market rule's pin
  (up to three stable machines, walked on `no_such_ask`). Caution machines are skipped at the pin, and the power floor is
  checked on the box.
- **Pins:** e4b at each kit's merge; torch 2.8.0+cu128, triton 3.4.0, transformers 5.17.0, bitsandbytes 0.50.2.
  grouped-nf4-gemm is at the serve stack's pin for box K, which stages the prototypes, and at the release that ships the
  int8 kernels for box Q+S. The Q+S box also pins Mixtral at `eba92302` (P141's lock).
- **Box K** (`p149k-5090-N`):
  - no checkpoint;
  - guard 0.5 h, disk ≥ 60 GB;
  - worst 0.5 h × $0.85 + 5 GB × $0.011/GB = **$0.48** (P148's form ran at about $0.03 actual).
- **Box Q+S** (`p149-5090-N`), only after K_GO:
  - disk ≥ 200 GB (the 93.4 GB checkpoint and the venv), declared as `lane_record.min_disk_gb`;
  - guard **3.0 h**: eight Mixtral loads at about 7 min, Q's decode in S0a and S1a at about 15-20 min, the speed reps,
    and the fetch;
  - worst 3.0 h × $0.85 + (93.4 GB checkpoint + 5 GB install) × $0.011/GB = **$3.63**.
- **Both** are under the standing rule (at most $15 each).
- **Run ids:** each box has `-1` as its reading and `-2` and `-3` as spares for pre-create refusals and host refusals (17
  and 18, and 15 for an unreadable power limit, are not strikes). Shakedowns are `p149-dry-N` on the lab A2000 ($0, correctness only: the suite compiled, the
  lever's census and engagement counters on a Mixtral slice, and Q's windows and shas).
- **Strikes:** two box-started HARNESS_ERROR, VOID or NOISY reads, across both boxes, stop the lane for a
  simplification.

## The rulings this draft carries (2026-10-19)

1. **K in its own box first**, P148's form, no checkpoint. The Q+S box only after K_GO; it runs the suite and never
   re-times K.
2. **Power:** refuse below 575 W (rc 17, before the install, not a strike). `power.limit`, `clocks.max.sm` and the enforced
   limit go in every record.
3. **M > 16:** transient dequantization. A persistent 1.34 GB bf16 copy would change the server's memory against S0, and S
   compares the arms at the same slots. Prefill TTFT is reported, never ruled.
4. **Q is ruled at decode,** through the served KV path with graphs on, at the kernels that make the speed, with
   engagement asserted. Prefill-Q is reported as continuity, and SANE stays the function check.
