# P149 (DRAFT): int8 weight-only attention projections for Mixtral-8x7B on `serve_paged`, a kernel premise and then the served read, on one RTX 5090

**Status: DRAFT, prospective.** This is $0 design. It is **not registered**: nothing here authorizes a box or spend. The
maintainer's RED ruling decides whether to register it, revise it or park it. **Mixtral-8x7B only.** Nothing carries over
from Qwen3-30B-A3B, whose attention parked at 1.26× its bound under the same quality rule (the W8A16 split premise).

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

### Phase 0, the engine ($0, code and CI before any box)

`experts4bit_qlora/engines/int8_attn.py`:
- **`Int8Linear`.** Weights int8 `[N, K]` with an fp32 per-row scale from `quantize_rows_int8`. Its bytes must be
  **bit-identical** to `i8_fakequant.roundtrip_`'s, so the served weights are exactly P147's I8ATTN weights.
- **Paths:**
  - M == 1: a bandwidth GEMV, P148's `gemv_i8` design re-shaped for 4096 × 4096 and 1024 × 4096, with an optional
    split-K under K33's split contract (fixed-order host sum).
  - 2 ≤ M ≤ 16: a small-M kernel that dequantizes int8 tiles to bf16 in registers and accumulates in fp32.
  - M > 16 (prefill): dequantize to a transient bf16 weight and call `F.linear`. No persistent bf16 copy is kept.
  - Buffers are preallocated at the swap, so CUDA-graph capture holds.
- **The lever.** `E4B_SERVE_ATTN_INT8=1` goes in `_apply_levers`, beside `E4B_SERVE_ATTN_INT4`. It refuses to combine
  with it, is recorded in `LEVER_ENV` and `info["levers_env"]`, and is off by default.
  - **Census refusal:** exactly the 128 attention projections on Mixtral are swapped, and the head is untouched.
  - The module keeps the `{q,k,v,o}_proj` child names, so R2's rope fold (`glue_r2._patch_attention_rope_only`) stays
    engaged. The census asserts it.
- **CI tests** (the interpreter, in their own process as P148's):
  - every path and config at ragged shapes and at Mixtral's shapes against an fp64 reference of the dequantized weights;
  - a one-hot activation reads one column exactly;
  - split-K agrees with unsplit;
  - the quantizer is bit-identical to `i8_fakequant`;
  - the lever's census and refusals on a tiny Mixtral.
- **The lab A2000**, $0 and correctness only: the suite compiled on a card, all passing, none skipped.

## The box (Phase 1): one RTX 5090, three stages, each gated

The order is fixed. The first stage that fails decides the verdict, and later stages do not run.

### 1. Host floors (before any install; refusals, not strikes)

- The card is a 170-SM RTX 5090. Anything else refuses (18).
- **POWER:** `nvidia-smi --query-gpu=power.limit` must read **≥ 575 W**, or the box refuses with **rc 17** (the launcher's
  host-limited power-cap code, P61's pattern) before the install. An unreadable limit refuses (15).
  - Every record carries `power.limit`, `clocks.max.sm` and the enforced limit.
  - At the pin, the host registry's caution machines are skipped (machine 40521 is marked for a 400 W cap).
  - A compute-sensitive read on a capped board measures the cap. P143 A3's FLIP and P145 both ran at 450 W, so this lane
    screens rather than records.

### 2. K, the kernel premise (P148's form, at Mixtral's shapes; falsify-first)

After the install, the tripwire, P136's premise tests and the kernel suite compiled on the card (all passing):
- **Contract:** per shape (4096 × 4096 and 1024 × 4096) at M == 1 and M == 16, the relative RMS error against fp64 of
  the dequantized weights must be ≤ 1.05 × cuBLAS bf16's against fp64 of the bf16 weights.
- **Select and read:** the same steps as P148.
  - Select the fastest contract-passing config.
  - Fresh graphs A (cuBLAS bf16 `F.linear`) and B (the choice), timed **A1 B1 B2 A2**: 20 warm and 200 timed replays,
    medians.
  - Every launch reads its own layer's weight, above L2.
  - **NOISY** if a self-pair (A2/A1 or B2/B1) is outside [0.985, 1.015].
- **The harness transfers:** at M == 1, each shape's cuBLAS medians lie within [0.8, 1.25] × P145's in-situ per-call
  time (q and o 1.4648 / 64 ms, k and v 0.5675 / 64 ms), and the profiled eager cuBLAS call runs P145's grid. Otherwise
  VOID.
- **K_GO** iff the W1 projected in-situ saving reaches the bar.
  - The saving per shape is the smaller of P145's in-situ time × (1 − B1/A1) and × (1 − B2/A2). It is summed over
    q, k, v and o, at 32 calls each.
  - The bar is **2 % of P145's W1 node period**, i.e. ≥ 0.1625 ms.
  - M == 16 is reported, never ruled.
- Otherwise **KERNEL_NO_GAIN**: a reading, not a strike. The box stops, and no served read follows.

### 3. Q, the quality re-check on the served model (term 2: before any speed term rules)

P147's rule, re-read on this box through `serve_paged`'s own model:
- **The model:** S0, the default server's model, and S1, the same with `E4B_SERVE_ATTN_INT8=1`.
- **The windows:** P147's check set and gates, bit-identical by sha256 to P147's: W1 has 196 windows at stride 1024, one
  per pass; W16 has 123, 16 per pass.
- **The forward:** teacher-forced prefill, which serves through the M > 16 path. NLL is read over positions 512..639
  exactly as in P147.
- **GO terms**, at **both** gates:
  - argmax agreement ≥ 0.95;
  - d + 1.645·SE ≤ ln(1 + 0.05 / ppl_S0), with d the mean per-window NLL shift S1 − S0 on this box and SE from the
    sample SD.
- **The decode path,** which prefill does not exercise: P143's SANE pass at T == 1, 12 wikitext windows, S1 against S0,
  |ΔNLL| ≤ 0.02 nats and argmax agreement ≥ 0.95.
- **Either fails → QUALITY_FAIL:** a PARK of the lever on this path. No speed term rules.

### 4. S, the served speed read (term 1: P143 Amendment 3's form, verbatim except the arms)

- **Arms:** S0 is the default server; S1 is `E4B_SERVE_ATTN_INT8=1`.
- **Eight fresh processes, ABBA twice:** S0a S1a S1b S0b S0c S1c S1d S0d.
- **The workloads:** P109's W16 and W1, 512-token wikitext prompts, SHORT 32 and LONG 160 new tokens, five timed reps
  after one untimed warm pass.
  - tok/s = B · (LONG − SHORT) / (min wall_LONG − min wall_SHORT), with `decode_ms_per_step` beside it.
  - The server is Mixtral's: 16 slots × 768 tokens, graphs on.
- **The rule** (the first rung that applies):
  1. VOID;
  2. **NOISY** (a VOID and a strike): any W1 self-pair outside [0.985, 1.015];
  3. **FUNCTION_FAIL:** an arm decodes tokens different from its own arm's first record, or its reps digest
     differently;
  4. **FLIP** iff g1 = the **minimum** of the four S1/S0 W1 pairs ≥ **1.03**, **and** g16 ≥ **0.99**, both compared
     unrounded;
  5. **NO_FLIP** otherwise, with each failed bar named.

**The verdict order:** host REFUSED (17 or 18) → VOID → NOISY → KERNEL_NO_GAIN → QUALITY_FAIL → FUNCTION_FAIL →
FLIP / NO_FLIP.

## What a FLIP starts (never automatic)

A separate PR makes `E4B_SERVE_ATTN_INT8` default on Mixtral only, citing this read. Nothing changes for any other family.
A NO_FLIP keeps the lever opt-in.

## Predictions (graded in the read, never ruling)

- **K:** K_GO. q and o's projected W1 saving is 0.45-0.70 ms (an int8 GEMV at 0.80-0.95 of the copy rate). k and v's is
  under 0.20 ms (256 programs and small rows). The M == 16 kernel is slower than cuBLAS at 1024 × 4096.
- **Q:** GO at both gates, with upper bounds within 1.5× P147's (0.0023 at W1).
- **S:** FLIP, with g1 between 1.04 and 1.08 and g16 between 0.99 and 1.03.

## The box and the run

- **Host:** one RTX 5090, Vast verified/secure, at most $0.85/h including the disk, under the market rule's pin (up to
  three stable machines, walked on `no_such_ask`).
  - Disk ≥ 200 GB (the 93.4 GB checkpoint and the venv), declared as `lane_record.min_disk_gb`.
  - The power floor is checked on the box (above). Caution machines are skipped at the pin.
- **Pins:** e4b at the kit's merge; grouped-nf4-gemm at the serve stack's pin; torch 2.8.0+cu128, triton 3.4.0,
  transformers 5.17.0, bitsandbytes 0.50.2; Mixtral at `eba92302` (P141's lock).
- **Guard:** 2.5 h.
  - Eight Mixtral loads at about 7 min each.
  - K at about 10 min.
  - Q at about 10 min (319 windows × 2 arms, resident experts).
  - The fetch.
- **Worst:** 2.5 h × $0.85 + 93.4 GB × $0.011/GB = **$3.15**, under the standing rule (at most $15).
- **Run ids:** `p149-5090-1`, with `-2` and `-3` as spares for pre-create refusals and host refusals (17 and 18 are not
  strikes). Shakedowns are `p149-dry-N` on the lab A2000 ($0, correctness only: the suite, the lever's census on a
  Mixtral slice, the quality pass's windows and shas).
- **Strikes:** two box-started HARNESS_ERROR, VOID or NOISY reads stop the lane for a simplification.

## Open questions for the maintainer's ruling

1. **K inside the same box, or its own box first?** Inside is cheaper: one fetch. A separate microbench box (P148's
   $0.32 form, no checkpoint) would fail faster and cheaper if the kernel is slow.
2. **The power floor at 575 W (refuse), or record-only like P143?** This draft refuses, per your term 3.
3. **The M > 16 path:** transient dequantization (no memory cost, a prefill cost) or a persistent bf16 copy (1.34 GB
   more on a card that already serves Mixtral at 768 tokens a slot)? This draft is transient. Prefill is not scored, but
   TTFT moves.
4. **Q's decode check:** P143's 12-window SANE at T == 1, or a larger decode-path sample?
