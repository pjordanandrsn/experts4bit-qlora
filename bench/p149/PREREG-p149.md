# P149: int8 weight-only attention projections for Mixtral-8x7B on `serve_paged`: a kernel premise, then quality at decode and the served read

**Status: registered 2026-10-19, before any data** (the maintainer's RED rulings and final pass, after one independent
review). Box K runs under the standing rule after Phase 0a's kit merges; box Q+S only after K_GO and Phase 0b.
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

**Process order:** S0a, S1a, then the Q reduce, then S1b S0b S0c S1c S1d S0d (Amendment 4 adds T1 after the Q reduce and
T2 after S1c). This is P143 Amendment 3's eight fresh
processes, ABBA twice.
- In S0a and S1a, the timed speed reps run first and Q runs after them in the same process. Q's work never precedes a
  timed rep **in the same process**. Across processes, S0a's 15-20 min of decode is followed only by S1a's load before
  S1a's timed reps, while S0a's own reps start cold. That load is short: P145 measured 23-27 s (`load_s`), so S1a's
  reps start about 25 s after S0a's decode ends, which is the stronger carry-over case. (Amendment 5 corrects the
  "about 7 min" first written here.)
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
- **The forward:** each window's first **511** tokens are prefilled (Amendment 4), so all 128 of P147's positions are
  read at decode. Its tokens 511..638 are then fed **teacher-forced at T == 1** through the served decode step with
  graphs on, and the NLL and argmax are read at positions 512..639, as in P147.
  - **Why 511, not 512:** prefilling 512 tokens would read position 512 from the prefill, on the transient path, and
    the step fed token 639 would score nothing.
  - Q's decode graphs are re-captured with the step also copying its last-position logits into a static buffer, after
    the timed reps, which run on the build's own graphs.
  - The forcing is checked as T1's and T2's is. Each step reads the bucket's input buffer back after the step's own
    sync, and checks its pointer against the re-capture. A mismatch is a VOID.
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

#### Amendment 4 (2026-10-19): the routing qualifier on g16, token-pinned (the maintainer's ruling, 20:17Z)

**Why.** Each arm decodes its own greedy tokens. At W16, a token where S1 differs from S0 changes that row's MoE routing
at every later step, so g16 carries a routing term besides the kernels. Mixtral's 32 picks per layer-step at B = 16 over
8 experts touch nearly every expert. The routing term therefore lives in the per-expert row histogram (the grouped
small-M GEMM's balance and its 16-row tiles), not in the distinct count. W1 has no such term: every step reads exactly
two experts per layer, whichever they are, so the bytes and the launches do not depend on the tokens.

**The measurement: two token-pinned processes, T1 and T2.**
- Each is a fresh S0 process: the default server, built exactly as S0's arms are, with graphs on.
- It runs only P109's W16 workload, **teacher-forced**. After each decode step, the next input token is overwritten with
  the recorded stream's token at that position, in place of the argmax the step wrote. Both streams are forced, so the
  forcing's own cost is on both sides of the ratio.
  - **A** is S0's own W16 tokens, from S0a's record. LONG is 160 tokens per row, and SHORT is LONG's first 32.
  - **B** is S1's W16 tokens, from S1a's record.
- Each process runs four W16 blocks. Each block is P143 Amendment 3's W16 form: one untimed warm pass, five timed reps of
  SHORT and LONG, and tok/s by its formula.
  - T1 runs **A B B A** and T2 runs **B A A B**, so the streams' order rotates within each process and across the two.
  - The decode graphs are captured at build, before either stream is fed, and both streams replay the same graphs.
- **Where they run:** in the registered rotation, after S1a, since they read S0a's and S1a's records. The order is S0a,
  S1a, the Q reduce, **T1**, S1b, S0b, S0c, S1c, **T2**, S1d, S0d. They do not run after a QUALITY_FAIL (the box stops
  there) or when S0a or S1a failed.
- **Engagement, asserted in T1 and T2.**
  - The forced token is written with `copy_` into the captured graph's own static input buffer. Before every block, that
    buffer's storage is checked against the pointer taken at capture: a rebound tensor would replay its own argmax and
    give r = 1 by construction.
  - Each step, after the sync the served step already makes (decode lookahead is off by default, so every step reads
    its tokens back), a count accumulates two things, read from the graph's own buffers: where the input buffer differs
    from the forced token, which must read 0; and where the step's own argmax differs from the next forced token.
  - The second count must read 0 on A, S0 reproducing its own stream. On B it must read above 0, unless S1 decoded S0's
    tokens exactly; then g16_routing is 1 by construction and the label is ROUTING_CLEAR.
  - Each block's fed tokens are recorded and must equal the stream.
  - Any failure is **QUALIFIER_VOID**. It holds the default-flip PR as UNRESOLVED does, it is not a strike, and the
    reading stands.

**The quantities** (tok/s, unrounded):
- **Per process:** r = (B1 + B2) / (A1 + A2). That is S0's kernels on S1's tokens over S0's kernels on S0's own tokens.
- **g16_routing** = the mean of T1's and T2's r.
- **The check:** the four token-pinned self-pairs, A2/A1 and B2/B1 in each process. e = the largest |pair − 1| of the
  four.
- **Reported beside them:** both streams' routing, from an **eager** replay of A and B through S0's model after T1's
  timed blocks. A Python counter under CUDA graphs runs at capture, not per replay. Each statistic is per MoE layer, and
  its support is the 16 rows × 160 LONG positions:
  - the distinct experts per layer-step;
  - the per-expert row histogram;
  - the 16-row tile count, Σ over experts of ⌈rows / 16⌉;
  - the largest per-expert row count.

**How it acts, stated before any data.** It never moves the verdict: FLIP, NO_FLIP and every other label stand as read.
It gates only the separate default-flip PR that a FLIP starts.
- **The terms:**
  - m = g16 − 0.99, the margin, where g16 is the S rule's minimum of four pairs;
  - ρ = g16_routing − 1, the routing term. A positive ρ means S1's tokens are cheaper on S0's own kernels, so part of
    g16 is routing.
- **The labels:**
  - **QUALIFIER_NOISY:** any of the four self-pairs is outside [0.985, 1.015]. It counts as unresolved. It is not a
    VOID and not a strike; the reading stands.
  - **ROUTING_CONFOUNDED:** ρ − e > m / 2.
  - **ROUTING_UNRESOLVED:** ρ − e ≤ m / 2 < ρ + e.
  - **ROUTING_CLEAR:** ρ + e ≤ m / 2.
- **The effect:** after a FLIP, the default-flip PR proceeds only on ROUTING_CLEAR. Any other label holds it for a
  routing-matched read, which needs a later registration.
- **Resolution, stated before data.** CLEAR needs ρ + e ≤ m / 2. So even at ρ = 0, a FLIP reads CLEAR only if
  g16 ≥ 0.99 + 2e. With ρ ≥ 0 it never clears when e > m / 2. Only a negative ρ, routing working against S1, can clear
  a thinner margin.
  - P143 Amendment 3's W16 self-pairs (Qwen3-30B-A3B, fresh processes) deviated by 0.0002 to 0.0010, with one outlier
    at 0.0141: S0b/S0a, 0.9859, against the first process.
  - At e = 0.001, CLEAR needs g16 ≥ 0.992. At the outlier's 0.0141, it needs g16 ≥ 1.018, the upper half of the
    predicted 0.99-1.03 band.
  - A thin margin's UNRESOLVED therefore means the margin is too thin to separate a routing term, not that one was
    found.
- The verdict always reports g16_routing, ρ, e, m and the label. Without a FLIP, the label is reported only.

**The router pre-read: a prediction, USD 0, on the lab A2000, posted before box Q+S is created.**
- The same W16 prompts are decoded greedily twice: as S0, and as S1 with `E4B_SERVE_ATTN_INT8=1` through K37 compiled on
  sm_86. The experts sit on the host, as the A2000 requires.
- Both streams are then replayed eagerly through S0's model, reporting the routing statistics above.
- They are priced two ways: by bytes (∝ the distinct experts) and by tiles (∝ the 16-row tile count).
- The tile model's g16_routing, with a band, joins the Predictions. It is graded in the read and never rules.
- It is a prediction because the A2000's tokens are not the 5090's: the other GEMMs' reduction orders differ by
  architecture.

**Box Q+S's guard and rate, re-fitted with these processes (≥ 25% headroom).**
- Each term is the larger of the runner's allowance and the largest value measured on a Mixtral RTX 5090 box. The
  allowances are the ones the kit's runner checks time left against: P141 and P145 sized Mixtral's fetch at 3,000 s and
  its bake at 1,500 s, and P143 sized SANE at 900 s. P145's receipt carries phase stamps: an install of 68 s, a fetch of
  174 s, a bake of 43 s, and a speed arm of 158 to 162 s with its load. P135's and P141's receipts do not.
- **The terms (s):**

  | term | s |
  |---|---|
  | boot | 300 |
  | install, self-tests and the compiled suites | 450 |
  | fetch (93.4 GB) | 3,000 |
  | bake | 1,500 |
  | eight S arms, 420 s load + 180 s reps each | 4,800 |
  | Q in S0a and S1a, 1,200 s each | 2,400 |
  | the Q reduce | 120 |
  | T1 and T2, 420 s load + four W16 blocks of 90 s + a 60 s eager replay each | 1,680 |
  | two SANE processes, run apart as P143 ran them, 900 s each | 1,800 |
  | the reserve | 600 |
  | **total** | **16,650 (4.63 h)** |

- **The guard is 6.0 h** (21,600 s), headroom 1.30. The registered 3.0 h (10,800 s) fails even without T1 and T2: the
  other terms sum to 14,970 s, more than the guard itself.
- **The rate takes Amendment 2's form:** up to $1.05/h including the disk, with a base offer at ≤ $0.85/h taken first
  whenever one passes. Amendment 2's evidence (no 575 W RTX 5090 listed at ≤ $0.85/h) holds for this box too.
- **Worst:** 6.0 h × $1.05 + (93.4 + 5) GB × $0.011/GB = **$7.38**, under the standing rule. The expected cost is set by
  the measured terms (about 2 h), not by the guard.
- The kit's runner refuses before the install when the time left cannot fit the rest of this table with its reserve (rc
  19, A3's form). A test asserts the table against the registered guard.
- Every phase records its UTC start and end, so the next fit has measured terms: each process's load, Q, each T block,
  and the eager replay. Q, the T blocks and the replay have no measurement yet.

**The verdict order:** host REFUSED (17, 18, or 15 for an unreadable power limit) → VOID → NOISY → KERNEL_NO_GAIN (box K) → QUALITY_FAIL →
FUNCTION_FAIL → FLIP / NO_FLIP.

#### Amendment 5 (2026-10-20): the dry mode and the early gates (the maintainer's ruling, 06:23Z)

**Why.** `p149-dry-1` (the lab A2000, USD 0) met the registered shakedown's scope:
- K37's suite and e4b's int8 kernel path compiled on the card: 28 and 4 passed.
- P147's W1 and W16 windows matched by sha256.
- On a one-layer cut of the checkpoint, the fold census was exactly the one-layer form written down before the run
  (`fuse_qkv_n` 0, `fuse_t1_glue_n` 3, R2 [1, 1], router 1).
- S1's lever swapped 4 projections on 1 attention module, with 1 R2 rope fold. Through eager decode its 511-row prefill
  took the transient path (64 calls), and its 16-row and 1-row decode steps took the kernel path (16 calls each).
- S0 carried no int8 module and made no int8 call.

It could not build the registered server. Below sm_89, `E4B_PAGED_GRAPHS=auto` resolves off, because the fused FP8 KV
append casts to e4m3 in Triton. So every speed process refused, and SANE's padded bind raised. The bucket capture
census, Q's re-capture and forcing checks, T1 and T2, and SANE would first run for real on the box. There, a kit fault
VOIDs the reading: strike two, which stops the lane.

**The dry mode (`P149_DRY=1`).**
- **Run ids:** `p149-dry-2`, `-3` and `-4` (the lab run was `-1`). The runner refuses a run whose `P149_DRY` does not
  match its id (rc 78), before any install.
- **The model:** the fetch takes shards 1 and 19 of `eba92302` (9.1 GB) with the config and the tokenizer.
  `p149_slice.py` cuts them to one decoder layer: the embedding, layer 0, the final norm and the head, byte for byte, and
  the config at `num_hidden_layers` = 1.
- **The run:** everything after the cut is the box's, at the registered reps and lengths: the bake, the prompts, P147's
  windows, the early gates below, the eight arms with Q in S0a and S1a, the Q reduce, T1 and T2, both SANE phases, and the
  reduce.
- **The reducer** runs `--dry`: the same closed forms at one layer (4 projections, 1 attention module and rope fold, the
  one-layer census, 3 × 4 int8 calls per captured bucket) on the slice.
- **The verdict is DRY, hard-coded: never a reading.** The reduce lists every instrument fault the box's rule would VOID
  on, across every process. The rule's own line on the slice is reported and means nothing. A dry run's gates and its Q
  reduce report and go on, so one dry finds every fault. rc 0 when there is none, else 29.
- **An INSTRUMENT run: never a reading and never a strike.** If it finds a fault, the fix is checked at USD 0 where
  possible, then the next dry id runs.
- **Host and cost:** ONE Vast RTX 5090 (per-second billing) at the box's floors, except a 60 GB disk, at Amendment 2's
  rate form, from main after this kit merges.
- **The dry table (s):**

  | term | s |
  |---|---|
  | boot | 300 |
  | install, self-tests and the compiled suites | 450 |
  | fetch (two shards, 9.1 GB) | 900 |
  | the cut and the bake | 600 |
  | the early gates' process S1w | 240 |
  | eight S arms, 240 s each | 1,920 |
  | Q in S0a and S1a, 480 s each | 960 |
  | the Q reduce | 120 |
  | T1 and T2, 240 s each | 480 |
  | two SANE processes, 240 s each | 480 |
  | the reserve | 600 |
  | **dry total** | **7,050 (1.96 h)** |

- **The dry guard is 3.0 h** (10,800 s). The runner's own check needs 6,750 × 1.25 = 8,437 s left when it runs, so
  boot may take up to 2,363 s; the static fit is 7,050 × 1.25 = 8,813 s. Worst 3.0 h × $1.05 + (9.1 + 5) GB ×
  $0.011/GB = **$3.31**; expected about 40 minutes, under $1.

**The early gates, in both modes (P158's form).**
- **S1w**, a process before any scored arm, is never scored. It builds S1 as every S1 arm builds it. Then it runs Q's
  re-capture and ONE teacher-forced pass per gate, with Q's forcing checks: W1 is one window at bucket 1, and W16 is
  sixteen at bucket 16.
- **rc 43, the build:** the pins, the server and the census at its 32-layer form. Every bucket captured. S1's int8
  census: 128 projections on 32 attention modules, with 32 rope folds. Every captured bucket's int8 calls at the closed
  form: exactly 3 × 128 = 384 at b rows on the kernel path, and none transient. `paged_runner` captures each bucket
  after two warm-up steps.
- **rc 44, the instrument:** Q's re-capture as at the build. Each gate's one pass on its bucket's graph: 128 replays, no
  eager step, no Python decode call. The forced tokens in the graph's own buffer, at its captured address, on every
  step. Prefill rows only on the transient path.
- **rc 45, S0's build:** read from S0a's record before S1a runs: the same build checks for S0, with no int8 module and
  no int8 call at capture.
- **Each is an instrument refusal, not a strike.** The same gate firing on two consecutive runs counts as one strike:
  the closed form, not the host, is then wrong.
- **The VOID rule stays as the backstop,** and the capture count is now part of it: a captured bucket whose S1 int8
  calls differ from 3 × 128 VOIDs.
- **The closed forms are the dry run's, scaled:** `p149-dry-1` read the one-layer census on the A2000. A dry run checks
  the one-layer census, 4 projections and 3 × 4 capture calls on the 5090 before any box reads the 32-layer forms.
- **Box Q+S's guard table, re-fitted (the maintainer's ruling of 10:40Z, option (b)):** S1w adds 600 s, and each
  speed arm's allowance drops from 600 s to 480 s, about twice what a P149 arm should take. P145's measured Mixtral
  speed arm on an RTX 5090 took 158-162 s, its 23-27 s load included, over 3 timed reps; P149's arm runs a warm pass
  and 5, about 230 s.

  | term | s |
  |---|---|
  | Amendment 4's total | 16,650 |
  | the early gates' process S1w (Amendment 5), a load and one Q pass per gate | 600 |
  | eight S arms at 480 s instead of 600 s | -960 |
  | **total** | **16,290 (4.53 h)** |

- **Box Q+S keeps its 6.0 h guard** (21,600 s), the rental policy's 6 h cap.
  - Static: 16,290 × 1.25 = 20,362 ≤ 21,600.
  - The runner's own check needs 15,990 × 1.25 = 19,987 s left when it runs, so boot may take up to 1,612 s. Before the
    trim, S1w alone would have left boot 413 s (cerin-maintainer's re-check).
  - The worst stays **$7.38**.
- **Why one box and not a Q box then an S box** (option (a)): each box fetches and bakes the 93.4 GB checkpoint. At
  these guard forms the split's two boxes and the dry run come to about $14.9 of worst, against the $10.89 left of the
  lane's $15 ($4.11 spent). One box and the dry run come to $7.38 + $3.31 = **$10.69**. A QUALITY_FAIL still stops the
  one box after the Q reduce, before any further arm.

## What a FLIP starts (never automatic)

A separate PR makes `E4B_SERVE_ATTN_INT8` default on Mixtral only, citing this read. Under Amendment 4 it proceeds only
on ROUTING_CLEAR. Nothing changes for any other family.
A NO_FLIP keeps the lever opt-in.

## Predictions (graded in the read, never ruling)

- **K:** K_GO. q and o's projected W1 saving is 0.45-0.70 ms (an int8 GEMV at 0.80-0.95 of the copy rate). k and v's is
  under 0.20 ms (256 programs and small rows). The M == 16 kernel is slower than cuBLAS at 1024 × 4096.
- **Q (at decode):** GO at both gates, with upper bounds within 1.5× P147's (0.0023 at W1). Both kernels differ from the
  fake-quant's bf16-stored weights only by the unrounded fp32 scale and the summation order, the less lossy direction.
- **S:** FLIP, with g1 between 1.04 and 1.08 and g16 between 0.99 and 1.03.
- **Routing (Amendment 4):** ROUTING_CLEAR, with |ρ| < 0.005: S1 changes few tokens, and Mixtral's B = 16 step reads
  nearly every expert either way. g16_routing's band from the A2000 pre-read is added here before box Q+S is created.
  - **The A2000 router pre-read** (2026-10-20, USD 0, a prediction; [`preread/preread.json`](preread/preread.json), sha256
    `efaab639…`), run at e4b `b17fa2ba` with grouped-nf4-gemm 0.49.0. P109's 16 W16 prompts were decoded greedily for
    160 tokens as S0, then as S1 (all 128 projections swapped, the 16-row decode calls on K37, the prefill transient).
    The two streams differ in all 16 rows, with first divergences at steps 5 to 140. Replayed through S0's router, they
    read 7.8919 and 7.8927 distinct experts per layer-step, and every expert's rows sit inside one 16-row tile, so the
    tile and bytes models agree: g16_routing 0.99991.
  - **The predicted band: g16_routing in [0.9979, 1.0019]**, graded in the read and never ruling.
- **Dry (Amendment 5):** `p149-dry-2` reads DRY with no instrument fault, and every captured bucket's S1 int8 calls are
  exactly 3 × 4 on the slice.

## The boxes and the run

- **Host:** one RTX 5090 per box, Vast verified/secure, at most $0.85/h including the disk, under the market rule's pin
  (up to three stable machines, walked on `no_such_ask`). Caution machines are skipped at the pin, and the power floor is
  checked on the box.
- **Pins:** e4b at each kit's merge; torch 2.8.0+cu128, triton 3.4.0, transformers 5.17.0, bitsandbytes 0.50.2.
  grouped-nf4-gemm is at the serve stack's pin for box K, which stages the prototypes, and at the release that ships the
  int8 kernels for box Q+S. The Q+S box also pins Mixtral at `eba92302` (P141's lock).
- **Box K** (`p149k-5090-N`):
  - no checkpoint;
  - guard 0.5 h, disk ≥ 60 GB (0.75 h by Amendment 3);
  - worst 0.5 h × $0.85 + 5 GB × $0.011/GB = **$0.48** (P148's form ran at about $0.03 actual).
  - **Amendment 2 (2026-10-19), box K's rate term only:** box K may rent at up to $1.05/h including the disk (the
    launcher's RTX 5090 class ceiling of $1.05/h, raised 2026-10-19), and takes a base offer at ≤ $0.85/h first whenever one
    passes. Worst 0.5 h × $1.05 + 5 GB × $0.011/GB = **$0.58**. Why: the 575 W floor and the $0.85 rate stopped meeting.
    At 15:07Z on 2026-10-19, 10 hosts with a recorded cap ≥ 575 W listed an RTX 5090, every one above $0.85/h (most at
    $0.9244), and 8 of them pass every other term at ≤ $1.05/h. The day's one such pin under $0.85 (`p149k-5090-5`) was
    taken before the create (`no_such_ask`). Everything else is unchanged, box Q+S included.
  - **Amendment 3 (2026-10-19), box K's guard, after `p149k-5090-6`:** the guard is **0.75 h**. `-6` reached the bench
    (575 W, the kernels' suite compiled on the card, 135 passed) and stopped at rc 40: the bench's check needs `NEED_BENCH`
    1200 s + the 600 s reserve = 1800 s left, the whole 0.5 h guard, so no box could pass it after any setup (175 s on
    `-6`). That is **strike one** (a box-started HARNESS_ERROR from the kit's own guard/budget fit). At 0.75 h = 2700 s the
    fit is 300 s boot + 450 s install-to-bench + 1200 + 600 = 2550 s. Worst 0.75 h × $1.05 + 5 GB × $0.011/GB = **$0.84**.
    The runner now refuses BEFORE the install when the time left cannot fit the install, the bench and the reserve (rc 19,
    a refusal before the install, not a strike), prints the reserve in its STOP-2 line, and records the bench's own wall
    time (`BENCH_WALL_S`). A test asserts the fit against the registered guard. Everything else is unchanged.
- **Box Q+S** (`p149-5090-N`), only after K_GO:
  - disk ≥ 200 GB (the 93.4 GB checkpoint and the venv), declared as `lane_record.min_disk_gb`;
  - guard **3.0 h** (**6.0 h** and Amendment 2's rate form, by Amendment 4): eight Mixtral loads at about 7 min, Q's
    decode in S0a and S1a at about 15-20 min, the speed reps,
    and the fetch;
  - worst 3.0 h × $0.85 + (93.4 GB checkpoint + 5 GB install) × $0.011/GB = **$3.63** (**$7.38** by Amendment 4).
- **Both** are under the standing rule (at most $15 each).
- **Run ids:** each box has `-1` as its reading and `-2` and `-3` as spares for pre-create refusals and host refusals (17
  and 18, and 15 for an unreadable power limit, are not strikes). Shakedowns are `p149-dry-N` on the lab A2000 ($0, correctness only: the suite compiled, the
  lever's census and engagement counters on a Mixtral slice, and Q's windows and shas).
  - **Amendment 5 (2026-10-20):** `p149-dry-2`, `-3` and `-4` are dry runs on one RTX 5090 (`P149_DRY=1`, the one-layer
    cut, guard 3.0 h, worst $3.31): instrument runs, never a reading and never a strike.
  - **Amendment 1 (2026-10-19), box K:** `p149k-5090-4`, `-5` and `-6` are spares for refusals BEFORE THE INSTALL only:
    any of the runner's refusals ahead of its first `pip install` (rc 18, 10, 15, 13, 16, 17 or 28, in its order), a
    pre-create refusal, or `no_such_ask`. Which of those count as strikes is unchanged; a box that reaches the install and
    fails is still a strike.
    `-4` is already spent: it was created before this amendment registered it, and the driver refused the unregistered
    id (rc 78) before any box work, at $0.001. That was a launch-side error, not a box read.
  - **Amendment 3 (2026-10-19), box K:** `p149k-5090-7`, `-8` and `-9` are spares on the same terms as Amendment 1's: any
    refusal the runner makes before its first `pip install`, by position, which now includes rc 19.
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
