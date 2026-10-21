# P149 results: box K reads **K_GO**; box Q+S reads **FLIP**, its routing qualifier VOID (2026-10-20)

The registration is [`PREREG-p149.md`](PREREG-p149.md) (#1806, with Amendments 1-5). P149 asks whether int8 weight-only
attention projections (q, k, v and o, through grouped-nf4-gemm's int8 kernels) make Mixtral-8x7B's served decode faster
at an unchanged quality bar. It reads in two boxes: box K, the kernel premise, then box Q+S, quality at decode and the
served speed. **Box Q+S read FLIP, but its routing qualifier read QUALIFIER_VOID**, which by Amendment 4 holds the
default-flip PR for a routing-matched read. So `E4B_SERVE_ATTN_INT8` stays opt-in (e4b v0.56.0 and main at 2026-10-20),
and the default does not change.

## Box K (`p149k-5090-8`): K_GO

The read (one RTX 5090 at 575 W, 2026-10-19, $0.038; receipts under
[`receipts/p149k-5090-8/`](receipts/p149k-5090-8/)) timed the int8 kernels against cuBLAS bf16 at Mixtral's attention
shapes. Every launch read its own layer's weight, and A1 B1 B2 A2 ran as fresh graphs.

| | |
|---|---|
| **W1 projected in-situ saving** (q, k, v, o at 32 calls each) | **0.984 ms**, against the 0.1625 ms bar: **K_GO** |
| q and o (4096 × 4096) at M == 1 | 0.716 ms (kernel 0.7588 ms against cuBLAS 1.4850 ms for 64 calls) |
| k and v (1024 × 4096) at M == 1 | 0.268 ms (0.3064 against 0.5799) |
| M == 16, reported only | q, o 0.7903 against 1.5547 ms; k, v 0.5225 against 0.5686 ms |
| the harness's own saving, reported | 0.999 ms |
| self-pairs | all within [0.9993, 1.0012] |
| contract (error ≤ 1.05 × cuBLAS's) | every shape passes |

The bar is 2 % of P145's 8.125 ms W1 node period; the saving per shape is P145's in-situ time × (1 − B/A), the smaller
of the two pairs. **This is a microbench premise:** it opened box Q+S and is never quoted as a served gain. The verdict
was re-derived byte-identical from the private receipt store.

### The predictions, graded

| prediction | read | |
|---|---|---|
| K_GO | K_GO | hit |
| q and o's projected W1 saving 0.45-0.70 ms | 0.716 ms | missed, just above |
| k and v's under 0.20 ms | 0.268 ms | missed |
| the M == 16 kernel slower than cuBLAS at 1024 × 4096 | 0.5225 against 0.5686 ms, faster | missed |

### The run ids

| run | outcome | cost |
|---|---|---|
| `-1`, `-5` | refused before any instance was created | $0 |
| `-2`, `-3`, `-7` | the power floor refused before the install: 400 W, 475 W and 475 W against 575 W (rc 17, not a strike) | $0.042 |
| `-4` | the driver refused an id not yet registered (rc 78), before any box work | $0.001 |
| `-6` | **strike one:** the kernel suite passed on the card (135), then the bench's time-left check could not fit the 0.5 h guard (rc 40). Amendment 3 re-fitted the guard to 0.75 h | $0.046 |
| `-8` | **K_GO** | $0.038 |

Box K cost $0.127 in all.

## What K_GO started

- **The kernels shipped:** grouped-nf4-gemm 0.49.0 carries them as K37 (`dense_i8.linear_i8`), at the configs `-8`
  chose.
- **The engine shipped, opt-in:** `E4B_SERVE_ATTN_INT8=1` (#1848) swaps Mixtral's 128 attention projections for
  `Int8Linear`. Decode calls of 1-16 rows run K37, and larger calls dequantize transiently. It refuses on other
  families, without K37, and alongside int4 attention or `E4B_DENSE_GEMV_BW=1`.
- **The quality premise had already passed:** P147 (`p147-a2000-1`) read I8ATTN GO for the int8 round trip on the lab
  A2000 ([results](../p147/RESULTS-p147.md)).

## Box Q+S (`p149-5090-2`): FLIP

One RTX 5090 (cc 12.0, 170 SMs, 32,607 MiB, a **575 W** power limit read on the box before any install, a 3,090 MHz
maximum SM clock) on an AMD Ryzen 7 9800X3D host (8 cores, 16 CPUs). The lane ran 2026-10-20T19:03Z to 20:38Z. Cost
$1.574. Receipts are under [`receipts/p149-5090-2/`](receipts/p149-5090-2/).
- **Code:** e4b `835874d1` (the pre-read band's merge, #1893, carrying the kit #1858 byte for byte) and grouped-nf4-gemm
  0.49.0 at `a83f6bb3` (K37). Model `mistralai/Mixtral-8x7B-Instruct-v0.1` at `eba92302`.
- **Stack:** torch 2.8.0+cu128, triton 3.4.0, transformers 5.17.0, bitsandbytes 0.50.2 (`versions.txt`).
- **Before the arms:**
  - The power floor and refusals passed.
  - The self-tests passed: `p149_reduce.py` 47, `p149_box.py` 23, `p115_quality.py` 11.
  - On the card: K37's suite (28) and e4b's int8 path (4).
  - P147's windows matched by sha256 (W1 196, W16 123).
  - **Amendment 5's early gates:** S1's build and instrument gates and S0's build gate all read OK.

**Q at decode: GO at both gates** (`q_verdict.json`). S1 is the int8 arm; d is S1's NLL minus S0's at each position.

| gate | windows | mean d | upper bound | bound | argmax agreement | S0 perplexity |
|---|---|---|---|---|---|---|
| W1 | 196 | 0.001024 | 0.002311 | 0.011537 | 0.9702 | 4.3088 |
| W16 | 123 | 0.001272 | 0.002894 | 0.011313 | 0.9691 | 4.3948 |

**S, the served read: FLIP** (`verdict_p149.json`). Decode tokens/s in the eight fresh processes, ABBA twice:

| | S0a | S1a | S1b | S0b | S0c | S1c | S1d | S0d |
|---|---|---|---|---|---|---|---|---|
| W1 | 128.86 | 149.58 | 149.51 | 128.73 | 128.80 | 149.54 | 149.59 | 128.78 |
| W16 | 324.72 | 330.60 | 330.44 | 323.38 | 323.40 | 330.39 | 330.52 | 323.49 |

- **g1 = 1.160795**, the minimum of the four S1/S0 W1 pairs (1.1608, 1.1614, 1.1610, 1.1616), against the 1.03 bar.
- **g16 = 1.018108**, the minimum of the four W16 pairs (1.0181, 1.0218, 1.0216, 1.0217), against the 0.99 bar.
- **Self-pairs:** W1 within [0.9990, 1.0003] and W16 within [0.9959, 1.0004], inside the [0.985, 1.015] NOISY band.
  Each arm decoded its own first record's tokens, so there was no FUNCTION_FAIL.
- **GPU state, reported:** SM clock 2,647-2,842 MHz and power 436-582 W across the timed blocks.

The verdict was re-derived byte-identical from the receipt subset in this tree with `p149_reduce.py` at `835874d1`.

### The routing qualifier: QUALIFIER_VOID

Amendment 4's token-pinned processes T1 and T2 each ran four blocks. A blocks feed S0's own recorded stream through
S0's kernels, and B blocks feed S1's stream through S0's kernels. Engagement held in every block: the forced token sat
in the graph's own input buffer, at its captured address, and every block's fed tokens equalled its stream.

| block | stream | argmax differs from the next forced token | steps | decode tokens/s |
|---|---|---|---|---|
| T1 0, T1 3, T2 1, T2 2 | A | 5 in each | 1,100 | 323.28, 324.61, 324.64, 324.64 |
| T1 1, T1 2, T2 0, T2 3 | B | 410 in each | 1,100 | 325.64, 325.79, 324.41, 325.68 |

A must read 0: S0 reproducing its own stream. It read 5 in every A block, so the label is **QUALIFIER_VOID** (margin m =
0.028108). The count is the same in all four blocks across two processes, so this is a fixed difference between S0's
token-pinned replay and S0a's own decode, not run-to-run noise. Its cause is given below. By Amendment 4, the
reading stands, this is not a strike, and the default-flip PR is held for a routing-matched read, which needs a later
registration. g16_routing and its band are not graded.

#### QUALIFIER_VOID cause (found after the run, from the receipt, no new run)

S0 is deterministic across processes. The instrument forced the 160-token stream into the 32-token pass, which
assumes the served stream does not depend on `max_new_tokens`, and it does (below):

- **S0 reproduces itself across processes.** S0b, S0c and S0d reproduce S0a's W16 32- and 160-token streams and its W1
  streams exactly, 16 of 16 rows (`arm_S*.json`).
- **The forced stream was the wrong length's.** The token-pinned harness (`p149_box.py`, `tf_main`) forces the
  160-token run's stream into both passes, including the 32-token pass. S0a's own 32-token run differs from its
  160-token run's first 32 tokens in one place: W16 row 10, step 29 (token 4777 against 6795). The forced prefix through
  step 28 is identical in both runs, and at step 29 the 32-token pass's argmax repeats the 32-token run's own choice
  (4777), not the forced 6795, so block A cannot read 0. Five per A block is that one position in each of the five
  timed reps.

**Reported, not attributed:** the served decode's tokens depend on `max_new_tokens`. In W16, S0a's 32-token run differs
from its 160-token run's prefix in one row (row 10, step 29), and S1a's in two (row 8, step 31; row 12, step 27); W1
differs in neither arm. All three sit in the last five steps of the 32-token run. This is a serving-correctness question
of its own, outside P149's reading, and it is not attributed here. The routing-matched read forces each pass's own
length's stream.

### The predictions, graded

| prediction | read | |
|---|---|---|
| Q: GO at both gates, upper bounds within 1.5 × P147's (0.0023 at W1) | GO; W1 0.002311, W16 0.002894 | hit |
| S: FLIP | FLIP | hit |
| g1 between 1.04 and 1.08 | 1.1608 | missed, faster |
| g16 between 0.99 and 1.03 | 1.0181 | hit |
| routing: ROUTING_CLEAR, with \|ρ\| < 0.005 | QUALIFIER_VOID | not read |
| the pre-read band: g16_routing in [0.9979, 1.0019] | not computed (VOID) | not read |
| dry: `p149-dry-2` reads DRY with no instrument fault | DRY, no fault | hit |

### The run ids

| run | outcome | cost |
|---|---|---|
| `p149-dry-2` | DRY with no instrument fault (Amendment 5's one-layer cut, an instrument run) | $0.231 |
| `p149-5090-1` | refused before any instance existed: the offer was taken between search and create. A pre-create refusal spends the id as a spare, not a strike | $0 |
| `p149-5090-2` | **FLIP**, routing qualifier VOID | $1.574 |

## What FLIP started, and what it holds

- **Nothing changes by default.** The default-flip PR, int8 only where K37's `dense_i8` is present and only on Mixtral,
  proceeds only on ROUTING_CLEAR. The qualifier read VOID, so that PR is held.
- **The claim** is `e4b.serve.p149.int8-attn.mixtral.5090.2026-10-20` in `docs/claims.json`. It is the opt-in lever's
  measured served speed on this host: one request 1.16 × faster, sixteen requests 1.018 ×, with Q GO.
- **The open question** is why the served decode's tokens depend on `max_new_tokens` (the QUALIFIER_VOID cause,
  above). A routing-matched read that settles g16's routing share must be designed around it, and must be registered
  before it runs.

Re-derive: `python bench/p149/p149_reduce.py --dir bench/p149/receipts/p149-5090-2 --out {out} --e4b-sha 835874d1449314fc569442228a3e46e7a6602748` writes `bench/p149/receipts/p149-5090-2/verdict_p149.json`
