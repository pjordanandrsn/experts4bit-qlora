# P149 results: box K reads **K_GO**; box Q+S has not run (2026-10-20)

The registration is [`PREREG-p149.md`](PREREG-p149.md) (#1806, with Amendments 1-5). P149 asks whether int8 weight-only
attention projections (q, k, v and o, through grouped-nf4-gemm's int8 kernels) make Mixtral-8x7B's served decode faster
at an unchanged quality bar. It reads in two boxes: box K, the kernel premise, then box Q+S, quality at decode and the
served speed. **Nothing here licenses a default or quotes a gain.** `E4B_SERVE_ATTN_INT8` stays opt-in until box Q+S
reads.

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

## Box Q+S: not run

Box Q+S rules quality at decode first (P147's gates through the served step, graphs on), then the served speed (P143
Amendment 3's form). It carries Amendment 4's token-pinned routing qualifier and Amendment 5's dry mode and early gates.
Its kit is #1858. Before it runs, `p149-dry-2` exercises the whole rotation on a one-layer cut on one RTX 5090. That is
an instrument run, never a reading. Until box Q+S reads, no speed is claimed for the lever.
