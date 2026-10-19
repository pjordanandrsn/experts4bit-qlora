# P150 results: D2's quality ruling, the int8 lm_head on held-out windows (e4b#1313)

**Status:** closed. **GO**: rounding only the lm_head to its per-row int8 grid keeps P125's quality gate at both ruled
gates, on windows nobody had read. This is a ruling on **quality only**. It licenses the int8 head on quality and
nothing else: its speed is a served read of its own, and no default, gate or speed claim changes here.

[P148](../p148/RESULTS-p148.md) read D2, an int8 lm_head, GO in a microbench, a premise. Before any kernel reaches the
served path, [the registration](PREREG-p150.md) ruled on its quality with the W8A16 split premise's rule, unchanged. The
earlier I8HEAD reads were reported, never ruled.

| run | what | cost | verdict |
|---|---|---|---|
| [`p150-5090-1`](receipts/p150-5090-1/README.md) | arms A and I8HEAD on held-out windows | $0.615 | **GO** |

The maintainer re-derived `verdict_p150.json` byte for byte from the private receipt store (the reducer at `658a0665`,
equal to the staged pin and the box's copy).

## The read

- **The windows nobody had read.** The split box had already read I8HEAD on P125's own wikitext-test windows. This read
  drew the same counts (t1 196, k16 123, stride 1024) from the same corpus's validation split. Their sha256s were
  registered before data, and the box read exactly them in both arms (t1 `5c444982…`, k16 `7ab52a50…`).
- **The lever.** I8HEAD rounded exactly the lm_head, one module (151936 x 2048, untied), to its per-row int8 grid. That
  grid is P148's `quantize_rows_int8`, bit for bit. Every attention projection, expert, the router, the embeddings and
  the norms stayed bf16.
- **The subject.** P125's (e4b `4934aba9`, grouped-nf4-gemm 0.43.0, Qwen3-30B-A3B @ `ad44e777`), on one RTX 5090 at
  575 W.

| gate | windows | argmax | d (nats) | d + 1.645 SE | P125's bound | ruled |
|---|---|---|---|---|---|---|
| t1 | 196 | 0.98904 | +0.000747 | 0.001225 | 0.005205 | pass |
| k16 | 123 | 0.99035 | +0.000547 | 0.001130 | 0.005659 | pass |

The bars compare unrounded values. The two gates overlap (k16's windows are t1's first 123), so they are two views of
overlapping text, not two samples.

**Continuity** (cited, seen before the rule). The split box's I8HEAD read on P125's test windows was t1 argmax 0.9892,
upper bound 0.00103; k16 0.9904, 0.00125. The fresh read agrees.

## Predictions (graded, never ruling)

| prediction | read | grade |
|---|---|---|
| GO | GO | hit |
| each gate's upper bound below half its bound | 0.24 and 0.20 of the bound | hit |
| argmax at least 0.985 at both gates | 0.98904 and 0.99035 | hit |

## What the GO starts

D2's speed, in this order:
- **The kernel:** a grouped-nf4-gemm int8 head GEMV (lane K36). It ships P148's `_gemv_i8_kernel` and
  `quantize_rows_int8` verbatim, under a tolerance contract against cuBLAS's bf16 head on the dequantized weights.
- **The knob:** an e4b knob that routes the lm_head's single-row decode GEMV through it.
- **The served read:** in P143's form, registered on its own, with the 575 W floor and the unprofiled step as its
  denominator.

Nothing flips before that read.
