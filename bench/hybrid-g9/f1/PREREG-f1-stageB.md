# PREREG — F1 Stage B: removing the elementwise block

Registered 2026-08-25, before any Stage B measurement. Basis:
RESULTS-f1-stageA (4.74 ms/step elementwise over 4006 launches, 47
named sites, four mechanisms owning 3.75 ms).

## The arms

All arms run the **graph loop** (`--b1d-loop graph`), B=1, on one box,
same prompt and seed, in one session. Anchor first, treatments after.

- **B0 — anchor.** No compile. Re-establishes the certified ladder's
  step time on this box so every delta below is within-box.
- **B1 — compile the dense layer body.** `--compile-layers
  --compile-mode default`. This treatment is ALREADY IN THE HARNESS and
  has never been measured at B=1 under the graph loop: dynamo owns the
  decoder-layer body while the paged-attention shim and the MoE tier
  stay `dynamo.disable`d, so inductor should fuse exactly the two
  mechanisms living in that body — **RMSNorm (1.26 ms) and RoPE
  (0.59 ms), 1.85 ms of the 4.74**. Mode is `default`, NOT
  `reduce-overhead`: reduce-overhead makes inductor manage its own
  CUDA graphs, which would nest inside our capture.
- **B2 — fuse the fp8 KV path** (only if B1 lands short of PASS). The
  remaining 1.58 ms is our own code and sits inside the dynamo-disabled
  shim, so compile cannot reach it: `_write_side`'s two symmetric
  `copy_` halves (1.08 ms) and `quantize_kv_fp8`'s abs/amax/where/div/gt
  chain (0.50 ms). Registered now so the treatment order is fixed
  before any number is seen.

## Bars (unchanged from the merged prereg)

Measured on the graph basis, median of the same estimator the ladder
used:

- **PASS**: step ≤ 10.5 ms (≥ 95.2 tok/s)
- **PARTIAL**: 10.5–12.0 ms, ships only if the arm's own A/A spread is
  < half the measured gain
- **REFUTED**: > 12.0 ms

## Refusals (any one voids the arm)

1. **Token identity.** Greedy continuations must be token-identical to
   B0 across the full generation. A fusion that moves a token is a
   REFUSE, not a disclosure — this is the hard gate, and it is why
   `default` mode (fp32-accumulating inductor kernels) is used rather
   than any fast-math option.
2. **A/A first.** B0 runs twice before any treatment; if the two B0
   arms differ by more than half the PASS margin (1.48 ms), the box is
   disqualified and the cycle re-rents.
3. **Capture neutrality.** The graph arm must record identical
   positions and step count to the eager arm, as b1d requires.
4. **No silent recompile.** Compiled arms report dynamo recompile
   counts; a recompilation inside the timed window voids the arm (it
   would time compilation, not the fused kernels).

## Verdict calculator

`f1_verdict.py`, self-tested both directions, reads the arm JSONs and
applies the bars above. Receipts in `receipts-f1/stageB/`.
