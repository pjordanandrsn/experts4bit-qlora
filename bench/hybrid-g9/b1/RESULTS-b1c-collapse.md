# RESULTS — B1c collapse: CERTIFIED-WITH-OPEN-MECHANISM (default ON)

Run 2026-08-24 against `PREREG-b1c-collapse.md` (#225), on a fresh
EPYC 9655 + RTX 5090 box (vast 48593098, $0.553/hr, destroyed +
verified zero; device triad 1573.9 GB/s). e4b main `8483735`.
Receipts in `receipts-b1c/`; verdict by `b1c_verdict.py` (self-tested,
8 branches).

## Verdict table

| gate/bar | registered | measured | result |
|---|---|---|---|
| GS: B=16 sanity, flag ON | in certified band, flag inert | 126.9 ms, attn 41.0, collapse=true | PASS |
| G0 A/A (C0 / C1 / R1) | < 7.5% each | 0.15% / 0.18% / 2.77% | PASS² |
| G1 identity | C0 ≡ C1 bitwise | **C0 ≡ C1 ≡ R1, 128/128** | PASS |
| H-C wall | ≥ 15% vs C0 | **15.99%** (59.27 → 49.80 ms) | PASS |
| H-M bracket | C1 moe_host ≤ 60% of C0 | 63.0% (28.2 → 17.8 ms) | FAIL |

² The first R1 pair G0-failed at 184% spread: its first run was
uniformly ~2.8× slow across every bracket (prefill 113 vs 39 ms/step)
during a co-tenant CPU burst (box load average 3.3 on a 48-of-96-core
slice). The registered consequence ran: a fresh pair (2.77% spread,
42.7 ms — matching the prior box's R1 within class variance). Both
pairs' receipts are kept; the transiently-poisoned pair decided
nothing.

**⇒ CERTIFIED-WITH-OPEN-MECHANISM: ship, default ON** (this PR flips
`enable_hybrid_tier(collapse_resident=True)`; `--no-collapse` runs the
C0 arm; `--collapse` stays accepted as a no-op — the #220 pattern, in
the same cycle as the cert this time).

## What the collapse buys (single-stream, all-VRAM placement)

- Step 59.27 → **49.80 ms**; per-stream **16.9 → 20.1 tok/s** on this
  box — with bit-identical decoded tokens, as the arithmetic-identity
  construction claimed and the (real) G1 gate verified.
- **57.2% of the gap to the pipelined reference closed**; C1 now sits
  16.6% above R1 (49.8 vs 42.7 ms).
- The B=16 hybrid point is structurally untouched (placement-static
  predicate; sanity run in-band with the flag on).

## The open mechanism (filed, not hand-waved)

H-M missed its 60% bar at 63.0%. Two candidate explanations, both
recorded for the next attribution pass: (1) the region bracket carries
fixed per-call overhead (timer + record_function guards), which
inflates a bracket's READING proportionally more as the bracketed work
shrinks — the same instrument-tax class as AMENDMENT-t5b-h-a and the
e4b#222 fast-gate fix, now appearing in a ratio of two instrumented
quantities; (2) the collapsed path still carries real per-layer work
the model under the bar didn't price (the activation gather, the
fused-kernel host launch, the reshape/sum). Distinguishing them needs
an overhead-calibrated bracket (measure the wrapper's own cost on a
no-op region) — registered as part of the next attribution instrument,
not asserted here.

## The 425 single-stream ladder after this rung

Roofline on this 5090 (measured 1573.9 GB/s): ~3.2 GB of weights per
decoded token ⇒ **~2.0 ms/token ≈ 480-490 tok/s ceiling**; 425 needs
~88% of roofline end-to-end.

1. ~~Collapse~~ — **this cert**: 49.8 ms (20.1 tok/s).
2. **Device-driven decode loop** (next registration): the collapsed
   all-resident path is static-shaped with no host decisions — the
   shape CUDA graphs capture. The T1b cudagraph refutation was at the
   hybrid point full of graph breaks and does not bind here; a fresh
   line is licensed by this cert's receipts. Host → ~0 targets the
   step toward its ~13 ms device floor (~75 tok/s).
3. **M=1 kernel roofline**: the device work itself (13.3 ms) runs at
   ~15% of bandwidth at M=1 — batched 8-expert reads per layer,
   fused epilogues, attention/dense/lm_head at B=1 shapes.
   Ceiling of this rung ≈ 2.5–4 ms ≈ 250–400 tok/s.
4. **Speculative decoding**: the rung that crosses 425 — amortize the
   per-token weight read over N accepted tokens; greedy-identity
   verifiable with the token gate. Its own registration when the
   ladder reaches it.
