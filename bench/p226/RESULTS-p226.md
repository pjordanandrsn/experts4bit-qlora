# P226 results: K25's plan at bucket 32 is not a lever (NO_GO; c1 stays)

**Registration:** [PREREG-p226.md](PREREG-p226.md) (e4b#2012, with amendment A2), read once, as registered.
**Reading:** `p226-5090-1`, one RTX 5090 at a 575 W power limit, e4b `4913f959`, grouped-nf4-gemm v0.50.0, $0.305.
Records: [receipts/p226-5090-1/](receipts/p226-5090-1/README.md).

## The question

At Qwen3-30B-A3B's bucket 32, the served decode sends 32 x top-8 = 256 routed rows through K25, on `c1`: the plan P160
read GO at 16 token rows (128 routed rows) and P161 made qwen3_moe's default. Does any plan in P156's 144-plan grid beat
c1 by at least 5 % at 256 rows, on 32 independent requests' routing, under P160's instrument?

## The answer: NO_GO

| arm | plan (gate_up / down) | median C/arm (64 reps) | lower bound | rules |
|---|---|---|---|---|
| C2, the self-pair | c1 / c1 | 1.0001 | 0.9995 | inside [0.99, 1.01]: the instrument is sound |
| **K1**, the selection's first | 64/256/8/3 / 128/128/8/4 | **0.9676** | **0.9669**; clock-adjusted 0.9644 | **< 1.05: NO_GO** |
| K2 (reported) | 64/256/4/3 / 32/256/4/2 | 0.9643 | 0.9636 | |
| K3 (reported) | 32/256/2/2 / 128/128/4/2 | 0.9797 | 0.9792 | |
| D, K25's built-in plan (reported) | 32/64/4/3 / 32/64/4/3 | 1.7712 | 1.7676 | |

- **No plan the selection carried is faster than c1 at 256 rows.** All three are slower, so a different choice of carried
  plan could not have produced a GO.
- **No contract finding.** Every one of the 144 plans per projection was bit-identical to c1 at selection, and every
  arm was bit-identical to C on every confirmation rep. There were no tile faults.
- **c1 stays qwen3_moe's plan at bucket 32.** The lane ends here: the first GO or NO_GO ends it.

## Reported, never ruling

- **Headroom:** c1 ran 7.883 ms a step for both GEMMs over 48 layers. The byte floor of the captured distinct experts
  (3,605 a step over the layers, at P141's 1,498.9 GB/s) is 6.384 ms, so c1 runs at **81.0 %** of the copy rate, with at
  most 1.50 ms a step left to any plan at this bucket.
- **D/C 1.771:** c1 is 1.77x faster than K25's built-in plan at 256 rows. The P160/P161 default carries over from 16 to
  32 token rows.
- **Drift probe:** c1 opened and closed each projection's sweep at 1.016 (gate_up) and 0.997 (down).
- **Compiled metadata:** the reader returned no compiled variants for any arm (empty records), so registers and spills
  are not reported for this run.
- **Time:** the bench took 573 s (selection 467 s, confirmation 103 s) against the registered 2,170 s allowance. The
  allowance doubled the 16-row times; on this kernel the work did not scale twofold with rows.

## What this does not say

- Nothing at bucket 64 or wider: at 512 routed rows the served route is the M-tile GEMM, not K25 (a route question).
- Nothing for other families, cards or buckets, and no served speed (none was read).
