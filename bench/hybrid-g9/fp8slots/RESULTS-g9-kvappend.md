# RESULTS — batched KV append: KVAPPEND-CERTIFIED, +56% throughput

Scored under [PREREG-g9-kvappend.md](PREREG-g9-kvappend.md); receipts
in [receipts-kvappend/](receipts-kvappend/). Box: EPYC 9655 + RTX 5090
(the baseline's host class), CUDA unit tests green on-box (void gate),
A/B/A at the frozen operating point. Cycle ≈ $0.55, box destroyed,
zero instances.

## Verdict

| arm | step ms | attention_host | other | dram | uniques total |
|---|---|---|---|---|---|
| A1 unbatched | 204.7 | 112.6 | 76.9 | 14.3 | 13,186 |
| **B batched** | **131.3** | **40.2** | 76.4 | 14.0 | **13,186** |
| A2 unbatched | 207.0 | 114.6 | 77.5 | 14.0 | 13,186 |

* **B0 (void) PASS** — 4/4 CUDA unit tests (batched vs sequential
  bit-equal pages, counters, tables) and uniques EXACTLY equal across
  all three arms: the fix changes no byte of model behavior.
* **B1 PASS** — attention_host 40.2 ms vs the ≤ 57.2 bar: a 65% cut of
  the bucket, from replacing ~1,500 single-token quantize kernels per
  step with 96 batched ones.
* **B2 PASS** — step 131.3 ms vs the ≤ 166 bar; delta 74.6 ms against
  a 7.0 ms 3×spread — scoreable at 10×.
* **Throughput: 78.2 → 121.9 tok/s aggregate at B=16 (+56%)**, 7.62
  tok/s per sequence, from one bit-identical batching change.

## What remains on the step (the next lines, if wanted)

131.3 ms = 40.2 attention-host (residual: per-sequence block writes
and ~28 ms of true device attention) + **76.4 other-submission** (now
the tallest host bucket: router/norm/projection launch path across 48
layers) + 14.0 dram experts + ~0.7 bookkeeping. The other-submission
bucket is the successor candidate; the cProfile receipts from the
hostbill run already cover it.

## Disposition

The call-site switch stays available behind
`Fp8PagedKV(batched_append=True)`; flipping the DEFAULT to True is a
one-line follow-up the certification supports, left for the owner (it
changes every consumer's receipts baseline). The G9 arc so far:
hostbill NO-GO (python-fix premise refuted) → attribution named the
append storm → this fix certified. The campaign's tok/s story:
45.4 (G9-era) → 83 (this program's serving stack) → **122** (this
line), same model, same batch class.
