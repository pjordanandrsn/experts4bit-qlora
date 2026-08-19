# Full-stack re-measure: G8 balance and G9 throughput after the fix round

Same host class as the pre-fix baseline (`../box/RESULTS-box-run.md`):
Threadripper PRO 7975WX (Zen 4 AVX-512, 32c/64t, all 64 CPUs visible),
RTX 5090, $0.936/hr. Stack under test = everything merged today: dequant
hoist + row chunking (gnf4#107), placement concentration via solver
constants + 32-worker pool + KV stream-sync removals (e4b#155), fused
expert-FFN kernel + adoption (gnf4#108, e4b#156). Calibration on-box;
exactness suites on-box (36/36 kernel x2, hybrid subset 8/8).

## G8 balance clause (bar 0.80)

| arm | config | B=8 | B=16 | dram ms (8/16) |
|-----|--------|-----|------|----------------|
| pre-fix (same host class, yesterday) | threads=0, bw-only solver | 0.124 | 0.074 | 160 / 236 |
| T | 32w + constants, **two-call** | **0.427** | **0.455** | 42.9 / 41.9 |
| F | 32w + constants, **fused** | 0.380 | 0.339 | 48.7 / 57.4 |

- **Still a MISS**, but 3.4-6x closer, and the batch trend now IMPROVES
  with batch on the two-call arm (0.427 -> 0.455) instead of collapsing
  (0.124 -> 0.074). The DRAM wall fell 160 -> 43 ms/step.
- **The fused kernel LOSES in-executor on serving hardware** — 13% at
  B=8, 37% at B=16, and its batch trend DEGRADES (48.7 -> 57.4 ms) as
  batching grows multi-row groups. This is the coarse-item imbalance the
  Zen5 micro-box predicted (multi-row groups: 1.25 vs 1.01 ms on a
  floor-free box): a (group, row-chunk) partition makes ~50 items of
  very different sizes for 32 workers, and the tail eats more than the
  saved wake. The two-call path's thousands of fine column tiles balance
  nearly perfectly. Consequence: `fused_ffn` DEFAULT FLIPS TO FALSE in
  this commit — the fused call remains available for the regime that
  measured in its favor (floor-heavy small-pool hosts: +24% on the
  loaded AVX2 dev box) and decode-singleton workloads (parity floor-free,
  so saved wakes are pure win when the floor is real and rows stay 1).
- unique_dram per step ~equal across arms (placement identical:
  4045/2099); per-layer histogram now embedded in the receipts.

## G9 throughput (bar 140 tok/s aggregate)

| run | aggregate | per-stream | TTFT p50 |
|-----|-----------|------------|----------|
| pre-fix (same host class) | 23.9 | 3.0 | 12.2 s |
| this run (full stack, fused ON) | **43.9** | 5.5 | 17.3 s |

- **Still a MISS**, 1.84x better on the same host class. Note the G9 arm
  ran fused ON (the then-default); by the G8 arm delta (~6 ms of a
  ~190 ms step) a two-call G9 would read ~1-3% higher — NOT measured,
  stated as inference.
- TTFT p50 ~17 s unchanged: prefill still serializes one sequence per
  chunk forward (`ids[None]`) — batching prefill across sequences
  remains the top TTFT lever, untouched this round.

## Where the remaining gap lives

G8: dram 42 ms vs gpu 19 ms needs another ~2.2x on the DRAM side. The
per-call floor lever that remains is the per-worker SUBSET WAKE (wake a
job-sized worker set instead of all 32; needs per-worker futexes). G9:
at 43.9 tok/s the step is ~183 ms of which experts are ~61 ms — the
non-expert majority (attention + engine python at B=8 x 48 layers) is
the dominant term and now the primary G9 target, along with prefill
batching for TTFT.

Receipts: `calib.json` (host block), `g8_T.json`, `g8_F.json`
(per-layer dram histograms embedded), `g9_fused.json`, `rows_zen4.log`,
`rows_curve.json`. Box destroyed; SSH refused + API null verified.
