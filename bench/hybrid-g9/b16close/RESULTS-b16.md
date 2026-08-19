# The B=16 axis: call-level row-scaling is real on BOTH ISAs; B=8 confirmed on a third host class; B=16 remains open at 0.69

Two boxes: TR PRO 7975WX (Zen 4, `../b16axis/`) and EPYC 9655 (Zen 5,
this directory), both with RTX 5090, capped harness, full merged stack.
Both destroyed + verified.

## The law this round establishes (and the one it retires)

Kernel-only, same expert set, rows doubled:

| host | B=8 (64 rows) | B=16 (128 rows) | bytes / time growth |
|---|---|---|---|
| Zen 4 7975WX | 543-588 us (50-53 GB/s) | 1130-1636 us (18-26 GB/s) | +0% / +92-178% |
| Zen 5 9655 | 516-525 us (124-131 GB/s) | 914-941 us (82-84 GB/s) | +12% / +78% |

**Call time scales with rows on BOTH architectures.** The earlier "rows
are free on Zen 5" law came from a single-expert rows-curve fit (flat to
T=32) and does NOT transfer to multi-expert serving calls — at the call
level, doubling rows costs ~80% at near-equal weight bytes even on true
512-bit Zen 5. (Also of note: the 9655's 12-channel platform reads
124-131 GB/s at serving shapes — 2.5x the TR — yet its in-executor wall
is only ~25% better; per-call structure, not raw DRAM, governs.)

## Arms

| host | arm | B=8 | B=16 |
|---|---|---|---|
| Zen 4 (axis run) | CTRL | — | 32.1 ms / 0.608 |
| Zen 4 | THIN8 | — | 25.7 / 0.731 |
| Zen 4 | FX30 | — | 28.4 / 0.680 |
| Zen 5 (close run) | CTRL | 13.1 / **0.873** | 22.9 / 0.688 |
| Zen 5 | THIN8 | 14.0 / **0.932** | 23.0 / 0.698 |

- **B=8: over the bar on a THIRD host class** (0.87-0.93 here; 0.92-0.98
  on the reference TR; the clause's B=8 half is closed beyond argument).
- **B=16: still open** — best 0.731 (Zen 4 THIN8, partially confounded)
  and 0.688-0.698 clean on Zen 5; the wall needs another ~14%.
- Structural constraint confirmed: VRAM is capacity-clamped (4045), so
  the solver cannot rebalance B=16 by moving experts — the CPU tier must
  get faster per row, or serve fewer rows (thin/offload routing).

## What closes B=16 (ordered)

1. **Attack the per-row call cost in the kernel**: the row chunking
   re-streams activations per column tile; cache-blocking columns
   against row chunks (tile N so x stays L1-resident across a column
   group) is the standard fix and the receipts' scaling signature
   (rows-cost at equal bytes) is its exact fingerprint.
2. **Row-aware routing**: the offload_rows switch exists; a per-call
   rows/expert threshold routing high-row calls to the GPU stream at
   B>=16 trades PCIe for row compute (bounded win — the GPU side has
   ~4 ms of headroom at B=16).
3. **Bare-metal counters** (Latitude) for the residual interference
   term from the intracall receipts.

Receipts here: `g8_{CTRL,THIN8}.json`, `g8_diag_b{8,16}.json`,
`calib.json`, `rows_curve.json`; Zen-4 twin set in `../b16axis/`.
