# Bar confirmation on reference silicon: B=8 balance 0.978 — the clause's B=8 half is closed

Reference TR PRO 7975WX (32c) + RTX 5090, $0.936/hr, capped harness,
the full merged stack (work-stealing gnf4#110 + thin routing e4b#159 +
fixed=55 + spin window gnf4#111). Exactness on-box 39/39 x2. Box
destroyed; SSH-refused + API-null verified.

| arm | B=8 dram / gpu -> ratio | B=16 dram / gpu -> ratio |
|---|---|---|
| spin 0 (default) | 16.87 / 18.28 -> **0.923** | 35.36 / 19.99 -> 0.565 |
| spin 1500 | 17.76 / 18.16 -> **0.978** | 31.88 / 19.71 -> **0.618** |

- **B=8: the 0.80 bar is crossed on both arms — 0.978 at spin 1500 is
  balance within 2.2%.** The day's stacked levers took this host class
  from 0.124 (yesterday's baseline) through 0.675 (adjudication,
  pre-stealing) to 0.92-0.98: concentration, hoist, torch cap, thin
  routing, work-stealing, spin window — each landed with receipts.
- **B=16 remains open at 0.618** (dram 31.9 vs the <=24.6 the bar
  demands): the batch grows touched-expert count and rows/expert, and
  the wall grows faster than the GPU side. The clause requires BOTH
  batches, so G8 stays formally open with exactly one axis left.
- Spin-0 already reading 0.923 here (vs 0.563 on the 7965WX) says this
  host instance's idle-entry cost is milder; spin 1500 still helps
  B=16 (35.4 -> 31.9) and never hurts. Serving default: ~1500.

## What closes B=16

dram(B=16) needs 31.9 -> ~24.6 ms (-23%). The B=16 wall's growth is
touched-expert amortization already modeled by the solver — candidates:
re-derive cpu_us_fixed at B=16 shapes (the 55 us constant came from
B=8-class calls), deepen thin routing's threshold for B=16 (more
layers qualify when more experts are touched per call elsewhere), and
the fat-call bandwidth ramp (larger calls at B=16 SHOULD amortize
better — that they do not, entirely, points back at the per-call ramp
measured in the intracall receipts).

Receipts: `g8_spin{0,1500}.json` (per-layer histograms embedded),
`calib.json`, `rows_curve.json`.
