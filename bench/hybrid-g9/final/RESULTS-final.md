# Stage-2 gate standing after the full stack: G8 open at 0.51-0.68, G9 at 45.4 tok/s — every identified lever landed and measured

TR PRO 7965WX (24c/48t, 8-channel) + RTX 5090, $0.936/hr, capped
harness, full stack (work-stealing + fused + thin routing + fixed=55 +
warm G9). Exactness on-box 38/38 x2, 9/9. Box destroyed; SSH-refused +
API-null verified. This closes the day's measurement campaign.

## G8 balance across serving hosts, full stack

| host | best B=8 (dram/gpu -> ratio) | best B=16 |
|---|---|---|
| 7975WX 32c (adjudication, pre-stealing) | 26.6 / 17.9 -> 0.675 | 33.8 / 19.1 -> 0.564 |
| 7965WX 24c (this run, full stack) | 28.3 / 15.7 -> 0.556 | 39.1 / 16.3 -> 0.416 |
| pool 24-vs-32 on the 24-core host | wash (within the +-10% band) | wash |

Two structural facts the table teaches:
1. **The DRAM wall converges at ~26-31 ms/step at B=8** (~178 unique
   experts, ~155-175 us/expert against a ~90 us bandwidth floor at the
   achieved 16-18 GB/s) on every healthy serving host, whatever the
   lever mix — the remaining cost is INSIDE the fat calls, not in call
   count, placement, partition, or thread interference anymore.
2. **The balance ratio penalizes GPU-side speed**: this box's faster
   fused-GPU path (15.7 vs 17.9 ms) LOWERS the ratio at an equal DRAM
   wall. The bar therefore moves with the GPU: 0.80 here means
   dram <= ~19.6 ms — a ~1.5x intra-call improvement (streaming/NT
   loads, per-CCD tiling of the expert bytes) is the remaining frontier.

Fused-vs-two-call on this host: a wash at B=8 (28.3 vs 28.4-30.9),
fused slightly behind at B=16 — consistent with gate-2's finding that
stealing removed the penalty; the +22% seen on the wobbly EPYC did not
reproduce on TR silicon. `fused_ffn` default stays False; the fused
call remains the opt-in for floor-heavy hosts.

## G9

**45.4 tok/s aggregate — program best** (23.9 this morning), TTFT p50
7.88 s (warm fix holding; the 4.91 s reading remains the best observed).
The bar is 140: the decomposed remainder is unchanged — non-expert step
time (attention + engine python at B=8 x 48 layers) and serial B=1
prefill chunks. Those are Phase-10-class engine work, not tier work.

## Day ledger (2026-08-18 -> 19)

G8: 0.124/0.074 -> 0.675/0.564 best-host, 0.51-0.68 across hosts.
G9: 23.9 -> 45.4 tok/s; TTFT 12-17 s -> 4.9-7.9 s.
Mechanisms landed: placement concentration, dequant hoist, three KV
stream-sync removals, torch-thread cap (12x on big-core boxes), thin
routing, subset engagement + race fix, work-stealing. Instrument laws:
warm both shapes, pin torch intraop, tee stage outputs, probe
claimed-dead boxes directly, pool = physical cores.

Receipts: `g8_{TWOCALL,FUSED}{,24}.json`, `g9_gate.json`, `calib.json`,
`rows_curve.json`.
