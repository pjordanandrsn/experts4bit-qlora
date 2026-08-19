# The clean adjudication: G8 at 0.675 / 0.564, G9 TTFT 4.9 s — every confound removed

Quiet TR PRO 7975WX + RTX 5090 (the reference host, re-rented), capped
harness (`torch.set_num_threads(8)` baked in per the gate1 thrash law),
tee'd stage outputs, warm-both-shapes G9. Exactness on-box: 38/38
kernel x2, 9/9 hybrid. Box destroyed; SSH-refused + API-null verified.

## G8 balance (bar 0.80)

| arm | B=8 dram / ratio | B=16 dram / ratio |
|---|---|---|
| this morning (pre-fix, same host class) | 160 ms / 0.124 | 236 ms / 0.074 |
| CTRL (capped, concentrated, 32w) | **26.6 ms / 0.675** | 40.4 ms / 0.473 |
| THIN (+ static thin-layer routing) | 28.4 ms / 0.650 | **33.8 ms / 0.564** |

- The torch cap bought ~35% even on this "accidentally immune" host
  (41 -> 26.6 ms) — the thrash law was seasoning every number all day.
- Thin routing reproduces its same-batch B=16 win exactly (+16%,
  matching the sick box's delta); at B=8 it is a wash here (the calls
  it rescues are cheaper on a quiet capped host).
- **Verdict: still a MISS, now by single-digit milliseconds** — the bar
  needs dram <= 22.4 ms at B=8 (4 ms away) and <= 23.9 at B=16 (10 ms
  away). Day's arc: 0.124/0.074 -> 0.675/0.564 (5.4x / 7.6x).

## G9 (bar 140 tok/s)

**43.3 tok/s aggregate, TTFT p50 4.91 s** — the program's best TTFT
(17.6 s this morning), delivered by warm-both-shapes + the cap + thin
routing. The aggregate's remaining distance is owned by non-expert step
time (attention + engine at B=8 x 48 layers) and serial B=1 prefill,
as decomposed in the fixbox receipts.

## What closes the last 4-10 ms of G8

The remaining DRAM wall is ~27-34 ms for ~178 unique experts/step:
per-expert ~150-190 us against a ~90 us bandwidth floor (2.65 MB at
the achieved 17.7 GB/s x 32w). Candidates, in order: deeper placement
concentration (fewer fat calls), per-call floor structure inside the
fat calls (the U-curve hot floor is ~470 us/call on this host), and
bus overlap. Each is a bounded, measurable step — no unknowns left in
the instrument.

Receipts: `g8_CTRL.json`, `g8_THIN.json` (per-layer histograms
embedded), `g9_gate.json`, `calib.json`, `rows_curve.json`.
