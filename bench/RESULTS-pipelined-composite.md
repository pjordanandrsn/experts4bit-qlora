# Pipelined-engine composite — gpt-oss-20b, NUMA-pinned — 2026-07-20 (C2)

The composite cell the earlier A/B never ran: the **production pipelined
engine** (`enable_pipelined_residency` — one pinned arena + address-dispatched
gather + device-id GEMV) with **routing-informed hot sets** and **NUMA
affinity pinning** (`taskset -c 0-3`). Driver: `bench/bench_gptoss_hybrid.py
ENGINE=pipelined HOT_MODE=informed`; receipts in `bench/receipts-pipelined-20260720/`.

Box: RunPod SECURE **L40S** 46 GB, AMD EPYC 9354, e4b @ the
`bench/pipelined-composite-4090` tip. gpt-oss-20b, routed k=4 (`K_SLOTS=4`),
128-token greedy decode, all cells `taskset -c 0-3`.

## Same-box results (all three cells, one L40S)

| cell | decode tok/s | peak GPU GB | informed coverage |
|---|---|---|---|
| pipelined **informed** K=8 | **17.61** | 18.43 | 68.9 % |
| pipelined naive K=8 | 14.67 | 18.43 | (ids 0..7 = 24.3 %) |
| pipelined K=0 (pure stream) | 17.08 | 15.75 | — |

## Reading (same-box, so these comparisons are clean)

- **The informed-hot-set law holds on the production engine:** informed K=8
  is **+20 %** over naive K=8 at *identical* VRAM (17.61 vs 14.67), with the
  informed top-8 covering 68.9 % of routed selections vs naive's 24.3 %. This
  is the same law the v0 receipts showed (`RESULTS-informed-hotsets.md`),
  now confirmed on `enable_pipelined_residency`.
- **K=0 pure-stream (17.08) ≈ informed K=8 (17.61) on this box** — a genuine
  finding, not a null result: for gpt-oss (k=4 routing) on a **fat-PCIe L40S**,
  the pipelined stream is fast enough that keeping 8 experts hot buys almost
  nothing, because the PCIe stream isn't the bottleneck here. The residency
  dial pays where the bus is thin or the CPU is weak (the regime map in
  `RESULTS-gptoss-hybrid-ab.md`), not on a fat-pipe server card — the dial's
  value is a property of the *host*, not the engine.

## Two honest caveats (do not over-read)

1. **VRAM here is NOT lean (peak 15.7–18.4 GB).** The pipelined engine keeps
   the module's base packed weights resident because its **prefill (T>1)
   fallback reads them** — so the driver-level free-base trick (valid for the
   v0 hot path, which never touches base in decode) can't apply. A pipelined
   configuration that actually fits a 12 GB card is the **offload+hot compose**
   library increment, not this bench. These numbers are a *speed* result, not
   a fits-small-card result.
2. **No cross-engine claim.** The v0 hot cell (8.83 tok/s informed+pinned) was
   measured on an **A5000**, these pipelined cells on an **L40S** — different
   silicon, so 8.83→17.6 is **not** a clean engine A/B and no "pipelined is 2×
   hot" claim is made. A same-box hot-vs-pipelined pass is a cheap follow-up if
   that comparison is ever wanted; it is not needed for the informed-hot-set
   law, which is established same-box above.

## Ops

One L40S pod (`rzxl99wg64w2r1`), ~$0.4, torn down on evidence-complete and
404-verified. It re-ran only the pipelined cells after the first composite pod
(A5000) crashed them on the free-base bug (fixed: skip free under pipelined,
`bench/bench_gptoss_hybrid.py`); the hot + llama comparators from that pod are
in `RESULTS-informed-hotsets.md` / `RESULTS-gptoss-hybrid-ab.md`.
