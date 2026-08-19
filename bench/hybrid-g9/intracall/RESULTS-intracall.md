# Intra-call frontier, scoped: call-size bandwidth ramp + an unexplained context-interference term; two mechanisms refuted

TR PRO 7965WX + 5090, capped harness, real 30B stacks. Instrument:
`g8_diag.py` extended with cold/warm kernel-only modes (the original
re-timed ONE expert set — warm after iter 1 — a silent bias, now
split), plus a gap-injection probe. Box destroyed; verified.

## Established

1. **Call size is the bandwidth determinant.** Serving-size calls
   (~16 MB, 6 uniques) achieve 19-23 GB/s at the KERNEL level, where
   30 MB U-curve jobs reach 40-50. The executor's in-situ 16.8-17.7
   GB/s is already ~75-80% of what the kernel itself delivers at these
   shapes — the earlier "executor loses 2x" framing was a cross-shape
   misextrapolation.
2. **L3-warmth refuted**: cold (fresh expert set per iteration) vs warm
   (same set re-timed) differ ~15% at most (18.9-22.9 GB/s) — nothing
   like the 2x the working-set argument predicted.
3. **Spin-window refuted**: injecting 0-3000 us gaps between gu and dn
   produces no cliff at the ~100 us worker spin boundary (dn holds
   600-900 us throughout). The first call after ~5 ms idle pays ~2.5x
   (gu 1.5 ms cold vs dn 0.6 ms warm at comparable work) — a real but
   bounded cold-entry cost.
4. **An unexplained context-interference term remains**: inside the
   full phase harness (torch ops + CUDA transfers interleaved), the dn
   call reads 4138 us — 5-6x its isolated cost — while gu reads 792.
   Not reproduced by gaps alone; needs hardware counters
   (perf/CAP_PERFMON), which rented containers do not grant.

## Consequence for G8

The remaining 1.5x to the bar decomposes into: per-call bandwidth ramp
(fatter calls would help — but placement concentration is already at
its useful limit), the ~2.5x cold-entry cost on the first call after
each layer's GPU phase, and the interference term. Next moves, in
order: (a) keep workers warm ACROSS a step (one pool "session" spanning
a forward's ~32 calls — e.g. a step-scoped spin extension or a
dedicated streaming thread), (b) bare-metal profiling session (Latitude
grants perf) for the interference term. Both are kernel-side; the
executor and instrument are done.

Receipts: `g8_diag.json` (cold/warm + phases), `calib.json`; probe
outputs quoted above from the live session log.
