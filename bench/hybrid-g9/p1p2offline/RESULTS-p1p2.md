# P1 and P2 run down offline — and the receipts sharpen where the −14% lives

Scoring the first two registered predictions of
`docs/hybrid/OBJECTIVE-REVISION-2026-08-23.md` from receipts already in
this repo (`b16close/`), plus the finding the accounting surfaced. No box
was rented; every number below is from committed receipts.

## P1 — headroom-gating A/B: condition FALSE, closed by accounting

The registration made P1 conditional: "**if** overlapped bytes/step ≈
prefetch+KV traffic is material, this is a chunk of the −14%." The
receipts answer the condition:

* CPU-tier weight traffic at B=16: 188 unique DRAM experts × 3.54 MB
  (`rows_curve.bytes_per_expert`) = **665 MB/step** under the 22.9 ms wall.
* DRAM-crossing traffic that overlaps it (thin-call H2D gathers, prefetch
  landings, KV): bounded above by a few hundred MB/step, and only the
  interleave fraction (κ−1 ≈ 0.3, gnf4 G1c) taxes:
  `0.3 × 300 MB / 212 GB/s ≈ 0.42 ms ≈ 1.8% of the wall` — an upper
  bound; the realistic figure is under 1%.
* gnf4 G4′'s 18% tax arose where storage bytes RIVALLED compute bytes
  per step. Here overlapped copies are ~5-10% of GEMV bytes. Same law
  (L1), different byte ratio — the law predicts its own irrelevance here.

**Verdict: P1's box A/B is registered-uninformative at the G8 B=16
operating point (predicted effect ≤ 2%, inside blip noise) and is not
run.** The objective revision stands — headroom scheduling matters where
DRAM-crossing traffic is material (the tribrid's cold tier; TTFT
prefill bursts) — but it is not the B=16 lever.

## P2 — thin routing vs the per-call cost model: CONSISTENT

The crossover's per-call form is the repo's own fit
(`rows_curve`: a = 182.8 µs fixed + 6.4 µs/row, 3.54 MB/expert): a thin
call (1–2 rows/unique) costs the CPU 189–196 µs/expert, vs a GPU
H2D+kernel cost of ~100–200 µs/expert at the measured 25–56 GB/s pinned
path — **near parity on Zen 5**. The model therefore predicts thin
re-routing moves little there, and that is what the arms measured:
B=16 walls 22.88 → 22.96 ms (CTRL → THIN8, ~0), B=8 13.14 → 14.02 ms
(slightly worse wall, better balance only because the CPU wall moved
toward the fixed GPU wall). Zen 4's −20% THIN8 gain sits where the
per-call floor is worse (its kernel-only rate is 2.4× lower at B=16), on
the favorable side of the same crossover. **Consistent; no anomaly for
the model to explain away.**

## The finding: a 3.5–3.8× executor-structure loss IS the −14% (and more)

Same host, same shape, same stacks:

| | kernel-only | in-executor | ratio |
|---|---|---|---|
| B=8 | 124.3 GB/s | 35.3 GB/s | **3.5×** |
| B=16 | 83.5 GB/s | 21.8 GB/s | **3.8×** |

At the kernel's own achieved rate, the B=16 CPU wall would be
**~8 ms against a 15.7 ms GPU wall** — balance ~1.0+, the clause closed
with room. The bytes are not the wall and the overlap is not the wall:
**per-call structure is** — 188 uniques fragmented across 48 layers
(~3.9 uniques/call, ~14 rows/call), each call paying the fixed floor,
the row-chunking re-stream (`b16close` item 1's fingerprint), the
phase overheads (`g8_diag_b16`: x_transfer 106 µs + sort 63 µs +
scatter_back 696 µs per capture window).

The ordered close-B=16 list in `b16close/RESULTS-b16.md` (cache-blocking
first) is therefore confirmed BY the objective-revision accounting rather
than competing with it — P4 in the revision's terms. The registered
sequencing updates: **P4 (kernel call-structure: cache-blocking columns
against row chunks + per-call floor amortization, gnf4 side) is the
B=16 lever**; P1-class headroom scheduling is deferred to the tiers where
its bytes are material; P3 stands as a pricing constraint whenever a
throttle is next tuned.
