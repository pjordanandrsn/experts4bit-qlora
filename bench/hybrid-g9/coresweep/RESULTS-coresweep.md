# Worker-count sweep: extra cores hurt decode, help prefill — so the runner now sizes the pool per phase

Same host as the re-measure (TR PRO 7975WX 32c/64t + RTX 5090, re-rented;
receipts here, box destroyed + verified). Question under test: is 32
workers — the optimum from the quota'd Zen5 U-curve — leaving cores on
the table on a bare 64-thread part?

## Kernel-level U-curve (this host, ~30 MB serving-shape calls)

```
workers=64: cold 1411 us | hot 672 us (28.1 GB/s)
workers=32: cold 1392 us | hot 470 us (40.2 GB/s)   <- optimum again
workers=16: cold 1876 us | hot 822 us
```

## In-executor, two-call arm, B=8/16

| workers | dram ms (8/16) | ratio (8/16) | G9 agg | G9 TTFT p50 |
|---------|----------------|--------------|--------|-------------|
| 32 | 40.6 / 41.1 | 0.448 / 0.461 | **44.0** | 17.6 s |
| 48 | 110.3 / 129.3 | 0.163 / 0.147 | 30.9 | **6.3 s** |
| 64 | 118.0 / 134.8 | 0.157 / 0.140 | 30.4 | **6.3 s** |

Decode: oversubscription beyond the 32 physical cores is a 2.7x
REGRESSION in-executor (far worse than the bare U-curve's 1.4x —
spinning workers on SMT siblings fight torch's own threads once real
GPU/python work runs concurrently). w32 reproduces the re-measure
within ~5% (0.448/0.461 vs 0.427/0.455) — the instrument's band.

**The inversion: TTFT improved 2.8x at 48/64 workers** while decode got
worse. Prefill-phase steps ran 2.65 s at w32 vs 1.26 s at w48/64 on
identical schedules (71 steps, 4096 prefill tokens, zero queue wait).

## What the probes ruled out

- Mixed-mode prefill IS engaged in the engine: an engine-path probe
  (one prompt through scheduler+runner) shows `dram_ns = 0.0` during the
  prefill drain, tiers bound (48/48), flags restored after.
- A bare 512-token mixed-mode chunk is pool-size-INVARIANT (897 vs
  898 ms at 32/64 workers, dram_ns 0) — and CPU-tier prefill
  (gpu_only=False) is 2.6-2.8 s, which matches the gate's w32 chunk
  time. The mechanism that makes the GATE's prefill-phase steps
  pool-size-sensitive despite dram_ns=0 in isolation is NOT resolved
  (candidates: routing breadth of real text vs the probe's degenerate
  prompt; resident decoders sharing drain-phase steps). Open diagnostic;
  operationally mooted by the fix below.

## The fix, validated on the same box

`threads` is a per-call kernel parameter, so `PagedModelRunner` now
takes `prefill_threads` and flips every tier's `_threads` around each
regime (decode restores the enable-time value). One run, decode=32 +
prefill=64:

```
G9 aggregate 44.1 tok/s (= w32's best)  TTFT p50 6.31 s (= w48's best)
```

Both bests in one config; TTFT 17.6 -> 6.3 s at zero decode cost.

Receipts: `pool_floor_zen4.txt`, `g8_w{32,48,64}.json`,
`g9_w{32,48,64}.json`, `g9_phased.json`, `calib.json`,
`rows_curve.json`.
