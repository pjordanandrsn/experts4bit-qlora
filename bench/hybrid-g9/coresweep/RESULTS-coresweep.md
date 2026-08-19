# Worker-count sweep: extra cores hurt decode; the TTFT signal was a first-run compile artifact

Same host as the re-measure (TR PRO 7975WX 32c/64t + RTX 5090,
re-rented; box destroyed + verified). Question under test: is 32
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
| 48 | 110.3 / 129.3 | 0.163 / 0.147 | 30.9 | 6.3 s |
| 64 | 118.0 / 134.8 | 0.157 / 0.140 | 30.4 | 6.3 s |

**Decode: STANDS.** Oversubscription beyond the 32 physical cores is a
2.7x in-executor regression (far worse than the bare U-curve's 1.4x —
spinning workers on SMT siblings fight torch's threads once real
GPU/python work runs concurrently). The w32 arm ran FIRST and measured
BEST — opposite the warm-up direction — so order cannot explain it.
w32 reproduces the re-measure within ~5%.

**TTFT: RETRACTED as a pool-size effect.** The apparent 2.8x TTFT win
at 48/64 workers is a run-ORDER artifact, established three ways:
1. mixed-mode prefill makes NO CPU kernel calls (engine-path probe:
   `dram_ns = 0.0` through scheduler+runner), and per-call `threads` is
   ignored under an active pool anyway (`pool_run` broadcasts to all
   `P.n`) — there is no mechanism for pool size to touch prefill;
2. a bare mixed-mode chunk is pool-size-INVARIANT (897 vs 898 ms at
   32/64 workers), measured late in the box's life;
3. every ~17 s TTFT in the program's history was a FIRST-on-box G9 run
   (pre-fix 12.2 s, remeasure 17.3 s, this sweep's w32 17.6 s); every
   ~6.3 s reading ran after other processes had warmed the box. A
   per-phase-threads runner patch — mechanically inert per (1) — also
   "reproduced" 6.31 s when run last, which is what exposed the confound.

Likely mechanism: the first process to run PREFILL shapes pays triton
compile/autotune inside the TTFT window (decode/balance shapes are
warmed by earlier arms; prefill T=512 shapes are not). The same
order-correlated bimodality explains the fixbox B-vs-C spread (105 vs
49 ms on provably identical configs). Consequence for the instrument,
not the engine: **the gate runner must warm BOTH shapes (one throwaway
prefill chunk + one decode step) BEFORE arrival timestamps start TTFT
clocks.** With a warm box, TTFT p50 ~6.3 s is the true current number —
still bounded below by serial B=1 chunk forwards (~1.1 s x 4.5 prompts
at p50), which prefill batching addresses.

## Standing conclusions

- Pool size 32 = physical cores, both host classes, decode-confirmed.
- G8's remaining ~2.2x rests on the per-worker subset wake (cut the
  470-1392 us per-call floor), not on more workers.
- G9's path: non-expert step decomposition at serving scale + batched
  prefill; TTFT measurement discipline fixed per above.

Receipts: `pool_floor_zen4.txt`, `g8_w{32,48,64}.json`,
`g9_w{32,48,64}.json`, `g9_phased.json` (the confound exposer),
`calib.json`, `rows_curve.json`.
