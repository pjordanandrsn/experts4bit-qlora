# RESULTS — BV3b: parity, the corrected attribution, and the PASS

Adjudicated 2026-08-26 by `bv3b_verdict.py` (amended degeneracy
handling, self-tests green) on `receipts-bv3b/bv3v2_report.json`.
Boxes: parity + floor on instance 48728047 (anchor 7.27 ms); the v2
wall arms on the same box.

## VERDICT: PASS — the batched graph loop ships

```
graph B=16: 38.16 / 38.17 ms   eager B=16: 118.2 / 120.0 ms
ratio 3.10x — 135 -> 419 aggregate tok/s
parity: BITWISE, all 48 layers (worst ratio 0.000e+00)
b1 sanity: 7.28 ms (on class)
identity: 11/16 rows fully token-identical; 4 diverge at >=103
          (registered floor: >=32); row 8 excluded (below)
```

## The attribution arc, corrected in public

1. BV3 refused on early token divergence; RESULTS-bv3 attributed it
   to device-vs-eager grouping numerics. **WRONG** — the parity probe
   (live inputs, T=16, dispatch verified non-vacuous through three
   review rounds of binds) measured the two paths BITWISE-IDENTICAL
   on every layer.
2. The remaining uncontrolled difference was dynamo's recompile
   budget: the graph lane raised it, the eager arms ran the default-8
   fallback — compiled vs fallback kernels differ numerically. With
   limits equalized (#270) on every arm, identical rows went 2/16 →
   11/16, residual divergence moved to step >=103, and the EAGER
   baseline itself got ~9% faster (118 vs 129 ms) — compile coverage
   was distorting the baseline too.
3. The residual 4 late divergences are consistent with
   graph-replay-vs-eager execution differences at greedy near-ties;
   all sit far inside the registered >=32 floor.

## Amendment record (disclosed)

Wikitext 512-token windows produce a looping continuation in ~1 of
16 rows at ANY offset (row 2 at offset 0, row 8 at 20000), and in
every observed case BOTH arms loop IDENTICALLY. Amended: a row
degenerate in both arms with identical streams is excluded and
reported (row 8 here); graph-only degeneration still refuses as
treatment-induced; disagreeing degenerate pairs refuse; >=75% of
rows must remain clean. Four-direction self-test cells.

## Scope

One box class, B=16, fixed 128-step rows (no EOS/early-exit — that
is serving polish, not this measurement). The 421-class number from
the confounded v1 run is superseded by this certified 419 at equal
compile coverage.
