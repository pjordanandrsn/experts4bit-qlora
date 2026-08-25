# AMENDMENT — F1 Stage B, arm B2: the fused KV append, concretely

Registered 2026-08-25, before any B2 measurement. The Stage B prereg
registered B2's existence and its ordering (only after B1 lands short
of PASS — it did, PARTIAL at 10.62 ms); this fixes its mechanism and
bars.

## Mechanism

`append_graph_t1`'s eager body is ~25 launches per layer (two 5-kernel
quantizes, ~10 address-math ops, 4 scatters), all at the ~1.2 µs launch
quantum. gnf4's `fp8_kv_append_t1` (gnf4#253) replaces each side with
ONE launch: address math from device state (seq_lens + block table),
fp32 group amax, `amax/E4M3_MAX` with all-zero groups pinned to 1.0,
saturating-RNE e4m3 cast, payload + fp32 scales stored in place. The
`seq_lens` publish stays the separate in-stream op it is today, so
ordering against the attention kernel is unchanged. Per layer: 25
launches → 3. Opt-in via `E4B_FUSED_KV_APPEND=1`, default OFF until
the B2 RESULTS merge.

## Arms

Same box, one session, merged main both repos, clean tree asserted:

- **C0 ×2** — the shipped Stage B config (`--compile-layers
  --compile-mode default`, graph loop), twice for A/A.
- **C1** — C0 + `E4B_FUSED_KV_APPEND=1`.

## Bars (fixed now, from the census, before measurement)

The addressable block is 1.58 ms of fp8 KV work; some quanta remain
(2 fused launches + the publish), so the honest expectation is
~1.0–1.4 ms realized.

- **PASS**: C1 step ≤ 9.6 ms (≥ 104 tok/s) — ≥ 1.0 ms removed.
- **PARTIAL**: 9.6–10.5 ms, ships only if the A/A spread is under half
  the gain.
- **REFUTED**: > 10.5 ms — no meaningful gain over B1; the fused
  kernel does not ship.

## Refusals (any one voids the arm)

0. **Bitwise pool gate, on-box, before any timed arm**: gnf4's
   `test_fp8_kv_append.py` must pass on the box's GPU (the e4m3 cast is
   hardware `cvt.rn.satfinite`; certifying it anywhere else certifies
   the wrong instruction), AND a 32-step fused-vs-eager pool comparison
   on the real cache must match byte-for-byte with a nonzero
   touched-byte count (anti-vacuity).
1. Token identity C1 vs C0 — full-length, element-wise (the Stage B
   gate, unchanged).
2. A/A before A/B — C0 spread must be under half the PASS margin.
3. Capture neutrality — same step count both arms.
4. No dynamo recompiles inside the timed window.

`f1_verdict.py` adjudicates with `treatment_name = "B2"` and the bars
above via its `pass_ms`/`partial_ms` overrides; self-test covers the
override path in both directions.
