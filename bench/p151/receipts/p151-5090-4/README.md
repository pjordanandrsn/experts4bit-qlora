# `p151-5090-4`: the served read (NOISY: a VOID and strike one)

One RTX 5090 (cc 12.0, 170 SMs, driver 595.71.05, a **600 W** power limit) on an AMD host (128 CPUs): Vast instance
56714196 on machine 45379, 2026-10-19T10:43:48Z to 11:14:04Z. Cost $0.429.
- **Code:** e4b `63b21b13` (Amendment 1's merge, #1819), grouped-nf4-gemm `9b824155` (K35's merge, 0.47.1 + `dense_gemv`).
- **Stack:** torch 2.8.0+cu128, triton 3.4.0, transformers 5.17.0; Qwen/Qwen3-30B-A3B @ `ad44e777`.

The install, tripwire, self-tests and premise passed (31 passed, K35's compiled contract on this card among them). All
eight arms and both SANE phases completed with rc 0, and the engagement was on shape in every record. S1 routed all 96
projections at one row with no one-row fallback; S0 swapped nothing.

**Verdict: NOISY**, a VOID and the lane's strike one (`verdict_p151.json`). The W1 self-pair S0b/S0a read **1.0345**,
outside [0.985, 1.015]. S0a, the box's first process, ran slow:
- W1 at 252.86 tok/s, against 261.59 / 261.51 / 261.59 for S0b, S0c and S0d;
- its 160-token walls 0.749-0.755 s, against 0.732 s;
- its load 50.5 s, against about 31 s for every later arm.

Its timed blocks show no throttle: the power limit read 600 W either side of each, SM clocks 2917-2925 MHz, 51-55 C.
Every other self-pair agrees within 0.06 % (W16 within 0.11 %).

Reported, never ruling (a NOISY reads no flip):

| W1 pair | S1a/S0a | S1b/S0b | S1c/S0c | S1d/S0d |
|---|---|---|---|---|
| ratio | 1.0603 | 1.0244 | 1.0246 | 1.0247 |

- g16 0.99991 (pairs 0.9999-1.0013).
- SANE: bias +0.000457, argmax 0.9681, KL 0.0081 (inside A2's band), from `sane_off.json` and `sane_on.json` with the
  reducer's own formula (the reducer stops at NOISY before SANE).
- W1 tokens: S1 identical to S0 at both lengths.

The launcher's receipt and ledger row are in the private receipt store. `SHA256SUMS` covers every file here. Twelve
logs carry carriage returns from the box's progress output; they stay only in that store, byte-exact, and
`STORE-ONLY.sha256` lists them.
