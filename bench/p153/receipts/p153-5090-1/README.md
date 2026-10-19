# `p153-5090-1`: the served read (FLIP)

One RTX 5090 (cc 12.0, 170 SMs, driver 615.71.09, a **575 W** power limit, enforced 575 W) on an AMD Ryzen 7 9800X3D
host (16 CPUs, 91 GiB RAM): Vast instance 56745420 on machine 147099, 2026-10-19T15:00:37Z to 15:23:19Z. Cost $0.445,
at the registered fallback tier ($0.7915/h including 170 GB, quoted at the pin).
- **Code:** e4b `585cf68b` (the registration's merge, #1826), grouped-nf4-gemm `929a6906` (K36's merge).
- **Stack:** torch 2.8.0+cu128, triton 3.4.0, transformers 5.17.0, bitsandbytes 0.50.2; Qwen/Qwen3-30B-A3B @ `ad44e777`.

The power floor, install, tripwire and self-tests passed: the reducer 52 cases, the box 23, P109's box 5, P115's quality
instrument 11. The premise read 28 passed, K36's compiled contract on this card among them. The warm-up process, all
eight arms and both SANE phases completed with rc 0.
- **On shape:** S1 swapped one module, the `[151936, 2048]` head (311,164,928 int8 bytes). It routed at one row and
  passed at 2 / 4 / 8 / 16 / 512 rows, with no one-row fallback. S0 swapped nothing.
- **Off in every arm:** the dense attention GEMV. The split cap read its `qwen3_moe` default of 16.
- **Power:** every timed block read 575 W either side of it, at most 59 C, with SM clocks at least 2760 MHz in every arm.

**Verdict: FLIP** (`verdict_p153.json`): g1 **1.052019** (the min of four W1 pairs) ≥ 1.03 and g16 **0.991794** ≥
0.99, both unrounded.

| | S1a/S0a | S1b/S0b | S1c/S0c | S1d/S0d |
|---|---|---|---|---|
| W1 | 1.0530 | 1.0520 | 1.0522 | 1.0527 |
| W16 | 0.9918 | 0.9970 | 0.9924 | 0.9929 |

- **Self-pairs:** W1 0.9993-1.0003; W16 0.9958-1.0010.
- **The warm-up S0w** (reported, never scored): W1 263.38 tok/s, 0.9989 of S0a. W16 1.0004 of S0a. Its load took 38.0 s,
  against 20.6-21.8 s for every later arm.
- **SANE:** bias +0.002158, argmax 0.988932, KL 0.000601 over 12 windows.
- **Tokens:** at W1, S1 is identical to S0 at both lengths. At W16, 11 of 16 rows are identical at 32 tokens and 4 of
  16 at 160 (the knob rounds the head at every row count).
- **Memory at load:** S1's `memory_allocated` is 311,772,672 bytes above S0's, exactly the int8 codes plus their fp32
  scales. Its `memory_reserved` is 400 MiB above (nvidia-smi 24487 vs 24087 MiB).

The launcher's receipt and ledger row are in the private receipt store. `SHA256SUMS` covers every file here. Thirteen
logs carry carriage returns from the box's progress output; they stay only in that store, byte-exact, and
`STORE-ONLY.sha256` lists them.
