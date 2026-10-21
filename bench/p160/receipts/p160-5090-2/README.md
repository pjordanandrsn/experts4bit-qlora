# `p160-5090-2`: decode x plan (GO)

One RTX 5090 (cc 12.0, 170 SMs, a **575 W** power limit, enforced 575 W in every sample) on an AMD Ryzen 7 9800X3D host
(16 CPUs, 91 GiB RAM, 60 GB disk), machine 147099, 2026-10-20T10:08:03Z to 10:14:35Z. Cost **$0.0710**, against the
registered worst of $1.16.
- **Code:** e4b `2267635d` (the registration's merge, #1870), grouped-nf4-gemm v0.50.0 at `41ef086e`.
- **Stack:** torch 2.8.0+cu128, triton 3.4.0.
- **The run id before it:** `p160-5090-1` was refused at create (the pinned offer was taken between the search and the
  create). No instance existed, it cost $0, and it is not a strike.

## Before the bench

- **The floors:** the power limit read 575 W before the install. The guard fit read `fit=1800s need=2250s left=3534s`.
- **The self-tests:** the bench 15 checks, the reducer 44.
- **The premise:** 27 passed on this card: K25's row-exact GPU tests (4) and K39's compiled contract (23).

## The bench

260 s (S1 173 s, S0R 83 s), 16.3 GB of stores, 41,523 clock samples. Medians of each arm's per-rep median (ms, both GEMMs
of 48 layers):

| block | P | P2 | C1 | C2 | C3 | T |
|---|---|---|---|---|---|---|
| S1 (126 reps) | 10.348 | 10.340 | **6.231** | 6.422 | 6.343 | 12.061 |
| S0R (60 reps) | 10.379 | 10.392 | 6.304 | 6.496 | 6.411 | 12.156 |

## The verdict

**GO** (`verdict_p160.json`): on S1, C1's median P/C1 is **1.659827**, SE (of the median) 0.0015, lower bound
**1.656785**, clock-adjusted **1.652477**, all >= 1.05.

| S1 | median | SE | lower bound | clock-adjusted | per rep |
|---|---|---|---|---|---|
| C1 over P | **1.6598** | 0.0015 | 1.6568 | 1.6525 | 1.6401-1.7021 |
| C2 over P | 1.6100 | 0.0017 | 1.6066 | 1.6066 | 1.5910-1.6600 |
| C3 over P | 1.6304 | 0.0016 | 1.6273 | 1.6230 | 1.6084-1.6745 |
| T/P | 1.1660 | | 1.1648 | | |
| P/P2 (self-pair) | 0.9999 | | | | 0.9952-1.0040 |

- **The S1 splits (reported):** C1 reads 1.6557 on steps 64-125 (62 reps, the half least like S0) and 1.6594 without
  P156's selection step numbers (118 reps).
- **S0R (reported):** C1 1.6508, C2 1.6002, C3 1.6204, T/P 1.1703, self-pair 1.0001.
- **The contract:** every arm kept T's bits, and P2 kept P's, on all 186 reps.
- **Clocks:** no arm is labelled. On S1 every arm's median SM clock read 2685-2692 MHz (C1 1.0026 of P's), and the
  memory clock 13801 MHz in every arm. The software power-cap event (0x4) is in 0.894-0.897 of each arm's S1 samples, the
  same for every arm. The enforced limit read 575 W in every sample.
- **Compiled (sm_120):** P 64 registers and 6,400 B of shared memory; T 80 and 14,592 B; C1 gate_up 194 and 17,408 B,
  down 127 and 26,624 B; no arm spills.

## Files

- **Here:** `bench_p160.json`, `verdict_p160.json`, `summary.txt`, `versions.txt`, `gpu_phases.csv`, and the bench,
  premise and pip logs. `SHA256SUMS` covers each file in this directory.
- **Only in the private receipt store,** byte-exact, listed by sha256 in `STORE-ONLY.sha256`: `forensics.txt` and the
  runner's console log (they carry the GPU's UUID; no reading depends on them), and the launcher's instance record and
  teardown proof (they name the provider's instance).
- **No identifiers:** no file here names an instance, an offer, a host name, an endpoint or the GPU's UUID.

The verdict re-derives byte for byte from the bench record with the registered reducer:

    python bench/p160/p160_reduce.py --bench bench/p160/receipts/p160-5090-2/bench_p160.json --out verdict.json

Re-derive: `python bench/p160/p160_reduce.py --bench bench/p160/receipts/p160-5090-2/bench_p160.json --out {out}` writes `bench/p160/receipts/p160-5090-2/verdict_p160.json`
