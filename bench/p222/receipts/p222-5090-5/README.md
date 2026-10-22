# `p222-5090-5`: K25's decode x plan at Mixtral-8x7B's W16 shapes (GO)

One RTX 5090 (cc 12.0, 170 SMs, a **575 W** power limit, enforced 575 W in every sample) on an AMD EPYC 9334 host
(128 CPUs, 755 GiB RAM, 60 GB disk), 2026-10-22T17:05:26Z to 17:15:49Z. Cost **$0.14**, against the
registered worst of $1.69.
- **Code:** e4b `683b85f2` (Amendment 3's merge, #1974), grouped-nf4-gemm v0.50.0 at `41ef086e`.
- **Stack:** torch 2.8.0+cu128, triton 3.4.0.
- **The run ids before it:** `p222-5090-1` to `-3` were refused at create (Amendment 3, "Why"), and `p222-5090-4` was
  refused at create when its pinned offer left the market before the create. No instance existed for any of them, each
  cost $0, and none is a strike.
- **How the host was found:** an RTX 5090 host whose 575 W cap was recorded by a power probe the same day. The screen
  admits only machines with a recorded power cap of at least 575 W, read within 3 days, and this one's was first recorded
  that afternoon by a short explore-tier probe: one box booted, read
  nvidia-smi's `power.limit` (575 W), and was torn down for about $0.01, with its receipt in the private receipt store.
  The probe only picked the host. This box read its own limit before the install, and the runner refuses under 575 W
  (rc 17).

## Before the bench

- **The floors:** the power limit read 575 W before the install. The guard fit read `fit=3300s need=4125s left=5310s`.
- **The self-tests:** the bench 17 checks, the reducer 55.
- **The premise:** 27 passed on this card: K25's row-exact GPU tests and K39's compiled contract.

## The bench

444.8 s (the sweep 277.5 s, the reading 165.6 s), 6.34 GB of stores over 8 MoE layers, 23,396 clock samples.

- **Block A, the sweep:** 144 plans a projection under `prmt`, on seed 2220's steps 0, 16, ..., 112. Every plan kept the
  tree's bits, and none failed to compile or launch.
- **The selection (M):** gate_up 64/256/4/2, which is c1's gate_up, and down **32/256/4/2**, where c1's down is
  128/128/8/3. The runner-up over the winner was **1.0154** for gate_up and **1.1293** for down.
- **Block B, the reading:** seed 2221's steps 0-119, 120 reps, 12 Williams squares of five arms. Medians of each arm's
  per-rep median, in ms, for both GEMMs of 8 layers:

| T (tree, today's plan) | T2 | P (prmt, today's plan) | C1 (prmt, c1) | **M** (prmt, the selection) |
|---|---|---|---|---|
| 11.365 | 11.366 | 11.152 | 5.470 | **4.951** |

## The verdict

**GO** (`verdict_p222.json`): M's median T/M is **2.267811**, with SE (of the median) 0.0027, lower bound **2.262375**,
and the same bound clock-adjusted, all >= 1.10. `served_arm` is **M**.

| block B | median | SE | lower bound | clock-adjusted | per rep |
|---|---|---|---|---|---|
| M over T | **2.2678** | 0.0027 | 2.2624 | 2.2624 | 2.2406-2.3461 |
| C1 over T | 2.0630 | 0.0009 | 2.0613 | 2.0613 | 2.0422-2.0849 |
| P over T (the decode alone) | 1.0187 | 0.0002 | 1.0183 | 1.0183 | 1.0135-1.0208 |
| M over C1 | 1.0999 | 0.0009 | 1.0980 | | 1.0897-1.1253 |
| T/T2 (self-pair) | 0.9999 | 0.0001 | | | 0.9983-1.0020 |

- **The served arm, by the registered rule:** M is neither today's plan nor c1's pair, and M over C1 (1.0999) is above
  1.02, so the arm is M.
- **The contract:** P, C1 and M kept T's bits, and T2 kept T's, on all 120 reps.
- **The self-pair band:** T/T2's median 0.9999 is inside 0.99-1.01.
- **Clocks:** no arm is labelled. Every arm's median SM clock read 2707 MHz (range 2640-2820) and its memory clock 13801
  MHz. The software power-cap event (0x4) is in 0.719-0.723 of each arm's samples, the same for every arm, and the
  enforced limit read 575 W in every sample.
- **Compiled (sm_120):**
  - T and T2: 80 registers and 14,592 B of shared memory.
  - P: 64 and 6,400 B.
  - C1: gate_up 194 and 17,408 B, down 127 and 26,624 B.
  - M: gate_up 194 and 17,408 B, down 128 and 12,800 B.
  - No arm spills.

## Files

- **Here:** `bench_p222.json`, `verdict_p222.json`, `summary.txt`, `versions.txt`, `gpu_phases.csv`, and the bench,
  premise and pip logs. `SHA256SUMS` covers each file in this directory.
- **Only in the private receipt store,** byte-exact, listed by sha256 in `STORE-ONLY.sha256`:
  - `forensics.txt` and the runner's console log. They carry the GPU's UUID, and no reading depends on them.
  - The launcher's instance record and teardown proof. They name the provider's instance and machine.
- **No identifiers:** no file here names an instance, a machine, an offer, a host name, an endpoint or the GPU's UUID.

The verdict re-derives from the bench record with the registered reducer
([`../../REDERIVE.md`](../../REDERIVE.md)). Under Python 3.11, CI's version, it is byte-identical to the box's:

    python bench/p222/p222_reduce.py --bench bench/p222/receipts/p222-5090-5/bench_p222.json --out verdict.json

Under Python 3.9 the last digit of four SE fields differs (M's SE in `M` and `read`, and M over
C1's in `read` and `reported`); no value that decides anything moves.
