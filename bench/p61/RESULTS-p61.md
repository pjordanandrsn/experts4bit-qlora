# Results — P61: what bounds the served expert GEMV at B=16 (RTX 5090, 2026-09-23)

Pre-registration: [`P61-PREREG.md`](P61-PREREG.md) (#689, merged before the first run). Amendments 1 (#690) and 2 (#691) were both dated before any P1–P4 read. Receipts: [`receipts/`](receipts/) holds the read run's rows, sweep and self-test logs (`sweep_log.txt`, `self_test_log.txt`), versions, forensics, staged pin and teardown proof, plus [`receipts/run1-gate-failed/`](receipts/run1-gate-failed/), run 1's forensics and summary (the P0 evidence only; its rows are never read). Every number below comes from [`p61_reduce.py`](p61_reduce.py), and its output is reproduced unedited.

## Six attempts, one read

| run | machine | card | outcome | cost | receipt (adertha-receipts) |
|---|---|---|---|---|---|
| `p61-5090-1` | 34261 | 5090, **450 W** | rc 0; **P0 FAILS**: served 7.253 vs 6.479 ms/step (+11.9 %), dedup within 1.5 %; not read | $0.0436 | `56c5170` |
| `p61-5090-2` | 53098 | 5090, **520 W** | refused before install (Amendment 1: power.limit < 575 W) | $0.026 | `25b8ba6` |
| `p61-5090-3` | 147454 | — | NOT_RUN: pre-flight, instance stuck `loading` 600 s | $0.0909 | `9763c2a` |
| `p61-5090-4` | 112352 | — | NOT_RUN: pre-flight, ssh key refused 180 s | $0.055 | `1b5478d` |
| `p61-5090-5` | — | — | refused by the launcher before renting (machine exclusion needs the receipt store at its pushed upstream) | $0 | `561bc21` |
| **`p61-5090-6`** | 137563 | 5090, **575 W**, driver 580.119.02 | **rc 0; P0 HOLDS: the read** | **$0.0632** | `4a07e46` |

The lane cost **$0.2787** in total, against its $1 ceiling. On the way it fixed a launcher gap: adertha-agents #128 lets a lane's power-cap refusal (exit 17) exclude its machine. The receipts branch was pushed so exclusions could run.

## The read (`p61_reduce.py`, unedited)

device NVIDIA GeForce RTX 5090 (170 SMs), torch 2.8.0+cu128, L2 96 MiB; 48 layers x 128 recorded steps; eids sha256 c050961e7f7d...; copy bandwidth DRAM 1528 GB/s, L2-resident 5753 GB/s (24 MiB x2); plans (sk by R) {'gate_up': {'1': 16, '2': 16, '4': 16, '8': 16, '16': 16, '32': 16, '64': 16, '128': 16}, 'down': {'1': 6, '2': 6, '4': 6, '8': 6, '16': 6, '32': 6, '64': 6, '128': 6}}

| proj | R | D | rows/expert | graph us/call | gemv us/call | reduce us/call |
|---|---|---|---|---|---|---|
| gate_up | 16 | 1 | 16 | 11.45 | 9.35 | 2.22 |
| gate_up | 16 | 2 | 8 | 12.64 | 9.60 | 2.23 |
| gate_up | 16 | 4 | 4 | 12.90 | 9.86 | 2.21 |
| gate_up | 16 | 8 | 2 | 15.00 | 11.97 | 2.21 |
| gate_up | 16 | 16 | 1 | 22.69 | 19.62 | 2.23 |
| gate_up | 32 | 1 | 32 | 19.85 | 17.35 | 2.17 |
| gate_up | 32 | 2 | 16 | 20.78 | 17.91 | 2.17 |
| gate_up | 32 | 4 | 8 | 21.05 | 18.22 | 2.18 |
| gate_up | 32 | 8 | 4 | 21.51 | 18.71 | 2.19 |
| gate_up | 32 | 16 | 2 | 24.91 | 22.04 | 2.17 |
| gate_up | 32 | 32 | 1 | 44.39 | 41.41 | 2.20 |
| gate_up | 64 | 1 | 64 | 37.06 | 34.13 | 2.47 |
| gate_up | 64 | 2 | 32 | 38.24 | 35.19 | 2.48 |
| gate_up | 64 | 4 | 16 | 38.82 | 35.75 | 2.50 |
| gate_up | 64 | 8 | 8 | 39.43 | 36.35 | 2.52 |
| gate_up | 64 | 16 | 4 | 40.42 | 36.57 | 2.58 |
| gate_up | 64 | 32 | 2 | 48.02 | 45.38 | 2.66 |
| gate_up | 64 | 64 | 1 | 91.12 | 87.54 | 2.88 |
| gate_up | 128 | 1 | 128 | 68.84 | 64.76 | 3.82 |
| gate_up | 128 | 2 | 64 | 70.28 | 66.16 | 3.81 |
| gate_up | 128 | 4 | 32 | 71.20 | 66.94 | 3.82 |
| gate_up | 128 | 8 | 16 | 71.92 | 67.75 | 3.85 |
| gate_up | 128 | 16 | 8 | 72.83 | 68.60 | 3.88 |
| gate_up | 128 | 32 | 4 | 75.12 | 69.93 | 3.94 |
| gate_up | 128 | 64 | 2 | 106.39 | 101.23 | 4.86 |
| gate_up | 128 | 128 | 1 | 185.19 | 177.18 | 7.38 |
| down | 16 | 1 | 16 | 6.35 | 4.89 | 1.11 |
| down | 16 | 2 | 8 | 6.40 | 4.98 | 1.10 |
| down | 16 | 4 | 4 | 7.46 | 5.54 | 1.10 |
| down | 16 | 8 | 2 | 8.49 | 6.62 | 1.10 |
| down | 16 | 16 | 1 | 12.15 | 10.20 | 1.10 |
| down | 32 | 1 | 32 | 9.79 | 8.42 | 1.14 |
| down | 32 | 2 | 16 | 10.22 | 8.40 | 1.15 |
| down | 32 | 4 | 8 | 10.70 | 8.82 | 1.14 |
| down | 32 | 8 | 4 | 11.01 | 9.12 | 1.14 |
| down | 32 | 16 | 2 | 13.07 | 11.16 | 1.14 |
| down | 32 | 32 | 1 | 20.77 | 18.79 | 1.14 |
| down | 64 | 1 | 64 | 17.01 | 15.62 | 1.29 |
| down | 64 | 2 | 32 | 17.35 | 15.63 | 1.31 |
| down | 64 | 4 | 16 | 17.90 | 16.08 | 1.30 |
| down | 64 | 8 | 8 | 18.15 | 16.34 | 1.31 |
| down | 64 | 16 | 4 | 18.67 | 16.88 | 1.32 |
| down | 64 | 32 | 2 | 21.76 | 19.86 | 1.31 |
| down | 64 | 64 | 1 | 38.38 | 36.24 | 1.30 |
| down | 128 | 1 | 128 | 31.87 | 30.21 | 1.80 |
| down | 128 | 2 | 64 | 32.33 | 30.28 | 1.79 |
| down | 128 | 4 | 32 | 32.84 | 30.71 | 1.79 |
| down | 128 | 8 | 16 | 33.16 | 30.98 | 1.80 |
| down | 128 | 16 | 8 | 33.69 | 31.53 | 1.81 |
| down | 128 | 32 | 4 | 35.15 | 33.54 | 1.80 |
| down | 128 | 64 | 2 | 44.57 | 43.47 | 1.82 |
| down | 128 | 128 | 1 | 75.55 | 73.10 | 1.85 |

| proj | fit on | a us | b us/row | c us/expert | R^2 | max rel resid |
|---|---|---|---|---|---|---|
| gate_up | graph | 1.44 | 0.4774 | 0.8655 | 0.9757 | 24.3 % |
| gate_up | gemv kernel | -0.71 | 0.4629 | 0.8414 | 0.9752 | 25.1 % |
| down | graph | 2.42 | 0.2097 | 0.3281 | 0.9791 | 21.0 % |
| down | gemv kernel | 0.71 | 0.2088 | 0.3253 | 0.9813 | 23.3 % |

| proj | b1 (gemv, D=1) us/row | L2 re-read floor us | b1 / floor | c_gemv us/expert | DRAM floor us | c / floor |
|---|---|---|---|---|---|---|
| gate_up | 0.4950 | 0.3076 | 1.61 | 0.8414 | 1.1578 | 0.73 |
| down | 0.2264 | 0.1538 | 1.47 | 0.3253 | 0.5789 | 0.56 |

| recorded routing | served ms/step | dedup ms/step | served - dedup | served gemv kernel ms/step |
|---|---|---|---|---|
| shared store | 6.548 | 5.565 | +0.983 | 6.434 |
| per_layer store | 6.588 | 5.683 | +0.905 | 6.497 |

Recorded distinct experts per call: mean 54.66 (39–87); rows removed per step by dedup: 3520.3.

- **P0 (instrument: shared-store served within ±5 % of P60's 6.479):** HOLDS — 6.548 ms/step (+1.1 %)
- **P1 (b_graph × rows removed predicts the per-layer served − dedup gap within ±25 %):** REFUTED — predicted 2.419 vs measured 0.905 ms/step (+167.1 %)
- **P2 (b1 / L2 re-read floor; ≤ 1.3 both → L2-BOUND, ≥ 2.0 both → NOT-BYTES):** MIXED — gate_up 1.61, down 1.47
- **P3 (informational, c_gemv / DRAM floor per expert):** gate_up 0.73, down 0.56
- **P4 (informational, per-layer served − shared served):** +0.040 ms/step

**Decision rule:** P0 ∧ ¬P1 → the grid's linear model does not transfer to the recorded routing (skew or an R x D interaction); P2 is informational only and no lever lane follows from this read.

## The grid, arranged

This is the `gemv_int4_b32` kernel time per call (µs), from the rows above. It adds no new number and is not a registered verdict.

| `gate_up` R \ D | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|
| **16** | 9.3 | 9.6 | 9.9 | 12.0 | 19.6 | · | · | · |
| **32** | 17.3 | 17.9 | 18.2 | 18.7 | 22.0 | 41.4 | · | · |
| **64** | 34.1 | 35.2 | 35.8 | 36.4 | 36.6 | 45.4 | 87.5 | · |
| **128** | 64.8 | 66.2 | 66.9 | 67.7 | 68.6 | 69.9 | 101.2 | 177.2 |

| `down` R \ D | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|
| **16** | 4.9 | 5.0 | 5.5 | 6.6 | 10.2 | · | · | · |
| **32** | 8.4 | 8.4 | 8.8 | 9.1 | 11.2 | 18.8 | · | · |
| **64** | 15.6 | 15.6 | 16.1 | 16.3 | 16.9 | 19.9 | 36.2 | · |
| **128** | 30.2 | 30.3 | 30.7 | 31.0 | 31.5 | 33.5 | 43.5 | 73.1 |

## What the read establishes

- **The instrument is anchored.** The shared-store served arm reads 6.548 ms/step against P60's 6.479 (+1.1 %), and dedup reads 5.565 against 5.560 (P0). Giving every layer its own weight store changes served by only +0.040 ms/step (P4), so P60's single shared store was not flattering its replay through cross-layer L2 reuse. The per-layer served kernel row is 6.497 ms/step, against P57's census 6.340.
- **The cost does not split additively (P1 refuted).** The least-squares model t = a + b·R + c·D predicts that removing the recorded repeated rows saves 2.419 ms/step. The measured per-layer served − dedup is **0.905**. By the registered rule that ends the lane's attribution: P2 (MIXED, 1.61 / 1.47× the L2 floor) is informational only, and **no lever lane follows from this read**.
- **Why the model fails (informational).** The arranged grid shows it directly:
  - at fixed R, time is nearly flat in D up to D ≈ 32 (gate_up at R=128: 64.8 µs at D=1, 69.9 at D=32), then climbs with D (101.2 at D=64, 177.2 at D=128);
  - at small D, time grows linearly with R (~0.5 µs per row on gate_up, ~0.23 on down).

  So the row work and the distinct-expert bytes **overlap**: the larger term dominates, instead of adding. An additive fit cannot represent that, and its per-row and per-expert coefficients come out distorted. c from the kernel fit sits below the DRAM copy floor (P3: 0.73 / 0.56), which no real DRAM stream can do. A max-of-terms model is a new hypothesis, and it would need its own pre-registration.
- **What bounds any lever here** (measured, not modelled):
  - on the recorded routing, everything the repeated rows cost, loads and arithmetic together, is **0.905 ms/step** on per-layer stores;
  - one row per distinct expert sits **1.22×** above the byte floor on the shared store, and 1.25× on per-layer stores. That is dedup 5.565 / 5.683 ms/step against 4.556, where 4.556 is the distinct bytes at the mean 54.66 experts per call ÷ this box's 1528 GB/s DRAM copy bandwidth, derived from the rows. P60 read 1.21 (its open P2).

  Those two numbers bound any B=16 expert-GEMV lever on this card. Neither is licensed by this read.

## Decision (as registered)

**P0 ∧ ¬P1 → the grid's linear model does not transfer to the recorded routing, and no lever lane follows.** Nothing changes a default. K18's refutation stands on its own evidence. P61 adds that the repeated rows' total cost on real routing is 0.9 ms/step: an upper bound on what any per-row lever could recover at B=16, and not a split of it.
