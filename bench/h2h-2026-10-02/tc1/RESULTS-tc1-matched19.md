# TC1 amendment 19 read: the matched-work positions after amendments 10–15 — Unsloth/e4b 1.997 (P30 HELD), axolotl/e4b 2.775 (P32 HELD), matched set equivalent (P31 HELD)

Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendment 19. Both boxes pinned e4b at `14a9a17` and grouped-nf4-gemm
at `00929a4`, with every default from amendments 10–15:

- the single-read grouping and pinned ring (#945);
- the trimmed LoRA delta (#440);
- the cost tile rule (#442);
- the fused rotary (#965);
- the fused RMSNorm (#975).

No environment variables were set. The tokens are unchanged from TC1 (`qwen3`, the matched set) and amendment 3 (`qwen3axolotl`).

## The boxes

| box | token | host | cost |
|---|---|---|---|
| `tc1-5090-49` | `qwen3` (box A) | instance 54075516, AMD EPYC 7C13, driver 580.95.05 | $1.15 |
| `tc1-5090-50` | `qwen3axolotl` (box B) | instance 54075633, AMD Ryzen 9 9950X3D, driver 610.57.04 | $0.33 |

## Readings (matched work: fp32 adapters, one per-slot init, the same tokens; s/step medians of steps 11..20 over two draws)

| position | box | ratio, other / e4b | e4b s/step | other s/step | peak GB (other vs e4b) | energy per step (other / e4b) | verdict |
|---|---|---|---|---|---|---|---|
| **Unsloth (`grouped_mm`)** | -49 | **1.997** [1.980, 2.014] | 3.973 | 7.933 | 24.27 vs 27.22 | ×1.17 | **P30 HELD** (band 1.6–2.8) |
| **axolotl** | -50 | **2.775** [2.735, 2.814] | 2.147 | 5.956 | 26.88 vs 27.22 | ×2.97 | **P32 HELD** (band 1.6–2.8) |
| axolotl, one draw (reported, not registered) | -49 | 1.979 [1.965, 1.994] | 3.973 | 7.863 | 26.88 vs 27.22 | ×2.73 | — |
| Unsloth mb1 × accum 8 (secondary; HF OOMed) | -49 | 2.101 | 6.840 | 14.374 | 24.24 vs 26.05 | ×1.60 | — |

**P31 HELD.** On box A the matched set is inside the draw noise of e4b fused:

- e4b reference: median step |Δ| 0.0017, held-out |Δ| at N 0.0002;
- Unsloth: median step |Δ| 0.0020, held-out |Δ| at N 0.0003;
- the draw-noise floor is 0.0034.

The fused RMSNorm (near-exact) moves e4b's own step-0 held-out by about 0.006 (1.9441 against the reference's 1.9508). That takes
Unsloth's step-0 class against e4b from SAME-BYTES-CLASS on `tc1-5090-16` to NEAR (0.0118, recorded, VALID). The trajectories
agree as before.

**Against the 2026-10-02 positions** (e4b before #945): Unsloth/e4b moved from 1.437 to **1.997**, and axolotl/e4b from 1.416 to
**1.979–2.775**. P30 landed on its point estimate of about 2.0; P32's point sits inside its band with its interval just past the top.

**Ratios no longer travel between hosts as well as they did.** e4b's step is host-launch-bound, so its own s/step varies by host more
than axolotl's: 2.147 on the 9950X3D against 3.973 on the EPYC 7C13. axolotl/e4b therefore reads 1.979 on one host and 2.775 on
the other. A position is a within-box reading of its host.

Also recorded:

- HF+PEFT OOMs at the matched recipe and at mb1, as before.
- Unsloth's profiled arm hit its 2,400 s alarm. It is the profile instrument, not a position.
- In axolotl's scattermoe native-best row on box B (3.363 s/step, its own init), e4b's matched path is the faster one this time.

## What follows

Two new register rows give the current positions:

- `e4b.train.h2h.unsloth.qwen3.5090.2026-10-03`;
- `e4b.train.h2h.axolotl.qwen3.5090.2026-10-03`.

The 2026-10-02 rows stay active, labelled as the code before #945. STATUS quotes the new positions.
