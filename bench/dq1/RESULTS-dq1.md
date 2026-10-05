# DQ1 — results

Pre-registration: [DQ1-PREREG.md](DQ1-PREREG.md). Work item: #1083.

## Run 1 — `dq1-5090-1`, 2026-10-05: NOISY (the registered rule withholds every verdict)

| | |
|---|---|
| box | RTX 5090 (driver 595.71.05, PCIe gen max 4, width 16), AMD EPYC 7C13, Vast verified-secure, instance 54242398 |
| software | torch 2.8.0+cu128, CUDA 12.8, triton 3.4, bitsandbytes 0.50.2, grouped-nf4-gemm 0.39.0 `a5edec87` (tripwire) |
| launch | e4b `5ed6ba45` (#1084's merge), adertha `abb8b907`, `bench/tc1/tc1_drive.sh` + `dq1_run.sh` |
| cost / wall | $0.067 actual (receipt `cost_usd.actual`); allocation 03:05:19Z → destroy 03:11:00Z; census ≈ 3 min |
| teardown | `vast-destroy` HTTP 200, instance absent from the listing afterwards (`teardown-proof.json`) |
| evidence | [`receipts/dq1-5090-1/`](receipts/dq1-5090-1/) (`SHA256SUMS`); launcher receipt + ledger row in the receipt store, commit `aae1719c` |

**What held.**
- Every registered cell is present: 5 shapes × 5 rows, 5 arms each, and the streaming probe at 1024 / 2048 / 4096.
- No arm failed parity, engagement or execution anywhere. The reducer found nothing to VOID.
- The census ran to the end.

**What failed: the self-pair gate.** 37 of 250 self-pairs (14.8%) fell outside [0.95, 1.05], against a 10% budget. So
the lane reads NOISY, and the prereg withholds H, G1, GF, L and the streaming verdict. They are not reported here.

### Diagnosis: the instrument, not the arms

The out-of-band pairs fall into two groups.

1. **bf16, forward, the large compute-bound shapes.** Its self-pair reads 1.087–1.116 on gate_up, down and o at
   M ≥ 2048. bf16 sits at position 1 of the palindrome, and it is the **first** position that is wrong, not the
   second:

   | shape, M | bf16 fwd pos 1 | bf16 fwd pos 2 | same GEMM, independently (`dq` fwd − bnb dequant) | pos 1 / est | pos 2 / est |
   |---|---|---|---|---|---|
   | gate_up, 2048 | 2.519 ms | 2.762 | 2.780 | 0.906 | 0.993 |
   | gate_up, 8192 | 10.138 | 11.018 | 11.035 | 0.919 | 0.998 |
   | down, 4096 | 5.060 | 5.646 | 5.670 | 0.892 | 0.996 |
   | o, 8192 | 3.234 | 3.582 | 3.572 | 0.905 | 1.003 |

   Across all 16 large cells, position 2 agrees with the independent estimate to 0.985–1.009. Position 1 runs about 212
   TF/s against a sustained ~192, and the decay is visible inside position 1's own draws. For down at M=8192 they are
   10.11, 10.11, 10.12, 10.76 and 11.14 ms, then position 2 reads 11.20–11.32.

   It is not a linear drift: bnb, at palindrome gap 7, reads 1.004. **Cause:** a boost transient. The 1.5 s warm-up left
   the card above its sustained clock for roughly its first 3 s of load, and bf16, the first arm timed in every cell,
   ran inside that window.
2. **Launch-bound small cells**, kv_proj at M ≤ 1024 (~0.13 ms a call). Here bnb (1.20) and gnf4a (1.07) read slower at
   position 2. This is host-side jitter on a shared EPYC host. Removing group 1's pairs leaves about 17 of 250 out
   (≈ 7%), inside the budget.

### Registered consequence, and what changes

The prereg's consequence for NOISY is one re-run on another host; a second VOID or NOISY stops the lane as an instrument
finding. Re-running the same instrument would re-read the same transient. So **Amendment 1**
([DQ1-PREREG.md](DQ1-PREREG.md#amendment-1-registered-2026-10-05-after-run-1-read-noisy-before-run-2)) changes the
warm-up, and nothing else, before run 2:

- the rule and its thresholds are unchanged;
- the predictions are unchanged;
- the arms, rows, timing and parity are unchanged.

## Run 2 — `dq1-5090-2`, 2026-10-05, under Amendment 1: READ

| | |
|---|---|
| box | RTX 5090 (driver 595.84, power limit 575 W, PCIe gen max 4, width 16), AMD EPYC 7B13. A different machine from run 1 (`avoid_vast_machine_receipts`). |
| software | torch 2.8.0+cu128, CUDA 12.8, triton 3.4, bitsandbytes 0.50.2, grouped-nf4-gemm 0.39.0 `a5edec87` (tripwire) |
| launch | e4b `b99ebefd` (#1103's merge), adertha `abb8b907` |
| cost / wall | $0.102 actual; allocation 07:37:03Z → destroy 07:47:38Z |
| teardown | `vast-destroy` HTTP 200, instance absent from the listing afterwards |
| evidence | [`receipts/dq1-5090-2/`](receipts/dq1-5090-2/) (`SHA256SUMS`); launcher receipt + ledger row in the store, commit `a737cf91` |
| instrument | every cell warmed to steady in 4.1–4.2 s at 188.9–190.5 TF/s and ended within 0.997–1.005 of that rate. Self-pairs out of band: **6 / 250** (all kv_proj dgrad, host-bound). |

**Verdicts (the registered rule):**

| lane | speed | G1 | fused | LoRA | stream |
|---|---|---|---|---|---|
| READ | S_DEAD | G1_PARITY | GF_LOSS | L_LOW | C_MARGINAL |

An independent re-computation that did not import the reducer reproduces every reading and verdict to four decimals.

| M | LC bnb ms/layer | LC bf16 | **H** | **G1** | **GF** | DQ (bnb / dq) | **L** | dequant share | bnb's forward dispatch |
|---|---|---|---|---|---|---|---|---|---|
| 512 | 10.53 | 7.33 | 0.304 | 1.007 | 0.317 | 1.007 | 0.211 | 0.289 | dequant |
| 1024 | 16.45 | 13.37 | 0.188 | 1.010 | 0.262 | 1.004 | 0.137 | 0.184 | dequant |
| 2048 | 29.23 | 26.39 | **0.097** | 1.007 | 0.243 | 1.002 | 0.102 | 0.104 | dequant |
| 4096 | 53.94 | 51.25 | **0.050** | 1.005 | 0.227 | 1.001 | 0.094 | 0.056 | dequant |
| 8192 | 104.85 | 101.99 | 0.027 | 1.002 | 0.222 | 1.000 | 0.094 | 0.029 | dequant |

| M | H2D alone GB/s | under GEMM load | GEMM slowdown under DMA | X ms/layer | R_fwd | R_bwd | **Rmin** |
|---|---|---|---|---|---|---|---|
| 1024 | 27.7 | 27.8 | 1.012 | 9.06 | 0.60 | 1.22 | 0.60 |
| 2048 | 27.8 | 27.9 | 1.009 | 9.02 | 1.07 | 2.17 | **1.07** |
| 4096 | 27.4 | 27.3 | 1.025 | 9.23 | 1.93 | 3.91 | **1.93** |

### Against the stamped predictions

| prediction | registered | measured | |
|---|---|---|---|
| H(512) / H(2048) / H(4096) | [0.15, 0.40] / [0.04, 0.15] / [0.02, 0.10] | 0.304 / 0.097 / 0.050 | held ×3 |
| speed verdict | S_DEAD or S_MARGINAL | S_DEAD | held |
| G1 at M ≥ 2048 | [0.97, 1.08] | 1.002–1.007 | held |
| GF at every M | < 0.80 | 0.222–0.317 | held |
| L(2048) | [0.03, 0.12] | 0.102 | held |
| loaded H2D on PCIe 4.0 x16 | ≈ 22 GB/s | 27.9 GB/s | **missed** (under-predicted the link) |
| Rmin(2048) on 4.0 | ≥ 1.0 | 1.07 | held |
| GEMM slowdown under DMA | ≤ 1.03 | 1.009 (1.025 at 4096) | held (narrowly at 4096) |

### What the read supports, and how far

1. **The W4A16 dense speed-kernel line closes** at QLoRA training rows (registered consequence).
   - **Scope.** H bounds a perfect weight-only kernel that does its math in bf16 on the base linears. bf16 runs at
     180–243 TF/s in every cell; every census cell is compute-bound by ~2.5–3.5× at the 5090's ridge. So a W4A16
     kernel beats the bf16 floor only in the decode regime (M ≲ 60–130).
   - **Not bounded:** FP8/FP4-compute kernels (a different primitive with its own quality question), or whole-step
     effects. H overstates whole-step savings, because attention, norms and LoRA sit outside it.
   - **Noise.** H(2048) = 0.097 is 0.003 under the S_DEAD line and inside draw noise: bootstrap 95% interval
     [0.095, 0.104], P(H ≥ 0.10) ≈ 11%. Run 1's position-2-only reading was 0.094. If it crossed 0.10 the label
     would read S_MARGINAL, but the registered consequence for S_MARGINAL with H(2048) < 0.12 is the same.
     H(4096) = 0.050 is robust (interval [0.048, 0.055]).
   - **Below the registered rows the headroom is real:** H(1024) = 0.188 and H(512) = 0.304. A workload that trains
     at ≤ 1k tokens a micro-batch (very large models on small cards) is where a fused dequant-GEMM could still pay.
     DQ1 does not register that question.
2. **The headroom is the dequant, within about ±10%.** Across a layer, (bnb − bf16) / (3 × bnb's standalone
   `dequantize_4bit`) is 1.05 / 1.02 / 0.93 / 0.89 / 0.94 from M=512 to 8192. The dequant is a fixed ~0.25 ms per call
   on gate_up/down and ~0.07 ms on q/o, so its share falls as 1/M.
   - **Exception: kv_proj at M ≤ 2048.** The gap there is host-side launch overhead (bnb's kv forward is flat at
     ~137 µs while bf16 takes 29–114 µs), not dequant. In a real step that host time overlaps the big linears, so it
     inflates H at small M slightly, in the S_DEAD direction.
3. **G=1 grouped-nf4-gemm is a correct dense route, at parity** (1.002–1.010).
   - Under 0.39.0's `auto` it is dequant + `torch.mm`, structurally bnb's path.
   - Its Triton decoder is 0.94–1.01× bnb's on the big shapes and 0.65× on kv, which buys under 1%.
   - Nothing for a dense backend to add on speed.
4. **The packed (fused) G=1 kernel is 3.2–4.5× slower** than bnb at every row (GF 0.22–0.32). This is consistent with
   the 2026-08-15 A6000 measurement (0.22–0.80×, kept in session notes, not in either repo), now on sm_120.
5. **bitsandbytes 0.50.2 takes dequant + `F.linear` at every census row on sm_120.** This is read from bnb's
   heuristic per cell, and corroborated by timing: bnb and the explicit `dq` arm agree within 0.7% at every row.
6. **The unfused LoRA delta costs as much as the whole dequant headroom at M ≥ 2048, and more beyond.**
   - L(2048) = 0.102 vs H = 0.097; L(4096) = 0.094 vs H = 0.050; L holds at ~0.094 at 8192 while H keeps falling.
   - That makes it the larger remaining speed lever on the linears.
   - It is also the lever Unsloth already pulls (fused LoRA). Any lane on it is a matched-work comparison against
     Unsloth, not a new primitive.
7. **Streaming is a link question, and on this PCIe 4.0 x16 host it is marginal.** DMA costs the GEMMs ≤ 2.5% and
   keeps its full bandwidth under load. The forward phase binds: Rmin = R_fwd = 1.07 at 2048 tokens a micro-batch,
   1.93 at 4096, and 0.60 at 1024 (where a 4.0 link would stall the forward by roughly 20%).
   - R counts linears only. Adding LoRA's forward and an estimate of causal attention lifts R_fwd(2048) to ~1.2.
   - **Derived, not measured: a model-size-free rule.** Compute and bytes both scale with parameter count, so
     R_fwd ≈ 2·M·B / (b·F) with B the link GB/s, F the sustained bf16 TF/s, and b = 0.516 B/param for NF4 + nested
     state (one layer's 251.5 MB over its 487.6 M parameters). Break-even is M* ≈ 0.26·F/B.
     - On this host (B ≈ 27.9 GB/s), with F set to the warm-up GEMM's 190 TF/s: M* ≈ 1,760 tokens. The measured R_fwd
       is 0.92× that at 2048 and 0.83× at 4096. The layer's own linears run faster than the 4096³ warm-up GEMM: about
       207 TF/s across a layer at 2048 and 224 at 4096. A faster GEMM leaves less time to hide each copy, so F must be
       the layer's measured rate. With it, the rule reproduces the measured R.
     - On a PCIe 5.0 x16 host (~55 GB/s), at the layer's ~207 TF/s: M* ≈ 970, and scaling the measured Rmin(2048) by the
       link gives ≈ 2.1, which would clear C_ALIVE.
   - Neither DQ1 run drew a 5.0 host. C_MARGINAL is a PCIe 4.0 result.

**Registered consequences** (DQ1-PREREG.md):
- **S_DEAD with G1_PARITY:** the dense W4A16 speed-kernel line closes; G=1 stays an internal gnf4 route; no repository
  is proposed for speed.
- **C_MARGINAL:** the registered consequence table names no action, so DQ1 licenses no streaming prototype. The
  capacity axis needs its own registration: a 5.0-x16 host, and the real step's per-layer forward time in place of the
  linears-only estimate.
- **Instrument note:** run 2 changed host as well as warm-up (575 W vs 400 W power limit, EPYC 7B13 vs 7C13), so the
  NOISY → READ improvement cannot be credited to Amendment 1 alone. Every warm-up hit the 4 s floor before the
  steadiness test (16 blocks); the end/steady rates (0.997–1.005) show no in-cell drift either way.
