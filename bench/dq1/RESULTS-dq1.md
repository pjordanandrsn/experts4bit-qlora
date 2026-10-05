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
