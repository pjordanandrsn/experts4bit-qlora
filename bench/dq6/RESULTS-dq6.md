# DQ6 — results

Pre-registration: [DQ6-PREREG.md](DQ6-PREREG.md) (DQ4's capacity read on a 24 GB RTX 4090). Work item: #1083.

**Scope:** one RTX 4090 with 24 GB (Vast instance 54498479; AMD EPYC 7742; PCIe gen 4 x16; driver 580.95.05), e4b
`1f4f639b`, adertha `7566b18a`, DQ3's subject and DQ4's recipe.

## Verdict — `dq6-4090-1`, 2026-10-06: **CAP_REAL**, G = **4.75** (graded configuration c_def)

With the frozen NF4 weights streamed (`train_prefetch` plus #1183's late-bound backward), one 24 GB RTX 4090 trains
**4.75×** the resident model's longest sequence. On DQ4's 32 GB RTX 5090 the factor was 2.00×.

| c_def: chunked loss, default allocator (**graded**) | R (resident) | S (streamed) |
|---|---|---|
| L\*: the largest passing rung, confirmed in a fresh process | **2048** | **9728** |
| first OOM rung, confirmed in a fresh process | 2560 | 10240 |
| allocated after setup, before the first rung | 18.43 GiB | 3.90 GiB |
| peak allocated / reserved at L\* (fresh) | 21.97 / 23.04 GiB | 17.30 / 22.99 GiB |

- **G = L\*_S / L\*_R = 9728 / 2048 = 4.75. CAP_REAL** (≥ 1.5).
- Each L\* is a rung on a 512-token ladder, so G is bracketed by [L\*_S / first-OOM_R, first-OOM_S / L\*_R] =
  [9728 / 2560, 10240 / 2048] = **[3.80, 5.00]**. The verdict does not depend on the rung step.
- **Every confirmation agreed:** all four fresh processes in each configuration matched their ladders. No configuration
  was VOID, FUNCTION_FAIL or NOISY.
- **Step time is unchanged:** T(S)/T(R) = **0.993** at 2048 tokens, the largest common rung (descriptive).
- **Losses agree to within run-to-run nondeterminism, not bitwise.** At every rung both arms passed, R's and S's
  losses differ by at most about 2e-3 (12.95693 against 12.95857 at 1024 tokens). R differs from *itself* by the same
  amount between c_def and c_exp (12.95693 against 12.95832), because DQ4's harness does not run deterministic
  kernels. Bitwise parity is DQ3's and DQ5's claim, from their deterministic pass; DQ6 neither measures nor claims it.
- **The saving is the weights.** R − S peak allocated is **14.072–14.084 GiB at every common rung** (512–2048 tokens
  in c_def, 512–2560 in c_exp). That is the 14.53 GiB of NF4 weights less the two layers the schedule keeps resident
  (2 × 243,793,920 B = 0.45 GiB), which is 14.08 GiB.

**Why the factor more than doubled on the smaller card.** Both arms' static footprints are the same as on the 5090:
- R: 18.43 GiB, against DQ4's 18.42;
- S: 3.90 GiB, against 3.89.

A 4090 leaves torch about 23.0 GiB to reserve. Resident training therefore has about 4.6 GiB for activations, and
streamed training has about 19.1 GiB.

## Secondary configuration: `expandable_segments:True` (c_exp)

| | R | S | G (bracket) | verdict | T(S)/T(R) |
|---|---|---|---|---|---|
| L\* / first OOM | 2560 / 3072 | 13312 / 13824 | **5.20** [4.33, 5.40] | CAP_REAL (secondary) | 0.990 at 2560 |
| peak allocated / reserved at L\* | 22.55 / 23.04 GiB | 21.65 / 22.99 GiB | | | |

**The descriptive s_def pair was skipped for time,** as the runner's registered rule provides: 466 s were left against
the 600 s needed. Builds took about 340 s per process on this host's EPYC 7742, against about 84 s on DQ4's 5090
host, and 12 processes ran.

## Against the registered predictions

All predictions were seeded from DQ4's rented-5090 receipts only.

| | predicted (band) | read | |
|---|---|---|---|
| c_def L\*_R | ~1536 [512, 2048] | **2048** | at the band's upper edge |
| c_def L\*_S | ~9216 [6144, 12800] | **9728** | inside |
| c_def G | ~6 [3.0, 25] | **4.75** | inside, below central |
| c_exp L\*_R | ~2048 [1536, 2560] | 2560 | at the band's upper edge |
| c_exp L\*_S | ~13312 [12288, 14336] | 13312 | on the central value |
| c_exp G | ~6.5 | 5.20 | below central |
| s_def | R 512 or VOID; S ~4608 | — | skipped for time |
| T(S)/T(R) | [1.0, 1.3] | 0.993 / 0.990 | **0.7–1.0 % below the band's floor**: S was not slower. A descriptive miss. For scale, R's fresh confirmation at 2048 read 4.73 / 4.47 s, against its ladder's 4.41 / 4.43 s |

**Where the model of the boundary was off: R, by one rung.** The prediction made two assumptions.
1. **Usable memory.** It assumed about 23.25 GiB usable: the card's 23.99 GiB less the 0.73 GiB the 5090 never let
   torch reserve. In fact torch reports this card's total as **23.52 GiB** (25,250,627,584 B). Both arms reserved up
   to **22.99–23.04 GiB**, about 0.48 GiB below that total, so the usable budget was about 0.2 GiB *smaller* than
   assumed.
2. **R's stranded memory at its boundary.** It assumed 1.46–2.50 GiB, from DQ4. R's reserved minus allocated at L\* was
   only **1.07 GiB** (default) and 0.49 GiB (expandable), so R reached one rung further than central.

**S's stranded memory reproduced DQ4's.** Under the default allocator, S's peak reserved sits at 17.7 GiB from the very
first rung (17.73 at 512 tokens, against 6.57 allocated). That is the same pattern as DQ4's 5090, which read 17.65 at
2048. At S's boundary 5.69 GiB was reserved but unallocated, against 6.22 on the 5090. expandable_segments recovers
it: 1.34 GiB at L\*, and L\*_S rises from 9728 to 13312.

## Run record

| | |
|---|---|
| box | RTX 4090, 24,564 MiB (`nvidia-smi`), PCIe gen 4 x16 (recorded, not gated), 450 W, driver 580.95.05; AMD EPYC 7742 |
| host gates | card gate `NVIDIA GeForce RTX 4090, 24564, 4, 16` in band; VRAM probe OK at 21.5 GiB; egress 3.12 MB/s |
| software | torch 2.8.0+cu128, bitsandbytes 0.50.2, transformers 5.18.0, peft 0.21.2, grouped-nf4-gemm `a5edec87` (pinned by the runner); tripwire: e4b `1f4f639b` |
| staged files | all eight box files (`dq6_run.sh`, `dq6_vram_probe.py`, `dq6_reduce.py`, `dq4_cap.py`, `dq4_reduce.py`, `dq3_arm.py`, `dq3_vram_probe.py`, `dq3_egress_probe.py`) match `1f4f639b` by sha256 |
| pre-launch gate | A2000 at `1f4f639b`, all four PASS: runner rc 19; probe rc 3; rehearsal L\*_S 6656 > L\*_R 5120, `late_bound_4bit` 56; reducer self-test and 12 lane tests |
| cost | **$0.802**, against the registered ≤ $1.20 (the launcher's estimate of $2.30 includes a download allowance); teardown proven (`teardown-proof.json`) |
| evidence | [`receipts/dq6-4090-1/`](receipts/dq6-4090-1/) (`SHA256SUMS`); launcher receipt and ledger row in the receipt store, commit `c28182c8` |

## Not shown

- Real weights. Values don't change bytes.
- A micro-batch above 1.
- Another model.
- A 4090 variant other than the 24 GB card.
- The stock-loss pair on this card (skipped for time).
- Step time as anything but descriptive.

## Reproduce

```
cd bench/dq6/receipts/dq6-4090-1
python3 ../../dq6_reduce.py . | cmp - dq6_read.json
python3 ../../dq6_reduce.py --self-test
```

The output is byte-identical. `SHA256SUMS` covers every committed file.
