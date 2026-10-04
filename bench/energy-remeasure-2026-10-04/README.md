# Energy remeasure with a recorded bitsandbytes version (e4b#392), 2026-10-04

**Harness:** `bench/_upstream/bench_energy.py`, unchanged (sha256 `ece6b5c8…`, the file `docs/METHODOLOGY.md` §10 was
measured with).

**Setup:**
- **Card:** the same card, the NAS RTX A2000 12 GB (70 W cap, driver 575.64.05), in a throwaway
  `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel` container.
- **Versions, recorded in `versions.txt`:** experts4bit-qlora 0.45.0 @`0c36bf8`, **bitsandbytes 0.50.2** (the released
  build), torch 2.8.0+cu128.
- **Runs:** three passes, 19:50:07–19:53:50Z by the runs' own timestamps (`t0_*`, `t1_*`). `e392_in.sh` is the driver.

**Contamination guard.** The NAS card also serves home services, so `nvidia-smi pmon` logged every process's SM use
throughout each pass (`pmon_*.txt`). Only this run's own process ever appears, one PID per pass. `pre_3.txt` reads
94 % / 68 W just before pass 3; that is the lagging average of pass 2, which had just exited, and `pre_2.txt` shows the
same decay after pass 1.

## Results: total J/op relative to native bf16, by pass

| cell | `matmul_4bit` | dequant → linear | fork receipt 2026-07-01 (`matmul_4bit` / dequant) |
|---|---|---|---|
| decode M=1 | **0.91 / 1.04 / 1.06** | 2.37 / 2.41 / 2.43 | 1.18 / 2.68 |
| prefill M=512 | **1.29 / 1.30 / 1.49** | 1.29 / 1.29 / 1.29 | 1.29 / — |
| train fwd+bwd M=32 | **1.86 / 2.15 / 1.64** | 1.59 / 1.68 / 1.31 | 2.25 / 1.57 |

On the released build the `matmul_4bit` decode cell is at **break-even**: it draws 37–43 W against native's 69 W while
running ~1.7× slower. The fork read 1.18×. The prefill cell is unchanged, and the train cell is 1.6–2.2× (fork 2.25×).

## Caveats

- **Pass 1 is the cleanest** (idle read 19.9 W). Passes 2–3 started on a warm card: idle read 51.7 W before pass 3,
  and the train cell's native power rose 59.5 → 69.1 W across passes. That is why the train ratios drift. As in §10,
  only total J is meaningful; the idle-subtracted column is not.
- One card, one projection (a microbenchmark): nothing here is about grouped MoE execution or other cards.
