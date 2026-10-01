# P87 rehearsal on the NAS RTX A2000 — NOT a reading

`rehearse.sh` staged the lane's files exactly as `p87_drive.sh` would, in a throwaway
`pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel` container. It then ran `p87_run.sh` in its proving mode on 2026-10-01
(`P87_PROVE=1`), under rehearsal knobs: `P87_GPU_CLASS=A2000` and `P87_MIN_DISK_GB=20`. The runner marked the run
`REHEARSAL`.

The e4b installed was #804's head, `46a7840`, from GitHub. The launch commit will be this registration's merge, which
carries the same route and test bytes.

| step | result |
|---|---|
| staged pins (`sha256sum -c`) and reducer self-test | OK, 17 cases |
| install + tripwire | K19, the #803 route, the T == 1 extension, K16, the #405 knobs, the fused reduce off: OK |
| **premise** (`test_k19_row_exact_gpu.py` on the card) | **3 passed**: rows bit-equal alone vs inside B=16; oracle; captured T == 1 replay equals eager |
| K19 + K16 contract tests compiled (gnf4 `3351c9d`) | 20 passed |
| egress probe | 35.4 MB/s |
| lane rc | 0, `PROVED` |

What it cannot show: anything on sm_120 or about Qwen3-30B-A3B's decode loops. The A2000 cannot run the fp8 paged-KV
kernel (sm_89+). The proving rental runs the same path on the 5090.
