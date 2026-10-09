### TC1 harness for P129 Amendment 3 (#835): the step-0 floor, measured on the box

- **The floor block.** `tc1_arm.qkv_floor_rows` (`TC1_QKV_FLOOR=1`): on the q0 arms, before any training step, the step-0 held-out per
  row under registered bf16 schedules of the q/k/v base projections. They are A0 (the stock arithmetic through the hook, a self-check),
  D1 (fp32 matmuls, the reference), D2 (q split in two), D3 (k and v as one matmul) and D4 (q split in four). Only each projection's base
  is swapped; the adapters add their delta the stock way. The receipt gains `qkv_floor`.
- **The family.** `qwen3fqkv3` in `tc1_run.sh`: Amendment 2's box with the floor on its q0 arms.
- **The reducer.** `tc1_reduce.py` reads its step-0 clause as an envelope: e_x is the mean over rows of |x − D1|, and the fused path
  passes when e_B ≤ max(e_A, e_D2, e_D3, e_D4). A q0 draw whose floor is incomplete, whose A0 misses the stock rows, or whose D3 did not
  share is VOID. Self-test case 127. Amendment 2's family reads as before.
- **Tested** on an RTX A2000 on the two-layer model (`bench/p129/instrument/p129_floor_check.py`).
