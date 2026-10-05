### RD1 registered: grouped-nf4-gemm's frozen-expert GEMM routes per call at MoE training shapes, on one RTX 5090 (bench and tests only)

- **Why.** On sm_120, `auto` keeps the fused NF4 kernels for calls with more than 16 present experts.
  - The per-expert `dense` loop is launch-bound there (TC1 amendment 22: 2.947× on Qwen3-30B-A3B).
  - The fused kernels run TF32 and decode the weight once per M-tile inside the GEMM loop.
  - No torch release through 2.14.1 has a single-launch grouped bf16 GEMM for that card.
  - A decoded route needs no per-expert launches: grouped-nf4-gemm's `dequant_groups` (one launch), then one Triton grouped
    bf16 GEMM launch.
  - RD1 asks whether it beats the best of {v1 as shipped, the fused kernels' own bf16 MMA (v3, plus a probe-local bf16
    dgrad), `dense`} per call.
- **What.** `bench/moegen/rd1/`:
  - `RD1-PREREG.md`: the bar, the correctness gate, the decision and the hypotheses, registered before the box;
  - `rd_probe.py`: five arms, eight families' shapes, seq 512 / 2048, uniform and skewed routers. It records device and
    event time, peak bytes, and each arm's error against an fp32 reference;
  - `rd_table.py`: the gate and the bar, with the decision read only on an RTX 5090 receipt;
  - `rd1_run.sh`: the box side, under `tc1_drive.sh`, with the train anchor strict;
  - `a2000/`: the correctness rehearsal, with no timing field.
  - `tests/test_rd1_lane.py` pins the staged pieces, the shape table, the bar and the gate to the registration. It also shows
    that a fast arm with 3× dense's error on one call cannot count.
- **Budget.** One RTX 5090 with no checkpoint fetch, about $0.64 under a 0.75 h guard.
