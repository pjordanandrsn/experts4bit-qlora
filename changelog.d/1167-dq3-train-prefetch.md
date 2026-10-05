### DQ3 stage 2 (#1083): an opt-in training prefetch for dense offload, `enable_dense_offload(..., train_prefetch=True)` (off by default), and the lane harness (#1167)

- **What it does.** Under grad mode, layer i's forward prefetches layer i+1's frozen weights on the prefetch stream.
  The last layer prefetches L−2, the first layer reused in backward, and backward prefetches i−1. Residency is at
  most two layers: the current one and its scheduled neighbour. A steady step issues 2×(L−2) prefetches and blocks
  on none. The first use of each prefetched block passes a `record_stream` fence at bind time.
- **The off path is unchanged:** no stream or schedule is built, and the output is bitwise equal to running with no
  offload. `dense_offload_report` gains a `train_prefetch` counters key, which is `None` when off.
- **Trainable parameters** follow #1165's per-call rule on both paths.
- **Tests** (`tests/test_dense_offload_train_prefetch.py`):
  - the pure schedule;
  - CPU and CUDA bitwise parity, with and without checkpointing;
  - bounded residency and exact counter values;
  - a fence race test with a warmed reader, so it can race; with the fence removed it fails. On the A2000, a cold
    first cuBLAS GEMM synchronized the device and made the test vacuous.
- **The lane harness** (`bench/dq3/`):
  - `dq3_arm.py` runs one arm per process on Qwen3-32B's architecture with random NF4 weights.
  - `dq3_reduce.py` has a 25-case self-test.
  - `dq3_run.sh` refuses with rc 13 any host that is not a gen 5 x16 RTX 5090.
  - `tests/test_dq3_lane.py` kills all 16 rule mutants.
  - `DQ3-PREREG.md` Amendment 2 (pre-data) records the A2000 rehearsal's S0 crash and the #1165 fix.
