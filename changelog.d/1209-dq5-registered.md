### DQ5 registered (#1083): DQ3's streamed-QLoRA lane on a PCIe gen 4 x16 RTX 5090 (bench only; #1209)

- **What runs.** DQ3's subject, arms, rule and reducer, unchanged, on a gen 4 x16 link.
- **The runner.** `bench/dq5/dq5_run.sh` is DQ3's runner with two changes:
  - a gen 4 gate on `nvidia-smi pcie.link.gen.max/width.max`. Anything else exits rc 19, which is not an admitted
    machine-evidence code, so a good gen 5 host is never excluded;
  - a descriptive pinned-H2D probe that never refuses.
- **The search.** The offer search takes adertha-agents#177's new PCIe generation ceiling.
- **The prediction.** T(S)/T(R) in [1.00, 1.05], so PROTO_PASS.
