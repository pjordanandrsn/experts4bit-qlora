### P149 box Q+S kit: quality at decode, the served read and Amendment 4's token-pinned processes, on Mixtral-8x7B

The kit for P149's box Q+S (`bench/p149/`). Nothing here is a default, and nothing runs until the box is created.
- **`p149_box.py`**, one process per arm or phase, on Mixtral's server (16 slots of 768 tokens, buckets 1 to 16, graphs
  on). The arms differ only in `E4B_SERVE_ATTN_INT8`.
  - **Speed:** P143 Amendment 3's form. The GPU's temperature and SM clock are recorded around every timed block.
  - **Q**, in S0a and S1a after their timed reps: P147's W1 and W16 windows, sha256-checked, teacher-forced at T == 1
    through the served decode step, on graphs re-captured with a logits copy. All 128 of P147's positions are read at
    decode.
  - **Reported only:** prefill-Q, prefill TTFT and SANE.
  - **Amendment 4's T1 and T2:** W16 teacher-forced on S0a's and S1a's recorded tokens. The forcing is checked against
    the captured graph's own input buffer, and routing comes from an eager replay.
  - **Engagement:** per captured bucket, the int8 projections' calls by row count and path.
- **`p149_reduce.py`:** P147's quality rule (S1 against S0 on the box), P143 Amendment 3's speed rule, the engagement
  VOIDs, and Amendment 4's qualifier.
  - The verdict order is VOID, NOISY, QUALITY_FAIL, FUNCTION_FAIL, then FLIP or NO_FLIP.
  - A Q-only reduce after S1a lets a QUALITY_FAIL stop the box.
- **`p149_run.sh` / `p149_drive.sh`:** the host floors, the power floor and Amendment 4's guard fit, all before the
  install. Then the compiled premises (grouped-nf4-gemm K37's suite, vendored byte for byte, and e4b's int8 kernel path),
  the rotation, and the receipts.
- **`p149_preread.py`:** Amendment 4's USD 0 A2000 router pre-read, a prediction.
