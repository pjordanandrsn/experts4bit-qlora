### SC1g A5 read (#846): K-A REFUTED, L1 HOLDS, L2 HOLDS on gpt-oss-20b by full-vocabulary KL; vLLM's served output is not run-to-run deterministic (bench and tests only)

- **The proof.** Box I's proof `sc1g-prove-a5-8` was PROVED ($0.955): every engine read zero masked reference mass, so no
  common-support rule.
- **The reading.** `sc1g-5090-a5-1` ($1.557), from the same commit and host, graded conv1–conv4 against box R's
  reference:
  - **K-A REFUTED:** NF4 is ≥ 3× MXFP4 on conv1–conv3 (10.8×, 4.7×, 4.0×) but 2.5× on conv4.
  - **L1 HOLDS:** e4b MXFP4 pooled KL 0.0224 is 1.85× vLLM's 0.0121, under 2×. conv3 is dropped, because vLLM is within
    the floor there.
  - **L2 HOLDS:** every native-MXFP4 engine sits below R's NF4-requant scale.
- **Descriptive:**
  - vLLM's conv1 KL moved 0.0332 → 0.0250 between two runs on one host, while e4b and llama.cpp were bit-identical. L1's
    0.0027 margin is inside that spread.
  - On conv1 and conv2, e4b NF4 has the lowest NLL and the largest KL: NLL flattery, now graded.
  - e4b MXFP4's conv2 KL, 0.043, is 5× vLLM's. That is open.
- **Receipts.** Both are committed, and the reading re-derives from its committed receipt, every verdict identical and every number to 1e-12 relative (pinned by a test).
- **PREREG.** The proof record gains `-a5-7` and `-a5-8`. The lane is at $8.850.
