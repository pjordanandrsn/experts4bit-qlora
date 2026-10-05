### Read: TC1 amendment 35 — under triton 3.7.1 the prebound launches read 0.986 (matched) and 0.996 (shipped); triton 3.7 stays covered (P66, P67, P68 HELD)

- `tc1-5090-79` ($1.68, EPYC 7B13, 60-step load-gated draws): amendment 26's A/B in venv-unsloth (torch 2.12.1, triton 3.7.1).
  Matched `_pb1`/`_pb0` 0.986 [0.975, 0.997]; shipped 0.996 [0.978, 1.013]; held-out within 0.003. Every `_pb1` arm counted 216,335
  e4b and 72,162 grouped-nf4-gemm prebound launches.
- By the registered rule triton 3.7 stays in the prebound path's supported versions (#1108, grouped-nf4-gemm#471). Row
  `e4b.train.prebind.triton37.qwen3.5090.2026-10-05`. The shipped arm's interval reaches 1.0.
- The gate voided five draws; the shipped arm's first draws stood at load 8.2 after the retries ran out. First attempts read 0.980 /
  0.995: no verdict changes.
