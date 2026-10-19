### P148 registered (#1313): the dense bf16 decode GEMVs, a falsify-first microbench

P142's per-name table puts the dense bf16 GEMVs at 41 % of the one-request step on an RTX 5090. Fused q/k/v and o run
at 75-79 % of the card's DRAM rate, and the lm_head at 92 %. `bench/p148/` times cuBLAS against two prototype kernels in
one CUDA-graph harness, with no model:
- **D1:** a bandwidth-targeted bf16 GEMV for q/k/v and o.
- **D2:** an int8 lm_head.

The rule is fixed before data: each lever GOes if it saves >= 2 % of the step within a tolerance contract, with a NOISY
rung and a transfer anchor. A GO starts kernel work only, and the saving is a premise, never a quoted gain. The int8
head also carries a quality gate before any flip. Nothing changes by default.
