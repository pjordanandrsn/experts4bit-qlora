### Read: TC1 amendment 37 — with grouped-nf4-gemm#473 the compact delta lowers the matched peak 0.29 GB and runs 0.967 / 0.948; it stays opt-in (P75 FALSIFIED on the fast side)

- `tc1-5090-83` ($0.78, EPYC 7702P, machine 45379, 60-step load-gated draws, venv-unsloth, grouped-nf4-gemm after #473):
  `NF4_QLORA_COMPACT_DELTA` 0 vs 1. Matched 0.967 [0.959, 0.975], peak 27.477 → 27.189 GB (P73, P74 HELD); shipped 0.948
  [0.933, 0.964], below its [0.95, 0.99] band (P75 FALSIFIED); held-out within 0.003 (P76 HELD).
- #473 turned amendment 36's +0.229 GB into −0.288 GB. The speed replicated on a second host.
- By the registered rule it stays opt-in pending its own registration: a ratio below 0.95 is the rule's "otherwise" branch.
