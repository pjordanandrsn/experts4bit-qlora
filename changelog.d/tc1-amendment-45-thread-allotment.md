### TC1 amendment 45 registered: OMP_NUM_THREADS at the host's physical cores against the container's CPU allotment (P104–P106)

- Token `qwen3ompab`: e4b's and Unsloth's matched arms in venv-unsloth, `_om0` (the host's physical cores, as every box so far) vs `_om1`
  (the container's allotment: cgroup `cpu.max` or the CFS quota, capped by affinity), two load-gated draws a side. P104: e4b ≤ 0.97;
  P105: Unsloth ≤ 1.01; P106: held-out within 0.005. A host with no allotment below its cores refuses at setup (rc 18) for about a cent.
- `tc1_run.sh` computes the allotment for every box and records it in `summary.txt`. `OMP_NUM_THREADS=$PHYS` now precedes the per-arm
  environment, so a family can override it; arms that do not are unchanged. `tc1_reduce.py`: the family, `ompab_why`, `score_ompab`;
  self-test 107 cases.
