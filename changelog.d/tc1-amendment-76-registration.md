### TC1 amendment 76 registered: grouped-nf4-gemm#584's reordered compact-delta backward, before vs after, two draws a side

- One rented RTX 5090 runs amendment 75's best arm (cd1cl1) at grouped-nf4-gemm main just before #584 and at its merge, two draws a side
  in ABBA order. grouped-nf4-gemm is reinstalled between the sides, and the installed commit is re-read. Before any arm, the merge's
  `kernel/test_compact_delta.py` runs on the box's GPU and refuses at cents on a failure. Worst case $3.20.
- A CPU-allotment floor in the pre-flight: a box under 42 refuses before any venv or fetch. The floor is sized from box 155's measured
  phases, with every arm run three times and 25 % headroom inside the 2 h guard. The pre-check installs pytest beside the measured venv,
  never into it, and refuses if the venv's package set changes.
- Each arm is bound to its code: `tc1_arm.py` now records the grouped-nf4-gemm commit and the sha256 of the `nf4_qlora.py` it imports
  (`env.gnf4_code`). The read leads with the reserved and nvidia-smi peaks, and its LM-loss bytes name the held logits.
- The rule takes medians of two draws. A candidate's reserved and device drops must clear the larger of 0.10 GB and the draw spread. The
  4090 arithmetic is reported beside, never as a verdict.
- `bench/tc1/tc1_run.sh` (`qwen3pos76co`, `gnf4_t212_at`), `bench/tc1/tc1_drive.sh` (`GNF4_SHA_B` forwarded and checked), and
  `bench/tc1/tc1_reduce.py` (`pos76_why`, `pos76_rule`, `pos76_block`, self-test case 131). Tests cover the helper, box 155's cd1cl1
  cell, and the family's pre-check and switch order, run from the script's own functions against a fake python and git.
