### TC1 arm: an OOM row keeps its environment and the allocator's text; the CPU pools and allocator setting recorded; an opt-in per-step allocator record

- A training OOM's row (`status: oom`) now carries the arm's `env` block and `arm_facts`, as a trained row does, plus the micro-batches'
  padded lengths up to the failure. Its reason keeps the allocator's message to 700 characters (it was 200), so the split between
  memory PyTorch has allocated and memory it has reserved but not allocated survives into the receipt.
- `env` records `PYTORCH_CUDA_ALLOC_CONF`, the allocator setting the arm ran under (`tc1_run.sh` sets `expandable_segments:True`).
- `arm_facts.threads` records the CPU pools every arm ran with: torch's intra- and inter-op pools, OMP / MKL / OpenBLAS as set, the
  affinity CPUs, the cgroup quota, and the cgroup's throttled-period count at the arm's start and end. These are recorded only and
  never gated. The default path adds exactly these two keys to a trained row (`arm_facts.threads`, `env.pytorch_cuda_alloc_conf`);
  real receipts reduce and render identically with or without them.
- `TC1_STEP_MEM=1` (opt-in; off by default, and then no new key) adds `step_mem`: at each optimizer step's end, the running maxima of
  allocated and reserved memory, the current allocated and reserved memory, and the device's used bytes. The record never resets peak
  statistics, so `mem_census` and the record's window peaks are unchanged. Under `--phase-peaks 1`, where each phase boundary resets
  the allocator's peaks, both maxima are folded back over the reset phases, so they stay running maxima. The row at which one first
  rises bounds the step that set the window peak; in-loop evals run after a step's row, so a rise first seen at row N+1 may have been
  set by the eval after step N. An OOM row carries the steps it completed.
- `tc1_drive.sh` forwards `TC1_STEP_MEM` with the other knobs. `bench/q235/q235.sha256` re-seals `tc1_arm.py`.
