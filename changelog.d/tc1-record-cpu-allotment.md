### TC1 harness: every box records the container's CPU allotment beside the host's cores

- **Why.** The lane sets `OMP_NUM_THREADS` to the host's physical cores, and a rented container can be held to far fewer. The RunPod H100
  pod `tc1c-h100-22` had 18 vCPUs under 72 threads, and e4b's draws there came 18 % and 29 % apart while Unsloth's held. Vast lists 5090
  rentals at a fraction of the host too (for example 24 of 96 cores on machine 152440). No receipt has recorded the allotment.
- **What.** `forensics.txt` gains the cgroup's `cpu.max` (v2) or CFS quota and period (v1), `cpuset.cpus.effective` and the affinity count.
  `box.json` gains `cgroup_cpu_max`, `cgroup_cpuset_effective` and `affinity_cpus`. Recorded only: no arm runs differently.
