### TC1 amendment 77: amendment 76's arms run their CPU pools at the container quota's floor

- `qwen3pos76co`'s arms run OMP, MKL and OpenBLAS at max(1, floor(the cgroup quota)) instead of the host's physical cores, on both sides
  of the ABBA pair. This removes a possible oversubscription when a host's physical cores exceed its quota, and does not change the
  memory read. Every other family is unchanged.
- Before anything loads, the arm refuses if torch's intra-op pool is not that count. `arm_facts.threads` records the pools, the
  variables, the physical cores, the quota and the cgroup's throttled-period count at the arm's start and end. The reducer VOIDs an
  arm whose pools were not set.
- `bench/tc1/tc1_run.sh` (`quota_floor_threads`), `bench/tc1/tc1_arm.py` (`thread_facts`, the preamble check), `bench/tc1/tc1_reduce.py`
  (`pos77_threads_why`, self-test case 131's thread cases). `bench/q235/q235.sha256` re-seals `tc1_arm.py`.
