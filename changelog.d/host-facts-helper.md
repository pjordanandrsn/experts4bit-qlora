### Added

- `bench/common/host_facts.py`: a runner can write the box's own measured host facts as `host_facts.json`, for the
  host registry. It records the host's CPU topology and NUMA nodes (`host_*`), the run's own CPU share (affinity and
  the cgroup quota, v2 or v1), transparent-hugepage state and GPU 0's power, clock and memory, each with its source,
  plus a `measured_at` stamp. Only numbers and small enums are written, never an identifier. A fact it cannot read is
  left out, and the helper never fails a run.
