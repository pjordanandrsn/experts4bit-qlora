### Added

- `bench/common/host_facts.py`: a runner can write the box's own measured host facts as `host_facts.json`, for the
  host registry. It records CPU topology, NUMA and transparent-hugepage state, the cgroup CPU quota and GPU 0's power,
  clock and memory, each with its source and a `measured_at` stamp. Only numbers and small enums are written, never an
  identifier. A fact it cannot read is left out, and the helper never fails a run.
