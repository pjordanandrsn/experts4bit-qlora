### Added

- `bench/hostprobe/`: a registered host-facts probe (`hostprobe-<machine>-<n>`). It rents one named machine, reads the
  host facts other lanes name as bounds on their reads (the transparent-huge-page mode, kernel, cgroup memory limit,
  CPUs and NUMA nodes), writes them to the receipt and tears down. Standard library only, never a reading, worst $0.20
  a probe.
