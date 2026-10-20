### Added

- `bench/common/thp_window.py`: a standard-library helper that records transparent huge pages around a lane's timed
  window, inside the measured process. It records the host's THP mode, the process's `THP_MEM_ALLOC_ENABLE`, its
  `AnonHugePages`, `Anonymous` and `Rss`, and the host's `thp_*` counter deltas. A reading that can depend on huge pages
  can then say whether huge pages were possible at the window, and whether the process held any.
