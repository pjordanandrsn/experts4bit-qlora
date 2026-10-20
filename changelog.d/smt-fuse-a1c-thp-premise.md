### Added

- `bench/smt-fuse/` Amendment 1c: one more reading, `smt-sib-9`, re-runs `smt-sib-2`'s measurement unchanged on the same
  host, with `bench/common/thp_window.py` records around every timed block and a `fast_state` summary. The definitions
  and outcomes are fixed before data. If the fast fuse state recurs while the measured process holds no huge pages, huge
  pages are falsified as its mechanism; if it does not recur, the result is inconclusive and said so.
