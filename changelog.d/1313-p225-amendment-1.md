### Changed

- P225 Amendment 1 (pre-data): the refusal-retry run ids `p225-5090-r1` to `p225-5090-r4`, exactly the registered
  HOST_REFUSED cap of 4, accepted by the runner and driver at the same tier and terms (`bench/p225/PREREG-p225.md`). A
  refused box keeps its own id and receipt with no strike; a retry box that starts takes the next unused box-started
  slot. The 3-box-started cap, the strike rules and the stop are unchanged.
