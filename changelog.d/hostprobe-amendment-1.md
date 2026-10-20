### Added

- `bench/hostprobe/` Amendment 1: the host-facts probe also records GPU 0's driver, CUDA and VBIOS versions, ECC mode
  and error counts, retired pages, remapped rows and the clock-event reasons at the probe's moment. Every rented box it
  reads then carries the facts that could rule a card fault in or out. Each value is left out when the driver does not
  report it. A field the driver does not list is never queried, and an `nvidia-smi` call that hangs stops every later
  one. Cost and scope are unchanged.
