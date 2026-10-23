### Added

- P230 registered: grouped-nf4-gemm's programmatic dependent launch, as it ships (on, capped at 8 rows), read for the
  first time on the DEFAULT NF4 server (`bench/p230/PREREG-p230.md`). P113 read it on SC1's int4 configuration; since
  K33 took the switch, the default server at one request launches four switched kernels a layer. Four arms in a
  palindrome (`GNF4_PDL=0` against the shipped default) through W1, where the cap lets PDL engage, and W16, the control;
  decode-only timing; identical tokens required; KEEP / NULL / SLOWER bars set from the instrument's self-pair spread
  only. The runner refuses before any install, checks the pre-install fit with 1,048 s of boot slack, writes a start
  marker before the first install, and every fetched box is classified mechanically (`p230_reduce.py --classify`). No
  default, gate or claim moves.
