### Added

- `bench/gpu-contract/`: a kernel PR's GPU contract on one rented card of a declared class (`GC_CARD`, compute capability and
  SM count). At two pinned commits (`GC_GNF4_SHA`, `GC_E4B_SHA`; anything but 40 hex is refused), it runs grouped-nf4-gemm's
  `kernel/test_nf4_gemv_bw.py` and `kernel/test_dispatch_counts.py`, and the three served-route premise tests P133's box ran.
  Each suite runs with `-rs` and a junit report. The driver refuses a dirty launch tree and records the tool's commit. It is
  a correctness contract, not a reading. `tests/test_gpu_contract.py`.
