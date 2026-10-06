### DQ3's box runner gates the PCIe width negotiated under load, and refuses out-of-band hosts at rc 19, not 13 (bench only; #1262, closes #1216)

- **The width gate.** `bench/dq3/dq3_run.sh` runs DQ5's link gate verbatim (`bench/dq5/dq5_link_gate.py`).
  `nvidia-smi`'s `width.max` is what the slot can negotiate, not what it did, so the gate reads the width during a
  pinned copy instead.
- **The exit code.** An out-of-band link now exits 19, a code adertha does not admit as machine evidence. Before, a
  good gen 4 5090 refused by DQ3 could have been excluded from other lanes' searches.
- **DQ3's read is unaffected.** Its host measured 53.9 GB/s, which is only possible at x16.
