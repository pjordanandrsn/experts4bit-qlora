### Changed

- TC3 amendment 6: `bench/tc1/tc1_reduce.py`'s `resident_comparison` now requires the offload anchor and TC1's resident arm to have
  the same expert absmax storage (the receipts' `absmax_dq`; absent = fp32, as before any compression path existed). A mismatched
  pair reads N-A instead of being scored: offloaded stacks keep the fp32 absmax while a resident fused arm double-quantizes it by
  default. The one committed scored pair (`tc3-4090-8` against `tc1-5090-16`) was fp32 on both sides; its verdict stands.
