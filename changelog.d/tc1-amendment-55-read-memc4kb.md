### Read: TC1 amendment 55 -- on packed rows e4b's run peak is its held-out evaluation; with the double-quantized absmax the gap to Unsloth is 2.01 GB (P141-P143 HELD)

- `tc1-5090-112` ($0.89, an EPYC 7K62): amendment 47's census at the current defaults. Peaks: e4b defaults 28.23 GB, e4b with
  `E4B_ABSMAX_DQ=1` 26.88 GB, Unsloth 24.86 GB. The gaps are +3.36 GB and +2.01 GB. Row
  `e4b.train.memory.packed-4k-census.5090.2026-10-06`.
- e4b's peak falls in the held-out evaluation, where the stock LM loss's fp32 logits and their working copies take about 6.3 GB. The
  chunked loss takes training forwards only, by design. Unsloth's peak falls in a training backward. grouped-nf4-gemm's groups hold
  nothing at e4b's peak. STATUS says so.
- By the registered rule, the next registration reads the double-quantized absmax as a library default on packed rows. It will record
  training and evaluation peaks separately.
