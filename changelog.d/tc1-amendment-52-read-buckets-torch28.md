### Read: TC1 amendment 52 -- bucketed padding costs torch 2.8 nothing on packed rows (P130-P133 HELD)

- `tc1-5090-109` ($2.35, an AMD EPYC 7713): amendment 48's packed box in the field image's torch 2.8. Buckets against the single block:
  matched 0.983 [0.974, 0.993], shipped 0.939 [0.929, 0.949]; the matched peak drops 4.24 GB; held-out within 0.0002. Row
  `e4b.train.pad-buckets.torch28.qwen3.5090.2026-10-06`. grouped-nf4-gemm's `auto` default stands in torch 2.8, and amendment 51's
  environment ratio (0.739) is not the buckets' cost.
- Reported, not scored: under torch 2.8 the bucketed arms left the GPU idle more of the step (median utilisation 87 % against 97 %).
  Host-side time in the bucketed delta under torch 2.8 is the next lead; STATUS says so.
