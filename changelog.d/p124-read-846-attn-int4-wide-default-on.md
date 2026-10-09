### P124 read (#846): DEFAULT_ON -- the attention projections above 16 rows on the int4 small-M GEMM make Qwen3-30B-A3B's 64- and 32-row decode steps 3.2–3.8 % faster, within P110's quality bar (bench, docs)

- **The reading** (`p124-5090-4`, $0.787; the lane $2.285): SC2e's served stack, interleaved blocks under Amendment 1.
  With `E4B_ATTN_INT4_WIDE=1` the 64-row step reads ON/OFF 0.9648 / 0.9670 and the 32-row step 0.9616 / 0.9679. The
  blocks agree within 0.65 %. The teacher-forced NLL moves +0.0011 nats at 64 rows and +0.0014 at 32, inside P110's
  bar. Both mutants fail it, and the captured run matches the eager one at all 8,128 positions.
- **Why:** the route replaces cuBLAS on a 1.81 GB bf16 copy (12.4 % of the eager step) with K16's int4 tile. The
  64-row tile still costs 0.74 of the time it replaces (0.59 at 32 rows), so a plan census for it is the next lever.
- **Recorded:** `bench/p124/RESULTS-p124.md` and the receipts (SHA256SUMS; the verdict re-derives byte for byte).
  The register row is `e4b.serve.p124.attn-int4-wide.qwen3-int4.5090.2026-10-09`, and STATUS cites it. The registered
  default flip is a separate PR.
