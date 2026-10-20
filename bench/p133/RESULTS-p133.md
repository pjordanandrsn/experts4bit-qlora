# P133 results: the bandwidth GEMV on an RTX 4090 (LICENSED, g1 1.5145)

**The read** (`p133-4090-2`, one RTX 4090, 2026-10-18; receipts under [`receipts/p133-4090-2/`](receipts/p133-4090-2/)).
On the default `serve_paged` server with Qwen3-30B-A3B NF4, `GNF4_GEMV_BW=1` routes the single-row decode GEMVs through
grouped-nf4-gemm K33's bandwidth-targeted kernel. That decoded one request **1.5145×** as fast as the shipped route, which
at 128 SMs is the scalar GEMV. That is the smaller of two ABBA pairs; the bar was 1.03. Sixteen requests read
**0.9998** (bar 0.99), and quality stayed within P110's bar. **LICENSED.**

The registration is [`PREREG-p133.md`](PREREG-p133.md) (#1695, with Amendments 1, #1704, and 2, #1711). The box ran e4b
`c8d7e629` and grouped-nf4-gemm 0.46.0 at `6b7c2e74`. The verdict was re-derived from the private receipt store,
byte-identical to the box's, and it re-derives from the public records here too (the receipt README gives the command).
Under P133's consequence, grouped-nf4-gemm 0.47.0 added the RTX 4090 configuration (sm_89, exactly 128 SMs) to `auto`, at
Qwen3-30B-A3B's two expert shapes only.

## The numbers

| | B0 (scalar) | B1 (`GNF4_GEMV_BW=1`) | B1 / B0 |
|---|---|---|---|
| W1 decode, ms a step (a / b) | 8.4759 / 8.4664 | 5.5967 / 5.5870 | **1.5145**, 1.5155 |
| W16 decode, tok/s (a / b) | 573.39 / 570.99 | 573.28 / 571.07 | **0.9998**, 1.0001 |

- **Engagement:** 480 build calls a process, all to the scalar GEMV under B0 and all to `bw_prmt32` under B1.
- **Tokens:** B1 decoded B0's tokens on every row, at W1 and W16 and at both lengths.
- **Self-pairs:** W1 1.0011 (B0) and 1.0017 (B1); W16 0.9958 and 0.9961.
- **Quality** (24 windows a text, T == 1): wikitext bias +0.00325 (bar 0.01387) and K8 Δppl +0.0283 (budget 0.05);
  c4val1 bias -0.00208 (bar 0.01053). Both spreads were within their bars, and the scale mutant failed as it must.
- **Cost:** $0.919. The earlier `p133-4090-1` ($0.50) was aborted before its quality phase, under the registered
  time-left check. It is not a reading.

## The predictions, graded

| # | prediction | read | grade |
|---|---|---|---|
| Q1 | the fit holds at `max_seqs` 16 | built at 16, buckets 1-16, graphs on | HELD |
| Q2 | engagement exact, every bucket captured, B1 ≠ B0 on some W1 rows, self-pairs bitwise | all except B1 ≠ B0: the tokens were identical | HELD, but for B1 ≠ B0 |
| Q3 | g1 ∈ [1.05, 1.30] | 1.5145 | MISSED (high) |
| Q4 | gW ∈ [0.99, 1.01] at W16 | 0.9998 | HELD |
| Q5 | quality within the bar | within it on both texts | HELD |
| Q6 | LICENSED, about 60 % | LICENSED | HELD |

## Caveats

- **One card, one configuration:** an RTX 4090 at 450 W (its stock limit), with Qwen3-30B-A3B's two expert shapes.
  The plans were the RTX 5090's (K33's), unswept on sm_89.
- **Not bitwise:** the bandwidth GEMV has a different reduction tree from the scalar GEMV. It ships on P110's quality
  gate, not a bitwise one, even though this read's tokens were identical.
