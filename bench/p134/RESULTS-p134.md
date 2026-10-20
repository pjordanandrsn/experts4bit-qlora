# P134 results: the bandwidth GEMV on Granite and OLMoE (LICENSED for both, g1 1.3979 and 1.5011)

**The read** (`p134-5090-1`, one RTX 5090, 2026-10-18; receipts under [`receipts/p134-5090-1/`](receipts/p134-5090-1/)).
On the default `serve_paged` server, `GNF4_GEMV_BW=1` routes the single-row decode GEMVs through grouped-nf4-gemm K33's
bandwidth-targeted kernel. That decoded one request faster than the shipped scalar GEMV on both families:
- Granite-3.1-3b-a800m **1.3979×** and OLMoE-1B-7B **1.5011×**, each the smaller of two ABBA pairs, against a 1.03 bar;
- sixteen requests 0.9992 and 0.9954 (bar 0.99), and quality within P110's bar on both.

**LICENSED, per family.**

The registration is [`PREREG-p134.md`](PREREG-p134.md) (#1697, with Amendment 1, #1704). The box ran e4b `c8d7e629` and
grouped-nf4-gemm 0.46.0 at `6b7c2e74`. Both verdicts were re-derived from the private receipt store, byte-identical to
the box's, and they re-derive from the public records here too (the receipt README gives the command). Under P134's
consequence, grouped-nf4-gemm 0.47.0 added each family's two expert shapes to `auto` on >= 160-SM parts. It licenses
shapes, not families.

## The numbers

| | Granite B0 → B1 | OLMoE B0 → B1 |
|---|---|---|
| W1 decode, ms a step (block a) | 5.0574 → 3.6142 | 4.0499 → 2.6980 |
| **g1**, the smaller W1 pair | **1.3979** (1.3993, 1.3979) | **1.5011** (1.5011, 1.5069) |
| **gW** at W16 | **0.9992** (0.9992, 1.0016) | **0.9954** (0.9954, 1.0005) |
| wikitext bias (bar); K8 Δppl (budget 0.05) | +0.00260 (0.01338); +0.0161 | -0.00238 (0.01087); -0.0234 |
| c4val1 bias (bar); K8 (reported) | +0.00459 (0.01042); +0.0503 | -0.00089 (0.01380); -0.0124 |

- **Engagement:** a build's calls went all to the scalar GEMV under B0 and all to `bw_prmt32` under B1 (Granite 192,
  OLMoE 96).
- **Tokens:** not bitwise, as registered. OLMoE's B1 matched B0 on every W1 row and on 31 of 32 W16 rows. Granite's
  differed on the W1 row at 160 tokens and on 3 of 32 W16 rows.
- **Self-pairs:** inside the registered bands, the widest W16's 0.9927 (OLMoE B0).
- **Cost:** $0.480; the proof `p134-5090-prove-1` cost $0.414. The rerun id `p134-5090-2` was unused.

## The predictions, graded

| # | prediction | read | grade |
|---|---|---|---|
| Q1 | both families fit at `max_seqs` 64 | both built at 64 | HELD |
| Q2 | engagement exact, every bucket captured, B1 ≠ B0 on some W1 rows, self-pairs bitwise | Granite's W1 differed; OLMoE's W1 tokens were identical | HELD for Granite; for OLMoE, all but B1 ≠ B0 |
| Q3 | Granite g1 ∈ [1.17, 1.37] | 1.3979 | MISSED (high) |
| Q4 | OLMoE g1 ∈ [1.218, 1.483], the band the proof narrowed it to before the reading (provisional [1.10, 1.70]) | 1.5011 | MISSED (high) |
| Q5 | gW ∈ [0.99, 1.01] for both | 0.9992, 0.9954 | HELD |
| Q6 | quality within the bar for both | both within it | HELD |

## Caveats

- **A capped card:** the RTX 5090 read a **400 W** power limit, below the class's stock 575 W. This read predates the
  575 W floor that later lanes register. Both arms ran on the same card, so the ratios are paired, but they are a 400 W
  card's.
- **Per family, at these shapes:** Granite's c4val1 K8 (+0.0503) is above the 0.05 budget that is gated on wikitext only.
  It is reported, as registered, and does not rule.
- **Not bitwise:** the bandwidth GEMV ships on P110's quality gate.
