# AMENDMENT-t5b-h-a — re-run Phase A with the block region (2026-08-24)

Registered AFTER the Phase A calculator returned its verdict and BEFORE
any Phase B work. The calculator said **PHASE-A-STOP**: coverage 79.37%
vs the 80.0% bar, top region `moe` at 45.8%. The receipts
(`receipts-t5b/verdict_a.json`) show the compound bar fired on its
secondary clause by 0.63 points while the primary attribution succeeded
decisively — and the miss is attributable to the instrument, not the
bill:

- The attribution arm's step reads 165.4 ms against the 141.0 ms
  un-instrumented baseline (the profiler + region guards inflate every
  bracket). The attention bracket, present in BOTH runs, measures the
  inflation directly: 54.4 profiled vs 46.0 unprofiled = 1.18×.
- Deflating all regions by that factor puts true coverage at ~79% and
  the true residual at ~21%: ~30 ms of dense projections, router,
  norms, embedding and inter-layer glue that Phase A's region set
  simply did not name — a KNOWN composite, not a diffuse mystery.

The STOP branch's premise ("the bill is diffuse") is contradicted by
its own receipts (one region owns 46%). The registered fix is a BETTER
INSTRUMENT, not a lowered bar:

1. Re-run the attribution arm on the `b1-instruments` step_decomp
   (PR #222), whose `moe_block` region wraps the whole sparse-MoE block
   — naming router/top-k (= block − experts) that currently lands in
   the residual.
2. `t5b_verdict.py` gains the block region: regions become {attn,
   moe (experts), router_topk (block − experts), lmhead, sched, drain};
   the SAME 80% coverage and 25% top-region bars apply unchanged.
3. Phase B proceeds only if the re-run passes both clauses; a second
   miss is a REAL diffuse bill and the original STOP consequence
   (re-point at T4 overlap) executes with no further amendment.

The first attribution run's receipts stay in `receipts-t5b/` and the
STOP verdict stays on the record as the calculator's output for that
instrument set.
