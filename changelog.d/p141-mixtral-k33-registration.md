### Registered: P141, K33's bandwidth GEMV at Mixtral's expert shapes (one RTX 5090)

`bench/p141/` registers a read of grouped-nf4-gemm K33 (`GNF4_GEMV_BW=1`) on Mixtral-8x7B at one request. Mixtral's
expert GEMVs take 76 % of its one-request step (P136's census), and K33 has no plan at its shapes, so they run the scalar
GEMV. Phase 1 picks each shape's plan on the card by K33's rule (contract first, then the lowest median of twelve plans
over 40 rounds); phase 2 is P134's served ABBA read with P110's quality bar, with W16 as a no-regression check. The
$0 premise is recorded under `bench/p141/premise/`: an A2000 trace shows e4b's decode calls the GEMV at exactly (28672,
4096) and (4096, 14336), and all twelve plans meet K33's numerics contract there. One reading (`p141-5090-1`), guard
2.0 h, worst $2.80.
