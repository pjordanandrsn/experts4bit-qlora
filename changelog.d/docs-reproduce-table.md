### Docs: REPRODUCE.md's "More readings" table

The remaining headline readings, in the page's columns: the claim, the e4b commit and the first release containing it,
the GPU and host, the lane and run, and the cost of one reading. It covers layer-major offload training, the serving
defaults (P116, P127, P133, P135, P141, P143, P153), server capacity, grouped-nf4-gemm's kernel reads and loggetta's
planner estimate. The negatives and limits stand beside them: P155's NO_GO, Gemma-4's served-quality defect, and
vLLM Marlin ahead of grouped-nf4-gemm at M=16. It also says which controllers pin, refuse an override, or need
`GNF4_SHA`.
