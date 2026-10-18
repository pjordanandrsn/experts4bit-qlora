### Read: P141, K33's bandwidth GEMV at Mixtral-8x7B's expert shapes (one RTX 5090): LICENSED, one request 2.41× as fast

At the plans its phase 1 selected on the card (gate_up (16, 1024, 4, 1), down (16, 512, 4, 2)), grouped-nf4-gemm's
bandwidth-targeted NF4 decode GEMV decodes Mixtral-8x7B NF4 one request 2.4075× as fast as the shipped scalar GEMV on the
default `serve_paged` server (19.5 → 8.1 ms a step). Sixteen requests are unchanged (0.9964), and quality is within P110's
bar. `bench/p141/RESULTS-p141.md`, the receipts under `bench/p141/receipts/`, and claim
`e4b.serve.p141.gemv-bw.mixtral.5090.2026-10-18`. The default moves with a grouped-nf4-gemm release carrying #556; until
then `GNF4_GEMV_BW=1` with `GNF4_GEMV_BW_PLAN="28672,4096=16,1024,4,1;4096,14336=16,512,4,2"` reproduces it.
