### FAM speed read (#1362): the router epilogue decodes one Qwen3.6 request in 0.973 of the step

Amendment 4's reading (`fam-speed-1`, one RTX 5090, $0.632) reads FASTER. On the default graph server, the fused router
epilogue, the only B=1 fusion that engages on Qwen3.6, cuts the one-row decode step from 12.64 to 12.30 ms: ON/OFF
0.9727 and 0.9732 in two interleaved blocks, with tokens identical. With FAM's quality PASS it licenses the allowlist
flip, which ships separately. Receipts: `bench/fam/receipts/fam-speed-prove-1/`, `fam-speed-1/`; claim
`e4b.serve.fam.router-epilogue-speed.qw36.5090.2026-10-09`.
