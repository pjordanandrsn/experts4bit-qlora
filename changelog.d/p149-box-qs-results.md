### P149 box Q+S read: int8 attention projections on Mixtral-8x7B, FLIP (g1 1.1608), routing qualifier VOID

On one RTX 5090, `E4B_SERVE_ATTN_INT8=1` (K37's int8 kernels for Mixtral's 128 attention projections) decoded one
request 1.1608x as fast as the default server (the smallest of four ABBA pairs), and sixteen requests 1.0181x. Quality
at decode passed first: P147's gates through the served step read GO at both widths (W1 upper bound 0.002311 against
0.011537). The token-pinned routing qualifier read QUALIFIER_VOID: S0, fed its own stream, differed from it at 5 forced tokens over
1,100 decode steps in every block. So the default-flip PR is held for a routing-matched read, and the lever stays opt-in. The g1
prediction band [1.04, 1.08] was missed toward faster. Record: `bench/p149/RESULTS-p149.md`, receipts under
`bench/p149/receipts/p149-5090-2/`, and the claim `e4b.serve.p149.int8-attn.mixtral.5090.2026-10-20`.
