### FAM Mixtral read (#1362): the B=1 fused stack passes on Mixtral-8x7B, resolved at a ×0.90 softmax-scale change

`fam-mixtral-4` (one RTX 5090, $2.59) reads PASS on ON_glue, ON_r2, ON_epi and ON_auto against Mixtral's own neutral
floor. At reading scale ×0.90 fails every gated cell, so the result is a null read of that size: no effect as large as
a ×0.90 change of the decode softmax scale. The knobs stay off on Mixtral by default until a speed read (Amendment 5).
Receipts: `bench/fam/receipts/fam-mixtral-prove-1/`, `fam-mixtral-prove-2/`, `fam-mixtral-3/`, `fam-mixtral-4/`; claim
`e4b.serve.fam.fused-stack-t1.mixtral.5090.2026-10-09`.
