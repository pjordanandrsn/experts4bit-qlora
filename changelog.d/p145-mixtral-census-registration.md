### Registered: P145, where Mixtral-8x7B's one-request step goes after K33 (an instrument)

`bench/p145/` runs the per-name census (`bench/pername/`) on Mixtral-8x7B at W1 and W16. The subject is the default
server on grouped-nf4-gemm 0.47.0 (#556), which serves Mixtral's expert GEMVs on K33 at P141's plans with no knob set. It
ranks every kernel line by its removable time less its floor (P141's isolated times for the two expert GEMVs, zero
otherwise), and proposes the top W1 line as the next lever. A dense_gemm line on top names a quality premise first, not
a kernel. It changes no default and makes no speed claim. One reading (`p145-5090-1`), guard 1.25 h, worst $2.16.
