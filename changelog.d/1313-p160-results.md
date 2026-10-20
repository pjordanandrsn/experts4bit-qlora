### P160 results (#1313): decode x plan, GO (C1 1.6598x over the served default)

`p160-5090-2` (one RTX 5090, $0.0710): K39's decode at P156's first carried plan pair, C1 (gate_up 64/256/4/2, down
128/128/8/3), ran both W16 K25 GEMMs 1.6598x faster than prmt at today's plan, the served default. The lower bound was
1.6568, or 1.6525 clock-adjusted, against the 1.05 bar. The read is on S1, a router-replay stream no K25 microbench had
timed. Every arm kept the tree's bits on every rep, and all eight predictions held. The least-overlapping half of S1
read 1.6557, and P156's own held-out steps 1.6508 (P156 reported 1.6167). Microbench only: no default changes and no
speed is claimed. Next come an opt-in per-projection K25 plan knob and a served read, each its own PR. Files:
`bench/p160/RESULTS-p160.md` and `bench/p160/receipts/p160-5090-2/`.
