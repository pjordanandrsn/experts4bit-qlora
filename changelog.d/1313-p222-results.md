### P222 results (#1313): K25's decode x plan at Mixtral-8x7B's W16 shapes, GO (M 2.2678x over the served default)

`p222-5090-5` (one RTX 5090, $0.14). At Mixtral-8x7B's W16 shapes, K39's decode at a sweep-selected plan pair (M:
gate_up 64/256/4/2, down 32/256/4/2) ran both K25 GEMMs 2.2678x faster than the served default (the tree at today's
plan). The lower bound was 2.2624, the same clock-adjusted, against the 1.10 bar, on routing draws the selection never
saw. Almost all of the gain is the plan: the decode alone read 1.0187, and c1's pair 2.0630. Every arm kept the tree's
bits on every rep. Four of six predictions held; the decode band missed below and the headline band missed above, at
equal clocks. Microbench only: no default changes and no speed is claimed. Next come an opt-in `_K25_PLANS` entry for
M's pair and a served read, each its own PR. Files: `bench/p222/RESULTS-p222.md` and `bench/p222/receipts/p222-5090-5/`.
