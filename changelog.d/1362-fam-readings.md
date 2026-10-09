### FAM readings (#1362): the B=1 fused stack at T == 1 on Granite, gpt-oss and Qwen3.6

Each family is read against its own neutral floor on one RTX 5090 ($6.456 for the lane). Granite-MoE fails on one gated
entry of 12. gpt-oss-20b fails on every knob, because its own floor sits below the 0.90 agreement backstop; the anchor
places Phase C's 0.924 inside that floor, so Phase C's miss was the gate, not the knob. Qwen3.6-MoE passes, under
Amendment 3's re-reduction. No default changes here. Receipts: `bench/fam/receipts/`; results:
`bench/fam/RESULTS-fam.md`; claims: `e4b.serve.fam.fused-stack-t1.*.5090.2026-10-09`.
