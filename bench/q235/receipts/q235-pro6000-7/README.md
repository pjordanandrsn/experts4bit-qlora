# `q235-pro6000-7`: VOID at the box's time-left guard (rc 142), strike two

One RTX PRO 6000 Blackwell Workstation Edition on machine 115349 (Vast instance 56635622; 32 logical CPUs, 16 physical
cores), e4b `d65793e4`. Cost $4.299. The load's expert fuse and quantize took about 2,088 s, against about 661 s on
`q235-pro6000-6`'s host. Step 0 began with about 850 s left, and the box's 2,997 s time-left guard ended it (rc 142).
That is a box-started VOID: strike two, so the lane stopped for a simplification.

Amendment 16 is that simplification. It sizes the box's cap on the box (half the time left, less 300 s). It records the
CPU topology (`cpu.json`, never gated) and a per-layer load profile (`load_layers.jsonl`). It makes a host-bound load that
the cap ends a `VOID_BY_HOST` (rc 18), not a strike. Strikes restarted at 0.

The launcher's receipt and ledger row are in the private receipt store. `SHA256SUMS` covers every file here.
