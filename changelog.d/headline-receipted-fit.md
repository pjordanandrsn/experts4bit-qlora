### Docs: the 12 GB headline cites the receipted A2000 training read

The README's first screen, its results table, STATUS, CHOOSING, STORAGE-MODES and the larger-than-VRAM solution page
now lead with `e4b.train.frontier.qwen3.a2000-12gb.2026-10-02`. That read has Qwen3-30B-A3B training TC1's field recipe
at micro-batch 1 × accum 8 at 10.463 GB peak on an RTX A2000 12 GB, from a pinned revision with receipts in
`bench/h2h-2026-10-02/tc3/`. The older 7.16 GB reading (`e4b.offload.fits-30b-class`, 5 steps at seq 128) recorded no
model revision or run id and was not reproduced. It stays beside the new one, labelled that way. No claim value changes.
