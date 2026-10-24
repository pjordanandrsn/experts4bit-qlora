### P236 Amendment 2: the gnf4 engagement gate accepts K33's split-K mark (after p236-5090-1 read VOID)

`p236-5090-1` read VOID, a strike: the gate refused `mixtral.down` at every M and those cells were never timed. In
grouped-nf4-gemm 0.51.0 a K33 call whose plan splits K (Mixtral-8x7B's down projection, split 2) bumps `bw_splitk`
beside `bw_prmt32` for its one launch, which the registered gate did not allow; grouped-nf4-gemm's own K33 bench
excludes that mark from its route tally. The gate now expects `bw_splitk == calls` exactly on the registered split shape
and 0 elsewhere. The lane test checks the constant against the pinned kernel's plan table, and the box's tripwire checks
it on the card. Tests run on the VOID box's real engagement records first (no timing is read). The rerun is
`p236-5090-2` from this amendment's merge, under a new Approved.
