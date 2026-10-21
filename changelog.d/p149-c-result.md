### P149: the bucket-1 residual is not the NF4 expert route

`bench/p149/RESULTS-p149.md`, "The schedule dependence": follow-up (c) (`p149b1-5090-5`) sent a one-row decode step's NF4
experts through K25 or through the M-tile GEMM. Either way, the single bucket-1 difference stayed at the same position
and magnitude, to the bit, so the expert route at one row is not its source. The difference stays unattributed. No
verdict, gate or claim moves.
