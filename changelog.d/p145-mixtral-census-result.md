### Read: P145, Mixtral-8x7B's one-request step after K33, kernel by kernel (one RTX 5090): READ

With K33 serving the expert GEMVs by default (grouped-nf4-gemm 0.47.0), the top addressable line at one request is the bf16
attention q/o projections: 1.471 ms of an 8.125 ms node replay period. The expert GEMVs now run below their isolated plan
time. The proposed next lever is a format premise for those projections, not licensed by this read. `bench/p145/RESULTS-p145.md`
and the receipts under `bench/p145/receipts/`. An instrument: no default, gate or speed claim moves.
