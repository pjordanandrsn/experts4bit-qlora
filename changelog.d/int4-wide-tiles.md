### `E4B_INT4_WIDE_TILES=1`: the one-launch tile table above 256 routed rows (opt-in)

A decode step above 32 rows at top-k 8 routes more than 256 rows and takes K19's prefill route, whose expert-major
tile table the chained builder makes (argsort, scatter, cumsum, searchsorted, index_select). Lane P119 read it at
2.87 ms of a 15.64 ms 64-row step on an RTX 5090, against 0.93 ms for the one-launch table at 256 rows.

`E4B_INT4_WIDE_TILES=1` builds that table in one launch with grouped-nf4-gemm's cumsum rank
(`build_group_tiles_fused(..., rank="cumsum")`) for every device-grouped call of 257 to 1024 routed rows (the int4
store's K19 route and the NF4 store's M-tile alike). Its integers are the chained
builder's, so outputs are bit-identical. The default stays `0` until a registered read licenses it; `1` on a kernel
package without `rank=` is refused; prefill chunks (more rows) keep the chained builder.
