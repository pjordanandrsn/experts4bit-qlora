# P68 -- generated results (p68_reduce.py; read against P68-PREREG.md)

## G0 -- validity
- **nf4**: OK
- **int4**: OK

## The ablation (first differing layer.site per position)

| stack | arm | mode | exact / positions | first differences | KL mean | flips | predicted | verdict |
|---|---|---|---|---|---|---|---|---|
| nf4 | hf.base | verify17 | 0/68 | {'L0.attn_core': 68} | 5.77e-04 | 0 | not exact; first diff at L0 in ['attn_core'] | HELD |
| nf4 | hf.base | verify16 | 0/64 | {'L0.attn_core': 64} | 5.75e-04 | 0 | not exact; first diff at L0 in ['attn_core'] | HELD |
| nf4 | hf.base | prefill | 0/160 | {'L0.attn_core': 160} | 1.42e-03 | 2 | not exact; first diff at L0 in ['attn_core'] | HELD |
| nf4 | hf.proj | verify17 | 0/68 | {'L0.attn_core': 68} | 8.73e-04 | 1 | not exact; > 50 % first diff at L0 in ['attn_core', 'attn_out'] | HELD |
| nf4 | hf.proj | verify16 | 0/64 | {'L0.attn_core': 64} | 8.17e-04 | 1 | not exact; > 50 % first diff at L0 in ['attn_core', 'attn_out'] | HELD |
| nf4 | hf.proj | prefill | 0/160 | {'L0.attn_core': 126, 'L9.attn_core': 29, 'L8.mlp_out': 2, 'L1.attn_core': 2, 'logits': 1} | 8.56e-04 | 1 | None | INFO |
| nf4 | hf.core | verify17 | 0/68 | {'L0.attn_core': 55, 'L0.attn_out': 12, 'L0.mlp_out': 1} | 1.10e-03 | 1 | not exact; > 50 % first diff at L0 in ['attn_core', 'attn_out'] | HELD |
| nf4 | hf.core | verify16 | 0/64 | {'L0.attn_core': 64} | 7.66e-04 | 1 | not exact; > 50 % first diff at L0 in ['attn_core', 'attn_out'] | HELD |
| nf4 | hf.core | prefill | 0/160 | {'L0.attn_core': 160} | 1.36e-03 | 3 | None | INFO |
| nf4 | hf.proj_core | verify17 | 0/68 | {'L0.mlp_out': 46, 'L1.attn_core': 20, 'L1.mlp_out': 1, 'L2.mlp_out': 1} | 4.82e-04 | 0 | no first diff at L0 in ['attn_core', 'attn_in', 'attn_out'] | HELD |
| nf4 | hf.proj_core | verify16 | 0/64 | {'L0.mlp_out': 40, 'L1.attn_core': 23, 'L1.mlp_out': 1} | 6.63e-04 | 0 | no first diff at L0 in ['attn_core', 'attn_in', 'attn_out'] | HELD |
| nf4 | hf.proj_core | prefill | 0/160 | {'L6.attn_core': 119, 'L9.attn_core': 37, 'L8.mlp_out': 2, 'logits': 1, 'L5.mlp_out': 1} | 4.85e-04 | 0 | no first diff at L0 in ['attn_core', 'attn_in', 'attn_out'] | HELD |
| nf4 | hf.all | verify17 | 68/68 | {'exact': 68} | 0.00e+00 | 0 | EXACT | HELD |
| nf4 | hf.all | verify16 | 64/64 | {'exact': 64} | 0.00e+00 | 0 | EXACT | HELD |
| nf4 | hf.all | prefill | 160/160 | {'exact': 160} | 0.00e+00 | 0 | EXACT | HELD |
| nf4 | hf.fp32red | verify17 | 0/68 | {'L0.attn_core': 68} | 5.44e-04 | 0 | None | INFO |
| nf4 | hf.fp32red | verify16 | 0/64 | {'L0.attn_core': 64} | 4.88e-04 | 0 | None | INFO |
| nf4 | hf.fp32red | prefill | 0/160 | {'L0.attn_core': 159, 'L0.attn_out': 1} | 9.00e-04 | 1 | None | INFO |
| nf4 | paged.base | verify17 | 0/68 | {'L0.attn_core': 68} | 1.51e-03 | 2 | None | INFO |
| nf4 | paged.base | verify16 | 0/64 | {'L0.attn_core': 64} | 1.14e-03 | 0 | None | INFO |
| nf4 | paged.base | prefill | 0/160 | {'L0.attn_core': 160} | 3.28e-03 | 4 | None | INFO |
| nf4 | paged.proj | verify17 | 0/68 | {'L0.attn_core': 68} | 1.44e-03 | 1 | None | INFO |
| nf4 | paged.proj | verify16 | 0/64 | {'L0.attn_core': 64} | 1.69e-03 | 0 | None | INFO |
| nf4 | paged.proj | prefill | 0/160 | {'L0.attn_core': 160} | 3.59e-03 | 2 | not exact; first diff at L0 in ['attn_core'] | HELD |
| int4 | hf.base | verify17 | 0/68 | {'L0.attn_core': 68} | 1.81e-03 | 1 | not exact; first diff at L0 in ['attn_core'] | HELD |
| int4 | hf.base | verify16 | 0/64 | {'L0.attn_core': 64} | 1.63e-03 | 0 | not exact; first diff at L0 in ['attn_core'] | HELD |
| int4 | hf.base | prefill | 0/160 | {'L0.attn_in': 160} | 2.60e-03 | 2 | not exact; first diff at L0 in ['attn_in'] | HELD |
| int4 | hf.proj | verify17 | 0/68 | {'L0.attn_core': 68} | 1.84e-03 | 0 | not exact; > 50 % first diff at L0 in ['attn_core', 'attn_out'] | HELD |
| int4 | hf.proj | verify16 | 0/64 | {'L0.attn_core': 64} | 1.82e-03 | 0 | not exact; > 50 % first diff at L0 in ['attn_core', 'attn_out'] | HELD |
| int4 | hf.proj | prefill | 0/160 | {'L0.attn_in': 160} | 2.14e-03 | 4 | None | INFO |
| int4 | hf.core | verify17 | 0/68 | {'L0.attn_core': 68} | 2.10e-03 | 0 | not exact; > 50 % first diff at L0 in ['attn_core', 'attn_out'] | HELD |
| int4 | hf.core | verify16 | 0/64 | {'L0.attn_core': 64} | 1.88e-03 | 0 | not exact; > 50 % first diff at L0 in ['attn_core', 'attn_out'] | HELD |
| int4 | hf.core | prefill | 0/160 | {'L0.attn_in': 160} | 2.78e-03 | 3 | None | INFO |
| int4 | hf.proj_core | verify17 | 0/68 | {'L0.mlp_out': 68} | 1.60e-03 | 0 | no first diff at L0 in ['attn_core', 'attn_in', 'attn_out'] | HELD |
| int4 | hf.proj_core | verify16 | 0/64 | {'L0.mlp_out': 64} | 1.50e-03 | 0 | no first diff at L0 in ['attn_core', 'attn_in', 'attn_out'] | HELD |
| int4 | hf.proj_core | prefill | 0/160 | {'L0.attn_in': 160} | 2.19e-03 | 4 | not exact; first diff at L0 in ['attn_in'] | HELD |
| int4 | hf.all | verify17 | 68/68 | {'exact': 68} | 0.00e+00 | 0 | EXACT | HELD |
| int4 | hf.all | verify16 | 64/64 | {'exact': 64} | 0.00e+00 | 0 | EXACT | HELD |
| int4 | hf.all | prefill | 160/160 | {'exact': 160} | 0.00e+00 | 0 | not exact; first diff at L0 in ['attn_core'] | REFUTED |
| int4 | hf.fp32red | verify17 | 0/68 | {'L0.attn_core': 68} | 2.10e-03 | 0 | None | INFO |
| int4 | hf.fp32red | verify16 | 0/64 | {'L0.attn_core': 64} | 1.67e-03 | 0 | None | INFO |
| int4 | hf.fp32red | prefill | 0/160 | {'L0.attn_in': 160} | 2.49e-03 | 2 | None | INFO |
| int4 | paged.base | verify17 | 0/68 | {'L0.attn_core': 68} | 3.83e-03 | 0 | None | INFO |
| int4 | paged.base | verify16 | 0/64 | {'L0.attn_core': 64} | 2.07e-03 | 2 | None | INFO |
| int4 | paged.base | prefill | 0/160 | {'L0.attn_in': 160} | 4.46e-03 | 3 | None | INFO |
| int4 | paged.proj | verify17 | 0/68 | {'L0.attn_core': 68} | 1.78e-03 | 0 | None | INFO |
| int4 | paged.proj | verify16 | 0/64 | {'L0.attn_core': 64} | 2.42e-03 | 0 | None | INFO |
| int4 | paged.proj | prefill | 0/160 | {'L0.attn_in': 160} | 4.34e-03 | 4 | not exact; first diff at L0 in ['attn_in'] | HELD |

## The size reading (served default, unforced; 95 % CI over (row, window) clusters)

| stack | arm | mode | positions | KL mean [CI] | top-1 [CI] | class | prediction |
|---|---|---|---|---|---|---|---|
| nf4 | size.hf | prefill | 2048 | 1.917e-03 [1.611e-03, 2.321e-03] | 0.9673 [0.9600, 0.9746] | WITHIN-BAR | HELD |
| nf4 | size.hf | verify16 | 896 | 8.927e-04 [7.160e-04, 1.092e-03] | 0.9766 [0.9676, 0.9855] | WITHIN-BAR | HELD |
| nf4 | size.hf | verify17 | 952 | 8.118e-04 [6.878e-04, 9.440e-04] | 0.9821 [0.9727, 0.9905] | WITHIN-BAR | HELD |
| nf4 | size.paged | prefill | 2048 | 4.587e-03 [4.242e-03, 4.955e-03] | 0.9507 [0.9429, 0.9590] | WITHIN-BAR | HELD |
| nf4 | size.paged | verify16 | 896 | 1.760e-03 [1.512e-03, 2.016e-03] | 0.9732 [0.9621, 0.9833] | WITHIN-BAR | HELD |
| nf4 | size.paged | verify17 | 952 | 1.658e-03 [1.452e-03, 1.870e-03] | 0.9769 [0.9653, 0.9874] | WITHIN-BAR | HELD |
| int4 | size.hf | prefill | 2048 | 2.141e-03 [1.966e-03, 2.330e-03] | 0.9697 [0.9614, 0.9775] | WITHIN-BAR | HELD |
| int4 | size.hf | verify16 | 896 | 1.161e-03 [1.009e-03, 1.332e-03] | 0.9710 [0.9598, 0.9810] | WITHIN-BAR | HELD |
| int4 | size.hf | verify17 | 952 | 1.281e-03 [1.096e-03, 1.476e-03] | 0.9748 [0.9643, 0.9853] | WITHIN-BAR | HELD |
| int4 | size.paged | prefill | 2048 | 5.043e-03 [4.480e-03, 5.728e-03] | 0.9521 [0.9419, 0.9609] | WITHIN-BAR | HELD |
| int4 | size.paged | verify16 | 896 | 2.087e-03 [1.799e-03, 2.398e-03] | 0.9676 [0.9554, 0.9788] | WITHIN-BAR | HELD |
| int4 | size.paged | verify17 | 952 | 2.084e-03 [1.823e-03, 2.352e-03] | 0.9601 [0.9454, 0.9727] | WITHIN-BAR | HELD |

## Verdicts
- ablation predictions held: 27; refuted: ['int4 hf.all prefill']
- a verify built from T = 1 kernels row by row is bit-exact (hf.all, verify): {'nf4': True, 'int4': True}
- size classes: ['WITHIN-BAR']; size prediction refuted: none
