# P68 -- generated results (p68_reduce.py; read against P68-PREREG.md)

## G0 -- validity
- **nf4**: OK
- **int4**: OK

## The ablation (first differing layer.site per position)

| stack | arm | mode | exact / positions | first differences | KL mean | flips | predicted | verdict |
|---|---|---|---|---|---|---|---|---|
| nf4 | hf.base | verify17 | 0/68 | {'L0.attn_core': 68} | 1.64e-02 | 4 | not exact; first diff at L0 in ['attn_core'] | HELD |
| nf4 | hf.base | verify16 | 0/64 | {'L0.attn_core': 64} | 2.89e-02 | 3 | not exact; first diff at L0 in ['attn_core'] | HELD |
| nf4 | hf.base | prefill | 0/160 | {'L0.attn_core': 160} | 1.85e-02 | 11 | not exact; first diff at L0 in ['attn_core'] | HELD |
| nf4 | hf.proj | verify17 | 0/68 | {'L0.attn_core': 58, 'L0.mlp_out': 6, 'L1.attn_core': 4} | 2.62e-02 | 7 | not exact; > 50 % first diff at L0 in ['attn_core', 'attn_out'] | HELD |
| nf4 | hf.proj | verify16 | 0/64 | {'L0.attn_core': 54, 'L0.mlp_out': 7, 'L1.attn_core': 3} | 1.20e-02 | 3 | not exact; > 50 % first diff at L0 in ['attn_core', 'attn_out'] | HELD |
| nf4 | hf.proj | prefill | 0/160 | {'L0.mlp_out': 70, 'L0.attn_core': 64, 'L1.attn_core': 25, 'L2.mlp_out': 1} | 2.03e-02 | 11 | None | INFO |
| nf4 | hf.core | verify17 | 0/68 | {'L0.attn_core': 68} | 1.41e-02 | 2 | not exact; > 50 % first diff at L0 in ['attn_core', 'attn_out'] | HELD |
| nf4 | hf.core | verify16 | 0/64 | {'L0.attn_core': 64} | 2.27e-02 | 6 | not exact; > 50 % first diff at L0 in ['attn_core', 'attn_out'] | HELD |
| nf4 | hf.core | prefill | 0/160 | {'L0.attn_core': 160} | 1.82e-02 | 11 | None | INFO |
| nf4 | hf.proj_core | verify17 | 0/68 | {'L0.mlp_out': 46, 'L1.attn_core': 21, 'L2.mlp_out': 1} | 2.13e-02 | 2 | no first diff at L0 in ['attn_core', 'attn_in', 'attn_out'] | HELD |
| nf4 | hf.proj_core | verify16 | 0/64 | {'L0.mlp_out': 46, 'L1.attn_core': 14, 'L1.mlp_out': 3, 'L2.mlp_out': 1} | 1.45e-02 | 3 | no first diff at L0 in ['attn_core', 'attn_in', 'attn_out'] | HELD |
| nf4 | hf.proj_core | prefill | 0/160 | {'L0.mlp_out': 111, 'L1.attn_core': 48, 'L2.mlp_out': 1} | 1.94e-02 | 8 | no first diff at L0 in ['attn_core', 'attn_in', 'attn_out'] | HELD |
| nf4 | hf.all | verify17 | 68/68 | {'exact': 68} | 0.00e+00 | 0 | EXACT | HELD |
| nf4 | hf.all | verify16 | 64/64 | {'exact': 64} | 0.00e+00 | 0 | EXACT | HELD |
| nf4 | hf.all | prefill | 160/160 | {'exact': 160} | 0.00e+00 | 0 | EXACT | HELD |
| nf4 | hf.fp32red | verify17 | 0/68 | {'L0.attn_core': 66, 'L0.attn_out': 2} | 2.35e-02 | 6 | None | INFO |
| nf4 | hf.fp32red | verify16 | 0/64 | {'L0.attn_core': 62, 'L0.attn_out': 2} | 2.62e-02 | 4 | None | INFO |
| nf4 | hf.fp32red | prefill | 0/160 | {'L0.attn_core': 160} | 2.09e-02 | 12 | None | INFO |
| nf4 | paged.base | verify17 | 0/68 | {'L0.attn_core': 68} | 1.96e-02 | 1 | None | INFO |
| nf4 | paged.base | verify16 | 0/64 | {'L0.attn_core': 64} | 1.59e-02 | 2 | None | INFO |
| nf4 | paged.base | prefill | 0/160 | {'L0.attn_core': 160} | 3.08e-02 | 12 | None | INFO |
| nf4 | paged.proj | verify17 | 0/68 | {'L0.mlp_out': 40, 'L1.attn_core': 27, 'L1.mlp_out': 1} | 3.30e-02 | 4 | None | INFO |
| nf4 | paged.proj | verify16 | 0/64 | {'L0.mlp_out': 46, 'L1.attn_core': 16, 'L1.mlp_out': 2} | 2.70e-02 | 2 | None | INFO |
| nf4 | paged.proj | prefill | 0/160 | {'L0.attn_core': 160} | 3.18e-02 | 14 | not exact; first diff at L0 in ['attn_core'] | HELD |
| int4 | hf.base | verify17 | 0/68 | {'L0.attn_core': 68} | 1.07e-02 | 6 | not exact; first diff at L0 in ['attn_core'] | HELD |
| int4 | hf.base | verify16 | 0/64 | {'L0.attn_core': 64} | 1.38e-02 | 4 | not exact; first diff at L0 in ['attn_core'] | HELD |
| int4 | hf.base | prefill | 0/160 | {'L0.attn_in': 160} | 1.57e-02 | 8 | not exact; first diff at L0 in ['attn_in'] | HELD |
| int4 | hf.proj | verify17 | 0/68 | {'L0.attn_core': 56, 'L0.mlp_out': 6, 'L1.attn_core': 6} | 1.22e-02 | 4 | not exact; > 50 % first diff at L0 in ['attn_core', 'attn_out'] | HELD |
| int4 | hf.proj | verify16 | 0/64 | {'L0.attn_core': 53, 'L0.mlp_out': 9, 'L1.attn_core': 2} | 1.39e-02 | 4 | not exact; > 50 % first diff at L0 in ['attn_core', 'attn_out'] | HELD |
| int4 | hf.proj | prefill | 0/160 | {'L0.attn_in': 160} | 1.65e-02 | 11 | None | INFO |
| int4 | hf.core | verify17 | 0/68 | {'L0.attn_core': 68} | 1.83e-02 | 6 | not exact; > 50 % first diff at L0 in ['attn_core', 'attn_out'] | HELD |
| int4 | hf.core | verify16 | 0/64 | {'L0.attn_core': 64} | 1.41e-02 | 5 | not exact; > 50 % first diff at L0 in ['attn_core', 'attn_out'] | HELD |
| int4 | hf.core | prefill | 0/160 | {'L0.attn_in': 160} | 1.71e-02 | 11 | None | INFO |
| int4 | hf.proj_core | verify17 | 0/68 | {'L0.mlp_out': 44, 'L1.attn_core': 22, 'L1.mlp_out': 1, 'L2.mlp_out': 1} | 1.29e-02 | 3 | no first diff at L0 in ['attn_core', 'attn_in', 'attn_out'] | HELD |
| int4 | hf.proj_core | verify16 | 0/64 | {'L0.mlp_out': 50, 'L1.attn_core': 14} | 1.08e-02 | 3 | no first diff at L0 in ['attn_core', 'attn_in', 'attn_out'] | HELD |
| int4 | hf.proj_core | prefill | 0/160 | {'L0.attn_in': 160} | 1.64e-02 | 11 | not exact; first diff at L0 in ['attn_in'] | HELD |
| int4 | hf.all | verify17 | 68/68 | {'exact': 68} | 0.00e+00 | 0 | EXACT | HELD |
| int4 | hf.all | verify16 | 64/64 | {'exact': 64} | 0.00e+00 | 0 | EXACT | HELD |
| int4 | hf.all | prefill | 1/160 | {'L0.attn_core': 159, 'exact': 1} | 1.58e-02 | 11 | not exact; first diff at L0 in ['attn_core'] | REFUTED |
| int4 | hf.fp32red | verify17 | 0/68 | {'L0.attn_core': 68} | 1.21e-02 | 4 | None | INFO |
| int4 | hf.fp32red | verify16 | 0/64 | {'L0.attn_core': 64} | 1.05e-02 | 4 | None | INFO |
| int4 | hf.fp32red | prefill | 0/160 | {'L0.attn_in': 160} | 1.91e-02 | 7 | None | INFO |
| int4 | paged.base | verify17 | 0/68 | {'L0.attn_core': 68} | 1.76e-02 | 4 | None | INFO |
| int4 | paged.base | verify16 | 0/64 | {'L0.attn_core': 64} | 2.90e-02 | 3 | None | INFO |
| int4 | paged.base | prefill | 0/160 | {'L0.attn_in': 160} | 3.06e-02 | 8 | None | INFO |
| int4 | paged.proj | verify17 | 0/68 | {'L0.mlp_out': 45, 'L1.attn_core': 21, 'L1.mlp_out': 2} | 1.76e-02 | 4 | None | INFO |
| int4 | paged.proj | verify16 | 0/64 | {'L0.mlp_out': 35, 'L1.attn_core': 25, 'L1.mlp_out': 3, 'L2.attn_core': 1} | 1.80e-02 | 2 | None | INFO |
| int4 | paged.proj | prefill | 0/160 | {'L0.attn_in': 160} | 3.27e-02 | 10 | not exact; first diff at L0 in ['attn_in'] | HELD |

## The size reading (served default, unforced; 95 % CI over (row, window) clusters)

| stack | arm | mode | positions | KL mean [CI] | top-1 [CI] | class | prediction |
|---|---|---|---|---|---|---|---|
| nf4 | size.hf | prefill | 2048 | 5.056e-03 [3.856e-03, 6.468e-03] | 0.9814 [0.9746, 0.9873] | WITHIN-BAR | HELD |
| nf4 | size.hf | verify16 | 896 | 2.514e-03 [2.067e-03, 3.045e-03] | 0.9855 [0.9777, 0.9922] | WITHIN-BAR | HELD |
| nf4 | size.hf | verify17 | 952 | 3.345e-03 [2.115e-03, 5.451e-03] | 0.9821 [0.9748, 0.9895] | WITHIN-BAR | HELD |
| nf4 | size.paged | prefill | 2048 | 1.688e-02 [1.170e-02, 2.514e-02] | 0.9546 [0.9453, 0.9634] | WITHIN-BAR | HELD |
| nf4 | size.paged | verify16 | 896 | 9.531e-03 [6.608e-03, 1.406e-02] | 0.9743 [0.9643, 0.9833] | WITHIN-BAR | HELD |
| nf4 | size.paged | verify17 | 952 | 1.423e-02 [7.103e-03, 2.677e-02] | 0.9580 [0.9454, 0.9695] | WITHIN-BAR | HELD |
| int4 | size.hf | prefill | 2048 | 5.793e-03 [4.912e-03, 6.908e-03] | 0.9600 [0.9497, 0.9692] | WITHIN-BAR | HELD |
| int4 | size.hf | verify16 | 896 | 3.328e-03 [2.834e-03, 3.913e-03] | 0.9654 [0.9554, 0.9743] | WITHIN-BAR | HELD |
| int4 | size.hf | verify17 | 952 | 3.293e-03 [2.677e-03, 4.030e-03] | 0.9685 [0.9569, 0.9790] | WITHIN-BAR | HELD |
| int4 | size.paged | prefill | 2048 | 1.451e-02 [1.266e-02, 1.643e-02] | 0.9463 [0.9355, 0.9570] | WITHIN-BAR | HELD |
| int4 | size.paged | verify16 | 896 | 1.164e-02 [7.527e-03, 1.882e-02] | 0.9587 [0.9464, 0.9710] | WITHIN-BAR | HELD |
| int4 | size.paged | verify17 | 952 | 9.404e-03 [8.063e-03, 1.098e-02] | 0.9569 [0.9464, 0.9674] | WITHIN-BAR | HELD |

## Verdicts
- ablation predictions held: 27; refuted: ['int4 hf.all prefill']
- a verify built from T = 1 kernels row by row is bit-exact (hf.all, verify): {'nf4': True, 'int4': True}
- size classes: ['WITHIN-BAR']; size prediction refuted: none
