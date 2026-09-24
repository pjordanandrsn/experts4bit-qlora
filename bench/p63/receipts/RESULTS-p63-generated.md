# P63 — generated results (p63_reduce.py; read against P63-PREREG.md)

Stacks: int4, int4nf, nf4; devices: {'nf4': 'NVIDIA GeForce RTX 5090', 'int4': 'NVIDIA GeForce RTX 5090', 'int4nf': 'NVIDIA GeForce RTX 5090'}

## G0 — instrument gates
- **nf4**: OK
- **int4**: OK
- **int4nf**: OK

## P1 — kernel census (layer-0 gate_up)
| stack | route pair | predicted | observed (by T) | verdict |
|---|---|---|---|---|
| int4 | int4.gemv -> int4.gemv | EXACT | {'16': 'EXACT', '17': 'EXACT', '160': 'EXACT'} | HELD |
| int4nf | int4.gemv -> int4.gemv | EXACT | {'16': 'EXACT', '17': 'EXACT', '160': 'EXACT'} | HELD |
| int4 | int4.gemv -> int4.deq_bf16 | PRECISION | {'16': 'PRECISION', '17': 'PRECISION', '160': 'PRECISION'} | HELD |
| int4nf | int4.gemv -> int4.deq_bf16 | PRECISION | {'16': 'PRECISION', '17': 'PRECISION', '160': 'PRECISION'} | HELD |
| int4 | int4.gemv -> int4.grouped_gemm | REORDER | {'16': 'REORDER', '17': 'REORDER', '160': 'REORDER'} | HELD |
| int4nf | int4.gemv -> int4.grouped_gemm | REORDER | {'16': 'REORDER', '17': 'REORDER', '160': 'REORDER'} | HELD |
| nf4 | nf4.dotpad -> nf4.dotpad | EXACT | {'16': 'EXACT', '17': 'EXACT', '160': 'EXACT'} | HELD |
| nf4 | nf4.dotpad -> nf4.mtile | PRECISION | {'16': 'PRECISION', '17': 'PRECISION', '160': 'PRECISION'} | HELD |
| nf4 | nf4.gemv_scalar[dotpad=0] -> nf4.gemv_scalar[dotpad=0] | REORDER | {'16': 'REORDER', '17': 'REORDER', '160': 'REORDER'} | HELD |

## P2 — module replay (same bytes in)
| stack | sub-arm | module | predicted | observed | modules exact | verdict |
|---|---|---|---|---|---|---|
| int4 | hf.default | experts | {'16': 'NOT', '17': 'NOT', '160': 'NOT'} | {'16': 'NOT', '17': 'NOT', '160': 'NOT'} | {'16': '0/48 exact', '17': '0/48 exact', '160': '0/48 exact'} | HELD |
| int4 | hf.device | experts | {'16': 'EXACT', '17': 'EXACT', '160': 'NOT'} | {'16': 'EXACT', '17': 'EXACT', '160': 'NOT'} | {'16': '48/48 exact', '17': '48/48 exact', '160': '0/48 exact'} | HELD |
| int4 | hf.singleton | experts | {'16': 'EXACT', '17': 'EXACT', '160': 'EXACT'} | {'16': 'EXACT', '17': 'EXACT', '160': 'EXACT'} | {'16': '48/48 exact', '17': '48/48 exact', '160': '48/48 exact'} | HELD |
| int4 | hf.combine0 | experts | {'16': 'NOT', '17': 'NOT', '160': 'NOT'} | {'16': 'NOT', '17': 'NOT', '160': 'NOT'} | {'16': '0/48 exact', '17': '0/48 exact', '160': '0/48 exact'} | HELD |
| int4 | paged.default | experts | {'16': 'NOT', '17': 'NOT', '160': 'NOT'} | {'16': 'NOT', '17': 'NOT', '160': 'NOT'} | {'16': '0/48 exact', '17': '0/48 exact', '160': '0/48 exact'} | HELD |
| int4 | paged.device | experts | {'16': 'EXACT', '17': 'EXACT', '160': 'NOT'} | {'16': 'EXACT', '17': 'EXACT', '160': 'NOT'} | {'16': '48/48 exact', '17': '48/48 exact', '160': '0/48 exact'} | HELD |
| int4 | hf.default | attn | {'16': 'NOT', '17': 'NOT', '160': 'NOT'} | {'16': 'NOT', '17': 'NOT', '160': 'NOT'} | {'16': '0/96 exact', '17': '0/96 exact', '160': '0/96 exact'} | HELD |
| int4 | hf.default | lm_head | {'16': 'NOT', '17': 'NOT', '160': 'NOT'} | {'16': 'NOT', '17': 'NOT', '160': 'NOT'} | {'16': '0/1 exact', '17': '0/1 exact', '160': '0/1 exact'} | HELD |
| int4 | hf.default | router | {'16': 'NOT', '17': 'NOT', '160': 'NOT'} | {'16': 'NOT', '17': 'NOT', '160': 'NOT'} | {'16': '0/48 exact', '17': '0/48 exact', '160': '0/48 exact'} | HELD |
| int4 | hf.default | input_layernorm | {'16': 'EXACT', '17': 'EXACT', '160': 'NOT'} | {'16': 'EXACT', '17': 'EXACT', '160': 'NOT'} | {'16': '48/48 exact', '17': '48/48 exact', '160': '0/48 exact'} | HELD |
| int4nf | hf.default | experts | {'16': 'NOT', '17': 'NOT', '160': 'NOT'} | {'16': 'NOT', '17': 'NOT', '160': 'NOT'} | {'16': '0/48 exact', '17': '0/48 exact', '160': '0/48 exact'} | HELD |
| int4nf | hf.device | experts | {'16': 'EXACT', '17': 'EXACT', '160': 'NOT'} | {'16': 'EXACT', '17': 'EXACT', '160': 'NOT'} | {'16': '48/48 exact', '17': '48/48 exact', '160': '0/48 exact'} | HELD |
| int4nf | paged.device | experts | {'16': 'EXACT', '17': 'EXACT', '160': 'NOT'} | {'16': 'EXACT', '17': 'EXACT', '160': 'NOT'} | {'16': '48/48 exact', '17': '48/48 exact', '160': '0/48 exact'} | HELD |
| int4nf | hf.default | attn | {'16': 'NOT', '17': 'NOT', '160': 'NOT'} | {'16': 'NOT', '17': 'NOT', '160': 'NOT'} | {'16': '0/96 exact', '17': '0/96 exact', '160': '0/96 exact'} | HELD |
| int4nf | hf.default | lm_head | {'16': 'NOT', '17': 'NOT', '160': 'NOT'} | {'16': 'NOT', '17': 'NOT', '160': 'NOT'} | {'16': '0/1 exact', '17': '0/1 exact', '160': '0/1 exact'} | HELD |
| nf4 | hf.default | experts | {'16': 'NOT', '17': 'NOT', '160': 'NOT'} | {'16': 'NOT', '17': 'NOT', '160': 'NOT'} | {'16': '0/48 exact', '17': '0/48 exact', '160': '0/48 exact'} | HELD |
| nf4 | hf.singleton | experts | {'16': 'EXACT', '17': 'EXACT', '160': 'EXACT'} | {'16': 'EXACT', '17': 'EXACT', '160': 'EXACT'} | {'16': '48/48 exact', '17': '48/48 exact', '160': '48/48 exact'} | HELD |
| nf4 | hf.device | experts | {'16': 'NOT', '17': 'NOT', '160': 'NOT'} | {'16': 'NOT', '17': 'NOT', '160': 'NOT'} | {'16': '0/48 exact', '17': '0/48 exact', '160': '0/48 exact'} | HELD |
| nf4 | hf.combine0 | experts | {'16': 'NOT', '17': 'NOT', '160': 'NOT'} | {'16': 'NOT', '17': 'NOT', '160': 'NOT'} | {'16': '0/48 exact', '17': '0/48 exact', '160': '0/48 exact'} | HELD |
| nf4 | hf.singleton.dotpad0 | experts | {'16': 'NOT', '17': 'NOT', '160': 'NOT'} | {'16': 'NOT', '17': 'NOT', '160': 'NOT'} | {'16': '0/48 exact', '17': '0/48 exact', '160': '0/48 exact'} | HELD |
| nf4 | paged.default | experts | {'16': 'NOT', '17': 'NOT', '160': 'NOT'} | {'16': 'NOT', '17': 'NOT', '160': 'NOT'} | {'16': '0/48 exact', '17': '0/48 exact', '160': '0/48 exact'} | HELD |
| nf4 | hf.default | attn | {'16': 'NOT', '17': 'NOT', '160': 'NOT'} | {'16': 'NOT', '17': 'NOT', '160': 'NOT'} | {'16': '0/192 exact', '17': '0/192 exact', '160': '0/192 exact'} | HELD |
| nf4 | hf.default | lm_head | {'16': 'NOT', '17': 'NOT', '160': 'NOT'} | {'16': 'NOT', '17': 'NOT', '160': 'NOT'} | {'16': '0/1 exact', '17': '0/1 exact', '160': '0/1 exact'} | HELD |
| nf4 | hf.default | router | {'16': 'NOT', '17': 'NOT', '160': 'NOT'} | {'16': 'NOT', '17': 'NOT', '160': 'NOT'} | {'16': '0/48 exact', '17': '0/48 exact', '160': '0/48 exact'} | HELD |

## P3 — combine census
- nf4 combine_rows: EXACT (predicted EXACT), max bound ratio 0.995, vs chain [True, True, True, False, False] — **HELD**
- nf4 chain: EXACT (predicted None), max bound ratio 0.995, vs chain [] — **INFO**
- int4 combine_rows: EXACT (predicted EXACT), max bound ratio 0.995, vs chain [True, True, True, False, False] — **HELD**
- int4 chain: EXACT (predicted None), max bound ratio 0.995, vs chain [] — **INFO**
- int4nf combine_rows: EXACT (predicted EXACT), max bound ratio 0.995, vs chain [True, True, True, False, False] — **HELD**
- int4nf chain: EXACT (predicted None), max bound ratio 0.995, vs chain [] — **INFO**

## P4 / P5 — T = 1 controls across sub-arms
- nf4 hf.default vs hf.singleton (T=1 controls): EXACT (predicted EXACT); flips 0, KL mean 0.00e+00, first non-equal layer None — **HELD**
- nf4 hf.default vs hf.device (T=1 controls): EXACT (predicted EXACT); flips 0, KL mean 0.00e+00, first non-equal layer None — **HELD**
- nf4 hf.default vs hf.combine0 (T=1 controls): NOT (predicted NOT); flips 0, KL mean 1.18e-04, first non-equal layer 40 — **HELD**
- nf4 hf.default vs hf.singleton.dotpad0 (T=1 controls): NOT (predicted NOT); flips 17, KL mean 2.12e-02, first non-equal layer 0 — **HELD**
- nf4 hf.default vs paged.default (T=1 controls): NOT (predicted NOT); flips 12, KL mean 2.61e-02, first non-equal layer 0 — **HELD**
- int4 hf.default vs hf.device (T=1 controls): EXACT (predicted EXACT); flips 0, KL mean 0.00e+00, first non-equal layer None — **HELD**
- int4 hf.default vs hf.singleton (T=1 controls): EXACT (predicted EXACT); flips 0, KL mean 0.00e+00, first non-equal layer None — **HELD**
- int4 hf.default vs hf.combine0 (T=1 controls): NOT (predicted NOT); flips 7, KL mean 1.70e-02, first non-equal layer 0 — **HELD**
- int4 hf.default vs paged.default (T=1 controls): NOT (predicted NOT); flips 8, KL mean 3.30e-02, first non-equal layer 0 — **HELD**
- int4 hf.default vs paged.device (T=1 controls): NOT (predicted NOT); flips 8, KL mean 3.30e-02, first non-equal layer 0 — **HELD**
- int4nf hf.default vs hf.device (T=1 controls): EXACT (predicted EXACT); flips 0, KL mean 0.00e+00, first non-equal layer None — **HELD**
- int4nf hf.default vs paged.device (T=1 controls): NOT (predicted NOT); flips 11, KL mean 2.73e-02, first non-equal layer 0 — **HELD**
- **P5** nf4 combine0 vs default: KL mean 1.18e-04 (max 2.02e-03), flip fraction 0.000 — **HELD**
- **P5** int4 combine0 vs default: KL mean 1.70e-02 (max 2.20e-01), flip fraction 0.044 — **BETWEEN**

## P7 — router output dtype by row count
- int4: dtype differs by n {'16': False, '17': False, '160': True} (predicted {'16': False, '17': False, '160': True}) — **HELD**
- int4nf: dtype differs by n {'16': False, '17': False, '160': False} (predicted {'16': False, '17': False, '160': False}) — **HELD**
- nf4: dtype differs by n {'16': False, '17': False, '160': False} (predicted {'16': False, '17': False, '160': False}) — **HELD**

## D — accuracy against fp64 (the DEFECT line): NONE over 179 records
- 26 cuBLAS record(s) exceed the bound only under torch's default bf16 reduced-precision reduction (inside it with fp32 split-K reduction); max default-flag ratio 3.341

## P6 — end to end (control vs verify / prefill)
| stack | sub-arm | mode | exact / positions | first diff layer | first-diff sites | flips | top-1 | KL mean | KL max | size | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nf4 | hf.default | verify17 | 0/68 | 0 | {'attn_core': 68} | 3 | 0.9559 | 9.19e-03 | 1.20e-01 | IN-BAND | HELD |
| nf4 | hf.default | verify16 | 0/64 | 0 | {'attn_core': 64} | 5 | 0.9219 | 2.66e-02 | 9.44e-01 | OVER-BAR | HELD |
| nf4 | hf.default | prefill | 0/160 | 0 | {'attn_core': 160} | 15 | 0.9062 | 2.27e-02 | 2.90e-01 | OVER-BAR | HELD |
| nf4 | hf.singleton | verify17 | 0/68 | 0 | {'attn_core': 68} | 4 | 0.9412 | 1.64e-02 | 2.55e-01 | BETWEEN | HELD |
| nf4 | hf.singleton | verify16 | 0/64 | 0 | {'attn_core': 64} | 3 | 0.9531 | 2.89e-02 | 8.91e-01 | BETWEEN | HELD |
| nf4 | hf.singleton | prefill | 0/160 | 0 | {'attn_core': 160} | 11 | 0.9313 | 1.85e-02 | 2.45e-01 | BETWEEN | HELD |
| nf4 | hf.device | verify17 | 0/68 | 0 | {'attn_core': 68} | 3 | 0.9559 | 9.19e-03 | 1.20e-01 | IN-BAND | HELD |
| nf4 | hf.device | verify16 | 0/64 | 0 | {'attn_core': 64} | 5 | 0.9219 | 2.66e-02 | 9.44e-01 | OVER-BAR | HELD |
| nf4 | hf.device | prefill | 0/160 | 0 | {'attn_core': 160} | 15 | 0.9062 | 2.27e-02 | 2.90e-01 | OVER-BAR | HELD |
| nf4 | hf.combine0 | verify17 | 0/68 | 0 | {'attn_core': 68} | 4 | 0.9412 | 9.07e-03 | 1.20e-01 | BETWEEN | HELD |
| nf4 | hf.combine0 | verify16 | 0/64 | 0 | {'attn_core': 64} | 5 | 0.9219 | 2.63e-02 | 9.44e-01 | OVER-BAR | HELD |
| nf4 | hf.combine0 | prefill | 0/160 | 0 | {'attn_core': 160} | 16 | 0.9 | 2.35e-02 | 2.90e-01 | OVER-BAR | HELD |
| nf4 | hf.singleton.dotpad0 | verify17 | 0/68 | 0 | {'attn_core': 68} | 5 | 0.9265 | 9.33e-03 | 1.14e-01 | OVER-BAR | HELD |
| nf4 | hf.singleton.dotpad0 | verify16 | 0/64 | 0 | {'attn_core': 64} | 7 | 0.8906 | 1.58e-02 | 2.69e-01 | OVER-BAR | HELD |
| nf4 | hf.singleton.dotpad0 | prefill | 0/160 | 0 | {'attn_core': 160} | 10 | 0.9375 | 2.07e-02 | 3.27e-01 | BETWEEN | HELD |
| nf4 | paged.default | verify17 | 0/68 | 0 | {'attn_core': 68} | 2 | 0.9706 | 2.17e-02 | 1.98e-01 | BETWEEN | HELD |
| nf4 | paged.default | verify16 | 0/64 | 0 | {'attn_core': 64} | 1 | 0.9844 | 2.19e-02 | 5.60e-01 | BETWEEN | HELD |
| nf4 | paged.default | prefill | 0/160 | 0 | {'attn_core': 160} | 15 | 0.9062 | 3.23e-02 | 6.66e-01 | OVER-BAR | HELD |
| int4 | hf.default | verify17 | 0/68 | 0 | {'attn_core': 68} | 5 | 0.9265 | 1.56e-02 | 2.18e-01 | OVER-BAR | HELD |
| int4 | hf.default | verify16 | 0/64 | 0 | {'attn_core': 64} | 4 | 0.9375 | 1.35e-02 | 2.41e-01 | BETWEEN | HELD |
| int4 | hf.default | prefill | 0/160 | 0 | {'attn_in': 160} | 9 | 0.9437 | 1.91e-02 | 2.64e-01 | BETWEEN | HELD |
| int4 | hf.device | verify17 | 0/68 | 0 | {'attn_core': 68} | 6 | 0.9118 | 1.07e-02 | 1.28e-01 | OVER-BAR | HELD |
| int4 | hf.device | verify16 | 0/64 | 0 | {'attn_core': 64} | 4 | 0.9375 | 1.38e-02 | 1.78e-01 | BETWEEN | HELD |
| int4 | hf.device | prefill | 0/160 | 0 | {'attn_in': 160} | 5 | 0.9688 | 1.55e-02 | 1.92e-01 | IN-BAND | HELD |
| int4 | hf.singleton | verify17 | 0/68 | 0 | {'attn_core': 68} | 6 | 0.9118 | 1.07e-02 | 1.28e-01 | OVER-BAR | HELD |
| int4 | hf.singleton | verify16 | 0/64 | 0 | {'attn_core': 64} | 4 | 0.9375 | 1.38e-02 | 1.78e-01 | BETWEEN | HELD |
| int4 | hf.singleton | prefill | 0/160 | 0 | {'attn_in': 160} | 8 | 0.95 | 1.57e-02 | 2.00e-01 | IN-BAND | HELD |
| int4 | hf.combine0 | verify17 | 0/68 | 0 | {'attn_core': 68} | 3 | 0.9559 | 7.74e-03 | 7.21e-02 | IN-BAND | HELD |
| int4 | hf.combine0 | verify16 | 0/64 | 0 | {'attn_core': 64} | 5 | 0.9219 | 8.42e-03 | 8.73e-02 | OVER-BAR | HELD |
| int4 | hf.combine0 | prefill | 0/160 | 0 | {'attn_in': 160} | 9 | 0.9437 | 2.23e-02 | 3.12e-01 | BETWEEN | HELD |
| int4 | paged.default | verify17 | 0/68 | 0 | {'attn_core': 68} | 3 | 0.9559 | 1.63e-02 | 1.08e-01 | IN-BAND | HELD |
| int4 | paged.default | verify16 | 0/64 | 0 | {'attn_core': 64} | 3 | 0.9531 | 1.94e-02 | 3.52e-01 | IN-BAND | HELD |
| int4 | paged.default | prefill | 0/160 | 0 | {'attn_in': 160} | 9 | 0.9437 | 2.68e-02 | 2.15e-01 | BETWEEN | HELD |
| int4 | paged.device | verify17 | 0/68 | 0 | {'attn_core': 68} | 4 | 0.9412 | 1.76e-02 | 1.80e-01 | BETWEEN | HELD |
| int4 | paged.device | verify16 | 0/64 | 0 | {'attn_core': 64} | 3 | 0.9531 | 2.90e-02 | 4.87e-01 | BETWEEN | HELD |
| int4 | paged.device | prefill | 0/160 | 0 | {'attn_in': 160} | 7 | 0.9563 | 3.30e-02 | 3.81e-01 | BETWEEN | HELD |
| int4nf | hf.default | verify17 | 0/68 | 0 | {'attn_core': 68} | 4 | 0.9412 | 1.06e-02 | 1.01e-01 | BETWEEN | HELD |
| int4nf | hf.default | verify16 | 0/64 | 0 | {'attn_core': 64} | 4 | 0.9375 | 1.03e-02 | 1.42e-01 | BETWEEN | HELD |
| int4nf | hf.default | prefill | 0/160 | 0 | {'attn_core': 160} | 9 | 0.9437 | 1.52e-02 | 1.61e-01 | BETWEEN | HELD |
| int4nf | hf.device | verify17 | 0/68 | 0 | {'attn_core': 68} | 4 | 0.9412 | 1.01e-02 | 9.99e-02 | BETWEEN | HELD |
| int4nf | hf.device | verify16 | 0/64 | 0 | {'attn_core': 64} | 5 | 0.9219 | 1.06e-02 | 8.28e-02 | OVER-BAR | HELD |
| int4nf | hf.device | prefill | 0/160 | 0 | {'attn_core': 160} | 10 | 0.9375 | 1.46e-02 | 1.51e-01 | BETWEEN | HELD |
| int4nf | paged.device | verify17 | 0/68 | 0 | {'attn_core': 68} | 8 | 0.8824 | 1.65e-02 | 1.42e-01 | OVER-BAR | HELD |
| int4nf | paged.device | verify16 | 0/64 | 0 | {'attn_core': 64} | 5 | 0.9219 | 2.35e-02 | 2.77e-01 | OVER-BAR | HELD |
| int4nf | paged.device | prefill | 0/160 | 0 | {'attn_core': 160} | 11 | 0.9313 | 2.76e-02 | 3.72e-01 | BETWEEN | HELD |
