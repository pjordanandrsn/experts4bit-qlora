# P63 — generated results (p63_reduce.py; read against P63-PREREG.md)

Stacks: int4, int4nf, nf4; devices: {'nf4': 'NVIDIA RTX A2000 12GB', 'int4': 'NVIDIA RTX A2000 12GB', 'int4nf': 'NVIDIA RTX A2000 12GB'}

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
| nf4 | nf4.gemv_scalar -> nf4.gemv_scalar | EXACT | {'16': 'EXACT', '17': 'EXACT', '160': 'EXACT'} | HELD |
| nf4 | nf4.gemv_scalar -> nf4.mtile | PRECISION | {'16': 'PRECISION', '17': 'PRECISION', '160': 'PRECISION'} | HELD |
| nf4 | nf4.gemv_scalar[dotpad=0] -> nf4.gemv_scalar[dotpad=0] | REORDER | {'16': 'EXACT', '17': 'EXACT', '160': 'EXACT'} | REFUTED |

## P2 — module replay (same bytes in)
| stack | sub-arm | module | predicted | observed | modules exact | verdict |
|---|---|---|---|---|---|---|
| int4 | hf.default | experts | {'16': 'NOT', '17': 'NOT', '160': 'NOT'} | {'16': 'NOT', '17': 'NOT', '160': 'NOT'} | {'16': '0/16 exact', '17': '0/16 exact', '160': '0/16 exact'} | HELD |
| int4 | hf.device | experts | {'16': 'EXACT', '17': 'EXACT', '160': 'NOT'} | {'16': 'EXACT', '17': 'EXACT', '160': 'NOT'} | {'16': '16/16 exact', '17': '16/16 exact', '160': '0/16 exact'} | HELD |
| int4 | hf.singleton | experts | {'16': 'EXACT', '17': 'EXACT', '160': 'EXACT'} | {'16': 'EXACT', '17': 'EXACT', '160': 'EXACT'} | {'16': '16/16 exact', '17': '16/16 exact', '160': '16/16 exact'} | HELD |
| int4 | hf.combine0 | experts | {'16': 'NOT', '17': 'NOT', '160': 'NOT'} | {'16': 'NOT', '17': 'NOT', '160': 'NOT'} | {'16': '0/16 exact', '17': '0/16 exact', '160': '0/16 exact'} | HELD |
| int4 | paged.default | experts | {'16': 'NOT', '17': 'NOT', '160': 'NOT'} | {'16': 'NOT', '17': 'NOT', '160': 'NOT'} | {'16': '0/16 exact', '17': '0/16 exact', '160': '0/16 exact'} | HELD |
| int4 | paged.device | experts | {'16': 'EXACT', '17': 'EXACT', '160': 'NOT'} | {'16': 'EXACT', '17': 'EXACT', '160': 'NOT'} | {'16': '16/16 exact', '17': '16/16 exact', '160': '0/16 exact'} | HELD |
| int4 | hf.default | attn | {'16': 'NOT', '17': 'NOT', '160': 'NOT'} | {'16': 'NOT', '17': 'NOT', '160': 'NOT'} | {'16': '0/64 exact', '17': '0/64 exact', '160': '0/64 exact'} | HELD |
| int4 | hf.default | lm_head | {'16': 'NOT', '17': 'NOT', '160': 'NOT'} | {'16': 'NOT', '17': 'NOT', '160': 'NOT'} | {'16': '0/1 exact', '17': '0/1 exact', '160': '0/1 exact'} | HELD |
| int4 | hf.default | router | {'16': 'NOT', '17': 'NOT', '160': 'NOT'} | {'16': 'NOT', '17': 'NOT', '160': 'NOT'} | {'16': '0/16 exact', '17': '0/16 exact', '160': '0/16 exact'} | HELD |
| int4 | hf.default | input_layernorm | {'16': 'EXACT', '17': 'EXACT', '160': 'NOT'} | {'16': 'EXACT', '17': 'EXACT', '160': 'NOT'} | {'16': '16/16 exact', '17': '16/16 exact', '160': '0/16 exact'} | HELD |
| int4nf | hf.default | experts | {'16': 'NOT', '17': 'NOT', '160': 'NOT'} | {'16': 'NOT', '17': 'NOT', '160': 'NOT'} | {'16': '0/16 exact', '17': '0/16 exact', '160': '0/16 exact'} | HELD |
| int4nf | hf.device | experts | {'16': 'EXACT', '17': 'EXACT', '160': 'NOT'} | {'16': 'EXACT', '17': 'EXACT', '160': 'NOT'} | {'16': '16/16 exact', '17': '16/16 exact', '160': '0/16 exact'} | HELD |
| int4nf | paged.device | experts | {'16': 'EXACT', '17': 'EXACT', '160': 'NOT'} | {'16': 'EXACT', '17': 'EXACT', '160': 'NOT'} | {'16': '16/16 exact', '17': '16/16 exact', '160': '0/16 exact'} | HELD |
| int4nf | hf.default | attn | {'16': 'NOT', '17': 'NOT', '160': 'NOT'} | {'16': 'NOT', '17': 'NOT', '160': 'NOT'} | {'16': '0/64 exact', '17': '0/64 exact', '160': '0/64 exact'} | HELD |
| int4nf | hf.default | lm_head | {'16': 'NOT', '17': 'NOT', '160': 'NOT'} | {'16': 'NOT', '17': 'NOT', '160': 'NOT'} | {'16': '0/1 exact', '17': '0/1 exact', '160': '0/1 exact'} | HELD |
| nf4 | hf.default | experts | {'16': 'NOT', '17': 'NOT', '160': 'NOT'} | {'16': 'NOT', '17': 'NOT', '160': 'NOT'} | {'16': '0/16 exact', '17': '0/16 exact', '160': '0/16 exact'} | HELD |
| nf4 | hf.singleton | experts | {'16': 'EXACT', '17': 'EXACT', '160': 'EXACT'} | {'16': 'EXACT', '17': 'EXACT', '160': 'EXACT'} | {'16': '16/16 exact', '17': '16/16 exact', '160': '16/16 exact'} | HELD |
| nf4 | hf.device | experts | {'16': 'NOT', '17': 'NOT', '160': 'NOT'} | {'16': 'NOT', '17': 'NOT', '160': 'NOT'} | {'16': '0/16 exact', '17': '0/16 exact', '160': '0/16 exact'} | HELD |
| nf4 | hf.combine0 | experts | {'16': 'NOT', '17': 'NOT', '160': 'NOT'} | {'16': 'NOT', '17': 'NOT', '160': 'NOT'} | {'16': '0/16 exact', '17': '0/16 exact', '160': '0/16 exact'} | HELD |
| nf4 | hf.singleton.dotpad0 | experts | {'16': 'NOT', '17': 'NOT', '160': 'NOT'} | {'16': 'EXACT', '17': 'EXACT', '160': 'EXACT'} | {'16': '16/16 exact', '17': '16/16 exact', '160': '16/16 exact'} | REFUTED |
| nf4 | paged.default | experts | {'16': 'NOT', '17': 'NOT', '160': 'NOT'} | {'16': 'NOT', '17': 'NOT', '160': 'NOT'} | {'16': '0/16 exact', '17': '0/16 exact', '160': '0/16 exact'} | HELD |
| nf4 | hf.default | attn | {'16': 'NOT', '17': 'NOT', '160': 'NOT'} | {'16': 'NOT', '17': 'NOT', '160': 'NOT'} | {'16': '0/64 exact', '17': '0/64 exact', '160': '0/64 exact'} | HELD |
| nf4 | hf.default | lm_head | {'16': 'NOT', '17': 'NOT', '160': 'NOT'} | {'16': 'NOT', '17': 'NOT', '160': 'NOT'} | {'16': '0/1 exact', '17': '0/1 exact', '160': '0/1 exact'} | HELD |
| nf4 | hf.default | router | {'16': 'NOT', '17': 'NOT', '160': 'NOT'} | {'16': 'NOT', '17': 'NOT', '160': 'NOT'} | {'16': '0/16 exact', '17': '0/16 exact', '160': '0/16 exact'} | HELD |

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
- nf4 hf.default vs hf.combine0 (T=1 controls): NOT (predicted NOT); flips 1, KL mean 4.57e-04, first non-equal layer 0 — **HELD**
- nf4 hf.default vs hf.singleton.dotpad0 (T=1 controls): EXACT (predicted NOT); flips 0, KL mean 0.00e+00, first non-equal layer None — **REFUTED**
- nf4 hf.default vs paged.default (T=1 controls): NOT (predicted NOT); flips 3, KL mean 3.33e-03, first non-equal layer 0 — **HELD**
- int4 hf.default vs hf.device (T=1 controls): EXACT (predicted EXACT); flips 0, KL mean 0.00e+00, first non-equal layer None — **HELD**
- int4 hf.default vs hf.singleton (T=1 controls): EXACT (predicted EXACT); flips 0, KL mean 0.00e+00, first non-equal layer None — **HELD**
- int4 hf.default vs hf.combine0 (T=1 controls): NOT (predicted NOT); flips 1, KL mean 1.21e-03, first non-equal layer 0 — **HELD**
- int4 hf.default vs paged.default (T=1 controls): NOT (predicted NOT); flips 0, KL mean 4.53e-03, first non-equal layer 0 — **HELD**
- int4 hf.default vs paged.device (T=1 controls): NOT (predicted NOT); flips 0, KL mean 4.53e-03, first non-equal layer 0 — **HELD**
- int4nf hf.default vs hf.device (T=1 controls): EXACT (predicted EXACT); flips 0, KL mean 0.00e+00, first non-equal layer None — **HELD**
- int4nf hf.default vs paged.device (T=1 controls): NOT (predicted NOT); flips 3, KL mean 4.41e-03, first non-equal layer 0 — **HELD**
- **P5** nf4 combine0 vs default: KL mean 4.57e-04 (max 5.15e-03), flip fraction 0.006 — **HELD**
- **P5** int4 combine0 vs default: KL mean 1.21e-03 (max 1.46e-02), flip fraction 0.006 — **HELD**

## P7 — router output dtype by row count
- int4: dtype differs by n {'16': False, '17': False, '160': True} (predicted {'16': False, '17': False, '160': True}) — **HELD**
- int4nf: dtype differs by n {'16': False, '17': False, '160': False} (predicted {'16': False, '17': False, '160': False}) — **HELD**
- nf4: dtype differs by n {'16': False, '17': False, '160': False} (predicted {'16': False, '17': False, '160': False}) — **HELD**

## D — accuracy against fp64 (the DEFECT line): NONE over 227 records
- 35 cuBLAS record(s) exceed the bound only under torch's default bf16 reduced-precision reduction (inside it with fp32 split-K reduction); max default-flag ratio 7.352

## P6 — end to end (control vs verify / prefill)
| stack | sub-arm | mode | exact / positions | first diff layer | first-diff sites | flips | top-1 | KL mean | KL max | size | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nf4 | hf.default | verify17 | 0/68 | 0 | {'attn_core': 68} | 1 | 0.9853 | 9.44e-04 | 6.71e-03 | IN-BAND | HELD |
| nf4 | hf.default | verify16 | 0/64 | 0 | {'attn_core': 64} | 0 | 1.0 | 5.20e-04 | 3.99e-03 | IN-BAND | HELD |
| nf4 | hf.default | prefill | 0/160 | 0 | {'attn_core': 160} | 4 | 0.975 | 1.29e-03 | 7.98e-03 | IN-BAND | HELD |
| nf4 | hf.singleton | verify17 | 0/68 | 0 | {'attn_core': 68} | 0 | 1.0 | 5.77e-04 | 6.44e-03 | IN-BAND | HELD |
| nf4 | hf.singleton | verify16 | 0/64 | 0 | {'attn_core': 64} | 0 | 1.0 | 5.75e-04 | 4.24e-03 | IN-BAND | HELD |
| nf4 | hf.singleton | prefill | 0/160 | 0 | {'attn_core': 160} | 2 | 0.9875 | 1.42e-03 | 1.93e-02 | IN-BAND | HELD |
| nf4 | hf.device | verify17 | 0/68 | 0 | {'attn_core': 68} | 1 | 0.9853 | 9.44e-04 | 6.71e-03 | IN-BAND | HELD |
| nf4 | hf.device | verify16 | 0/64 | 0 | {'attn_core': 64} | 0 | 1.0 | 5.20e-04 | 3.99e-03 | IN-BAND | HELD |
| nf4 | hf.device | prefill | 0/160 | 0 | {'attn_core': 160} | 4 | 0.975 | 1.29e-03 | 7.98e-03 | IN-BAND | HELD |
| nf4 | hf.combine0 | verify17 | 0/68 | 0 | {'attn_core': 68} | 1 | 0.9853 | 1.19e-03 | 1.61e-02 | IN-BAND | HELD |
| nf4 | hf.combine0 | verify16 | 0/64 | 0 | {'attn_core': 64} | 0 | 1.0 | 7.10e-04 | 4.87e-03 | IN-BAND | HELD |
| nf4 | hf.combine0 | prefill | 0/160 | 0 | {'attn_core': 160} | 3 | 0.9812 | 1.29e-03 | 8.46e-03 | IN-BAND | HELD |
| nf4 | hf.singleton.dotpad0 | verify17 | 0/68 | 0 | {'attn_core': 68} | 0 | 1.0 | 5.77e-04 | 6.44e-03 | IN-BAND | HELD |
| nf4 | hf.singleton.dotpad0 | verify16 | 0/64 | 0 | {'attn_core': 64} | 0 | 1.0 | 5.75e-04 | 4.24e-03 | IN-BAND | HELD |
| nf4 | hf.singleton.dotpad0 | prefill | 0/160 | 0 | {'attn_core': 160} | 2 | 0.9875 | 1.42e-03 | 1.93e-02 | IN-BAND | HELD |
| nf4 | paged.default | verify17 | 0/68 | 0 | {'attn_core': 68} | 2 | 0.9706 | 1.29e-03 | 1.65e-02 | IN-BAND | HELD |
| nf4 | paged.default | verify16 | 0/64 | 0 | {'attn_core': 64} | 0 | 1.0 | 1.07e-03 | 6.27e-03 | IN-BAND | HELD |
| nf4 | paged.default | prefill | 0/160 | 0 | {'attn_core': 160} | 4 | 0.975 | 3.59e-03 | 2.21e-02 | IN-BAND | HELD |
| int4 | hf.default | verify17 | 0/68 | 0 | {'attn_core': 68} | 0 | 1.0 | 2.10e-03 | 3.26e-02 | IN-BAND | HELD |
| int4 | hf.default | verify16 | 0/64 | 0 | {'attn_core': 64} | 0 | 1.0 | 1.20e-03 | 9.39e-03 | IN-BAND | HELD |
| int4 | hf.default | prefill | 0/160 | 0 | {'attn_in': 160} | 5 | 0.9688 | 2.59e-03 | 3.43e-02 | IN-BAND | HELD |
| int4 | hf.device | verify17 | 0/68 | 0 | {'attn_core': 68} | 1 | 0.9853 | 1.81e-03 | 2.15e-02 | IN-BAND | HELD |
| int4 | hf.device | verify16 | 0/64 | 0 | {'attn_core': 64} | 0 | 1.0 | 1.63e-03 | 1.49e-02 | IN-BAND | HELD |
| int4 | hf.device | prefill | 0/160 | 0 | {'attn_in': 160} | 3 | 0.9812 | 2.41e-03 | 2.72e-02 | IN-BAND | HELD |
| int4 | hf.singleton | verify17 | 0/68 | 0 | {'attn_core': 68} | 1 | 0.9853 | 1.81e-03 | 2.15e-02 | IN-BAND | HELD |
| int4 | hf.singleton | verify16 | 0/64 | 0 | {'attn_core': 64} | 0 | 1.0 | 1.63e-03 | 1.49e-02 | IN-BAND | HELD |
| int4 | hf.singleton | prefill | 0/160 | 0 | {'attn_in': 160} | 2 | 0.9875 | 2.60e-03 | 3.53e-02 | IN-BAND | HELD |
| int4 | hf.combine0 | verify17 | 0/68 | 0 | {'attn_core': 68} | 0 | 1.0 | 2.78e-03 | 6.34e-02 | IN-BAND | HELD |
| int4 | hf.combine0 | verify16 | 0/64 | 0 | {'attn_core': 64} | 0 | 1.0 | 1.70e-03 | 1.07e-02 | IN-BAND | HELD |
| int4 | hf.combine0 | prefill | 0/160 | 0 | {'attn_in': 160} | 5 | 0.9688 | 2.57e-03 | 3.37e-02 | IN-BAND | HELD |
| int4 | paged.default | verify17 | 0/68 | 0 | {'attn_core': 68} | 2 | 0.9706 | 2.53e-03 | 2.28e-02 | IN-BAND | HELD |
| int4 | paged.default | verify16 | 0/64 | 0 | {'attn_core': 64} | 0 | 1.0 | 2.07e-03 | 2.74e-02 | IN-BAND | HELD |
| int4 | paged.default | prefill | 0/160 | 0 | {'attn_in': 160} | 5 | 0.9688 | 4.33e-03 | 3.39e-02 | IN-BAND | HELD |
| int4 | paged.device | verify17 | 0/68 | 0 | {'attn_core': 68} | 0 | 1.0 | 3.83e-03 | 2.77e-02 | IN-BAND | HELD |
| int4 | paged.device | verify16 | 0/64 | 0 | {'attn_core': 64} | 2 | 0.9688 | 2.07e-03 | 1.62e-02 | IN-BAND | HELD |
| int4 | paged.device | prefill | 0/160 | 0 | {'attn_in': 160} | 2 | 0.9875 | 4.03e-03 | 3.11e-02 | IN-BAND | HELD |
| int4nf | hf.default | verify17 | 0/68 | 0 | {'attn_core': 68} | 0 | 1.0 | 1.64e-03 | 1.19e-02 | IN-BAND | HELD |
| int4nf | hf.default | verify16 | 0/64 | 0 | {'attn_core': 64} | 0 | 1.0 | 1.39e-03 | 6.43e-03 | IN-BAND | HELD |
| int4nf | hf.default | prefill | 0/160 | 0 | {'attn_core': 160} | 4 | 0.975 | 2.19e-03 | 2.16e-02 | IN-BAND | HELD |
| int4nf | hf.device | verify17 | 0/68 | 0 | {'attn_core': 68} | 0 | 1.0 | 1.42e-03 | 9.55e-03 | IN-BAND | HELD |
| int4nf | hf.device | verify16 | 0/64 | 0 | {'attn_core': 64} | 0 | 1.0 | 1.55e-03 | 1.43e-02 | IN-BAND | HELD |
| int4nf | hf.device | prefill | 0/160 | 0 | {'attn_core': 160} | 4 | 0.975 | 2.53e-03 | 2.49e-02 | IN-BAND | HELD |
| int4nf | paged.device | verify17 | 0/68 | 0 | {'attn_core': 68} | 1 | 0.9853 | 2.87e-03 | 2.85e-02 | IN-BAND | HELD |
| int4nf | paged.device | verify16 | 0/64 | 0 | {'attn_core': 64} | 0 | 1.0 | 1.96e-03 | 1.19e-02 | IN-BAND | HELD |
| int4nf | paged.device | prefill | 0/160 | 0 | {'attn_core': 160} | 4 | 0.975 | 3.88e-03 | 1.95e-02 | IN-BAND | HELD |

