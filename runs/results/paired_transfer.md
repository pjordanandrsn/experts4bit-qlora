# Paired transfer statistics (same-adapter and seed-paired contrasts)

- source: query jobs, train rows ['int8-offload', 'int8-resident', 'nf4-offload', 'nf4-resident'], query columns ['int8-resident', 'nf4-resident'], seeds [1337, 2027, 3407]
- ruler: PAIRED deltas (marginal cell sds are the wrong ruler for a paired design)

## Same-adapter query-pair deltas, per seed

| train row | pair | per-seed deltas | mean | paired sd | t | same-sign |
|---|---|---|---|---|---|---|
| int8-offload | L(→int8-resident) − L(→nf4-resident) | -0.0022 / -0.0068 / -0.0091 | -0.0061 | 0.0035 | -3.00 | 0/3 |
| int8-resident | L(→int8-resident) − L(→nf4-resident) | -0.0069 / -0.0106 / -0.0050 | -0.0075 | 0.0028 | -4.55 | 0/3 |
| nf4-offload | L(→int8-resident) − L(→nf4-resident) | -0.0003 / -0.0002 / +0.0140 | +0.0045 | 0.0082 | 0.95 | 1/3 |
| nf4-resident | L(→int8-resident) − L(→nf4-resident) | -0.0021 / -0.0033 / +0.0047 | -0.0002 | 0.0044 | -0.10 | 1/3 |

## Seed-paired cross-adapter contrasts, per query column

| query col | contrast | per-seed deltas | mean | paired sd | t | same-sign |
|---|---|---|---|---|---|---|
| int8-resident | int8-offload − int8-resident | -0.0045 / -0.0044 / -0.0100 | -0.0063 | 0.0032 | -3.42 | 0/3 |
| int8-resident | int8-offload − nf4-offload | -0.0030 / +0.0030 / -0.0193 | -0.0064 | 0.0115 | -0.97 | 1/3 |
| int8-resident | int8-offload − nf4-resident | -0.0039 / -0.0069 / -0.0061 | -0.0056 | 0.0015 | -6.29 | 0/3 |
| int8-resident | int8-resident − nf4-offload | +0.0015 / +0.0074 / -0.0093 | -0.0001 | 0.0085 | -0.03 | 2/3 |
| int8-resident | int8-resident − nf4-resident | +0.0007 / -0.0025 / +0.0039 | +0.0007 | 0.0032 | 0.37 | 2/3 |
| int8-resident | nf4-offload − nf4-resident | -0.0008 / -0.0099 / +0.0132 | +0.0008 | 0.0117 | 0.12 | 1/3 |
| nf4-resident | int8-offload − int8-resident | -0.0092 / -0.0081 / -0.0058 | -0.0077 | 0.0017 | -7.83 | 0/3 |
| nf4-resident | int8-offload − nf4-offload | -0.0011 / +0.0097 / +0.0038 | +0.0041 | 0.0054 | 1.32 | 2/3 |
| nf4-resident | int8-offload − nf4-resident | -0.0038 / -0.0034 / +0.0078 | +0.0002 | 0.0066 | 0.06 | 1/3 |
| nf4-resident | int8-resident − nf4-offload | +0.0080 / +0.0179 / +0.0097 | +0.0119 | 0.0053 | 3.91 | 3/3 |
| nf4-resident | int8-resident − nf4-resident | +0.0054 / +0.0048 / +0.0137 | +0.0079 | 0.0050 | 2.77 | 3/3 |
| nf4-resident | nf4-offload − nf4-resident | -0.0026 / -0.0131 / +0.0040 | -0.0039 | 0.0086 | -0.79 | 1/3 |
