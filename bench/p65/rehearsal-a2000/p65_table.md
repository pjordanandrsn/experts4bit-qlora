| family | signal | r_split | r_cross_half [95 % CI] | penalty | r_cross_full | Colla-Q cosine | top-10 % Jaccard (chance) | survives |
|---|---|---|---|---|---|---|---|---|
| granite | entropy | 0.977 | 0.895 [0.872, 0.913] | 0.082 | 0.903 | 0.9998 | 0.711 (0.053) | yes |
| granite | rel_act | 0.974 | 0.919 [0.898, 0.938] | 0.055 | 0.924 | 0.9999 | 0.730 (0.053) | yes |
| granite | freq | 0.988 | 0.893 [0.867, 0.917] | 0.095 | 0.898 | 0.9831 | 0.540 (0.053) | yes |
| granite | rho_x_err | 0.974 | 0.891 [0.857, 0.920] | 0.083 | 0.895 | 0.9994 | 0.621 (0.053) | yes |
| olmoe | entropy | 0.970 | 0.783 [0.715, 0.844] | 0.187 | 0.781 | 0.9998 | 0.576 (0.053) | no |
| olmoe | rel_act | 0.988 | 0.938 [0.912, 0.964] | 0.050 | 0.940 | 0.9999 | 0.576 (0.053) | yes |
| olmoe | freq | 0.940 | 0.454 [0.359, 0.549] | 0.487 | 0.469 | 0.9141 | 0.156 (0.053) | no |
| olmoe | rho_x_err | 0.986 | 0.938 [0.909, 0.960] | 0.047 | 0.944 | 0.9994 | 0.576 (0.053) | yes |
| mixtral | — | NOT_READ: no census file | | | | | | |

Descriptive, read by no rule:

| family | signal | r_split | r_cross_half | penalty | rel_gu~rel_dn (wiki / c4) |
|---|---|---|---|---|---|
| granite | rel_gu | 0.993 | 0.967 | 0.027 | -0.516 / -0.529 |
| granite | rel_dn | 0.980 | 0.954 | 0.026 | -0.516 / -0.529 |
| granite | rho_in | 0.965 | 0.852 | 0.112 | -0.516 / -0.529 |
| granite | h_energy | 0.979 | 0.933 | 0.046 | -0.516 / -0.529 |
| olmoe | rel_gu | 0.993 | 0.942 | 0.051 | -0.003 / 0.050 |
| olmoe | rel_dn | 0.974 | 0.897 | 0.077 | -0.003 / 0.050 |
| olmoe | rho_in | 0.966 | 0.744 | 0.222 | -0.003 / 0.050 |
| olmoe | h_energy | 0.962 | 0.891 | 0.071 | -0.003 / 0.050 |

| family | entropy~rel_act (wiki / c4) | entropy redundant | Colla-Q claim (entropy − freq) | RTN tail, c4val1 (P3 statistic) | premise | selector |
|---|---|---|---|---|---|---|
| granite | -0.039 / -0.043 | no | UNRESOLVED (0.002 [-0.033, 0.034]) | 0.1185 | holds (P44-a P3 0.0992 (recipe/GPTQ, 32/32 layers)) | TWO_ARMS |
| olmoe | 0.370 / 0.510 | no | REPLICATES (0.329 [0.177, 0.482]) | 0.2422 | no (P3 on this census (RTN, c4val1 full): fraction 0.2421875) | NOT_WRITTEN (no per-expert premise) |
