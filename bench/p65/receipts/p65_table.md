| family | signal | r_split | r_cross_half [95 % CI] | penalty | r_cross_full | Colla-Q cosine | top-10 % Jaccard (chance) | survives |
|---|---|---|---|---|---|---|---|---|
| granite | entropy | 0.968 | 0.882 [0.860, 0.902] | 0.086 | 0.894 | 0.9995 | 0.546 (0.053) | yes |
| granite | rel_act | 0.974 | 0.932 [0.916, 0.945] | 0.043 | 0.940 | 0.9999 | 0.750 (0.053) | yes |
| granite | freq | 0.984 | 0.891 [0.863, 0.914] | 0.093 | 0.895 | 0.9795 | 0.680 (0.053) | yes |
| granite | rho_x_err | 0.974 | 0.910 [0.892, 0.928] | 0.063 | 0.918 | 0.9995 | 0.787 (0.053) | yes |
| olmoe | entropy | 0.964 | 0.800 [0.764, 0.835] | 0.164 | 0.794 | 0.9997 | 0.569 (0.052) | no |
| olmoe | rel_act | 0.982 | 0.931 [0.921, 0.941] | 0.051 | 0.936 | 0.9999 | 0.729 (0.052) | yes |
| olmoe | freq | 0.935 | 0.380 [0.310, 0.453] | 0.556 | 0.395 | 0.8940 | 0.146 (0.052) | no |
| olmoe | rho_x_err | 0.980 | 0.928 [0.914, 0.940] | 0.052 | 0.935 | 0.9992 | 0.700 (0.052) | yes |
| mixtral | entropy | 0.927 | 0.789 [0.708, 0.853] | 0.138 | 0.796 | 0.9999 | 0.733 (0.053) | yes |
| mixtral | rel_act | 0.878 | 0.719 [0.651, 0.791] | 0.159 | 0.747 | 1.0000 | 0.529 (0.053) | no |
| mixtral | freq | 0.757 | 0.310 [0.215, 0.403] | 0.446 | 0.290 | 0.9944 | 0.182 (0.053) | no |
| mixtral | rho_x_err | 0.918 | 0.724 [0.653, 0.795] | 0.194 | 0.757 | 0.9998 | 0.529 (0.053) | no |

Descriptive, read by no rule:

| family | signal | r_split | r_cross_half | penalty | rel_gu~rel_dn (wiki / c4) |
|---|---|---|---|---|---|
| granite | rel_gu | 0.993 | 0.971 | 0.022 | -0.436 / -0.452 |
| granite | rel_dn | 0.979 | 0.956 | 0.023 | -0.436 / -0.452 |
| granite | rho_in | 0.957 | 0.895 | 0.062 | -0.436 / -0.452 |
| granite | h_energy | 0.974 | 0.914 | 0.060 | -0.436 / -0.452 |
| olmoe | rel_gu | 0.990 | 0.946 | 0.044 | -0.127 / -0.086 |
| olmoe | rel_dn | 0.961 | 0.896 | 0.065 | -0.127 / -0.086 |
| olmoe | rho_in | 0.946 | 0.782 | 0.163 | -0.127 / -0.086 |
| olmoe | h_energy | 0.962 | 0.869 | 0.093 | -0.127 / -0.086 |
| mixtral | rel_gu | 0.958 | 0.855 | 0.103 | 0.037 / 0.036 |
| mixtral | rel_dn | 0.864 | 0.750 | 0.113 | 0.037 / 0.036 |
| mixtral | rho_in | 0.887 | 0.776 | 0.110 | 0.037 / 0.036 |
| mixtral | h_energy | 0.906 | 0.795 | 0.110 | 0.037 / 0.036 |

| family | entropy~rel_act (wiki / c4) | entropy redundant | Colla-Q claim (entropy − freq) | RTN tail, c4val1 (P3 statistic) | premise | selector |
|---|---|---|---|---|---|---|
| granite | -0.000 / -0.055 | no | UNRESOLVED (-0.009 [-0.044, 0.024]) | 0.0142 | holds (P44-a P3 0.0992 (recipe/GPTQ, 32/32 layers)) | TWO_ARMS |
| olmoe | 0.427 / 0.545 | no | REPLICATES (0.421 [0.329, 0.501]) | 0.2119 | no (P3 on this census (RTN, c4val1 full): fraction 0.2119140625) | NOT_WRITTEN (no per-expert premise) |
| mixtral | 0.326 / 0.332 | no | REPLICATES (0.479 [0.369, 0.588]) | 0.2344 | no (P44-a P3 0.2266 (RTN, 16/32 layers)) | NOT_WRITTEN (no per-expert premise) |
