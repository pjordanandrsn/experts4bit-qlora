# P129 Amendment 1, re-derived: **PASS**

| measure | fused, worst of 5 | floor, worst of 15 |
|---|---|---|
| `D_traj` | 0.00311 | 0.00361 |
| `D_held` | 0.00064 | 0.00078 |

- end-to-end gate: HELD (backstop: every fused `D_held` <= 0.005)
- counts: launches 785 -> 675 (-110), Python calls 16440 -> 14466 (-12.01 %): HELD; dequant bitwise: True

| draw | `D_traj` | `D_held` |
|---|---|---|
| F1 211 | 0.00182 | 0.00040 |
| F2 211 | 0.00210 | 0.00025 |
| F3 211 | 0.00223 | 0.00009 |
| F1 223 | 0.00263 | 0.00021 |
| F2 223 | 0.00244 | 0.00013 |
| F3 223 | 0.00206 | 0.00008 |
| F1 227 | 0.00270 | 0.00014 |
| F2 227 | 0.00224 | 0.00019 |
| F3 227 | 0.00342 | 0.00011 |
| F1 229 | 0.00243 | 0.00033 |
| F2 229 | 0.00209 | 0.00033 |
| F3 229 | 0.00307 | 0.00002 |
| F1 233 | 0.00244 | 0.00076 |
| F2 233 | 0.00171 | 0.00024 |
| F3 233 | 0.00361 | 0.00078 |
| FUSED 211 | 0.00228 | 0.00050 |
| FUSED 223 | 0.00311 | 0.00048 |
| FUSED 227 | 0.00237 | 0.00041 |
| FUSED 229 | 0.00236 | 0.00041 |
| FUSED 233 | 0.00234 | 0.00064 |

Reported -- the fused projection in isolation (relative difference, bar, within):
- qkv out: 6.71e-03 (bar 3.64e-02): True
- d x: 1.06e-02 (bar 1.08e-03): True
- d q_proj.lora_A: 4.05e-07 (bar 9.54e-06): True
- d q_proj.lora_B: 3.95e-07 (bar 2.19e-05): True
- d k_proj.lora_A: 3.80e-07 (bar 3.59e-06): True
- d k_proj.lora_B: 3.58e-07 (bar 2.16e-05): True
- d v_proj.lora_A: 3.78e-07 (bar 3.91e-06): True
- d v_proj.lora_B: 3.55e-07 (bar 1.92e-05): True

Reported -- the first step's worst relative gradient difference from the baseline, per draw:
- F1 211: 0.0096 (model.layers.0.self_attn.k_proj.lora_B)
- F2 211: 0.0484 (model.layers.1.mlp.experts.down_lora_A)
- F3 211: 0.1489 (model.layers.0.mlp.experts.down_lora_A)
- FUSED 211: 0.1235 (model.layers.1.mlp.experts.down_lora_B)
- F1 223: 0.0093 (model.layers.0.self_attn.k_proj.lora_A)
- F2 223: 0.0091 (model.layers.0.self_attn.k_proj.lora_A)
- F3 223: 0.1285 (model.layers.0.mlp.experts.gate_up_lora_B)
- FUSED 223: 0.1099 (model.layers.0.mlp.experts.gate_up_lora_B)
- F1 227: 0.0085 (model.layers.0.self_attn.k_proj.lora_A)
- F2 227: 0.0573 (model.layers.1.mlp.experts.gate_up_lora_B)
- F3 227: 0.1427 (model.layers.0.mlp.experts.gate_up_lora_B)
- FUSED 227: 0.1407 (model.layers.0.mlp.experts.gate_up_lora_B)
- F1 229: 0.0099 (model.layers.0.self_attn.k_proj.lora_A)
- F2 229: 0.0498 (model.layers.1.mlp.experts.down_lora_A)
- F3 229: 0.1430 (model.layers.0.mlp.experts.gate_up_lora_B)
- FUSED 229: 0.2215 (model.layers.0.mlp.experts.gate_up_lora_B)
- F1 233: 0.0097 (model.layers.0.self_attn.q_proj.lora_B)
- F2 233: 0.0648 (model.layers.1.mlp.experts.down_lora_B)
- F3 233: 0.1494 (model.layers.0.mlp.experts.down_lora_B)
- FUSED 233: 0.1453 (model.layers.0.mlp.experts.down_lora_B)
