P67 re-read of existing receipts (bench/p67/P67-PREREG.md): TOL=0.05, K=3, N_MIN=3. TRAIN-loss quantities; held-out beside.

| family | fixture | session | judged arm | status | D_final train | D_med train | Δ held-out | constant 0.05 | floor band | carried by | admissible floor draws |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma4 | N=20 seq=2048 mb=2 acc=4 r=16 tok=4ea779a5 init=7588b481 | tp4-c-parity-2 | `fused_attn4` | VALID | 0.08257 | 0.12421 | 0.15985 | FAIL | **NO-FLOOR** | — | 0 |
| gemma4 | N=20 seq=2048 mb=2 acc=4 r=16 tok=4ea779a5 init=7588b481 | p56-gemma4-ladder-3 | `batched_attn4` | VALID | 0.05425 | 0.08487 | 0.08595 | FAIL | **NO-FLOOR** | — | 0 |
| gemma4 | N=20 seq=2048 mb=2 acc=4 r=16 tok=4ea779a5 init=7588b481 | p56-gemma4-ladder-3 | `fused_attn4` | VALID | 0.10223 | 0.10639 | 0.16378 | FAIL | **NO-FLOOR** | — | 0 |
| gemma4 | N=20 seq=2048 mb=2 acc=4 r=16 tok=4ea779a5 init=7588b481 | p56-gemma4-ladder-3 | `fused_attn4_nodgrad` | VALID | 0.10274 | 0.11678 | 0.18265 | FAIL | **NO-FLOOR** | — | 0 |
| gemma4 | N=20 seq=2048 mb=2 acc=4 r=16 tok=f758d35e init=7588b481 | tp4-c-4 | `fused_attn4` | VALID | 0.09037 | 0.11801 | 0.09828 | FAIL | **NO-FLOOR** | — | 0 |
| gemma4 | N=60 seq=512 mb=None acc=None r=8 tok=tp1-clin init=a059fa05 | tp1 (clinical, bf16 attention) | `batched_attn4` | VOID | — | — | — | — | — | — | 0 (kernel calls/step min 0 < 2*n_patched=60 (tp1: fallback took layers)) |
| gemma4 | N=60 seq=512 mb=None acc=None r=8 tok=tp1-clin init=a059fa05 | tp1 (clinical, bf16 attention) | `fused_attn4` | VALID | 0.02385 | 0.04742 | 0.00219 | PASS | **PASS** | tolerance | 0 |
| granite | N=20 seq=2048 mb=2 acc=4 r=16 tok=858f7190 init=87fd1d94 | tp4-b-p46cut-A | `fused_attn4` | VALID | 0.00063 | 0.00135 | 0.00020 | PASS | **PASS** | tolerance | 0 |
| granite | N=60 seq=2048 mb=2 acc=4 r=16 tok=858f7190 init=87fd1d94 | tp4-a | `fused_attn4` | VALID | 0.00030 | 0.00148 | 0.00030 | PASS | **PASS** | tolerance | 0 |
| granite | N=60 seq=2048 mb=2 acc=4 r=16 tok=858f7190 init=87fd1d94 | tp4-d | `fused_attn4` | VALID | 0.00273 | 0.00162 | 0.00089 | PASS | **PASS** | tolerance | 0 |
| granite | N=60 seq=512 mb=None acc=None r=8 tok=tp1-clin init=9da33dcf | tp1 (clinical, bf16 attention) | `batched_attn4` | VALID | 0.01553 | 0.01681 | 0.00551 | PASS | **PASS** | tolerance | 0 |
| granite | N=60 seq=512 mb=None acc=None r=8 tok=tp1-clin init=9da33dcf | tp1 (clinical, bf16 attention) | `fused_attn4` | VALID | 0.01329 | 0.01270 | 0.00495 | PASS | **PASS** | tolerance | 0 |
| mixtral | N=20 seq=2048 mb=2 acc=4 r=16 tok=4a41b3f4 init=e88e53c0 | tp4-c-4 | `fused_attn4` | VALID | 0.00403 | 0.00415 | 0.00087 | PASS | **PASS** | tolerance | 0 |
| mixtral | N=60 seq=512 mb=None acc=None r=8 tok=tp1-clin init=46ea74b4 | tp1 (clinical, bf16 attention) | `batched_attn4` | VALID | 0.00766 | 0.01057 | 0.00717 | PASS | **PASS** | tolerance | 0 |
| mixtral | N=60 seq=512 mb=None acc=None r=8 tok=tp1-clin init=46ea74b4 | tp1 (clinical, bf16 attention) | `fused_attn4` | VALID | 0.00953 | 0.00945 | 0.00068 | PASS | **PASS** | tolerance | 0 |
| olmoe | N=20 seq=2048 mb=2 acc=4 r=16 tok=de1b303b init=9180a93d | tp4-b-p46cut-A | `fused_attn4` | VALID | 0.00203 | 0.00117 | 0.00116 | PASS | **PASS** | tolerance | 0 |
| olmoe | N=60 seq=2048 mb=2 acc=4 r=16 tok=de1b303b init=9180a93d | tp4-a | `fused_attn4` | VALID | 0.00019 | 0.00125 | 0.00119 | PASS | **PASS** | tolerance | 0 |
| olmoe | N=60 seq=512 mb=None acc=None r=8 tok=tp1-clin init=b14ec247 | tp1 (clinical, bf16 attention) | `batched_attn4` | VOID | — | — | — | — | — | — | 0 (kernel calls/step min 24 < 2*n_patched=32 (tp1: fallback took layers)) |
| olmoe | N=60 seq=512 mb=None acc=None r=8 tok=tp1-clin init=b14ec247 | tp1 (clinical, bf16 attention) | `fused_attn4` | VALID | 0.01327 | 0.01249 | 0.00381 | PASS | **PASS** | tolerance | 0 |
| qwen3 | N=20 seq=2048 mb=2 acc=4 r=16 tok=6d5783ce init=0e2825d3 | p46-qwen3lora | `fused_attn4` | VALID | 0.00236 | 0.00177 | 0.00153 | PASS | **PASS** | tolerance | 0 |
| qwen3 | N=20 seq=2048 mb=2 acc=4 r=16 tok=6d5783ce init=0e2825d3 | tp4-b-p46cut-4 | `fused_attn4` | VALID | 0.00140 | 0.00142 | 0.00050 | PASS | **PASS** | tolerance | 0 |
| qwen3 | N=60 seq=512 mb=None acc=None r=8 tok=tp1-clin init=2b362f7b | tp1 (clinical, bf16 attention) | `batched_attn4` | VOID | — | — | — | — | — | — | 0 (kernel calls/step min 12 < 2*n_patched=96 (tp1: fallback took layers)) |
| qwen3 | N=60 seq=512 mb=None acc=None r=8 tok=tp1-clin init=2b362f7b | tp1 (clinical, bf16 attention) | `fused_attn4` | VALID | 0.01315 | 0.01050 | 0.00244 | PASS | **PASS** | tolerance | 0 |
| qwen3_5 | N=20 seq=2048 mb=2 acc=4 r=16 tok=20d39334 init=12802937 | tp4-b-p46cut-4 | `fused_attn4` | VALID | 0.00159 | 0.00154 | 0.00007 | PASS | **PASS** | tolerance | 0 |

Disclosed, never admitted (different sessions): the same arm on two boxes of the same fixture.

| family | arm | session a | session b | D_final train | D_med train | Δ held-out | Δ step-0 | first differing step |
|---|---|---|---|---|---|---|---|---|
| gemma4 | `fused_attn4` | tp4-b-p46cut-C2 | tp4-c-parity-2 | 0.06442 | 0.08156 | 0.11060 | 0.000000 | 2 |
| gemma4 | `fused_attn4` | tp4-b-p46cut-C2 | p56-gemma4-ladder-3 | 0.06268 | 0.05639 | 0.06319 | 0.000000 | 2 |
| gemma4 | `fused_attn4` | tp4-c-parity-2 | p56-gemma4-ladder-3 | 0.00174 | 0.04620 | 0.04741 | 0.000000 | 2 |
| gemma4 | `reference_attn4` | tp4-c-parity-2 | p56-gemma4-ladder-3 | 0.02140 | 0.02560 | 0.05134 | 0.000000 | 2 |
| granite | `fused_attn4` | tp4-a | tp4-d | 0.00186 | 0.00122 | 0.00159 | 0.000000 | 2 |
| granite | `reference_attn4` | tp4-a | tp4-d | 0.00057 | 0.00090 | 0.00100 | 0.000000 | 5 |
| qwen3 | `fused_attn4_mb1` | p43-t1-qwen3-2 | tp4-b-p46cut-4 | 0.00181 | 0.00084 | 0.00011 | 0.000000 | 2 |
| qwen3 | `fused_attn4` | p43-t1-qwen3-2 | p46-qwen3lora | 0.00174 | 0.00086 | 0.00018 | 0.000000 | 2 |
| qwen3 | `fused_attn4` | p43-t1-qwen3-2 | tp4-b-p46cut-4 | 0.00276 | 0.00119 | 0.00079 | 0.000000 | 2 |
| qwen3 | `fused_attn4` | p46-qwen3lora | tp4-b-p46cut-4 | 0.00102 | 0.00100 | 0.00061 | 0.000000 | 2 |
| qwen3 | `reference_attn4` | p46-qwen3lora | tp4-b-p46cut-4 | 0.00198 | 0.00093 | 0.00164 | 0.000000 | 2 |
