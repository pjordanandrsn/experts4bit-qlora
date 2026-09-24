# tp4 — e4b (GitHub main) vs Unsloth vs plain HF+PEFT+bnb per MoE family, same box, one fixture (/root/tp4)
Rule (tp4/TP4-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per TP4's validity rules with the registered n_layers (granite 32, olmoe 16, qwen3 48, qwen3_5 40, gemma4 30, mixtral 32, gptoss 24, deepseek_v4 43, kimi_k3 None); positions = s/step ratios other/e4b from the primary triple (median of steps 11..N; > 1 = e4b faster), quoted only when both arms are VALID, with the quality reading (held-out |Δ| at N ≤ 0.05 nats → COMPARABLE, else the ratio carries the flag) and the other arm's 4-bit REGIME beside it; e4b internal parity in tp1's B2/C2 units, informational; Qwen3's tp2-fixture anchor pair vs tp2's 1.457 / P38's 1.413 within ±10 % or a stated finding. VOID never enters a ratio. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.37.3 @cd2716decf77f0c7d0e81b7e58dbe5dd5345cc29 (GitHub main)
gnf4 0.32.1 @9206352f7c49d76432824f7ce8872f0b285f2a4e (GitHub main)
torch(e4b/hf) 2.8.0+cu128
triton(e4b/hf) 3.4.0
transformers(e4b/hf) 5.17.0
bitsandbytes(e4b/hf) 0.50.2
peft(hf) 0.20.0
unsloth 2026.9.11
unsloth_zoo 2026.9.7
torch(unsloth) 2.8.0+cu128
triton(unsloth) 3.4.0
transformers(unsloth) 5.5.0
bitsandbytes(unsloth) 0.50.2
peft(unsloth) 0.21.0
torchao None
moe_backend native_torch
```
`box.json`
```
{
 "box": "C",
 "run_id": "p67-gemma4-floor-1",
 "instance_id": "52449460",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "580.105.08",
 "cpu": "AMD EPYC 7302 16-Core Processor",
 "nproc": 64,
 "mem_total_kb": "528207196",
 "cgroup_memory_max": "259623223296",
 "disk_root": "overlay         320G   76M  320G   1% /",
 "hostname": "ba15a6aa9460",
 "registered_gpu_class": "5090",
 "prereg": "tp4/TP4-PREREG.md"
}
```

### Gemma-4-26B-A4B-it (`gemma4`, registered n_layers 30)
- model `google/gemma-4-26B-A4B-it` @ `4d7ae4984b7d`; tokens sha `4ea779a5b6a1`; N=20; fixture template alpaca seq 2048 micro-batch 2 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 487280640; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | engagement | n_attn4 | trainable | adapter MB (dtypes) | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4 | **OK** | VALID | 20 | 3.725 | 415.3 | 23.899 | 895.4 | 5.7057→1.1422 | 5.1227→1.2034 | patched 30 / kcalls 480 | 115 | 487280640 | 997.7 (torch.bfloat16,torch.float32) | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth | **NOT_RUN** | — | 20 | — | — | — | — | —→— | —→— | stacks — / fwd — / u8 — | — | — | 0.0 (?) | — | skipped by TP4_SKIP |
| hf | hf_peft | **NOT_RUN** | — | 20 | — | — | — | — | —→— | —→— | peft mods — / params — / fwd — | — | — | 0.0 (?) | — | skipped by TP4_SKIP |
| e4b | reference_attn4 | **OK** | VALID | 20 | 42.026 | 37.6 | 23.899 | 3531.2 | 5.7396→1.1289 | 5.2121→1.1921 | patched 0 / kcalls 0 | 115 | 487280640 | 997.7 (torch.bfloat16,torch.float32) | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | batched_attn4 | **OK** | VALID | 20 | 4.056 | 386.9 | 23.899 | 1520.5 | 5.8049→1.0920 | 5.1854→1.1542 | patched 30 / kcalls 0 | 115 | 487280640 | 997.7 (torch.bfloat16,torch.float32) | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_nodgrad | **NOT_RUN** | — | 20 | — | — | — | — | —→— | —→— | patched — / kcalls — | — | — | 0.0 (?) | — | skipped by TP4_SKIP |
| e4b | reference_attn4_perm1 | **OK** | VALID | 20 | 41.806 | 37.1 | 23.899 | 3632.7 | 5.5223→1.2499 | 5.1909→1.3044 | patched 0 / kcalls 0 | 115 | 487280640 | 997.7 (torch.bfloat16,torch.float32) | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | reference_attn4_perm2 | **OK** | VALID | 20 | 42.194 | 37.4 | 23.899 | 3536.6 | 5.6821→1.1402 | 5.2117→1.1705 | patched 0 / kcalls 0 | 115 | 487280640 | 997.7 (torch.bfloat16,torch.float32) | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | reference_attn4_perm3 | **OK** | VALID | 20 | 42.534 | 37.2 | 23.899 | 3679.2 | 5.7914→1.2436 | 5.1747→1.3811 | patched 0 / kcalls 0 | 115 | 487280640 | 997.7 (torch.bfloat16,torch.float32) | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | reference_attn4_perm4 | **OK** | VALID | 20 | 42.483 | 37.5 | 23.899 | 3471.1 | 5.8215→1.1546 | 5.1948→1.2455 | patched 0 / kcalls 0 | 115 | 487280640 | 997.7 (torch.bfloat16,torch.float32) | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | reference_attn4_repeat | **OK** | VALID | 20 | 40.889 | 38.3 | 23.899 | 3432.9 | 5.7396→1.1513 | 5.2121→1.2168 | patched 0 / kcalls 0 | 115 | 487280640 | 997.7 (torch.bfloat16,torch.float32) | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4` **92.7 s** before step 1 (52% of the arm): load_weights 26.9, c1_before 25.4, c1_after 24.4, eval0 19.2; unattributed 1.555; budget 1260.0
- prologue `e4b/reference_attn4` **166.2 s** before step 1 (15% of the arm): eval0 91.6, load_weights 29.7, c1_after 26.4, c1_before 24.9; unattributed 1.425; budget 1890.0
- prologue `e4b/batched_attn4` **83.9 s** before step 1 (48% of the arm): load_weights 26.4, c1_after 26.1, c1_before 26.0, eval0 10.8; unattributed 1.45; budget 1890.0
- prologue `e4b/reference_attn4_perm1` **162.2 s** before step 1 (15% of the arm): eval0 89.7, load_weights 27.0, c1_before 25.3, c1_after 24.9; unattributed 1.373; budget 1890.0
- prologue `e4b/reference_attn4_perm2` **161.1 s** before step 1 (15% of the arm): eval0 90.4, load_weights 26.6, c1_before 24.2, c1_after 23.6; unattributed 1.387; budget 1683.1
- prologue `e4b/reference_attn4_perm3` **160.6 s** before step 1 (15% of the arm): eval0 90.6, load_weights 26.3, c1_after 24.8, c1_before 23.9; unattributed 1.395; budget 1288.0
- prologue `e4b/reference_attn4_perm4` **161.1 s** before step 1 (15% of the arm): eval0 90.4, load_weights 26.6, c1_after 26.2, c1_before 24.1; unattributed 1.443; budget 891.1
- prologue `e4b/reference_attn4_repeat` **163.6 s** before step 1 (15% of the arm): eval0 89.4, load_weights 27.7, c1_before 25.7, c1_after 25.0; unattributed 1.518; budget 1890.0
- e4b internal parity (tp1's rule, informational): fused_attn4 vs reference_attn4 Δfinal 0.01331, median step |Δ| 0.03565 → **PASS** (band 0.05/0.05); ×11.28 faster per step, peak ×1.000
- **NO POSITION QUOTED (unsloth)** — arms not both VALID: e4b VALID / unsloth —
- **NO POSITION QUOTED (hf)** — arms not both VALID: e4b VALID / hf —

## Cross-family summary
| family | e4b primary | unsloth primary | hf primary | unsloth/e4b s/step | hf/e4b s/step | quality | e4b internal parity | notes |
|---|---|---|---|---|---|---|---|---|
| Granite-3.1-3B-A800M-instruct | (no receipts) | | | | | | | |
| OLMoE-1B-7B-0924-Instruct | (no receipts) | | | | | | | |
| gpt-oss-20b | (no receipts) | | | | | | | |
| Qwen3-30B-A3B | (no receipts) | | | | | | | |
| Qwen3.6-35B-A3B (qwen3_5_moe) | (no receipts) | | | | | | | |
| Gemma-4-26B-A4B-it | OK VALID | NOT_RUN — | NOT_RUN — | not quoted | not quoted | — | PASS | e4b/reference_attn4: OK; e4b/batched_attn4: OK; e4b/fused_attn4_nodgrad: NOT_RUN; e4b/reference_attn4_perm1: OK; e4b/reference_attn4_perm2: OK; e4b/reference_attn4_perm3: OK; e4b/reference_attn4_perm4: OK; e4b/reference_attn4_repeat: OK |
| Mixtral-8x7B-Instruct-v0.1 | (no receipts) | | | | | | | |
| DeepSeek-V4-Flash | (no receipts) | | | | | | | |
| Kimi-K3 | (no receipts) | | | | | | | |

## Predictions P1–P9 (TP4-PREREG.md, scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P1 | granite | **UNTESTED** | no receipts |
| P1 | olmoe | **UNTESTED** | no receipts |
| P2 | qwen3 | **UNTESTED** | no receipts |
| P3 | qwen3 | **UNTESTED** | no anchor-pair receipts |
| P4 | qwen3_5 | **UNTESTED** | no receipts |
| P5 | gemma4 | **FALSIFIED** | e4b OK, unsloth NOT_RUN, hf NOT_RUN |
| P6 | mixtral | **UNTESTED** | no receipts |
| P7 | gptoss | **UNTESTED** | no receipts |
| P8 | all | **UNTESTED** | no pair with both arms VALID |
| P9 | all | **HELD** | gemma4 PASS |
