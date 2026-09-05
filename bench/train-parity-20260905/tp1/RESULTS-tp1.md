# tp1 — training parity on real weights, results (2026-09-05, box 49937730, one RTX 5090 on an EPYC 7Q83 host)

**PARTIAL — pending rows arrive before merge.** The table below is the verbatim output of `python tp1_reduce.py .` over
the JSON receipts in this directory at the 12:49Z snapshot (box, cut, fixture, the three amendments, the verdict rule,
the predictions and the layout: [`README.md`](README.md)). Rows in: Granite `reference` / `batched` (+ its `fused`
first run, a `harness_error` row), OLMoE `reference` / `fused` / `batched`, gpt-oss `attn_only` / `fused` / `batched` /
`mxfp4`, and Qwen3's load receipt. **Pending (arrive before merge):** Granite `fused` re-run (`logs/tp1b.sh`, after
`TP_DONE`, marker `TP2_DONE`), Qwen3 `reference` / `fused` / `batched` (the reference arm was at step 20/60 at the
snapshot), Gemma-4 ×3, Mixtral ×3 (offload). The reducer prints `not run` / `(no receipts)` for them; nothing else on
this page changes when they land — the final snapshot replaces this file's table and adds their reading.

Verdicts are in the registered units and no others (PREREG-flagship-matrix B2 / -model2 C2: `|Δ final train loss| ≤ 0.05`
AND `median step-wise |Δ| ≤ 0.05`, fused/batched vs the family's own `reference` arm, same box, same session). A VOID row
carries no parity number. Cost (s/step, tok/s, peak GB, J/step) is reported, never gated; every ratio is within one
family on this box. Eval-loss deltas are printed beside the band and labelled "not the band".

---

# tp1 — training parity table (/Users/jordananderson/.claude/worktrees/e4b-tp1-bundle/bench/train-parity-20260905/tp1)
Verdict rule: PASS iff |Δ final train loss| ≤ 0.05 AND median step-wise |Δ| ≤ 0.05 vs the family's reference arm (PREREG-flagship-matrix B2 / -model2 C2); VOID when the arm cannot be read (init_sha, C1, n_patched, kernel engagement, steps). Cost is reported, not gated. Eval deltas are not the band. No cross-family, cross-lane or cross-card ratio appears here.

### Granite-3.1-3B-A800M (`granite`)
- load: status=ok model_type=granitemoe load_s=3.2 loaded_gb=2.351 verify={'n_quantized': 32, 'n_unquantized': 0} wrapped=32 bare=0 classes={'Experts4bit': 32} offload=False geometry(H/I/L/E/k)=1536/512/32/40/8
- env: e4b 0.35.0 gnf4 0.30.0 torch 2.8.0+cu128 transformers 5.16.1 gpu NVIDIA GeForce RTX 5090 box_class pcie-full/launch-fast host_mem_gib 251.5
| arm | status | n_patched | kcalls/step min (need) | C1 | steps | s/step | tok/s | peak GB | J/step | loss first→last | eval 0→final (not the band) | Δ final train | median step \|Δ\| | verdict (registered units) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reference | ok | 0 | 0 (0) | ok | 60 | 4.266 | 23.7 | 2.810 | 399.5 | 3.4176→0.2274 | 3.4311→0.2582 | — | — | **REF** |
| batched | ok | 32 | 132 (64) | ok | 60 | 1.086 | 93.0 | 2.971 | 121.2 | 3.4135→0.2429 | 3.4241→0.2637 (Δ vs ref +0.0055) | 0.01553 | 0.01681 | **PASS** (×3.93 faster, peak ×1.057, J ×0.303) |
- init_sha (reference) `9da33dcfb11c4259`; dataset `ds_clinical.json` sha `76fb9036de80`; trainable 49807360 params in 384 tensors

### OLMoE-1B-7B-0924-Instruct (`olmoe`)
- load: status=ok model_type=olmoe load_s=3.9 loaded_gb=4.695 verify={'n_quantized': 16, 'n_unquantized': 0} wrapped=16 bare=0 classes={'Experts4bit': 16} offload=False geometry(H/I/L/E/k)=2048/1024/16/64/8
- env: e4b 0.35.0 gnf4 0.30.0 torch 2.8.0+cu128 transformers 5.16.1 gpu NVIDIA GeForce RTX 5090 box_class pcie-full/launch-fast host_mem_gib 251.5
| arm | status | n_patched | kcalls/step min (need) | C1 | steps | s/step | tok/s | peak GB | J/step | loss first→last | eval 0→final (not the band) | Δ final train | median step \|Δ\| | verdict (registered units) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reference | ok | 0 | 0 (0) | ok | 60 | 3.022 | 27.9 | 5.234 | 294.7 | 4.1678→0.2995 | 4.0130→0.3241 | — | — | **REF** |
| fused | ok | 16 | 64 (32) | ok | 60 | 0.939 | 89.9 | 5.234 | 100.0 | 4.1626→0.2863 | 4.0159→0.3203 (Δ vs ref -0.0038) | 0.01327 | 0.01249 | **PASS** (×3.22 faster, peak ×1.000, J ×0.339) |
| batched | ok | 16 | 24 (32) | ok | 60 | 1.911 | 44.2 | 6.211 | 201.5 | 4.1541→0.2821 | 4.0159→0.3197 (Δ vs ref -0.0045) | — | — | **VOID** (×1.58 faster, peak ×1.187, J ×0.684) — kernel calls/step min 24 < 2*n_patched=32 (not engaged on every layer) |
- init_sha (reference) `b14ec24782b3eb8a`; dataset `ds_clinical.json` sha `76fb9036de80`; trainable 60817408 params in 192 tensors

### gpt-oss-20b (`gptoss`)
- load: status=ok model_type=gpt_oss load_s=6.0 loaded_gb=14.359 verify={'n_quantized': 24, 'n_unquantized': 0} wrapped=0 bare=24 classes={'GptOssExperts4bit (bare)': 24} offload=False geometry(H/I/L/E/k)=2880/2880/24/32/4
- env: e4b 0.35.0 gnf4 0.30.0 torch 2.8.0+cu128 transformers 5.16.1 gpu NVIDIA GeForce RTX 5090 box_class pcie-full/launch-fast host_mem_gib 251.5
| arm | status | n_patched | kcalls/step min (need) | C1 | steps | s/step | tok/s | peak GB | J/step | loss first→last | eval 0→final (not the band) | Δ final train | median step \|Δ\| | verdict (registered units) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fused | refused | 0 | — (0) | — | 0 | — | — | — | — | —→— | —→— | — | — | **REFUSED** — 0 patched: [e4b.fast] fused TRAINING path on 0 ExpertsLoRA module(s) (dgrad kernel backward) |
| batched | refused | 0 | — (0) | — | 0 | — | — | — | — | —→— | —→— | — | — | **REFUSED** — 0 patched: [e4b.batched] batched training path on 0 ExpertsLoRA module(s) |
| attn_only | ok | 0 | 0 (0) | ok | 60 | 1.802 | 44.9 | 14.687 | 223.4 | 5.1060→0.3664 | 5.1405→0.3435 | — | — | **NO-PAIR** |
| mxfp4 (gnf4 ExpertsMxfp4LoRA, own text) | ok | — | — | prov pre==post: True | 60 | 5.160 | — | 6.221 | — | 5.461→2.622 | 4.591→2.184 | — | — | **EXPERIMENTAL — NOT LICENSED** (canary top1=0.906 kl=0.02495) |

### Qwen3-30B-A3B (`qwen3`)
- load: status=ok model_type=qwen3_moe load_s=16.6 loaded_gb=20.019 verify={'n_quantized': 48, 'n_unquantized': 0} wrapped=48 bare=0 classes={'Experts4bit': 48} offload=False geometry(H/I/L/E/k)=2048/768/48/128/8
- env: e4b 0.35.0 gnf4 0.30.0 torch 2.8.0+cu128 transformers 5.16.1 gpu NVIDIA GeForce RTX 5090 box_class pcie-full/launch-fast host_mem_gib 251.5
| arm | status | n_patched | kcalls/step min (need) | C1 | steps | s/step | tok/s | peak GB | J/step | loss first→last | eval 0→final (not the band) | Δ final train | median step \|Δ\| | verdict (registered units) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

## Cross-family summary (verdicts in registered units; cost ratios within family, this box)
| family | reference | fused | batched | other rows |
|---|---|---|---|---|
| Granite-3.1-3B-A800M | ok 4.27 s/step, 2.81 GB | not run | **PASS** (0.0155/0.0168; ×3.93) | — |
| OLMoE-1B-7B-0924-Instruct | ok 3.02 s/step, 5.23 GB | **PASS** (0.0133/0.0125; ×3.22) | **VOID** | — |
| gpt-oss-20b | not run | **REFUSED** | **REFUSED** | attn_only: ok; mxfp4: ok EXPERIMENTAL |
| Qwen3-30B-A3B | not run | not run | not run | — |
| Gemma-4-26B-A4B-it | (no receipts) | | | |
| Mixtral-8x7B-Instruct-v0.1 | (no receipts) | | | |

---

## Reading it

**Rule (pre-registered, P36 §"Registered criteria", applied by the reducer):** VOID if the arm's `init_sha` ≠ the
reference's, C1 not bit-exact (or 0 bytes hashed, or empties skipped, or the byte-flip control silent) in either arm,
`n_patched == 0` on an accelerated arm, `kernel_calls_per_step_min < 2 × n_patched`, or a different step count; else
PASS iff both registered numbers are ≤ 0.05; else FAIL. A `refused` / `harness_error` / `alarm` / `oom` / `load_fault` /
`verify_failed` receipt is a row with that status. `attn_only` has no pair and gets no verdict. `mxfp4` is EXPERIMENTAL —
NOT LICENSED, printed from the runner's own artifact.

### Granite-3.1-3B-A800M (`granitemoe`) — first real-weight pass through the direct path

- **Load + verify (measured, new):** `load_moe_4bit_streaming(offload=False)` quantised **32/32** MoE layers (40 experts
  each) in 3.2 s from the staged snapshot, **2.351 GB** loaded, `verify_moe_4bit(strict=True)` 32 quantised / 0
  unquantised, **32 `ExpertsLoRA`** installed over `Experts4bit` (0 bare). C1 hashed **1,698,693,120 B** of frozen expert
  bytes per arm — exactly the 1.70 GB the pre-registration's arithmetic (E·L·3·H·I × 0.5625) predicted — with 0 empties
  and the byte-flip control firing. Until this lane, Granite's real weights had only ever been loaded through the serving
  lanes' arena path (MATRIX §1: "not_tested" for the direct `from_float` + `ExpertsLoRA` path).
- **`reference` — REF.** 60 steps, loss 3.4176 → 0.2274, held-out eval 3.4311 → 0.2582; **4.266 s/step**, 23.7 tok/s
  (6,060 tokens over 60 rows — the clinical rows are ≈100 tokens, so "seq 512" is a ceiling), peak 2.810 GB, 399.5 J/step.
  The per-expert loop over 40 experts × 32 layers is launch-bound: P7 assumed 0.5–1 s/step for this family and the
  reference arm is 4–8× that (an alarm-sizing assumption, replaced by the measurement; no alarm fired).
- **`batched` — PASS.** 32/32 patched, whole-stack dequant calls per step 132–174 (min 132 ≥ 2 × 32 = 64: the kernel
  reached every layer on every step), `init_sha` equal to the reference's (`9da33dcfb11c4259…`), C1 clean in both arms.
  **|Δ final train loss| 0.01553, median step-wise |Δ| 0.01681** — both inside the 0.05 band. Cost on this box: **×3.93
  faster** per step (1.086 vs 4.266 s), peak ×1.057 (2.971 GB), **J/step ×0.303** (121.2 vs 399.5). Eval Δ vs reference
  +0.0055 (not the band).
- **`fused` — `harness_error` (kept; amendment 3).** The arm loaded (32/32, receipt `granite_train_load.json` written by
  `reference`; `logs/run_granite_fused.log`) and died in the first forward with `TypeError: _dequant_whole() got an
  unexpected keyword argument 'weights_fn'` — the harness's kernel-call counter (`KernelCounter.install`, D5) wrapped
  `nf4_qlora.fused_grouped_lora` and `batched._dequant_whole` in one scope with the same closure variable `orig`, so
  Python's late binding sent the fused kernel's call to `_dequant_whole`. Harness, not shipped code: the shipped
  `enable_fast_train` had patched and was calling the kernel. **The re-run on the patched harness is the row that
  counts (pending, `logs/tp1b.sh`).** No receipt JSON exists for this arm (the process died before writing one), so the
  reducer shows the arm as `not run`; the log is the row.

### OLMoE-1B-7B-0924-Instruct (`olmoe`) — first parity reading on a registered text with real weights

- **Load + verify:** 16/16 (64 experts each), 4.695 GB loaded (the register's `e4b.train.olmoe-fits` measured 4.70 GB on
  the base checkpoint), 16 `ExpertsLoRA`, C1 3,623,878,656 B hashed per arm, control fires.
- **`reference` — REF.** loss 4.1678 → 0.2995, eval 4.0130 → 0.3241; 3.022 s/step, 27.9 tok/s, peak 5.234 GB, 294.7 J/step.
- **`fused` — PASS.** `enable_fast_train(dgrad=True)` patched 16/16; `fused_grouped_lora` called **64 times on every
  step** (≥ 32); `init_sha` equal (`b14ec24782b3eb8a…`); C1 clean. **|Δ final| 0.01327, median 0.01249.** Cost: **×3.22
  faster** (0.939 vs 3.022 s), peak ×1.000 (5.234 GB — identical to the reference's), **J/step ×0.339** (100.0 vs 294.7).
  Eval Δ −0.0038 (not the band). This is the first fused-vs-reference reading for this family on a registered text with
  real weights — the dgrad-gate receipt (`e4b.train.fast-train-dgrad`) used 8 steps of synthetic random tokens.
- **`batched` — VOID (not engaged on every layer).** 16/16 patched, C1 clean, `init_sha` equal, 60 steps — and the
  whole-stack dequant count per step ran **24–54 with a minimum of 24 < 2 × 16 = 32**. `engines/batched.py:208` returns
  the reference forward whenever `n_grp × widest > _PAD_WASTE_LIMIT × total` (4.0), silently and without a counter; on
  ≈85-token rows routed over 64 experts the padded gather exceeded that ratio on some layers on some steps, so the arm
  is a mixture of the batched kernel and the reference loop — "a reference re-run wearing a batched label" on those
  layers. P2 named this risk for the 128-expert families; it fired first on the 64-expert one. **No parity number is
  quoted for a VOID row** (reading rule 5; the arm's loss curve is in its receipt for the record). Its cost columns
  (×1.58, peak ×1.187 = 6.211 GB, J ×0.684) describe the mixed path, not the batched kernel.

### gpt-oss-20b (`gpt_oss`) — every row says what the code says: no expert-LoRA path

- **Load + verify:** the released MXFP4 blocks/scales dequantised and rebuilt as **24 bare `GptOssExperts4bit`** (0
  `ExpertsLoRA`; `loader.py`: "Built bare (no ExpertsLoRA): GPT-OSS-aware training LoRA is a separate change"),
  14.359 GB loaded, `verify_moe_4bit(strict=True)` 24/24 quantised (the subclass counts as quantised — assumption 2 of
  MATRIX §4, now observed).
- **`fused` — REFUSED; `batched` — REFUSED.** Probed from the `attn_only` process (D8): `enable_fast_train(dgrad=True)`
  → 0 (`[e4b.fast] fused TRAINING path on 0 ExpertsLoRA module(s) (dgrad kernel backward)`); `enable_batched_train` → 0
  (`[e4b.batched] batched training path on 0 ExpertsLoRA module(s)`). Two refusal rows, as P3 predicted; not a zero, a
  build-out item.
- **`attn_only` — NO-PAIR (trains).** `add_attention_lora` wrapped 96 projections (q/k/v/o × 24 layers; gpt-oss's
  `attention_bias=True` projections keep their bias inside `LoRALinear`), 3,981,312 trainable fp32 parameters in 192
  tensors; loss 5.1060 → 0.3664, eval 5.1405 → 0.3435 over 60 steps; **C1 on the bare stacks: 10,749,542,400 B hashed
  (96 tensors), 0 empties, control fires, bit-exact.** 1.802 s/step, 44.9 tok/s, peak 14.687 GB, 223.4 J/step. This row
  is attention-only QLoRA over frozen NF4-requantised experts; it never says "trains the experts".
- **`mxfp4` — EXPERIMENTAL, NOT LICENSED.** grouped-nf4-gemm's `run_mxfp4_20b_qlora` (`mxfp4_qlora.ExpertsMxfp4LoRA`:
  LoRA over the released e2m1/e8m0 bytes, faithful clamped GLU + biases, recompute-in-backward) on **its own text**
  (wikitext-2 train, its seed-41 chunking): step-0 canary vs transformers' dequant path **top-1 0.906 over 32 compared
  positions, KL 0.02495** (P4's `≥ 0.9` clause holds by 0.006); `provenance.pre_equals_post` **True** over 96 file
  tensors; train loss 5.461 → 2.622, eval 4.591 → 2.184 (@0 → @60, every 10 steps in `run_artifact.json`); mean 5.160
  s/step; peak 6.221 GB (`cuda.loaded_gb` 3.971 with 13.8 GB host RSS after the native patch — the runner's own
  residency scheme, `logs/run_gptoss_mxfp4.log`; not compared to the e4b arms' `loaded_gb`). No pair, no parity verdict,
  different text, different runner: it is never compared to the e4b arms, and this lane licenses nothing on it.

### Qwen3-30B-A3B (`qwen3_moe`) — load row in; arms pending

- **Load + verify (measured):** 48/48 MoE layers × 128 experts quantised in 16.6 s from the staged snapshot, **20.019 GB**
  loaded resident on the 32 GB card, `verify_moe_4bit(strict=True)` 48/0, 48 `ExpertsLoRA`. The pre-registration's
  arithmetic said ≈18.7 GB; the flagship receipts loaded this family with `offload=True` and the dgrad-gate resident on a
  48 GB card, so this is the first resident load of the family on a 32 GB card in the register.
- **`reference` in progress at the snapshot** (`logs/run_qwen3_reference.log`: step 20/60, loss 0.350, 12.8–13.0 s/step —
  P7 assumed 6–8 s; the 3000-s alarm holds with 60 × 13 s ≈ 13 min of steps). `fused`, `batched`: pending.

### Gemma-4-26B-A4B-it, Mixtral-8x7B-Instruct-v0.1 — pending

No receipts at the snapshot. Gemma-4-it may fault at load per #344 (P6: a `load_fault` row, not a retry); Mixtral runs
`offload=True` (P5). Their rows arrive before merge.

### Cross-family, in one line each (nothing here is a ratio across families)

- Granite: direct real-weight load measured; batched **PASS** (×3.93, J ×0.303); fused pending (harness re-run).
- OLMoE: fused **PASS** on a registered text (×3.22, J ×0.339); batched **VOID** (fallback engaged, uncounted in the code).
- gpt-oss: fused / batched **REFUSED** (0 patched); attention-only trains with the frozen stacks bit-exact; MXFP4 route
  **EXPERIMENTAL** (canary passes, provenance holds, loss falls) — not licensed.
- Qwen3: resident load on 32 GB measured; three arms pending. Gemma-4, Mixtral: pending.

### What this table does not say

No convergence claim: 60 steps rank arms against each other on one text, they do not train a model. "PASS on one
text" is what a PASS here means (the flagship needed five). No cross-family, cross-lane or cross-card ratio: tp1's
Qwen3 and Gemma-4 rows, when they land, sit beside the flagship receipts and are never divided into them. No training
throughput position and no anchor-class projection: s/step and tok/s are this box's (train-anchor class
`pcie-full/launch-fast`), and two 5090s have differed 1.65× on one training config. Experimental ≠ licensed.
