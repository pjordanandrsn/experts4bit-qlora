# tp1 — training parity on real weights, results (2026-09-05, box 49937730, one RTX 5090 on an EPYC 7Q83 host)

**PARTIAL — the pending rows arrive before merge.** The table below is the verbatim output of `python tp1_reduce.py .`
(reducer **v2**, this directory's copy; the copy that ran on the box is [`logs/tp1_reduce.py`](logs/tp1_reduce.py), v1,
whose table it supersedes) over the artefacts in this directory at the 12:49Z snapshot. Box, cut, fixture, the four
amendments, the verdict rule verbatim, the predictions and the layout: [`README.md`](README.md).

**Row status vocabulary (phase directive 2026-09-05 14:45Z):** every row is exactly one of **OK, REFUSED,
HARNESS_ERROR, ALARM, OOM, NOT_RUN, EXPERIMENTAL**, classified mechanically from `summary.txt` (rc per attempt), the
receipt or stub, `logs/outer.log` and the run logs — never from a missing error; a missing receipt is NOT_RUN with the
reason from `outer.log` (a fetch or arm alarm is ALARM). The **parity verdict** (PASS / FAIL / VOID) is a separate
column, computed only for OK rows of an accelerated arm, in the registered units and no others
(PREREG-flagship-matrix B2 / -model2 C2: `|Δ final train loss| ≤ 0.05` AND `median step-wise |Δ| ≤ 0.05` against the
family's own `reference` arm, same box, same session; VOID when the arm cannot be read). A VOID row carries no parity
number. Cost is reported, never gated; every ratio is within one family on this box. Eval deltas are "not the band".
Every attempt is a row: Granite's first `fused` attempt stays as HARNESS_ERROR beside its corrected-counter re-run
when that lands, and the amendment that touched a row is named in its last column.

Rows in at this snapshot: Granite `reference` OK, `fused` **HARNESS_ERROR** (attempt 1, amendment 3), `batched` OK;
OLMoE `reference` / `fused` / `batched` OK; gpt-oss `fused` / `batched` REFUSED, `attn_only` OK, `mxfp4` EXPERIMENTAL;
Qwen3 load OK. **NOT_RUN at this snapshot, arriving before merge:** Granite `fused` re-run (`logs/tp1b.sh`, marker
`TP2_DONE`), Qwen3 `reference` / `fused` / `batched` (started 12:41:43Z; the private record says all three completed on
the box at 13:27Z), Gemma-4 ×3 (**amendment 4**: its fetch hung at 13:46Z and was left to its 4800-s alarm; `tp1b.sh`
redoes the family), Mixtral ×3 (`offload=True`; the amendment-4 redo rule applies should its fetch fail the same way).

---

# tp1 — training parity table (`tp1`, reducer v2)
Row status ∈ {OK, REFUSED, HARNESS_ERROR, ALARM, OOM, NOT_RUN, EXPERIMENTAL}, classified from summary.txt (rc per attempt), the receipt / stub, outer.log and the run logs — never from a missing error. Verdict (OK accelerated rows only): PASS iff |Δ final train loss| ≤ 0.05 AND median step-wise |Δ| ≤ 0.05 vs the family's reference arm (PREREG-flagship-matrix B2 / -model2 C2); VOID when the arm cannot be read (init_sha, C1, n_patched, kernel engagement, steps). Cost is reported, not gated. Eval deltas are not the band. No cross-family, cross-lane or cross-card ratio appears here.
- lane header (summary.txt): ANCHOR rc=0 class=pcie-full/launch-fast · DATASET clinical sha=76fb9036de80f3bb495fe4c8894159fcb1d399d2437293e012e264d81949f791
- anchor.json: OK class per summary; flops median 184.36 TFLOP/s, launch 186947 /s, h2d 25.93 GB/s
- markers shipped: none; tp1b marker in outer.log: no; lane end lines: GRANITE DONE, OLMOE DONE, GPTOSS DONE
- lane-level amendments: 1 (pre-flight floor 40 MB/s, fetch alarms doubled — logs/tp1_run.sh vs logs/tp1_run.sh.pre-amend, both shipped); 2 (helper fetch by archive tarball after the fetch-by-sha false start — logs/outer.attempt1.log shipped); 3 and 4 are attached to their rows below.

### Granite-3.1-3B-A800M (`granite`)
- load: status=ok model_type=granitemoe load_s=3.2 loaded_gb=2.351 verify={'n_quantized': 32, 'n_unquantized': 0} wrapped=32 bare=0 classes={'Experts4bit': 32} offload=False geometry(H/I/L/E/k)=1536/512/32/40/8
- env: e4b 0.35.0 gnf4 0.30.0 torch 2.8.0+cu128 transformers 5.16.1 gpu NVIDIA GeForce RTX 5090 box_class pcie-full/launch-fast host_mem_gib 251.5
| arm | attempt | status | verdict | n_patched | kcalls/step min (need) | C1 | steps | s/step | tok/s | peak GB | J/step | loss first→last | eval 0→final (not the band) | Δ final train | median step \|Δ\| | reason / amendment |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reference | 1/1 | **OK** | **REF** | 0 | 0 (0) | ok | 60 | 4.266 | 23.7 | 2.810 | 399.5 | 3.4176→0.2274 | 3.4311→0.2582 | — | — | — |
| fused | 1/1 | **HARNESS_ERROR** | — | — | — (0) | — | — | — | — | — | — | —→— | —→— | — | — | rc=1; TypeError: _dequant_whole() got an unexpected keyword argument 'weights_fn' (innermost frame tp1_train_smoke.py, run_granite_fused.attempt1.log); no CELL line in summary.txt (the next arm's result shares its line) — amendment 3 (late-binding counter closure in the harness; patched in flight; re-run by tp1b) |
| batched | 1/1 | **OK** | **PASS** (×3.93 faster, peak ×1.057, J ×0.303) | 32 | 132 (64) | ok | 60 | 1.086 | 93.0 | 2.971 | 121.2 | 3.4135→0.2429 | 3.4241→0.2637 (Δ vs ref +0.0055) | 0.01553 | 0.01681 | — |
- init_sha (reference) `9da33dcfb11c4259`; dataset `ds_clinical.json` sha `76fb9036de80`; trainable 49807360 params in 384 tensors

### OLMoE-1B-7B-0924-Instruct (`olmoe`)
- load: status=ok model_type=olmoe load_s=3.9 loaded_gb=4.695 verify={'n_quantized': 16, 'n_unquantized': 0} wrapped=16 bare=0 classes={'Experts4bit': 16} offload=False geometry(H/I/L/E/k)=2048/1024/16/64/8
- env: e4b 0.35.0 gnf4 0.30.0 torch 2.8.0+cu128 transformers 5.16.1 gpu NVIDIA GeForce RTX 5090 box_class pcie-full/launch-fast host_mem_gib 251.5
| arm | attempt | status | verdict | n_patched | kcalls/step min (need) | C1 | steps | s/step | tok/s | peak GB | J/step | loss first→last | eval 0→final (not the band) | Δ final train | median step \|Δ\| | reason / amendment |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reference | 1/1 | **OK** | **REF** | 0 | 0 (0) | ok | 60 | 3.022 | 27.9 | 5.234 | 294.7 | 4.1678→0.2995 | 4.0130→0.3241 | — | — | — |
| fused | 1/1 | **OK** | **PASS** (×3.22 faster, peak ×1.000, J ×0.339) | 16 | 64 (32) | ok | 60 | 0.939 | 89.9 | 5.234 | 100.0 | 4.1626→0.2863 | 4.0159→0.3203 (Δ vs ref -0.0038) | 0.01327 | 0.01249 | — |
| batched | 1/1 | **OK** | **VOID** (×1.58 faster, peak ×1.187, J ×0.684) | 16 | 24 (32) | ok | 60 | 1.911 | 44.2 | 6.211 | 201.5 | 4.1541→0.2821 | 4.0159→0.3197 (Δ vs ref -0.0045) | — | — | kernel calls/step min 24 < 2*n_patched=32 (not engaged on every layer) |
- init_sha (reference) `b14ec24782b3eb8a`; dataset `ds_clinical.json` sha `76fb9036de80`; trainable 60817408 params in 192 tensors

### gpt-oss-20b (`gptoss`)
- load: status=ok model_type=gpt_oss load_s=6.0 loaded_gb=14.359 verify={'n_quantized': 24, 'n_unquantized': 0} wrapped=0 bare=24 classes={'GptOssExperts4bit (bare)': 24} offload=False geometry(H/I/L/E/k)=2880/2880/24/32/4
- env: e4b 0.35.0 gnf4 0.30.0 torch 2.8.0+cu128 transformers 5.16.1 gpu NVIDIA GeForce RTX 5090 box_class pcie-full/launch-fast host_mem_gib 251.5
| arm | attempt | status | verdict | n_patched | kcalls/step min (need) | C1 | steps | s/step | tok/s | peak GB | J/step | loss first→last | eval 0→final (not the band) | Δ final train | median step \|Δ\| | reason / amendment |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fused | 1/1 | **REFUSED** | — | 0 | — (0) | — | 0 | — | — | — | — | —→— | —→— | — | — | 0 patched: [e4b.fast] fused TRAINING path on 0 ExpertsLoRA module(s) (dgrad kernel backward) (stub written by the probing arm, D8) |
| batched | 1/1 | **REFUSED** | — | 0 | — (0) | — | 0 | — | — | — | — | —→— | —→— | — | — | 0 patched: [e4b.batched] batched training path on 0 ExpertsLoRA module(s) (stub written by the probing arm, D8) |
| attn_only | 1/1 | **OK** | **no pair** | 0 | 0 (0) | ok | 60 | 1.802 | 44.9 | 14.687 | 223.4 | 5.1060→0.3664 | 5.1405→0.3435 | — | — | — |
| mxfp4 (gnf4 ExpertsMxfp4LoRA, own text) | 1/1 | **EXPERIMENTAL** | — | — | — | prov pre==post: True | 60 | 5.160 | — | 6.221 | — | 5.461→2.622 | 4.591→2.184 | — | — | NOT LICENSED; canary top1=0.906 kl=0.02495; grouped-nf4-gemm run_mxfp4_20b_qlora on its own text; never licensed |

### Qwen3-30B-A3B (`qwen3`)
- load: status=ok model_type=qwen3_moe load_s=16.6 loaded_gb=20.019 verify={'n_quantized': 48, 'n_unquantized': 0} wrapped=48 bare=0 classes={'Experts4bit': 48} offload=False geometry(H/I/L/E/k)=2048/768/48/128/8
- env: e4b 0.35.0 gnf4 0.30.0 torch 2.8.0+cu128 transformers 5.16.1 gpu NVIDIA GeForce RTX 5090 box_class pcie-full/launch-fast host_mem_gib 251.5
| arm | attempt | status | verdict | n_patched | kcalls/step min (need) | C1 | steps | s/step | tok/s | peak GB | J/step | loss first→last | eval 0→final (not the band) | Δ final train | median step \|Δ\| | reason / amendment |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reference | 1/1 | **NOT_RUN** | — | — | — (0) | — | — | — | — | — | — | —→— | —→— | — | — | started 2026-09-05T12:41:43Z (outer.log), no result line at the snapshot -- in progress or lost |
| fused | 1/1 | **NOT_RUN** | — | — | — (0) | — | — | — | — | — | — | —→— | —→— | — | — | not started at the snapshot (the checkpoint had staged; an earlier arm of the family was still running) |
| batched | 1/1 | **NOT_RUN** | — | — | — (0) | — | — | — | — | — | — | —→— | —→— | — | — | not started at the snapshot (the checkpoint had staged; an earlier arm of the family was still running) |

### Gemma-4-26B-A4B-it (`gemma4`)
- load: (no receipt at this snapshot)
| arm | attempt | status | verdict | reason / amendment |
|---|---|---|---|---|
| reference | 1/1 | **NOT_RUN** | — | not reached at the snapshot (no fetch or arm line in outer.log) |
| fused | 1/1 | **NOT_RUN** | — | not reached at the snapshot (no fetch or arm line in outer.log) |
| batched | 1/1 | **NOT_RUN** | — | not reached at the snapshot (no fetch or arm line in outer.log) |

### Mixtral-8x7B-Instruct-v0.1 (`mixtral`)
- load: (no receipt at this snapshot)
| arm | attempt | status | verdict | reason / amendment |
|---|---|---|---|---|
| reference | 1/1 | **NOT_RUN** | — | not reached at the snapshot (no fetch or arm line in outer.log) |
| fused | 1/1 | **NOT_RUN** | — | not reached at the snapshot (no fetch or arm line in outer.log) |
| batched | 1/1 | **NOT_RUN** | — | not reached at the snapshot (no fetch or arm line in outer.log) |

## Per-family matrix (the directive's ten columns; support words in the capability vocabulary, read from the LAST attempt of each arm)
| family | reference support | fused support | batched support | native-format route | loss-parity result | s/step (ref / fused / batched) | peak GB (ref / fused / batched) | tok/s (ref / fused / batched) | evidence tier | limitations / refusal reason |
|---|---|---|---|---|---|---|---|---|---|---|
| Granite-3.1-3B-A800M | supported (OK) | harness_error — rc=1; TypeError: _dequant_whole() got an unexpected keyword argument 'weights_fn' (innermost frame tp1_train_smoke.py, run_granite_fused.attempt1.log) | supported (PASS) | n/a (NF4 checkpoint) | fused — (HARNESS_ERROR); batched PASS 0.0155 / 0.0168 | 4.266 / — / 1.086 | 2.810 / — / 2.971 | 23.7 / — / 93.0 | measured (receipts in this directory: reference, batched) | fused: HARNESS_ERROR — rc=1; TypeError: _dequant_whole() got an unexpected keyword argument 'weights_fn' (innermost frame tp1_train_smoke.py, run_granite_fused.attempt1.log) [amendment 3 (late-binding counter closure in the harness; patched in flight; re-run by tp1b)] |
| OLMoE-1B-7B-0924-Instruct | supported (OK) | supported (PASS) | void (VOID) | n/a (NF4 checkpoint) | fused PASS 0.0133 / 0.0125; batched VOID | 3.022 / 0.939 / 1.911 | 5.234 / 5.234 / 6.211 | 27.9 / 89.9 / 44.2 | measured (receipts in this directory: reference, fused, batched) | batched: OK/VOID |
| gpt-oss-20b | refused — the loader builds the experts bare (no ExpertsLoRA); attention-only QLoRA over them: attn_only OK (no pair) | refused — 0 patched: [e4b.fast] fused TRAINING path on 0 ExpertsLoRA module(s) (dgrad kernel backward) (stub written by the probing arm, D8) | refused — 0 patched: [e4b.batched] batched training path on 0 ExpertsLoRA module(s) (stub written by the probing arm, D8) | experimental — grouped-nf4-gemm run_mxfp4_20b_qlora on its own text; never licensed | fused — (REFUSED); batched — (REFUSED) | 1.802 / — / — (ref = attn_only) | 14.687 / — / — | 44.9 / — / — | measured (receipts in this directory: fused, batched, attn_only, mxfp4); mxfp4 experimental, not licensed | fused: REFUSED — 0 patched: [e4b.fast] fused TRAINING path on 0 ExpertsLoRA module(s) (dgrad kernel backward) (stub written by the probing arm, D8); batched: REFUSED — 0 patched: [e4b.batched] batched training path on 0 ExpertsLoRA module(s) (stub written by the probing arm, D8); mxfp4: EXPERIMENTAL — grouped-nf4-gemm run_mxfp4_20b_qlora on its own text; never licensed |
| Qwen3-30B-A3B | not_tested — started 2026-09-05T12:41:43Z (outer.log), no result line at the snapshot -- in progress or lost | not_tested — not started at the snapshot (the checkpoint had staged; an earlier arm of the family was still running) | not_tested — not started at the snapshot (the checkpoint had staged; an earlier arm of the family was still running) | n/a (NF4 checkpoint) | fused — (NOT_RUN); batched — (NOT_RUN) | — / — / — | — / — / — | — / — / — | pending (no receipt at this snapshot) | reference: NOT_RUN — started 2026-09-05T12:41:43Z (outer.log), no result line at the snapshot -- in progress or lost; fused: NOT_RUN — not started at the snapshot (the checkpoint had staged; an earlier arm of the family was still running); batched: NOT_RUN — not started at the snapshot (the checkpoint had staged; an earlier arm of the family was still running) |
| Gemma-4-26B-A4B-it | not_tested — not reached at the snapshot (no fetch or arm line in outer.log) | not_tested — not reached at the snapshot (no fetch or arm line in outer.log) | not_tested — not reached at the snapshot (no fetch or arm line in outer.log) | n/a (NF4 checkpoint) | fused — (NOT_RUN); batched — (NOT_RUN) | — / — / — | — / — / — | — / — / — | pending (no receipt at this snapshot) | reference: NOT_RUN — not reached at the snapshot (no fetch or arm line in outer.log); fused: NOT_RUN — not reached at the snapshot (no fetch or arm line in outer.log); batched: NOT_RUN — not reached at the snapshot (no fetch or arm line in outer.log) |
| Mixtral-8x7B-Instruct-v0.1 | not_tested — not reached at the snapshot (no fetch or arm line in outer.log) | not_tested — not reached at the snapshot (no fetch or arm line in outer.log) | not_tested — not reached at the snapshot (no fetch or arm line in outer.log) | n/a (NF4 checkpoint) | fused — (NOT_RUN); batched — (NOT_RUN) | — / — / — | — / — / — | — / — / — | pending (no receipt at this snapshot) | reference: NOT_RUN — not reached at the snapshot (no fetch or arm line in outer.log); fused: NOT_RUN — not reached at the snapshot (no fetch or arm line in outer.log); batched: NOT_RUN — not reached at the snapshot (no fetch or arm line in outer.log) |

---

## Reading it

**Rule (pre-registered, P36 §"Registered criteria", applied by the reducer):** for an OK accelerated row, VOID if the
arm's `init_sha` ≠ the reference's, C1 not bit-exact (or 0 bytes hashed, or empties skipped, or the byte-flip control
silent) in either arm, `n_patched == 0`, `kernel_calls_per_step_min < 2 × n_patched`, or a different step count; else
PASS iff both registered numbers are ≤ 0.05; else FAIL. Row statuses come from the artefacts as the reducer's
docstring lists them; nothing below is inferred from the absence of an error.

### Granite-3.1-3B-A800M (`granitemoe`) — first real-weight pass through the direct path

- **Load + verify (OK, new):** `load_moe_4bit_streaming(offload=False)` quantised **32/32** MoE layers (40 experts
  each) in 3.2 s, **2.351 GB** loaded, `verify_moe_4bit(strict=True)` 32 quantised / 0 unquantised, **32 `ExpertsLoRA`**
  over `Experts4bit` (0 bare). C1 hashed **1,698,693,120 B** per arm — exactly the 1.70 GB the pre-registration's
  arithmetic predicted — 0 empties, control fires. Until this lane the family's real weights had only been loaded
  through the serving lanes' arena path (MATRIX §1: "not_tested" for the direct `from_float` + `ExpertsLoRA` path).
- **`reference` — OK, REF.** 60 steps, loss 3.4176 → 0.2274, held-out eval 3.4311 → 0.2582; **4.266 s/step**, 23.7 tok/s
  (6,060 tokens over 60 rows — the clinical rows are ≈100 tokens, so "seq 512" is a ceiling), peak 2.810 GB, 399.5
  J/step. P7 assumed 0.5–1 s/step; the per-expert loop over 40 experts × 32 layers is launch-bound and reads 4–8× that
  (alarm sizing only; no alarm fired).
- **`fused` attempt 1 — HARNESS_ERROR (kept; amendment 3).** rc=1 in `summary.txt` (no CELL line — the next arm's
  result shares its line), no receipt JSON (the process died before writing one),
  [`logs/run_granite_fused.attempt1.log`](logs/run_granite_fused.attempt1.log): the load stage passed (32/32), then
  the first forward died with `TypeError: _dequant_whole() got an unexpected keyword argument 'weights_fn'` from the
  harness frame `tp1_train_smoke.py:234 in w` — `KernelCounter.install` (D5) wrapped `nf4_qlora.fused_grouped_lora`
  and `batched._dequant_whole` in one scope with the same closure variable `orig`, so late binding sent the fused
  kernel's call to `_dequant_whole`. Harness, not shipped code: `enable_fast_train` had patched the layer and was
  calling the kernel. Patched on the box before OLMoE's fused arm ([`logs/tp1_closure_patch.py`](logs/tp1_closure_patch.py)).
  **The corrected-counter re-run by [`logs/tp1b.sh`](logs/tp1b.sh) (same box, install, fixture; after `TP_DONE`) is the
  row that counts**; it lands as attempt 2 of this arm, and the reducer labels it from the `tp1b:` marker in
  `outer.log`. Register: `e4b.train.parity.tp1.granite.fused.attempt1.2026-09-05` (this row),
  `e4b.train.parity.tp1.granite.fused.2026-09-05` (the re-run; NOT_RUN at the snapshot).
- **`batched` — OK, PASS.** 32/32 patched, whole-stack dequant calls per step 132–174 (min 132 ≥ 2 × 32 = 64: the
  kernel reached every layer on every step), `init_sha` equal to the reference's (`9da33dcfb11c4259…`), C1 clean in
  both arms. **|Δ final train loss| 0.01553, median step-wise |Δ| 0.01681** — both inside the 0.05 band. Cost on this
  box: **×3.93 faster** per step (1.086 vs 4.266 s), peak ×1.057 (2.971 GB), **J/step ×0.303** (121.2 vs 399.5). Eval Δ
  vs reference +0.0055 (not the band). `granitemoe.batched_train: supported` in the capability contract;
  `fast_train` waits for the re-run.

### OLMoE-1B-7B-0924-Instruct (`olmoe`) — first parity reading on a registered text with real weights

- **Load + verify (OK):** 16/16 (64 experts each), 4.695 GB (the register's `e4b.train.olmoe-fits` measured 4.70 GB on
  the base checkpoint), 16 `ExpertsLoRA`, C1 3,623,878,656 B hashed per arm, control fires.
- **`reference` — OK, REF.** loss 4.1678 → 0.2995, eval 4.0130 → 0.3241; 3.022 s/step, 27.9 tok/s, peak 5.234 GB,
  294.7 J/step.
- **`fused` — OK, PASS.** `enable_fast_train(dgrad=True)` patched 16/16; `fused_grouped_lora` called **64 times on every
  step** (≥ 32); `init_sha` equal (`b14ec24782b3eb8a…`); C1 clean. **|Δ final| 0.01327, median 0.01249.** Cost: **×3.22
  faster** (0.939 vs 3.022 s), peak ×1.000 (5.234 GB, identical to the reference's), **J/step ×0.339** (100.0 vs 294.7).
  Eval Δ −0.0038 (not the band). The first fused-vs-reference reading for this family on a registered text with real
  weights — the dgrad-gate receipt (`e4b.train.fast-train-dgrad`) used 8 steps of synthetic random tokens.
  `olmoe.fast_train: supported`.
- **`batched` — OK, VOID (not engaged on every layer).** 16/16 patched, C1 clean, `init_sha` equal, 60 steps — and the
  whole-stack dequant count per step ran **24–54 with a minimum of 24 < 2 × 16 = 32**. `engines/batched.py:208` returns
  the reference forward whenever `n_grp × widest > _PAD_WASTE_LIMIT × total` (4.0), silently and without a counter; on
  ≈85-token rows routed over 64 experts the padded gather exceeded that ratio on some layers on some steps, so the arm
  is a mixture of the batched kernel and the reference loop on those layers. P2 named this risk for the 128-expert
  families; it fired first on the 64-expert one. **No parity number is quoted for a VOID row** (reading rule 5; the
  arm's loss curve is in its receipt for the record). Its cost columns (×1.58, peak ×1.187 = 6.211 GB, J ×0.684)
  describe the mixed path. `olmoe.batched_train: void`.

### gpt-oss-20b (`gpt_oss`) — every row says what the code says: no expert-LoRA path

- **Load + verify (OK):** the released MXFP4 blocks/scales dequantised and rebuilt as **24 bare `GptOssExperts4bit`**
  (0 `ExpertsLoRA`; `loader.py`: "Built bare (no ExpertsLoRA): GPT-OSS-aware training LoRA is a separate change"),
  14.359 GB loaded, `verify_moe_4bit(strict=True)` 24/24 (the subclass counts as quantised — MATRIX §4 assumption 2,
  now observed).
- **`fused` — REFUSED; `batched` — REFUSED.** Probed from the `attn_only` process (D8; the stubs carry the enablers'
  own lines): `enable_fast_train(dgrad=True)` → 0 (`[e4b.fast] fused TRAINING path on 0 ExpertsLoRA module(s) (dgrad
  kernel backward)`); `enable_batched_train` → 0 (`[e4b.batched] batched training path on 0 ExpertsLoRA module(s)`).
  Two refusal rows, as P3 predicted; a build-out item, not a zero. `gpt_oss.fast_train / batched_train: refused`.
- **`attn_only` — OK, no pair.** `add_attention_lora` wrapped 96 projections (q/k/v/o × 24 layers; gpt-oss's
  `attention_bias=True` projections keep their bias inside `LoRALinear`), 3,981,312 trainable fp32 parameters in 192
  tensors; loss 5.1060 → 0.3664, eval 5.1405 → 0.3435 over 60 steps; **C1 on the bare stacks: 10,749,542,400 B hashed
  (96 tensors), 0 empties, control fires, bit-exact.** 1.802 s/step, 44.9 tok/s, peak 14.687 GB, 223.4 J/step.
  Attention-only QLoRA over frozen NF4-requantised experts; it never says "trains the experts".
  `gpt_oss.reference_train: refused` (no expert adapter; this row is the attention-only evidence).
- **`mxfp4` — EXPERIMENTAL, NOT LICENSED.** grouped-nf4-gemm's `run_mxfp4_20b_qlora` (`mxfp4_qlora.ExpertsMxfp4LoRA`:
  LoRA over the released e2m1/e8m0 bytes, faithful clamped GLU + biases, recompute-in-backward) on **its own text**
  (wikitext-2 train, its seed-41 chunking): step-0 canary vs transformers' dequant path **top-1 0.906 over 32 compared
  positions, KL 0.02495** (P4's `≥ 0.9` clause holds by 0.006); `provenance.pre_equals_post` **True** over 96 file
  tensors; train loss 5.461 → 2.622, eval 4.591 → 2.184 (@0 → @60, every 10 steps in `run_artifact.json`); mean 5.160
  s/step; peak 6.221 GB (`cuda.loaded_gb` 3.971 with 13.8 GB host RSS after the native patch — the runner's own
  residency scheme, `logs/run_gptoss_mxfp4.log`; not compared to the e4b arms' `loaded_gb`). No pair, no parity verdict,
  different text, different runner: never compared to the e4b arms, and this lane licenses nothing on it.
  `gpt_oss.native_mxfp4_train: experimental` (under the MXFP4 capability).

### Qwen3-30B-A3B (`qwen3_moe`) — load row in; three arms NOT_RUN at this snapshot

- **Load + verify (OK):** 48/48 MoE layers × 128 experts quantised in 16.6 s, **20.019 GB** loaded resident on the 32 GB
  card, `verify_moe_4bit(strict=True)` 48/0, 48 `ExpertsLoRA`. The pre-registration's arithmetic said ≈18.7 GB; the
  flagship receipts loaded this family with `offload=True` and the dgrad-gate resident on a 48 GB card, so this is the
  first resident load of the family on a 32 GB card in the register.
- **`reference` — NOT_RUN at the snapshot** (`logs/run_qwen3_reference.log`: step 20/60, loss 0.350, 12.8–13.0 s/step —
  P7 assumed 6–8 s; the 3000-s alarm holds). **`fused`, `batched` — NOT_RUN** (not started at the snapshot). The private
  record (P36 amendment 4) says all three completed on the box at 13:27Z; their receipts and rows arrive with the
  final snapshot.

### Gemma-4-26B-A4B-it — NOT_RUN at this snapshot (amendment 4); Mixtral-8x7B-Instruct-v0.1 — NOT_RUN (not reached)

No receipts at the snapshot (the lane had not reached either family). **Amendment 4 (14:15Z):** the Gemma-4 fetch
started 13:27Z on a ≥40 MB/s link, hit "The read operation timed out" on shard 1 of 2 at 13:46Z with 46 GB cached, and
was left to its 4800-s fetch alarm (no kill); the lane continues with Mixtral, and `tp1b.sh` redoes any family whose
reference receipt is missing — Gemma-4 (fetch resuming the cached blobs, then the three arms with the pre-registered
alarms) and, should its fetch fail the same way, Mixtral — before `TP2_DONE`. When the final `summary.txt` lands, the
reducer attaches amendment 4 to the Gemma-4 rows from the `gemma4: FETCH FAILED rc=142` line (ALARM for the first
attempt) and to the redo from the `tp1b:` marker. Gemma-4-it may still fault at load per #344 (P6: a `load_fault`
receipt is a NOT_RUN row with that reason, not a retry); Mixtral runs `offload=True` (P5).

### Cross-family, in one line each (nothing here is a ratio across families)

- Granite: direct real-weight load OK; batched **OK · PASS** (×3.93, J ×0.303); fused **HARNESS_ERROR** (attempt 1) →
  corrected-counter re-run pending.
- OLMoE: fused **OK · PASS** on a registered text (×3.22, J ×0.339); batched **OK · VOID** (fallback engaged, uncounted
  in the code).
- gpt-oss: fused / batched **REFUSED** (0 patched); attn_only **OK** with the frozen stacks bit-exact; mxfp4
  **EXPERIMENTAL** (canary passes, provenance holds, loss falls) — not licensed.
- Qwen3: resident load on 32 GB OK; three arms NOT_RUN at the snapshot (completed on the box, receipts pending).
  Gemma-4: NOT_RUN (amendment 4). Mixtral: NOT_RUN (not reached).

### What this table does not say

No convergence claim: 60 steps rank arms against each other on one text, they do not train a model. "PASS on one
text" is what a PASS here means (the flagship needed five). No cross-family, cross-lane or cross-card ratio: tp1's
Qwen3 and Gemma-4 rows, when they land, sit beside the flagship receipts and are never divided into them. No training
throughput position and no anchor-class projection: s/step and tok/s are this box's (train-anchor class
`pcie-full/launch-fast`), and two 5090s have differed 1.65× on one training config. Experimental ≠ licensed. A VOID
row is not a PASS however its loss curve reads. A HARNESS_ERROR row is not a shipped-code failure and not a result;
it is kept so the re-run cannot look like a first run.
