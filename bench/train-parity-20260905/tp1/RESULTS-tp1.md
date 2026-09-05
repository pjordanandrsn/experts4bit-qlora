# tp1 — training parity on real weights, results (2026-09-05, box 49937730, one RTX 5090 on an EPYC 7Q83 host)

**All 18 rows are in: `TP_DONE` 2026-09-05T15:22:26Z (the six families through the registered arms) and `TP2_DONE`
15:33Z (Granite's `fused` corrected-counter re-run, `logs/tp1b.sh`, on the same box).** The table below is the verbatim
output of `python tp1_reduce.py .` (reducer **v2**, this directory's copy; the copy that ran on the box is
[`logs/tp1_reduce.py`](logs/tp1_reduce.py), v1, whose table — [`logs/RESULTS.txt`](logs/RESULTS.txt), the box's own print
at `TP_DONE`, before the re-run — it supersedes) over the artefacts in this directory. Box, cut, fixture, the five
amendments, the verdict rule verbatim, the predictions and the layout: [`README.md`](README.md).

**Row status vocabulary (phase directive 2026-09-05 14:45Z):** every row is exactly one of **OK, REFUSED,
HARNESS_ERROR, ALARM, OOM, NOT_RUN, EXPERIMENTAL**, classified mechanically from `summary.txt` (rc per attempt), the
receipt or stub, `logs/outer.log` and the run logs — never from a missing error; a missing receipt is NOT_RUN with the
reason from `outer.log` (a fetch or arm alarm is ALARM). The **parity verdict** (PASS / FAIL / VOID) is a separate
column, computed only for OK rows of an accelerated arm, in the registered units and no others
(PREREG-flagship-matrix B2 / -model2 C2: `|Δ final train loss| ≤ 0.05` AND `median step-wise |Δ| ≤ 0.05` against the
family's own `reference` arm, same box, same session; VOID when the arm cannot be read). A VOID row carries no parity
number. Cost is reported, never gated; every ratio is within one family on this box. Eval deltas are "not the band".
Every attempt is a row: Granite's first `fused` attempt stays as HARNESS_ERROR (attempt 1/2) beside its
corrected-counter re-run (attempt 2/2), the launcher start that aborted between them is listed in the re-run's reason,
and the amendment that touched a row is named in its last column.

---

# tp1 — training parity table (`tp1`, reducer v2)
Row status ∈ {OK, REFUSED, HARNESS_ERROR, ALARM, OOM, NOT_RUN, EXPERIMENTAL}, classified from summary.txt (rc per attempt), the receipt / stub, outer.log and the run logs — never from a missing error. Verdict (OK accelerated rows only): PASS iff |Δ final train loss| ≤ 0.05 AND median step-wise |Δ| ≤ 0.05 vs the family's reference arm (PREREG-flagship-matrix B2 / -model2 C2); VOID when the arm cannot be read (init_sha, C1, n_patched, kernel engagement, steps). Cost is reported, not gated. Eval deltas are not the band. No cross-family, cross-lane or cross-card ratio appears here.
- lane header (summary.txt): ANCHOR rc=0 class=pcie-full/launch-fast · DATASET clinical sha=76fb9036de80f3bb495fe4c8894159fcb1d399d2437293e012e264d81949f791
- anchor.json: OK class per summary; flops median 184.36 TFLOP/s, launch 186947 /s, h2d 25.93 GB/s
- markers shipped: none; tp1b marker in outer.log: yes (line 214); lane end lines: GRANITE DONE, OLMOE DONE, GPTOSS DONE, QWEN3 DONE, GEMMA4 DONE, MIXTRAL DONE, 2026-09-05T15:22:26Z TP_DONE, GRANITE FUSED RERUN DONE, 2026-09-05T15:33:13Z TP1B DONE
- lane-level amendments: 1 (pre-flight floor 40 MB/s, fetch alarms doubled — logs/tp1_run.sh vs logs/tp1_run.sh.pre-amend, both shipped); 2 (helper fetch by archive tarball after the fetch-by-sha false start — logs/outer.attempt1.log shipped); 3 and 4 are attached to their rows below.

### Granite-3.1-3B-A800M (`granite`)
- load: status=ok model_type=granitemoe load_s=3.2 loaded_gb=2.351 verify={'n_quantized': 32, 'n_unquantized': 0} wrapped=32 bare=0 classes={'Experts4bit': 32} offload=False geometry(H/I/L/E/k)=1536/512/32/40/8
- env: e4b 0.35.0 gnf4 0.30.0 torch 2.8.0+cu128 transformers 5.16.1 gpu NVIDIA GeForce RTX 5090 box_class pcie-full/launch-fast host_mem_gib 251.5
| arm | attempt | status | verdict | n_patched | kcalls/step min (need) | C1 | steps | s/step | tok/s | peak GB | J/step | loss first→last | eval 0→final (not the band) | Δ final train | median step \|Δ\| | reason / amendment |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reference | 1/1 | **OK** | **REF** | 0 | 0 (0) | ok | 60 | 4.266 | 23.7 | 2.810 | 399.5 | 3.4176→0.2274 | 3.4311→0.2582 | — | — | — |
| fused | 1/2 | **HARNESS_ERROR** | — | — | — (0) | — | — | — | — | — | — | —→— | —→— | — | — | rc=1; TypeError: _dequant_whole() got an unexpected keyword argument 'weights_fn' (innermost frame tp1_train_smoke.py, run_granite_fused.attempt1.log); no CELL line in summary.txt (the next arm's result shares its line) — amendment 3 (late-binding counter closure in the harness; patched in flight; re-run by tp1b) |
| fused | 2/2 | **OK** | **PASS** (×5.86 faster, peak ×1.000, J ×0.197) | 32 | 128 (64) | ok | 60 | 0.728 | 138.7 | 2.810 | 78.6 | 3.4198→0.2407 | 3.4264→0.2631 (Δ vs ref +0.0050) | 0.01329 | 0.01270 | 1 earlier start line(s) without a result line at 2026-09-05T11:44:40Z (a launcher abort before the harness ran; outer.log) — amendment 3: the corrected-counter re-run (tp1b) -- the row that counts |
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
| reference | 1/1 | **OK** | **REF** | 0 | 0 (0) | ok | 60 | 12.797 | 6.7 | 22.716 | 1219.8 | 3.6093→0.2732 | 3.7459→0.2859 | — | — | — |
| fused | 1/1 | **OK** | **PASS** (×2.56 faster, peak ×1.000, J ×0.407) | 48 | 192 (96) | ok | 60 | 5.003 | 17.2 | 22.715 | 496.2 | 3.6185→0.2600 | 3.7362→0.2884 (Δ vs ref +0.0024) | 0.01315 | 0.01050 | — |
| batched | 1/1 | **OK** | **VOID** (×1.08 faster, peak ×1.040, J ×0.915) | 48 | 12 (96) | ok | 60 | 11.885 | 7.3 | 23.629 | 1116.0 | 3.6460→0.2677 | 3.7444→0.2915 (Δ vs ref +0.0056) | — | — | kernel calls/step min 12 < 2*n_patched=96 (not engaged on every layer) |
- init_sha (reference) `2b362f7b19ad8830`; dataset `ds_clinical.json` sha `76fb9036de80`; trainable 321257472 params in 576 tensors

### Gemma-4-26B-A4B-it (`gemma4`)
- load: status=ok model_type=gemma4 load_s=14.7 loaded_gb=18.113 verify={'n_quantized': 30, 'n_unquantized': 0} wrapped=30 bare=0 classes={'Experts4bit': 30} offload=False geometry(H/I/L/E/k)=2816/704/30/128/None
- env: e4b 0.35.0 gnf4 0.30.0 torch 2.8.0+cu128 transformers 5.16.1 gpu NVIDIA GeForce RTX 5090 box_class pcie-full/launch-fast host_mem_gib 251.5
| arm | attempt | status | verdict | n_patched | kcalls/step min (need) | C1 | steps | s/step | tok/s | peak GB | J/step | loss first→last | eval 0→final (not the band) | Δ final train | median step \|Δ\| | reason / amendment |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reference | 1/1 | **OK** | **REF** | 0 | 0 (0) | ok | 60 | 7.344 | 12.3 | 20.179 | 704.8 | 8.2431→0.2879 | 8.7561→0.3022 | — | — | — |
| fused | 1/1 | **OK** | **PASS** (×2.37 faster, peak ×1.000, J ×0.477) | 30 | 120 (60) | ok | 60 | 3.101 | 29.1 | 20.179 | 336.1 | 8.3132→0.2640 | 7.7324→0.3044 (Δ vs ref +0.0022) | 0.02385 | 0.04742 | — |
| batched | 1/1 | **OK** | **VOID** (×1.01 faster, peak ×1.072, J ×0.999) | 30 | 0 (60) | ok | 60 | 7.259 | 12.4 | 21.623 | 703.9 | 8.2350→0.2773 | 8.8105→0.2944 (Δ vs ref -0.0078) | — | — | kernel calls/step min 0 < 2*n_patched=60 (not engaged on every layer) |
- init_sha (reference) `a059fa05b41f1c32`; dataset `ds_clinical.json` sha `76fb9036de80`; trainable 242606080 params in 320 tensors

### Mixtral-8x7B-Instruct-v0.1 (`mixtral`)
- load: status=ok model_type=mixtral load_s=50.9 loaded_gb=3.421 verify={'n_quantized': 32, 'n_unquantized': 0} wrapped=32 bare=0 classes={'Experts4bit': 32} offload=True geometry(H/I/L/E/k)=4096/14336/32/8/2
- env: e4b 0.35.0 gnf4 0.30.0 torch 2.8.0+cu128 transformers 5.16.1 gpu NVIDIA GeForce RTX 5090 box_class pcie-full/launch-fast host_mem_gib 251.5
| arm | attempt | status | verdict | n_patched | kcalls/step min (need) | C1 | steps | s/step | tok/s | peak GB | J/step | loss first→last | eval 0→final (not the band) | Δ final train | median step \|Δ\| | reason / amendment |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reference | 1/1 | **OK** | **REF** | 0 | 0 (0) | ok | 60 | 3.073 | 32.8 | 10.562 | 482.6 | 3.0157→0.2170 | 2.9904→0.2501 | — | — | — |
| fused | 1/1 | **OK** | **PASS** (×1.26 faster, peak ×0.494, J ×0.734) | 32 | 128 (64) | ok | 60 | 2.429 | 41.5 | 5.215 | 354.4 | 3.0192→0.2266 | 2.9898→0.2508 (Δ vs ref +0.0007) | 0.00953 | 0.00945 | — |
| batched | 1/1 | **OK** | **PASS** (×1.27 faster, peak ×0.802, J ×0.976) | 32 | 192 (64) | ok | 60 | 2.412 | 41.8 | 8.466 | 471.1 | 3.0165→0.2247 | 2.9894→0.2572 (Δ vs ref +0.0072) | 0.00766 | 0.01057 | — |
- init_sha (reference) `46ea74b4d8860671`; dataset `ds_clinical.json` sha `76fb9036de80`; trainable 111673344 params in 384 tensors

## Per-family matrix (the directive's ten columns; support words in the capability vocabulary, read from the LAST attempt of each arm)
| family | reference support | fused support | batched support | native-format route | loss-parity result | s/step (ref / fused / batched) | peak GB (ref / fused / batched) | tok/s (ref / fused / batched) | evidence tier | limitations / refusal reason |
|---|---|---|---|---|---|---|---|---|---|---|
| Granite-3.1-3B-A800M | supported (OK) | supported (PASS) | supported (PASS) | n/a (NF4 checkpoint) | fused PASS 0.0133 / 0.0127; batched PASS 0.0155 / 0.0168 | 4.266 / 0.728 / 1.086 | 2.810 / 2.810 / 2.971 | 23.7 / 138.7 / 93.0 | measured (receipts in this directory: reference, fused, batched) | fused: 2 attempts (1: HARNESS_ERROR [amendment 3 (late-binding counter closure in the harness; patched in flight; re-run by tp1b)]; 2: OK [amendment 3: the corrected-counter re-run (tp1b) -- the row that counts]) |
| OLMoE-1B-7B-0924-Instruct | supported (OK) | supported (PASS) | void (VOID) | n/a (NF4 checkpoint) | fused PASS 0.0133 / 0.0125; batched VOID | 3.022 / 0.939 / 1.911 | 5.234 / 5.234 / 6.211 | 27.9 / 89.9 / 44.2 | measured (receipts in this directory: reference, fused, batched) | batched: OK/VOID |
| gpt-oss-20b | refused — the loader builds the experts bare (no ExpertsLoRA); attention-only QLoRA over them: attn_only OK (no pair) | refused — 0 patched: [e4b.fast] fused TRAINING path on 0 ExpertsLoRA module(s) (dgrad kernel backward) (stub written by the probing arm, D8) | refused — 0 patched: [e4b.batched] batched training path on 0 ExpertsLoRA module(s) (stub written by the probing arm, D8) | experimental — grouped-nf4-gemm run_mxfp4_20b_qlora on its own text; never licensed | fused — (REFUSED); batched — (REFUSED) | 1.802 / — / — (ref = attn_only) | 14.687 / — / — | 44.9 / — / — | measured (receipts in this directory: fused, batched, attn_only, mxfp4); mxfp4 experimental, not licensed | fused: REFUSED — 0 patched: [e4b.fast] fused TRAINING path on 0 ExpertsLoRA module(s) (dgrad kernel backward) (stub written by the probing arm, D8); batched: REFUSED — 0 patched: [e4b.batched] batched training path on 0 ExpertsLoRA module(s) (stub written by the probing arm, D8); mxfp4: EXPERIMENTAL — grouped-nf4-gemm run_mxfp4_20b_qlora on its own text; never licensed |
| Qwen3-30B-A3B | supported (OK) | supported (PASS) | void (VOID) | n/a (NF4 checkpoint) | fused PASS 0.0131 / 0.0105; batched VOID | 12.797 / 5.003 / 11.885 | 22.716 / 22.715 / 23.629 | 6.7 / 17.2 / 7.3 | measured (receipts in this directory: reference, fused, batched) | batched: OK/VOID |
| Gemma-4-26B-A4B-it | supported (OK) | supported (PASS) | void (VOID) | n/a (NF4 checkpoint) | fused PASS 0.0239 / 0.0474; batched VOID | 7.344 / 3.101 / 7.259 | 20.179 / 20.179 / 21.623 | 12.3 / 29.1 / 12.4 | measured (receipts in this directory: reference, fused, batched) | batched: OK/VOID |
| Mixtral-8x7B-Instruct-v0.1 | supported (OK) | supported (PASS) | supported (PASS) | n/a (NF4 checkpoint) | fused PASS 0.0095 / 0.0095; batched PASS 0.0077 / 0.0106 | 3.073 / 2.429 / 2.412 | 10.562 / 5.215 / 8.466 | 32.8 / 41.5 / 41.8 | measured (receipts in this directory: reference, fused, batched) | — |

---

## Reading it

**Rule (pre-registered, P36 §"Registered criteria", applied by the reducer):** for an OK accelerated row, VOID if the
arm's `init_sha` ≠ the reference's, C1 not bit-exact (or 0 bytes hashed, or empties skipped, or the byte-flip control
silent) in either arm, `n_patched == 0`, `kernel_calls_per_step_min < 2 × n_patched`, or a different step count; else
PASS iff both registered numbers are ≤ 0.05; else FAIL. Row statuses come from the artefacts as the reducer's
docstring lists them; nothing below is inferred from the absence of an error. C1 held in every OK arm of every family
(bytes hashed: Granite 1,698,693,120; OLMoE 3,623,878,656; gpt-oss 10,749,542,400; Qwen3 16,307,453,952; Gemma-4
12,846,366,720; Mixtral 25,367,150,592 — each exactly the pre-registration's NF4 arithmetic — 0 empties, control fires),
and `init_sha` matched the reference's in every accelerated arm.

### Granite-3.1-3B-A800M (`granitemoe`) — first real-weight pass through the direct path; both accelerated arms PASS

- **Load + verify (OK, new):** 32/32 MoE layers (40 experts each) in 3.2 s, **2.351 GB**, `verify_moe_4bit(strict=True)`
  32/0, **32 `ExpertsLoRA`** (0 bare). Until this lane the family's real weights had only been loaded through the
  serving lanes' arena path.
- **`reference` — OK, REF.** loss 3.4176 → 0.2274, eval 3.4311 → 0.2582; **4.266 s/step**, 23.7 tok/s (6,060 tokens),
  peak 2.810 GB, 399.5 J/step. P7 assumed 0.5–1 s/step; the per-expert loop over 40 experts × 32 layers is
  launch-bound and reads 4–8× that.
- **`fused` attempt 1/2 — HARNESS_ERROR (kept; amendment 3).** rc=1 in `summary.txt` (no CELL line — the next arm's
  result shares its line), no receipt JSON, [`logs/run_granite_fused.attempt1.log`](logs/run_granite_fused.attempt1.log):
  load passed (32/32), then the first forward died with `TypeError: _dequant_whole() got an unexpected keyword argument
  'weights_fn'` from the harness frame `tp1_train_smoke.py:234 in w` — the kernel-call counter's late-binding closure
  (D5), not the shipped code. Register: `e4b.train.parity.tp1.granite.fused.attempt1.2026-09-05`.
- **`fused` attempt 2/2 — OK, PASS (the corrected-counter re-run, `logs/tp1b.sh`, the row that counts).** Started
  15:31:49Z after `TP_DONE` on the same box, install and fixture (Granite re-fetched from the cache in 0.0 min); the
  reducer labels it from the `tp1b:` marker and lists, in its reason, the one earlier start line without a result —
  15:30:43Z, the follow-up script's own `DATA: unbound variable` abort before the harness ran (amendment 5, not a lane
  arm). 32/32 patched, `fused_grouped_lora` **128 calls every step** (≥ 64), `init_sha` equal (`9da33dcfb11c4259…`), C1
  clean (1,698,693,120 B, bit-exact). **|Δ final| 0.01329, median 0.01270** — both inside the band. Cost on this box:
  **×5.86 faster** (0.728 vs 4.266 s), peak ×1.000 (2.810 GB), **J ×0.197** (78.6 vs 399.5); 138.7 tok/s; loss 3.4198 →
  0.2407, eval 3.4265 → 0.2631 (Δ +0.0050, not the band). Register: `e4b.train.parity.tp1.granite.fused.2026-09-05`.
  `granitemoe.fast_train: supported` → **`granitemoe` enters `model_families`**.
- **`batched` — OK, PASS.** 32/32 patched, dequant calls/step 132–174 (min 132 ≥ 64), `init_sha` equal, C1 clean.
  **|Δ final| 0.01553, median 0.01681.** **×3.93 faster** (1.086 vs 4.266 s), peak ×1.057 (2.971 GB), **J ×0.303**.
  `granitemoe.batched_train: supported`. At this family's width the batched path is ×3.93 over the per-expert loop and
  the fused path ×5.86 — the register's "no speed-up at real width" is a 30B-width statement.

### OLMoE-1B-7B-0924-Instruct (`olmoe`) — fused PASS on a registered text; batched VOID

- **Load + verify (OK):** 16/16, 4.695 GB, 16 `ExpertsLoRA`. **`reference` — OK, REF:** 3.022 s/step, peak 5.234 GB.
- **`fused` — OK, PASS.** 16/16, `fused_grouped_lora` 64 calls every step (≥ 32). **|Δ final| 0.01327, median 0.01249.**
  ×3.22 (0.939 s), peak ×1.000, J ×0.339. First fused-vs-reference reading on a registered text with real weights for
  this family (the dgrad-gate used synthetic tokens). `olmoe.fast_train: supported`.
- **`batched` — OK, VOID.** 16/16 patched, calls/step 24–54, min 24 < 32: `engines/batched.py:208` returned the
  reference forward per call above `_PAD_WASTE_LIMIT` (4.0) on some layers on some steps, silently and uncounted in the
  code. No parity number for a VOID row. Cost of the mixed path: ×1.58, peak ×1.187 (6.211 GB), J ×0.684.
  `olmoe.batched_train: void`.

### gpt-oss-20b (`gpt_oss`) — no expert-LoRA path; attention-only trains; MXFP4 route experimental

- **Load + verify (OK):** 24 bare `GptOssExperts4bit` (0 `ExpertsLoRA`), 14.359 GB, 24/24.
- **`fused` — REFUSED; `batched` — REFUSED** (probed from `attn_only`, D8; 0 patched, the enablers' own lines). P3.
- **`attn_only` — OK, no pair.** 96 `LoRALinear` (biased projections keep their bias), 3,981,312 params; loss 5.1060 →
  0.3664, eval 5.1405 → 0.3435; C1 on the bare stacks bit-exact; 1.802 s/step, 44.9 tok/s, peak 14.687 GB, 223.4 J/step.
- **`mxfp4` — EXPERIMENTAL, NOT LICENSED.** grouped-nf4-gemm's `run_mxfp4_20b_qlora` on its own text: canary top-1
  0.906 / KL 0.02495 vs transformers' dequant path, `pre_equals_post` True over 96 tensors, train loss 5.461 → 2.622,
  eval 4.591 → 2.184, 5.160 s/step, peak 6.221 GB. Never compared to the e4b arms.

### Qwen3-30B-A3B (`qwen3_moe`) — resident on 32 GB; fused PASS; batched VOID

- **Load + verify (OK):** 48/48 × 128 experts in 16.6 s, **20.019 GB resident** (the flagship ran this family with
  `offload=True` on a 4090; the dgrad-gate resident on a 48 GB A6000), 48 `ExpertsLoRA`; trainable 321,257,472 params in
  576 tensors (the `lora.py` docstring's count); C1 16,307,453,952 B — the fused-train-gate's 16.31 GB, hashed again.
- **`reference` — OK, REF.** loss 3.6093 → 0.2732, eval 3.7459 → 0.2859; **12.797 s/step**, 6.7 tok/s (5,175 tokens),
  peak 22.716 GB, 1219.8 J/step (P7 assumed 6–8 s; the 3000-s alarm held).
- **`fused` — OK, PASS.** 48/48 patched, `fused_grouped_lora` **192 calls every step** (≥ 96), `init_sha` equal
  (`2b362f7b19ad8830…`). **|Δ final| 0.01315, median 0.01050.** **×2.56 faster** (5.003 vs 12.797 s), peak ×1.000
  (22.715 GB), **J ×0.407** (496.2 vs 1219.8). Eval Δ +0.0024 (not the band). Sits beside the flagship's five-dataset PASS
  and the dgrad-gate's ×2.52 on an A6000 — a different card, never divided into them. `qwen3_moe.fast_train: supported`.
- **`batched` — OK, VOID.** 48/48 patched, calls/step **12–30 against 96 required** (median 21): on ≈86-token rows
  routed over 128 experts the `_PAD_WASTE_LIMIT` fallback took most layers on every step — exactly P2's stated risk.
  Cost of the mixed path: ×1.08 (11.885 s), peak ×1.040 (23.629 GB, the highest), J ×0.915. `qwen3_moe.batched_train:
  void` (the dgrad-gate's 20-step Alpaca trajectory, where the kernel engaged, stands as measured on its own fixture).

### Gemma-4-26B-A4B-it (`gemma4_text`) — loaded without #344; fused PASS (the tightest cell); batched VOID

- **Load + verify (OK):** the `-it` checkpoint loaded on this host in 14.7 s with **no #344 fault** (P6: the
  load-succeeds branch), 30/30 × 128 experts, **18.113 GB resident**, 30 `ExpertsLoRA`; trainable 242,606,080 in 320
  tensors; C1 12,846,366,720 B — the model-2 flagship's byte count, hashed again. The config prints no `num_experts_per_tok`
  (the geometry line shows `k=None`; top-8 per `arch/moe_conventions.py`).
- **`reference` — OK, REF.** loss 8.2431 → 0.2879, eval 8.7561 → 0.3022; **7.344 s/step**, 12.3 tok/s (5,421 tokens),
  peak 20.179 GB, 704.8 J/step.
- **`fused` — OK, PASS.** 30/30, **120 calls every step** (≥ 60), `init_sha` equal (`a059fa05b41f1c32…`). **|Δ final|
  0.02385, median step-wise 0.04742 — inside the 0.05 band by 0.0026, the tightest cell of the lane** (the model-2
  flagship's worst cell was 0.03653 on the final loss; this family's step-wise curves diverge more than any other's
  here, on one text, 60 steps — a PASS as registered, read with that margin). **×2.37 faster** (3.101 vs 7.344 s), peak
  ×1.000 (20.179 GB), **J ×0.477** (336.1 vs 704.8). Eval Δ +0.0022 (not the band). `gemma4_text.fast_train: supported`.
- **`batched` — OK, VOID.** 30/30 patched, calls/step **0–12 against 60 required** (median 6; **9 of 60 steps had zero
  calls** — the fallback took every layer on those steps). ×1.01 (7.259 s), peak ×1.072 (21.623 GB), J ×0.999: a
  reference re-run wearing a batched label, as P2 predicted for this family. `gemma4_text.batched_train: void`.

### Mixtral-8x7B-Instruct-v0.1 (`mixtral`) — first real-weight pass through the `w1/w3/w2` fusion, under offload; both arms PASS

- **Load + verify (OK, new):** `offload=True` — 32/32 layers × 8 experts fused gate-first from the per-expert
  `w1/w3/w2` tensors into `Experts4bit` in 50.9 s, **3.421 GB on the GPU**, the 32 layers' NF4 experts pinned in host
  RAM and streamed one layer at a time (`run_mixtral_reference.log`); 32 `ExpertsLoRA`; trainable 111,673,344 in 384
  tensors; C1 25,367,150,592 B from the CPU home — 25.37 GB, the pre-registration's arithmetic to the byte.
- **`reference` — OK, REF.** loss 3.0157 → 0.2170, eval 2.9904 → 0.2501; **3.073 s/step**, 32.8 tok/s (6,044 tokens),
  peak 10.562 GB, 482.6 J/step (P7 assumed 8–12 s: ~3× under).
- **`fused` — OK, PASS.** 32/32, **128 calls every step** (≥ 64), `init_sha` equal (`46ea74b4d8860671…`). **|Δ final|
  0.00953, median 0.00945.** ×1.26 (2.429 s), **peak ×0.494 (5.215 GB)** — under offload the fused path never holds
  the reference loop's per-expert dequant intermediates — J ×0.734 (354.4). Eval Δ +0.0007. `mixtral.fast_train:
  supported` → **`mixtral` enters `model_families`** (the first family to enter on a tp1 row).
- **`batched` — OK, PASS.** 32/32, **192 calls every step** (≥ 64): the 8-expert / top-2 shape never crosses
  `_PAD_WASTE_LIMIT`, the one family where the batched path engaged everywhere. **|Δ final| 0.00766, median 0.01057.**
  ×1.27 (2.412 s), peak ×0.802 (8.466 GB), J ×0.976. Eval Δ +0.0072. `mixtral.batched_train: supported`.

### Cross-family, in one line each (nothing here is a ratio across families)

- Granite: fused **OK · PASS** on the corrected-counter re-run (×5.86, J ×0.197; attempt 1 a kept HARNESS_ERROR);
  batched **OK · PASS** (×3.93, J ×0.303).
- OLMoE: fused **OK · PASS** (×3.22, J ×0.339); batched **OK · VOID**.
- gpt-oss: fused / batched **REFUSED**; attn_only **OK**; mxfp4 **EXPERIMENTAL** — not licensed.
- Qwen3: fused **OK · PASS** (×2.56, J ×0.407, resident on 32 GB); batched **OK · VOID**.
- Gemma-4-it: loaded (no #344); fused **OK · PASS** (×2.37, J ×0.477; median 0.04742, the tightest); batched **OK · VOID**.
- Mixtral (offload): fused **OK · PASS** (×1.26, peak ×0.494); batched **OK · PASS** (×1.27, peak ×0.802).

**The batched path's engagement envelope, read across the lane:** `enable_batched_train` reached every layer on every
step only where the expert count is small (Mixtral, 8 experts; Granite, 40) and fell back — silently, in the code —
on 64 experts (OLMoE) and on both 128-expert families, on ≈100-token rows. Three VOID rows, one finding: assert
kernel engagement, not the patch count (`docs/capabilities.json`, `qlora-fused-moe-experts` limitations); the release
that carries this bundle adds a counter for exactly this fallback (`batched_fallback_stats`, #402).

**The fused path, read across the lane:** PASS on all five families that have one — Granite ×5.86, OLMoE ×3.22, Qwen3
×2.56, Gemma-4 ×2.37, Mixtral ×1.26 (offload) per step over the per-expert loop, each on its own box and never divided
into another — with the frozen stacks bit-exact and the kernel reached on every layer of every step.

### What this table does not say

No convergence claim: 60 steps rank arms against each other on one text. "PASS on one text" is what a PASS here means
(the flagship needed five; Gemma-4's median sits 0.0026 inside the band). No cross-family, cross-lane or cross-card
ratio: tp1's Qwen3 and Gemma-4 rows sit beside `bench/flagship-matrix` / `-model2` (a 4090, `offload=True`) and the
dgrad-gate (an A6000) and are never divided into them. No training throughput position and no anchor-class
projection: s/step and tok/s are this box's (train-anchor class `pcie-full/launch-fast`); two 5090s have differed 1.65×
on one training config. Experimental ≠ licensed. A VOID row is not a PASS however its loss curve reads. A
HARNESS_ERROR row is not a shipped-code failure and not a result; it is kept so the re-run cannot look like a first run,
and the launcher abort between the two attempts (amendment 5) is listed in the re-run's reason for the same reason.
