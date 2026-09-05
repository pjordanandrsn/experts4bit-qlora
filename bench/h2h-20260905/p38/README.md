# Head-to-head lane p38 — 2026-09-05, one RTX 5090: e4b vs Unsloth, QLoRA end-to-end on ONE identical training problem

**Status: complete — `TP_DONE` 2026-09-05T15:57:41Z (the lane's own run: no e4b row and no comparator, amendments 1–2),
`TP2_DONE` 16:00Z (`p38b.sh`: the rebuilt Unsloth venv, still no comparator), `TP3_DONE` 16:40:39Z (`p38c.sh`: the
eight registered arms in the registered order — five e4b rows and the Unsloth 200-step row VALID, the two Unsloth
60-step arms dead at load, amendment 3), `TP4_DONE` 16:52:00Z (`p38d.sh`: the two Unsloth 60-step arms), `U8_DONE`
16:53:04Z (the amendment-4 proof) and `TP5_DONE` 16:55:45Z (re-reduced under amendment 4: all eight rows VALID).**
Eight registered arms, eight VALID rows; every earlier attempt is a line in [`summary.txt`](summary.txt) and a stub in
[`attempts/`](attempts/) or an error tail in [`outer.log`](outer.log). No arm alarm fired, no OOM, no refusal, no C1
failure. Four amendments, all environment or instrument; **no workload, fixture, threshold or knob changed.**

**What was compared.** The same training problem through two frameworks on one box, one session: `Qwen/Qwen3-30B-A3B`
at revision `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` (bf16, the serving lanes' bytes; both frameworks quantise from
these bytes on load), the registered `clinical` set tokenised **once** into [`tokens_clinical.json`](tokens_clinical.json)
(sha256 `81dc24c3c195667fd7d0829a5f73f401e91708bd911ef21cb89794515cd1e583`; every arm of either framework loads that
file, asserts its sha and trains on those ids), sequences truncated at 512, LoRA r 8 / α 16 on attention q/k/v/o and on
every expert's `gate_up_proj` / `down_proj` in all 48 layers with the router frozen (321,257,472 trainable parameters,
asserted in every arm), `torch.optim.AdamW(lr=1e-4)` with torch defaults, batch 1, grad-accum 1, loss over all tokens,
`torch.manual_seed(0)` before the load, N = 60 steps for the position and N = 200 for the loss curve, held-out eval =
the first 48 rows of the 200-row split at step 0 and every 20 steps (evals outside the timed window). e4b runs in the
image python — **experts4bit-qlora 0.35.0 + grouped-nf4-gemm 0.30.0** (main @`f4b639fd2640` and @`ddcb850e05c3`, the
tp1 cut; transformers 5.16.1, bitsandbytes 0.50.1, torch 2.8.0+cu128, triton 3.4.0). Unsloth runs in its own venv —
**unsloth 2026.9.2 + unsloth_zoo 2026.9.1** (transformers 5.5.0, bitsandbytes 0.50.2, peft 0.20.0, torch 2.8.0+cu128,
triton 3.4.0; MoE backend `native_torch`, separated LoRA; `is_transformers_v5_moe_quantization_available()` asserted
True by the tripwire). **The transformers / peft version difference between the two pythons is a recorded environment
difference, not a workload difference** (amendment 1). Every version is in [`versions.txt`](versions.txt) and in each
receipt's `env`.

**Box:** Vast.ai instance **49975389**, RTX 5090 (sm_120, driver 595.84, 32607 MiB, power limit 600 W, PCIe gen 5 ×16)
on an Intel Core i9-14900K host (32 threads, 188.6 GiB; [`forensics.txt`](forensics.txt) is the lane script's probe at
install; each receipt's `env.host` carries the GPU UUID, bus id, VBIOS and the instance id). **Train-anchor class
`pcie-full/launch-fast`** ([`anchor.json`](anchor.json), gate log [`logs/anchor_gate.log`](logs/anchor_gate.log): FLOPs
188.4 TFLOP/s = 1.017× the reference band, self-consistency 1.0012; launch 451,198/s; H2D 23.66 GB/s; `BOX ACCEPTED`).
A training receipt without its class is not a receipt — two 5090s ran an identical training config 1.65× apart
([`../../train-anchor/README.md`](../../train-anchor/README.md)) — so every absolute on this page is *this box's* and
every ratio is between two arms on this box in this session. Image `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`.

**When (box time = UTC, from [`outer.log`](outer.log)):** the lane launched 15:44Z; e4b installed 15:46:50Z (`pip rc=0`;
tripwire `p38 tripwire OK (e4b): 0.35.0 gnf4 0.30.0 torch 2.8.0+cu128 triton 3.4.0 bnb 0.50.1`; `cuda ok`); the Unsloth
venv install failed 15:47:16Z (`ResolutionImpossible`, amendment 1); helpers fetched as the repository's archive
tarball at the cut 15:47:19Z; train anchor 15:47:27Z → `BOX ACCEPTED`; dataset regenerated and sha-verified 15:47:30Z
(`DATASET clinical sha=76fb9036de80…`); Qwen3 fetched at the pinned revision (staged in 9.7 min); tokenised once
15:57:11Z (`TOKENS n_train=1200 n_eval=48 train_tokens=103298 mean=86.1 sha=81dc24c3…`); 15:57:13Z–15:57:41Z every e4b
arm died at load (`LocalEntryNotFoundError`, amendment 2) and the three Unsloth arms wrote `install_failed` stubs;
`TP_DONE` 15:57:41Z with **no e4b row and no comparator**. `p38b.sh` (amendment 1) rebuilt the venv 15:58:51Z (`pip rc=0`)
and its tripwire failed on `torchao` (amendment 2) → `P38B: NO COMPARATOR`, `TP2_DONE`. `p38c.sh` (amendment 2) ran the
eight registered arms in the registered order from 16:03:44Z: e4b `reference_attn4` 16:03:44Z → `fused_attn4` 16:11:50Z
→ `fused` 16:14:57Z → `fused_attn4_nosamp` 16:17:48Z → Unsloth `ckpt_unsloth` 16:19:24Z and `ckpt_hf` 16:19:33Z (both
dead at load inside the Unsloth venv, amendment 3) → e4b `fused_attn4_200` 16:19:41Z → Unsloth `ckpt_unsloth_200`
16:27:25Z (on the amendment-3 harness: the engagement banner, `Unsloth: MoE bnb4bit using dequantize-plus-native_torch
loop`, `CELL OK`) → reduced 16:40:39Z, `TP3_DONE`. `p38d.sh` (amendment 3) re-ran `ckpt_unsloth` 16:42:09Z and `ckpt_hf`
16:47:05Z (the latter on the amendment-4 harness) → `TP4_DONE` 16:52:00Z. The amendment-4 proof ran 16:52:47Z →
`U8_DONE` 16:53:04Z; the final reduction under amendment 4 (`--u8-proof`) 16:55:45Z → `TP5_DONE`. Box torn down 16:57Z
(proven; the guard killed). **Wall 15:44Z → 16:56Z ≈ 1 h 12 min** against the pre-registration's ≈ 2.5–3 h estimate;
**cost ≈ 1.3 h of box.** The 10-h hard-kill guard on the mini did not fire.

## The identical training problem (every line applies to BOTH arms unless it says otherwise)

The pre-registration is [`PREREG.md`](PREREG.md), verbatim (the private `INT4B16/h2h/P38-UNSLOTH-PREREG.md`); this is
its fixture as it ran, with what the receipts record.

- **Checkpoint:** `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`, fetched once with the revision
  pinned; both frameworks quantise from the same snapshot bytes (`HF_HUB_OFFLINE=1` for every arm). Not the
  pre-quantised `unsloth/Qwen3-30B-A3B-bnb-4bit`, not the `unsloth/Qwen3-30B-A3B` mirror.
- **Text:** the registered `clinical` set (`bench/flagship-matrix/drivers/n9_datasets.py`, seed 1000, 1,200 train /
  200 held-out rows; sha256 `76fb9036de80f3bb495fe4c8894159fcb1d399d2437293e012e264d81949f791` against
  [`ds_manifest.json`](ds_manifest.json), verified before any arm; a mismatch refuses the lane). Row text
  `### Instruction:\n{instruction}\n\n### Response:\n{output}`; **tokenised once** (`p38_arm.py --prepare`, the
  checkpoint's `AutoTokenizer` at the pinned revision, truncation at 512, rows < 8 tokens dropped) into
  [`tokens_clinical.json`](tokens_clinical.json): 1,200 train rows, 103,298 train tokens, mean 86.1 tokens per row, the
  first 48 held-out rows, sha `81dc24c3…`. Each arm re-tokenises 8 rows with its own framework's tokenizer object and
  records `tokenizer_agree` (true in every receipt; informational — the ids used are the file's either way). Seq 512 is a
  truncation ceiling: the same 5,175 tokens flow through every 60-step arm and the same 17,173 through both 200-step arms.
- **Loss and eval:** causal LM loss over all tokens (`labels = input_ids`), one row per step in fixed order, batch 1,
  no packing, padding, masking, TRL or template on either side — the same `model(input_ids, labels).loss` call. Held-out
  eval = mean loss over the first 48 held-out rows, `no_grad`, `model.eval()`, at step 0 and every 20 steps; evals are
  excluded from the timed window and from time-to-target's clock.
- **Adapters:** r 8, α 16 (scaling 2.0), dropout 0, no bias; attention q/k/v/o in all 48 layers and every expert's
  `gate_up_proj` / `down_proj` (128 experts × 48 layers); router frozen; **321,257,472 trainable parameters in 576
  tensors, asserted — a different count is a VOID arm.** e4b: `add_attention_lora(model, 8, 16, torch.float32)` +
  `ExpertsLoRA` from the loader. Unsloth: `FastLanguageModel.get_peft_model(r=8, lora_alpha=16, lora_dropout=0,
  bias="none", target_modules=[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj], random_state=0)` — its
  loader maps the last three onto the bare 3-D expert parameters through PEFT's `ParamWrapper`
  (`mlp.experts.gate_up_proj`, `mlp.experts.down_proj`, wrap depth 2); the banner `Unsloth: Detected MoE model with
  num_experts = 128 … Enabling LoRA on MoE parameters: ['mlp.experts.gate_up_proj', 'mlp.experts.down_proj']` is
  asserted and is in every Unsloth receipt's `engagement_banners`. No trainable parameter outside the adapters in
  either arm (`non_adapter_trainable: []`). LoRA init differs and is disclosed: e4b `A ~ N(0, 1/r)`, PEFT Kaiming-uniform;
  `B = 0` on both, so step-0 eval is the pure 4-bit model in both arms.
- **Precision — what the receipts record.** bf16 compute on both. Unsloth's 576 adapter tensors are fp32 before and
  after the harness's normalisation (`lora_cast_to_fp32: 0`). **e4b's adapter is 384 fp32 tensors (the attention LoRA)
  plus 192 bf16 tensors (the expert LoRA)** — `load_moe_4bit_streaming` passes the model dtype to `ExpertsLoRA`, so the
  pre-registration's "fp32 adapters (e4b by construction)" did not hold for the expert adapter. Recorded in every e4b
  receipt (`adapter_dtypes_before` / `after`) and not corrected in flight: it is what the shipped loader does, and it is
  the reason the two adapters differ in size (656.1 MB vs 1,285.1 MB for the same 321,257,472 parameters), not a
  format overhead. Optimizer states fp32 on both.
- **What is quantised:** experts → NF4 on both (e4b `Experts4bit`, block 64, fp32 absmax per block, computed on the
  packed bytes by grouped-nf4-gemm's fused NF4 grouped GEMM and `dgrad` kernel; Unsloth: bitsandbytes `Params4bit` nf4
  via transformers' `Bnb4BitHfQuantizer` with `unsloth_zoo`'s `moe_utils_bnb4bit` patches, Unsloth's default double
  quantisation, dequantised on the fly to bf16 and dispatched to the `native_torch` grouped loop — **this mechanism
  difference IS the comparison**); attention q/k/v/o → NF4 on both (e4b: `quantize_attention_projections_4bit`, the
  shipped `TRAIN_ATTN_4BIT` mechanism, 192 `Linear4bit` asserted; Unsloth: transformers' bitsandbytes integration);
  embeddings and `lm_head` bf16 on both; the router `gate` bf16 on both (`census.router_gate`, 48 `Qwen3MoeTopKRouter`
  each). The 4-bit census is in every receipt (`census`): e4b `Experts4bit 48 / ExpertsLoRA 48 / LoRALinear 192 /
  Linear4bit 192`; Unsloth `Params4bit_expert_stacks 96 / Linear4bit 384`.
- **Gradient checkpointing (named modes):** e4b HF `gradient_checkpointing_enable(use_reentrant=False)`, resident (no
  expert offload; the model fits). Unsloth primary `use_gradient_checkpointing="unsloth"` (its documented default:
  per-layer checkpoint with activation offload to host RAM); Unsloth secondary `use_gradient_checkpointing=True` (HF's
  mode, the semantic match to e4b's). Both Unsloth modes ran at N = 60; the 200-step curve ran the primary.
- **Measured per arm** (the harness [`p38_arm.py`](p38_arm.py) = tp1's `tp1_train_smoke.py` (= `n17_cell.py`) plus the
  deltas U1–U10 named in its header): peak VRAM (`torch.cuda.max_memory_allocated` over the training window, the
  comparable figure; a 1-s `nvidia-smi memory.used` trace beside it in `vram_<fw>_<arm>.txt`), s/step (**median of steps
  11..N** is the headline; the mean over all N printed beside it), tokens/s (measured tokens / training wall), J/step
  (`nvidia-smi power.draw` at 200 ms, idle median subtracted, over the training window; the sampler runs identically in
  both frameworks' arms), held-out loss at every eval, time-to-target (cumulative training wall at the first eval whose
  held-out loss ≤ **0.32**), adapter bytes / tensors / dtypes, C1 (the frozen expert bytes — and the 192 attention
  `Linear4bit` in the attn4 arms — hashed before → after from the bytes that persist, bytes > 0, empties = 0, the
  byte-flip control fires), `init_sha` over every trainable tensor at step 0 (identical across the five e4b arms,
  `2b362f7b19ad…`, and across the three Unsloth arms, `83332aed4cb5…`), engagement counters (e4b: `n_patched` and
  `nf4_qlora.fused_grouped_lora` calls per step; Unsloth: `forward_moe_backend_bnb4bit` calls per step and the experts
  forward pre-hooks), `tokens_sha`, versions, host fingerprint, box class, instance id.

## Registered units, validity rules and the reading rule (pre-registered; applied by `p38_reduce.py`)

**Units:** s/step (median of steps 11..N), tok/s, GB (allocator peak), J/step, held-out loss in nats, time-to-0.32 in
seconds, adapter MB. No "%" position, no floor, no anchor-class projection (training has no anchor class; P36 rule 3).

**A row is VALID or VOID before it is read** (the reducer's `validity()`; a VOID arm never enters a ratio):
`trainable_params == 321,257,472`; no trainable parameter outside the adapters; `tokens_sha` equals the file's; C1
clean (bit-exact, bytes > 0, empties = 0, control fired); the registered step count; e4b `fused*` arms with
`n_patched == 48` and `fused_grouped_lora` calls per step ≥ 96 on every step (every arm read 192); e4b `*attn4` arms
with 192 NF4 attention projections; Unsloth arms with ≥ 96 `Params4bit` expert stacks in the census, the engagement
banner, the `moe_bnb4bit_backend` counter ≥ 96 calls per step on every step (every Unsloth arm read 96 on every step),
and **U8**: `unsloth_zoo`'s `_moe_uses_bnb4bit_expert_weights` predicate True on ≥ 48 experts modules after training —
**as amended (amendment 4): the predicate is evaluated on the innermost experts module** (`n_bnb4bit_unwrapped`, in the
receipts written after the amendment) **and, for the two receipts written before it, the proof + the receipt's own
census (96 stacks) + its own backend counter (96/step) stand in for it.** Thresholds (48 / 96) unchanged.

**The reading rule:** (1) the primary pair is `unsloth/ckpt_unsloth` vs `e4b/fused_attn4` at N = 60 — the arm that
matches Unsloth's quantised set (NF4 experts *and* NF4 attention) on the shipped fused `dgrad` path; (2) the two
held-out losses at step 60 are printed with their difference; **|Δ| ≤ 0.05 nats reads "comparable quality"** (a
reading threshold, never a gate) and the s/step ratio is then the position; outside it, time-to-0.32 is the headline;
(3) e4b's internal fused-vs-reference pair is read in the flagship's B2/C2 units (|Δ final train loss| ≤ 0.05 AND median
step-wise |Δ| ≤ 0.05) as PASS/FAIL, **informational — tp1 owns that licence**; (4) no "best" tally, no winner rule beyond
the printed numbers; (5) no cross-box or cross-lane ratio (tp1's Qwen3 rows and the 2026-08-26 numbers are cited beside,
never divided into); (6) nothing here licenses anything.

## The four amendments (each dated in the private record before the data it touches)

1. **Amendment 1 (16:05Z; the failure at 15:47:16Z, before any data) — the Unsloth venv's pins.** The lane pinned
   `transformers==5.16.1` (the e4b environment's version) inside the Unsloth venv and Unsloth requires its own
   transformers range: `pip` answered `ResolutionImpossible` ([`logs/pip_unsloth.log`](logs/pip_unsloth.log)), the lane
   continued e4b-only (`NO COMPARATOR`) and the three Unsloth arms wrote `install_failed` stubs (kept as
   [`attempts/*.attempt1.json`](attempts/)). Fix: [`p38b.sh`](p38b.sh) (waiter [`p38b_wait.sh`](p38b_wait.sh), same box)
   rebuilds the venv with Unsloth's own resolution — no transformers / bitsandbytes / peft pins — and records every
   version ([`logs/pip_unsloth.attempt2.log`](logs/pip_unsloth.attempt2.log): transformers 5.5.0, bitsandbytes 0.50.2,
   peft 0.20.0, torch 2.8.0+cu128). The e4b side is untouched; both frameworks still share the one tokens file. **What
   changed: the comparator's environment. Not changed: the workload, the fixture, any threshold or knob.** The different
   transformers versions across the two pythons are a recorded environment difference.
2. **Amendment 2 (16:03Z, before any comparator data) — `refs/main` and `torchao`.** Two defects, neither in the
   workload. (a) The lane's own run produced no e4b row either: every e4b arm died at load with
   `LocalEntryNotFoundError` under `HF_HUB_OFFLINE=1` ([`outer.log`](outer.log) lines 40–64). The lane fetched the
   checkpoint with `snapshot_download(revision=<sha>)`, which writes no `refs/main` pointer, and the e4b loader resolves
   the model id at `main` (`AutoConfig.from_pretrained` without `revision`), so the pinned snapshot was on disk and
   unreachable; tp1's unpinned fetch of the same repository had staged the same snapshot, so `main == the pin` at fetch
   time. Fix: write `refs/main := ad44e777…` — a cache pointer, the bytes unchanged — proven by an offline `AutoConfig`
   load. The loader's inability to honour a pinned revision is filed as e4b#404 (reproducibility, not a P38 matter).
   (b) The amendment-1 venv installed but its tripwire failed at `import peft` → transformers' lazy module → `torchao
   0.18.0`, which imports `torch.nn.functional.ScalingType` (torch ≥ 2.9): an optional import on that path; `torchao`
   removed from the venv and the tripwire passes ([`logs/tripwire_unsloth.log`](logs/tripwire_unsloth.log):
   `moe_backend native_torch separated_lora True`). [`p38c.sh`](p38c.sh) re-runs the eight registered arms in the
   registered order on the same box, same fixture (`TOKENS(p38c) … sha=81dc24c3…` in `summary.txt`), marker `TP3_DONE`;
   the failed rows stay in `summary.txt` above its lines. **What changed: a cache pointer and one optional package in
   the comparator's venv. Not changed: the workload, the fixture, any threshold or knob.**
3. **Amendment 3 (16:25Z, before any Unsloth data) — snapshot-directory resolution in the Unsloth branch.** p38c's four
   60-step e4b arms and the 200-step e4b arm ran to `CELL OK`; both Unsloth 60-step arms died before training at the
   harness's `snapshot_download(a.model, revision=<sha>)` inside the Unsloth venv: huggingface_hub 1.30.0 (the same
   version in both pythons) raises `LocalEntryNotFoundError` for an offline sha-revision lookup on this cache although
   `snapshots/<sha>/` is present with all 23 files (reproduced standalone in the venv; [`outer.log`](outer.log) lines
   142–149). Fix: the harness's Unsloth branch resolves the pinned snapshot directory directly when it exists
   ([`amend3.py`](amend3.py), applied from the unpatched copy [`p38_arm.py.pre_amend3`](p38_arm.py.pre_amend3); the e4b
   branch untouched; same bytes) — the same class as tp1's amendment 3, harness plumbing. A first in-flight patch attempt
   through nested ssh quoting left the file with a `SyntaxError` for about a minute (the running e4b arm had already
   loaded the module; no arm started in that window); the scp'd script replaced it. `ckpt_unsloth_200` ran in-lane on the
   patched harness (`ENGAGE` banner, `CELL OK`, 2.157 s/step); [`p38d.sh`](p38d.sh) (waiter
   [`p38d_wait.sh`](p38d_wait.sh)) re-ran `ckpt_unsloth` and `ckpt_hf`, marker `TP4_DONE`. The two p38c failures wrote no
   receipt JSON (they died before the harness's first write), so their record is `summary.txt` (`rc=1`) and the
   `outer.log` tails; their run logs were overwritten by p38d's runs under the same names. **What changed: how the
   comparator's harness finds the pinned snapshot. Not changed: the workload, the fixture, any threshold or knob.**
4. **Amendment 4 (16:43Z instrument, additive; 16:55Z reducer, after the proof) — U8 on PEFT's wrapper.** p38c's
   reduction marked `unsloth/ckpt_unsloth_200` VOID on one of its four Unsloth validity checks, U8: `n_bnb4bit 0 < 48`,
   while its other three passed (census 96 `Params4bit` expert stacks; `moe_bnb4bit_backend` intercepted 96 calls per
   step on every step; the engagement banner). Reading: U8 applied `unsloth_zoo`'s `_moe_uses_bnb4bit_expert_weights` to
   the module named `mlp.experts`, which after `get_peft_model` is PEFT's `ParamWrapper` (twice wrapped); the predicate
   inspects the wrapper's attributes and reports False for all 48 while the very function it gates
   (`forward_moe_backend_bnb4bit`) fired 96 times per step. An instrument defect of the check, not a silent fallback —
   but the VOID stood until proven. (a) [`amend4.py`](amend4.py) (applied 16:43:36Z, before `ckpt_hf` started;
   `ckpt_unsloth` was already running) adds `n_bnb4bit_unwrapped` — the same predicate on the innermost module — and the
   inner parameter types to U8; additive, nothing removed (the pre-amendment harness is
   [`p38_arm.py.pre_amend4`](p38_arm.py.pre_amend4)). (b) After `TP4_DONE`, with the GPU free, the standalone proof
   [`u8_proof.py`](u8_proof.py) (waiter [`u8_wait.sh`](u8_wait.sh); a fresh process, the same load and PEFT call, 10.6 s
   load; [`u8_proof.json`](u8_proof.json), [`logs/u8_proof.log`](logs/u8_proof.log)) recorded the predicate on the 48
   `mlp.experts` modules: **before `get_peft_model`: 48/48 True (`Qwen3MoeExperts`, `Params4bit`); after, on the module
   named `mlp.experts` (PEFT `ParamWrapper`, attributes `NoneType`): 0/48; after, on the innermost module (wrap depth 2,
   `Qwen3MoeExperts`, `Params4bit`): 48/48.** U8 as written had inspected the wrapper. The `ckpt_hf` receipt, run on the
   amendment-4 harness, carries `n_bnb4bit 0 / n_bnb4bit_unwrapped 48 / inner Params4bit`. Only then was the reducer
   amended ([`amend_reduce_u8.py`](amend_reduce_u8.py); the pre-amendment copy is
   [`p38_reduce.py.pre_amend4`](p38_reduce.py.pre_amend4)): U8 reads the unwrapped count when the receipt carries it; the
   two receipts written before the amendment (`ckpt_unsloth`, `ckpt_unsloth_200`) are judged by the proof together with
   their own census (96 stacks) and their own per-step backend counter (96 calls of the gated function). Thresholds
   unchanged (48 / 96). Re-reduced with `--u8-proof u8_proof.json` (marker `TP5_DONE`): **all eight rows VALID; no number
   in any row changed.** **What changed: which module the U8 predicate is evaluated on. Not changed: the workload, the
   fixture, any threshold or knob.**

Full text of the amendments: the private receipt `INT4B16/P25-PARITY.md` ("P37 / P38 launch log", amendments 1–4, the
proof, the close-out) and [`PREREG.md`](PREREG.md) here.

## The table, exactly as reduced ([`RESULTS-p38.md`](RESULTS-p38.md) = [`RESULTS.txt`](RESULTS.txt), the reducer's output under amendment 4)

| framework | arm | status | validity | N | s/step med(11+) | s/step mean | tok/s | peak GB | J/step | train first→last | held-out 0→final | t→target s | adapter MB (dtype) | trainable | ckpt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused | ok | **VALID** | 60 | 1.473 | 1.951 | 58.8 | 22.715 | 145.6 | 3.6185→0.2642 | 3.7362→0.2851 | 58.4 | 656.1 (bf16, f32) | 321257472 | hf:use_reentrant=False |
| e4b | fused_attn4 | ok | **VALID** | 60 | 1.522 | 2.053 | 56.0 | 21.371 | 157.1 | 3.6282→0.2629 | 3.7022→0.2923 | 92.5 | 656.1 (bf16, f32) | 321257472 | hf:use_reentrant=False |
| e4b | fused_attn4_200 | ok | **VALID** | 200 | 1.515 | 2.035 | 56.6 | 21.372 | 177.7 | 3.6282→0.3017 | 3.7022→0.2881 | 91.5 | 656.1 (bf16, f32) | 321257472 | hf:use_reentrant=False |
| e4b | fused_attn4_nosamp | ok | **VALID** | 20 | 1.530 | 2.042 | 56.4 | 21.371 | — | 3.6282→0.3375 | 3.7022→0.3459 | not reached by 20 steps | 656.1 (bf16, f32) | 321257472 | hf:use_reentrant=False |
| e4b | reference_attn4 | ok | **VALID** | 60 | 4.437 | 6.607 | 19.5 | 21.371 | 268.1 | 3.6535→0.2642 | 3.7021→0.2909 | 176.7 | 656.1 (bf16, f32) | 321257472 | hf:use_reentrant=False |
| unsloth | ckpt_hf | ok | **VALID** | 60 | 2.146 | 3.493 | 39.8 | 23.137 | 221.3 | 3.6550→0.2606 | 3.6968→0.2975 | 130.1 | 1285.1 (F32) | 321257472 | hf:True (via get_peft_model) |
| unsloth | ckpt_unsloth | ok | **VALID** | 60 | 2.151 | 3.499 | 39.7 | 23.141 | 224.7 | 3.6550→0.2606 | 3.6968→0.2975 | 130.3 | 1285.1 (F32) | 321257472 | unsloth |
| unsloth | ckpt_unsloth_200 | ok | **VALID** | 200 | 2.157 | 3.522 | 39.2 | 23.141 | 258.3 | 3.6550→0.2795 | 3.6968→0.2713 | 135.9 | 1285.1 (F32) | 321257472 | unsloth |

- Energy control: sampler perturbation 1.61 % (steps 11..20: sampled 1554 ms vs unsampled 1530 ms) → energy reported
  without caveat.
- e4b internal parity (informational; tp1 owns the licence): `fused_attn4` vs `reference_attn4` Δ final train loss
  0.00131, median step-wise |Δ| 0.01138 → **PASS** in the B2/C2 band; cost ×2.92 faster per step, peak ×1.000.

**Primary pair — Unsloth (4-bit MoE, "unsloth" checkpointing) vs e4b (fused dgrad path + NF4 attention), N = 60:**

- **s/step ratio Unsloth/e4b = 1.413** (2.151 vs 1.522 s; e4b faster per step at this workload).
- Peak VRAM: Unsloth 23.14 GB vs e4b 21.37 GB (Δ +1.77 GB).
- tokens/s: Unsloth 39.7 vs e4b 56.0; J/step: Unsloth 224.7 vs e4b 157.1 (energy without caveat).
- Held-out loss at the shared evals: step 0 e4b 3.7022 / Unsloth 3.6968 (Δ −0.0055 ≤ 0.05); step 20 0.3489 / 0.5798
  (Δ +0.2309 > 0.05); step 40 0.3205 / 0.3672 (Δ +0.0467 ≤ 0.05); step 60 0.2923 / 0.2975 (Δ +0.0052 ≤ 0.05).
- Reading: **COMPARABLE QUALITY at N = 60** (|Δ| ≤ 0.05) — the s/step ratio is the position.
- Time-to-target (held-out ≤ 0.32, cumulative training wall, evals excluded): e4b 92.5 s; Unsloth 130.3 s (both from
  the 60-step arms).
- Adapters: e4b 656.1 MB (bf16 + f32) / Unsloth 1,285.1 MB (F32); both 321,257,472 parameters (asserted).
- Step-0 held-out — the two quantisers on the same bytes with B = 0: e4b 3.7022 vs Unsloth 3.6968 (Δ −0.0055).

**Secondary rows (never the headline):** e4b `fused` (the tp1 fixture, bf16 attention): 1.473 s/step, peak 22.71 GB,
held-out 3.7362→0.2851 — vs the primary e4b arm s/step ×0.968, peak +1.34 GB. Unsloth `ckpt_hf` (HF checkpointing, the
semantic match to e4b's mode): 2.146 s/step, peak 23.14 GB, held-out 3.6968→0.2975 — vs the primary Unsloth arm s/step
×0.998, peak −0.00 GB.

**200-step curves (held-out loss at every eval; cumulative training wall):**

- e4b (VALID): 0: 3.7022 @ 0 s · 20: 0.3504 @ 31 s · 40: 0.3200 @ 61 s · 60: 0.2848 @ 92 s · 80: 0.2873 @ 122 s ·
  100: 0.3184 @ 152 s · 120: 0.3004 @ 182 s · 140: 0.2933 @ 212 s · 160: 0.2947 @ 243 s · 180: 0.2900 @ 273 s ·
  **200: 0.2881 @ 303 s**.
- Unsloth (VALID): 0: 3.6968 @ 0 s · 20: 0.5798 @ 49 s · 40: 0.3672 @ 92 s · 60: 0.2975 @ 136 s · 80: 0.2918 @ 179 s ·
  100: 0.2894 @ 222 s · 120: 0.2844 @ 265 s · 140: 0.2799 @ 309 s · 160: 0.2768 @ 352 s · 180: 0.2753 @ 395 s ·
  **200: 0.2713 @ 438 s**.

## Close-out (registered units; the register's ids in `docs/claims.json`)

| framework / arm | s/step med(11+) | tok/s | peak GB | J/step | held-out 0→N | t→0.32 | adapter |
|---|---|---|---|---|---|---|---|
| **e4b fused_attn4 (primary)** | **1.522** | 56.0 | **21.371** | 157.1 | 3.7022→0.2923 (N=60) | 92.5 s | 656.1 MB (bf16+f32) |
| e4b fused (bf16 attention, the tp1 fixture) | 1.473 | 58.8 | 22.715 | 145.6 | 3.7362→0.2851 | 58.4 s | 656.1 MB |
| e4b reference_attn4 (control) | 4.437 | 19.5 | 21.371 | 268.1 | 3.7021→0.2909 | 176.7 s | 656.1 MB |
| **Unsloth ckpt_unsloth (primary)** | **2.151** | 39.7 | **23.141** | 224.7 | 3.6968→0.2975 (N=60) | 130.3 s | 1,285.1 MB (F32) |
| Unsloth ckpt_hf | 2.146 | 39.8 | 23.137 | 221.3 | 3.6968→0.2975 | 130.1 s | 1,285.1 MB |
| e4b fused_attn4_200 | 1.515 | 56.6 | 21.372 | 177.7 | →0.2881 (N=200) | 91.5 s | — |
| Unsloth ckpt_unsloth_200 | 2.157 | 39.2 | 23.141 | 258.3 | →0.2713 (N=200) | 135.9 s | — |

- **Position** (`e4b.train.h2h.unsloth.qwen3.5090.2026-09-05`): s/step ratio Unsloth/e4b = **1.413** (2.151 vs
  1.522 s; e4b faster per step); peak VRAM e4b −1.77 GB; J/step e4b ×0.70; time-to-0.32 e4b 92.5 s vs 130.3 s. Same box,
  same session, identical tokens; a ratio between two arms on this box, not a number that travels.
- **Quality at N = 60: COMPARABLE** (`…quality-n60`): held-out 0.2923 vs 0.2975, |Δ| 0.0052 ≤ 0.05 — the registered
  reading threshold, not a gate.
- **At N = 200 the curves separate in Unsloth's favour** (`…curve-n200`): **0.2713 vs 0.2881 (Δ −0.017)**. e4b's
  curve flattens near 0.29 from step 60 while Unsloth's keeps falling. This is a registered row in Unsloth's favour and
  is quoted beside the position wherever the position is quoted. Candidate causes are **NOT established** — the eval
  schedule and sampler, the checkpointing mode, the different transformers / peft versions of the two stacks, and (from
  the receipts' own dtype census) the expert-adapter precision, bf16 on the e4b side against fp32 on Unsloth's; the e4b
  bf16-attention arm reaches 0.2851 at N = 60. Deciding among them is a separately registered lane, not this one.
- **e4b internal fused vs reference** (`…e4b-internal-parity`): PASS (Δ final 0.00131, median 0.01138), ×2.92 per
  step at identical peak — informational; tp1's row `e4b.train.parity.tp1.qwen3.fused.2026-09-05` owns the licence.
- Per-arm rows: `e4b.train.h2h.unsloth.qwen3.5090.2026-09-05.arm.<framework>.<arm>` (eight, all VALID).
- Nothing in the register is superseded by this lane. The 2026-08-26 "1.17× ahead" memory (private, never registered;
  the Unsloth arm trained at a uniform-prediction loss on unmatched work) is disqualified as a current comparison by
  this lane; it was never a claim and gets no row.

## Predictions P1–P10 (pre-registered; scored at this snapshot, the lane ships whichever way they fall)

| | predicted | outcome |
|---|---|---|
| **P1** | both frameworks run the registered configuration on 31.8 GiB; e4b peak 22–26 GB, Unsloth below its 21.6 GB at seq 192 × mb 4 | **Held on the fit, wrong on the numbers:** e4b 21.371 GB (below the 22–26 range; NF4 attention −1.34 GB vs the bf16-attention arm's 22.715), Unsloth 23.141 GB (above, not below, its 2026-08-26 figure). |
| **P2** | peak VRAM: Unsloth lower by 1–4 GB | **Wrong in sign:** Unsloth higher by 1.77 GB. |
| **P3** | s/step: Unsloth faster per step by 1.3–3× (central 2×); "if e4b comes out ahead, that is the finding" | **Wrong in sign:** e4b faster per step, ratio Unsloth/e4b 1.413. The finding, as pre-registered. |
| **P4** | tokens/s follows s/step | **Held:** 56.0 vs 39.7 on the same 5,175 tokens. |
| **P5** | the faster arm has the lower J/step | **Held:** 157.1 vs 224.7 J/step. |
| **P6** | step-0 held-out within 0.02; step 60 within the 0.05 reading threshold; the e4b fused-vs-reference pair passes its band | **Held on all three:** step 0 Δ 0.0055; step 60 Δ 0.0052; internal parity PASS 0.00131 / 0.01138. (At N = 200 the curves separate by 0.017 in Unsloth's favour — outside P6's scope, registered as its own row.) |
| **P7** | identical parameter count (asserted); ≈ 1.285 GB fp32 adapters on both sides; "if PEFT/Unsloth writes bf16, ≈ 0.64 GB — a format difference, reported as such" | **Count held** (321,257,472 both). **The size prediction held for Unsloth (1,285.1 MB F32) and not for e4b (656.1 MB):** it is e4b, not Unsloth, whose expert adapter is bf16 — the loader passes the model dtype to `ExpertsLoRA` — so the pre-registration's "fp32 adapters (e4b by construction)" was wrong about the shipped loader; recorded, not corrected. |
| **P8** | time-to-0.32 reached by both inside 200 steps; the lower-s/step arm reaches it first unless the curves separate | **Held:** both inside 60 steps (92.5 s / 130.3 s); e4b first. The curves separate later than the target crossing (N = 200, Unsloth lower), which P8's "unless" names — that row is quoted beside the position. |
| **P9** | "unsloth" checkpointing 1–3 GB below HF mode at ≤ ±10 % s/step | **VRAM part wrong, s/step part held:** 23.141 vs 23.137 GB (−0.00 GB at ≈ 100 tokens per step, nothing to offload); s/step ×0.998. |
| **P10** | step-time assumptions (alarm sizing): e4b reference_attn4 ≈ 12–14 s, fused ≈ 5–8 s, Unsloth ≈ 2–4.5 s | **e4b wrong by 3–4× (in its favour), Unsloth held:** reference 4.437 s, fused 1.522 s, Unsloth 2.151 s. No alarm fired; the lane took ≈ 1 h 12 min against ≈ 2.5–3 h. |

## Every attempt, a row (from [`summary.txt`](summary.txt), in lane order)

| attempt | arm | outcome | record |
|---|---|---|---|
| lane (15:57Z) | e4b reference_attn4 / fused_attn4 / fused / fused_attn4_nosamp / fused_attn4_200 | `rc=1`, died at load: `LocalEntryNotFoundError` (amendment 2) | `summary.txt` lines 4–7, 10; `outer.log` lines 40–64 (the error tails; the run logs were overwritten by p38c's runs) |
| lane (15:57Z) | unsloth ckpt_unsloth / ckpt_hf / ckpt_unsloth_200 | `INSTALL_FAILED` (amendment 1) | [`attempts/*.attempt1.json`](attempts/), [`logs/pip_unsloth.log`](logs/pip_unsloth.log) |
| p38b (16:00Z) | (the Unsloth venv only) | pip ok, tripwire FAIL on `torchao` (amendment 2) | [`logs/pip_unsloth.attempt2.log`](logs/pip_unsloth.attempt2.log), `outer.log` lines 112–121 |
| p38c (16:03Z) | e4b reference_attn4 / fused_attn4 / fused / fused_attn4_nosamp / fused_attn4_200 | `CELL OK` — **the rows** | the receipts and [`logs/run_e4b_*.log`](logs/) |
| p38c (16:19Z) | unsloth ckpt_unsloth / ckpt_hf | `rc=1`, died at load in the Unsloth venv: `LocalEntryNotFoundError` (amendment 3) | `summary.txt` lines 22–23; `outer.log` lines 142–149 (no receipt JSON written) |
| p38c (16:27Z) | unsloth ckpt_unsloth_200 | `CELL OK` on the amendment-3 harness — **the row** (VOID on U8 at `TP3_DONE`, VALID under amendment 4) | the receipt, [`logs/run_unsloth_ckpt_unsloth_200.log`](logs/run_unsloth_ckpt_unsloth_200.log) |
| p38d (16:42Z) | unsloth ckpt_unsloth / ckpt_hf | `CELL OK` — **the rows** (`ckpt_hf` on the amendment-4 harness) | the receipts, [`logs/run_unsloth_ckpt_unsloth.log`](logs/run_unsloth_ckpt_unsloth.log), [`logs/run_unsloth_ckpt_hf.log`](logs/run_unsloth_ckpt_hf.log) |
| U8 proof (16:52Z) | (no arm; the amendment-4 proof) | `rc=0`: 48/48 → 0/48 → 48/48 | [`u8_proof.json`](u8_proof.json), [`logs/u8_proof.log`](logs/u8_proof.log) |

## Layout

- `<framework>_train_<arm>.json` — one receipt per VALID arm (eight); [`attempts/`](attempts/) — the lane's three
  `install_failed` stubs. [`RESULTS-p38.md`](RESULTS-p38.md) and [`RESULTS.txt`](RESULTS.txt) (byte-identical: the
  reducer's output under amendment 4, `python p38_reduce.py . --u8-proof u8_proof.json`), [`summary.txt`](summary.txt)
  (one line per attempt, in lane order, plus the amendment markers), [`versions.txt`](versions.txt) (the two tripwires'
  appended lines; the Unsloth block repeats once per tripwire run), [`u8_proof.json`](u8_proof.json),
  [`anchor.json`](anchor.json), [`ds_manifest.json`](ds_manifest.json) (byte-identical to
  `bench/flagship-matrix/ds_manifest.json`), [`forensics.txt`](forensics.txt), [`tokens_clinical.json`](tokens_clinical.json)
  (the one tokenised fixture, 0.9 MB), the eight `vram_<framework>_<arm>.txt` 1-s `nvidia-smi` traces, the markers
  `TP_DONE` … `TP5_DONE` and `U8_DONE` (empty files, kept because the waiters keyed on them).
- Harness and lane: [`p38_arm.py`](p38_arm.py) (the patched harness, amendments 3 and 4) beside
  [`p38_arm.py.pre_amend3`](p38_arm.py.pre_amend3) and [`p38_arm.py.pre_amend4`](p38_arm.py.pre_amend4) (`diff` them);
  [`p38_reduce.py`](p38_reduce.py) (amendment 4) beside [`p38_reduce.py.pre_amend4`](p38_reduce.py.pre_amend4);
  [`amend3.py`](amend3.py), [`amend4.py`](amend4.py), [`amend_reduce_u8.py`](amend_reduce_u8.py) (the patches as applied);
  [`u8_proof.py`](u8_proof.py); [`tripwire.py`](tripwire.py) (the Unsloth tripwire p38c/p38d re-ran);
  [`p38_run.sh`](p38_run.sh) (the pre-registered lane script — its Unsloth pip line is amendment 1's defect, its
  `snapshot_download` fetch is amendment 2's), [`p38b.sh`](p38b.sh), [`p38c.sh`](p38c.sh), [`p38d.sh`](p38d.sh) and the
  waiters [`p38b_wait.sh`](p38b_wait.sh), [`p38d_wait.sh`](p38d_wait.sh), [`u8_wait.sh`](u8_wait.sh); the lane console
  [`outer.log`](outer.log) (the lane, p38b, p38c, p38d and the proof all appended to it).
- [`logs/`](logs/) — `pip_e4b.log`, `pip_unsloth.log` (amendment 1's failure), `pip_unsloth.attempt2.log`,
  `tripwire_unsloth.attempt3.log` and `tripwire_unsloth.log`, `anchor.log`, `anchor_gate.log`, `datasets.log`,
  `fetch_qwen3.log`, `prepare.log`, **every per-arm `run_*.log`** (the load line, `step k/N` every 10 steps with the
  kernel-call counts, the `CELL OK` line; the Unsloth logs carry the engagement banner and the backend line) and
  `u8_proof.log`. The logs are force-added past the repository's `*.log` ignore rule, as tp1's were.
- [`PREREG.md`](PREREG.md) — the pre-registration, verbatim from the private tree, added by this bundle.
- Nothing in this directory is edited: every file is the byte-for-byte copy of the box's snapshot (checked with `cmp`
  when the bundle was written) except the two written here, `README.md` and `PREREG.md`. Not shipped: `adapters/`
  (1.3 GB of trained adapters; their bytes, tensor counts and dtypes are in the receipts), `e4b-src/` (the repository's
  archive at the cut), `venv-unsloth/`, `unsloth_compiled_cache/` (Unsloth's 39 generated patched-forward files, a
  cache), the regenerated `data/ds_*.json` (their registered shas are in `ds_manifest.json`; the lane refuses a
  mismatch), the helper copies the lane fetched from the repository (`n9_datasets.py`, `train_anchor.py`,
  `train_anchor_gate.py` — byte-identical to `bench/flagship-matrix/drivers/` and `bench/train-anchor/` at `f4b639f`),
  `INSTANCE_ID` (49975389, also in every receipt's `env.host.vast_instance_id`), the empty waiter logs and `__pycache__`.

## Reproduce

Rent the class in `forensics.txt` (a 5090 whose train anchor reads `pcie-full/launch-fast`; ≥ 320 GB disk, ≥ 98 GB host
RAM, a link that sustains ≥ 40 MB/s), stage `p38_run.sh`, `p38_arm.py`, `p38_reduce.py` and `tripwire.py` under
`/root/p38/`, and run `p38_run.sh` **with amendments 1–2 applied** (the Unsloth pip line without transformers /
bitsandbytes / peft pins, `torchao` removed from the venv, and `refs/main` written to the pinned revision after the
fetch — or the loader's fix once e4b#404 ships); `p38_arm.py` here already carries amendments 3–4. Then
`python p38_reduce.py . --u8-proof u8_proof.json` (a fresh proof from `u8_proof.py` if the harness's `n_bnb4bit_unwrapped`
is absent from any receipt).

## What is NOT claimed

- **No convergence claim, no winner.** 60 steps rank two arms against each other on one text; 200 steps draw one curve
  each. The register's convergence claims are different runs and are neither superseded nor extended by this lane.
- **No general speed claim.** The s/step ratio is this workload (≈ 86 tokens per step, batch 1, resident 30B MoE) on
  this box; the pre-registration expected the opposite sign at exactly this workload and was wrong. Larger batches,
  longer sequences and other families are separately registered lanes.
- **The 200-step separation is not explained.** 0.2713 vs 0.2881 is measured; every candidate cause above is a
  candidate. The position is never quoted without it.
- **Nothing is licensed.** Not e4b's paths (tp1 does that, on its own rows) and not Unsloth's; the internal parity PASS
  here is informational.
- **No cross-box ratio.** tp1's Qwen3 rows (box 49937730: reference 12.797 s/step, fused 5.003) sit beside these
  (reference_attn4 4.437, fused_attn4 1.522 on box 49975389 with NF4 attention) with their dates and are never divided
  into them; the two boxes differ in host class and the arms differ in attention precision.
- **The prediction table is scored, not argued.** P2, P3, P7 (e4b's half) and P10 (e4b's half) were wrong; the lane
  ships them.
