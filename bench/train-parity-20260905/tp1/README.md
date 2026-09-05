# Training parity lane tp1 — 2026-09-05, one RTX 5090: the six serving families through the SHIPPED training path, verdicts in the registered units

**Status: complete — `TP_DONE` 2026-09-05T15:22:26Z (the six families through the registered arms, 24 receipts incl.
the six load receipts) and `TP2_DONE` 15:33Z (Granite's `fused` corrected-counter re-run, `logs/tp1b.sh`, same box).**
18 result lines, every attempt a row, nothing pending. Every arm ran to a receipt: no alarm, no OOM, no load fault, no
verify failure; one HARNESS_ERROR (kept) and one launcher abort (amendment 5, listed in the re-run's reason).

**Bundle shape (phase directive, 2026-09-05 14:45Z — applied to this bundle before the final snapshot so the
finalisation is mechanical).** (1) Every row is exactly one of **OK, REFUSED, HARNESS_ERROR, ALARM, OOM, NOT_RUN,
EXPERIMENTAL**, with the parity verdict (PASS / FAIL / VOID) as a separate column for OK rows; the reducer
([`tp1_reduce.py`](tp1_reduce.py), **v2**) classifies mechanically from `summary.txt` (rc per attempt), the receipt or
stub, `logs/outer.log` and the run logs — never from a missing error; a missing receipt is NOT_RUN with the reason from
`outer.log` (an alarm is ALARM). The copy that ran on the box (v1) is kept verbatim as [`logs/tp1_reduce.py`](logs/tp1_reduce.py);
its table is superseded. Granite's first `fused` attempt is a HARNESS_ERROR row (its log
[`logs/run_granite_fused.attempt1.log`](logs/run_granite_fused.attempt1.log) — renamed from the box's
`run_granite_fused.log`, bytes unchanged, so tp1b's re-run log lands under the original name; likewise
[`logs/fetch_granite.attempt1.log`](logs/fetch_granite.attempt1.log)); the tp1b re-run is the row that counts and the
reducer labels it "the corrected-counter re-run" from the `tp1b:` marker. Every amendment (1–4) stays on this page and
is referenced from the rows it touched. The reducer's per-family matrix states, per family: reference / fused / batched
support, the native-format route, the loss-parity result, s/step, peak VRAM, tok/s, the evidence tier and the
limitations or refusal reason. (2) `docs/capabilities.json` states training support **per path** (`training_support.by_model_type`,
keyed by model_type: quantize / reference_train / fast_train / batched_train / nvme_train / native_mxfp4_train, each
`supported` / `refused` / `void` / `harness_error` / `not_tested` / `experimental` / `n/a` with its claim ids), and
`model_families` is exactly the families whose `fast_train` is `supported`. (3) `granitemoe.fast_train` is decided by
the tp1b re-run only; `batched_train` is `supported` (PASS). No gate, threshold or fixture text changed.

**Box:** Vast.ai instance **49937730**, RTX 5090 (sm_120, driver 595.84, 32607 MiB, power limit 550 W, PCIe gen 4 ×16)
on an AMD EPYC 7Q83 64-core host (128 CPUs, 251.5 GiB; [`forensics.txt`](forensics.txt) is the lane script's probe at
install; the per-receipt `env.host` block carries the GPU UUID, bus id, VBIOS and the instance id). **Train-anchor class
`pcie-full/launch-fast`** ([`anchor.json`](anchor.json), gate log [`logs/anchor_gate.log`](logs/anchor_gate.log):
FLOPs 184.36 TFLOP/s = 0.996× the reference band, self-consistency 1.0066; launch 186,947/s; H2D 25.93 GB/s; `BOX
ACCEPTED`). A training receipt without its class is not a receipt — two 5090s ran an identical training config 1.65×
apart ([`../../train-anchor/README.md`](../../train-anchor/README.md)) — so every absolute on this page is *this box's*
and every ratio is within one family, same box, same session. torch 2.8.0+cu128, transformers 5.16.1, bitsandbytes
0.50.1 (image `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`); installed `experts4bit-qlora 0.35.0`,
`grouped-nf4-gemm 0.30.0` (the tripwire line in [`logs/outer.log`](logs/outer.log)).

**When (box time = UTC, from [`logs/outer.log`](logs/outer.log)):** install 11:32:21Z (`pip rc=0`; tripwire `tp1
tripwire OK: e4b 0.35.0 gnf4 0.30.0 (no usercustomize hook: training lane)`; `cuda ok`); helpers fetched as the
repository's archive tarball 11:32:43Z (amendment 2); train anchor 11:32:53Z → `BOX ACCEPTED`; dataset regenerated and
sha-verified 11:32:57Z (`DATASET clinical sha=76fb9036de80…`); Granite fetched 11:32:57Z (5.5 min), arms 11:38:26Z
`reference` → 11:44:40Z `fused` (TypeError, amendment 3) → 11:45:01Z `batched` → `GRANITE DONE` 11:46:51Z; OLMoE fetched
(7.0 min), arms 11:53:52Z / 11:58:35Z / 12:00:26Z → `OLMOE DONE` 12:03:39Z; gpt-oss fetched (15.8 min), `attn_only`
12:19:26Z, `mxfp4` 12:23:14Z → `GPTOSS DONE` 12:32:58Z; Qwen3 fetched (8.7 min), arms 12:41:43Z / 13:01:01Z /
13:09:19Z → `QWEN3 DONE` 13:27:26Z; Gemma-4 fetched 13:27:26Z — **staged in 47.1 min** (the resumed connection
recovered inside the 4800-s alarm; amendment 4's trigger never fired), arms 14:14:30Z / 14:26:03Z / 14:31:35Z →
`GEMMA4 DONE` 14:42:57Z; Mixtral fetched 14:42:57Z (15.6 min), arms under `offload=1` 14:58:31Z / 15:07:00Z / 15:14:43Z
→ `MIXTRAL DONE` 15:22:26Z; the box's own reducer print ([`logs/RESULTS.txt`](logs/RESULTS.txt)) and **`TP_DONE`
15:22:26Z**. **Lane wall 11:32Z → 15:22Z = 3 h 50 min** against the pre-registration's ≈ 5.5 h estimate (P7's step
times were wrong in both directions, below); no arm or fetch alarm fired on any of the 17 result lines. Then
`logs/tp1b.sh` on `TP_DONE`: `tp1b: Granite fused arm re-run on the patched harness` 15:23:13Z, `harness patched OK`,
Granite re-fetched (its conditional Gemma-4 / Mixtral redo, amendment 4, found both reference receipts present and
did not run); its first `arm granite/fused` start at 15:30:43Z died on the follow-up script's own `DATA: unbound
variable` before the harness ran (amendment 5), the fixed script's start at 15:31:49Z is the re-run → `GRANITE FUSED
RERUN DONE`, **`TP1B DONE` / `TP2_DONE` 15:33:13Z**. **Wall on this box for the six families plus the re-run: 11:32Z →
15:33Z = 4 h 01 min** against the pre-registration's ≈ 5.5 h. **Honest box time and cost:** the first box (49937447)
≈ 5 min before amendment 1 destroyed it; this box rented ≈ 07:05Z, **≈ 4.4 h of it idle** after the fetch-by-sha false
start (amendment 2, ≈ $3.4), the lane and re-run ≈ 4.0 h, `TP2_DONE` at 15:33Z — **≈ 8.5 h ≈ $6.8 at ~$0.8/h, of which
the measurements themselves ≈ $3.2**; teardown follows `TP2_DONE` (label-guarded, proven before the guard is killed).
Hard-kill guard on the mini: re-armed to 20:00Z at the relaunch, then 12:50Z to 23:30Z before Qwen3's arms; it did not
fire.

**Cut — the shipped code, both packages at their `main` after the 0.35.0 / 0.30.0 releases** (the `pip install` line in
[`logs/tp1_run.sh`](logs/tp1_run.sh); [`logs/pip.log`](logs/pip.log) shows pip resolving both as revisions, not tags):

- **experts4bit-qlora main @`f4b639fd2640`** = v0.35.0 + #396 (docs). The tripwire asserts `__version__ == "0.35.0"` and
  imports `enable_fast_train`, `enable_batched_train`, `ExpertsLoRA`, `load_moe_4bit_streaming`, `verify_moe_4bit`,
  `engines.int4_experts.enable_serve_experts_int4_calibrated` (the 0.35.0 symbol bo7 also asserts) and
  `engines.batched._dequant_whole` / `_PAD_WASTE_LIMIT` (the fallback tp1 counts).
- **grouped-nf4-gemm main @`ddcb850e05c3`** = v0.30.0 + #341 (docs). The tripwire asserts `nf4_qlora.fused_grouped_lora`
  carries a `dgrad_kernel` parameter and imports `mxfp4_qlora.ExpertsMxfp4LoRA` and `run_mxfp4_20b_qlora`.
- **No `usercustomize` hook**: that hook is the serving K8 int4 instrument and has no role in training.

**Fixture (the registered one; [`../../../docs/PREREG-flagship-matrix.md`](../../../docs/PREREG-flagship-matrix.md) B2 /
[`-model2.md`](../../../docs/PREREG-flagship-matrix-model2.md) C1–C3, applied verbatim):** per arm one process, seed 0
before the load; `load_moe_4bit_streaming(model, "cuda", bf16, r=8, alpha=16, quant_type="nf4", offload=<family>)`;
`verify_moe_4bit(strict=True)`; gradient checkpointing (`use_reentrant=False`, `use_cache=False`);
`add_attention_lora(r=8, alpha=16, fp32)`; expert LoRA fp32 (the `ExpertsLoRA` default); AdamW 1e-4, no weight decay,
no schedule, no clipping; batch 1, one example per step, loss over all tokens, sequences truncated at 512 (the rows are
≈100 tokens, so `tokens_per_s` is quoted with the measured tokens); held-out eval = the first 48 rows of the 200-row eval
split, before and after; **N = 60 steps**. Text: the fused-train-gate's synthetic **`clinical`** set (1,200 train / 200
eval, seed 1000), regenerated on the box by `bench/flagship-matrix/drivers/n9_datasets.py` at the cut and **refused
unless its sha256 equals the registered `76fb9036de80f3bb495fe4c8894159fcb1d399d2437293e012e264d81949f791`**
([`ds_manifest.json`](ds_manifest.json), byte-identical to `bench/flagship-matrix/ds_manifest.json`). Arms:
**`reference`** (the per-expert loop), **`fused`** (`enable_fast_train(model, dgrad=True)`, the documented default),
**`batched`** (`enable_batched_train(model)`); gpt-oss instead runs **`attn_only`** (the reference path over its bare
experts, which also probes both enablers and writes the `fused` / `batched` refusal stubs) and the **EXPERIMENTAL
`mxfp4`** run (grouped-nf4-gemm's `run_mxfp4_20b_qlora`, its own wikitext-2 text, never compared to the e4b arms, never
licensed); Mixtral runs `offload=True`. Every arm records `init_sha` (sha256 over every trainable tensor at step 0, so
the reducer *proves* the arms started bit-identical), the kernel-call count per step (`fused_grouped_lora` /
`_dequant_whole`, wrapped and counted — a patch count is not a call count), C1 from `state_dict()` with bytes > 0,
empties == 0 and a byte-flip positive control, per-step loss and time, and the `nvidia-smi` power sampler at 200 ms with
the idle baseline subtracted. Harness: [`logs/tp1_train_smoke.py`](logs/tp1_train_smoke.py) = `n17_cell.py` + deltas
D1–D8 (named in its header). Order Granite → OLMoE → gpt-oss → Qwen3 → Gemma-4 → Mixtral, the checkpoint freed between
families.

## The five amendments, and why (each dated in the private record before the data it touches)

1. **Amendment 1 (07:05Z, before any data) — the first box's link.** The queue fired on bo7's `TP_DONE` at 07:00:27Z
   and rented Vast **49937447** (machine 37675). It passed the renter's 15-s pre-flight at 20 MB/s (bo7's 15 MB/s floor)
   but sustained **10.4 MB/s** on a 20-s range read of a Qwen3 shard while the install ran: at that rate Qwen3 (60 GB),
   Gemma-4 (52 GB) and Mixtral (93 GB) would have died by their fetch alarms — three of six families lost before a step.
   Decision at install time, nothing measured: lane stopped, instance destroyed by the label-guarded teardown (proven
   absent; bo7's box untouched), machine blacklisted; **≈ 5 min billed**. Parameters amended: the pre-flight floor
   **40 MB/s** (a training lane downloads ≈ 230 GB of checkpoints) and **every fetch alarm doubled** (Granite/OLMoE 2400,
   gpt-oss 3000, Qwen3/Gemma-4 4800, Mixtral 7200 s — the `fetch` lines in [`logs/tp1_run.sh`](logs/tp1_run.sh) against
   [`logs/tp1_run.sh.pre-amend`](logs/tp1_run.sh.pre-amend)). Arm alarms, fixture, criteria, verdict rule, predictions
   unchanged. The first relaunch failed at the rent step (`KeyError: VAST_API_KEY` — the relaunch chain had not sourced
   the secrets the queue sources); relaunched with the environment.
2. **Amendment 2 (11:40Z) — the short-sha fetch, and the idle time.** On the re-rolled box **49937730** the install and
   tripwire passed ([`logs/outer.attempt1.log`](logs/outer.attempt1.log): 07:05:35Z) but the helper fetch `git init &&
   git fetch --depth 1 origin f4b639fd2640` failed within a second — `fatal: couldn't find remote ref f4b639fd2640`
   (GitHub does not serve a fetch-by-abbreviated-sha) — `SRC FETCH FAIL`, exit 9, `TP_DONE` touched. **The box then idled
   ≈ 4.4 h (07:06Z → 11:32Z, ≈ $3.4 at ~$0.8/h) before it was noticed:** the lane's own watcher reported the failure, but
   the session-side tail had not survived the queue-log rewrite at the relaunch. Fix: the helpers (`n9_datasets.py`,
   `ds_manifest.json`, the train-anchor pair) are fetched as the repository's **archive tarball at the cut**
   (`archive/<sha>.tar.gz`, 600-s alarm; all four byte-identical to the repository at `f4b639f`, checked when this
   bundle was written). Relaunched on the same box with the install intact (`outer.attempt1.log` kept), guard re-armed
   to 20:00Z before the old one was killed, watcher rebuilt. Fixture, arms, criteria, defaults unchanged.
3. **Amendment 3 (11:55Z, harness defect, fixed in flight) — the closure bug.** Granite's `fused` arm died in its first
   forward with `TypeError: _dequant_whole() got an unexpected keyword argument 'weights_fn'`
   ([`logs/run_granite_fused.log`](logs/run_granite_fused.log)) while `reference` and `batched` passed. Cause, in the
   harness and not in the shipped code: `KernelCounter.install` (D5) wraps `nf4_qlora.fused_grouped_lora` and
   `batched._dequant_whole` in one scope with the same closure variable `orig`, so Python's late binding made the fused
   wrapper call `_dequant_whole` — every family's `fused` arm would have failed identically. Patched on the box before
   OLMoE's fused arm ran ([`logs/tp1_closure_patch.py`](logs/tp1_closure_patch.py): each wrapper binds its original as a
   default argument, `def w(*a, _orig=orig, **k)`; the shipped harness copy is the patched one, `grep _orig=orig`) and in
   every copy. Each arm is a fresh process, so OLMoE and everything after ran on the fixed file. **Granite's `fused` arm
   is re-run after `TP_DONE` on the same box, same install, same fixture by [`logs/tp1b.sh`](logs/tp1b.sh)** (waiter
   [`logs/tp1b_wait.sh`](logs/tp1b_wait.sh); Granite re-fetched; marker `TP2_DONE`). The first receipt is a
   `harness_error` row, kept (its log is the row — the process died before writing a JSON); the re-run is the row that
   counts. Fixture, criteria, verdict rule, predictions unchanged. **The re-run PASSED** (attempt 2/2 in the table:
   0.01329 / 0.01270, ×5.86, J ×0.197, 128 kernel calls every step); its own launcher hiccup is amendment 5.

4. **Amendment 4 (14:15Z) — the Gemma-4 fetch hang, left to its alarm; conditional redo.** The Gemma-4-26B-A4B-it
   fetch (started 13:27Z on a ≥ 40 MB/s link, XET already disabled by the lane) hit `The read operation timed out` on
   shard 1 of 2 at 13:46Z with 46 GB cached, and its resumed connection went silent. Decision: **no kill** — it is left
   to its 4800-s fetch alarm (≈ 14:47Z), after which the lane continues with Mixtral; the first Gemma-4 attempt will
   read **ALARM** (fetch) in `summary.txt` (`gemma4: FETCH FAILED rc=142`) and the reducer attaches this amendment to
   those rows from that line. `logs/tp1b.sh` (the post-`TP_DONE` follow-up on the same box) now also **redoes any
   family whose reference receipt is missing** — Gemma-4 (fetch 4800 s resuming the cached blobs, then
   `reference` / `fused` / `batched` with the pre-registered alarms) and, should its fetch fail the same way, Mixtral —
   before touching `TP2_DONE`; the redo's rows are labelled from the `tp1b:` marker. Worst-case `TP2_DONE` ≈ 20:30Z
   against the 23:30Z guard. Fixture, criteria, verdict rule, predictions unchanged. **Outcome:** the trigger never
   fired — the fetch's resumed connection recovered and the checkpoint staged at 47.1 min (`outer.log` line 89), inside
   the 4800-s alarm; `summary.txt` carries no `FETCH FAILED` line, all three Gemma-4 arms ran on the lane itself, and
   `tp1b.sh`'s redo (conditional on a missing reference receipt) did not run. The amended `tp1b.sh` as of `TP_DONE` is
   shipped as [`logs/tp1b.sh`](logs/tp1b.sh) (its redo block is the diff against [`logs/tp1b.sh.amend3`](logs/tp1b.sh.amend3),
   the amendment-3 copy). Gemma-4-it also loaded on this host **without the #344 fault** (P6's load-succeeds branch).
5. **Amendment 5 (15:50Z in the private record; the event at 15:30:43Z) — the follow-up script's unset variable.**
   `tp1b.sh` reuses `tp1_run.sh`'s `arm` helper, which reads the dataset path (`DATA`, `DATA_SHA`) that the lane script
   sets in its dataset step; the follow-up never set them, so under `set -u` its first `arm granite/fused` start
   (15:30:43Z, `outer.log` line 220) died with `/root/tp1/tp1b.sh: line 32: DATA: unbound variable` before the harness
   ran — a defect of the follow-up script, not a lane arm and not the harness. Fixed by setting the dataset variables
   (and re-verifying the registered sha; a mismatch would have refused with exit 13) and the anchor class at the top of
   the script: [`logs/tp1b.sh`](logs/tp1b.sh) is that copy, the one that ran the re-run; the `TP_DONE`-time copy with
   the defect is [`logs/tp1b.sh.amend4`](logs/tp1b.sh.amend4). The aborted start is an `arm` line in `outer.log` with
   no result line; the reducer aligns result lines to the last start lines and prints the extra start in the re-run's
   reason, so it cannot be mistaken for an attempt of the harness. Fixture, criteria, verdict rule, predictions
   unchanged.

Full text: the private receipt `INT4B16/P25-PARITY.md` (§P36, amendments 1–5, the launch and false-start logs) and
`INT4B16/tp1/P36-PREREG.md`.

## Registered criteria (verbatim) and the verdict rule per arm

> **B2 — loss parity, fused vs reference.** For each dataset, train twice from the same seed: once through the reference
> path, once through `enable_fast_train`. Registered band: **|Δ final-train-loss| ≤ 0.05** and the step-wise loss curves'
> median absolute difference **≤ 0.05**. The fused path reorders expert summation (group-sorted vs ascending id), so exact
> equality is not expected; a difference larger than this band means it is not the same computation.

> **C1 — bit-exactness (HARD GATE).** SHA-256 of every frozen expert's packed bytes before and after training must be
> **identical, 100 %**, in both arms. Hashes must come from `state_dict()` (offload maps experts to their CPU home), with
> **bytes hashed > 0** and **zero empty tensors skipped** asserted, and a byte-flip positive control demonstrating the
> check can fail. Any mismatch voids every performance number in the document.

> **C2 — loss parity, fused vs reference.** Registered band, unchanged from the first ten so the two halves are
> comparable: **|Δ final train loss| ≤ 0.05** **and** median step-wise |Δ| ≤ 0.05, per dataset.

> **C3 — cost, reported for every cell, not gated:** s/step, tokens/s, peak VRAM, and energy as J/step from `nvidia-smi`
> at 200 ms across the timed window, idle baseline subtracted.

**Verdict rule per (family, accelerated arm), applied by [`tp1_reduce.py`](tp1_reduce.py) in these units and no others
(P36 §"Registered criteria", verbatim):**

> 1. **VOID** (no verdict, the row says why) if any of: the arm's `init_sha` ≠ the family's `reference` `init_sha` (the
>    arms did not start identical); `C1_bit_exact` false or `C1_bytes_hashed == 0` or `C1_empties_skipped > 0` or the
>    byte-flip control did not fire, in either arm; `n_patched == 0` on a `fused` / `batched` arm that reached training
>    ("the arm must be the arm", n17); `kernel_calls_per_step < 2 × n_patched` on a `fused` / `batched` arm (the patched
>    forward never reached the kernel on some layer — for `batched` this is the `_PAD_WASTE_LIMIT` fallback, which is
>    silent in the code and counted here); the two arms trained a different number of steps.
> 2. Otherwise **PASS** iff `|loss_last_fused − loss_last_reference| ≤ 0.05` **and** `median_i |loss_i^fused −
>    loss_i^reference| ≤ 0.05` over the N steps; else **FAIL**. Both numbers are printed with the verdict; eval-loss
>    deltas are printed beside them labelled "not the band" (the flagship results corrected exactly this confusion once).
> 3. **Cost is reported, never gated**: s/step, tokens/s, peak VRAM, J/step, and the within-family ratios `reference/arm`
>    for s/step and `arm/reference` for peak VRAM and J/step — on this box, this session.
> 4. No "best result" / C4 tally: the model-2 matrix showed the 0.99 % separability threshold does not transfer across
>    models (the winner flipped sign on a re-run), so tp1 registers no winner rule at all.
> 5. A `refused` / `oom` / `load_fault` / `alarm` / `verify_failed` receipt is a **row with that status**, and the family's
>    other arms are still read.
> 6. `attn_only` (gpt-oss) has no reference pair and gets no parity verdict: its row reports that the reference path
>    trains attention-only LoRA over the bare frozen experts (loss finite and falling, eval before → after), C1 on the
>    bare stacks, and the two refusal stubs.
> 7. The gpt-oss `mxfp4` row is **EXPERIMENTAL, NOT LICENSED**: printed from the runner's `run_artifact.json` (step-0
>    canary vs transformers' dequant path: `top1`, `kl`; `provenance.pre_equals_post`; eval before → after; peak) with
>    that label in the verdict column. A failed canary or `pre_equals_post == false` is reported as such and changes
>    nothing else.

Reading rules 1–9 of the pre-registration apply as written: registered units only (nothing in %, nats or against a
floor); no cross-lane, cross-card or cross-family ratio; no training throughput position and no anchor projection; a
refusal is a row; a green skipped arm is not evidence; C1 is a hard gate per arm; gpt-oss rows never say "trains" (the
experts); licence labels for serving are the register's; the dataset sha and `init_sha` are printed in every row.

## Predictions P1–P7, scored at this snapshot (falsifiable; the lane ships whichever way they fall)

| | predicted | so far |
|---|---|---|
| **P1** | Granite, OLMoE, Qwen3, Gemma-4, Mixtral: `verify_moe_4bit(strict=True)` passes; `enable_fast_train(dgrad=True)` patches exactly the MoE-layer count (32 / 16 / 48 / 30 / 32); `fused` PASSES both criteria; C1 holds in every arm | **Held on all five:** verify 32/0, 16/0, 48/0, 30/0, 32/0; patched 32/32, 16/16, 48/48, 30/30, 32/32; fused PASS 0.01329 / 0.01270 (Granite, on the corrected-counter re-run — attempt 1 was a HARNESS_ERROR of the harness, not of the shipped code), 0.01327 / 0.01249 (OLMoE), 0.01315 / 0.01050 (Qwen3), 0.02385 / 0.04742 (Gemma-4 — inside the band by 0.0026), 0.00953 / 0.00945 (Mixtral, offload); C1 clean in every OK arm of every family. |
| **P2** | `enable_batched_train` patches the same counts, PASSES parity where it engages, is slower than `fused` at 30B width with the highest peak VRAM; *stated risk:* on the 128-expert families (Qwen3, Gemma-4) `_PAD_WASTE_LIMIT` falls back on some or all layers → VOID (not engaged), a finding about the batched path's engagement envelope, not about parity | **Counts held** (32 / 16 / 48 / 30 / 32). **PASS where it engaged** (Granite 0.01553 / 0.01681; Mixtral 0.00766 / 0.01057, the 8-expert shape engaged 192 calls every step). **Slower than fused at 30B width held** (Qwen3 11.885 vs 5.003 s; Gemma-4 7.259 vs 3.101). **Highest peak held resident** (Qwen3 23.629, Gemma-4 21.623, OLMoE 6.211, Granite 2.971 — each the family's highest); *not* under offload (Mixtral batched 8.466 < reference 10.562). **The stated risk fired on both 128-expert families as predicted — Qwen3 VOID (min 12 < 96), Gemma-4 VOID (min 0 < 60; 9 of 60 steps with no kernel call at all) — and on the 64-expert one it did not name (OLMoE VOID, min 24 < 32).** *Not predicted:* at Granite's width the batched path is ×3.93 over the per-expert loop — the register's "no speed-up at real width (1.05×)" is a 30B-width statement (Qwen3 here reads ×1.08 on the mixed path), not a general one. |
| **P3** | gpt-oss: `ExpertsLoRA` absent (0 wrapped, 24 bare); both enablers return 0 → two `refused` rows; `attn_only` trains with C1 holding on the bare stacks | **Held exactly.** 0 wrapped / 24 bare `GptOssExperts4bit`; `[e4b.fast] … on 0 ExpertsLoRA module(s)`, `[e4b.batched] … on 0 ExpertsLoRA module(s)`; `attn_only` 5.106 → 0.366 (eval 5.1405 → 0.3435), C1 10,749,542,400 B bit-exact, control fires. |
| **P4** | gpt-oss `mxfp4`: canary passes (`top1` ≥ 0.9 over 32 positions, small `kl`), `pre_equals_post` true, loss falls; s/step reported without a prediction | **Held.** top-1 0.906 (by 0.006), KL 0.02495; `pre_equals_post` True (96 tensors); loss 5.461 → 2.622, eval 4.591 → 2.184; 5.16 s/step, peak 6.22 GB. Experimental, not licensed. |
| **P5** | Mixtral offload arms run; the resident probe, if enabled, OOMs | **Held:** all three `offload=True` arms ran to OK (3.421 GB on the GPU at load, peaks 10.562 / 5.215 / 8.466 GB); the probe stayed off, so the OOM clause is untested. |
| **P6** | Gemma-4-it loads on this host or faults per #344; either is a row | **Held (the load-succeeds branch):** loaded in 14.7 s, 30/30, 18.113 GB, no `CUDA error: invalid argument`. The fetch hang (amendment 4) resolved itself at 47.1 min; no ALARM row. |
| **P7** | Step-time assumptions (alarm sizing only): Granite ≈ 0.5–1 s, OLMoE ≈ 1–2 s, gpt-oss attention-only ≈ 3–5 s, Qwen3 reference ≈ 6–8 s / fused ≈ 3 s / batched ≈ 6–8 s, Gemma-4 ≈ Qwen3, Mixtral-offload reference ≈ 8–12 s / fused ≈ 4–6 s | **Wrong in both directions, harmlessly:** Granite reference 4.27 s (4–8× the assumption — the 40-expert per-expert loop is launch-bound), batched 1.09; OLMoE 3.02 / 0.94 / 1.91; gpt-oss attention-only 1.80 (under); Qwen3 12.80 / 5.00 / 11.89 (reference ≈ 1.7× over); Gemma-4 7.34 / 3.10 / 7.26 (under Qwen3, not ≈); Mixtral-offload 3.07 / 2.43 / 2.41 (≈ 3× under). No alarm fired on any of the 17 result lines; the lane took 3 h 50 min against the ≈ 5.5 h estimate. |

## What the matrix now says (from the receipts in this directory; the register's ids in `docs/claims.json`)

The reducer's **per-family matrix** (the last table in [`RESULTS-tp1.md`](RESULTS-tp1.md)) is the ten-column statement
the directive asks for — reference / fused / batched support, native-format route, loss-parity result, s/step, peak
VRAM, tok/s, evidence tier, limitations or refusal reason — read from the LAST attempt of each arm with every earlier
attempt listed in the last column. The per-path reading below is the same evidence in the capability vocabulary; the
machine-readable form is `training_support` in `docs/capabilities.json`.

| family | direct real-weight load + `verify(strict)` | `ExpertsLoRA` | `fused` (`enable_fast_train(dgrad=True)`) | `batched` (`enable_batched_train`) | other |
|---|---|---|---|---|---|
| Granite-3.1-3B-A800M | **measured** — 32/32, 2.351 GB (was: fixtures + the arena path only) | **measured** — 32 | **OK · PASS** 0.01329 / 0.01270 — ×5.86, peak ×1.000, J ×0.197 (attempt 2/2, the corrected-counter re-run; attempt 1 a kept HARNESS_ERROR, amendment 3; the launcher abort between them, amendment 5) | **OK · PASS** 0.01553 / 0.01681 — ×3.93, peak ×1.057, J ×0.303 | — |
| OLMoE-1B-7B-Instruct | measured — 16/16, 4.695 GB | measured — 16 | **OK · PASS** 0.01327 / 0.01249 — ×3.22, peak ×1.000, J ×0.339 (first reading on a registered text with real weights) | **OK · VOID** — `_PAD_WASTE_LIMIT` fallback engaged (min 24 < 32 kernel calls/step) | — |
| gpt-oss-20b | measured — 24/24, bare, 14.359 GB | **unsupported** (built bare) | **REFUSED** (0 patched) | **REFUSED** (0 patched) | `attn_only` **OK** (no pair); `mxfp4` **EXPERIMENTAL** — canary 0.906 / 0.025, provenance holds, loss falls; not licensed |
| Qwen3-30B-A3B | measured — 48/48, **20.019 GB resident on 32 GB** | measured — 48 | **OK · PASS** 0.01315 / 0.01050 — ×2.56, peak ×1.000, J ×0.407 | **OK · VOID** — fallback on most layers every step (min 12 < 96 kernel calls/step) | — |
| Gemma-4-26B-A4B-it | measured — 30/30, 18.113 GB resident; **no #344 fault on this host** (amendment 4's fetch hang recovered at 47.1 min) | measured — 30 | **OK · PASS** 0.02385 / 0.04742 (inside the band by 0.0026) — ×2.37, peak ×1.000, J ×0.477 | **OK · VOID** — 9 of 60 steps with no kernel call (min 0 < 60) | — |
| Mixtral-8x7B-Instruct | measured — 32/32 through the `w1/w3/w2` fusion, `offload=True`, 3.421 GB on the GPU (25.37 GB of NF4 experts pinned in host RAM) | measured — 32 | **OK · PASS** 0.00953 / 0.00945 — ×1.26, peak ×0.494, J ×0.734 | **OK · PASS** 0.00766 / 0.01057 — ×1.27, peak ×0.802, J ×0.976 (engaged 192 calls every step) | — |

Per path (`training_support` in `docs/capabilities.json`; `model_families` = the families whose `fast_train` is `supported`):

| model_type | quantize | reference_train | fast_train | batched_train | nvme_train | native_mxfp4_train |
|---|---|---|---|---|---|---|
| `olmoe` | supported | supported | **supported** (tp1 PASS) | void (tp1) | not_tested | n/a |
| `qwen3_moe` | supported (tp1 resident on 32 GB) | supported (tp1; flagship) | **supported** (tp1 PASS; flagship 5/5) | void (tp1; the dgrad-gate trajectory stands on its own fixture) | not_tested | n/a |
| `gemma4_text` | supported (tp1: the `-it` checkpoint on this host; flagship: base) | supported (tp1; flagship) | **supported** (tp1 PASS, median inside the band by 0.0026; flagship 5/5) | void (tp1: 9 steps with no kernel call) | not_tested | n/a |
| `granitemoe` | supported (tp1) | supported (tp1) | **supported** (tp1 PASS on the corrected-counter re-run; attempt 1 a kept harness_error) → enters `model_families` | supported (tp1 PASS) | not_tested | n/a |
| `gpt_oss` | supported (bare) | refused | refused | refused | refused (`enable_mxfp4_nvme_residency` refuses bias-carrying modules, #402; the training-arena wrap is refused on structure) | experimental (tp1 canary) |
| `mixtral` | supported (tp1, offload) | supported (tp1) | **supported** (tp1 PASS) → enters `model_families` | supported (tp1 PASS) | not_tested | n/a |

**`olmoe` is confirmed on real weights and a registered text** (it entered on the dgrad-gate's synthetic tokens);
**`qwen3_moe` and `gemma4_text` are confirmed on this lane** beside their flagship receipts (never divided into them);
**`mixtral` enters `model_families` on its fused PASS** — under offload, the first real-weight pass through its
`w1/w3/w2` fusion — and **`granitemoe` enters on its corrected-counter re-run's PASS** (its `batched_train` was already
`supported`); **`gpt_oss` stays out** — every e4b training enabler refuses it, and the one route that trains its
experts is the kernel package's experimental `ExpertsMxfp4LoRA`, which this lane exercised and did not license. The
three `batched` VOIDs (OLMoE, Qwen3, Gemma-4) are one limitation the capability carries: in the shipped 0.35.0 code
`enable_batched_train` falls back per call above `_PAD_WASTE_LIMIT` with no counter, so a batched arm must assert kernel
engagement, not the patch count; it engaged everywhere only on the 8-expert (Mixtral) and 40-expert (Granite) shapes.
The release that carries this bundle adds `batched_fallback_stats(model)` for exactly this (#402).

## The table

[`RESULTS-tp1.md`](RESULTS-tp1.md) is the verbatim output of [`tp1_reduce.py`](tp1_reduce.py) (v2; `python tp1_reduce.py .`,
stdlib only) followed by the reading. The reducer reads `summary.txt` (one `rc` line per attempt, a family's `FETCH
FAILED` lines), `<fam>_train_<arm>.json` (receipts and stubs), the gpt-oss experimental envelope
`gptoss_train_mxfp4.json`, `logs/outer.log` (arm and fetch start lines, the `tp1b:` marker) and the per-attempt run
logs (`run_<fam>_<arm>[.attempt<k>].log`), and prints per family the load row, the environment line, **one row per
attempt of each arm** (attempt k/n · **row status** · **verdict** with the within-family cost ratios · `n_patched` ·
kernel calls/step min against `2 × n_patched` · C1 · steps · s/step · tok/s · peak GB · J/step · loss first → last ·
eval 0 → final, labelled "not the band" · |Δ final| · median step-wise |Δ| · the reason and the amendment that touched
the row), the reference's `init_sha` and the dataset sha, and then the **per-family matrix** (reference / fused / batched
support in the capability vocabulary, native-format route, loss-parity result, s/step, peak GB, tok/s, evidence tier,
limitations / refusal reason — every earlier attempt listed). The classification rules are the reducer's docstring;
the verdict rule is v1's, verbatim.

## Layout

- `<family>_train_<arm>.json` — one receipt per arm (`reference` / `fused` / `batched` / `attn_only` / `mxfp4`; refusal
  stubs for gpt-oss `fused` / `batched`); `<family>_train_load.json` — the load-stage receipt written by the first arm
  that passed `verify_moe_4bit` (six of them). [`gptoss_mxfp4/`](gptoss_mxfp4/) — the
  experimental runner's `run_artifact.json` (canary, provenance hashes, eval every 10 steps) and `steps.jsonl` (per-step
  loss, dt, peak). [`anchor.json`](anchor.json) (the train-anchor probes), [`ds_manifest.json`](ds_manifest.json) (the
  registered dataset shas), [`forensics.txt`](forensics.txt), [`summary.txt`](summary.txt) (one line per arm, in lane
  order: rc and the result line).
- [`logs/`](logs/) — [`tp1_run.sh`](logs/tp1_run.sh) (the amended lane script, the one that ran),
  [`tp1_run.sh.pre-amend`](logs/tp1_run.sh.pre-amend) (the pre-registered copy: fetch-by-sha, un-doubled fetch alarms —
  `diff` it), [`tp1_train_smoke.py`](logs/tp1_train_smoke.py) (the harness, patched copy),
  [`tp1_closure_patch.py`](logs/tp1_closure_patch.py) (amendment 3's patch as applied),
  [`tp1b_wait.sh`](logs/tp1b_wait.sh) (the post-`TP_DONE` waiter), `pip.log`, `anchor.log`,
  `anchor_gate.log`, `datasets.log`, the six `fetch_*.log` (Granite's lane fetch as `fetch_granite.attempt1.log`, tp1b's
  re-fetch as `fetch_granite.log`), **every per-arm `run_*.log`** (the load lines, `step k/60` every 10 steps with the
  kernel-call counts, the result line; Granite's fused attempt-1 traceback as `run_granite_fused.attempt1.log`, the
  re-run as `run_granite_fused.log`), [`tp1_reduce.py`](logs/tp1_reduce.py) (v1, the copy that ran on the
  box) and [`RESULTS.txt`](logs/RESULTS.txt) (its print at `TP_DONE`, the box's own table — superseded by this
  directory's v2 output), [`tp1b.sh`](logs/tp1b.sh) (the amendment-5 copy that ran the re-run) beside
  [`tp1b.sh.amend4`](logs/tp1b.sh.amend4) (the `TP_DONE`-time copy, the unset-variable defect) and
  [`tp1b.sh.amend3`](logs/tp1b.sh.amend3) (the amendment-3 copy),
  the lane console [`outer.log`](logs/outer.log) (through `TP1B DONE`; `tp1b.sh` appended to it, including the
  amendment-5 abort line) and the false start's [`outer.attempt1.log`](logs/outer.attempt1.log).
  The logs are force-added past the repository's `*.log` ignore rule, as bo6's and bo7's were.
- Nothing in this directory is edited: every file is the byte-for-byte copy of the box's snapshot (checked with `cmp`
  when the bundle was written), except the three written here — `README.md`, `RESULTS-tp1.md`, and `tp1_reduce.py`
  (v2, written for the bundle under the 14:45Z directive; the box's v1 copy is [`logs/tp1_reduce.py`](logs/tp1_reduce.py),
  byte-identical to the one that ran). Two verbatim files are renamed so the re-runs land under the original names:
  `run_granite_fused.log` → [`logs/run_granite_fused.attempt1.log`](logs/run_granite_fused.attempt1.log) and
  `fetch_granite.log` → [`logs/fetch_granite.attempt1.log`](logs/fetch_granite.attempt1.log) (bytes unchanged). Not shipped: the helper copies the
  lane fetched from the repository (`n9_datasets.py`, `train_anchor.py`, `train_anchor_gate.py` — byte-identical to
  `bench/flagship-matrix/drivers/` and `bench/train-anchor/` at `f4b639f`), the regenerated `data/ds_*.json` (their
  registered shas are in `ds_manifest.json`; the lane refuses a mismatch), the mini-side renter / queue / watcher
  scripts (the copy of the renter in the receipt tree predates amendment 1's 40 MB/s floor and is not the one that ran,
  so it is not offered as evidence), the empty `tp1b_wait.log`, `INSTANCE_ID` (49937730, also in every receipt's
  `env.host.vast_instance_id`), the box's `RESULTS-tp1.md` (byte-identical to `logs/RESULTS.txt`), the `TP_DONE`
  marker and `__pycache__`.

## Reproduce

Rent the class in `forensics.txt` (a 5090 whose train anchor reads `pcie-full/launch-fast`; ≥ 320 GB disk, ≥ 96 GB host
RAM, a link that sustains ≥ 40 MB/s), stage `logs/tp1_run.sh`, `logs/tp1_train_smoke.py` and `tp1_reduce.py` under
`/root/tp1/`, put the Hugging Face token at `~/.cache/huggingface/token` (Gemma-4-it and Mixtral-Instruct are gated; the
script exits 8 without it), and run `logs/tp1_run.sh` (its `pip install` line pins both cuts; its tripwire refuses
anything else; it fetches the dataset generator and the train-anchor pair from the repository at the cut and refuses a
dataset whose sha is not the registered one), then `logs/tp1b.sh` after `TP_DONE`. Then `python tp1_reduce.py .`.

## What is NOT claimed

- **No convergence claim.** 60 steps rank two arms against each other on one text; they do not train a model, and the
  eval column is "not the band". The register's convergence claims (`e4b.train.olmoe-converges`) are different runs and
  are neither superseded nor extended by this lane.
- **PASS means PASS on one text.** The flagship matrices needed five datasets; tp1 ran the registered `clinical` set only;
  Gemma-4's fused median sits 0.0026 inside the band.
- **Nothing is pending.** 18 result lines, 24 receipts; the register carries no placeholder.
- **No cross-family, cross-lane or cross-card ratio**, and no training throughput position: s/step and tok/s are this
  box's, quoted with its anchor class; no anchor-class projection is made for training. tp1's Qwen3 and Gemma-4 rows,
  when they land, sit beside `bench/flagship-matrix` and `bench/flagship-matrix-model2` (a 4090, `offload=True`) and are
  never divided into them.
- **Experimental ≠ licensed.** The gpt-oss `mxfp4` row is the kernel package's experimental route on its own text with
  its own runner; it licenses nothing, and gpt-oss rows never say the experts train under the shipped e4b path.
- **A VOID row is not a PASS**, however its loss curve reads: OLMoE `batched` carries no parity number. **A
  HARNESS_ERROR row is not a shipped-code failure and not a result**: Granite's first fused attempt is kept so the
  re-run cannot read as a first run.
- **A refusal is a row, not a zero**: gpt-oss `fused` / `batched` are build-out items.
- **Cost is reported, never gated**, and a within-family ratio on this box is not a general one (P2's 30B-width speed
  statement stands as measured there; Granite's ×3.93 is Granite's on this box).
- **No gate was changed, no threshold retuned, no umbrella claim** ("training parity across six families") is made: the
  register carries one claim per (family × arm) with this receipt — 26 entries, all `measured`, no placeholder.
