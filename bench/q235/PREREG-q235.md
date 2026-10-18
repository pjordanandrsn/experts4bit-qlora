# Q235 — one full-depth QLoRA optimizer step on Qwen3-235B-A22B-Instruct-2507 (DRAFT, Phase 0: registered before any run)

Issue: experts4bit-qlora#835. **Phase 0 is preparation only.** It runs no GPU work on the subject and rents nothing. It approves no
budget, changes no default and makes no public claim. A box runs only after an amendment registers it, and only once the Phase 0 → 1
gates below have passed. The lane is sequenced behind the active SD2 and DQ11 gates.

## The milestone

**One reproducible, real, full-depth QLoRA optimizer step** on Qwen3-235B-A22B-Instruct-2507: all 94 decoder layers, every routed
expert, one GPU. It runs on the release current at registration: experts4bit-qlora 0.53.0, grouped-nf4-gemm 0.45.0 and loggetta 0.5.0,
each pinned by commit in the amendment that registers the box. The step records:
- device memory: `max_memory_allocated`, `max_memory_reserved` and the driver's peak for the process;
- host memory: pinned bytes (e4b's own accounting) and the process's peak RSS;
- bytes moved: host-to-device and device-to-host over PCIe (e4b's staging counters, cross-checked against the profiler's copy events),
  and NVMe bytes read and written (`/proc/diskstats` deltas around load and around the step);
- nonzero adapter gradients (finite; every adapter tensor has at least one nonzero element; the global norm) and an optimizer delta
  (every trained adapter tensor changed, every frozen tensor unchanged);
- frozen-base hashes: sha256 of every frozen tensor (host homes and device-resident) after load and again after the step;
- fidelity against a bf16 reference (below);
- the step's actual elapsed time: forward, backward and `optimizer.step()`, synchronized, with load and download timed separately.

## Scale, and what is not inherited

From the pinned configs:

| | Qwen3-30B-A3B | Qwen3-235B-A22B | ratio |
|---|---|---|---|
| decoder layers | 48 | 94 | 1.96 |
| hidden size | 2,048 | 4,096 | 2 |
| experts / top-k | 128 / 8 | 128 / 8 | 1 |
| expert intermediate size | 768 | 1,536 | 2 |
| expert parameters | 29.0 B | 227.1 B | 7.83 |
| total parameters | 30.5 B | 235.1 B | 7.70 |

So the expert stacks are about 7.8× those of the 30B model that every TC1 position reads, and the whole model is 7.7× its size.

**Nothing is inferred from simulated kernels or from the July 235B receipts** (grouped-nf4-gemm `bench/phase3/flagship`). Those receipts
are forward-only, batch-1 decode on an older stack (grouped-nf4-gemm 62dbfe9 / 1a60c39, triton 3.4.0, bitsandbytes 0.49.2). They have
no backward, no LoRA and no bf16 reference, and experts4bit-qlora is not in their path. They are background: a real 94-layer decode
with experts in pinned host RAM ran on 15.1–15.2 GB of VRAM. They say nothing about training memory, training speed or quality.

## Provenance (Phase 0, deliverable 1)

- **Checkpoint.** `Qwen/Qwen3-235B-A22B-Instruct-2507` at revision `ac9c66cc9b46af7306746a9250f23d47083d689e`. This has been the hub's
  `main` since 2025-09-17, and the weights' LFS hashes are unchanged since the 2025-07-21 upload. At that revision the hub lists 129
  files: 118 safetensors (470,191,875,040 B, 437.9 GiB) and 11 others.
- **The local cache holds 125 files.** That is the 118 shards plus 7 JSON files; `.gitattributes`, `LICENSE`, `README.md` and
  `merges.txt` are absent. Every file was verified by content on 2026-10-10: 125/125 match the hub's hashes and none differ.
  - 119 files are checked by sha256: the 118 shards plus `tokenizer.json`, which is also stored in LFS.
  - 6 files are checked by git blob id.
  - Each shard's sha256 equals the Instruct-2507 shard at that revision, so the cache is this variant. The base model and
    Thinking-2507 also have 118 shards.
- **The four absent files** were fetched at the same revision into a sidecar directory beside the cache. Each was verified against
  the hub's git blob id before it was written, so the cached snapshot stays byte-identical. `LICENSE` is the licence record, and
  `merges.txt` serves a slow-tokenizer path.
- **The July receipts' "125 shards" are 125 files.** The July downloader called `snapshot_download` with
  `allow_patterns=["*.safetensors", "*.json", "tokenizer*"]` and no revision. That selects 118 shards and 7 JSON files, and the log
  reads "Fetching 125 files". The July record is left as published; this correction lives here.
- **July most likely read the same revision, but that is an inference, not a recorded hash match.** The July file set is this cache's
  file set, and `ac9c66cc` was the only `main` revision in the window. The July receipts record no revision hash.

## The step

**Load.**
- Experts in NF4, held in pinned host RAM and staged to the GPU one layer at a time (`load_moe_4bit_streaming(..., offload=True)`).
- Frozen attention projections in NF4 (`quantize_attention_projections_4bit`).
- LoRA r 16, alpha 16 on every attention projection and every routed expert.
- `enable_fast_train` at the release's defaults:
  - the fused q/k/v training projection;
  - the chunked LM loss under `auto`;
  - the default checkpoint flavour.

  The expert absmax stays fp32: `compress_expert_absmax_` refuses a layer under expert offload, so `enable_fast_train` keeps the fp32
  absmax and records why.
- `pin=True`. e4b prices each pinned tensor rounded up to a power of two (PyTorch's caching host allocator), which is why the host homes
  exceed the device stacks.

**Data.**
- TC1's field-recipe tokens: `unsloth/alpaca-cleaned` at TC1's pinned revision, the same template, tokenized at `ac9c66cc`.
- **Shape A, the milestone:** one micro-batch, rows right-padded to the longest, truncation 2,048 tokens.
- **Shape B, for comparability:** the field recipe's micro-batch 2 × accumulation 4.
- The token tensor's sha256 is recorded.

**Optimizer.**
- bitsandbytes AdamW8bit, lr 2e-4, weight decay 0.001.
- The milestone step runs at a constant lr of 2e-4. This deviates from TC1's warm-up, which holds lr at 0 for steps 0 and 1 and would
  leave no optimizer delta to record. The deviation is stated in every receipt.

**Adapter dtype.** The first box trains **bf16 adapters**, the shipped default, because the milestone is the release as shipped. An fp32
arm (TC1's matched arm) is optional on the same box, only within the cap, and is labelled matched-to-TC1.

**Reproducibility.**
- In the same process, two forward-backward passes on the same tokens (no optimizer step) give the path's run-to-run spread. That spread
  can be exactly 0.
- A fresh process on the same box repeats step 0, and its frozen-base hashes must be equal.
- **Tolerance:** the larger of the in-process spread and a **floor**. Amendment 1 fixes the floor from the 30B calibration's
  fresh-process repeats (below), before any 235B box. With a floor, one unit in the last place of fresh-process nondeterminism does not
  fail the check by construction.
- A fresh-process difference above the tolerance is **reported as a finding with its size**. It is not a silent VOID, and it does not
  pass.

## Fidelity: how a bf16 reference is made at this size

A resident bf16 reference needs 470 GB of device memory, so no single card holds one. Instead, the reference is a **layer-streamed bf16
forward** on the RTX A2000 (12 GB), at $0, from the verified cache:
- each decoder layer's bf16 weights (about 5 GB) are loaded alone, applied, and released;
- the micro-batch's hidden states (2,048 × 4,096 bf16 per sequence, 16 MiB) are carried from layer to layer;
- the embedding and the final norm and head (151,936 × 4,096 bf16, 1.16 GiB) are applied on their own.

It records, for the step's exact token ids:
- every layer's output;
- every layer's router top-8 sets;
- the final per-token loss.

**The NF4 side** is the step's own step-0 loss (before the optimizer step), plus per-layer outputs captured by hooks.

**The read:**
- the loss gap, NF4 minus bf16;
- the per-layer relative error of the outputs (median and maximum over the 94 layers);
- per-layer agreement of the router's top-8 sets.

**Calibration before any box** (on the RTX A2000 and the host CPU beside it, at $0). A full-depth resident bf16 reference of
Qwen3-30B-A3B (about 61 GB) fits neither the RTX A2000 (12 GB) nor the CPU host, which has about 53 GB free beside its services. The
calibration therefore splits the question in three:
1. **Streaming mechanics.** A **depth-truncated** resident reference: the first 6 decoder layers of Qwen3-30B-A3B in bf16 (6 × about
   1.25 GB, plus the embedding and head at 0.62 GB each: about 8.7 GB), resident on the RTX A2000. It is compared with the layer-streamed
   path over the same 6 layers on the same card. Same ops on the same device, so they are expected to agree bit for bit.
2. **CPU against CUDA.** The same 6 layers in bf16, resident on the host CPU: PyTorch's CPU bf16 matmuls, 6 intra-op threads (the host's
   6 cores, under `nice 19`). The CPU-against-CUDA reference difference is measured and stated. It is part of the calibration, not
   assumed away.
3. **The NF4-against-bf16 values, full depth.** The layer-streamed bf16 forward over all 48 layers on the RTX A2000, against e4b's NF4
   path with experts in pinned host RAM on the same card. That is the path the 235B box runs, at the step's shapes. It gives the 30B loss
   gap, per-layer relative error and router agreement. The fresh-process repeats behind the reproducibility floor run here too.
- Amendment 1 sets the fidelity bands from that calibration, before any 235B box. The bands cover the per-layer relative error and the
  router agreement, where a semantic error and rounding differ by orders of magnitude. The loss gap is reported, not gated.

**What fidelity here cannot say:**
- gradient fidelity at full depth. The step checks only that gradients are finite and nonzero. A layer-streamed backward is possible,
  storing 94 layer inputs of about 1.5 GiB, but is not part of this registration;
- anything about quality after training.

## Controls

- **One configuration per comparison.** Any comparison holds the model, revision, sequence length, micro-batch, adapter dtype,
  precision and host fixed. One host class runs per box.
- **Host facts on every receipt:** the CPU model, the PCIe generation and lane count, host RAM, NVMe model and free space, driver, CUDA
  and torch.
- **Fixed inputs:** the token tensor, the seeds, and the LoRA initialization (TC1's per-slot generator), each recorded by hash.

## Host classes, capacity and refusals (Phase 0, deliverable 2)

Two independent estimates were reconciled on the pinned config:
- **e4b 0.53.0's `estimate_qlora_footprint`:** the allocator's bytes at the step's peak, by item.
- **loggetta 0.5.0's planner:** a per-class device plan that adds a 0.5 GiB context and the planner's reserve, so its device totals sit
  above e4b's.

The two agree item for item. One disagreement, in the activations, was an input error: a config passed without `architectures`
silently turns the chunked loss off in the estimate, adding 1.65 GiB. The shared setup is:
- r 16 / alpha 16, AdamW8bit, NF4 attention;
- shape A (2,048 × 1 × 1) and shape B (2,048 × 2 × 4, priced at the nominal 4,096 tokens per micro-batch, which over-prices the
  field recipe's real rows).

**Experts on the device: every single-device class refuses.**
- **Needed:** 182.4 GiB (A, fp32 adapters) and 170.5 GiB (A, bf16), by loggetta's plan.
- **Available after headroom:** RTX 5090 30.25, H100 75.67, RTX PRO 6000 90.81, H200 133.38 GiB.
- The NF4 expert stacks alone take 118.97 GiB.

**Experts in pinned host RAM, attention plus every routed expert, the grouped NF4 path:**

| class | A fp32 | A bf16 | B fp32 | B bf16 | budget after headroom |
|---|---|---|---|---|---|
| RTX 5090 32 GB | refused (41.12) | **fits, 29.24** (1.0 GiB margin) | refused (45.08) | refused (32.56) | 30.25 |
| H100 80 GB | fits, 41.12 | fits, 29.24 | fits, 45.08 | fits, 32.56 | 75.67 |
| RTX PRO 6000 96 GB | fits | fits | fits | fits | 90.81 |
| H200 141 GB | fits | fits | fits | fits | 133.38 |

(GiB, loggetta's plan totals. e4b's raw estimate for A: 33.85 fp32 / 23.95 bf16, of which activations 3.30 / 2.77 under the chunked
loss.)

**Refusal points, named:**
1. Experts resident on the device: no single card (above).
2. Attention-only with the grouped NF4 kernel: `enable_fast_train` has no expert adapter to patch, on every class. Attention-only runs
   only on the reference kernel, which fits everywhere with host experts: 19.5 / 19.3 GiB for A, fp32 / bf16. That is not the milestone.
3. RTX 5090: every row but A-bf16. That row's 1.0 GiB margin must hold the fused path's offload staging transient, which no one has
   measured. The 5090 is therefore a stretch class until the rehearsal measures it.
4. Host RAM: every host-expert row needs 161.6 GiB. That is 158.63 GiB of pinned expert homes, each tensor rounded up to a power of two
   by PyTorch's caching host allocator, plus a 3.0 GiB process baseline. With `pin=False` the homes are 118.97 GiB, about 122 in total,
   at a speed cost. 161.6 GiB is 173.5 GB. With the OS and the loader's unmodelled transients on top, **the registration requires at
   least 192 GB of host RAM.** That rule, not the arithmetic alone, excludes one H100 NVL offer at 188 GB.
   The host homes carry the **fp32** expert absmax, which is correct for this path:
   absmax double-quantization is refused under expert offload (see "The step"). An estimator that prices the absmax by the setting in
   force must still price fp32 here, so this figure is not to be "corrected" downward.
5. Disk: the shards alone are 437.9 GiB (470 GB). The training planner has no NVMe tier, the streaming loader reads the shards
   directly, and no arena is planned. **The registration requires at least 550 GB free**: the shards plus the wheels, the
   workspace and the receipts.
6. The link floor sets the step time. Expert staging moves 237.94 GiB host-to-device per micro-batch: the slab twice, for the forward
   and for the recompute. That is about 12.3 s per micro-batch at the RTX 5090's measured 20.75 GB/s. No other class has a committed
   H2D measurement, so their floors stay unstated until one exists.

**Unmodelled** (both tools say so):
- the fused path's offload staging transient;
- the CUDA context beyond 0.5 GiB, workspaces and fragmentation;
- load-time transients and page cache;
- more than one device;
- time beyond the link floor.

**Cost per class: an estimate, not a quote.** It uses Vast's verified on-demand single-GPU offers at 2026-10-10 13:24 UTC, filtered to at
least 192 GB of host RAM and enough disk. The time model:
- about 1 h of install, load, hashing and a fresh-process repeat, which the rehearsal must replace with measured parts;
- plus the 470 GB fetch at the offer's listed download bandwidth;
- plus the offer's per-GB download charge (RunPod bills none).

| class | fitting offers, $/h | PCIe generation offered | ≈ $ per box, over those offers |
|---|---|---|---|
| RTX 5090 | 0.76–0.97 | gen4 ×16 | 3–8 |
| RTX PRO 6000 | 1.6–2.2 | gen3 to gen5 ×16 | 4–9 |
| H100 (PCIe / NVL) | 2.4–3.1 | gen4 / gen5 ×16 | 3–6 |
| H200 NVL | 4.3–4.4 | gen4 / gen5 ×16 | 6–23 |

The spread within a class comes mostly from the fetch time and the per-GB download charge, which vary by host. The H200 NVL's
upper end is one host charging about $0.04 per GB for the 470 GB fetch. Each is within the $35 per-run cap. An RTX 5090 offer under the class's $0.85 policy rate exists; its slow fetch puts it at about $5. Every class other than the 5090 needs its rate approved.

**The class to register first: an RTX PRO 6000 (96 GB, sm_120).**
- It fits every shape and dtype with room (90.81 GiB after headroom against 45.08 at most), and is offered at $1.6–2.2/h.
- It runs the fused grouped kernels, the same route TC1's RTX 5090 boxes run and the poisoned-empty audit exercised.
- sm_90 classes (H100, H200) take grouped-nf4-gemm's `grouped_mm` route under `auto`. That route is a second code path, worth a later
  cross-check, not the first box.
- The RTX 5090 stays a stretch class until the staging transient is measured.

This draft approves no rate and rents nothing. The registering amendment approves the class's rate, after SD2 and DQ11 close and gates
1–5 pass.

## Decision rules for the eventual box

**MILESTONE PASS** iff all of these hold:
1. The step completes on the registered host class.
2. The loss is finite.
3. Every adapter tensor's gradient is finite, and each has at least one nonzero element.
4. Every trained adapter tensor changes under `optimizer.step()`, and no frozen tensor does.
5. The frozen-base hashes after the step equal those after load.
6. The fresh-process repeat of step 0 is within the tolerance: the larger of the in-process spread and Amendment 1's floor. A miss is
   reported as a finding with its size.
7. Fidelity is within the bands Amendment 1 sets.

**VOID** (no reading; the box log says why):
- a harness error;
- a host fault (a container restart, a lost NVMe, a link below the registered generation);
- a time-left guard firing;
- a refusal: a named refusal point is reached on the registered class.

**Reported, not gating:**
- the measured device peak against e4b's estimate plus its named unmodelled items (a deviation above 15 % opens an estimator issue);
- bytes moved against the estimate's link line;
- the elapsed step time.

## Before any rental: the Phase 0 → 1 gates

1. **A tiny-scale end-to-end rehearsal of the box script**, on CPU and on the RTX A2000. It runs a two-layer Qwen3-MoE of this
   architecture through the exact box script: install, fetch gate, load, step, records and teardown. It follows DQ11's opt-in rehearsal,
   which is never a scientific draw.
2. **A time-left fit test.** Every time-left check's budget is measured on the rehearsal after install, and each guard must fit what it
   guards. A 235B budget is scaled from measured parts, explicitly, never copied from a 30B box.
3. **A pre-rental fetch gate over every locked URL.** This covers the wheels and the 129 model files at `ac9c66cc`. Each is
   hash-locked and checked from a CPU host before the rental controller can quote (DQ11's launch gate).
4. **The box installs its own tools.** git and everything else the script calls are installed on the box before any fetch (DQ11's
   `require_git`).
5. **No placeholder values anywhere.** The manifest, the scripts and the amendment are checked for unfilled values before launch.
6. **Budget.** The launcher's per-run hard cap of $35 applies. Only the RTX 5090 has a class rate in the launcher's policy, so any other
   class needs its rate approved before a launch.

## What this registration cannot say

- **Speed.** One step's elapsed time is a record, not a benchmark, and nothing here compares frameworks or cards.
- **Training dynamics.** Nothing about multi-step training or quality after training.
- **Other shapes.** Nothing about sequence lengths, models or adapter scopes it does not run.
- **The `grouped_mm` route.** grouped-nf4-gemm's `auto` takes this route on sm_90 (H100, H200). It applies only if such a class is the
  registered host; the RTX A2000 cannot exercise it.
- **Serving.**
- **Hardware not run.** Anything about the card classes in the capacity table that no box runs.

## Amendment 1 (2026-10-10, after the 30B calibration, before any 235B box): the floor, the bands, and one correction

**The calibration.**
- **Run:** 2026-10-10 on an RTX A2000 and its host CPU, as the "Fidelity" section registers it, on 8 rows of TC1's field text (1,014
  real tokens).
- **Receipts and tools:** `bench/q235/calibration-30b/` and `bench/q235/instrument/`.
- **Exclusions:** padded positions are excluded from every read.

| comparison | layers | per-layer relative error, median / max | router top-8 sets equal (worst layer) | router overlap (worst layer) | loss gap |
|---|---|---|---|---|---|
| layer-streamed against resident (CUDA) | 6 | 0 / 0 | 100 % | 100 % | 0 |
| CPU against CUDA (both resident bf16) | 6 | 0.0067 / 0.0078 | 86.3 % | 98.2 % | +0.0069 |
| e4b NF4 (experts in host RAM) against streamed bf16 | 48 | 0.107 / 0.204 | 46.4 % | 91.3 % | −0.0055 (loss 1.8592 against 1.8647) |
| e4b NF4, second fresh process against the first | 48 | 0 / 0 | 100 % | 100 % | 0 |

**What it shows:**
- **The streamed reference is the resident model**, bit for bit on one device. A CPU-test against transformers' `from_pretrained` model
  also matches every layer's output by sha256.
- **The CPU and CUDA references differ** by under 1 % per layer, but exact top-8 set equality falls to 86–92 % from device numerics
  alone: the router has near-ties. **Exact set agreement is therefore reported, not gated.** The gate is the overlap, |A ∩ B| / 8,
  which stays at 0.98–0.99 there against about 0.06 for random sets.
- **The NF4-against-bf16 error does not compound much with depth.** It reaches about 0.106 by layer 3 and stays there to layer 36
  (0.106–0.112). It is 0.13 at layer 44 and 0.20 at the last layer. A log-linear fit over the 48 layers grows about 1 % a layer,
  which is about 1.6× over the 46 more layers of the 235B model.
- **e4b's NF4 forward reproduces exactly in a fresh process** on this path: identical loss and per-layer outputs.

**The fidelity bands (rule 7).** The 235B read passes fidelity iff:

| bound | 30B measured | margin |
|---|---|---|
| median per-layer relative error ≤ **0.25** | 0.107 | 2.3× |
| maximum per-layer relative error ≤ **0.50** | 0.204 | 2.5× |
| minimum per-layer router overlap ≥ **0.80** | 0.913 | — |

The margins cover the depth growth above (about 1.6×) and the 235B model's doubled hidden and expert widths. The loss gap is reported,
not gated.

These bands are a semantic-error detector, not a prediction of the 235B values. A wrong expert fusion, a dropped epilogue or misrouted
experts is expected to move a layer's output by order 1 and its overlap toward chance (0.06). **That signature is expected, not yet
measured.**

**The planted errors.** Gate 1's rehearsal therefore plants **each of these two errors**, named here before any rehearsal runs. Each must
fail the bands.
- **Where:** in the NF4 side under test; the bf16 reference stays correct.
- **The model:** the rehearsal's same-architecture model, with at least 4 decoder layers.
- **The layer:** the middle one, layer ⌊n / 2⌋. That is layer 47 of 94 at 235B, and layer 2 of a 4-layer rehearsal model.

1. **A routing error: every routed token goes to the wrong expert.** At the planted layer, the router's top-k expert indices are shifted
   by one (e → (e + 1) mod E) between the router and the experts. Each token is then served by a neighbouring expert's weights at its
   own routing weight. The router's own top-k at that layer is unchanged.
   - **Expected signature:** that layer's output relative error is of order 1, above 0.50. The router overlap at the later layers falls
     well below 0.80, toward chance as their inputs diverge.
   - **Must fail:** the maximum per-layer relative error exceeds 0.50, or the minimum overlap falls below 0.80.
2. **An arithmetic error: the expert epilogue is dropped.** At the planted layer, the experts compute `gate * up` instead of
   `act_fn(gate) * up`: the SiLU is skipped and nothing else changes. This is the error class the epilogue contract (#397) guards.
   - **Expected signature:** that layer's output relative error is of order 1, above 0.50. The later layers carry it forward.
   - **Must fail:** the maximum per-layer relative error exceeds 0.50.

If either planted error passes the bands, the bands are void, and a new amendment sets them before any box.

**The reproducibility floor (rule 6).**
- **Tolerance:** the larger of the in-process spread and **1e-4 relative on the step-0 loss**.
- **Why this floor.** The 30B fresh-process repeat was exact, so the floor only absorbs last-place nondeterminism on another card class.
  It is about 30× below the 30B NF4-against-bf16 loss gap (2.9e-3 relative), so a difference the floor admits is far too small to be a
  semantic error.
- **The floor covers the loss only.** The forward is deterministic on this path, so the step-0 loss can be compared across processes.
  Adapter gradients are not compared across runs.
  - Under PyTorch's default attention backend, the flash-attention backward is nondeterministic: about 0.9 % relative run to run on
    LoRA-B gradients (experts4bit-qlora #1592). Math SDPA makes it bitwise.
  - So the gradient checks are that they are finite and nonzero (rule 3). A bitwise gradient comparison would need the math SDPA
    backend, and this registration makes none.

**A correction to the capacity section.** e4b's estimate now prices the expert absmax as stored: double-quantized under
`enable_fast_train`'s default, which this registration's numbers predate. This moves only the rows with **experts on the device**. Each
falls by 11.83 GiB in loggetta's plan:

| shape | before | after |
|---|---|---|
| A, fp32 | 182.36 | 170.53 |
| A, bf16 | 170.48 | 158.65 |
| B, fp32 | 186.33 | 174.49 |
| B, bf16 | 173.81 | 161.97 |

Every row still refuses on every single card; the H200's budget after headroom is 133.38 GiB. The rows with **experts in host RAM** do
not move. Offload keeps the fp32 absmax, as the capacity section says, so the RTX 5090 fits only A-bf16, at 29.24 GiB.

**Planning facts from the calibration** (not claims):
- **The streamed bf16 reference** over Qwen3-30B-A3B's 48 layers took 1,124 s from a disk-pool cache, about 55 MB/s with per-expert
  tensor reads. At the same rate the 235B reference (470 GB) takes about 2.4 hours of RTX A2000 time.
- **e4b's NF4 side** with experts in host RAM loaded in 1,370 s and 1,798 s, and peaked at 5.55 GB of device memory on the RTX A2000.

## Amendment 2 (2026-10-10, before any 235B box): the gradient and optimizer rules by LoRA factor, and a second step

**Found by the box script's CPU rehearsal** (`bench/q235/instrument/q235_box.py`; `tests/test_q235_box.py`, a 4-layer Qwen3-MoE on
the reference path). The decision rules' 3 and 4 cannot pass as registered:
- LoRA's B is zero-initialized, and each A's gradient flows through its B. So at step 0 **every LoRA-A gradient is exactly zero**: 24 of
  the test's 48 adapter tensors, all of them A and none B.
- With a zero gradient, only weight decay moves an A, by about 2e-7 relative at lr 2e-4 and weight decay 0.001. That is below bf16's
  resolution, so **step 0's optimizer leaves every A unchanged**.

The same holds at any scale. Rules 3 and 4 are replaced by the rules below. The milestone becomes **two optimizer steps** at the
registered shape.

**Step 0:**
- every adapter gradient is finite;
- every LoRA-B gradient has a nonzero element, and every B changes under `optimizer.step()`;
- every LoRA-A gradient is exactly zero. A nonzero A gradient at step 0 means B was not zero-initialized, and the box is VOID as a
  harness error.

**Step 1** (after step 0 has moved B; same tokens, same shape, same box):
- every adapter gradient is finite;
- every LoRA-A gradient has a nonzero element, and every A changes under `optimizer.step()`;
- B's rules repeat.

Requiring step 0's A gradients to be zero cannot, alone, tell a zero-initialized B from a broken A path. Step 1 is what can: a bug that
zeroed A's gradient forever would fail it.

**Both steps:**
- no frozen tensor changes; the frozen-base hashes after step 1 equal those after load;
- each step's records are kept: loss, elapsed time, bytes moved, memory peaks, gradients and deltas.

**What else changes:**
- **Time-left fit (gate 2):** it includes step 1, which is one more forward, backward and update at the same shape.
- **Unchanged:** the fidelity read (step 0's forward) and the reproducibility floor (step 0's loss).

## Amendment 3 (2026-10-11, before any 235B box): the depth read, the contribution statistic, the plants on the box, and the run

**Why.** Gate 1 set and checked Amendment 1's bands on a 4-layer slice. Before any box, a full-depth test asked whether those bands,
and a per-layer contribution statistic, can see one wrong layer deep in a model. Its reading lines were registered before the runs.
Receipts: `bench/q235/depth-30b/`.

**The depth test** (2026-10-10, RTX A2000, $0). e4b's NF4 training forward over all 48 layers of Qwen3-30B-A3B, on the calibration
tokens, in three fresh processes: the box unplanted, and each of Amendment 1's planted errors at layer 24. Each is read against
Amendment 1's 48-layer streamed bf16 reference.

| run | full state: median / max | full state at L24 | min overlap | u_24 | loss gap |
|---|---|---|---|---|---|
| box | 0.107 / 0.204 | 0.107 | 0.913 | 0.346 | −0.0055 |
| routing plant | 0.112 / 0.241 | 0.112 | 0.890 | 1.063 | +0.0440 |
| epilogue plant | 0.2555 / 0.371 | 0.287 | 0.834 | 8.505 | +0.1902 |

u_i is defined below. Three independent implementations agree on every quantity to three decimals.
- **The full-state bands do not see either plant at depth.** The routing plant sits inside all three bands. The epilogue plant's
  maximum, 0.371, does not reach its must-fail line (> 0.50).
- **The contribution statistic sees both.** At the planted layer the routing plant reads 3.07× the box and the epilogue plant 24.6×.
  Both also clear 2 × the box's largest interior u (0.388, at layer 34) anywhere in the model.
- **Why the state dilutes** (exploratory). Position 0 of each row is an attention sink, with a hidden-state norm of about 1,217
  against 5–46 elsewhere. From layer 3 it holds 97–99.8 % of the reference's squared norm, so the mid-depth full state measures mostly
  that one position's error (0.106, set by about layer 3).
- **The router overlap cannot see the routing plant at its own layer.** The recorded top-k is the router's choice before the plant
  shifts it: layer 24 reads 0.9275 in all three runs.

**What changes.**

1. **Gate 1.** Its formal PASS stands. Its fidelity evidence covers a 4-layer depth only, and says nothing about the full-state bands
   at the box's depth.
2. **The full-state bands are void for any deep box,** under Amendment 1's void rule. The full-state profile, its median and its
   maximum are reported, not gated. The median is the upper middle, `s[⌊n / 2⌋]` of the n sorted layers, as `q235_compare` computes it.
3. **Rule 7 for the box: the contribution statistic.**
   - **Definition.** u_i = ‖d_box,i − d_ref,i‖ / ‖d_ref,i‖, where d_i = h_i − h_(i−1) is layer i's update to the residual. The norm
     is Frobenius over the real (unpadded) positions, in fp32; layer 0 has no stored input and no u.
   - **The interior class:** layers ⌈4n / 48⌉ to ⌊44n / 48⌋, the 30B calibration's layers 4–44 scaled to the model's depth. That is
     layers 8–86 of 94.
   - **STRICT band:** u_i ≤ **0.775** on every interior layer. This is 2 × the 30B box's largest interior u (0.3876). It is a 30B
     calibration applied to the 235B model, and (4) below is what makes that safe.
   - **Reported without a bar:** u_1–u_7 and u_87–u_93, and the box's full u profile.
   - **The null and the ceiling, before the bar:** on the 30B calibration the unplanted box's interior maximum (0.388) sits below the
     band (0.775), and both plants sit above it (1.063 and 8.505).
4. **The box's own plants, at its own depth.** The fresh-process repeat (`q235_box.py --repeat --plants`) runs, in that one process:
   the clean step-0 forward (the **null**, which is also rule 6's repeat), then the routing plant's forward, then the epilogue
   plant's forward. Each plant is at layer ⌊n / 2⌋ = 47, as Amendment 1 registers, and is removed before the next forward.
   - **The null must be the box.** Its per-layer outputs are expected to equal the box's step-0 outputs bit for bit, and did in every
     rehearsal: on the 30B model at full depth and on the 235B slice. Another card class may differ in the last place, so the registered
     tolerance is a per-layer relative difference ‖null − box‖ / ‖box‖ ≤ **1e-3** on every layer, about 100× below the box's own
     NF4-against-bf16 error. Bit-for-bit equality is reported. Beyond the tolerance the instrument broke, and the box is **VOID**,
     not FAIL.
   - **Each plant must engage, and only at its layer.** Its layers before 47 must match the null's within the same tolerance,
     ‖plant_i − null_i‖ / ‖null_i‖ ≤ 1e-3 for every i < 47. In-process equality was bit for bit in every rehearsal, but the RTX A2000
     cannot certify the registered card's dispatch: a reduction with atomics could differ in the last bit within one process. Bit-for-bit
     equality is reported. Its layer 47 must differ from the null's. Otherwise: VOID.
   - **The unplant is checked structurally,** in code. After each plant comes off, the planted layer's hooks, instance-level forwards
     and expert activation must equal their snapshot from before it, by identity. Otherwise the repeat raises, a harness error (VOID).
     The routing plant cannot leak into the epilogue forward.
   - **Each plant must fail the band:** its u_47 > 0.775. A plant inside the band means the band cannot see that error on this model:
     **VOID**. That is the guard against a band too loose for the 235B model.
   - **The box must pass the band.** An unplanted box above it is a **FAIL**, reported with its full u profile. That is the guard against
     235B NF4 noise beyond the 30B's.
   - **Both directions fail closed.**
   - **Rehearsed, $0.** One such repeat process on Qwen3-30B-A3B at full depth, on the RTX A2000, reproduces the depth test's three
     separate processes bit for bit: the null is the box, each plant equals its separate run on all 48 layers, and u_24 reads
     0.346, 1.063 and 8.505 again (`depth-30b/`). Each in-process plant's forward took about 5.5 s.
5. **The router overlap moves to reporting.** Making it STRICT again needs the top-k actually executed recorded, and a fresh depth test.
6. **Unchanged:** rules 1–6, Amendment 2's STRICT rules by LoRA factor, and the reproducibility floor (1e-4 relative on step 0's loss).
   The loss gap is reported, not gated.

**A $0 calibration on the 235B model itself at depth: what it can and cannot give.** A 235B slice fed the streamed bf16 reference's
hidden state at about layer 45 could run NF4 and bf16 side by side on the RTX A2000. That gives the 235B model's **local** contribution
error at depth: each layer's NF4 error from an exact input. It does not give the box's u, which also carries the drift of the 44 NF4
layers before it. Turning the local error into a band would take a local-to-propagated ratio from the 30B model, which is again a 30B
calibration. It would also need about 45 layers of the 235B bf16 reference on the RTX A2000, plus a slice-input harness that does not
exist yet. So the band stays as in (3), with (4)'s plants on the box as the 235B model's own sensitivity check. The box's own unplanted
u profile, reported in full, is the 235B measurement.

**The registered run.**
- **Host class:** an RTX PRO 6000 96 GB (the box refuses any other card), with at least 192 GB of host RAM and 550 GB of free disk,
  as registered. The host is pinned by the ACK, and nothing substitutes for it.
- **Shape A, as run.** The 8 rows of TC1's field text the calibration used (`ds_alpaca.json`, sha256 `5324987a…`), tokenized by the
  checkpoint's own tokenizer. This is one micro-batch, right-padded to the longest row: 8 × 331 = 2,648 positions, of which 1,014 are
  real. The 235B tokenizer gives exactly the calibration's token ids. The capacity section's 2,048-token rows describe the planned
  shape; this is the run's.
- **Capacity at that shape,** from e4b's estimate on main at 13b55990 (`gate2/capacity-2648.json`), with r 16 / alpha 16, NF4
  attention, experts in pinned host RAM, the grouped NF4 kernel and AdamW8bit:

  | adapters | device | pinned host | host-to-device per micro-batch |
  |---|---|---|---|
  | bf16 (the box) | 24.76 GiB | 158.62 GiB | 237.94 GiB |
  | fp32 | 34.82 GiB | 158.62 GiB | 237.94 GiB |

  At the planned 2,048 positions e4b's estimate was 23.95 GiB of device memory for A-bf16 (the capacity section). The planner totals
  there (29.24 GiB for A-bf16) add a CUDA context and the planner's reserve, and are not re-run here; the registered class's budget
  after headroom is 90.81 GiB.
- **Pins:** experts4bit-qlora at the launching checkout's merged HEAD, which must contain this amendment; grouped-nf4-gemm at
  `724ccc45`; every wheel hash-locked (`wheels.json`). Each pin is checked on the box against pip's record.
- **Sealed caps and floors** (`q235_run.sh`, fitted in `gate2/`). The time-left caps:

  | phase | cap (s) |
  |---|---|
  | require-git | 240 |
  | create-venv | 120 |
  | install | 1,800 |
  | hostrate | 600 |
  | fetch | 12,000 |
  | tokens | 300 |
  | box | 8,820 |
  | repeat (null + two plants) | 4,710 |

  The host floors are a pinned host-to-device rate of 5.8 GB/s, a host copy of 5.14 GB/s, an O_DIRECT disk read of 0.743 GB/s and a
  checkpoint pull of 0.062 GB/s. A host below any of them is refused before the step.
- **Cost:** the worst-case wallclock is the caps, the runner's 300 s reserve and 1,800 s for the controller (its receipt fetch now
  carries about 8.2 GB of per-layer outputs): 8.53 h. That is $18.76 at $2.20 an hour and $13.64 at $1.60. The launcher's per-run
  cap is $35, and a run above $15 needs its ACK approved.

**Gate status.**

| gate | state |
|---|---|
| 1. rehearsal | the core PASS stands (4-layer depth only, as above). The driver's lifecycle passed on CPU and on the RTX A2000, and passed again on both with the in-process plants at f7714702 (`gate2/`) |
| 2. time-left fit | passes with the caps above: the first A2000 leg's phases plus the two plant forwards (3.38 s on the slice), scaled (`gate2/`) |
| 3. fetch gate | passed over every locked URL; it re-runs within 24 h of a launch, and the drive refuses without a fresh report |
| 4. the box installs its own tools | yes (`q235_bootstrap.py`, `dq11_require_git.sh`) |
| 5. no placeholders | the fetch gate scans the closure, the locks, the drive and this registration |
| 6. budget | as above |

**After the box** (off the box, $0). The layer-streamed bf16 reference runs on the box's own tokens on the RTX A2000, about 2.4 h.
`q235_verdict.py` then applies the rules mechanically:
- PASS iff rules 1–6, Amendment 2's STRICT rules and this amendment's band all hold;
- FAIL when the box fails any of them;
- VOID on the instrument checks in (4), a refusal, a host fault, a time-left guard or a harness error.

Nothing is quoted before that verdict is re-derived independently on CPU.

## Amendment 4 (2026-10-16, after box `q235-pro6000-2`, before any further 235B box): the check copies move to the host

**Why.** The box's step recorded a device peak (`max_memory_allocated`) of 58.85 GiB at step 0 and 58.91 GiB at step 1. That is
2.38× the registered 24.76 GiB estimate. The box record and the script show that the gap is not a training-step term:
`q235_box.py` kept four fp32 copies of every adapter tensor on the device for its checks, all of them alive at the step's peak.

| copy | used by |
|---|---|
| `g0`, `g1` | the in-process pair's gradient spread |
| `before` | the optimizer delta (the parameters before the step) |
| `grads` | this step's gradient checks (finite, nonzero counts, global norm) |

- **Size.** Each copy is 2 × 4.685 GiB (the bf16 adapters held as fp32), 9.369 GiB; the four together are 37.476 GiB.
- **Accounting.** The estimate's terms co-resident at that peak are everything but the activations, which are freed after the
  backward: 21.18 GiB. With the copies that accounts for 58.655 GiB. The residual is +0.195 GiB (0.33 %) at step 0 and +0.255 GiB
  (0.43 %) at step 1.
- **Where the peak is.** Step 1 exceeds step 0 by 0.06 GiB, not by the optimizer state's 4.76 GiB, so the peak sits inside the
  optimizer step in both steps.
- **Consequence.** The reported device deviation and its estimator-issue flag measured the harness. They fire on any run of the old
  script at this shape.

**What changes.**

1. **The copies.** `q235_box.py` takes all four with `host_copy`: the tensor leaves the device at its own dtype, is widened to fp32
   on the host, and never shares storage with its source. The checks compute the same quantities on the host copies, and the
   optimizer delta is `host_copy(p) − before`. No rule, threshold or verdict line changes.
2. **The step record** adds `grads_host_copy_s`, the time taken to move `grads` off the device. It falls inside `elapsed_s`, as the
   device clone did, and is recorded so that step times stay comparable with the first box.
3. **Expected on the next box** (reported, not gated). The device peak falls by about the copies' 37.5 GiB, to the estimate's
   co-resident terms plus the residual.
4. **The host-RAM floor: 192 GB becomes 235 GB** (`q235_run.sh`, science class). The copies are anonymous host memory, which the
   kernel cannot reclaim, so the floor is derived again with the same method as the registered one (refusal point 4):
   - **Registered:** 158.63 GiB of pinned expert homes plus a 3.0 GiB process baseline, 161.6 GiB (173.5 GB), plus 18.5 GB for the
     OS and the loader's unmodelled transients: 192 GB.
   - **Added:** the four host copies, 37.476 GiB (40.24 GB). Also the step-0 layer outputs that the box holds on the host for the
     fidelity read, 94 × 2,648 × 4,096 × 2 B = 2.04 GB, alive through the step and not itemized before.
   - **Floor:** 173.5 + 40.24 + 2.04 + 18.5 = 234.3 GB, so 235 GB.
   - **What it excludes.** The first box's peak RSS, 517.4 GB on a 1,081 GB host, is a lifetime peak (`ru_maxrss`). It includes the
     checkpoint's memory-mapped file pages, which are reclaimable, so it does not size the floor.
   - **The record.** The box now records `host_memory` (`/proc/self/status`: `VmHWM`, `VmRSS`, `RssAnon`, `RssFile`, `RssShmem`) at
     the end of the load and of each step, after the optimizer step, when all four copies are alive. This is reported, not gated.
     At full depth, `RssAnon + RssShmem` at a step's end is expected at or below the floor less its allowance (215.8 GB). The
     rehearsal in (5) reads it at its own depth.
5. **Rehearsal.** The rehearsals behind gates 1 and 2 ran the old script. Before any further 235B box, gate 1's rehearsal on the
   RTX A2000 re-runs with this one, and the closure (`q235.sha256`) carries the new script and the new floor.
6. **Tests.**
   - `tests/test_q235_box.py`: the copies are fp32 host copies that never share storage with their source. `fwd_bwd` and
     `one_step` contain no `.clone()` or `.float()`, either of which would put a copy back on the device. The memory record is
     present.
   - `tests/test_q235_driver.py`: the science floor is this derivation's 235 GB, and the RAM-floor refusal test reads the floor from
     the script.

## Amendment 5 (2026-10-16, after Amendment 4's rehearsal, before any further 235B box): the host-RAM floor, measured

**Why.** Amendment 4 derived the science host's RAM floor, 235 GB, from the estimate's pinned expert homes, a 3.0 GiB process
baseline, the check copies, the held layer outputs and an 18.5 GB allowance. It also registered an expectation for the box's new
`host_memory` record: at a full-depth step end, `RssAnon + RssShmem` would be at most 215.8 GB, the floor less its allowance.

**The rehearsal** (gate 1's A2000 leg at 9f6e0088, `gate2/a2000-leg-am4-summary.json`). Every phase exited 0, Amendment 2's nine
rules held, the frozen base was unchanged (111 tensors), and the null and both plants behaved as before.
- **Device.** Amendment 4's device prediction holds. The step-0 peak fell by 1.194 GiB and the step-1 peak by 1.195 GiB, against
  1.197 predicted (three 0.399 GiB copies at the backward peak).
- **Host.** `RssAnon + RssShmem` at the step end was 13.53 GB, against 12.27 GB for Amendment 4's terms at this depth. `RssShmem`
  alone was 2.25 GB above the estimate's pinned homes.

**The scaling read** (`gate2/memslope.json`, $0 on the RTX A2000). The box's `--repeat`, with the leg's environment, ran in a fresh
process at 2, 4, 6 and 8 layers of the same checkpoint. The load-end `host_memory` was fitted against depth by least squares:

| term | per layer | fixed | largest residual |
|---|---|---|---|
| `RssShmem` (the pinned homes) | 1.8119 GB (1.6874 GiB) | 2.231 GB | 52 kB |
| `RssAnon` | 0.015 GB | 3.780 GB | 0.14 GB |
| `RssAnon + RssShmem` | 1.8269 GB | 6.011 GB | 0.14 GB |

- **The estimator's pinned-homes term is exact per layer:** 1.6875 GiB estimated, 1.6874 GiB measured (ratio 0.99995).
- **The excess is fixed, not per layer.** It is about 2.23 GB of shared memory and an anonymous baseline about 0.56 GB above the
  3.0 GiB assumed, plus 15 MB of anonymous memory per layer. The estimator does not price these. That is a follow-up for the
  estimator, not a change here.

**The floor, derived again.**
- At the load end, at 94 layers: 6.011 + 1.8269 × 94 = 177.74 GB.
- Add the four check copies (40.24 GB) and the held layer outputs (2.04 GB): **220.02 GB at the step end.**
- **No heap reuse is assumed. That is the conservative choice.** At 4 layers the copies reused about 1.7 GB of heap that the load had
  freed, but that cannot cover 40 GB at full depth.
- 220.02 GB exceeds Amendment 4's 215.8 GB expectation, and 235 GB less the allowance (216.5 GB). With the registered 18.5 GB
  allowance the floor is 238.52 GB: **`q235_run.sh` now requires 239 GB of host RAM for the science class.**
- The expectation for the box's `host_memory` becomes `RssAnon + RssShmem` at most 220.0 GB at a step end. It is reported, not gated.

**Tests.** `tests/test_q235_driver.py` pins the science floor to this derivation's 239 GB.

## Amendment 6 (2026-10-16, after Q235's PASS, before any Shape B box): Shape B, and the accumulation it rests on

**Why.** The registration names Shape B "for comparability": the field recipe's micro-batch 2 × accumulation 4. The first box ran
Shape A only, one micro-batch of 8 rows. The question Shape B answers is whether the milestone's step extends to the shape TC1 trains
at. Under accumulation, every expert crosses the link four times per optimizer step, and four backwards run before `optimizer.step()`.
Anything in the offload, eviction or fused paths that assumes one backward per step, or that clears or overwrites `p.grad` between
micro-batches, changes the gradient here and nowhere in the first box.

**The step at Shape B** (`q235_box.py --micro-batch 2`; `q235_run.sh` runs the box at it).
- The same 8 TC1 rows as four micro-batches of two rows, in order. Each micro-batch is trimmed to its longest row.
- Each micro-batch's mean loss is weighted by its share of the window's valid labels (125, 252, 480 and 149 of 1,006 on these rows),
  so the window's loss is the token mean over all 8 rows, as the field recipe's trainer normalizes. Shape A and Shape B are therefore
  the same gradient in exact arithmetic.
- Amendment 2's nine STRICT rules and the frozen-base hashes apply to both Shape B steps.
- Rule 6 (the fresh-process repeat) and rule 7 (Amendment 3's band, on the repeat's null and plants) stay on Shape A's forward,
  where the band is calibrated.

**Rule 8 as first registered, and its calibration: STOPPED.** The first form compared each step's accumulated gradient with Shape A's
at the same parameters (`g0` at step 0; a fresh Shape A gradient at step 1), per-tensor relative difference at most BAR. BAR's rule
(below, unchanged) was registered before its runs. The calibration ran on 2026-10-16 ($0, RTX A2000; `q235_box.py --micro-batch 2`
with the leg's environment; every depth exited 0 with all of Amendment 2's rules true):

| depth | largest per-tensor difference s | p50 | global cosine (step 0) | same-shape in-process spread |
|---|---|---|---|---|
| 235B, 2 layers | 0.0219 | 0.0055 | 0.999993 | 0.0034 |
| 235B, 4 layers | 0.0344 | 0.0093 | 0.999977 | 0.0053 |
| 235B, 8 layers | 0.0558 | 0.0203 | 0.999938 | 0.0078 |
| 30B, 48 layers | 0.5826 | 0.2342 | 0.981525 | 0.0294 |

- **The fit.** The exponent is 0.676 ± 0.014 (95 % interval 0.50 to 0.86) with residuals within 1.1 %. Extrapolated to 94 layers that
  gives s₉₄ = 0.29 (0.13 to 0.66). The 30B's s(48) sits 3.13× above the fit's value at 48 layers.
- **Two SOFT triggers fired:** the exponent is above 0.5, and the 30B is more than 3× off the fit. The p99 view gives no relief: it
  equals the maximum on the 235B cuts (12 to 96 tensors), and its exponent is the same.
- **The plants.** BAR by the rule would be 1.75. Both plants then read 0.19 to 0.59 BAR at every depth (mean of means 0.33 to 0.63,
  overwrite 0.88 to 1.04), so neither could be registered.
- **The reading.** Shape A against Shape B is not the accumulation. With the same 8 rows at the same parameters, the per-tensor
  gradients at 48 layers differ by a median of 0.23 and a maximum of 0.58 (cosine 0.98) from the micro-batch shape alone. The
  difference is spread evenly over the layers, attention as much as experts, and is about 12× the same-shape run-to-run spread at that
  depth. At full depth it reaches the mean-of-means plant's size, so it cannot gate. It stays reported.

**Rule 8, registered again: the same-shapes reference (STRICT, Shape B only).**
- **The comparison.** Each step's accumulated gradient (the four backwards into `p.grad`) is compared with the host fp32 sum of the
  same four micro-batches' weighted gradients, each taken on its own (`p.grad` cleared before each backward). Both sides use the same
  shapes and the same parameters, so they differ only in how the backwards accumulate. The reference is taken at step 0's parameters
  before the plants, and at step 1's parameters before that step's accumulation (`ga_mb_check`).
- **The STRICT quantity** is the per-tensor relative difference against that reference: at most **BAR** on every tensor whose
  reference is nonzero. A tensor exactly zero in the reference must be exactly zero in the accumulation.
- **The plants**, the same two (mean of means; overwrite), are read against the same reference at step 0's parameters. Each
  REGISTERED plant must read above BAR, else **VOID**.
- **Reported:** the global cosine and the norm ratio; Shape A against Shape B in every view (`ga_check`); and the router-flip count.
  - **Router flips:** per layer, the real tokens whose top-k expert set differs between Shape A's forward and Shape B's micro-batch
    forwards at step 0's parameters. If such flips explain the A-against-B gap, that is a finding about batch-variant MoE training in
    bf16, reported with prior art and claiming nothing here.
- **Host memory.** At most four fp32 host copies are alive at once, as Amendment 5 prices: `g0` and the reference with the step's
  `before` and `grads`; at step 1, the fresh Shape A gradient and its reference.
- **Cost on the box:** four more micro-batch passes per step, about 4 minutes.

**BAR, read by the rule already registered, on the new quantity.**
1. **Runs** ($0, RTX A2000): the box at `--micro-batch 2` on 2-, 4- and 8-layer cuts of `ac9c66cc` and on Qwen3-30B-A3B at its 48
   layers, with the same tokens.
2. **s(n)** is the larger of the two steps' largest per-tensor difference against the same-shapes reference. p50 and p99 are fitted
   the same way and reported.
3. **The fit:** log s against log n over the three 235B cuts, extrapolated to n = 94 (s₉₄), with the exponent's interval from the
   residuals. The 30B's s(48) is reported against the fit at 48.
4. **BAR = 3 × max(s₉₄, the 30B's s(48)).**
5. **The plants are fixed in advance, before the data.** A plant registers only if it clears BAR by at least 3× at every calibration
   depth. If mean of means falls short, it stays unregistered and the overwrite plant alone carries rule 8's instrument check. If
   neither clears, the amendment stops.
6. **The SOFT stops are unchanged:** an exponent above 0.5, a point more than 30 % off the fit, or the 30B more than 3× off. Any of
   them stops the amendment, which then proposes a form for review before any box.

**The diagnostic, as run** (2026-10-16, the 4-layer slice). Under `torch.use_deterministic_algorithms(True, warn_only=True)` with
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, one warning appeared: memory-efficient attention's backward defaults to a nondeterministic
algorithm. Shape A's in-process gradient pair became bit for bit equal (spread 0.0, against 0.0053 by default). The same-shapes
difference fell from 0.0039 to 0.0033. Shape A against Shape B (0.0343) and the router flips (140 of 4,056 token-layers) did not
change, so those are effects of the batch shape alone.

**The second calibration, as run: one SOFT trigger.** Every depth exited 0, with all of Amendment 2's rules true and no zero mismatch.

| depth | s (same-shapes) | p50 | same-shape in-process spread | plants: mean / overwrite | Shape A vs B | router flips |
|---|---|---|---|---|---|---|
| 235B, 2 layers | 0.00309 | 0.00261 | 0.0034 | 0.331 / 0.875 | 0.022 | 2.1 % |
| 235B, 4 layers | 0.00393 | 0.00291 | 0.0053 | 0.342 / 0.876 | 0.034 | 3.5 % |
| 235B, 8 layers | 0.00431 | 0.00311 | 0.0069 | 0.341 / 0.894 | 0.056 | 5.2 % |
| 30B, 48 layers | 0.02663 | 0.00924 | 0.0273 | 0.645 / 1.036 | 1.114 | 16.2 % |

- **The fit.** The exponent is 0.239 ± 0.061 (95 % −0.54 to 1.02) with residuals within 5 %, giving s₉₄ = 0.0079.
- **The trigger.** The 30B sits 3.93× above the fit at 48 layers: the trigger fired. The other two stops were clean.
- **What BAR would have been.** The rule gives 0.080, which both plants clear at every depth (mean 4.1× to 8.1×, overwrite 11× to 13×).
- **The reading.** The 30B's s(48) equals its own same-shape run-to-run spread (0.0273): it measures the backward's nondeterminism
  compounding over 48 layers, not the accumulation.
- **Why the bar cannot stand.** At 94 layers that spread extrapolates to about 0.02 to 0.03, so a bar of 0.080 would leave rule 8
  about a coin flip from a VOID on the box.

**Rule 8, third form: the check passes under deterministic algorithms (STRICT, Shape B only).**
- **The check passes.** At each step's parameters, before that step, rule 8's check runs under
  `torch.use_deterministic_algorithms(True, warn_only=True)`:
  - the host-sum reference;
  - the device accumulation, a pass of its own, four backwards into `p.grad`;
  - a second accumulation, for the floor.

  The plants are check passes too. The previous setting is restored when the check ends, however it ends. The registered steps
  (rules 1 to 7, the fidelity read, the optimizer steps) run on the shipped default path, and every pass records which mode it ran in.
- **The STRICT quantity** is the per-tensor relative difference between the deterministic accumulation and the deterministic
  reference: at most **BAR** on every tensor whose reference is nonzero. A tensor exactly zero in the reference must be exactly zero
  in the accumulation.
- **Instrument checks (VOID):**
  - the check ran deterministic, with the default restored for the step;
  - the floor (the two deterministic accumulations against each other) is at most **BAR / 3**;
  - each registered plant reads above BAR.
- **Reported:**
  - the registered step's own default-mode accumulation against the deterministic reference;
  - Shape A against Shape B in every view;
  - the router flips.
- **Host memory and cost.** At most four fp32 host copies are alive at once. The check adds eight micro-batch passes per step on the
  box, about 10 minutes in all.
- **One scoping check before the third calibration.** `CUBLAS_WORKSPACE_CONFIG` is read when cuBLAS creates its handle, so it cannot
  be limited to the check passes. The calibration session first runs the 4-layer in-process pair under the torch flag alone. If that
  pair is bit for bit equal, the form stands as written. If not, the amendment stops for review before the calibration.

**BAR, from a third calibration,** read by the rule already registered (the depth fit, 3 × max(s₉₄, the 30B's s(48)), plants at least 3×
BAR at every depth, the SOFT stops). It uses the third form's quantity, the 235B cuts at 2, 4 and 8 layers, and the 30B at 48.

**The third calibration, as run: BAR = 0.0231, both plants registered** (2026-10-16, RTX A2000, $0; `bench/q235/am6/`).
- **The scoping check passed.** The 4-layer in-process pair under the torch flag alone, with `CUBLAS_WORKSPACE_CONFIG` unset, was
  bit for bit equal. The check passes therefore need no process-wide setting, and the registered steps keep the default path.
- **Modes.** At every depth, the check ran deterministic, the default was restored, and the steps ran on the default. Amendment 2's
  rules all held, and no tensor was a zero mismatch.

| depth | s (the deterministic check) | p50 | floor (two deterministic accumulations) | plants: mean / overwrite, in BAR |
|---|---|---|---|---|
| 235B, 2 layers | 0.00278 | 0.00249 | 0 at both steps | 14.3 / 37.9 |
| 235B, 4 layers | 0.00332 | 0.00248 | 0 at both steps | 14.8 / 37.9 |
| 235B, 8 layers | 0.00282 | 0.00247 | 0 at both steps | 14.8 / 38.7 |
| 30B, 48 layers | 0.00770 | 0.00427 | 0.0079 / 0.0087 | 29.0 / 44.9 |

- **The fit is flat:** exponent 0.010 ± 0.143, residuals within 12.1 %, so s₉₄ = 0.00305. The 30B sits 2.54× the fit at 48, inside 3×.
  No SOFT stop fired, and the p99 view is clean.
- **BAR = 3 × max(0.00305, 0.00770) = 0.0231** (`q235_verdict.GA_BAR`). Both plants clear it at least 14× at every depth, so both are
  registered.
- **The floor, read plainly.**
  - It is exactly 0 on the 235B's shapes at every depth measured.
  - On the 30B's shapes it is not: 0.0079 and 0.0087, just above BAR / 3 = 0.0077, so a 30B box would read rule 8 VOID. Some
    operation at those shapes stays nondeterministic under the torch flag. It is not identified; `warn_only` leaves
    memory-efficient attention's backward as it is, and custom kernels sit outside the flag.
  - The 235B box runs the cuts' layer shapes at 94 layers. Equal inputs stay equal through more layers of the same shapes, so its
    floor is expected to be 0. If it is not, the instrument check VOIDs rule 8 rather than reading noise as a FAIL.
- **Reported:**
  - **Router flips** are 2.1 %, 3.5 %, 5.2 % and 16.2 % of token-layers.
  - **Shape A against Shape B** reads as in the second calibration.
  - **The wording.** Shape A and Shape B are the same gradient in exact arithmetic; under bf16, routing differs by a few percent of
    tokens per layer, growing with depth.
- **Re-derivation.** `am6/am6_fit.py` re-derives each calibration's figures from its box records. Its per-depth summaries are
  `am6/calibration-1-a-vs-b.json`, `-2-same-shapes.json` and `-3-deterministic.json`.
- **The gate-1 rehearsal** on the RTX A2000, after this merges, measures the floor again on the merged code.

**The fp32 arm** stays unrun.

## Amendment 7 (2026-10-16, after Amendment 6, before any Shape B box): the MoE experts' staging and compute, timed in the box (REPORTED only)

**Why.** Training stages each MoE layer's experts synchronously: a whole-layer host-to-device copy before every experts call, twice
per micro-batch under gradient checkpointing (the forward and the recompute), and nothing overlaps it. Whether projection pipelining
or whole-layer prefetch is worth registering depends on C: each layer's experts compute time against its staging time. C depends on
the card's link (the RTX A2000 runs PCIe gen3 x8; the registered class, a PRO 6000, runs gen5), so a rehearsal-only reading would
answer for the wrong hardware. The box takes the reading at no extra rental.

**The pass** (`q235_box.py --time-moe 8 --time-moe-bound`; `q235_run.sh` runs the box with it).
- **When.** After both Shape B steps, their frozen-base hashes and every registered record, the box runs one extra forward-backward
  of the registered shape: Shape B's token accumulation over the four micro-batches, with no optimizer step. The steps run before any
  of its hooks exist.
- **Where.** On 8 evenly spaced decoder layers: 0, 13, 27, 40, 53, 66, 80 and 93 of 94. A slice with fewer layers is timed on every
  layer.
- **What, per layer and per micro-batch.** CUDA events, in stream order, around each experts module call:
  - `stage_ms`: across e4b's `_stage_pre_hook`, the synchronous whole-layer copy;
  - `compute_ms`: from the end of staging to the module's return, gate_up, the activation and down together (one fused call);
  - both of those in the forward and in the checkpoint recompute;
  - the module's backward window. `own_ms` is its part after the recompute, the dgrad and the LoRA gradients.

  Each pair sits beside its micro-batch's rows, width and valid labels.
- **Checkpoint early stop.** It is off for this pass only, so a non-reentrant recompute reaches the module's return.
- **The overhead read** (`--time-moe-bound`, registered here before the gate-1 rehearsal reads it):
  - One discarded warm-up pass, then four in ABBA order in the same process: untimed, timed, timed, untimed.
  - `overhead_s` is the mean of the two timed passes minus the mean of the two untimed ones.
  - `half_s`, the read's order and drift noise, is half the difference between the two timed-minus-untimed pairs,
    |(on₁ − off₁) − (on₂ − off₂)| / 2. Under a linear drift d per pass, the pairs read overhead + d and overhead − d, so their mean
    is the overhead and half their difference is |d|.
  - `pairs_s`, both pair differences, is in every record, so the read can be recomputed from it.
  - The read is decision-grade only when `half_s` ≤ `overhead_s` / 2. Otherwise it is reporting-only, and the refit below does not
    use it.
  - ABBA cancels a linear drift, and the warm-up removes the cold-first bias.
  - Each pass's device-to-host copy of the adapter gradients is recorded and left out of every comparison.
  - Both timed passes carry the hooks, so each sampled layer holds two pairs per micro-batch.

**What it is not.**
- REPORTED only. It lives under `moe_timing` in the box record. `q235_verdict.py` reads none of it, and no rule, threshold, verdict
  line or STRICT check reads it. A failure inside the pass is recorded as `error` and never fails the box.
- With the flag off, the box record is unchanged in every existing field.
- gate_up and down are not separated. They run as one fused call, and separating them would move the experts4bit-qlora pin.

**Its cost, and when it leaves the box.**
- The read adds five Shape B forward-backwards without an optimizer step to the box phase: the warm-up, two untimed and two timed.
- The gate-1 A2000 rehearsal at the merge of Amendments 6 and 7 is one rehearsal for both. It measures `overhead_s`, `half_s` and the
  passes' time on the slice.
- Gate 2's time-left fit then re-runs from that rehearsal with the pass in the box phase. The refitted box cap and the run's worst-case
  wallclock and cost are recorded here before any box. Both are UNKNOWN until that rehearsal runs.
- If, with the pass, the run exceeds 6 h at the slow end of the PRO 6000 spread, the pass leaves the box (the run line drops the flag)
  and stays in the rehearsals. The maintainer decides that on the measured numbers, before any box.

**Recorded at the gate-1 A2000 rehearsal** (e4b f17f0744, the merge of Amendments 6 and 7; 2026-10-17, $0; records in `gate2/`):
- **Amendment 6.** Rule 8's floor read exactly 0 at both steps. Its check read 0.00278 and 0.00393 against BAR 0.0231. Amendment 2's
  rules held (9 of 9), and the frozen base was unchanged across both steps. This is the A2000 correctness gate; the registered verdict
  is the PRO 6000 box's.
- **Amendment 7's overhead read.** `overhead_s` 0.0513 s (0.57 % of the pass), `half_s` 0.0109 s: decision-grade. The maintainer kept
  the pass in the box.
- **The refit** (`q235_fit.py --box-from` this leg, `--box-baseline` the Amendment 4 leg 9f6e0088):
  - the change adds 142.8 s outside the load on the slice (222.0 s against 79.2 s), so the box basis is 392.8 s;
  - the box's budget at 94 layers is 13,847 s;
  - **the box cap is 13,847 s** (it was 8,820). No other cap moves.
- **Wallclock and cost.** The launcher's declared wallclock, 6.0 h, bounds the run. The expected run is about 4.4–4.6 h (scaled
  phases, the 300 s reserve and the controller's 1,800 s). The Shape B packet states both figures from its own quote, under the
  maintainer's bounds:
  - expected all-in (scaled hours × quote + the 470.2 GB checkpoint × the listed $/GB + upload) at most $11.50;
  - worst case (6.0 h × quote + download + upload) at most $15.00;
  - advertised download at least 3,000 Mbps.
- **C on the A2000** (reporting only: PCIe gen 3 x8, and the card is correctness-only): staging about 222 ms per experts call,
  against 44–60 ms of compute. No pipelining or prefetch decision comes from it; the PRO 6000 box's own read decides.

## Amendment 8 (2026-10-17, after the R2 gate-1 ruling, before any further 235B box): the dX record and the held-out read

**Why.** R2-Mixtral's gate 1 asks that "Q235 has reached L2 and reproduced it" (`bench/fam235/PREREG-r2-mixtral.md:112`). The family
ladder's L2 (the fam235 matrix, draft 2.1 §5) is "full-depth nonzero LoRA dX and adapter gradients, frozen-base hashes before and
after, one real optimizer update, a held-out check".
- The maintainer's evidence packet maps it to runs 2 and 3 field by field.
- The maintainer ruled gate 1 **not yet met** on two gaps:
  1. no receipt records dX: it is only inferred from step 1's nonzero A gradients;
  2. no receipt holds a held-out check: this registration scoped one out ("What this registration cannot say").
- Everything else is met on both runs. Reproduction stands as rule 6 (both runs, gap 0.0) plus the bitwise cross-run forward (all 94
  layer outputs and the step-0 loss equal).
- Both boxes ran on Vast machine 8652. That is a stated limitation, not a blocker.

**1. The dX record** (`q235_box.py`, every stepped run).
- **What.** Each step records `dx`: every decoder layer's output gradient dL/dh_i, i = 0 … n−1, as its fp32 norm. It is taken once per
  forward-backward of the step's own pass: one under shape A, one per micro-batch under shape B.
  - It is the gradient that the layer's own adapters, and every layer below it, receive. Layer 0's is the deepest the backward must
    reach.
  - It is not the LoRA branch's own input gradient. At step 0, B is zero, so that term is exactly zero by construction (Amendment 2).
    The registered quantity is the full gradient arriving at each layer, from which every adapter gradient is computed.
- **How** (`DxProbe`).
  - Layer i's output is taken where it is an outer graph tensor: as the input of the next module, through a forward pre-hook on
    layer i+1, and on the final norm for the last layer. A tensor hook registered there receives dL/dh_i in the backward. That
    holds under either checkpoint mode.
  - The probe is armed around the step's own forward and disarmed before the backward. So the checkpoint recompute, which calls the
    modules again inside the backward, registers nothing.
  - A layer whose output carries no gradient, or whose hook never fires, stays unrecorded and fails the rule. The record fails closed.
  - The in-process pair, rule 8's check passes and the timed pass carry no probe.
  - **Found at the 30B rehearsal, before any box.** The first form hooked each layer's own output. On the CUDA fast path it recorded
    nothing: 192 of 192 entries were `None`, and the STRICT rule failed closed (exit 3), as designed. e4b's fast path checkpoints
    reentrantly (`ckpt_offload`), which runs the checkpointed forward without gradients, so the output a layer's own hook sees never
    carries one. The CPU tests ran HF's non-reentrant checkpoint and passed. A CPU test now shows the old hook recording nothing
    under a reentrant checkpoint, and the new probe recording every layer under both modes.
- **Rule 9 (STRICT).** At both steps, every layer's dX in every pass is recorded, finite and nonzero.
  - In the record: `step0_dx_finite_nonzero` and `step1_dx_finite_nonzero` in `rules`.
  - `strict_failures` requires them on any record that carries `amendment8`, fail-closed. A failure stops the run after the box
    (exit 12), as Amendment 2's do.
  - The verdict reports it as `9_dx`, with each step's smallest norm.

**2. The held-out read** (`q235_box.py --heldout`; `q235_run.sh` passes it to the box).
- **The registered held-out rows.**
  - They are the next `ROWS` (8) rows of the same split: rows 8–15 of `ds_alpaca.json`'s `train`.
  - `q235_tokens.py --heldout-out` writes them in the tokens phase, rendered, tokenized and padded as the step's rows. They are
    disjoint from the step's rows by construction.
  - Their sha256 is in the record.
- **Three reads, each one no-gradient forward at shape A on those rows.**
  - Before the update, after the in-process pair. B is zero then, so this is the frozen base's loss on them.
  - After step 0.
  - After step 1.

  They are recorded under `heldout`, with `rel_change_step0` and `rel_change` = (read − before) / before. `heldout_finite` (STRICT)
  requires all three reads finite.
- **The step's own rows are read the same way at the same three points** (`heldout.train`, with their `rel_change_step0` and
  `rel_change`). This is reporting only, not a pass/fail input. It lets a reader see that a held-out drop is shared with the step's
  own rows, not an artifact of the held-out rows.
- **Rule 10 (the verdict's), as the maintainer re-ruled it: a ONE-SIDED blow-up bound.**
  - Every held-out read must be finite.
  - Each post-update read (after step 0 and after step 1) must be at most `before × (1 + HELDOUT_BAND)`, with `HELDOUT_BAND` =
    **0.10**: an increase over 10 % fails.
  - There is **no lower bound**. The rule asks only whether the update made held-out text worse. It is not an equivalence test, and
    not a claim about quality after training, which stays out of scope.
  - **Why one-sided.** The first ruling, a two-sided cap of 0.10, would have failed a healthy update. At the 30B rehearsal (below)
    the held-out loss fell 38.6 %: two steps at the registered rate teach the shared Alpaca template, and the step's own rows fell
    the same way. The 235B's step loss halves likewise on runs 2 and 3 (1.24 → 0.65). A large drop is learning, not corruption.
  - An Amendment 8 box without its held-out read is a harness error. The verdict code also keeps rule 10 VOID for an unset band;
    a test exercises that path.
- **The 235B box records all three reads** and both relative changes. A later amendment tightens the cap from them, at the scale the
  rule applies to.
- **The 30B A2000 leg** is the mechanics rehearsal: recorded, not band-setting ($0).
  - Qwen3-30B-A3B runs this amendment's box step at full depth on the 30B calibration's path, with the held-out read over the same
    registered rows.
  - The rows are tokenized as the step's: the 30B's `tokenizer.json` is the same file as the 235B's (sha256 `aeb13307`), as in
    Amendment 6's calibrations.
  - **First run** (2026-10-17, e4b `7942eb39`, the first probe):
    - Amendment 2's nine rules held, and the frozen base was unchanged.
    - dX recorded nothing: the probe bug above. The run exited 3, a STRICT FAIL on rule 9.
    - Held-out loss read 1.7656 before the update, 1.2301 after step 0 and 1.0838 after step 1 (`rel_change` −0.386), all finite.
      The step's loss fell 1.871 → 0.910.
  - **At the fixed probe** (2026-10-17, e4b `1c12341c`; records in `bench/q235/am8/`):
    - The CUDA re-check ran first, on the 4-layer slice of `ac9c66cc`. It exited 0. dX was finite and nonzero at every layer in
      every pass at both steps (4 × 4), and every rule held.
    - Then the 30B leg ran. It exited 0.
      - dX was finite and nonzero at all 48 layers in all four passes at both steps (smallest 0.0068).
      - Amendment 2's nine rules and this amendment's held, and the frozen base was unchanged.
      - Held-out loss read 1.7656 → 1.2280 → 1.0851 (−30.4 %, −38.5 %), all finite. The step's own rows read 1.8592 → 0.6075.
      - Under rule 10 as ruled, it reads PASS: every held-out read fell.

**3. Host independence.** Per the ruling, the next Q235 box runs on a Vast machine other than 8652. Its envelope or packet names its
host by inclusion, and 8652 is not among them. That records the reproduction across hosts, the gate's stated limitation.

**4. Cost.** Six no-gradient forwards at shape A on 8 rows are added to the box phase: three on the held-out rows and three on the
step's rows. With the gradient path, box 1's forward took 7.2 s, so the reads add about 45 s. That is inside the 13,847 s cap's
margin. The dX hooks add n fp32 norms per pass.
- The gate-1 A2000 leg re-runs at this amendment's merge with this script. It records `dx` and `heldout` on the slice, as the
  mechanics' rehearsal.
- If the refit from that leg moves `CAP_BOX`, the new cap is recorded here before any box.

**What it is not.**
- No threshold of rules 1–8 changes.
- Records without `amendment8` read exactly as before: runs 2 and 3 carry neither field, and their verdicts are unchanged.
- The probe and the reads change no registered record. With and without `--heldout`, the step's losses, the dX norms, the layer
  outputs and the frozen hashes are equal (tested).

**Tests.**
- `tests/test_q235_box.py`:
  - every layer's dX is recorded, finite and nonzero, at both steps (four passes under shape B);
  - the probe records the armed pass and never the recompute, under a reentrant and a non-reentrant checkpoint;
  - the first probe's output hook records nothing under a reentrant checkpoint (the rehearsal's bug, proved on the CPU);
  - the held-out reads are finite and change no registered record, and the step's rows are read beside them;
  - `--heldout` rides the stepped run only;
  - the new STRICT rules fail closed.
- `tests/test_q235_verdict.py`:
  - PASS as "rules 1-7, 9-10" and "rules 1-10";
  - a zero, non-finite or missing dX fails rules 9 and 3–5;
  - rule 10 is one-sided: −39 % and +9.5 % pass, while +10.5 % at either read, an infinite read and a NaN read fail;
  - VOID while the band is unregistered, and on a missing read;
  - records from before this amendment read as before.

**The closure.** `q235.sha256` is re-sealed for `q235_box.py`, `q235_tokens.py` and `q235_run.sh`. `q235_verdict.py` runs off the box,
outside the closure. R2's `r2.sha256` cites Q235 at its own commit and is untouched.

## Amendment 9 (2026-10-17, after box `q235-pro6000-3` and Amendment 8, before any further 235B box): the staging rate per copy, and the host-rate probe by buffer (REPORTED only)

**Why.** On `q235-pro6000-3` (`box/box.json` `8b19c4c2`, `hostrate.json` `c82685f9`, `instance.json` `295f1dc7`), staging ran at
26.6 GB/s while the box's own host-rate probe read 20.39. `am7/READ-C.md` settles that the probe under-measures that host's H2D:
the staged bytes are exact, the timed pass bounds the rate from below at 23.0 GB/s without the stage timer, and the provider's
listing gave 26.6. The records could not say why:
- the box keeps only each step's staged bytes and copy count, though the offload records every copy's CUDA events during the step
  (`E4B_OFFLOAD_STATS=1`);
- the probe times one 1 GiB pinned buffer and records nothing about where it sits. On that host its D2H read 28.29 GB/s on both
  boxes while its H2D read 21.76, then 20.39. Staging varied by layer (25.8–27.9 GB/s) and held steady within a layer. H2D there
  depends on which buffer is read, and one buffer is one sample.

**(i) The box records `h2d_staging` at each step**, beside `h2d_bytes_offload` (`q235_box.py`, `staging_record`):
- `ms`: every staging call's event time, in issue order. These are the offload's own events, recorded on the stream that carries
  each copy and read after the step's synchronize, so nothing is added inside the step;
- `gb_s`: the bytes-weighted rate over all calls; `gb_s_min`, `gb_s_median` and `gb_s_max`: the per-call spread; `nbytes`,
  `policies` and `total_ms`;
- `by_layer`: calls attributed to decoder layers by the registered schedule. Per forward-backward pass, layers 1 to L−1 stage for
  the forward (layer 0 is still resident from the previous pass's backward), then L−1 down to 0 for the recompute: 2L − 1 calls,
  187 at 94 layers, 748 per Shape B step. This is DERIVED from the issue order, not observed. It is left out, with `by_layer_why`,
  when the count is not that schedule's.

**(ii) The probe records `h2d_by_buffer`, `single_buffer_pages` and `card`** (`q235_hostrate.py`):
- **Unchanged:** `h2d_gb_s` and `d2h_gb_s` are still one 1 GiB pinned buffer, timed as before. Every floor and gate 2's fit read
  what they read before.
- `single_buffer_pages`: where that buffer's pages sit.
- `h2d_by_buffer`:
  - K = 8 separately pinned 1 GiB buffers (`--multi-buffers 8 --multi-copies 5`), allocated after the pinned-block cache is emptied,
    so none of them is the single buffer's block reused;
  - buffer i is allocated with the process bound to the CPUs of NUMA node i mod N, round robin over the N nodes that have CPUs
    (unbound where the host exposes no nodes);
  - each buffer is timed alone with CUDA events: one warm-up copy, then 5;
  - buffer 0 is timed again after the rest (`first_last_gb_s`), so a link still warming up does not read as placement;
  - per buffer: the bound node, the node its pages report, and GB/s. Overall: min, median, max, the median per bound node, and the
    median per observed node (null unless every buffer's placement was observed);
  - the single-buffer read runs FIRST, so none of its pinned blocks can be one of these, and the by-buffer read hands its 8 GiB
    back before the host-copy and disk reads.
- Page placement comes from `move_pages(2)`, queried without moving at 16 pages per buffer. Where the kernel refuses that (a
  driver-pinned mapping), it comes from `/proc/self/numa_maps`'s counts for the mapping; otherwise the record says why.
- `card`: the card's NUMA node (sysfs `numa_node`), its IOMMU group's default-domain type (`identity` = passthrough; `DMA` or
  `DMA-FQ` = translated; no group = no IOMMU in use, or none exposed), and the kernel command line's IOMMU arguments.
- **Cost:** about 8 GiB of pinned host memory for the probe's few seconds, before the fetch. The hostrate phase's 600 s cap is
  unchanged.

**How the next box's records are read** (registered now; reporting-only):
- **R9.1, the staging rate of record**, is step 0's `h2d_staging.gb_s` (step 1's is the check). The probe's `h2d_gb_s` is a lower
  bound and is not quoted as the link rate.
- **R9.2, does the by-buffer probe find the link?** Expected: `h2d_by_buffer.gb_s_median` within ±10 % of R9.1's rate. Outside
  that band, the probe still misses what staging sees, and the read says so.
- **R9.3, placement.**
  - The per-node medians are `by_observed_node_median_gb_s` when every buffer's placement was observed, else
    `by_bound_node_median_gb_s` (and the read says which).
  - If those medians span more than 10 % between nodes, NUMA placement accounts for buffer-to-buffer H2D on that host.
  - If every node agrees within 5 % while the single buffer's `h2d_gb_s` sits more than 10 % below them, its low read is not
    placement. `card.iommu_type` and `first_last_gb_s` are read next.
  - Between those, or on a host with one node, no conclusion.
- The next box is not `q235-pro6000-3`'s machine and may show no gap at all. That is a result, not a failure.

**What it is not.**
- REPORTED only. `q235_verdict.py` reads none of it, and no rule, threshold, floor, verdict line or STRICT check reads it.
- A failure in either record is recorded (`error` in the box, `why` in the probe) and never fails the box or the phase.
- Every existing field is unchanged.

**Rehearsal** (the A2000, $0, correctness only; it sets no rate): the probe on the A2000 host, checking that the by-buffer read,
the placement fields and the card fields fill, that buffer 0 reads the same first and last, and that the single-buffer figure is
the one it was; and `h2d_staging` from a box step with the experts offloaded, its call count and per-layer attribution matching the
schedule.

**Recorded at the rehearsal** (2026-10-17, RTX A2000, $0; records in `am9/`):
- **The probe** (at `72c0663a`): single buffer 6.10 GB/s; by buffer 6.12 / 6.16 / 6.18 (min / median / max); buffer 0 first and
  last 6.15 and 6.11. Placement was not observable on that host, and the record says why: its kernel has no NUMA, and the
  container's seccomp filter refuses `move_pages`. The IOMMU group type is `identity`. The rehearsal first ran at `3757430f`,
  where those absent fields were left out silently; `72c0663a` records the reason instead.
- **The box** (Shape B, the 4-layer slice; `q235_box.py` byte-identical at both heads): `h2d_staging` holds 28 calls per step =
  4 × (2 × 4 − 1), the bytes exact, at 6.168 and 6.117 GB/s, with `by_layer` matching the schedule (layer 0: 4 calls; layers 1
  to 3: 8 each). Amendment 2's rules all hold.
- On that host staging sits +0.13 % (step 0) and −0.70 % (step 1) from the by-buffer median (6.16 GB/s), and the single buffer shows no gap.
- The rehearsed `q235_box.py` is Amendment 9 on the tree before Amendment 8 merged. At this amendment's merge the file carries both;
  Amendment 9's hunks are unchanged, and the merged file is rehearsed with the next gate-1 leg before any box. R9.2 and R9.3 are the
  next box's reads.

## Amendment 10 (2026-10-18, before any further 235B box): the shared TC1 harness gains default-off `--order-seed` and the HP1 export hooks

**What changes.** `bench/tc1/tc1_arm.py`, which this closure seals, moves from
`fbe790d7d6e868c31598c76c7e77afb8c5632f16e61cab85c457854af30bdadc` to
`9f43b45b267d3b4fbc0855fd26e8b1c15de7e92832344fe42c74ecba5e2a4939` (e4b #1709). It gains two default-off additions for the HP1
held-out parity lane:
1. `--order-seed S` permutes the TC1 train rows per seed.
2. `--hp1-export DIR` / `--hp1-export-base 1` write a canonical adapter, a training-base digest and the base as stored, after the
   step-N eval.

Registered Q235 runs pass neither.
- With them unset, the train rows are untouched, the receipt carries neither `order` nor `hp1` (its key set and content are
  unchanged; the harness selftest asserts both), and `hp1_lib` is never imported.
- A box that stages `tc1_arm.py` alone needs no new file.
- With the HP1 flags set and no `hp1_lib`, or with `--offload 1`, the arm is a `refused` stub before anything loads.

**The closure.** `q235.sha256` is re-sealed with `q235_stage.py --write`. Against main after Amendment 9 (053c5780), only the
`tc1_arm.py` row moves. A box already staged from an earlier main keeps its old bytes and is unaffected. R2 re-cites once, in its
own PR, after this merges.

**What it is not.** No rule, threshold, band or registered record changes. The scope and the re-pin are ruled by the maintainer;
the closure re-seal is reviewed.

## Amendment 11 (2026-10-18, before any further 235B box): gate 2's timing leg moves to a rented RTX PRO 6000, and the floors it implies

**Why.** Gate 2's time-left caps and the box's host floors are scaled from RTX A2000 legs (Amendment 3's table; `CAP_BOX` from the
Amendment 7 refit). An A2000 timing should not set a budget or a floor for another card class, and the quiet A2000 window that the
floors rule (`max(sealed, measured, quiet sample)`) waits on has not come: the host's load average stayed far above the quiet
gate all day. Gate 2's TIMING leg therefore moves to the registered class, a rented RTX PRO 6000. The A2000 keeps gate 1's
correctness rehearsal (the lifecycle and the plants) and sets no time.

**The leg.**
- **Mode `pro6000`** (`q235_run.sh`): the a2000 rehearsal's slice leg exactly (4 decoder layers, 18 files, 27.05 GB fetched and
  hash-checked like the full checkpoint; box, then repeat with the plants). The card must match `^NVIDIA RTX PRO 6000` with at
  least 94,000 MiB, and the host must have 239 GB of RAM (Amendment 5's science floor, so the leg host is a box-class host) and
  120 GB of free disk. It records only: `science_eligible` is false, and no rate floor refuses it.
- **Selected by the run id only.** A launcher run id `q235-leg-pro6000-N` sets the mode inside the sealed runner (`TC1_RUN_ID`,
  which the launcher forwards). `Q235_REHEARSAL` beside a leg id refuses (78), and `pro6000` without a leg id refuses. A science
  id stays science.
- **Run ids:** `q235-leg-pro6000-1`, with `-2` and `-3` as spares. A refusal before the create consumes an id and is not an
  attempt. At most two legs start a box, and the second runs only as the one re-run the quiet rule below allows.
- **Host terms:** a RTX PRO 6000 offer with at least 239 GB of host RAM and reliability at least 0.98, preferably a machine the
  science box would pin. Rate at most $1.70 an hour including its disk, download at most $0.011 per GB.
- **Guard 1.0 h:** the A2000 leg's 716 s × 1.3, plus the slice pull at the host's listed rate × 1.3, plus 15 minutes of boot and
  pre-flight. **Worst $2.04:** $1.70 × 1.0 h, plus 31.1 GB (the slice and 4.04 GB of wheels) × $0.011.

**Quiet on a rented box** (before the leg's first venv):
- no compute app on the card, and `utilization.gpu` at most 5 % over 10 s, or a refusal (18, `not quiet`);
- the load average and `nproc` are recorded in `receipts/quiet.json` and gate nothing, because a container can see its
  co-tenants' load;
- `q235_hostrate.py` runs twice, before the fetch (`hostrate`) and after the repeat (`hostrate2`), both with an O_DIRECT disk
  read. The two samples must agree within **15 %** on pinned H2D and on host copy. Otherwise the leg is unusable for floors: one
  re-run on another host, and after that the decision goes to the maintainer.

**The floors** (`q235_fit.leg_floors`). The box's budgets will come from a fast host, so the A2000's floors no longer protect
the fit. For pinned H2D and host copy,

    floor = max(sealed, the A2000's measured, min(the leg's two samples) / 1.2);

for the disk, the same with the leg's cold box load's effective read rate (bytes returned over read seconds) as the sample. Each
floor is rounded up to 0.001 GB/s, and floors only move up. The A2000's measured rates are registered here: gate 1's leg at
`d2d6b292`, H2D 5.83 GB/s, host copy 4.34 GB/s, and its box load's reads 22,391,367,680 bytes in 19.55 s (1.145 GB/s).
- **Why 1.2:** a box host at a floor runs a host-bound part at most 1.2 × slower than the leg did. The budget's `MARGIN` of 1.5
  then leaves 1.5 / 1.2 = 1.25 for everything else.
- **Worked example:** samples 24.0 and 23.0 GB/s H2D give 23.0 / 1.2 = **19.167**. Host copy 36.0 and 35.0 give **29.167**. A cold
  load reading 2.488 GB/s gives 2.0733, rounded up to **2.074**. Each is above the sealed floor and the A2000's measured rate, so
  each binds. Samples of 24.0 and 20.0 (16.7 % apart) make the leg unusable. A tested case pins this example.

**The caps.** `q235_fit.py --rehearsal <the leg>` scales the leg's box and repeat by 94 / 4 and `MARGIN` 1.5, as before; its
fetch record is the box's own slice fetch. A leg also binds the six-hour rule: the expected run (the scaled phases without
`MARGIN`, plus the 300 s reserve and the controller's 1,800 s) must be at most 0.8 × 6 h. If it is not, the science box waits.

**Amendment 12 is mechanical.** After the leg, the floors and caps this amendment's formulas give are sealed into
`q235_run.sh` (`MIN_*`, `CAP_*`) in their own change, which is reviewed as arithmetic only: re-derive both from the leg's
receipt. The science box's pin screen re-derives its download bound from that fit instead of holding a fixed one.

**The closure.** `q235.sha256` is re-sealed with `q235_stage.py --write`. Only the `q235_run.sh` row moves, from
`d88469ad66b7113d02489636a9302e92dcc061734670a1d1efea3616ff7dcc84` to `900e7357558d254089bc9e0f1192f5a04c98d6d1ab4ccbe3036bd2deffd94434`. `q235_fit.py` runs off the box and is not in the closure.

**Tests.**
- `tests/test_q235_driver.py`: a leg id runs mode `pro6000` and refuses a card outside the class; the mode comes from the run id
  only; a busy card (a compute app, or utilization above 5 %) refuses as `not quiet`; a quiet card passes and records its gate; the
  second host-rate sample follows the repeat.
- `tests/test_q235_fit.py`: the worked example; disagreeing samples make the leg unusable; the expected run against the guard; a
  synthetic leg read end to end, with `--box-from` refused for a leg.

**Unchanged.** The science box's rules, bands, plants, records and verdict; gate 1's A2000 rehearsal; the sealed floors and caps
until Amendment 12.

## Amendment 12 (2026-10-18, after the leg `q235-leg-pro6000-2`, before any further 235B box): the leg's floors and caps, sealed

**Mechanical.** This amendment applies Amendment 11's registered formulas to the leg's receipt and seals the results into
`q235_run.sh`. The receipt's inputs are committed under `bench/q235/am12/` with their sha256. A test re-derives every sealed
value from them.

**The leg.** `q235-leg-pro6000-1` was refused before the create, at $0: the declared hourly rate was rounded below the
offer's bill. `q235-leg-pro6000-2` ran on a rented RTX PRO 6000, PCIe gen 4 x16, at a cost of $0.52. Every phase exited 0. The
quiet gate saw no compute app and 0 % utilization.

**The floors** (`max(sealed, the A2000's measured, min(two samples) / 1.2)`, rounded up to 0.001 GB/s):
- **Pinned H2D:** samples 28.36 and 28.33 GB/s (0.1 % apart). 28.33 / 1.2 = 23.6083, so **23.609**.
- **Host copy:** samples 76.64 and 70.20 GB/s (8.4 % apart, inside 15 %). 70.20 / 1.2 = **58.5**.
- **Disk: 1.145.** Amendment 11 registers the leg's cold box load's read rate as the sample. This leg's load was warm: the
  fetch had just read and hashed the slice, and the load read it at 17.36 GB/s from the page cache. The registered term was
  not observed, so it contributes nothing, and the floor is max(sealed 0.743, the A2000's measured 1.145) = **1.145**. The leg's
  O_DIRECT samples (1.73 and 3.33) are not substituted: that would be a new rule chosen after seeing the data.
- **Sanity check of the fit at that disk floor:** the full checkpoint's reads are about 22.39 GB × 94 / 4 ≈ 526 GB. Read cold
  at 1.145 GB/s, they take about 460 s, against the leg-scaled warm read of about 30 s. So the box's scaled 1,997.5 s becomes
  about 2,427 s, inside its 2,997 s budget.
- **For a later leg:** this leg design cannot observe a cold read, because the fetch warms the cache. A leg that needs one must
  drop the page cache, or read with O_DIRECT, before its box load.

**The caps** (`q235_fit.py --rehearsal <the leg> --warm-install-s 155 --bandwidth-mbps 806 --usd-per-hour 1.627`):
- **Box:** 85 s × 94 / 4 = 1,997.5 s, × 1.5 = **2,997 s** (was 13,847, the A2000 refit of Amendment 7).
- **Repeat:** 20 s × 94 / 4 = 470 s, × 1.5 = **705 s** (was 4,710).
- **Unchanged,** because each still covers the leg fit's budget: install 1,800 (budget 293 at 806 Mbps), fetch 12,000 (budget
  7,111 at 806 Mbps), the fixed phases, and `MIN_FETCH_GBS` 0.062.
- **The expected run** (the scaled phases without `MARGIN`, plus the reserve and the controller) is **2.65 h**, within
  0.8 × 6 h.

**The pin screen.** The science box's download bound is the fit's expected run at the pinned host's listed bandwidth, at most
4.8 h, as Amendment 11 registered. The fixed 3,000 Mbps bound is retired.

**The science box.**
- **Run ids:** `q235-pro6000-4`, with `-5` and `-6` as spares. A refusal before the create consumes an id and is not an attempt.
- **Host terms:** a RTX PRO 6000 with at least 239 GB of host RAM, reliability at least 0.98, and 650 GB of disk. Rate at most
  $1.70 an hour including the disk, download at most $0.011 per GB. Prefer the leg's machine, whose rates set these floors.
- **Guard:** 6.0 h. **Worst** = 6.0 h × the quoted rate + 474.252 GB × the quoted per-GB download, at most $15 (at the leg
  machine's quote: $1.627 and $0.0026, so $11.00).

**The fit's one code change.** For an Amendment 11 leg, `q235_fit.py` passes no disk sample to `leg_floors`, and reports the
warm read beside it as not used.

**The closure.** `q235.sha256` is re-sealed with `q235_stage.py --write`. Only the `q235_run.sh` row moves, from
`900e7357558d254089bc9e0f1192f5a04c98d6d1ab4ccbe3036bd2deffd94434` to `1bc7a664ff791a3bfbc38f17ecb4829d4236e20424ace89599c7872157e51875`.

**Tests** (`tests/test_q235_fit.py`):
- The committed leg refits to exactly the sealed floors and caps, with `sealed_match` true and the expected run inside the guard.
  Each sealed constant moved by one unit fails it.
- Amendment 7's A2000 refit (13,847 s) and Amendment 3's plants refit (4,702 s) stay as records against the new caps.
- The binding of the expected run is exercised against the caps Amendment 11 registered it beside: under these tighter caps,
  every cap fitting already implies an expected run under 4.8 h.

**Unchanged.** The science box's rules, bands, plants, records and verdict, and gate 1's A2000 rehearsal.

## Amendment 13 (2026-10-18, after `q235-pro6000-5`, before any further 235B box): the host-copy floor, from the fit

**Why, stated plainly.** This amendment is motivated by `q235-pro6000-5`'s refusal. That box (24 CPUs, PCIe gen 5) read pinned
H2D at 49.02 GB/s and the disk at 11.91, but host copy at 17.69 GB/s against Amendment 12's floor of 58.5. Host floors are
feasibility gates: they protect the guard and the fit, they never touch the science reading, and the worst case stays bounded
by the guard × the rate. Amendment 11's rule (min(samples) / 1.2, applied as if the whole box were bound by each rate) turns
out to demand a host like the leg's own (56 CPUs, host copy 76.64 / 70.20). Yet its own fit shows a 17.69 GB/s host fitting
with about 900 s to spare: the host-copy-bound part of the box is the load's assemble, 1.45 s on the slice, 34.075 s scaled, and
at 17.69 GB/s it becomes about 135 s, so the box takes about 2,099 s against its 2,997 s budget.

**The rule.** The host-copy floor is derived per host-bound part. The load's assemble may grow by at most 20 % of the box's
scaled time on a slower host:

    MIN_HOST_COPY = max(5.14, the A2000's measured 4.34, min(two samples) × A / (A + 0.2 × B)),

with A the leg's scaled assemble and B the leg's scaled box, rounded up to 0.001 GB/s. The base 5.14 is the floor sealed before
Amendment 12. This keeps the 1.2 slack's intent, a slower host costs a bounded share, while dropping the strictness of charging
that share to the whole box.

**Worked example (the leg).** A = 1.45 × 94 / 4 = 34.075 s and B = 85 × 94 / 4 = 1,997.5 s, so 0.2 × B = 399.5 s and the
assemble may reach 433.575 s. 70.20 × 34.075 / 433.575 = 5.5171, rounded up to **5.518** GB/s (`MIN_HOST_COPY_GBS`, was 58.5).
A host at 17.69 passes; one at 5.0 is refused.

**Unchanged.** Pinned H2D stays at 23.609 and the disk at 1.145. Their parts already sit far inside the fit: H2D-bound
staging and offload rides the leg's measured box, and the disk's cold-read check gives about 2,427 s against 2,997. Nothing
else changes either: the caps, the expected run (2.65 h), the leg and its receipt.

**Run ids.** `q235-pro6000-7`, `-8` and `-9` are registered as spares for host-floor refusals. A registered host-floor
refusal after the create is not an attempt: the gate refused a host for a host property, before any fetch.

**The fit.** `q235_fit.leg_floors` takes the leg's scaled assemble and box and derives the host-copy floor by this rule (main
passes them for a pro6000 leg). Without them it keeps Amendment 11's rule, which the A11 worked-example test still pins.

**The closure.** `q235.sha256` is re-sealed with `q235_stage.py --write`. Only the `q235_run.sh` row moves, from
`1bc7a664ff791a3bfbc38f17ecb4829d4236e20424ace89599c7872157e51875` to `d09981f5f73198875cc5b93733c1c0864a6feec567f537c991227fbd2b96b1eb`.

**Tests** (`tests/test_q235_fit.py`):
- The worked example. A host at 17.69 passes and one at 5.0 is refused. The base 5.14 binds when the fit-derived value is lower.
- The committed leg refits to exactly the sealed floors (23.609, 5.518, 1.145).
- `HC_BOX_SHARE` 0.1 fails three tests, and `MIN_HOST_COPY_GBS` 5.517 fails the refit test.

## Amendment 14 (2026-10-18, before any further 235B box): the science box's host rate

For the science box's ids `q235-pro6000-6` to `-9`, the host rate is at most **$1.85 an hour including 650 GB of disk** (was
$1.70). This is a feasibility term. No PRO 6000 offer with at least 239 GB of host RAM billed under $1.70 for about 1.5 hours,
and the cheapest reliable one billed $1.83. The worst case must still be at most $15 at the pinned quote (6.0 h × the quoted
rate + 474.252 GB × the quoted per-GB). Amendment 13's floors and everything else are unchanged.
