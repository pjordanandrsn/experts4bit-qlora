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
