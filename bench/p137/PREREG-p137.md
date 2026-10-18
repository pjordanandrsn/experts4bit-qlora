# P137 — does layer-major accumulation cut the step time of expert-offloaded QLoRA training on an RTX 4090 (sm_89, 24 GB), in the configuration that makes Qwen3-30B-A3B trainable there?

Registered before any reading. Harness: `p137_run.sh` (box), `p137_drive.sh` (controller), `p137_box.py`,
`p137_reduce.py`, `p137_tiny.py` (the dry fixture), all at `staged.sha256`; `tests/test_p137_staged_pin.py` checks them
in CI.

## Why now

Layer-major accumulation is on main: the per-handle forward hold (#1708), `layer_major_step` (#1710) and its guards
(#1715: the drift guard, the reproducibility pre-check, the dropout refusal). The stock accumulation loop runs each
micro-batch's whole forward and backward in turn. Under expert offload that stages every layer twice per micro-batch:
M × (2n − 1) whole-layer stagings per step in steady state. Layer-major stages each layer once per direction: 2n − 1.
Every test so far is a correctness read (CPU, and the RTX A2000, which is correctness-only). This lane reads the speed,
on the card where offload is not optional.

**The subject is TC3's `fused_attn4_m_offload` configuration**, the arm that trains Qwen3-30B-A3B on a 24 GB card
(`tc3-4090-8`; `bench/h2h-2026-10-02/tc3/README.md`): the experts in pinned host RAM, `enable_fast_train(dgrad=True)`
(the grouped kernels, and its reentrant checkpoint). The resident arm OOMs on that card. The shipped CLI trainer
(`python -m experts4bit_qlora.train`) does not take the fused path. Under `OFFLOAD_EXPERTS=1` it runs the per-expert
reference chain (77.387 s/step on the same box), where staging is a small share of the step, so a read there would
measure the kernel path and not the staging schedule. That gap is out of this lane's scope.

**A pre-registration gate already passed** ($0, RTX A2000, torch 2.11.0+cu128, e4b `b4366b9a`). `layer_major_step`
composes with `enable_fast_train(dgrad=True)` under offload, on a tiny NF4 Qwen3-MoE:
- 3/3 wrappers were patched, with the reentrant checkpoint on 3 layers;
- the reproducibility pre-check and the drift guard pass (step-0 per-micro-batch `torch.equal` against the stock forward);
- stagings were 16 = M(2n − 1) + 1 cold for the stock loop, and 6 / 5 = 2n / 2n − 1 for layer-major;
- layer-major against the stock loop's adapter gradients gave max_rel 2.89e-3 and cosine 0.9999966.

## The pins

| | pin |
|---|---|
| e4b | `b4366b9af2234730b5df783c5f3b0c7aef3ab41c` (main with #1708, #1710, #1715), installed by the box; `E4B_PIN` in the runner and driver |
| grouped-nf4-gemm | **0.46.0**, `6b7c2e7474d20c8843814dd46dad6cfc3d94b440` |
| torch | 2.8.0+cu128, the image's (TC3's e4b box) |
| transformers / bitsandbytes | 5.18.0 / 0.50.2 (TC3's e4b box) |
| model | Qwen/Qwen3-30B-A3B @ `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` (TC3's) |

**Card:** the RTX 4090 class, compute capability 8.9, checked before any install (rc 15). A licence covers that card only.

## The arms

Four fresh processes, ABBA: **A0a L1a L1b A0b**, differing only in the accumulation loop:
- **A0:** the stock loop. Per micro-batch, `model(**b)`, then `(loss * w).backward()` with w = 1/4 (`train.py`'s loop).
- **L1:** `layer_major_step(model, batches, w)`, with the same weights.

**Recipe** (TC3's, `p137_box.py`):
- Qwen3-30B-A3B, NF4 experts, `offload=True, pin=True`, `enable_fast_train(dgrad=True)`;
- r 16 / α 16, bf16 adapters, AdamW 8-bit at lr 2e-4, seed 3407;
- alpaca (response-only loss) at seq 2048, micro-batch 2 rows × accumulation 4;
- every lever at its default: the chunked loss on `auto`, no pinned slab, no arena;
- 2 warm steps (step 0 is the equality step), then **8 timed steps**.

Every arm trains the same micro-batches from the same seed. A step's wall is CUDA-synced around zero-grad, the four
micro-batches, and the optimizer step. `E4B_OFFLOAD_STATS=1` is on in both arms to count stagings (one event pair per
copy, the same in both arms).

**Per step, recorded:** the wall, the loss, the stagings and the peak allocated memory. **Per arm:** step 0's
per-micro-batch losses (and their float hex), a held-out loss on 16 later rows after the last step, the versions and the
fast-path census.

## The rule (`p137_reduce.py`, self-tested on 29 cases)

First rung that applies:
1. **VOID** on any of these:
   - an arm record missing, or an arm OOM. An L1 OOM is VOID, never NO_GAIN;
   - another e4b commit than the pin, or the arms on different stacks, models or recipes;
   - `enable_fast_train` patched nothing, or no expert layer was offloaded;
   - a dry record in a reading, or the reverse; a reading whose model path lacks the pinned revision `ad44e777`;
   - the patched and the offloaded layer counts not both the registered **n = 48** (the dry fixture's 2): a partly
     offloaded run must not pass against a target derived from itself;
   - step 0's per-micro-batch losses not bit-identical across all four arms;
   - stagings per step after step 0 not exactly **4 × (2n − 1) = 380** (A0) and **2n − 1 = 95** (L1), with n = 48;
   - L1's peak allocated memory over the **timed** steps above A0's by more than **4.25 GiB** (below). Step 0 is left out:
     on L1 it also carries the drift guard's one-time stock forward. The all-steps peak is recorded.
2. **NOISY:** a self-pair (A0b/A0a, L1b/L1a, the medians of the timed walls) outside **[0.97, 1.03]**.
3. **LICENSED** iff **S = min(A0a/L1a, A0b/L1b) ≥ 1.10**; otherwise **NO_GAIN**.

**REPORTING** (never changes the verdict): held-out |L1 − A0| ≤ **0.01** (a missing held-out is recorded as missing);
S inside the registered band.

**The memory term, derived.** Layer-major keeps every micro-batch's saved state alive until its one backward; the stock
loop keeps one at a time. The extra is the other M − 1 = 3 micro-batches':
- their checkpointed layer inputs, 48 × tokens × 2048 × 2 B each;
- their loss's saved fp32 log-probabilities, tokens × 151,936 × 4 B each, while the chunked loss's `auto` gate leaves
  the stock loss on. It engages at 1 GiB of fp32 logits, so at 1,767 tokens per micro-batch.

The worst case over micro-batches of at most 2 × 2,048 tokens sits just under that gate: 3 × (1.0 GiB + 0.32 GiB)
≈ 3.97 GiB. Above the gate the log-probabilities go and 3 × 0.75 GiB = 2.25 GiB remain. **The registered term is
4.25 GiB:** the extra 0.25 GiB covers state the derivation does not model (about 10–20 MB per micro-batch), so that the
bound is not itself the likeliest false VOID at the gate boundary.

**Checked against the card before any data:** the anchor's peak (`tc3-4090-8`, `fused_attn4_m_offload`, 11.881 GB
`torch.cuda.max_memory_allocated`, 11.07 GiB) plus the 4.25 GiB term is 15.32 GiB (16.4 GB), against the RTX 4090's
23.52 GiB. An L1 OOM is not expected.

## Predictions (written before any data)

| # | prediction |
|---|---|
| Q1 | every arm fits; A0's peak near the anchor's 11.881 GB, L1's within the 4.25 GiB term |
| Q2 | step 0 bit-identical across arms (the drift guard already checks it per micro-batch on L1); stagings exactly 380 / 95 |
| Q3 | **S ∈ [1.25, 2.0], point 1.55–1.85** |
| Q4 | self-pairs inside [0.97, 1.03] |
| Q5 | held-out within 0.01 (the gradients are the same sum in another order) |
| Q6 | LICENSED, about 80 % |

**Q3's arithmetic.** One layer's experts are 128 × 2.654 MB = **339.7 MB** (hidden 2048, expert width 768, NF4 packed
plus fp32 absmax, as offloaded stacks keep it). Per step that is 380 × 339.7 MB = **129.1 GB**
for A0 and 95 × 339.7 MB = **32.3 GB** for L1. No H2D rate is on record for this card class. At gen-4 rates between
the 20.75 GB/s 5090 probe (`bench/q235/PREREG-q235.md`) and Q235's 26.6 GB/s staging rate of record
(`bench/q235/am7/READ-C.md`), staging takes 4.85–6.22 s for A0 and 1.21–1.56 s for L1: a saving of 3.6–4.7 s.
Against the anchor's 10.521 s/step (`tc3-4090-8`, a different harness on the same configuration),
S = 10.52 / (10.52 − saving) ≈ 1.53–1.81. The band's low end allows a faster link or a smaller staging share in this
harness; its high end allows a slower link.

## Consequence, registered now

- **LICENSED:** one line in the docs and the capability inventory, naming the configuration exactly: layer-major
  accumulation (`layer_major_step`) **with `enable_fast_train` + expert offload**, on an RTX 4090, Qwen3-30B-A3B, at
  this recipe, citing this receipt. **Never a bare `LAYER_MAJOR=1`:** that knob is the CLI trainer's, which does not
  take the fused path. No default flips.
- **NO_GAIN:** a note in `bench/expert-tiling/README.md`'s levers, and the hybrid lever (routed staging in the
  recompute) is reconsidered there before any registration.
- **VOID or NOISY:** one rerun (`p137-4090-2`). A VOID from the memory term or an OOM is reported as such; the lane does
  not rerun it under another term.

## Runs, cost and the dry run

- **Run ids:** `p137-4090-1` (the reading) and its one rerun `p137-4090-2`. `p137-dry-<n>` is the $0 dry run. There is
  no paid proof: the dry run drives the whole path.
- **Cost:** an RTX 4090 at ≤ the policy rate, **guard 2.5 h**, plus the 61 GB checkpoint at the offer's per-GB rate.
  Worst ≈ 2.5 h × $0.49 + 61 GB × $0.011 ≈ **$1.90**, inside the no-ask rental rule. Each launch posts its pin and
  expected/worst cost on the bus before the create, with the host pinned by inclusion and `--validate` on the exact
  manifest. Nothing runs before the envelope ACK.
- **Premise**, on the card, before anything is fetched (rc 25): `tests/test_layer_major.py`, `tests/test_offload_hold.py`
  and `tests/test_offload.py` at the pin give **47 passed, none skipped** (`-rs` lists any skip; one fails the
  premise). Those 47 passed on the A2000 at the pin's parents (#1715's leg).
- **Dry run** (`P137_DRY=1`, run ids `p137-dry-<n>`, $0): the driver and the whole box on `p137_tiny.py`'s fixture (a
  2-layer random Qwen3-MoE in the on-disk per-expert layout, Qwen3's tokenizer), on the RTX A2000 under a `gpu:a2000` bus
  claim.
  - The box is reached through `P137_BOX_EXEC` and tar instead of ssh, scp and rsync, under `P137_HOME`.
  - The dry box is shared, so nothing is installed: it runs the pin's e4b tree (staged by the driver) on the box's own
    stack, recorded and not pinned. The torch, card and version pins are the paid run's alone.
  - The reducer reads the fixture's records in its dry mode (n = 2: stagings 12 / 3).
  - The fixture's own floors apply: 10 GB of disk and 16 GiB of host RAM (the reading's: 120 GB and 64 GiB).
  - The A2000 is correctness-only: no timing from it enters any prediction.
- **First executed on the paid run:** the install and the tripwire's pins, the 61 GB fetch, the rsync transport, sm_89,
  and the timed walls at Qwen3-30B scale.

## What was seen

**Dry run `p137-dry-4`** ($0; RTX A2000, sm_86, torch 2.11.0+cu128, transformers 5.19.0, grouped-nf4-gemm 0.45.0, the box's
own stack; e4b the pin's tree; the harness at `57d401ee`). The whole path ran end to end, driver rc 0: staging, the nonce
handshake, the heartbeat, the self-tests, the premise, the data, the fixture, four arms, the reducer and the fetch.
- **Premise: 47 passed, none skipped** (`-rs` listed no skip). So the A2000 skipped nothing that sm_89 would run.
  - What sm_89 adds is not in the premise: the grouped kernels' sm_89 build and tuning on the fused path, and the
    timed walls. Both run first on the paid box.
- **Every arm:** `enable_fast_train` patched 2/2 layers, with the reentrant checkpoint on 2 layers.
- **Step 0's per-micro-batch losses were bit-identical across A0a L1a L1b A0b** (float hex, e.g. `0x1.59d7d4p+4`).
- **Stagings** after step 0 were exactly 12 (A0) and 3 (L1) at n = 2. Step 0 was 13 (A0, cold) and 14 (L1: cold plus
  the drift guard's stock forward).
- **Peak allocated memory** over the timed steps: A0 1.602 GiB, L1 2.179 GiB, so +0.58 GiB, inside the term at the
  fixture's size. The all-steps peak was the same.
- **Held-out:** A0 21.781191 vs L1 21.781043 (|Δ| 1.5e-4).
- **The reducer's dry verdict was NOISY** (A0b/A0a 0.956). The fixture's steps take about 0.17 s on the A2000, so its
  self-pairs are timer noise. The A2000 is correctness-only, and no timing from it enters any prediction or bar.

The harness changed twice on the way, each change followed by a dry run:
- The fixture's own disk and RAM floors (`p137-dry-1` refused at the reading's 120 GB floor on a 79 GB dry box: the
  refusal worked).
- The premise line may carry a warnings count (`p137-dry-2` had 47 passed with 15 warnings, and the strict match
  refused it).

The memory term was then set to read the timed steps only. `p137-dry-4` is that final tree.

## Amendment 1 (after `p137-4090-1`, before any measurement)

**What happened.** `p137-4090-1` (RTX 4090, machine 51617) ended HARNESS_ERROR at $0.028, before any arm. Everything up to
the data step passed on sm_89 for the first time:
- the refusals (cc 8.9, 128 SMs);
- the pinned install and its tripwire (torch 2.8.0+cu128, transformers 5.18.0, grouped-nf4-gemm 0.46.0, bitsandbytes
  0.50.2);
- both self-tests;
- the premise, **47 passed, none skipped**.

The data step (rc 19) failed inside the dataset download on the box: `load_dataset('tatsu-lab/alpaca')` raised in the hub
client's content decoder (`TypeError: process() takes no keyword arguments`). Nothing was fetched or measured.

**The change.** The training rows no longer come from a download on the box:
- The slice (alpaca `train[:400]`, as `data.json`, the bytes every dry run produced) is staged by the driver from the
  controller (`P137_DATA`, default `~/p137-data/data.json`). It is kept outside the repository: the dataset's licence is
  not the repository's.
- The driver refuses (78) unless its sha256 is
  **`68e3d31c194ebd8fee4330808a46642f443aecf4e16b78e5eb39fd21b16079c5`**. The runner checks it again on the box (rc 19 on a
  missing or different file).
- Nothing on the box imports `datasets`.

Nothing else changes: the arms, the recipe, the rule, the bars, the pins and the run ids. The reading runs as the declared
rerun `p137-4090-2`, after a $0 dry run at this amendment's merge and a fresh launch approval. Under the two-strikes rule,
one more HARNESS_ERROR or VOID stops the lane for a root-cause and simplification note.
