# PREREG — training a 284B model from its own released MXFP4 bytes, on one 24 GB card

**Pre-data.** Stamped before the arena is baked or any step is timed.

## What is under test

`enable_nvme_train_residency` read only NF4 quantize-at-bake arenas, so a
natively-MXFP4 checkpoint could **serve** from its released bytes but had to be
re-quantized to train from disk. The overlay under test
(`arena_offload_view`, branch `feat/mxfp4-arena-train`, one file, 70 lines)
resolves MXFP4 arenas through `fuse_gate_up_segments` into the same
four-segment staging the NF4 path uses.

**Claim:** DeepSeek-V4-Flash (43 layers x 256 experts, 284B) runs a QLoRA
training step on a single 24 GB card, with its frozen experts served from an
on-disk arena of the checkpoint's **own MXFP4 bytes** — no re-quantization.

**This code has never run against a real MXFP4 arena.** The CPU resolver check
covers NF4 passthrough and the reject path only.

## Arms, both on the same pod and card

| arm | what it is | expected |
|---|---|---|
| **STOCK** | `transformers` + `bitsandbytes` 4-bit + `peft`, same model, same card | **OOM** — 284B in 4-bit is ~142 GB against 24 GB of VRAM |
| **ARENA** | published e4b/gnf4 wheels + this overlay, `arena_train=True` + `enable_nvme_train_residency` | trains |

The STOCK arm is **run, not asserted**: its traceback is captured verbatim into
the receipt. An enablement claim whose control was never executed is an opinion.

## Frozen predictions

- **P1 — STOCK fails.** If STOCK *succeeds*, the enablement claim is refuted and
  that is the headline, not a footnote.
- **P2 — VRAM.** Peak < 24 GB, and predicted **4–8 GB** for the staged stacks
  from the module's own arithmetic (it keeps the full `[E, ...]` shape, quoted
  at ~15.7 GB for K3's 896 experts/layer, so 256/layer scales to ~4.5 GB) plus
  activations. Outside that band, the arithmetic in the header is wrong and gets
  corrected in public.
- **P3 — bytes unchanged.** Every frozen expert tensor hashes identical before
  and after training, and the arena rows match the checkpoint's released bytes.
  A single mismatch STOPS the run.
- **P4 — it learns.** Loss over the registered steps is finite and moves. No bar
  on how much: this is a feasibility run, not a quality result.
- **NO PREDICTION on s/step.** Reading routed rows off NVMe every step is a new
  cost curve; whatever it is, it is reported.

## Gates

- **G1** `enable_nvme_train_residency` returns > 0 modules patched. Zero means
  the overlay did not take and every later number is meaningless.
- **G2** The overlay is verified *in the imported module* before the run, and
  `overlay_provenance.json` records stock and patched sha256.
- **G3** Gradient checkpointing on — the tier requires it (evict fires on
  forward return; without recompute, backward reads 0-element placeholders).

## Stop rules

Whatever it shows is written up in either direction, including "the enabler
refused", "it OOMed", or "the bytes moved". No retry-until-green: a failure is
the result, and the code is only changed with a new stamped prereg. Hard bill
cap $35; teardown is evidence-first.

## Not blind

I read the tier's source and its serving figure (V4-Flash generating at 8.74 GiB
peak from a 147 GB arena) before writing this. P2's band is derived from that
code's own comment, not guessed.

---

## Amendment 1 (2026-08-13, pre-data for the second attempt)

**Attempt 1 measured nothing.** It is recorded here rather than discarded,
because three of its four failures were mine and two of them change the design.

- The arena bake raised `TypeError: bake_expert_tensors() missing 1 required
  keyword-only argument: 'name_template'` — **and the run continued**, because
  the step was written `... | tail -12 || fail BAKE_FAILED` and the exit status
  of a pipeline is its last command. Both arms then ran against an arena that
  did not exist. Fixed in all four banked runners (`set -o pipefail` plus an
  explicit `${PIPESTATUS[0]}` check).
- **The STOCK control failed for the wrong reason** — `FineGrainedFP8Config` vs
  `BitsAndBytesConfig`, i.e. transformers refusing bnb-4bit over an
  already-quantized checkpoint, because V4's dense half is block-scaled FP8.
  P1's direction held; its stated mechanism did not. A config rejection is not
  evidence of a memory ceiling, so the control is respecified: load with **no**
  quantization config and let it attempt to materialize.
- **The ARENA arm never started**: `quant_type="mxfp4"` is not an accepted
  value (`nf4/fp4/int8/fp8/bf16/fp16`). The overlay fixed segment RESOLUTION;
  the module still has to be MXFP4-SHAPED for the geometry check to match.

**Target excursion, and the reason it reversed.** After attempt 1 I recommended
switching to gpt-oss-120b as a cheaper proof-of-method, and that was **wrong**.
`lora.py::_epilogue` defers to the base's `_apply_gate`; `arch/deepseek_v4.py`
provides one and `arch/gptoss.py` does not. Wrapping gpt-oss in `ExpertsLoRA`
today would apply plain SwiGLU over a clamped-GLU base — the failure that
docstring names explicitly: *the model trains, the loss falls, and it is
optimising a function the frozen base does not compute*. V4 is therefore the
SAFER first target, not the more expensive one. Target returns to V4-Flash.

**Design added under test (option B).** Under `arena_train=True` with an MXFP4
arena, the loader builds the meta experts **MXFP4-shaped**, so their declared
dtype and per-expert width match the arena's segments. The base is on `meta` and
holds nothing, so this concerns declared shape, not data movement.

**Predictions unchanged** (P1–P4, gates G1–G3), except that P1 is now graded on
a control that fails on MEMORY rather than on a quantizer-class rejection. The
pre-amendment stamp is preserved as `PREREG.md.pre-amendment1.ots`.

---

## Amendment 2 (2026-08-14, pre-data — the compute half exists now)

Amendment 1 put option B under test as **staging only**, and said so in as many
words: *"the COMPUTE half (an MXFP4 fused kernel behind this module's forward) is
not wired here."* It is wired now, and that changes enough about what this
experiment runs that it gets a stamp before any of it is measured.

**What landed.** `_dequantize_expert` is overridden per instance, so every
reference lane — `ExpertsNbit.forward`, `_DeepseekV4ForwardMixin.forward`, and
`ExpertsLoRA._base_project` — decodes MXFP4 through `formats.mxfp4`. The module's
`forward` additionally routes to `mxfp4_grouped.gemm_mxfp4_grouped` under CUDA +
bf16 + **no autograd graph**.

**Parity-gated before this stamp**, on an L40S, against the pure-torch oracle
(`dequantize_mxfp4` then matmul) rather than another accelerated lane:
projections **0.000e+00**, whole forward **4.4–5.9e-3**, and an
unclamped-epilogue control at **~1.0** so the fixture demonstrably tells the
activations apart. Receipts in `receipts/`, write-up in
`RESULTS-forward-parity.md`.

### The consequence that matters for THIS run

**The training step will not use the fused kernel, and no number from this run
should be read as if it did.** `gemm_mxfp4_grouped` is raw Triton with no
`autograd.Function` behind it, so the router deliberately refuses it whenever
autograd needs a graph — otherwise there is no `dL/dx` and the model below the
layer silently stops learning. Every ARENA-arm step therefore runs the
**dequantize-per-expert reference lane**, recomputing each expert in backward.
That is the cost curve the "NO PREDICTION on s/step" line already declines to
predict; naming the mechanism here so the absence of a prediction is not mistaken
for an absence of a known bottleneck.

**`enable_fast_train` must NOT be called for the ARENA arm, and returns 0 if it
is.** It now refuses an MXFP4-arena base outright: such a base passes every
eligibility check it had (`quant_type="nf4"` by class, `_apply_gate` present) and
would then die inside the forward on `gate_up_absmax.view(E, n1, k1 // 64)`,
because e8m0 scales are one byte per 32 elements, not fp32 per 64. The usage line
in `nvme_train`'s docstring — which ends `enable_fast_train(model)` — does not
apply to this arm. A runner that asserts a nonzero return, as the standing
house rule for that opt-in says to, **will fail on a correct system.**

**P2's arithmetic is unchanged.** MXFP4 storage is ~5.6% smaller per element than
NF4 (0.53125 vs 0.5625 bytes: same K/2 blocks, one scale byte per 32 against fp32
absmax per 64), and the staged stacks still keep the full `[E, ...]` shape. The
4–8 GB band stands as stated; it is not re-derived to fit.

### Added gate

- **G4 — the arm must be on the lane it claims.** Assert, in the imported module
  and before the run: every patched module has `_e4b_mxfp4_arena` True; the
  frozen base's `_dequantize_expert` is the MXFP4 override, not the NF4 one; and
  `enable_fast_train` returns **0**. G1 counts modules patched, which no longer
  implies which arithmetic they will run — a module can be patched and still
  refuse the fused lane, which is the correct behaviour under grad and would
  otherwise be indistinguishable from a dead patch.

**Predictions P1–P4 and gates G1–G3 are otherwise unchanged.** Stop rules
unchanged: whatever it shows is written up in either direction, no
retry-until-green, hard cap $35, teardown evidence-first. The pre-amendment-2
stamp is preserved as `PREREG.md.pre-amendment2.ots`.

---

## Amendment 3 (2026-08-14, pre-data — the second attempt, after an upstream fix)

The first attempt under amendment 2 **measured P1 and nothing else**. It is
recorded in `RESULTS-284b-run.md`; the short version is that everything up to the
geometry check ran (149 GB downloaded in 128 s, the MXFP4 relocation arena baked
in 76.8 s, the model loaded in 20.7 s at 10.63 GiB peak on a 23.6 GiB card) and
then `enable_nvme_train_residency` raised `KeyError: 'F8_E8M0'` inside
`nvme_residency.segment_geometry`.

**P1 is already CONFIRMED and is not re-litigated by this run.** The STOCK control
died with a genuine `CUDA out of memory` on the 23.56 GiB card — the memory
ceiling amendment 1 respecified it to test, not attempt 1's quantizer-class
rejection. It is run again here only as a same-host control; a second failure adds
confidence, and a success would refute P1 and become the headline exactly as
amendment 1 says.

**P2, P3 and P4 have still never been graded.** No training step has run.

### What changed, and the provenance cost

`grouped-nf4-gemm` could bake a DeepSeek-V4 MXFP4 arena and serve from it, but
`_ST_TO_TORCH` — the table `segment_geometry` and `segment_tensor` both resolve —
had no entry for `F8_E8M0`, the tag V4 uses for its expert scales. Fixed in
[grouped-nf4-gemm#75](https://github.com/pjordanandrsn/grouped-nf4-gemm/pull/75),
merged as `0f68952`, with three tests that fail on the unfixed code with that
exact `KeyError`.

**The fix is not in any published wheel.** This run therefore applies it as part
of the same overlay, so the receipt reads **published `experts4bit-qlora` 0.18.0 +
published `grouped-nf4-gemm` 0.11.0 + a five-file overlay spanning both
packages** — four e4b files and gnf4's `nvme_residency.py` backported from merged
main. That is a **weaker provenance claim than the parity leg**, which was
published wheels plus one file, and it is stated here rather than left for a
reader to notice. `overlay_provenance.json` records the stock and overlay sha256
of all five.

**G2 is widened** to check the gnf4 half in the imported module
(`nvme_residency._ST_TO_TORCH["F8_E8M0"] == "uint8"`), because an overlay that
takes for e4b and silently misses for gnf4 would fail in exactly the same place as
before.

### Registered before the fact

- The CPU gate now bakes V4's real labels (`I8` blocks, `F8_E8M0` scales) rather
  than `U8` for everything. On stock gnf4 that test reproduces the pod's KeyError
  in 3.6 s; with the overlay it passes. The gate went 12 passed + 1 xfailed →
  **13 passed**, and that transition is the evidence the fix unblocks the path.
- Three harness defects from the last attempt are fixed and are **not** part of
  what is under test: the loader was handed the hub id instead of the local
  checkpoint path (starting a second 160 GB download onto a full disk),
  `capacity_for_bytes` was called with one of its two positionals, and the G2 gate
  grepped a docstring for a marker that lives in code. `hot_rows` is now sized
  from the **cgroup**, not from `free`.

**Everything else stands**: P1–P4 as written, G1/G3/G4, and the stop rules. A
refusal, an OOM or a wedge is the result. Hard cap $35. The pre-amendment-3 stamp
is preserved as `PREREG.md.pre-amendment3.ots`.

---

## Amendment 4 (2026-08-14, pre-data — a registered batch size, because there wasn't one)

The attempt under amendment 3 reached a real training step. G1 patched **43**
modules, **G4 held** (43/43 MXFP4-flagged, 43/43 on the MXFP4 dequantize
override, `enable_fast_train` returned 0), the model loaded in 23.9 s at 10.63
GiB, and the step then **exceeded the 24 GiB cap inside `_epilogue`** on a
44.39 GiB card that still had 19.75 GiB free.

**That is not yet a verdict on P2, and the reason is a hole in this prereg: no
batch size was ever registered.** The 8×64 shape was picked in the harness, not
here. "It exceeded 24 GiB at a shape I chose after the fact" is not a measurement
of the claim; it is a measurement of my harness. This amendment closes that hole
before any more data is taken.

### The registered ladder

Run **ascending**, every rung, **each as its own process** — so an OOM at one rung
costs that rung's receipt and not the ones already measured. The deliverable is
the **curve**, not the first cell that fits.

| rung | tokens × seqlen | total tokens | role |
|---|---|---|---|
| 1 | 1 × 128 | 128 | **PRIMARY** |
| 2 | 1 × 256 | 256 | |
| 3 | 2 × 256 | 512 | |
| 4 | 4 × 256 | 1024 | |
| 5 | 8 × 64 | 512 | reproduces the shape that exceeded the cap under amendment 3 |

Rung 5 is kept deliberately: it lets the previous observation be reproduced
**within this run** rather than compared across pods.

**Why 1×128 is the primary, stated plainly:** it is the smallest configuration
that is still a genuine QLoRA training step, and it is chosen precisely *because*
a larger shape already exceeded the cap. Picking it after seeing that failure is
exactly the move that needs declaring rather than hiding, so it is declared. The
claim it can support is correspondingly narrow.

### P2 restated for this ladder

- **P2a (primary).** At rung 1, peak VRAM **< 24 GiB**. This is the claim
  "trains on a single 24 GB card" reduced to the shape actually registered.
- **P2b (the curve).** Peak is reported for every rung that completes, and the
  **largest rung that fits in 24 GiB is the headline number** — including the
  case where that is *no rung at all*, which refutes the enablement claim and is
  written up as such.
- P2's original 4–8 GB band was about the **staged expert stacks**, not total
  peak, and the run now reports both so the two are not conflated.

**No prediction is offered on where the ladder breaks.** Naming a rung now would
be a guess dressed as a hypothesis.

### Also fixed, and not under test

`hot_rows` sizing was wrong twice in the last attempt and both are corrected here:

- **cgroup v2 counts page cache in `memory.current`**, so straight after the
  138 GiB bake the naive `memory.max − memory.current` read **18.3 MB**. The
  reclaimable `file` field from `memory.stat` is now subtracted before calling the
  remainder free, and the raw figures go in the receipt.
- The floor omitted the `min(…, num_experts)` clamp the tier's own docstring
  specifies, so it asked for `512 × 6 = 3072` rows where only **256** experts
  exist per layer — **38.2 GiB** of pinned DRAM instead of 3.2 GiB. Clamped now,
  and a floor that exceeds measured capacity **refuses** instead of silently
  taking the floor.

Neither touched VRAM, so neither caused the observed OOM; they are recorded
because the previous receipt's `hot_rows` must not be read as a measured capacity.

The pre-amendment-4 stamp is preserved as `PREREG.md.pre-amendment4.ots`.

---

## Amendment 5 (2026-08-14, pre-data — the dense side, and what this does NOT predict)

The amendment-4 ladder refuted P2 at every rung, and located the failure precisely:
not the expert tier but `formats/fp8_blocks.py::dequantize_fp8_blocks`, batch-
independently, with 51 MiB free on a 23.5 GiB card. See
`RESULTS-284b-ladder.md`.

**What changed.** That function's own class docstring claimed a transient of "one
weight at a time (~67 MB for V4's largest)". It counted the bf16 result only; the
fp32 route also held an expanded fp32 scale, `weight.float()` and the fp32 product
— **12–14 bytes per parameter, ~403 MB for `wq_b`**. The aligned path now decodes
in `dtype` with a broadcast scale, **2 bytes per parameter**.

**It is bit-exact, and that is verified rather than argued.** e4m3 → bf16 is
lossless (3 mantissa bits into 7) and an e8m0 scale is a power of two, so the
multiply cannot round. Checked on CPU across typical, wide and deliberately
over/underflowing exponent ranges, plus ragged shapes, before this stamp.

### P5, and the prediction deliberately NOT made

- **P5 — the transient shrinks.** The dense dequantize holds ~2 bytes per
  parameter instead of ~12. This is arithmetic plus a bit-exactness test, and it
  is the only thing claimed here.
- **NO PREDICTION that the ladder now fits.** At the failure the card held
  22.88 GiB with 51 MiB free; this change frees a few hundred MB at the peak
  moment. Whether that clears the rung, or simply moves the OOM to the next
  allocation, is **reported, not forecast**. Predicting a pass here would be
  guessing dressed as a hypothesis, and the previous amendment already refused
  that once.

### Method held constant

The ladder is re-run **unchanged** — same five rungs, same order, same one
process per rung — so the comparison against `RESULTS-284b-ladder.md` is
like-for-like and any difference is attributable to this one change.

**Provenance cost grows again, and is stated:** the overlay is now **six files** —
four e4b engine/adapter files, `experts4bit_qlora/formats/fp8_blocks.py`, and
gnf4's `nvme_residency.py` from merged main. G2 checks the fp8 half in the
imported module alongside the other two.

P1 stands confirmed twice and is not re-litigated. P3 and P4 remain ungraded. The
pre-amendment-5 stamp is preserved as `PREREG.md.pre-amendment5.ots`.
