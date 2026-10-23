# D0 follow-on: a merged, re-quantized expert arena (path (a)): design note

**Status:** a design note, USD 0, no code. It precedes the arena code PR and the registered cross-format served read.
Nothing here moves a default. Serving a merged arena stays opt-in until that served read licenses it.

**Why now:** D0 (its results are #2070, `RESULTS-d0.md`) found every cell RETAINED on one OLMoE-1B-7B fixture, NF4 re-quantized
from the trained base included. Path (a), baking the expert delta into the arena, is therefore viable on that fixture,
and this note says how to build it so that it fails closed.

## What exists today (main)

- **The paged server loads experts only from an NF4 arena.** `build_engine` requires `E4B_PAGED_ARENA`.
  - `load_moe_4bit_streaming(..., arena=...)` builds the expert modules on `meta`.
  - The hybrid tier then reads each row's four segments (gate_up and down blocks, fp32 absmax) from the arena.
  - No path loads experts resident from the checkpoint.
- **The arena is baked outside this repository,** by grouped-nf4-gemm's `bake_nf4`, with the same `quantize_4bit` call
  (nf4, blocksize 64, `compress_statistics=False`) that training's `from_float` makes.
  - An fp32-absmax trained base therefore equals its arena bit for bit (`tests/test_adapter_fingerprint_serve.py`).
  - The index records the HF snapshot path, the quantizer and the geometry. It has no revision field and no adapter field.
- **An adapter with nonzero expert LoRA is refused** (refusal 1, `adapter_dir.servable_refusals`).
  - Refusal 9 (an unverified expert base) is checked before it.
  - Refusal 12 compares the adapter's recorded expert fingerprint (qfp-v1, with `fp32_source` for a double-quantized
    trained base) against the served arena.
- **Attention adapters merge at load** (`merge_attention_lora`): fp32 on device, one rounding, every target checked before
  anything is merged.
- **INT4 experts** (`E4B_SERVE_EXP_INT4`) are repacked from the checkpoint at load, and **MXFP4** experts are gpt-oss's
  native blocks only. Both are refused under an adapter (refusal 5).

## The design

**1. An offline writer, not a load-time merge.** `experts4bit_qlora.merged_arena.write(base_arena, adapter_dir, out, source=...)`
writes a NEW arena with the base's geometry. Each (layer, expert) row's segments are re-quantized from S + ΔW, where
ΔW = scale · B[e] @ A[e] is computed in fp32.
- **Why offline:** the server has no resident expert path. An in-memory re-quantization of the VRAM tier would leave the
  DRAM and NVMe tiers reading the unmerged arena, so one model would serve two functions by tier. An arena is also an
  artifact that can carry provenance and be fingerprinted.
- **The quantizer is the bake's own call:** `F.quantize_4bit` with nf4, blocksize 64 and `compress_statistics=False`, in
  the arena's row layout and alignment.
  - **The parity test that makes this safe:** with ΔW = 0 (a zero-B adapter), the writer reproduces the base arena's rows
    BYTE FOR BYTE. That pins the layout and the quantizer to the bake without changing grouped-nf4-gemm.
  - If that test cannot be made to hold, the fallback is a grouped-nf4-gemm PR first, kernel-first: a bake hook that
    receives (layer, expert, projection) and returns a delta.

**2. The source S is the decision with evidence behind it.**
- **`source="arena"` (the default):** S is the base arena's own dequantized rows. For an fp32-absmax trained base these are
  Q_train(W) exactly, verified by the adapter's expert fingerprint against the base arena BEFORE the merge. This is D0's
  "NF4 from Q" cell: RETAINED, gap_to_ref +0.0011 and −0.0036 nats.
- **`source="checkpoint"`:** S is the raw checkpoint W in fp32. This is D0's "NF4 from W" cell: RETAINED, gap_to_ref +0.0007
  and +0.0027. It needs the checkpoint at the pinned revision.
- **A double-quantized trained base** (`fp32_source` present) is a third case. The arena's fp32 rows are not Q_train, and
  the server never re-compresses: that is deliberate, because the compression's statistics can differ between GPU models.
  The writer then merges into the fp32 arena rows and records `trained_base: "double-quantized"`. The served read prices
  that combination, which D0 did not measure directly.

**3. Provenance: what each side records, so no check passes by accident.** The merged arena's index keeps the base's
geometry and adds a `merge` block. `bake_mode` becomes `"nf4-merged"`, never the base's `"nf4-quantize"`. The `merge` block
records:
- the base index sha256;
- the base's qfp-v1 expert fingerprint;
- `adapter_id`;
- the sha256 of the adapter's expert tensors;
- scale = alpha / r;
- the source, and the trained-base form;
- the writer's version and quantizer parameters;
- the sha256 of the retention receipt (point 5).

**4. Server rules** (`adapter_dir.servable_refusals` and `serve_paged`), each with a test, in this order:
- **(i)** A merged arena is served only when the opt-in `E4B_PAGED_MERGED_ARENA=1` is set.
- **(ii)** A merged arena is served only with its OWN adapter: the adapter's id and its expert-tensor sha256 must equal the
  `merge` block's. Otherwise it is a new refusal, 13: a merged arena for a different adapter. A merged arena with NO adapter
  is refusal 13 too, because serving it as the base would serve a trained function under the base's name.
- **(iii)** Base verification uses the triple. The adapter's recorded expert fingerprint must equal `merge.base_qfp_v1`
  (refusal 12 against the base the merge started from, never against the merged rows, which can never match). Refusal 9
  keeps its meaning for a v1 adapter.
- **(iv)** Refusal 1 passes only when (i) to (iii) hold. The test removes each input in turn, so a new branch can never
  be shadowed by refusal 1 or 9 (the PA-1 lesson).
- **(v)** No double application. The server never merges expert tensors itself. With a merged arena, the adapter's expert
  tensors are verified by sha256 and otherwise ignored.
- **(vi)** Levers stay as today. `E4B_SERVE_EXP_INT4` re-reads the checkpoint and would drop the delta, so it stays refused
  (refusal 5).
- **(vii)** `/health` and `/v1/models` carry the `merge` block and the retention summary.

**5. A retention receipt, recorded, not gating.** The writer computes D0's Step-1 quantities for every projection:
ρ, c and f, and the energy-weighted ρ̄ per projection kind. It writes them as a receipt beside the arena.
- They are reported, not a gate. D0 showed ρ̄ 0.71 for NF4 from Q while loss was RETAINED, so a geometric threshold would
  refuse a merge that held, and choosing one now would be a rule chosen after the data.
- The gate is the registered served read's loss and logit result, per the train-to-serve spec's gated-cell rule.

**6. Formats.** NF4 first: the only expert arena format the paged server serves. INT4 (adding ΔW to the
fp32 stacks before `_pack_stack`) and MXFP4 are later items, each with its own test, outside the first code PR.

## Tests (CPU unless marked)

- **Parity:** ΔW = 0 reproduces the base arena byte for byte, on a synthetic arena written with the bake's call.
- **Graded against D0:** on a synthetic stack, the merged rows equal `NF4(S + ΔW)` exactly as D0's own repack computed it.
  The receipt's ρ, c and f equal D0's Step-1 functions on the same inputs.
- **Provenance:**
  - the `merge` block is complete;
  - `bake_mode` is `nf4-merged`;
  - a merged arena never passes refusals 3 and 4 as a plain base.
- **The server rules** (i)-(vii), each a refusal test with its input removed, and mutants:
  - drop the adapter-id check;
  - compare the fingerprint against the merged rows;
  - apply expert tensors at load;
  - let a merged arena serve without an adapter.
- **GPU (gated, through the GPU-contract harness):** one served forward on a merged OLMoE arena equals the same arena's
  dequantized forward on CPU within the server's own rounding. The registered served read prices the rest.

## The served read that follows (outline only; the PREREG is its own PR)

On the D0 fixture, the read compares:
- REF (the adapter unmerged, the trainer's function);
- the paged server on a merged arena (`source="arena"`);
- the paged server on a merged arena (`source="checkpoint"`);
- the paged base with the attention adapter only.

It measures teacher-forced logits against REF, held-out loss on D0's 64 unseen rows, the canary, and W1/W16 cost. Its
gate is the spec's gated-cell rule. The tools' sha256 are committed to the receipt store before any run. Its fetches import
no links (rsync `--no-links`, no tar fallback). Its launch needs an explicit approval comment on its merge PR.
