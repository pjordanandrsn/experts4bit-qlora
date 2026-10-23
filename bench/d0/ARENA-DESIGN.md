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
- **Adapter refusals run in two phases, and only under an adapter** (`if cfg.adapter` in `build_engine`):
  - **Before the load,** from records: refusals 1-6, 8 and 9 (`adapter_dir.servable_refusals`). An adapter with nonzero
    expert LoRA is refusal 1. Refusal 9 (an unverified expert base) is checked before it.
  - **After the load,** measured: refusals 11 and 12 (`fingerprint_check`). Refusal 12 compares the adapter's recorded
    expert fingerprint (qfp-v1, with `fp32_source` for a double-quantized trained base) against the served arena.
  - A branch that a pre-load refusal always raises before is unreachable, whatever its place in the list.
- **The manifest's `lora_scale_applied` holds both scales** (attention and experts), read from the trained modules.
  `adapter_dir` checks only the attention entry against alpha / r.
- **Attention adapters merge at load** (`merge_attention_lora`): fp32 on device, one rounding, every target checked before
  anything is merged.
- **INT4 experts** (`E4B_SERVE_EXP_INT4`) are repacked from the checkpoint at load, and **MXFP4** experts are gpt-oss's
  native blocks only. Both are refused under an adapter (refusal 5).

## Measured for this note (CPU, bitsandbytes 0.50.2, USD 0)

- **fp32 dequantization makes the parity exact.** On five synthetic stacks (Gaussian in bf16 and fp32, a zero block, an
  all-negative block, a Student-t heavy tail), quantize, then dequantize in fp32, then re-quantize reproduced the codes
  AND the fp32 absmax byte for byte.
  - The reason: a block's largest element normalizes to exactly ±1.0, which NF4's codebook holds, so in fp32 the
    re-quantized absmax equals the old one bit for bit.
  - A bf16 dequantization kept the codes but changed the absmax of every block whose absmax was not already
    bf16-exact.
- **The bake's own quantizer runs on CPU.** grouped-nf4-gemm's `default_quantize_expert(dev="cpu")` on a bf16 W gave
  the same codes and absmax as `quantize_4bit` on fp32(W). That is three of three trials on a 512 × 1024 stack.
- **CPU against CUDA.** The registration notes that a CPU quantize can differ from CUDA's at ties, since CUDA multiplies by
  the reciprocal. On D0's fp32-absmax fixture, a CPU re-quantization of the GPU-trained base still reproduced 2,048 of
  2,048 expert projections. The writer does not rely on that: it checks itself (point 1).

## The design

**1. An offline writer, not a load-time merge.** `experts4bit_qlora.merged_arena.write(base_arena, adapter_dir, out, source=...)`
writes a NEW arena with the base's geometry. Each (layer, expert) row's segments are re-quantized from S + ΔW, where
ΔW = s · B[e] @ A[e] is computed in fp32.
- **Why offline:** the server has no resident expert path. An in-memory re-quantization of the VRAM tier would leave the
  DRAM and NVMe tiers reading the unmerged arena, so one model would serve two functions by tier. An arena is also an
  artifact that can carry provenance and be fingerprinted.
- **The quantizer is the bake's own call:** `F.quantize_4bit` with nf4, blocksize 64 and `compress_statistics=False`, in
  the arena's row layout and alignment.
- **S is dequantized in fp32,** from the raw codes and the fp32 absmax, never through a bf16 dequantization (measured
  above).
- **The self-check:** before any merge, the writer re-quantizes every base row with ΔW = 0. It refuses unless every row
  reproduces the base arena exactly. Whatever device the writer runs on, a tie it would round differently from the
  bake's device is caught here and never written.
- **The parity test** (CI, CPU): with ΔW = 0, the writer reproduces a base arena byte for byte.
  - That base arena comes from grouped-nf4-gemm's REAL `bake_nf4`, with its default quantizer on CPU, through the
    pinned install, as `tests/test_adapter_fingerprint_serve.py` already bakes one. It is never a re-implementation,
    which would prove only that the writer agrees with itself.
  - If the test cannot be made to hold, the fallback is a grouped-nf4-gemm PR first, kernel-first: a bake hook that
    receives (layer, expert, projection) and returns a delta.
- **The scale:** s is the manifest's `lora_scale_applied.experts`. The writer refuses unless it equals alpha / r. The
  code PR adds the experts entry to `adapter_dir`'s existing check, beside the attention entry.

**2. The source S is the decision with evidence behind it.**
- **`source="arena"` (the default):** S is the base arena's own dequantized rows. For an fp32-absmax trained base these are
  Q_train(W) exactly, verified by the adapter's expert fingerprint against the base arena BEFORE the merge. This is D0's
  "NF4 from Q" cell: RETAINED, gap_to_ref +0.0011 and −0.0036 nats.
- **`source="checkpoint"`:** S is the raw checkpoint W in fp32, at a pinned revision. This is D0's "NF4 from W" cell:
  RETAINED, gap_to_ref +0.0007 and +0.0027.
  - It must prove that W is the base: quantize(W) must reproduce the base arena's rows exactly, or the writer refuses.
  - The `merge` block records the revision.
- **A double-quantized trained base** (`fp32_source` present) is a third case. The arena's fp32 rows are not Q_train, and
  the server never re-compresses: that is deliberate, because the compression's statistics can differ between GPU models.
  The writer then merges into the fp32 arena rows and records `trained_base: "double-quantized"`. The served read prices
  that combination, which D0 did not measure directly.

**3. Provenance: what each side records, so no check passes by accident.** The merged arena's index keeps the base's
geometry and adds a `merge` block. `bake_mode` becomes `"nf4-merged"`, never the base's `"nf4-quantize"`. The `merge` block
records:
- the base index sha256;
- `base_qfp_v1`, the base's expert fingerprint;
- `merged_qfp_v1`, the same fingerprint, which the writer computes over its own output;
- `adapter_id`;
- the sha256 of the adapter's expert tensors;
- the scale s;
- the source, its revision for `source="checkpoint"`, and the trained-base form;
- the writer's version and quantizer parameters;
- the sha256 of the retention receipt (point 5).

**4. Server rules** (`adapter_dir.servable_refusals`, `build_engine` and `serve_paged`), each with a test. The rules are
grouped by PHASE, because a pre-load refusal shadows every later branch.
- **Before the load, from records only** (the merged index's `merge` block and the adapter's files; nothing measured):
  - **(i)** A merged arena is served only when the opt-in `E4B_PAGED_MERGED_ARENA=1` is set. Without it, a merged arena
    is refused.
  - **(ii)** A new refusal, 13: a merged arena served with an adapter other than its OWN. The adapter's id and its
    expert-tensor sha256 must equal the `merge` block's.
  - **(ii-b)** Refusal 13 also covers a merged arena with NO adapter, because serving it as the base would serve a trained
    function under the base's name.
    - The adapter refusals run only under an adapter, so this check is `build_engine`'s own: it reads the arena index
      for EVERY config.
    - It refuses whether or not (i) is set.
  - **(iii)** The base check uses the triple. The adapter's recorded expert fingerprint must equal `merge.base_qfp_v1`,
    and `lora_scale_applied.experts` must equal both alpha / r and `merge.scale`. Refusal 9 keeps its meaning for a v1
    adapter.
  - **(iv)** Refusal 1's exemption is decided in this phase, and only when (i) to (iii) hold. Otherwise an adapter with
    live expert LoRA raises refusal 1 before 13 or 12 is ever evaluated.
- **After the load, measured:**
  - **(v)** The served expert fingerprint must equal `merge.merged_qfp_v1`. This is refusal 12's machinery, compared
    against the merged arena's own fingerprint, never against the base's (which merged rows can never match). It
    refuses an arena file swapped in under a copied index.
  - **(vi)** Each part is applied exactly once.
    - The expert delta is in the arena. The server never merges expert tensors itself: it verifies the adapter's expert
      tensors by sha256 and otherwise ignores them.
    - The attention LoRA still merges once at load (`merge_attention_lora`), as today.
- **Unchanged:**
  - **(vii)** The levers stay as today. `E4B_SERVE_EXP_INT4` re-reads the checkpoint and would drop the delta, so it
    stays refused (refusal 5).
  - **(viii)** `/health` and `/v1/models` carry the `merge` block and the retention summary.

**5. A retention receipt, recorded, not gating.** The writer computes D0's Step-1 quantities for every projection:
ρ, c and f, and the energy-weighted ρ̄ per projection kind. It writes them as a receipt beside the arena.
- They are reported, not a gate. D0 showed ρ̄ 0.71 for NF4 from Q while loss was RETAINED, so a geometric threshold would
  refuse a merge that held, and choosing one now would be a rule chosen after the data.
- The gate is the registered served read's loss and logit result, per the train-to-serve spec's gated-cell rule.

**6. Formats.** NF4 first: the only expert arena format the paged server serves. INT4 (adding ΔW to the
fp32 stacks before `_pack_stack`) and MXFP4 are later items, each with its own test, outside the first code PR.

## Tests (CPU unless marked)

- **Parity, against the real bake:**
  - ΔW = 0 reproduces a `bake_nf4` arena byte for byte, for `source="arena"`.
  - The same holds for `source="checkpoint"`, against a bake of the same synthetic checkpoint (bf16 W in the bake, fp32(W)
    in the writer).
  - Named cases: a zero block (absmax 0) and an all-negative block.
  - The self-check refuses an arena with one flipped code.
- **The function, computed independently of the writer:**
  - On a toy MoE layer, the unmerged expert LoRA (the trainer's module, forward on S) is compared with the forward on
    the dequantized merged arena. They must agree within NF4's rounding bound.
  - The toy uses square projections, so that a transpose is shape-legal.
  - Mutants that must fail: ΔW transposed, and the scale doubled.
- **Graded against D0:** on a synthetic stack, the merged rows equal `NF4(S + ΔW)` exactly as D0's own repack computed it.
  The receipt's ρ, c and f equal D0's Step-1 functions on the same inputs.
- **Provenance:**
  - the `merge` block is complete, `merged_qfp_v1` included;
  - `bake_mode` is `nf4-merged`;
  - a merged arena never passes refusals 3 and 4 as a plain base;
  - a scale other than alpha / r is refused.
- **The server rules** (i)-(viii): each new branch is probed with its input REMOVED, past every earlier PHASE, not just
  the earlier entries of its list.
  - A live-expert-LoRA adapter on a merged arena reaches 13, 12 or service, never refusal 1.
  - Attention and experts are each applied exactly once.
  - Mutants that must fail:
    - drop the adapter-id check;
    - compare the post-load fingerprint against `base_qfp_v1`;
    - apply the expert tensors at load;
    - let a merged arena serve without an adapter;
    - move refusal 13 to after the load (the live-expert-LoRA test must catch it).
- **GPU (gated, through the GPU-contract harness):** one served forward on a merged OLMoE arena equals the same arena's
  dequantized forward on CPU, within the server's own rounding. The registered served read prices the rest.

## The served read that follows (outline only; the PREREG is its own PR)

On the D0 fixture, the read compares:
- REF (the adapter unmerged, the trainer's function);
- the paged server on a merged arena (`source="arena"`);
- the paged server on a merged arena (`source="checkpoint"`);
- the paged base with the attention adapter only.

It measures teacher-forced logits against REF, held-out loss on D0's 64 unseen rows, the canary, and W1/W16 cost. Its
gate is the spec's gated-cell rule. The tools' sha256 are committed to the receipt store before any run. Its fetches follow
rule 8 of `bench/common/README.md` (rsync `--no-links`, no tar fallback). Its launch needs an explicit approval comment on its merge PR.
