### The merged-arena writer: a trained expert delta baked offline into a new NF4 arena (serves nothing yet)

- `experts4bit_qlora.merged_arena.write(base_arena, adapter_dir, out, source="arena" | "checkpoint")` writes an arena
  with the base's geometry, each row re-quantized from `S + s * B[e] @ A[e]` with the bake's own quantizer call
  (bench/d0/ARENA-DESIGN.md, path (a)).
  - It proves the base first. The base arena's qfp-v1 must equal the adapter's recorded expert fingerprint, and every
    row of S, re-quantized with no delta, must reproduce the base bytes exactly. The scale is the manifest's
    `lora_scale_applied.experts`, which must equal alpha / r.
  - `source="checkpoint"` runs grouped-nf4-gemm's own `bake_nf4` with a quantizer that adds the delta.
  - The index records a `merge` block (`bake_mode: "nf4-merged"`), including the merged arena's own fingerprint, and a
    retention receipt (D0's Step-1 sums) is written beside it, as a record that does not gate.
- **A merged arena is refused wherever an arena is opened** (the loader, the NF4, MXFP4 and training NVMe residencies,
  and the hybrid tier), so it is never read as a base. Serving one with its own adapter is a later, opt-in change.
- `adapter_schema.arena_expert_fingerprint(arena, stacks)`: an arena's expert qfp-v1 with no model loaded, read through
  the same path as the served one.
- `tests/test_merged_arena.py` (CPU, on a real `bake_nf4` arena) covers:
  - ΔW = 0 reproduces the bake byte for byte from both sources;
  - the merged rows equal D0's `NF4(S + ΔW)`;
  - the delta is checked against ExpertsLoRA's own low-rank path, with doubled-scale and transpose mutants caught by
    value;
  - the receipt equals D0's sums;
  - the merge block is complete;
  - every refusal.
