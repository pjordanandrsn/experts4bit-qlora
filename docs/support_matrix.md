# Support matrix

Not "works / doesn't work." Each cell is classified with the taxonomy so a user can tell whether
a path is validated, broken, impractical, unsupported, not tested, blocked, or impossible.

## Evidence scope

- Bundle: `olmoe-qlora-grid-20260705-1351` (OLMoE-1B-7B) + first Qwen3-30B-A3B probe, 2026-07-05
- Host: rented RunPod RTX A5000 24 GB, torch 2.8.0+cu128, bitsandbytes 0.49.2, transformers 5.13.0
- These classifications apply to the listed runs/host. "validated" here means reproduced under
  those stated conditions (OLMoE across 3 seeds), not universal.
- Provenance caveat: OLMoE repeat/decode jobs are `debug_only` on the per-job commit check only
  (git-archive worker trees); metrics/env/versions captured. See `docs/provenance_contract.md`.

## Taxonomy

`validated` reproduced under stated conditions · `broken` expected to work, fails ·
`impractical` runs but exceeds a realistic memory/time budget · `unsupported` not implemented or
not validated · `not_tested` no evidence yet · `blocked` gated by hardware/access/library ·
`impossible` representation/conversion cannot be made safe.

## OLMoE-1B-7B

| storage mode | train resident | train offload | decode resident |
|---|---|---|---|
| nf4 | validated (3 seeds) | validated (3 seeds) | validated (repeat-5) |
| int8 | validated (3 seeds) | validated (3 seeds) — best repeated eval | validated (repeat-5) |
| fp4 | observed (1 run) | observed (1 run) | validated (repeat-5) |
| fp8 | observed (1 run) | observed (1 run) | observed (1 run) |
| bf16 | observed (1 run) | observed (1 run) | observed (1 run) |
| fp16 | observed (1 run) | observed (1 run) | observed (1 run) |

"observed (1 run)" = ran cleanly in the six-mode grid but not seed-repeated; a rung below
`validated` on the claim ladder. Full numbers: `docs/OLMOE_EXPERTSNBIT_GRID.md`.

### OLMoE cross-cutting paths

| path | status | note |
|---|---|---|
| adapter portability (same-mode query) | validated pair (seed 0) | diagonal of the matrix; runnable + useful |
| adapter portability (upward: train 4-bit → query 8/16-bit) | observed, quality preserved | not yet multi-seed |
| adapter portability (downward: train 8-bit → query 4-bit) | quality_shift (mild) | small consistent degradation |
| adapter portability (any → fp4 query) | quality_shift | fp4 query degrades every adapter |
| expert-streaming hot-static pinning | unsupported (by decision) | concentration gate not met (diffuse wall); not built |
| offloaded training without gradient checkpointing | unsupported (fails loudly) | invariant enforced in code |

## Qwen3-30B-A3B (scale-transfer probe; no OLMoE numbers inherited)

| path | status | note |
|---|---|---|
| nf4 resident decode | validated (this host) | fits 24 GB at 20.04 GB, 4.14 tok/s |
| int8 resident | impractical | CUDA OOM > 24 GB VRAM on this card |
| nf4 / int8 offload (train or decode) | blocked | 25 GB container RAM cap kills streaming load; needs >40 GB-RAM pod |
| adapter portability at 30B | not_tested | gated follow-up |
| expert-streaming at 30B | not_tested | sentinel only if a larger model shows concentration (OLMoE did not) |

## Storage-mode representability

All six modes (`nf4`, `fp4`, `int8`, `fp8`, `bf16`, `fp16`) are implemented and round-trip
through `state_dict` with validated construction metadata (merged in the finish pass; tested on
CPU + four CUDA arches). No mode is `impossible` for the fused-expert layout; the un-validated
cells above are `observed`/`not_tested`, not representation failures.

## How to read this

The project does not name a winning mode. It records, per (model, mode, operation), whether the
path is validated, merely observed, impractical, blocked, or unsupported — so a user can decide
against their own fit/fidelity/speed/portability/VRAM constraints. Promotion to `validated`
requires seed-level reproduction; `blocked`/`impractical` are first-class, actionable answers.

---

<!-- ots-attestation-footer -->

**OpenTimestamps anchor (self-attestation footer):**

- **OTS proof timestamp for visible document:** `2026-07-05T14:00:25Z` (the moment the current `.ots` was submitted to the calendars; this is the legally operative timestamp for the visible file as published).
- **Disclosed pre-footer content hash:** `db71e2ace2609227ff367b5f898c2650c0389c72d98c8b6a95a5be3627de2c37` (the SHA-256 of the document *before* this footer was appended — disclosed inside the OTS-anchored visible document for human-readable historical reference; this hash is *not* the payload of the current `.ots` file).
- integrity-attestor glyph (`core.fingerprint`, first 8 bytes of the disclosed pre-footer hash): `[!@=:?+%&?+0.#++=]`
- Drunken-bishop randomart (full disclosed pre-footer SHA-256, OpenSSH-style):

```
+----[SHA256]-----+
| . O.            |
|. O +o           |
| + o+ .          |
|. .+ .           |
|. o .   S o .    |
|.. o .   B = .   |
|. + = . + * o    |
|   %oE+o.. .     |
|  o.B*==...      |
+-----------------+
```

- **Payload hash actually covered by the current `.ots`:** see `ots info support_matrix.md.ots`; by construction this is `SHA-256(this entire file including this footer)` and `ots verify support_matrix.md.ots support_matrix.md` succeeds against the on-disk bytes.
- Anchor file: `support_matrix.md.ots`
- Calendars: a.pool.opentimestamps.org, b.pool.opentimestamps.org, a.pool.eternitywall.com, ots.btc.catallaxy.com
- **Provenance posture (load-bearing):** the **OTS proof timestamp** above is the legal anchoring time for the visible document — that is what the calendars witnessed. The **disclosed pre-footer content hash** is *not* anchored by the current `.ots` file; it is *disclosed inside* the OTS-anchored visible document as a human-readable historical record of what the file's bytes hashed to immediately before this footer was appended. A reviewer verifying the visible file runs `ots verify` against the on-disk bytes; a reviewer wanting to confirm the disclosed pre-footer hash recomputes `SHA-256` of the file with everything from `<!-- ots-attestation-footer -->` onward stripped. Both checks are independent; neither replaces the other.
