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
| adapter portability (nf4/int8 same + cross-mode query) | validated (3 seeds) | int8-offload adapters transfer well to both columns; nf4-trained is query-agnostic |
| adapter portability (downward: train int8 → query nf4) | quality_shift (validated, 3 seeds) | +~0.007 eval degradation, holds across seeds |
| adapter portability (upward: train nf4 → query int8) | validated (3 seeds), quality preserved | nf4→int8 ≈ nf4→nf4 |
| adapter portability (any → fp4/bf16/fp16 query) | observed (seed 0) | fp4 query degraded every adapter; not re-tested in phase 3 |
| expert-streaming hot-static pinning | unsupported (by decision) | concentration gate not met (diffuse wall); not built |
| offloaded training without gradient checkpointing | unsupported (fails loudly) | invariant enforced in code |

## Qwen3-30B-A3B (scale-transfer probe; no OLMoE numbers inherited)

| path | status | note |
|---|---|---|
| nf4 resident decode | validated | fits 24 GB at 20.04 GB, ~4–5 tok/s |
| int8 resident | impractical on 24 GB | 34.5 GB peak — OOMs a 24 GB card, fits an A100 |
| nf4 offload decode | validated (A100 host) | peak **4.07–4.41 GB** — small-card fit for a 30B model; prefetch 1.44x |
| int8 offload decode | validated (A100 host) | peak **4.38–5.02 GB** — width-collapse topology transfers |
| adapter portability at 30B | not_tested | gated follow-up |
| expert-streaming at 30B | not_tested | no trigger — OLMoE's wall was diffuse |

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

- **OTS proof timestamp for visible document:** `2026-07-05T14:52:13Z` (the moment the current `.ots` was submitted to the calendars; this is the legally operative timestamp for the visible file as published).
- **Disclosed pre-footer content hash:** `167959271ad9d4470dbb94a6c6d539950a19c980b81c9616a0ffe303c1669313` (the SHA-256 of the document *before* this footer was appended — disclosed inside the OTS-anchored visible document for human-readable historical reference; this hash is *not* the payload of the current `.ots` file).
- **Prior disclosed pre-footer hashes (chain, newest first):**
  - `2026-07-05T14:00:25Z` `db71e2ace2609227ff367b5f898c2650c0389c72d98c8b6a95a5be3627de2c37`
- integrity-attestor glyph (`core.fingerprint`, first 8 bytes of the disclosed pre-footer hash): `[:0=#O#+=:%!#!oo=]`
- Drunken-bishop randomart (full disclosed pre-footer SHA-256, OpenSSH-style):

```
+----[SHA256]-----+
|   ...+ ..+=B.oo=|
|  .E * . ..X.o.==|
| .. * o o + . B+.|
|  .O o   o . * ..|
|  o.+   S   + .  |
|   ..  .   .     |
|    .o           |
|    ...          |
|     ..          |
+-----------------+
```

- **Payload hash actually covered by the current `.ots`:** see `ots info support_matrix.md.ots`; by construction this is `SHA-256(this entire file including this footer)` and `ots verify support_matrix.md.ots support_matrix.md` succeeds against the on-disk bytes.
- Anchor file: `support_matrix.md.ots`
- Calendars: a.pool.opentimestamps.org, b.pool.opentimestamps.org, a.pool.eternitywall.com, ots.btc.catallaxy.com
- **Provenance posture (load-bearing):** the **OTS proof timestamp** above is the legal anchoring time for the visible document — that is what the calendars witnessed. The **disclosed pre-footer content hash** is *not* anchored by the current `.ots` file; it is *disclosed inside* the OTS-anchored visible document as a human-readable historical record of what the file's bytes hashed to immediately before this footer was appended. A reviewer verifying the visible file runs `ots verify` against the on-disk bytes; a reviewer wanting to confirm the disclosed pre-footer hash recomputes `SHA-256` of the file with everything from `<!-- ots-attestation-footer -->` onward stripped. Both checks are independent; neither replaces the other.
