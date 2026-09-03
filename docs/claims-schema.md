# Claims register — schema (draft for docs/claims.json in both repos)

One JSON object per claim. A claim is one sentence a reader could act on,
with a number where there is one. Every README/docs number must map to an
entry; CI can enforce that later.

```json
{
  "id": "e4b.serve.b1.qwen3-30b.nf4.5090",         // stable slug, never reused
  "package": "experts4bit-qlora" | "grouped-nf4-gemm",
  "area": "train" | "offload" | "serve" | "kernel" | "parity" | "provenance" | "portability",
  "claim": "one sentence, present tense, the thing a user gets",
  "value": 98.3, "unit": "tok/s",                    // omit for qualitative claims
  "model": "Qwen/Qwen3-30B-A3B", "hardware": "RTX 5090 (sm_120), rented Vast host",
  "conditions": "B=1, NF4 experts, fp8 paged KV, 512-token prompt, --no-fuse-qkv",
  "measured_on": "2026-09-03",
  "status": "verified" | "measured" | "projected" | "retired" | "superseded" | "open",
  "tier": "confirmed" | "measured" | "projected",   // repo's existing evidence tiers
  "evidence": ["bench/hybrid-g9/b1/RESULTS-b1-decomposition.md"],  // PUBLIC paths only
  "evidence_private": ["INT4B16/P25-PARITY.md"],    // exists but not in this repo -- reader cannot check it
  "supersedes": ["<id>"], "superseded_by": "<id>",
  "retired_reason": "why, in one sentence, with the measurement that retired it",
  "quoted_in": ["README.md#L45", "docs/METHODOLOGY.md#13"]
}
```

Status meanings:
- **verified** — reproduced under stated conditions, receipt public in this repo.
- **measured** — one run, receipt public. (The repos' own tier language.)
- **measured-private** — the receipt exists only in the private audit tree; the
  number is real but a reader of this repo cannot check it. MUST be flagged in
  prose, or the receipt published.
- **projected** — arithmetic, not a run.
- **retired** — was published, now known wrong; keep the entry so the retraction
  is findable, never delete.
- **superseded** — still true as measured, but a later entry replaces it as the
  number to quote (e.g. v0 offload tok/s superseded by the paged engine).
- **open** — a claim the docs make that has no evidence either way yet.

Human-readable companion: docs/STATUS.md renders this as three lists —
"what you get today" (verified/measured), "what changed" (superseded/retired
with the one-line reason), "what is open".
