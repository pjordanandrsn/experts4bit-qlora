# Results summary — the decision surface

experts4bit-qlora does not produce one winning quantization mode. It produces a measured decision
surface across fit, fidelity, speed, portability, and residency budget, so a user can decide
whether a fused-MoE storage/offload path is validated, observed, impractical, blocked, or
unsupported for their constraints. The apparatus — atomic-claim multi-pod execution, job-local
provenance, a controller gate, OTS-stamped docs — is the point as much as the numbers.

## Evidence scope

Bundle `olmoe-qlora-grid-20260705-1351`, 2026-07-05, rented RunPod RTX A5000 (24 GB), torch
2.8.0+cu128, bitsandbytes 0.49.2, transformers 5.13.0. OLMoE-1B-7B is the validated model; Qwen3-30B-A3B
is a scale-transfer probe and inherits no OLMoE numbers. Per-job commit for OLMoE repeat/decode
jobs is bundle-attested, not self-reported (git-archive worker trees; runner since fixed) — see
`docs/provenance_contract.md`.

## What graduated (OLMoE-supported, host-specific — reproduced across seeds 1337/2027/3407)

- **Offload collapses the storage-width memory gap.** Resident training peak scales cleanly with
  width (5.28 / 8.50 / 14.54 GB for 4- / 8- / 16-bit); offload flattens all of them to
  2.41–2.72 GB. Offload width-delta 0.20 GB vs resident 3.22 GB (ratio 0.06), 3/3 seeds.
- **int8-offload is a low-VRAM / high-fidelity training candidate for OLMoE.** Best held-out eval
  in 3/3 seeds (aggregate 1.0261 ± 0.0079, lowest of the four repeated modes) at a ~2.72 GB peak
  near the 4-bit floor. A candidate regime for OLMoE — not a Qwen3 claim, and single-run
  bf16/fp16-offload are excluded from the ranking.
- **BEFORE-training eval tracks reconstruction fidelity** (int8 < nf4; 4-bit worst), 3/3 pairs.

## What a single run got wrong, and repeating fixed

- **fp4 decode is NOT faster than nf4.** The single-run grid showed fp4 12.59 vs nf4 10.12 tok/s;
  repeat-5 puts them within a standard deviation (fp4 12.87 ± 0.20, nf4 12.68 ± 0.22). The nf4
  10.12 was a slow outlier. The repeat grid doing exactly its job.

## Train-here / query-there (portability, seed 0, measured pairs — not universal)

Adapters were trained under each storage/offload mode and queried under each. Upward transfer
(train coarser, query finer) roughly preserved same-mode quality; downward degraded a little more
(mild asymmetry); **querying under fp4 degraded every adapter** (`quality_shift`); int8-offload →
int8 was the single strongest cell (1.0126). Every cell is a measured pair, not evidence of
universal compatibility. See `docs/MODE_DECOUPLED_ADAPTERS.md`.

## Expert-streaming: is the offload wall worth routing around? No, here.

Profiling OLMoE offload training (int8 and nf4) found the transfer stall **diffuse**, not
hot-expert concentrated: top 10% of layer/expert pairs hold ~20% of projected stall, top 20% hold
~35–36% — below the pre-registered 40% / 60% gate. Budget simulation shows no cheap knee (2 GB
pinned buys back only 48% / 74% of projected stall). Decision: **do not build hot-static pinning**
for this model/path. A clean negative that was declared useful in advance. See
`docs/EXPERT_STREAMING_PROFILE.md`.

## Scale transfer to Qwen3-30B-A3B (probe)

nf4 resident **fits** a 24 GB card (20 GB, 4.14 tok/s); int8 resident is **impractical** (OOM);
the offload path is **blocked** by the pod's 25 GB container RAM cap during streaming load — a
host limit, not a mechanism failure. The transferable question ("does offload preserve fit at
30B") remains open pending a higher-RAM pod. See `docs/QWEN3_30B_EXPERTSNBIT_GRID.md`.

## The umbrella result

There is no single best mode. There is a decision surface:

- need it to **fit** on a small card → offload flattens the storage-width memory axis.
- want **fidelity per VRAM** on OLMoE → int8-offload is the candidate.
- **decoding** → 4-bit and 16-bit are comparable; int8/fp8 are slower; fp4 is not the speed win a
  single run suggested.
- **train here, query there** → measured per pair; fp4-as-query costs quality; don't assume.
- **buy back offload speed with a little VRAM** → not available here; the wall is diffuse.
- **at 30B** → resident nf4 fits; int8 needs a bigger card; offload needs a bigger-RAM host.

## Provenance and reproduction

Every number traces to a job-local result under `runs/jobs/` or `runs/expert_streaming/jobs/`,
aggregated by controller-only summarizers into `runs/results/`. Re-run via the manifests in
`runs/job_manifest/` under the multi-pod protocol (`docs/RUNPOD_DISTRIBUTED_VALIDATION.md`); gate
with `scripts/validate_job_provenance.py`; docs are OTS-stamped (`docs/*.md.ots`).

## Open, gated follow-ups (not run without instruction)

- Seeded portability (phase 3): 24 query jobs over the repeat adapters, for `validated` (not just
  seed-0) portability cells.
- Qwen3-30B offload on a >40 GB-RAM pod, to answer the offload-preserves-fit question at scale.
- Related-work positioning: existing work made large-MoE inference practical (quantization,
  offload, expert caching, fused kernels) and QLoRA made low-bit adapter training practical for
  dense-style layouts; this apparatus targets the under-instrumented intersection — fused-MoE
  expert tensors during adapter training, with explicit storage modes, offload provenance,
  train/query portability, and reproducible validation grids.

---

<!-- ots-attestation-footer -->

**OpenTimestamps anchor (self-attestation footer):**

- **OTS proof timestamp for visible document:** `2026-07-05T14:00:26Z` (the moment the current `.ots` was submitted to the calendars; this is the legally operative timestamp for the visible file as published).
- **Disclosed pre-footer content hash:** `0b455296684992211f5b5b703cb21bebd38cfdb33e8b15575b54fbe12e672327` (the SHA-256 of the document *before* this footer was appended — disclosed inside the OTS-anchored visible document for human-readable historical reference; this hash is *not* the payload of the current `.ots` file).
- integrity-attestor glyph (`core.fingerprint`, first 8 bytes of the disclosed pre-footer hash): `[.@oOO+#00*o##++:]`
- Drunken-bishop randomart (full disclosed pre-footer SHA-256, OpenSSH-style):

```
+----[SHA256]-----+
|  . ++=**.      +|
|   o.==B+      ..|
|    o..o..    .o.|
|      o.     ..oo|
|      .+S . . ...|
|      o. . o  .  |
|     . =. .  E * |
|      + +oo   B .|
|       ..o==     |
+-----------------+
```

- **Payload hash actually covered by the current `.ots`:** see `ots info results_summary.md.ots`; by construction this is `SHA-256(this entire file including this footer)` and `ots verify results_summary.md.ots results_summary.md` succeeds against the on-disk bytes.
- Anchor file: `results_summary.md.ots`
- Calendars: a.pool.opentimestamps.org, b.pool.opentimestamps.org, a.pool.eternitywall.com, ots.btc.catallaxy.com
- **Provenance posture (load-bearing):** the **OTS proof timestamp** above is the legal anchoring time for the visible document — that is what the calendars witnessed. The **disclosed pre-footer content hash** is *not* anchored by the current `.ots` file; it is *disclosed inside* the OTS-anchored visible document as a human-readable historical record of what the file's bytes hashed to immediately before this footer was appended. A reviewer verifying the visible file runs `ots verify` against the on-disk bytes; a reviewer wanting to confirm the disclosed pre-footer hash recomputes `SHA-256` of the file with everything from `<!-- ots-attestation-footer -->` onward stripped. Both checks are independent; neither replaces the other.
