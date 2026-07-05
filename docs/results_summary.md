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
- **int8-offload best eval — CONFOUNDED; D3 now paid and the confound is worse than feared.**
  Best in 3/3 seeds (1.0261 ± 0.0079), but the one-step placement certificate came back
  **bitwise-exact** (offload training is numerically identical to resident at step granularity,
  all five configurations — `docs/TRAIN_PLACEMENT_CERTIFICATE.md`), and T5 forensics showed
  every seed-matched placement pair in the repeat grid was a **cross-architecture comparison**
  (4090↔A5000; evaluator offset 0.0026–0.0054 on identical bytes — the scale of the claimed
  effect). No placement mechanism exists at one step; the "wins" are best explained as
  architecture offset + run-level variance. Quarantined (`pending-mechanism`). Note also S9:
  the frozen-precision axis itself is within sampling noise at n=64 (n=1024 re-pin running,
  `docs/NULL_LADDER_1024_AMENDMENT.md`). **The honest headline is memory, not this eval:**
  bf16-offload trains at 2.41 GB — below the nf4 floor, no quantization — vs 14.54 GB resident
  (decomposition code-verified in `docs/OFFLOAD_MEMORY_FACTS.md`).
- **BEFORE-training eval tracks reconstruction fidelity** (int8 < nf4; 4-bit worst), 3/3 pairs.

## What a single run got wrong, and repeating fixed

- **fp4 decode is NOT faster than nf4.** The single-run grid showed fp4 12.59 vs nf4 10.12 tok/s;
  repeat-5 puts them within a standard deviation (fp4 12.87 ± 0.20, nf4 12.68 ± 0.22). The nf4
  10.12 was a slow outlier. The repeat grid doing exactly its job.

## Train-here / query-there (portability, seed 0, measured pairs — not universal)

Adapters were trained under each storage/offload mode and queried under each. Seed-0 across all
modes, then **phase 3 re-ran the nf4/int8 columns across 3 seeds** (24 jobs, all `claim_usable`).
Validated across seeds: the **downward-transfer penalty** (int8-trained degrades ~0.007 queried
under nf4; nf4-trained is query-agnostic) and **int8-offload adapters transfer well** to both
columns. The seed-0 absolutes were optimistic (int8-offload→int8 1.0126 → 3-seed 1.0260 — same
drift lesson as fp4-decode). Querying under fp4 degraded every adapter, but that stays a seed-0
observation (fp4 not a phase-3 column). Every cell is a measured pair, not universal
compatibility. See `docs/MODE_DECOUPLED_ADAPTERS.md`.

## Expert-streaming: is the offload wall worth routing around? No, here.

Profiling OLMoE offload training (int8 and nf4) found the transfer stall **diffuse**, not
hot-expert concentrated: top 10% of layer/expert pairs hold ~20% of projected stall, top 20% hold
~35–36% — below the pre-registered 40% / 60% gate. Budget simulation shows no cheap knee (2 GB
pinned buys back only 48% / 74% of projected stall). Decision: **do not build hot-static pinning**
for this model/path. A clean negative that was declared useful in advance. See
`docs/EXPERT_STREAMING_PROFILE.md`.

## Scale transfer to Qwen3-30B-A3B (probe) — the topology transfers

nf4 resident **fits** a 24 GB card (20 GB); int8 resident is **impractical** on 24 GB (34.5 GB).
And **offload preserves fit at 30B**: on a high-RAM host, both nf4 (4.07–4.41 GB) and int8
(4.38–5.02 GB) offload peak at ~4–5 GB — a small-card footprint for a 30-billion-parameter model,
with int8-offload only ~0.5 GB above nf4-offload (the same width-collapse topology OLMoE showed,
at 15x the parameters). The first probe's offload was `blocked` purely by a 25 GB container RAM
cap; cleared at 236 GB RAM, every offload config ran. Offload stays a memory-for-speed trade
(0.67–1.65 tok/s vs ~5 resident; prefetch 1.3–1.4x). No OLMoE eval numbers inherited. See
`docs/QWEN3_30B_EXPERTSNBIT_GRID.md`.

## The umbrella result

There is no single best mode. There is a decision surface:

- need it to **fit** on a small card → offload flattens the storage-width memory axis.
- want **fidelity per VRAM** on OLMoE → int8-offload is the candidate.
- **decoding** → 4-bit and 16-bit are comparable; int8/fp8 are slower; fp4 is not the speed win a
  single run suggested.
- **train here, query there** → measured per pair; fp4-as-query costs quality; don't assume.
- **buy back offload speed with a little VRAM** → not available here; the wall is diffuse.
- **at 30B** → resident nf4 fits a 24 GB card; int8 needs a bigger card; **offload fits a 30B
  model in ~4–5 GB of VRAM** (needs a host with enough system RAM to stage it).

## Provenance and reproduction

Every number traces to a job-local result under `runs/jobs/` or `runs/expert_streaming/jobs/`,
aggregated by controller-only summarizers into `runs/results/`. Re-run via the manifests in
`runs/job_manifest/` under the multi-pod protocol (`docs/RUNPOD_DISTRIBUTED_VALIDATION.md`); gate
with `scripts/validate_job_provenance.py`; docs are OTS-stamped (`docs/*.md.ots`).

## Open, gated follow-ups (not run without instruction)

- ~~Seeded portability (phase 3)~~ — **done**: 24 commit-attested query jobs; downward-transfer
  penalty and int8-offload transfer validated across 3 seeds (nf4/int8 columns). (Both
  placement-differentiated readings now quarantined pending the run-level anomaly resolution —
  see `docs/TRAIN_PLACEMENT_CERTIFICATE.md`.)
- ~~D1/D2/D3 (audit debts)~~ — **done** same-day: D2 bitwise placement-exact at serve, D3
  bitwise placement-exact at one training step, D1 fired S9 (precision ladder within sampling
  noise at n=64). See `docs/MEASUREMENT_AUDIT.md` §7.
- **n=1024 ∅-ladder re-pin** — preregistered (`docs/NULL_LADDER_1024_AMENDMENT.md`), running.
- **Divergence-onset probe** (gated): 150-step twin trio with per-step hashes — the remaining
  open question is why the same-host single-run bf16 pair differed 0.0108 when one step is
  bitwise-exact.
- Speculative lanes (S-A response curve, S-B adapter-routing, …): `docs/SPECULATIVE_LANES_PLAN.md`
  — second-class by construction, byproduct analyses of the scheduled runs.
- ~~Qwen3-30B offload on a >40 GB-RAM pod~~ — **done** (A100, 236 GB RAM): offload fits a 30B
  model in ~4–5 GB VRAM; the topology transfers. See the Qwen3 section above.
- Related-work positioning: existing work made large-MoE inference practical (quantization,
  offload, expert caching, fused kernels) and QLoRA made low-bit adapter training practical for
  dense-style layouts; this apparatus targets the under-instrumented intersection — fused-MoE
  expert tensors during adapter training, with explicit storage modes, offload provenance,
  train/query portability, and reproducible validation grids.

---

<!-- ots-attestation-footer -->

**OpenTimestamps anchor (self-attestation footer):**

- **OTS proof timestamp for visible document:** `2026-07-05T18:11:19Z` (the moment the current `.ots` was submitted to the calendars; this is the legally operative timestamp for the visible file as published).
- **Disclosed pre-footer content hash:** `4f6df7c749dc6cb00cf3c32fcad96c387a7b4c7ac87623391d8e2a1e52a9e558` (the SHA-256 of the document *before* this footer was appended — disclosed inside the OTS-anchored visible document for human-readable historical reference; this hash is *not* the payload of the current `.ots` file).
- **Prior disclosed pre-footer hashes (chain, newest first):**
  - `2026-07-05T16:48:56Z` `7bd7d4a5b3c9f18fdd5ce7c6035132d00b162428ac8926de56507558979f86ca`
  - `2026-07-05T14:52:18Z` `ce2030434782e4ea2b1ada367261fb4a2ae1f4e4f14b674787382b0b101df026`
  - `2026-07-05T14:00:26Z` `0b455296684992211f5b5b703cb21bebd38cfdb33e8b15575b54fbe12e672327`
- integrity-attestor glyph (`core.fingerprint`, first 8 bytes of the disclosed pre-footer hash): `[o$0!$=&=o#!&0&@.]`
- Drunken-bishop randomart (full disclosed pre-footer SHA-256, OpenSSH-style):

```
+----[SHA256]-----+
|                 |
|                 |
|            o .  |
|         . . *.+.|
|        E . o Bo+|
|       B o . .+*.|
|      + o .. @.o=|
|       ... .&=X..|
|       ...o==Xo. |
+-----------------+
```

- **Payload hash actually covered by the current `.ots`:** see `ots info results_summary.md.ots`; by construction this is `SHA-256(this entire file including this footer)` and `ots verify results_summary.md.ots results_summary.md` succeeds against the on-disk bytes.
- Anchor file: `results_summary.md.ots`
- Calendars: a.pool.opentimestamps.org, b.pool.opentimestamps.org, a.pool.eternitywall.com, ots.btc.catallaxy.com
- **Provenance posture (load-bearing):** the **OTS proof timestamp** above is the legal anchoring time for the visible document — that is what the calendars witnessed. The **disclosed pre-footer content hash** is *not* anchored by the current `.ots` file; it is *disclosed inside* the OTS-anchored visible document as a human-readable historical record of what the file's bytes hashed to immediately before this footer was appended. A reviewer verifying the visible file runs `ots verify` against the on-disk bytes; a reviewer wanting to confirm the disclosed pre-footer hash recomputes `SHA-256` of the file with everything from `<!-- ots-attestation-footer -->` onward stripped. Both checks are independent; neither replaces the other.
