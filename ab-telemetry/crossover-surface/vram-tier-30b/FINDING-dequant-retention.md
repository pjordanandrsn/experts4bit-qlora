# FINDING: bf16 dequant retention makes 30B-on-24GB impossible on the current stack
## (the fv ladder never ran — this replaces it, and it's bigger than what it replaces)

**Date:** 2026-07-12 · **Prereg:** `prereg_vram30b.json` @ `e611d96` · **Host:** RunPod
SECURE RTX 3090 24GB (L=19.22 GB/s), ~$0.9 incl. two wedged community rolls · pod
deleted, 404-verified, evidence pulled by watchdog.

## What happened

Every arm — probe, warmup, R (pure RAMStore streaming), and fused fv=0.0 — OOM'd
on the first forward pass. `LADDER_STOP first_oom_at_fv=0.0`. The offload installed
correctly (`homed … (14.50 GB)` pool to pinned CPU RAM), yet the process died with
**22.37 GiB allocated** while asking for one more 384 MiB dequant buffer, inside
`bitsandbytes/nn/parametrize.py:39 → dequantize_4bit` mid-network.

## Root cause (archaeology-confirmed)

The whole-layer path dequantizes each staged block's packed 4-bit experts to bf16
for the fused MoE forward (~1.1 GB per Qwen3-30B block). Those dequant outputs are
**retained across blocks** (saved-for-backward; gradient checkpointing is not
discarding the parametrize-forward products), so VRAM grows linearly with depth:

- 3090 24 GB: base ~6 GB + ~14 blocks × 1.1 GB ≈ 22.4 GB → **OOM at block ~14 of 48** ✓ (matches the 22.37 GiB reading)
- Prior PRO 6000 96 GB whole arms (same model, this repo's archived logs):
  `qwen30b-lo-decisive` max_active **56.9–57.0 GiB**, `conv-ab` **59.4–60.4 GiB**
  ≈ base + 48 × 1.1 GB — the full-depth retention, previously unremarked because
  the cards were big enough to absorb it.
- OLMoE (all boxes): pool 3.22 GB → retention ≈ 12.9 GB + small base ≈ the ~13 GiB
  max_active seen on EVERY OLMoE arm — and why resident-vs-offload footprints
  matched (12.89 vs 13.05 GiB on the H100): the dominant term was never the packed
  weights, it was the retained dequants.

Reconciliation note: the earlier "Qwen3-30B trains at 7.16 GB peak with offload"
note is WRONG as a training-VRAM claim and should not be relied on (the archived
logs above are authoritative).

## Consequences

1. **The 24 GB consumer story for 30B-class MoE is blocked by this, not by the
   packed pool.** Offload frees the 14.5 GB of packed weights; checkpointing then
   leaves ~53 GB of transient bf16 on the card. Fixing the retention (recompute
   dequants in backward / free-after-use in the parametrize path, or checkpoint
   the dequant inside the block region) is THE unlock — after it, 30B QLoRA fits
   24 GB with the pool on RAM/VRAM tiers, and the fv ladder becomes meaningful.
2. Yesterday's "unattributed 3090-30B hang" plausibly = the same retention
   thrashing before OOM. Treat as probably-identified, pending the fix.
3. The prereg's fv ladder is deferred, not failed: S1–S4 were never tested
   (no arm trained). A 4090 re-try would re-confirm a diagnosed failure for money
   — skipped deliberately (prereg fallback overridden by root cause).
4. All OLMoE-based surface results stand: retention is a constant additive
   footprint per model, invisible to s/step store comparisons.

## Next action (queued to the code lane, front of #2)

Fix the retention in the public seam (axolotl `feature/expert-store` +/or the bnb
parametrize path), gate with: 30B probe arm trains on 24 GB, max_active drops from
~57 GiB to base+O(1 block); then re-run this prereg's ladder unchanged.

## Evidence

`q30-evidence.tgz` at mini `~/q30-evidence/` (SENT for all attempts incl. the
`dataset_processes` wedge, OOM logs, configs). Ops traps this run: axolotl
`dataset_processes` defaults to os.cpu_count() → 256 workers on a cgroup-capped
pod wedged preprocessing for 60+ min (cap to 16 → 61 s); the probe hang-guard must
exclude one-time dataset prep.
