# Session 2 (design case): prefetch lane REFUTED as implemented; stripe delivers; residue closed

**Date:** 2026-07-12 · **Prereg:** `prereg_baremetal2.json` @ `2bbd2e2` (OTS, frozen
before provisioning) · **Host:** Latitude `g3-h100-small` DAL — **the same physical
server as session 1** (same id `sv_gXQvNeZGl5zpb`, same IP; caught re-entering stock
by the watcher 42 min after our morning teardown), re-provisioned `raid: raid-0`
(md0, 7 TB, both NVMe striped as /). L = 56.66 GB/s (matches 56.74 ✓).
**Cost:** ~$4 (1.75 h) · torn down by sentinel, 404-verified, ramcode shredded.
Same frozen OLMoE workload; floors from n=2 f=1.0 pairs: OFF spread **0.0002**,
ON spread 0.0012 — the cleanest floors of the program.

## S2 — the stripe point (PASS, top edge)

fio O_DIRECT 1MiB on the stripe: **QD1 3.61 GB/s** (= 2.4× single-drive QD1 1.52;
prereg band [2.4, 3.6] — lands at the boundary), QD2 3.88, QD4 3.93 (saturating).
Frozen-formula knee update: f_knee = 1 − S2_QD1/L = **0.936** (was 0.973 single).
Through-path effective: the all-flash OFF arm implies S_eff ≈ 6.44 GB / 0.528 s ≈
**12.2 GB/s** (per-slot reads overlap) → effective knee ≈ **0.78** — striping moved
the affordable flash share from ~3% to ~22% of blocks. All-flash step time dropped
0.752 vs box-1-single 1.183 s (−36%) on identical workload. **Striping works and
is worth its money for the flash tier.**

## K1/K2/K3 — the prefetch design case (ALL FAIL; diagnosis + fix path)

| f (ram share) | OFF s/step | ON s/step | ON vs OFF |
|---:|---:|---:|---:|
| 1.0 (floor, n=2) | 0.2237 | 0.2235 | −0.1% (lane inert ✓) |
| 0.9375 | 0.2323 | 0.2394 | +3.1% |
| 0.875 | 0.2720 | 0.2715 | −0.2% |
| 0.75 | 0.3495 | 0.3758 | +7.5% |
| 0.5 | 0.5187 | 0.6427 | +23.9% |
| 0.0 | 0.7519 | 1.0672 | +41.9% |
| centroid 3-tier | 0.4013 | 0.4284 | +6.8% |

- **K1 FAIL:** ON at flash-share 1/16 sits +7.1% over the ON floor (bar ≤5%). No
  flat region as implemented.
- **K2 FAIL:** ON ≤ OFF nowhere that matters; at f=0.5 the lane *adds* 42% to the
  flash penalty instead of removing ≥25%.
- **K3 FAIL** (composite): centroid ON > OFF; the f=1 inertness sub-check passed.

**Diagnosis (mechanism visible in the deficit's shape):** `FusedPrefetchReader` is
single-threaded double-buffer; when `take()` misses (read not finished within one
inter-flash-block gap), `stage()` issues its own synchronous read of the SAME block
while the background read is still in flight — **duplicate I/O contending on the
same device**. Misses become the common case as flash share rises (201 MB read ≈
56 ms at QD1 vs ~14 ms/block of compute), so the deficit grows with flash share —
exactly the +3% → +42% gradient measured. Second factor: the OFF path is already
mostly hidden at low share (f=15/16 pays only 8.6 ms/step over floor, ~4 ms/touch —
far below the 56 ms sync-read worst case), so the lane had little to win there.

**Fix path (queued to the code lane, after dequant-retention):**
1. **Wait-don't-duplicate:** on `take()` miss with the block in flight, block on the
   reader's completion instead of issuing a second read.
2. **Adaptive gating:** only read ahead when the predicted gap-compute exceeds the
   expected read time (placement knows the gaps; S_eff is measurable at install).
3. Deeper queue/multiple readers for consecutive-flash runs (contiguous placement).
Re-run this prereg's ON ladder unchanged after the fix.

## R1/R2 — the seq-residue, closed

Residue (R−V) reproduces on the same silicon to a tenth of a millisecond at the top:
**21.6 / 40.3 / 102.1 ms** at seq 64/512/2048 (box-1: 21.1 / 35.4 / 101.8) — **R2
PASS**, a real stable phenomenon. Discriminator: public `expert_offload_prefetch`
at seq2048 removed **0.8%** of the residue (0.3531 vs 0.3539 s) → **not**
copy-serialization hideable by lookahead → classification **H-B/H-C: the H2D path
itself degrades at long seq** (pinned-copy bandwidth under contention with larger
activation traffic, or allocator/event serialization). Copy-stream lookahead won't
fix it; it needs the copy path examined (pinned-buffer contention, per-copy event
overhead, allocator behavior at large max_active). Filed as the follow-on
instrumented probe — one profiled arm, next time a box is up for other reasons.

## Program state after this session

The measurement program registered at `2bbd2e2` is complete: stripe measured,
design case adjudicated (negative, mechanism identified, fix specified), residue
reproduced and classified. Combined with session 1 and the 3090 link-axis point:
the three-tier placement story is validated everywhere it was tested; the two
open items are now *code work*, both measurement-backed — (1) dequant retention
(the 24 GB/30B unlock, `vram-tier-30b/FINDING-dequant-retention.md`), (2) the
prefetch lane rebuild above. No further rentals needed until both land.

## Evidence

`bm2-final.tgz` (BM2_SENT, BUILD_SENT, results JSON, all 25 arm logs, configs)
sha256 `04082605317a5c6a54d08b9a7f56859aeb43620d677220d49795bcc6ed742970`
at mini `~/bm2-evidence/`; `bm2_results.json` alongside (this dir).
