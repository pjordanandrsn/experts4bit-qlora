# BM4: gated prefetch lane passes K1/K2/K4 (K3 misses by 1.6%); residue decomposed to H-D — the copies are exonerated

**Date:** 2026-07-12/13 · **Prereg:** `prereg_baremetal4.json` (OTS, frozen before
provisioning) · **Host:** Latitude `g3-h100-small` DAL — **same physical server as
bm1/bm2/bm3 (5th provision**, `sv_gXQvNeZGl5zpb`; re-caught twice from our own
teardowns), raid-0 (md0, 7 TB), fio QD1 3.57 GB/s, L = 56.6 GB/s. **Code:** axolotl
`feature/expert-store@5cb0dbc` + local probe commit `52a0661` (patch-file deploy;
no-push session), private `e4b-ssdtier@4073d01` (v3 reader; tests-only delta from
the prereg'd `0a5eb8b`, reader byte-identical). **Cost:** ~$3.2 across two
provisions (one lost to a watchdog race, below) · torn down, 404-verified,
ramcode shredded. Floors n=2: ON spread 0.0012, OFF 0.0018.

## Part A — the gated ladder (prereg 2bbd2e2 bars, third run, same silicon)

| f (ram share) | OFF s/step | ON v3 s/step | ON vs OFF | v2 rebuilt (bm3) | v1 (bm2) |
|---:|---:|---:|---:|---:|---:|
| 1.0 (n=2) | 0.2399 | 0.2401 | **+0.06%** (inert ✓) | −0.1% | −0.1% |
| 0.9375 | — | 0.2511 | vs ON floor **+4.6%** | +6.6% | +3.1% |
| 0.875 | — | 0.2813 | | | −0.2% |
| 0.75 | — | 0.3668 | | | +7.5% |
| 0.5 | 0.5160 | 0.4274 | **−17.2%** | +13.1% | +23.9% |
| 0.0 (all-flash) | 0.7704 | 0.6964 | **−9.6%** | −6.9% | +41.9% |
| centroid 3-tier | 0.3997 | 0.4060 | **+1.6%** | +10.4% | +6.8% |

- **K1 PASS** — ON at flash share 1/16 lands **+4.6%** over the ON floor (bar ≤5%;
  v2 measured +6.6%). The flat region exists.
- **K2 PASS** — ON ≤ OFF at every measured interior point, and at f=0.5 the lane
  **removes 32.2% of the OFF flash penalty** (bar ≥25%; v2 *added* 24%).
- **K3 FAIL (small)** — centroid ON 0.4060 vs OFF 0.3997 = **+1.6%** (v2: +10.4%).
  The f=1.0 inertness sub-check passed (+0.06%).
- **K4 PASS** — all-flash ON −9.6% (the run rule preserved and slightly improved
  the v3 rebuild's −6.9% win).

### What the gate actually did (decision counters, per 40-step arm)

| arm | enq | allow_run | allow_hide | deny | served (hit+wait) | read-EMA end | cadence-EMA end |
|---|---:|---:|---:|---:|---:|---:|---:|
| f=0.0 | 1121 | 1121 | 0 | 0 | 1121 | 17.5 ms | 18.1 ms |
| f=0.5 | 520 | 0 | 521 | 559 | 520 | 19.3 ms | 10.3 ms |
| f=0.75 | 3 | 0 | 8 | 992 | 3 | **89.9 ms** | 12.6 ms |
| f=0.875 | 1 | 0 | 6 | 834 | 1 | **119.8 ms** | 10.5 ms |
| f=0.9375 | 1 | 0 | 0 | 557 | 1 | **118.2 ms** | 10.3 ms |
| f=1.0 | 0 | — | — | — | 0 | — | 6.4 ms |
| centroid | 4 | 0 | 7 | 993 | 4 | 68.1 ms | 14.0 ms |

Two mechanisms, both legible in the counters:

1. **The interior win is real hiding, not luck:** at f=0.5 the warm read-EMA
   (19.3 ms) sits just under the 2-gap window (~20.6 ms), the gate admits ~half
   the opportunities, and each admitted read is mostly hidden (519 waits totaling
   4.45 s ≈ 8.6 ms residual wait per served block vs a 19 ms sync read).
2. **Deny-starvation (new finding, refinement queued):** at f ≥ 0.75 the only
   read samples are the first cold ones (drop_caches + fresh store), the read-EMA
   seeds at 90–120 ms, the gate then denies everything — and with no new reads
   the EMA **never warms**. A self-reinforcing quiet state. It is conservative in
   the right direction (ON degenerates to OFF + ~1%, which is why K1 passes and
   nothing regresses), but at f=0.75 a warm 19 ms read against a 50 ms window
   would have been admissible and winning. Fix candidate: decay the read-EMA
   toward optimistic over denied opportunities, or admit an exploratory read
   every N denials. **Not applied — frozen per the no-tune clause; report as-is.**

### K3 discussion

The centroid miss is 6.3 ms/step on single arms (no centroid floor pair), against
within-run f=1.0 floors of ~1–2 ms — so it is small but probably not pure noise.
Two candidate mechanisms, both consistent with the counters: the mixed-tier
cadence (VRAM-tier stages are aliases, ~instant, so the uniform cadence-EMA
underestimates the true wall-window to the next flash block — windows read too
small, useful reads get denied, and the lane pays observe overhead without
serving) and the same deny-starvation as above (centroid read-EMA stuck at
68 ms). A per-tier or per-gap-composition window estimate is the natural next
increment. Either way the v3 gate cut the centroid regression from +10.4% to
+1.6%.

### Product posture (unchanged, sharpened)

`prefetch: false` stays the default for the knee's target regime (small flash
share) — the gate's own verdict agrees: it denies there. What v3 adds is that
**prefetch: true is now safe everywhere and profitable at high flash share**
(−9.6% all-flash, −17.2% at f=0.5): the lane no longer needs the operator to
know the regime in advance.

## Part B — the seq-residue, decomposed: H-D (between-copy growth); H-B and H-C refuted

Residue (R − V), this box: **44.1 / 63.9 / 120.8 ms** at seq 64 / 512 / 2048.

- **R3 PASS** — 120.8 ms at seq2048 is within factor 2 of the 102 ms from boxes
  1–2. (The seq64 point runs hotter than the earlier 21 ms — see anchor-drift
  note below; R3 was registered on seq2048.)
- **R4 PASS** — probe overhead +0.63% (R@64 with vs without probe; bar ≤3%).
- **R5 → H-D** with unusually clean discriminators, per-stage medians
  (steps 10..40):

| arm | submit (host) | device copy per GB | alloc retries |
|---|---:|---:|---:|
| R seq64 | 0.033 ms | 17.67 ms/GB | 0 |
| R seq512 | 0.034 ms | 17.67 ms/GB | 0 |
| R seq2048 (n=2) | 0.034–0.035 ms | 17.66 ms/GB | 0 |

The nominally-async submit is flat (no allocator/serialization stall → **H-C
refuted**), and the device-side copy runs at **17.67 ms/GB = 56.6 GB/s — full
link speed at every sequence length** (no bandwidth degradation → **H-B
refuted**), while the exposed residue nearly triples. The growth lives entirely
**between** the stage copies.

**Profiler arm (H-D branch, identification-not-pass/fail; 3 trainer steps,
seq2048 R):** HtoD copies total 71 ms/step device time (124 calls, 1.72 ms avg
— consistent with the probe), while `cudaStreamSynchronize` blocks the host
**20.8 ms/step across 6 calls** (trainer sync points: loss `.item()`, grad-norm
clip, logging). Mechanism this points at: each host sync drains the queue; the
first stages after a drain have no queued compute to hide behind, so their
copies land exposed — and at long seq both the drains (more queued work per
sync) and the per-block compute between refills grow. This also closes the loop
with R1's earlier exclusion: lookahead cannot help, because the copy is already
enqueued as early as possible — it is the *queue*, not the copy, that empties.
Follow-on (LOW, opportunistic): instrument around the trainer's sync points
(defer `.item()`/clip syncs, or measure with `logging_steps` >> 1) to confirm
the drain attribution.

## Cross-provision anchor drift (caveat)

This provision's seq64 anchors sit ~7% above bm2/bm3 on identical configs
(OFF f=1.0: 0.2399 vs 0.2237; V64 0.1977 vs 0.2075 — note V moved the *other*
way), despite being the same physical serial. Fresh OS install + rebuilt env
(torch wheel lineage identical; cause not isolated). Every verdict in this
report is within-run, so none depends on cross-provision comparability; only
the v1/v2/v3 delta *table* carries this caveat.

## Ops ledger (each cost an iteration tonight)

1. **PEP-668:** Ubuntu 24 system pip refuses installs — `--break-system-packages`
   on every pip call.
2. **Non-root user installs land in `~/.local/bin`** — a nohup'd shell doesn't
   have it on PATH; every arm exited rc=127 while gates (python -m) passed.
   `export PATH="$HOME/.local/bin:..."` at runner top + a `command -v axolotl`
   gate + preprocess rc checks.
3. **`Path.exists()` raises EACCES** probing `/root/...` as ubuntu → private
   test collection crashed on bare metal (pods ran as root). Fixed in
   `e4b-ssdtier@4073d01` (tests-only).
4. **Watchdog friendly fire:** the fallback watcher latched a BUILD_ABORT state
   while the claim was cleared for "safety", its 12-min grace expired mid-fix,
   and it sentinel'd the billcap — **the box was torn down under an active
   debugging session** (~$1.1 lost, clean 404). Policy fixed in the watcher:
   gate failures now pull evidence + mail ONLY and never initiate teardown (the
   billcap deadline still caps the bill); teardown-on-completion still requires
   the unclaimed path.
5. **Host-key race on re-provision:** the reimaging box briefly answers with its
   old host key; `accept-new` pins it and the fresh key then hard-fails. Clear
   known_hosts immediately before the *first successful* contact, not at
   deploy start.

## Evidence

`bm4-final.tgz` (SENT incl. all attempt logs, 22 arm logs, configs, gates, probe
JSONLs, prefetch stats) sha256
`d61ec519f268e55c6f54f9e3518739ce78fdd02eb7df24f2a4a17709d1b6e357`; profiler
addendum `prof_r2048.txt` sha256 `6b37e072…`; both at mini `~/bm4-evidence/`.
`bm4_results.json` alongside (this dir). Server deleted → GET 404; RunPod at 0
pods; nothing billing.
