# N2 — ROUTED-STREAM PHASE 0–1, RECONSTRUCTED (preregistration)

```
status:   preregistration of a RECONSTRUCTION — committed BEFORE any run (R5).
          plan-routed-v3 is not in this repository; this document reconstructs a
          minimal Phase 0–1 from everything v3-adjacent that IS committed:
          A1–A4 (971065c) · Addendum 2 O1/O2 + P-B1..P-B4 (149fbbc) · A2 economics
          priors · T6/A3 T_ovh protocol · LAYOUT_FACTS host rule · the S-B pairing.
          If v3 lands, results re-grade against the real plan; deviations are the
          reconstruction's fault, not v3's.
host:     one A5000 (T5c single-architecture discipline), resident traces
          (placement is bitwise-innocent — D2/D3 — so traces collected resident are
          valid for offload-serving simulation).
authorized: operator "please" on option (b), 2026-07-06.
```

## Phase 0 — probes (reconstructing "0.2b" + T6/A3)

- **P0.a Host attestation (LAYOUT_FACTS rule):** bitwise eval-repeat on THIS pod before
  anything else (int8-resident n=64 per-example eval ×2; require 64/64 bitwise, else stop).
- **P0.b BW_gather:** pinned H2D bandwidth at *expert-gather granularity* — timed copies of
  k=8 slabs per layer at each precision's true slab size (nf4 3.54 MB, int8 6.68 MB,
  bf16 12.58 MB per expert incl. absmax), 100 reps, median GB/s per precision; plus the
  256 MB-block ceiling for reference. t_fetch(p) per token = 16 layers × 8 experts ×
  slab(p) / BW_gather(p).
- **P0.c T_ovh per precision (T6/A3 protocol):** resident greedy decode, back-to-back in the
  same pod session, ≥5 reps per mode via `scripts/decode_repeat.py` for {nf4, int8, bf16};
  median ± spread; T_ovh(p) = 1/tok_s_median.
- **P0.d A2 economics, re-derived from measurement:** ceiling gap = t_fetch/T_ovh and
  gain(h) = (T_ovh+t_fetch)/(T_ovh+(1−h)·t_fetch) − 1 per precision, replacing A2's priors
  (+20/+38/+79%) with measured values. A2's shape prediction (int8 ≈ 1.9× nf4's gap,
  bf16 ≈ 3.6–4× — byte-ratio-driven) is graded here.

## Phase 1 — traces + simulation (reconstructing §1.2 per O1, §1.4 per O2)

- **Traces (O1 pairing):** batch-1 greedy decode, 16 fresh alpaca instructions
  (train[11088:11104] — beyond every pinned set), 256 new tokens each ≈ 4096 decode tokens
  per config. Configs: **nf4 base**, **nf4 + nf4-adapter** (the O1 adapter-active pair), and
  **int8 base** (per-precision locality check). Per token per layer: routed top-k set and
  router top-k boundary margin (prefill excluded — decode-time locality is the question).
- **Locality metric:** consecutive-token routed-set Jaccard per layer; churn = 1 − mean J.
- **h(S) simulation (§1.4 reconstructed):** replay each trace through a global LRU over
  (layer, expert) slabs with byte budget S ∈ {0.25, 0.5, 1, 2, 4} GB per precision; h(S) =
  hit fraction. **Policies:** plain LRU, and O2's margin-aware LRU (eviction priority
  up-weighted for entries whose recent hits served bottom-global-decile-margin decisions).
- **Kill rule (reconstructed from A2's "<10%"):** for precision p, if max over S ≤ 2 GB of
  gain(h(S)) < 10%, the cache lane for p is killed; reactive stands. A2 expects: nf4 killed,
  int8/16-bit spared.

## Predictions graded here (filed at 149fbbc, before any trace existed)

- **P-B1:** across layers, Spearman corr(near-margin fraction, consecutive-token routed-set
  Jaccard) < −0.4. Operationalization (stated now, pre-join): near-margin fraction per layer
  = fraction of decode tokens whose top-k boundary margin falls below the GLOBAL bottom-decile
  threshold (per-layer deciles would make the fraction constant by construction).
- **P-B2:** top-quartile near-margin-fraction layers show ≥ 1.5× the churn of bottom-quartile.
- **O1 byproduct:** base-vs-adapted decode-trace routing Jaccard (S-B's trace-workload
  number; the eval-set number was 0.94).
- P-B3 (lane odds) does NOT grade here — it grades at lane end, after the build/kill decision.

## RESULT (2026-07-06, A5000, `runs/results/postaudit/n2b/`; rev1 margin-hook red fixed)

Host attest 64/64 bitwise (P0.a passed). Phase-0 probes: BW_gather 13.3 GB/s (nf4/int8) /
9.0 (bf16); T_ovh 79.7/95.6/72.9 ms; t_fetch 34.0/63.9/179.6 ms/token. Three traces ×4k
decode tokens.

**Core lane verdict — SPARE all three cache lanes (build is justified), correcting A2.**

| precision | ceiling gap | best h(2GB) | best gain | verdict |
|---|---|---|---|---|
| nf4 | +43% | 0.854 | +34.3% | SPARE |
| int8 | +67% | 0.626 | +33.7% | SPARE |
| bf16 (proxy stream) | +246% | 0.424 | +43.1% | SPARE |

Decode routing has **low sequential locality** (consecutive-token Jaccard ≈ 0.30, churn ≈ 0.70)
but a **bounded working set** a 2 GB LRU captures well — so caching pays on working-set capture,
not sequential reuse. A2's prior (**kill nf4**, spare int8/16-bit) is **corrected**: nf4 is
spared too (gain +34%). The cache is a plain-LRU working-set cache.

**O2 margin-aware eviction — NO benefit (clean negative).** Margin-LRU ≈ plain LRU (int8
0.626→0.630, nf4 unchanged, bf16 0.424→**0.386 worse**). Do not build the margin-aware policy.

**P-B1 (margin→locality) — BORDERLINE, directionally right.** Spearman corr(near-margin frac,
Jaccard) across the 16 layers: nf4-base −0.415 (HOLDS <−0.40), int8-base −0.435 (HOLDS),
nf4-adapter −0.350 (FAILS). All negative and ~−0.4, so low-margin layers do route less stably
— but with n=16 layers the estimate straddles the committed −0.40 bar (2/3 clear it). Weakly
supported, not robust.

**P-B2 (churn quartile ≥ 1.5×) — FAILS.** Top-vs-bottom near-margin-quartile churn ratio is
1.09–1.10× (real, tiny). The margin signal is too weak to stratify layers for a policy —
consistent with O2's no-benefit.

**O1 byproduct — base-vs-adapter decode routing Jaccard 0.771** (vs eval-set 0.942): adapters
move routing MORE under autoregressive decode than under teacher-forced eval, reinforcing S-B.

**Net:** build the simple working-set LRU cache (SPARE, A2-corrected); skip the margin-aware
refinement (O2 no help, P-B2 fails, P-B1 borderline). P-B3 (lane odds) grades whenever a real
build/kill decision is made — the kill table now supplies its input. Reconstruction caveat
stands: re-grade against plan-routed-v3 if it lands.

## What will not be claimed

- Nothing about v3's actual Phase 0–1 — this is a reconstruction, labeled as such throughout.
- No cache BUILD from this session (v3's own rule: no cache before Phase-1 h(S) exists;
  after this session h(S) exists — the build/no-build decision is then Jordan's, informed by
  the kill table).
- No cross-session decode comparisons (T6); all Phase-0/1 numbers are one pod session.
- One model (OLMoE), one host class; Qwen3 parts of the lane stay unrun (A100 work).

---

<!-- ots-attestation-footer -->

**OpenTimestamps anchor (self-attestation footer):**

- **OTS proof timestamp for visible document:** `2026-07-06T04:36:32Z` (the moment the current `.ots` was submitted to the calendars; this is the legally operative timestamp for the visible file as published).
- **Disclosed pre-footer content hash:** `faf7756b6d16598ad8bbe3b80c6fc488ec579394266af3e2ed18f56187983d12` (the SHA-256 of the document *before* this footer was appended — disclosed inside the OTS-anchored visible document for human-readable historical reference; this hash is *not* the payload of the current `.ots` file).
- **Prior disclosed pre-footer hashes (chain, newest first):**
  - `2026-07-06T02:51:02Z` `bc4db0ef06f20437bf188617876c1a6917f204ff926749e79bdaec0734b01288`
- integrity-attestor glyph (`core.fingerprint`, first 8 bytes of the disclosed pre-footer hash): `[$%$==O0@0!:0O#*%]`
- Drunken-bishop randomart (full disclosed pre-footer SHA-256, OpenSSH-style):

```
+----[SHA256]-----+
|                 |
|                 |
|        E  .     |
|        .=+.    .|
|     . oS**+.. .o|
|      *o.+B+o .o |
|     ooo.o.. .. +|
|      o=o+o.o. o=|
|     .o++o=++o.+ |
+-----------------+
```

- **Payload hash actually covered by the current `.ots`:** see `ots info N2_PHASE01_RECONSTRUCTION.md.ots`; by construction this is `SHA-256(this entire file including this footer)` and `ots verify N2_PHASE01_RECONSTRUCTION.md.ots N2_PHASE01_RECONSTRUCTION.md` succeeds against the on-disk bytes.
- Anchor file: `N2_PHASE01_RECONSTRUCTION.md.ots`
- Calendars: a.pool.opentimestamps.org, b.pool.opentimestamps.org, a.pool.eternitywall.com, ots.btc.catallaxy.com
- **Provenance posture (load-bearing):** the **OTS proof timestamp** above is the legal anchoring time for the visible document — that is what the calendars witnessed. The **disclosed pre-footer content hash** is *not* anchored by the current `.ots` file; it is *disclosed inside* the OTS-anchored visible document as a human-readable historical record of what the file's bytes hashed to immediately before this footer was appended. A reviewer verifying the visible file runs `ots verify` against the on-disk bytes; a reviewer wanting to confirm the disclosed pre-footer hash recomputes `SHA-256` of the file with everything from `<!-- ots-attestation-footer -->` onward stripped. Both checks are independent; neither replaces the other.
