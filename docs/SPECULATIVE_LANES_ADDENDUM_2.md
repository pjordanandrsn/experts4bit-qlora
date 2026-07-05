# ROUTED-STREAM / SPECULATIVE LANES — ADDENDUM 2: CACHE-LANE POSTURE (post-S9)

```
status:   prediction/order addendum — commit BEFORE (a) the n=1024 passes if not yet
          run (margins telemetry), and (b) any Phase-1 trace collection (spec changes).
          Shares Addendum 1's commit window; batch them.
cites:    plan-routed-v3 <hash> + PLAN_ROUTED_V3_AMENDMENTS_A1-A4 <hash>
          SPECULATIVE_LANES_PLAN_20260705.md <hash> + ADDENDUM 1 <hash>
          PREREG_AMENDMENT_LADDER_REPIN_n1024.md <hash>
          AGENT_HANDOFF_POST_AUDIT_20260705.md <hash>
```

## 1. Orders (modify plan-routed-v3 Phase 1 by citation; do not edit v3)

- **O1 — adapter-active traces (supersedes v3 §1.2 trace spec).** Collect each trace
  workload TWICE on identical teacher-forced token sequences: base model, and
  adapter-active (the nf4-trained adapter). Same prompts, same logging.
  Rationale: deployment locality is adapted-model locality, not base-model locality;
  and the paired collection IS lane S-B's measurement (base-vs-adapted routing
  Jaccard per example) as a byproduct. Marginal cost: 2× trace passes; the <1%
  passive-logging budget and trace-mode rules from A-series apply unchanged.
- **O2 — margin-aware eviction (extends v3 §1.4 policy list).** Add one replay
  policy: LRU with eviction priority up-weighted for cache entries whose recent hits
  served bottom-decile-margin routing decisions (churn-prone experts). Zero GPU;
  one more pass through the simulator.

## 2. Predictions (ledger entries, timestamped by this commit)

- **P-B1 (margin → locality).** Across layers, Spearman corr(near-margin fraction,
  consecutive-token routed-set Jaccard) < −0.4, where near-margin = bottom decile of
  that layer's margin distribution. Tested jointly by the n=1024 margins and the
  Phase-1 traces; this commit must predate the join.
- **P-B2 (derived).** Layers in the top quartile of near-margin fraction show ≥ 1.5×
  the routing churn (1 − Jaccard) of bottom-quartile layers.
- **P-B3 (lane odds, graded at program end).** Cache built for ≥ 1 precision lane:
  40%. Speculation built: 15%. routed-stream-reactive ships: 70%.
- **P-B4 (coupling, conditional).** If the re-pin primary does NOT clear (S9 holds),
  cache priority demotes one tier — the ≥8-bit serve lanes lose their quality
  justification; the 16-bit lane survives on memory-floor + no-dequant grounds only.
  If it clears, the int8/16-bit cache lanes hold the A2 economic priors unchanged.

## 3. Gate modification (handoff-level)

- **G-mod.** The routed-stream lane-portfolio decision reads the re-pin branch as an
  input alongside v3's per-precision kill rules. The kill rules themselves are
  unchanged; only the portfolio-level prioritization gains the input.

## 4. What will not be claimed

- No cache-viability claim from margins alone; P-B1/P-B2 require the trace join.
- The train-phase DO-NOT-BUILD (hot-static pinning) decision is not reopened here;
  it answered a different question and stands.
- Nothing denominated in G while S9 holds.

---

## Executor's filing notes (added at commit time; not part of the handoff text)

- **Commit-window status:** batched with Addendum 1's window as instructed.
  (a) The n=1024 passes are MID-COLLECTION (started 18:08Z); the router-margins
  telemetry this addendum consumes was enabled by the re-pin amendment's manifest
  BEFORE launch and is **unchanged by this file** — P-B1 consumes that data, it does
  not modify its collection. (b) **Zero Phase-1 trace collection has occurred**
  (queue item T7 untouched); O1/O2 therefore precede all trace collection, as
  required. No n=1024 join of any kind has been computed at filing time.
- Cite hashes: plan-routed-v3 — **not in repo** (standing note,
  POST_AUDIT_WORK_QUEUE.md); `PLAN_ROUTED_V3_AMENDMENTS_A1-A4` @ 971065c;
  `SPECULATIVE_LANES_PLAN` @ ba23461; Addendum 1 @ 1271f96;
  re-pin amendment @ 94c3931; post-audit handoff @ 8a8928e.
- O1's byproduct note interacts with the queued S-B jobs
  (`runs/job_manifest/spine_extras_jobs.jsonl`): those measure base-vs-adapted
  routing on the pinned EVAL set (lanes-plan S-B as written); O1's paired traces add
  the TRACE-workload version when Phase 1 runs. Both proceed; neither substitutes
  for the other.
- P-B1..P-B4 are indexed in `docs/PREDICTION_LEDGER.md` as filed-not-graded.

---

<!-- ots-attestation-footer -->

**OpenTimestamps anchor (self-attestation footer):**

- **OTS proof timestamp for visible document:** `2026-07-05T18:22:03Z` (the moment the current `.ots` was submitted to the calendars; this is the legally operative timestamp for the visible file as published).
- **Disclosed pre-footer content hash:** `c67e654ec5dd2e0374b09403cc55df86e4df6ff5b8bb4052bcd67a1bdca85019` (the SHA-256 of the document *before* this footer was appended — disclosed inside the OTS-anchored visible document for human-readable historical reference; this hash is *not* the payload of the current `.ots` file).
- integrity-attestor glyph (`core.fingerprint`, first 8 bytes of the disclosed pre-footer hash): `[&0=?0Oo?&O!!+?.~]`
- Drunken-bishop randomart (full disclosed pre-footer SHA-256, OpenSSH-style):

```
+----[SHA256]-----+
|         o.o=++  |
|          o+o* +o|
|            E.= *|
|       .   . B +.|
|        S . X + =|
|       o   X o *+|
|        . o + * =|
|         . . + = |
|            . =o |
+-----------------+
```

- **Payload hash actually covered by the current `.ots`:** see `ots info SPECULATIVE_LANES_ADDENDUM_2.md.ots`; by construction this is `SHA-256(this entire file including this footer)` and `ots verify SPECULATIVE_LANES_ADDENDUM_2.md.ots SPECULATIVE_LANES_ADDENDUM_2.md` succeeds against the on-disk bytes.
- Anchor file: `SPECULATIVE_LANES_ADDENDUM_2.md.ots`
- Calendars: a.pool.opentimestamps.org, b.pool.opentimestamps.org, a.pool.eternitywall.com, ots.btc.catallaxy.com
- **Provenance posture (load-bearing):** the **OTS proof timestamp** above is the legal anchoring time for the visible document — that is what the calendars witnessed. The **disclosed pre-footer content hash** is *not* anchored by the current `.ots` file; it is *disclosed inside* the OTS-anchored visible document as a human-readable historical record of what the file's bytes hashed to immediately before this footer was appended. A reviewer verifying the visible file runs `ots verify` against the on-disk bytes; a reviewer wanting to confirm the disclosed pre-footer hash recomputes `SHA-256` of the file with everything from `<!-- ots-attestation-footer -->` onward stripped. Both checks are independent; neither replaces the other.
