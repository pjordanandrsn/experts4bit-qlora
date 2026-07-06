# NEXT-CAMPAIGN LANES — LICENSED CONTINUATIONS (successor to "queue empty")

```
status:   index + one ledger correction. Opens nothing by curiosity; every lane below
          cites the pre-committed rule that licensed it. None precede submission —
          this file IS the PR's future-work section, verbatim if desired.
cites:    PREDICTION_LEDGER.md (filed-not-graded index) · SPECULATIVE_LANES_PLAN §S-B/§S-C
          SPEC_LANES_ADDENDUM_2 (P-B1–P-B4) · plan-routed-v3 + A1–A4 (T7)
          final results picture 2026-07-06
```

## Ledger correction (file as a new entry; gradings are never edited)

> **2026-07-06 — Reviewer closure claim CORRECTED.** "Queue state after this
> document: empty" (DIVERGENCE_ONSET_PROBE_GO) and the reviewer's "the queue reads
> empty" overstated closure: the ledger's own filed-not-graded index held P-B1–P-B4
> open, and two graduation clauses had fired. Correct statement: the campaign's
> confirmatory queue is empty; three licensed continuations remain (below).
> Caught by the operator, not the apparatus — the three-verb test does not check
> claims of absence. Rule adopted: a closure claim is a claim; grade it against the
> filed-not-graded index before making it.

## Lane N1 — Routing-pinned serve (licensed by S-B graduation clause; FIRED) — **RUN 2026-07-06: PARTIAL**

Mechanism-complete: adapters move 5.6% of routing, gains track the movement
(ρ 0.58–0.60), forfeit measured ~100% of certified G = 0.0166 ± 0.0023.
Experiment: freeze/pin routing decisions from the train-base forward while serving
int8 experts; artifacts on disk; ~2–3 eval passes + one hook.
Success = recovers ≥ 50% of G at resident-certified placement.

**Result (`docs/N1_ROUTING_PINNED_SERVE.md`): R = 0.365 ± 0.092 — PARTIAL (middle branch).**
Pinning recovers ~37% of G (real, |t|≈4) but below the 0.50 bar; routing mismatch is a
significant term, the majority is value-space co-adaptation. Next step (no clock): a
seed-replicated follow-up to confirm before any serving claim.

## Lane N2 — Routed-stream Phases 0–6 (T7) — **PHASE 0–1 RUN 2026-07-06 (reconstruction)**

P-B4's coupling resolved BULLISH earlier. Phase 0–1 now run as a reconstruction
(`docs/N2_PHASE01_RECONSTRUCTION.md`, v3 not-in-repo):

**Build the simple working-set LRU cache — SPARE all lanes, A2's "kill nf4" prior corrected.**
Decode locality is low (consecutive-token Jaccard ≈0.30) but the working set fits 2 GB
(h 0.85/0.63/0.42), so the cache pays +34–43% on all precisions. O2 margin-aware eviction adds
nothing (clean negative); P-B1 (margin→locality) is borderline (−0.4-ish, n=16 fragile), P-B2
fails (1.1×). O1: adapters move routing more under decode (Jaccard 0.771) than eval (0.942).
**Remaining in N2:** Phases 2–6 (the actual cache build + serving integration), the Qwen3
parts (A100), and P-B3 lane-odds grading at a real build/kill decision — all no-clock.

## Lane N3 — S-C mixed-precision cell (HALF-licensed; one CPU join short) — **RUN 2026-07-06: CLOSED**

G-cleared precondition: MET. Fragility transfer: P-A5 held at 5× enrichment.
But the tail carried 45% (< 50% bar) and the per-(layer,expert) flip attribution
join has not run. Order of operations: run the attribution join (CPU, data on
disk); preregister the cell only if concentration ≥ 2× over uniform; prediction
template already in SPECULATIVE_LANES_PLAN §S-C.

**Result (`runs/results/postaudit/n3_fragility_attribution.md`): CLOSED.** The committed
literal ≥2× gate passed (top-10% of layer/expert pairs hold 52% of top-decile flip mass =
5.2× uniform) — but the added fragility-specificity control shows the *least*-fragile decile is
*equally* concentrated (55%). Routing flips concentrate on the same experts regardless of
precision fragility, so "top-fragility experts" is not a set distinct from "top-flip experts,"
and they don't track |d_i|. The cell's targeting premise is unsupported → per-expert precision
is not the dial. (The literal gate was under-specified; closed on the control, transparently.)

## Closed, and stays closed

S-D (fp8−bf16 n.s. at n=1024) · S-A strong form (graduation failed; the monotone
W_RMS→flips→sd table survives as one results paragraph, no runs) · Qwen3 sentinel
(gate failed by the letter) · S-E beyond the probe (GO clause holds; onset answered)
· placement (bitwise, twice, attested) · the 150-step gap (chaos, graded).

## Sequencing

Submission first — unchanged. N1 is a first-week-at-the-job or first-item-next-
campaign task; N2 is its own arc; N3 waits on one join. Nothing here has a clock.

---

## Executor's filing notes (added at commit time; not part of the handoff text)

- Correction ACCEPTED without reservation: my "queue empty" (and the probe doc's
  "Queue state … Empty") was scoped to the confirmatory queue in my head but stated
  broadly, and the filed-not-graded index (P-B1–P-B4, `docs/PREDICTION_LEDGER.md`)
  plus the S-B/S-C graduation clauses were the counterexample. Filed as a new dated
  ledger entry, not an edit. The adopted rule ("a closure claim is a claim; grade it
  against the filed-not-graded index") is now recorded.
- In-repo names: `SPEC_LANES_ADDENDUM_2` = `docs/SPECULATIVE_LANES_ADDENDUM_2.md`;
  `plan-routed-v3` remains not-in-repo (its A1–A4 amendments are at 971065c).
- N3's CPU attribution join is a next-campaign item by the sequencing ("nothing here
  has a clock; submission first") — NOT run now. Its inputs (per-example
  `routed_sets.jsonl` telemetry for the six n=1024 modes) are on disk in
  `runs/results/postaudit/postaudit_jobs/null1024_*/routed_sets.jsonl.gz`, so the
  join needs no new GPU when it is authorized.
- N1's artifacts (seed-0 portability adapters + train-base routing telemetry) are on
  the RunPod volume and in-bundle; the experiment needs a pod when authorized.
- results_summary's future-work section now points here; this doc is submission-ready
  as the PR's future-work section verbatim.

---

<!-- ots-attestation-footer -->

**OpenTimestamps anchor (self-attestation footer):**

- **OTS proof timestamp for visible document:** `2026-07-06T04:37:03Z` (the moment the current `.ots` was submitted to the calendars; this is the legally operative timestamp for the visible file as published).
- **Disclosed pre-footer content hash:** `a7d1ead39cfda1a4ac5f5cbabea2764ad6cbf9991353f54673f99f159d101e6a` (the SHA-256 of the document *before* this footer was appended — disclosed inside the OTS-anchored visible document for human-readable historical reference; this hash is *not* the payload of the current `.ots` file).
- **Prior disclosed pre-footer hashes (chain, newest first):**
  - `2026-07-06T02:10:11Z` `5c34eb0f8d386ea774e2e63d5d1e4a08979ff049a880522e4f7b5257b976ade8`
  - `2026-07-06T02:01:46Z` `85f56abb01baff49d61993aa00a323cd2d4a458b1c24632d7fca1664ad6a6f05`
  - `2026-07-06T00:36:01Z` `2d7b369eddc7837d5218176cd66716803cd8bd2acb4b4bd9f22eeb9967b9e39b`
- integrity-attestor glyph (`core.fingerprint`, first 8 bytes of the disclosed pre-footer hash): `[%=!:?%!~#&$!%:%o]`
- Drunken-bishop randomart (full disclosed pre-footer SHA-256, OpenSSH-style):

```
+----[SHA256]-----+
|             +o +|
|            o .==|
|           E . o*|
|         ..   . =|
|        S o  ...+|
|         * .oo ..|
|        =o.o=o.  |
|       +oo**o= . |
|       .=BB+O+.  |
+-----------------+
```

- **Payload hash actually covered by the current `.ots`:** see `ots info NEXT_CAMPAIGN_LANES.md.ots`; by construction this is `SHA-256(this entire file including this footer)` and `ots verify NEXT_CAMPAIGN_LANES.md.ots NEXT_CAMPAIGN_LANES.md` succeeds against the on-disk bytes.
- Anchor file: `NEXT_CAMPAIGN_LANES.md.ots`
- Calendars: a.pool.opentimestamps.org, b.pool.opentimestamps.org, a.pool.eternitywall.com, ots.btc.catallaxy.com
- **Provenance posture (load-bearing):** the **OTS proof timestamp** above is the legal anchoring time for the visible document — that is what the calendars witnessed. The **disclosed pre-footer content hash** is *not* anchored by the current `.ots` file; it is *disclosed inside* the OTS-anchored visible document as a human-readable historical record of what the file's bytes hashed to immediately before this footer was appended. A reviewer verifying the visible file runs `ots verify` against the on-disk bytes; a reviewer wanting to confirm the disclosed pre-footer hash recomputes `SHA-256` of the file with everything from `<!-- ots-attestation-footer -->` onward stripped. Both checks are independent; neither replaces the other.
