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

## Lane N1 — Routing-pinned serve (licensed by S-B graduation clause; FIRED)

Mechanism-complete: adapters move 5.6% of routing, gains track the movement
(ρ 0.58–0.60), forfeit measured ~100% of certified G = 0.0166 ± 0.0023.
Experiment: freeze/pin routing decisions from the train-base forward while serving
int8 experts; artifacts on disk; ~2–3 eval passes + one hook.
Success = recovers ≥ 50% of G at resident-certified placement. This is the
highest-value open item and the most product-shaped.

## Lane N2 — Routed-stream Phases 0–6 (T7; never closed)

P-B1/P-B2 await the margins×traces join; P-B3 grades at lane end. Note for the
record: P-B4's coupling resolved BULLISH — G real ⇒ ≥8-bit serve is
quality-justified ⇒ the cache's int8/16-bit lanes keep their A2 priors (+38%/+79%
ceilings). The deciding measurement remains batch-1 decode temporal locality
(Phase 1, adapter-active per O1).

## Lane N3 — S-C mixed-precision cell (HALF-licensed; one CPU join short)

G-cleared precondition: MET. Fragility transfer: P-A5 held at 5× enrichment.
But the tail carried 45% (< 50% bar) and the per-(layer,expert) flip attribution
join has not run. Order of operations: run the attribution join (CPU, data on
disk); preregister the cell only if concentration ≥ 2× over uniform; prediction
template already in SPECULATIVE_LANES_PLAN §S-C.

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

- **OTS proof timestamp for visible document:** `2026-07-06T00:36:01Z` (the moment the current `.ots` was submitted to the calendars; this is the legally operative timestamp for the visible file as published).
- **Disclosed pre-footer content hash:** `2d7b369eddc7837d5218176cd66716803cd8bd2acb4b4bd9f22eeb9967b9e39b` (the SHA-256 of the document *before* this footer was appended — disclosed inside the OTS-anchored visible document for human-readable historical reference; this hash is *not* the payload of the current `.ots` file).
- integrity-attestor glyph (`core.fingerprint`, first 8 bytes of the disclosed pre-footer hash): `[+!=@~0#?!!&=*~=!]`
- Drunken-bishop randomart (full disclosed pre-footer SHA-256, OpenSSH-style):

```
+----[SHA256]-----+
|          + o.o.o|
|         . = . ==|
|            . +oo|
|         .   .. .|
|        S . .  + |
|         +o.  . .|
|        o=*.. oo |
|        oB*O.o.o+|
|        .B@E= .oo|
+-----------------+
```

- **Payload hash actually covered by the current `.ots`:** see `ots info NEXT_CAMPAIGN_LANES.md.ots`; by construction this is `SHA-256(this entire file including this footer)` and `ots verify NEXT_CAMPAIGN_LANES.md.ots NEXT_CAMPAIGN_LANES.md` succeeds against the on-disk bytes.
- Anchor file: `NEXT_CAMPAIGN_LANES.md.ots`
- Calendars: a.pool.opentimestamps.org, b.pool.opentimestamps.org, a.pool.eternitywall.com, ots.btc.catallaxy.com
- **Provenance posture (load-bearing):** the **OTS proof timestamp** above is the legal anchoring time for the visible document — that is what the calendars witnessed. The **disclosed pre-footer content hash** is *not* anchored by the current `.ots` file; it is *disclosed inside* the OTS-anchored visible document as a human-readable historical record of what the file's bytes hashed to immediately before this footer was appended. A reviewer verifying the visible file runs `ots verify` against the on-disk bytes; a reviewer wanting to confirm the disclosed pre-footer hash recomputes `SHA-256` of the file with everything from `<!-- ots-attestation-footer -->` onward stripped. Both checks are independent; neither replaces the other.
