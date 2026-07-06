# DIVERGENCE-ONSET PROBE — GO (final gate of the campaign)

```
status:   go decision + preregistration for the one open item. NON-BLOCKING:
          submission does not wait on this. Smallest document in the series,
          by design — it closes the queue.
cites:    final results picture (this date) · D3 certificate (bitwise, all objects)
          T5 forensics (cross-host evaluator offset 0.003–0.005) · Addendum 3 §3
target:   the same-host 150-step bf16 resident-vs-offload gap of 0.0108 —
          the only number in the campaign without an attributed mechanism.
```

## Sequencing (T5 earned first position)

1. **Static first (zero GPU):** full provenance/config diff of the two 150-step
   runs — commit, eval cadence, checkpoint-selection rule, batch hashes, dataloader
   workers, wall-clock interleaving. One harness confound already lived in this
   grid; rule that class out before spending.
2. **Twin probe (only if the diff is clean; ~30–60 min A5000):** paired twins from
   identical state, per-step adapter-weight divergence + per-step routed-set flip
   counts (the flip-annotated divergence curve). Log first-divergence step, first
   routing-flip step, and growth shape after onset.

## Committed odds (graded like everything else)

- 30% — resolved statically: a config/harness difference explains the gap; no GPU spent.
- 55% — nondeterministic-accumulation onset: bitwise for some steps (D3 says ≥1),
  then a first-divergence event (atomics-class), then amplification; routing flips
  follow onset rather than cause it.
- 15% — other (including: gap fails to reproduce on rerun).

## Interpretation guards

- D3's bitwise result at one step is a boundary condition any mechanism must respect.
- A reproduced gap with clean provenance is a *chaos measurement*, not a placement
  effect — the placement question is closed and stays closed.
- Whatever the branch, output is one paragraph in the results doc + a graded line in
  PREDICTION_LEDGER.md. No new lanes open from this item.

## RESULT (2026-07-06, A5000, `runs/results/postaudit/divergence_bf16_result.json`)

**Branch 2 confirmed, in its strongest form: the divergence is run-to-run nondeterminism;
placement is incidental.** Three legs from byte-identical state (bf16, seed 0, 60 steps):
resident-twin A, resident-twin B, offload C. Subset-sketch weight divergence + routing flips
vs twin A:

| step | wdiv A–B | wdiv A–C | flip A–B | flip A–C |
|---|---|---|---|---|
| 1 | 0 | 0 | 0 | 0 |
| 2 | 1.5e-4 | 1.4e-4 | 0 | 0 |
| 3 | 3.6e-3 | 3.8e-3 | 7 | 6 |
| 10 | 3.96e-2 | 3.59e-2 | 9 | 6 |
| 30 | 1.36e-1 | 1.22e-1 | 9 | 10 |
| 60 | 2.28e-1 | 2.03e-1 | 24 | 18 |

- **Step 1 bitwise across all three legs** — the D3 boundary is respected exactly.
- **First divergence at step 2, identical for the resident twin (A–B) and placement (A–C).**
  The offload pair diverges no earlier than an identical resident rerun does.
- **First routing flip at step 3, both** — flips FOLLOW weight divergence, they do not cause
  it (predicted).
- **A–C tracks A–B the whole way and is marginally SMALLER** (final ratio A–C/A–B = 0.89;
  mean flips/step 13.3 vs 14.4). Placement adds nothing beyond the chaos an identical rerun
  already produces.

**Attribution:** the same-host 150-step bf16 0.0108 best-eval gap is a **chaos measurement** —
the training process is nondeterministic (fp32 `index_add_` atomics in the MoE combine + SDPA
backward, per T1.0/LAYOUT_FACTS), so any two runs from identical state diverge and amplify over
150 steps; best-of-3-evals then samples different points of two chaotic trajectories. Offload
is numerically innocent (D3: bitwise at one step; here: A–C ≤ A–B across 60). **The placement
question is closed and stays closed** — this is not a placement effect. Committed odds graded:
branch 2 (55%) hit; branches 1 (30%, static — ruled out clean) and 3 (15%, other) did not.

## Queue state after this document

Empty. The campaign's last unattributed number now has a mechanism. New questions belong to
the job or to the next campaign.

---

## Executor's filing notes (added at commit time; not part of the handoff text)

- Committed BEFORE the static diff runs (R5). The bf16 pair under scrutiny is the
  single-run *grid* legs: `runs/jobs/`-adjacent grid outputs (`mode-ab-out/bf16` and
  `.../bf16-offload` on the volume) reporting resident best-eval 1.0220 vs offload
  1.0112 — Δ 0.0108, `docs/OLMOE_EXPERTSNBIT_GRID.md`.
- Step-2 twin probe reuses the D3 harness (`scripts/one_step_certificate.py` already
  logs per-object hashes + engagement); the divergence-curve variant runs N steps
  logging per-step adapter-weight L2 divergence and routed-set flips. Only spun if
  step-1 is clean AND explicitly instructed (the pod is currently down).

---

<!-- ots-attestation-footer -->

**OpenTimestamps anchor (self-attestation footer):**

- **OTS proof timestamp for visible document:** `2026-07-06T00:18:55Z` (the moment the current `.ots` was submitted to the calendars; this is the legally operative timestamp for the visible file as published).
- **Disclosed pre-footer content hash:** `18a88feb1357ccc94731c8e4c8daa054d9e46d67ef9bf05cb27fecfa5ac1b20a` (the SHA-256 of the document *before* this footer was appended — disclosed inside the OTS-anchored visible document for human-readable historical reference; this hash is *not* the payload of the current `.ots` file).
- **Prior disclosed pre-footer hashes (chain, newest first):**
  - `2026-07-05T22:57:37Z` `b856ab52a7094321eb9bc588e59b73223ece6843dde42c1a83a150c5a257d096`
- integrity-attestor glyph (`core.fingerprint`, first 8 bytes of the disclosed pre-footer hash): `[:*%**$?@:~O=&&&#]`
- Drunken-bishop randomart (full disclosed pre-footer SHA-256, OpenSSH-style):

```
+----[SHA256]-----+
|   .++..o.       |
|  .o.=+ ..       |
| .. +=o= o       |
|.. =  Bo+ .   .  |
|. o ....S  . . o |
|  .o.     .   o .|
|  .o.    E o o.. |
|  ..      = B .o |
| .o.       B.+*o |
+-----------------+
```

- **Payload hash actually covered by the current `.ots`:** see `ots info DIVERGENCE_ONSET_PROBE.md.ots`; by construction this is `SHA-256(this entire file including this footer)` and `ots verify DIVERGENCE_ONSET_PROBE.md.ots DIVERGENCE_ONSET_PROBE.md` succeeds against the on-disk bytes.
- Anchor file: `DIVERGENCE_ONSET_PROBE.md.ots`
- Calendars: a.pool.opentimestamps.org, b.pool.opentimestamps.org, a.pool.eternitywall.com, ots.btc.catallaxy.com
- **Provenance posture (load-bearing):** the **OTS proof timestamp** above is the legal anchoring time for the visible document — that is what the calendars witnessed. The **disclosed pre-footer content hash** is *not* anchored by the current `.ots` file; it is *disclosed inside* the OTS-anchored visible document as a human-readable historical record of what the file's bytes hashed to immediately before this footer was appended. A reviewer verifying the visible file runs `ots verify` against the on-disk bytes; a reviewer wanting to confirm the disclosed pre-footer hash recomputes `SHA-256` of the file with everything from `<!-- ots-attestation-footer -->` onward stripped. Both checks are independent; neither replaces the other.
