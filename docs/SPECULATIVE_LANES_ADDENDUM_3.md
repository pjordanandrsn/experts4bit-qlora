# ADDENDUM 3 — SOURCE-READ RESOLUTIONS (repo zip, 2026-07-05)

```
status:   resolution/correction addendum from static reads of experts4bit_qlora/
          {train.py, offload.py, _vendor/experts.py} in main_11.zip. Two entries
          CORRECT the reviewer's own prior analysis turn; graded in §4.
cites:    AGENT_HANDOFF_POST_AUDIT_20260705.md (T1.0, T4, D3 tree)
          SPECULATIVE_LANES_PLAN_20260705.md + Addenda 1–2
          n=64 D1/D2 report (bitwise eval determinism)
clock:    none urgent; batch with the standing commit window. LAYOUT_FACTS (§3)
          should land before any routed-stream Phase 0 code is written.
```

## 1. T1.0 — static path diff: PARTIALLY COMPLETE (one candidate killed, one residue)

**Resolved from source:** the trainer always enables gradient checkpointing
(`use_reentrant=False`) for BOTH placements (train.py:181; offload.py:29 states the
offload invariant). The "offload forces checkpointing, resident doesn't" candidate is
**eliminated** — the paths are checkpointing-symmetric.

**NOT resolved (correcting the prior chat turn):** this does not confirm an RNG
mechanism. `use_reentrant=False` defaults to `preserve_rng_state=True`, so recompute
dropout masks nominally match the original forward in both placements. Sub-mechanism
redistributed, modal "environment-not-math" branch held at ~75–80%:
RNG/dropout divergence via some unpreserved generator · kernel/numeric path
differences · stream-ordering effects in backward accumulation.

**Residue task T1.0b (zero GPU):** confirm in the installed transformers version that
the non-reentrant checkpoint path actually passes/honors `preserve_rng_state=True`
end-to-end, and enumerate any generator the staging path touches. D3's design is
unchanged — the dropout-OFF/ON contrast discriminates regardless of sub-mechanism.

## 2. T4 — workspace hypothesis: REFUTED; corrected mechanism hypothesis + T4b

**Refuted from source (offload.py:25–32):** checkpointing saves only the packed
buffers, never the full dequantized expert; the dequant is a transient inside each
layer's backward segment. **No persistent bf16 dequant workspace exists.** The
offload training-memory win is architectural, not an optimization left on the table.

**Corrected mechanism (hypothesis, pending T4b):** offload peaks = a common
per-layer dequant/compute transient (bf16-sized; passthrough pays only this) + a
**packed-bytes adder** for quantized formats (staged layer's packed weights on GPU).
First-order arithmetic fits the measured ordering: staged-layer packed ≈ 0.21 GB
(nf4) / 0.42 GB (int8) vs measured deltas over bf16 of 0.11 / 0.31 GB — direction
and rough magnitude right, residuals unexplained (double-buffering, absmax
granularity are candidates).

**T4b (zero GPU):** derive expected per-mode offload peaks from tensor shapes under
the current scheme; compare to the six measured peaks; only then name the mechanism
in OFFLOAD_MEMORY_FACTS. **P-C1 (committed):** the shape model reproduces the peak
ordering with per-mode residual < 0.10 GB.

**Optimization note (replaces the retracted 1.8 GB line):** the remaining lever for
quantized modes is chunking the backward-segment dequant transient; direction
committed, no number until T4b sizes the transient.

## 3. LAYOUT_FACTS — combine order (source fact + measurement precedence)

- Source fact (_vendor/experts.py:486–513): resident combine is a Python loop over
  `expert_hit` with `final_hidden_states.index_add_(0, token_idx, …)` — contribution
  order follows expert-hit order; `index_add_` is nominally non-deterministic on GPU.
- Measurement fact (D2 report): eval path repeated **bitwise** 64/64 on the A5000
  stack — on this host/dtypes/shapes the nominal non-determinism did not manifest.
- Rule for routed-stream (v3 §3.1): implementation must match the MEASURED behavior —
  run the bitwise repeat on the target host first; require bitwise match if it holds
  there, within-measured-null otherwise. Training-path determinism stays UNKNOWN
  pending D3 (backward adds its own accumulation atomics).

## 4. Ledger (graded)

- **RETRACTED (wrong):** "workspace removal → nf4-offload ≈ 1.8 GB." The mechanism
  it assumed does not exist (§2).
- **CORRECTED (overreach):** prior turn's "~85%, RNG mechanism named in a docstring."
  Actual content of the read: one candidate eliminated, none confirmed (§1).
- **CORRECTED (inference vs measurement):** "atomics → non-bitwise" deferred to D2's
  measured bitwise repeat (§3). R9 cuts both ways: source reads are facts about code,
  not about runtime behavior already measured otherwise.
- **STANDS:** the bf16-offload memory inversion and its user-facing claim ("full-
  precision training at the offload floor"); only its mechanism sentence changes.

## 5. Small orders

- Cross-link `audits/unsloth-zoo-4032/REPORT.md` from README/results_summary — it is
  currently reachable only by directory browsing and is the portfolio's strongest
  single artifact. Watch zoo #849/#850 for maintainer response; reply SLA = same-day.
- Serialization check: this session's planning layer (n=1024 amendment, Addenda 1–3,
  spec-lanes plan, mode-decoupled order sheet) is not in main_11.zip — confirm all
  land in the next commit window, hash-ordered per the standing rule.

## 6. What will not be claimed

- No D3 sub-mechanism claim until the certificate runs (T1.0b is enumeration, not
  attribution).
- No memory-mechanism sentence in any doc until T4b's residuals are in hand.
- No determinism claim for any host other than the one measured.

---

## Executor's filing notes (added at commit time; not part of the handoff text)

**Staleness reconciliation — this addendum was authored from `main_11.zip`, which
predates today's session. Three entries were overtaken by measurement before filing:**

1. **§1's "modal environment-not-math branch ~75–80%" and §6's "until the certificate
   runs": the certificate RAN today** (rev2, five trios): null AND placement bitwise
   on every object, dropout OFF and ON, default and deterministic kernels
   (`docs/TRAIN_PLACEMENT_CERTIFICATE.md`, 3d7a480). At one-step granularity NO
   divergence mechanism exists; the anomaly is run-level, and for the repeat grid it
   is the T5(c) architecture confound. Also: T1.0's committed static diff (8792b2f)
   found **no dropout exists anywhere on this training path** (no LoRA dropout;
   OLMoE `attention_dropout` = 0.0) — the RNG/dropout sub-mechanism has no surface
   even nominally. An offload-engagement attestation (rev3) is pending as a vacuity
   guard; the row-4 filing stands on eval-path evidence meanwhile.
2. **§3's "training-path determinism stays UNKNOWN pending D3"**: now PARTIALLY
   measured — one training step is bitwise-repeatable (null a≡b) on this host.
   Full-run (150-step) determinism remains unknown (the divergence-onset probe is
   the gated resolver). LAYOUT_FACTS (`docs/LAYOUT_FACTS.md`) records both.
3. **§2's packed-bytes-adder hypothesis**: T4b executed —
   `docs/OFFLOAD_MEMORY_FACTS.md` (T4b section). Stated from shapes, the
   packed-adder model and the fixed+slab+transient model coincide and underpredict
   quantized train peaks by ~+0.63 GB; a full-layer-dequant variant overpredicts by
   a constant ~+0.17 GB. **P-C1 FAILS as committed** (no shape model reaches
   per-mode residual < 0.10 GB). The n=64 eval-job peaks cannot disambiguate
   (load-phase transients pollute them; the eval script did not reset after load).
   Per §6, no mechanism sentence ships; the floor headline stands unchanged; the
   one-step memory-timeline probe remains the resolver.

- In-repo mapping: cites as in Addenda 1–2. `main_11.zip` corresponds to a pre-fcd65cf
  state; all of today's planning docs (re-pin amendment 94c3931, lanes ba23461,
  Addenda 1/2 @ 1271f96/149fbbc, review orders 6827896) are committed and pushed —
  §5's serialization check PASSES.
- T1.0b executed: `docs/LAYOUT_FACTS.md` carries the checkpoint/RNG enumeration.

---

<!-- ots-attestation-footer -->

**OpenTimestamps anchor (self-attestation footer):**

- **OTS proof timestamp for visible document:** `2026-07-05T20:03:52Z` (the moment the current `.ots` was submitted to the calendars; this is the legally operative timestamp for the visible file as published).
- **Disclosed pre-footer content hash:** `c48962214696975546f9a75cb13514d086d1b4b677f3e67d29948e89785250ce` (the SHA-256 of the document *before* this footer was appended — disclosed inside the OTS-anchored visible document for human-readable historical reference; this hash is *not* the payload of the current `.ots` file).
- integrity-attestor glyph (`core.fingerprint`, first 8 bytes of the disclosed pre-footer hash): `[&o*#0++:o0#0#=OO]`
- Drunken-bishop randomart (full disclosed pre-footer SHA-256, OpenSSH-style):

```
+----[SHA256]-----+
| .=..o.o+.  oO+. |
| o..o. +.o  o *. |
|   .o . B.   =o. |
|   . . o E. +. . |
|        S. +  o.o|
|         .o  o .+|
|        o . =   +|
|       o o o o +o|
|        o     . +|
+-----------------+
```

- **Payload hash actually covered by the current `.ots`:** see `ots info SPECULATIVE_LANES_ADDENDUM_3.md.ots`; by construction this is `SHA-256(this entire file including this footer)` and `ots verify SPECULATIVE_LANES_ADDENDUM_3.md.ots SPECULATIVE_LANES_ADDENDUM_3.md` succeeds against the on-disk bytes.
- Anchor file: `SPECULATIVE_LANES_ADDENDUM_3.md.ots`
- Calendars: a.pool.opentimestamps.org, b.pool.opentimestamps.org, a.pool.eternitywall.com, ots.btc.catallaxy.com
- **Provenance posture (load-bearing):** the **OTS proof timestamp** above is the legal anchoring time for the visible document — that is what the calendars witnessed. The **disclosed pre-footer content hash** is *not* anchored by the current `.ots` file; it is *disclosed inside* the OTS-anchored visible document as a human-readable historical record of what the file's bytes hashed to immediately before this footer was appended. A reviewer verifying the visible file runs `ots verify` against the on-disk bytes; a reviewer wanting to confirm the disclosed pre-footer hash recomputes `SHA-256` of the file with everything from `<!-- ots-attestation-footer -->` onward stripped. Both checks are independent; neither replaces the other.
