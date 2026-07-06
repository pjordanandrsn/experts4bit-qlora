# N1 — ROUTING-PINNED SERVE (preregistration)

```
status:   preregistration — committed BEFORE any GPU run (R5). Opens the next campaign's
          first lane, licensed by the S-B graduation clause (adapters steer routing;
          shift tracks gain, ρ 0.58–0.60).
cites:    NEXT_CAMPAIGN_LANES.md N1 · SPECULATIVE_LANES_PLAN §S-B · sb_routing_shift_1024.md
          n=1024 re-pin (certified G_int8 = 0.01657 ± 0.00227) · MODE_DECOUPLED_ADAPTERS
host:     one A5000, resident, per the T5c single-architecture discipline. ~2–4 eval passes.
```

## Question

The mode-decoupled **upgrade forfeit** — an nf4-trained adapter served on a *better* (int8)
base fails to capture the base's G = 0.0166 improvement (portability matrix: nf4→int8 ≈
nf4→nf4) — is, per S-B, partly a **routing-mismatch** effect: adaptation moves routing, and a
different serve-base routes differently, so the adapter's per-expert corrections land on
different experts than it trained against. Does **pinning the served model's routing to the
train-base reference** recover the forfeited gain?

## Article & conditions (nf4-trained seed-0 adapter, pinned n=1024 eval set, all resident)

- **C0 — home** (reference): nf4 adapter on **nf4** base, standard routing. Loss L0. Also the
  source of the per-(example, layer, position) top-k routing indices that C2 replays.
- **C1 — unpinned upgrade**: nf4 adapter on **int8** base, standard routing. Loss L1
  (the forfeit — expected ≈ L0, little/no gain from the better base).
- **C2 — routing-pinned upgrade**: nf4 adapter on **int8** base, but every MoE layer's top-k
  routing **forced to C0's captured indices** for that token (eval is teacher-forced, so the
  token sequence is identical across passes — routing replays by position). Loss L2.
- Determinism: eval forward is bitwise-repeatable on this host (D2), so a single pass per
  condition is a measurement, not a sample.

## Primary metric & success (pre-committed)

Fraction of the frozen serve-upgrade gain the pinned adapter recovers:
`R = (L1 − L2) / G`, G = 0.01657 (certified, n=1024).
- **Success = R ≥ 0.50** (pinning recovers ≥ half of G that the unpinned upgrade forfeits),
  with L2 < L1 by more than the eval repeat-null (bitwise, so any L1−L2 > 0 is real).
- Report R with G's SE propagated. Secondary: L2 vs the int8-trained→int8 native floor
  (does pinning close the gap to a natively-int8 adapter?), and the per-layer routing-override
  count actually applied (attestation the pin engaged).

## Committed odds (graded like everything else)

- 30% — **R ≥ 0.50**: routing mismatch was the dominant forfeit term; pinning is a real lever.
- 45% — **0 < R < 0.50**: routing mismatch is *a* term but not the whole forfeit (the adapter
  also co-adapted to nf4's *values*, which pinning can't fix — served int8 experts differ in
  magnitude, not just identity).
- 25% — **R ≤ 0** (pinning neutral or *hurts*): int8 experts "want" their own routing;
  forcing nf4's choices onto more-accurate experts degrades. Forfeit is a value-space effect,
  routing is a bystander. (Would also close S-B's serving relevance — a clean negative.)

## What will not be claimed

- No claim beyond this adapter/model/host/eval-set (one measured article; a positive result
  preregisters a seed-replicated follow-up before shipping).
- No G-denominated headline unless L1−L2 exceeds the (bitwise) repeat-null — it does by
  construction if nonzero, but R's CI carries G's SE.
- Nothing about training-time routing; N1 is a serve-side intervention only.
- No serving-framework claim — this is a provenance/validation measurement, per the repo's
  standing scope.

## RESULT (2026-07-06, A5000, `runs/results/postaudit/n1_routing_pinned_result.json`)

**PARTIAL — the committed middle branch (45%): routing mismatch is a real forfeit term, but
not the majority.** nf4-trained seed-0 adapter, pinned n=1024 set, resident; pin engaged fully
(16384/16384 layer×example overrides).

| condition | eval loss |
|---|---|
| C0 home (nf4 base + adapter) | 1.22486 |
| C1 unpinned upgrade (int8 base + adapter) | 1.22169 |
| C2 routing-pinned upgrade (int8 base, nf4-home routing) | 1.21565 |

**R = (L1 − L2) / G = 0.365 ± 0.092** (|t| ≈ 4.0 — recovery is real; the point estimate is
below the 0.50 success bar, CI ~[0.18, 0.55]). Pinning the int8-served model's routing to the
nf4 home reference recovers **~37% of the certified precision gap** — a significant lever, but
the majority of the forfeit survives pinning, i.e. it is value-space co-adaptation (the adapter
learned nf4's specific quantization errors; served int8 experts differ in *magnitude*, which
pinning routing cannot correct) rather than routing identity. This confirms S-B's serving
relevance (adapter-steered routing matters at serve time) while bounding it.

Honest caveats: (1) on this seed-0 adapter the *absolute* forfeit L1−L0 = −0.0032 was small
(the unpinned int8 serve was already ≈ nf4-home), so R measures pinning *recovery* against G,
not recovery of a large observed forfeit; the lane's "~100% forfeit" framing came from 3-seed
matrix means, not this single article. (2) One adapter, one host, one eval set — per the
no-claims list, this positive-but-partial result licenses a **seed-replicated follow-up**
before any serving claim ships; it does not itself ship one. Committed odds graded:
middle branch (0 < R < 0.50, 45%) **HIT**; R ≥ 0.50 (30%) and R ≤ 0 (25%) did not.

## Mechanism (how the pin is built)

A forward pre-hook on each `ExpertsLoRA`, active only in pinned mode: it overrides the
`top_k_index` argument (and matching `top_k_weights` gather) with the reference indices
captured from the C0 pass, keyed by (example, layer, token-position). C0 runs first with a
capture hook; C2 replays. No weight or router modification — only the routed-expert selection
is substituted, so the served int8 experts + the trained adapter run on the home routing.
Script: `scripts/routing_pinned_serve.py` (reuses the eval + telemetry scaffolding).

---

<!-- ots-attestation-footer -->

**OpenTimestamps anchor (self-attestation footer):**

- **OTS proof timestamp for visible document:** `2026-07-06T02:01:18Z` (the moment the current `.ots` was submitted to the calendars; this is the legally operative timestamp for the visible file as published).
- **Disclosed pre-footer content hash:** `0c8a62a865f73c37e0574569b42c92b9188afff432ce9988ea0277f93df202bc` (the SHA-256 of the document *before* this footer was appended — disclosed inside the OTS-anchored visible document for human-readable historical reference; this hash is *not* the payload of the current `.ots` file).
- **Prior disclosed pre-footer hashes (chain, newest first):**
  - `2026-07-06T00:52:06Z` `ea42c4880c89d6e6be1746d6c5d80cac4ff18aa72894589e51b893c1813ab787`
- integrity-attestor glyph (`core.fingerprint`, first 8 bytes of the disclosed pre-footer hash): `[.&*%0+%*0O$=~&~=]`
- Drunken-bishop randomart (full disclosed pre-footer SHA-256, OpenSSH-style):

```
+----[SHA256]-----+
|           .o.   |
|         o oo.   |
|      o + ..+    |
|.  o o = o o     |
|o.=.+.o S .      |
|++.o=+ . .       |
|o. ..+*.+        |
|.  .E=*Bo.       |
|+o. ..**o.       |
+-----------------+
```

- **Payload hash actually covered by the current `.ots`:** see `ots info N1_ROUTING_PINNED_SERVE.md.ots`; by construction this is `SHA-256(this entire file including this footer)` and `ots verify N1_ROUTING_PINNED_SERVE.md.ots N1_ROUTING_PINNED_SERVE.md` succeeds against the on-disk bytes.
- Anchor file: `N1_ROUTING_PINNED_SERVE.md.ots`
- Calendars: a.pool.opentimestamps.org, b.pool.opentimestamps.org, a.pool.eternitywall.com, ots.btc.catallaxy.com
- **Provenance posture (load-bearing):** the **OTS proof timestamp** above is the legal anchoring time for the visible document — that is what the calendars witnessed. The **disclosed pre-footer content hash** is *not* anchored by the current `.ots` file; it is *disclosed inside* the OTS-anchored visible document as a human-readable historical record of what the file's bytes hashed to immediately before this footer was appended. A reviewer verifying the visible file runs `ots verify` against the on-disk bytes; a reviewer wanting to confirm the disclosed pre-footer hash recomputes `SHA-256` of the file with everything from `<!-- ots-attestation-footer -->` onward stripped. Both checks are independent; neither replaces the other.
