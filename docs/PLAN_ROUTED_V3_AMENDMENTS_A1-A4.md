# Amendments A1–A4 to plan-routed-v3

```
status:  amendment/prediction document — commits BEFORE any routed-stream execution
amends:  plan-routed-v3 (ROUTED_STREAMING_AGENT_EXECUTION_PLAN.md) — cited, NOT edited
source:  POST_AUDIT_WORK_QUEUE.md §4 (2026-07-05 handoff), which is authoritative
         for the amendment text; this file is the standalone citable copy.
note:    plan-routed-v3 is not present in this repository at the time of this
         commit; the amendments bind against it by name. When v3 lands in-repo,
         these amendments attach to it unchanged.
```

## A1 — 16-bit routed-stream lane

Add bf16/fp16 to the routed-stream lane's precision set.

- Expert spans ≈ 4× nf4 (~10.6 MB/expert on Qwen3-30B-A3B).
- NO dequant workspace, NO dequant kernel — the simplest gather path (matches the
  §4 audit mechanism verified in `docs/OFFLOAD_MEMORY_FACTS.md`).
- Host pinned-memory requirement ≈ 61 GB for Qwen3 — run the S4 host-RAM check
  before scheduling this lane on any pod.
- Motivation: the ∅ ladder says serve ≥ int8 for quality; the audit says 16-bit is
  the offload memory floor; A2's economics say 16-bit is where a cache pays most.

## A2 — Per-precision cache economics as G1 priors

Reactive→ceiling tok/s gap = t_fetch/T_ovh:

| precision | ceiling gap | gain over reactive at h=0.5 |
|---|---|---|
| nf4  | +20% | +9%  |
| int8 | +38% | +16% |
| bf16 | +79% (T_ovh(nf4) placeholder until measured) | +28% |

Gain over reactive at hit rate h: `(T_ovh + t_fetch) / (T_ovh + (1−h)·t_fetch) − 1`.

All values pending Phase 0.2b's BW_gather measurement. Expected consequence: the
<10% kill rule fires on the nf4 lane and spares int8/16-bit — kill per precision,
as v3 already allows.

## A3 — T_ovh measurement protocol (supersedes v3 §0.3 acquisition)

= T6 in the work queue. Before any G1 commit: resident decode T_ovh measured
back-to-back, same session, ≥5 reps, median ± spread, per (model, precision).
Formula unchanged from v3 §0.3; only the acquisition discipline is upgraded.
Forced by the Qwen3 4.14 vs 5.19 tok/s same-mode cross-session finding (±13%
session noise exceeds the ±15% PvM bound's headroom).

## A4 — Decode-lane isolation note

The D3 anomaly mechanism (dropout/RNG/recompute) has no surface at decode: no
dropout, no backward, no recompute. The routed-stream lane proceeds in parallel
with T1 (certificate D3) and neither gates the other. Serve-side certificates
remain the deterministic strong family.

---

<!-- ots-attestation-footer -->

**OpenTimestamps anchor (self-attestation footer):**

- **OTS proof timestamp for visible document:** `2026-07-05T16:58:45Z` (the moment the current `.ots` was submitted to the calendars; this is the legally operative timestamp for the visible file as published).
- **Disclosed pre-footer content hash:** `29bad2005639a15429f336c93dcc17ff631819727ef8a0ba771526114a425123` (the SHA-256 of the document *before* this footer was appended — disclosed inside the OTS-anchored visible document for human-readable historical reference; this hash is *not* the payload of the current `.ots` file).
- integrity-attestor glyph (`core.fingerprint`, first 8 bytes of the disclosed pre-footer hash): `[+#@%!+..O0~#%:Oo]`
- Drunken-bishop randomart (full disclosed pre-footer SHA-256, OpenSSH-style):

```
+----[SHA256]-----+
| ..o=Eo+ ..      |
|.o.=  +oo+       |
| .* *  .* =      |
|.. * = . X +     |
|... . + S X .    |
|  .  . o . *     |
|   o. .   o .    |
|  . .o . .       |
|   ...o .        |
+-----------------+
```

- **Payload hash actually covered by the current `.ots`:** see `ots info PLAN_ROUTED_V3_AMENDMENTS_A1-A4.md.ots`; by construction this is `SHA-256(this entire file including this footer)` and `ots verify PLAN_ROUTED_V3_AMENDMENTS_A1-A4.md.ots PLAN_ROUTED_V3_AMENDMENTS_A1-A4.md` succeeds against the on-disk bytes.
- Anchor file: `PLAN_ROUTED_V3_AMENDMENTS_A1-A4.md.ots`
- Calendars: a.pool.opentimestamps.org, b.pool.opentimestamps.org, a.pool.eternitywall.com, ots.btc.catallaxy.com
- **Provenance posture (load-bearing):** the **OTS proof timestamp** above is the legal anchoring time for the visible document — that is what the calendars witnessed. The **disclosed pre-footer content hash** is *not* anchored by the current `.ots` file; it is *disclosed inside* the OTS-anchored visible document as a human-readable historical record of what the file's bytes hashed to immediately before this footer was appended. A reviewer verifying the visible file runs `ots verify` against the on-disk bytes; a reviewer wanting to confirm the disclosed pre-footer hash recomputes `SHA-256` of the file with everything from `<!-- ots-attestation-footer -->` onward stripped. Both checks are independent; neither replaces the other.
