# Expert-streaming profile

Offload answers whether the model can fit. Expert-streaming profiling asks whether the transfer
wall is concentrated enough to route around. Blunt rule: measure the wall before building the
door through it.

## 1. Purpose

Determine whether the offload slowdown is concentrated in a small number of layer/expert pairs.
If it is, a later pass can test static hot-expert pinning to recover speed while keeping most of
the offload VRAM benefit. If it is not, we document that the offload wall is diffuse for this
model/path and do not build a cache. This is a separate experiment family from the OLMoE
repeat-validation grid and the portability matrix; all outputs live under
`runs/expert_streaming/`.

## 2. Why offload is not enough

On the OLMoE grid (`docs/OLMOE_EXPERTSNBIT_GRID.md`), int8-offload holds a near-4-bit memory
floor (2.72 vs 2.52 GB) far below int8 resident (8.50 GB) but pays for it in step time (18.5
vs 12.1 s/step). The question this profile asks: can that memory advantage be kept while
identifying which expert transfers cause most of the speed penalty — the precondition for any
targeted pinning.

## 3. Profile-only methodology

No behavior change: no cache, no scheduler, no prefetcher, no residency-policy change. The
profiler (`experts4bit_qlora/expert_profile.py`, gated on `E4B_EXPERT_PROFILE=<out.jsonl>`,
attached by `train.py`/`infer.py` after the model is built) records two things and changes
nothing:

- **per-layer H2D staging** — count, bytes, and wall of each staging copy, via CUDA events
  recorded on the staging stream and reduced with a *single* `synchronize` at flush (never in the
  hot path — the same discipline as the offload stats path, so the profiler does not destroy what
  it measures), tagged sync / cold_miss / prefetch.
- **per-(layer, expert) routing** — `hits` (forwards that routed to the expert) and
  `tokens_routed`, as on-device bincounts in a forward pre-hook (one bincount per layer forward,
  no host sync until flush).

**Load-bearing caveat: staging is layer-granular.** `_ExpertOffload` moves a layer's *entire*
fused expert stack per visit; there is no per-expert transfer in the current design. So measured
H2D stall is per layer. Per-expert stall is a **projection** (the summarizer shares a layer's
stall among its routed experts, weighted by `tokens_routed`) — that projection is the number a
per-expert pinning policy would have to beat, not a measurement of isolated per-expert transfer.

Sampling: 100 training steps for the train profiles; 5 decode samples + 1 discarded warmup for
the decode profiles.

## 4. OLMoE int8-offload / nf4-offload traces

Jobs (`scripts/make_expert_streaming_manifest.py`), run under the distributed harness
(`docs/RUNPOD_DISTRIBUTED_VALIDATION.md`), training first:

- `profile_olmoe_int8_offload_train_seed1337_steps100`  ← best first target
- `profile_olmoe_nf4_offload_train_seed1337_steps100`
- `profile_olmoe_int8_offload_decode_repeat5`
- `profile_olmoe_nf4_offload_decode_repeat5`

*Traces filled from `scripts/summarize_expert_streaming.py` output when the profile jobs land.*

## 5. Hotness / stall concentration

*Filled from the summary: top layer/expert pairs by hits, tokens, and projected stall; and the
concentration table (share of total held by the top 1% / 5% / 10% / 20% of pairs).*

## 6. Estimated pinning budgets

*Filled from the summary: projected stall coverage at cache budgets 0.25 / 0.5 / 1.0 / 2.0 GB,
greedy by projected-stall-per-byte. A projection, not a measured speedup.*

## 7. Decision: build hot-static or not

Pre-registered criterion — build hot-static **only if** the top 10% of (layer, expert) pairs
account for ≥40% of projected stall, or the top 20% account for ≥60%. If stalls are diffuse,
do not build a cache; record that the offload wall is not hot-expert concentrated for this
model/path.

If (and only if) the criterion is met, one validation run follows —
`olmoe_int8_offload_hotstatic_seed1337_steps100_budget1gb` — compared to all-offload on peak GPU,
s/step, before/after eval, and stall reduction. Success = added peak GPU ≤ 1 GB, a measurable
s/step or stall reduction, and no eval/path regression beyond training variance. The policy, if
built, is `hot-static` only: no LRU, no router-aware prefetch, no async scheduler, no learned
policy in this family.

*Decision filled from §5 when the traces land.*

## 8. What is not claimed

- No measurement of isolated per-expert transfer (staging is layer-granular; §3).
- No speedup — pinning coverage is a projection until a hot-static validation run measures it.
- No universal claim: concentration (or its absence) is observed for this model/path/host.

## 9. Next steps

Qwen3-30B-A3B gets a later *sentinel* profile only if OLMoE shows concentrated stalls and the
policy looks worth scaling — not before, and not on Qwen3 first.

---

<!-- ots-attestation-footer -->

**OpenTimestamps anchor (self-attestation footer):**

- **OTS proof timestamp for visible document:** `2026-07-05T09:33:07Z` (the moment the current `.ots` was submitted to the calendars; this is the legally operative timestamp for the visible file as published).
- **Disclosed pre-footer content hash:** `e3990334d90b819081d12ce9a613048aef1bae9633dd22933d98ba9dc6513d8c` (the SHA-256 of the document *before* this footer was appended — disclosed inside the OTS-anchored visible document for human-readable historical reference; this hash is *not* the payload of the current `.ots` file).
- integrity-attestor glyph (`core.fingerprint`, first 8 bytes of the disclosed pre-footer hash): `[?~##.~~o!#.@*:#.]`
- Drunken-bishop randomart (full disclosed pre-footer SHA-256, OpenSSH-style):

```
+----[SHA256]-----+
|+*o+ ..          |
|*.+ .  +         |
|=.   += .        |
|.+  E.+o .       |
|o...  ..S        |
|o..    o +       |
| oX..   =        |
| &oO .   .       |
|*+X o            |
+-----------------+
```

- **Payload hash actually covered by the current `.ots`:** see `ots info EXPERT_STREAMING_PROFILE.md.ots`; by construction this is `SHA-256(this entire file including this footer)` and `ots verify EXPERT_STREAMING_PROFILE.md.ots EXPERT_STREAMING_PROFILE.md` succeeds against the on-disk bytes.
- Anchor file: `EXPERT_STREAMING_PROFILE.md.ots`
- Calendars: a.pool.opentimestamps.org, b.pool.opentimestamps.org, a.pool.eternitywall.com, ots.btc.catallaxy.com
- **Provenance posture (load-bearing):** the **OTS proof timestamp** above is the legal anchoring time for the visible document — that is what the calendars witnessed. The **disclosed pre-footer content hash** is *not* anchored by the current `.ots` file; it is *disclosed inside* the OTS-anchored visible document as a human-readable historical record of what the file's bytes hashed to immediately before this footer was appended. A reviewer verifying the visible file runs `ots verify` against the on-disk bytes; a reviewer wanting to confirm the disclosed pre-footer hash recomputes `SHA-256` of the file with everything from `<!-- ots-attestation-footer -->` onward stripped. Both checks are independent; neither replaces the other.
