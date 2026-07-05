# Expert-streaming profile and VRAM-budgeted residency

Offload makes the model fit. Expert residency budgeting asks how much spare VRAM the user wants
to spend to buy back speed. Blunt rule: measure the wall before building the door through it.

## 1. Purpose

Determine whether offload slowdown is concentrated in a small number of layer/expert pairs. If
it is, a later pass can test a user-dialable expert residency budget — pin the highest-value
experts, spend a little VRAM, recover some of the CPU/RAM→VRAM transfer penalty. If it is not,
we document that the offload wall is diffuse for this model/path and do not build a cache. This
is a separate experiment family from the OLMoE repeat grid, the portability matrix, and Qwen3
Tier-3; all outputs live under `runs/expert_streaming/` (job-local dirs + atomic locks, per
`docs/RUNPOD_DISTRIBUTED_VALIDATION.md` — parallelize profile jobs, never output files).

## 2. Offload is not binary

Expert residency budget turns offload into a dial. At 0 GB, all experts stream from CPU. As the
user allocates more VRAM, the system pins the layer/expert pairs with the highest observed stall
cost per resident byte:

- **0 GB extra** — stream everything; lowest peak GPU; slowest (today's `OFFLOAD_EXPERTS=1`).
- **0.25 / 0.5 / 1.0 / 2.0 GB extra** — pin the highest-value pairs; recover some stall.
- **full resident** — highest VRAM; fastest if it fits.

The future user-facing shape (NOT implemented in this pass — the dial is only built if the
profile proves it has something useful to spend):

```
--expert-residency-policy all-offload | hot-static      (later, maybe: auto)
--gpu-expert-cache-gb 0.5
```

The motivating asymmetry, from the OLMoE grid (`docs/OLMOE_EXPERTSNBIT_GRID.md`): int8-offload
holds a near-4-bit memory floor (2.72 vs 2.52 GB) far below int8 resident (8.50 GB) but pays
18.5 vs 12.1 s/step — and currently carries the best single eval, pending repeats. Can most of
that memory advantage be kept while buying back part of the stall?

## 3. Profile-only methodology

No behavior change: no cache, no scheduler, no prefetcher, no residency-policy change, no pinning.
The profiler (`experts4bit_qlora/expert_profile.py`, gated on `E4B_EXPERT_PROFILE=<out.jsonl>`,
attached by `train.py`/`infer.py` after the model is built) records two things and changes
nothing:

- **per-layer H2D staging** — count, bytes, and wall of each staging copy, via CUDA events
  recorded on the staging stream and reduced with a *single* `synchronize` at flush (never in the
  hot path, so the profiler does not destroy the timing being measured; methodology is recorded
  in the trace meta row), tagged sync / cold_miss / prefetch.
- **per-(layer, expert) routing** — `hits` and `tokens_routed`, as on-device bincounts in a
  forward pre-hook (one bincount per layer forward, no host sync until flush).

**Load-bearing caveat: staging is layer-granular.** `_ExpertOffload` moves a layer's *entire*
fused expert stack per visit; there is no per-expert transfer in the current design. Measured
H2D stall is per layer. Per-expert stall is a **projection** (the summarizer shares a layer's
stall among its routed experts, weighted by `tokens_routed`) — the number a per-expert pinning
policy would have to beat, not a measurement of isolated per-expert transfer. Policy files name
the field `stall_ms_projected` accordingly.

Sampling: 100 training steps per training profile; 5 decode samples + 1 discarded warmup per
decode profile.

## 4. OLMoE int8-offload and nf4-offload traces

First target: **int8-offload training** (the asymmetry above). Comparator: **nf4-offload**
training — the low-memory baseline; comparing the two shows whether wider storage changes the
transfer-stall *profile*, not just the bytes. Decode profiles follow after training profiles.

Jobs (`scripts/make_expert_streaming_manifest.py`), independent, one pod each is fine:

- `profile_olmoe_int8_offload_train_seed1337_steps100`  ← first
- `profile_olmoe_nf4_offload_train_seed1337_steps100`
- `profile_olmoe_int8_offload_decode_repeat5` (optional, after)
- `profile_olmoe_nf4_offload_decode_repeat5` (optional, after)

*Traces filled from `scripts/summarize_expert_streaming.py` output when the jobs land.*

## 5. Hotness and stall concentration

*Filled from the summary: top layer/expert pairs by hits / tokens routed / H2D bytes / projected
stall, and the concentration curves (share of stall, hits, tokens, and bytes held by the top
1% / 5% / 10% / 20% of pairs).*

## 6. Budget simulation

Before implementing any pinning, the summarizer simulates what residency would have bought:
greedy selection by `pin_score = projected_stall_ms_saved / resident_bytes` (fallbacks for noisy
timing: `h2d_bytes/resident_bytes` or `hits*h2d_bytes/resident_bytes`; greedy, not a knapsack —
v1 by design) at budgets 0.25 / 0.5 / 1.0 / 2.0 GB, reporting pinned experts, added VRAM,
projected stall covered, H2D GB avoided, and hit coverage.

*Budget table filled from the summary.*

## 7. Static hot-set recommendation

For each budget the summarizer (controller-only) writes a machine-readable policy file:

```
runs/expert_streaming/policies/olmoe_int8_offload_hotstatic_budget{0.25,0.5,1.0,2.0}gb.json
```

each listing the selected (layer, expert) pairs with resident bytes, score, projected stall,
hits, and tokens — plus the attribution rule, so no consumer can mistake projection for
measurement.

**Decision criterion (pre-registered): build hot-static only if the top 10% of pairs hold ≥40%
of projected stall, or the top 20% hold ≥60%.** If stalls are diffuse, no cache gets built and
this doc records that the model/path does not show hot-expert concentration.

## 8. Hot-static validation, if run

Only if §7's criterion is met: `olmoe_int8_offload_hotstatic_seed1337_steps100_budget1gb`
(its own job dir), compared against the plain int8-offload seed-1337 100-step run on added peak
GPU, s/step change, stall reduction, before/after eval, the selected experts, and whether the
measured gain roughly matches the projected gain. Success = spend ≤1 GB extra VRAM and recover a
measurable fraction of the offload slowdown without changing the intended math path. Resident
speed is NOT the goal. If built, the policy is `hot-static` only — no LRU, no LFU, no
router-aware prefetch, no async stream scheduler, no learned policy, no cross-layer prediction.

## 9. What is not claimed

- No global hot experts and no universal pinning sets — hotness is model/host/dataset/phase
  specific; query hotness is not assumed to equal train hotness, nor train hotness another
  dataset's.
- No guaranteed speedup, and no speedup at all from profile-only runs — budget coverage is a
  projection until a hot-static validation run measures it.
- No claim that one budget is best; the budget is user-selectable by design.
- No measurement of isolated per-expert transfer (staging is layer-granular; §3).

Claimed, at most: profile-derived recommendations, a measured-or-projected stall-per-GB
trade-off, and model/host/dataset/phase-specific findings.

## 10. Next steps

Qwen3-30B-A3B gets a later *sentinel* profile only if OLMoE shows concentrated stalls and
hot-static looks worth scaling — not before, and not on Qwen3 first.

---

<!-- ots-attestation-footer -->

**OpenTimestamps anchor (self-attestation footer):**

- **OTS proof timestamp for visible document:** `2026-07-05T09:38:20Z` (the moment the current `.ots` was submitted to the calendars; this is the legally operative timestamp for the visible file as published).
- **Disclosed pre-footer content hash:** `626c88bae971413df5ca73c8b8ffa11973e0ddea69c7849b7c4a2ce759d1cf87` (the SHA-256 of the document *before* this footer was appended — disclosed inside the OTS-anchored visible document for human-readable historical reference; this hash is *not* the payload of the current `.ots` file).
- **Prior disclosed pre-footer hashes (chain, newest first):**
  - `2026-07-05T09:33:07Z` `e3990334d90b819081d12ce9a613048aef1bae9633dd22933d98ba9dc6513d8c`
- integrity-attestor glyph (`core.fingerprint`, first 8 bytes of the disclosed pre-footer hash): `[0+0&**@%?#=:o:~!]`
- Drunken-bishop randomart (full disclosed pre-footer SHA-256, OpenSSH-style):

```
+----[SHA256]-----+
|       .         |
|    . . .        |
|   . o   .       |
|  .. o= o   .    |
|  ....=BS... .   |
| .  .oo.*.... o. |
|.. . . =.B=o  Eo.|
| oo   . @=*+    .|
|+.     ++O+      |
+-----------------+
```

- **Payload hash actually covered by the current `.ots`:** see `ots info EXPERT_STREAMING_PROFILE.md.ots`; by construction this is `SHA-256(this entire file including this footer)` and `ots verify EXPERT_STREAMING_PROFILE.md.ots EXPERT_STREAMING_PROFILE.md` succeeds against the on-disk bytes.
- Anchor file: `EXPERT_STREAMING_PROFILE.md.ots`
- Calendars: a.pool.opentimestamps.org, b.pool.opentimestamps.org, a.pool.eternitywall.com, ots.btc.catallaxy.com
- **Provenance posture (load-bearing):** the **OTS proof timestamp** above is the legal anchoring time for the visible document — that is what the calendars witnessed. The **disclosed pre-footer content hash** is *not* anchored by the current `.ots` file; it is *disclosed inside* the OTS-anchored visible document as a human-readable historical record of what the file's bytes hashed to immediately before this footer was appended. A reviewer verifying the visible file runs `ots verify` against the on-disk bytes; a reviewer wanting to confirm the disclosed pre-footer hash recomputes `SHA-256` of the file with everything from `<!-- ots-attestation-footer -->` onward stripped. Both checks are independent; neither replaces the other.
