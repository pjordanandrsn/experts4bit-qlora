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

Bundle `olmoe-qlora-grid-20260705-1351`, one RTX A5000, seed 1337, 100 steps, 16 MoE layers x 64
experts (1024 routed layer/expert pairs). Raw: `runs/expert_streaming/jobs/`; summaries:
`runs/results/expert_streaming_{int8,nf4}_offload.md`. Provenance caveat as in the grid doc
(git-archive trees, no per-job commit; metrics/env captured; runner since fixed).

Every expert is hit on ~992/992 forwards — OLMoE's top-k routing over 64 experts touches
essentially all of them across a batch, so **hit-count is flat**; the only variation is in
`tokens_routed` (~10x range) and thus the token-weighted projected stall.

## 5. Hotness and stall concentration

Both offload storage modes give the same diffuse picture:

| metric (share held by hottest pairs) | top 1% | top 5% | top 10% | top 20% |
|---|---|---|---|---|
| int8-offload projected stall | 3% | 11% | 20% | 36% |
| nf4-offload projected stall | 3% | 11% | 20% | 35% |
| hits (either mode) | 1% | 6% | 11% | 23% |

No small hot set carries the wall: the top 10% of pairs hold ~20% of projected stall, the top
20% hold ~35-36% — close to their proportional share, i.e. nearly uniform.

## 6. Budget simulation

Projected stall covered if the hottest experts were pinned within a VRAM budget (greedy by
projected-stall-per-byte; a projection under the attribution rule, not a measured speedup):

| budget GB | int8-offload stall covered | nf4-offload stall covered |
|---|---|---|
| 0.25 | 9% (37 experts) | 14% (70) |
| 0.50 | 15% (74) | 26% (141) |
| 1.00 | 28% (149) | 45% (282) |
| 2.00 | 48% (299) | 74% (565) |

No cheap knee: coverage rises roughly linearly with budget (nf4 higher only because its experts
are half the bytes, so a budget pins twice as many). There is no small budget that buys back a
large fraction of the stall.

## 7. Static hot-set recommendation

**None.** The pre-registered gate — build only if top 10% of pairs hold >=40% of projected stall
OR top 20% hold >=60% — is not met by either mode (top 10% = 20%, top 20% = 35-36%). No policy
files are shipped; the budget simulation above is retained as the evidence for *why not*.

## 8. Hot-static validation, if run

Not run — the concentration gate did not pass, so no hot-static policy was built or validated.
This is a clean negative result: for OLMoE-1B-7B offload training on this host, the transfer wall
is not hot-expert concentrated, so a small fixed residency budget cannot route around it. Offload
here is closer to genuinely binary than to a high-value dial. Both outcomes were declared useful
in advance; this is the "does not justify hot-static" branch.

## 9. What is not claimed

- No per-expert transfer measurement (staging is layer-granular; per-expert stall is a projection).
- No claim this generalizes: a different model, dataset, or routing temperature could concentrate
  stall differently. This is OLMoE-1B-7B, this host, this run.
- The absence of concentration here does not imply offload is un-optimizable — only that *static
  hot-expert pinning under this criterion* is not the lever for this model/path.

## 10. Next steps

Qwen3-30B-A3B gets a sentinel profile only if a larger model shows concentration worth scaling —
and OLMoE did not, so there is no trigger to profile Qwen3 for hotness now. If a future model's
grid shows a hot-expert knee, this same apparatus (profiler + gate + budget sim) applies
unchanged.

---

<!-- ots-attestation-footer -->

**OpenTimestamps anchor (self-attestation footer):**

- **OTS proof timestamp for visible document:** `2026-07-05T13:57:40Z` (the moment the current `.ots` was submitted to the calendars; this is the legally operative timestamp for the visible file as published).
- **Disclosed pre-footer content hash:** `f2c1379e8b3f06585eab5f74ee22db3093c61b155dbd87559a0f559965a33146` (the SHA-256 of the document *before* this footer was appended — disclosed inside the OTS-anchored visible document for human-readable historical reference; this hash is *not* the payload of the current `.ots` file).
- **Prior disclosed pre-footer hashes (chain, newest first):**
  - `2026-07-05T09:38:20Z` `626c88bae971413df5ca73c8b8ffa11973e0ddea69c7849b7c4a2ce759d1cf87`
  - `2026-07-05T09:33:07Z` `e3990334d90b819081d12ce9a613048aef1bae9633dd22933d98ba9dc6513d8c`
- integrity-attestor glyph (`core.fingerprint`, first 8 bytes of the disclosed pre-footer hash): `[$+&:~=#?*@~$.0O*]`
- Drunken-bishop randomart (full disclosed pre-footer SHA-256, OpenSSH-style):

```
+----[SHA256]-----+
|            .E .#|
|            ..+O+|
|            ..=o.|
|       .. .  ..+.|
|      .+S.o.o . o|
|      .oo=.* o   |
|        .o@ . .  |
|        .o+X..   |
|        .+*+...  |
+-----------------+
```

- **Payload hash actually covered by the current `.ots`:** see `ots info EXPERT_STREAMING_PROFILE.md.ots`; by construction this is `SHA-256(this entire file including this footer)` and `ots verify EXPERT_STREAMING_PROFILE.md.ots EXPERT_STREAMING_PROFILE.md` succeeds against the on-disk bytes.
- Anchor file: `EXPERT_STREAMING_PROFILE.md.ots`
- Calendars: a.pool.opentimestamps.org, b.pool.opentimestamps.org, a.pool.eternitywall.com, ots.btc.catallaxy.com
- **Provenance posture (load-bearing):** the **OTS proof timestamp** above is the legal anchoring time for the visible document — that is what the calendars witnessed. The **disclosed pre-footer content hash** is *not* anchored by the current `.ots` file; it is *disclosed inside* the OTS-anchored visible document as a human-readable historical record of what the file's bytes hashed to immediately before this footer was appended. A reviewer verifying the visible file runs `ots verify` against the on-disk bytes; a reviewer wanting to confirm the disclosed pre-footer hash recomputes `SHA-256` of the file with everything from `<!-- ots-attestation-footer -->` onward stripped. Both checks are independent; neither replaces the other.
