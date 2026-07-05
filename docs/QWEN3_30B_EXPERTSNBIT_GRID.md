# Qwen3-30B-A3B scale-transfer validation

Qwen3-30B-A3B is the gated larger-model target. Its job is to test whether the **topology**
observed on OLMoE-1B-7B (`docs/OLMOE_EXPERTSNBIT_GRID.md`) transfers — not to inherit OLMoE row
winners, memory numbers, or eval ordering. No OLMoE numbers appear in this document; no Qwen3
number is projected onto OLMoE. The transferable claim is the topology, not the exact rows.

## Evidence scope

- Model: `Qwen/Qwen3-30B-A3B` (base model, no adapter — decode-grid probe of the load/offload path)
- Bundle: first Qwen3 tier-3 probe, 2026-07-05
- Host: **one** RunPod RTX A5000 (24 GB VRAM), torch 2.8.0+cu128, bitsandbytes 0.49.2
- Runner: `bench/run-bigmoe-decode.sh`, `BENCH_TOKENS=96`
- These results apply only to this host and run.

## What ran, and its status

| config | result | status |
|---|---|---|
| nf4 resident decode | 4.14 tok/s, peak **20.04 GB** VRAM, prefill 1.75 s | **validated** (this host): nf4 30B fits resident on a 24 GB card |
| int8 resident decode | CUDA OOM (>24 GB VRAM) | **impractical** on 24 GB (expected — int8 doubles expert bytes) |
| nf4 / int8 offload (serial, prefetch, prefetch+dq) | OS-killed ("Killed") during the streaming quantize-load, before decode | **blocked** — host RAM cap, see below |

## Why the offload configs were blocked (not a mechanism failure)

The offload runs died during `fusing + quantizing experts ... (streaming)`, before any decode.
The container's memory cgroup cap is **25 GB** (`/sys/fs/cgroup/memory/memory.limit_in_bytes` =
24999997440) even though the host shows 251 GB free — a standard single-A5000 RunPod instance
caps container RAM well below the physical total. The streaming loader stages bf16 shards while
building the pinned CPU expert homes (~15 GB for nf4, ~30 GB for int8 at this model size), which
exceeds the 25 GB cap and trips the OS OOM-killer.

This is a **host resource cap, not an offload-mechanism failure**: the same offload path runs
Qwen3-30B-A3B nf4 offload+prefetch decode elsewhere on record (a 12 GB A2000 with ample host
RAM). Classified `blocked`, not `broken` and not "offload doesn't scale."

## Scale-transfer questions

| question | answer on this host |
|---|---|
| Does the resident support matrix transfer? | Partially: nf4 resident **fits** (20 GB); int8 resident is **impractical** on 24 GB. |
| Does offload preserve fit at 30B? | **Not answered here** — `blocked` by the 25 GB container RAM cap during load. |
| Does resident int8 become impractical on available hardware? | Yes on a 24 GB card (OOM). |
| Do profile concentration patterns transfer? | `not_tested` — OLMoE profiles run first (`docs/EXPERT_STREAMING_PROFILE.md`); Qwen3 is a sentinel only if OLMoE shows concentration. |

## What is not claimed

- No Qwen3 confirmation of any OLMoE row winner (int8-offload eval strength, fp4 decode speed,
  etc.). Those are OLMoE observations; Qwen3 does not inherit them.
- No offload throughput/memory number for Qwen3 — the offload path did not complete on this host.
- No eval-quality result — this probe used the base model with no adapter.

## Next step (explicit, not run here)

To answer "does offload preserve fit at 30B," re-run `bench/run-bigmoe-decode.sh` for
`Qwen/Qwen3-30B-A3B` on a pod whose **container RAM cap** exceeds ~40 GB (int8) / ~20 GB (nf4) —
i.e. select a RunPod instance by minimum system RAM, not just GPU. Until then the offload row
stays `blocked`. This is a gated follow-up; it is not run without explicit instruction.

---

<!-- ots-attestation-footer -->

**OpenTimestamps anchor (self-attestation footer):**

- **OTS proof timestamp for visible document:** `2026-07-05T13:19:31Z` (the moment the current `.ots` was submitted to the calendars; this is the legally operative timestamp for the visible file as published).
- **Disclosed pre-footer content hash:** `5861bf583dfa4ed6ac94dd469db20a99b40e4b09732fdc7cf3f749a30fa0b91c` (the SHA-256 of the document *before* this footer was appended — disclosed inside the OTS-anchored visible document for human-readable historical reference; this hash is *not* the payload of the current `.ots` file).
- integrity-attestor glyph (`core.fingerprint`, first 8 bytes of the disclosed pre-footer hash): `[O*0:@$O*~!$%o?!0]`
- Drunken-bishop randomart (full disclosed pre-footer SHA-256, OpenSSH-style):

```
+----[SHA256]-----+
|        o        |
|       . o .     |
|        . o o    |
|       o o o .  o|
|     o..S.o. . .o|
|      = * *..=oo |
|       * E o*o+oo|
|      . * ==+.+oo|
|       . + .ooooo|
+-----------------+
```

- **Payload hash actually covered by the current `.ots`:** see `ots info QWEN3_30B_EXPERTSNBIT_GRID.md.ots`; by construction this is `SHA-256(this entire file including this footer)` and `ots verify QWEN3_30B_EXPERTSNBIT_GRID.md.ots QWEN3_30B_EXPERTSNBIT_GRID.md` succeeds against the on-disk bytes.
- Anchor file: `QWEN3_30B_EXPERTSNBIT_GRID.md.ots`
- Calendars: a.pool.opentimestamps.org, b.pool.opentimestamps.org, a.pool.eternitywall.com, ots.btc.catallaxy.com
- **Provenance posture (load-bearing):** the **OTS proof timestamp** above is the legal anchoring time for the visible document — that is what the calendars witnessed. The **disclosed pre-footer content hash** is *not* anchored by the current `.ots` file; it is *disclosed inside* the OTS-anchored visible document as a human-readable historical record of what the file's bytes hashed to immediately before this footer was appended. A reviewer verifying the visible file runs `ots verify` against the on-disk bytes; a reviewer wanting to confirm the disclosed pre-footer hash recomputes `SHA-256` of the file with everything from `<!-- ots-attestation-footer -->` onward stripped. Both checks are independent; neither replaces the other.
