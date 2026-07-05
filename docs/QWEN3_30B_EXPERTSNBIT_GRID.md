# Qwen3-30B-A3B scale-transfer validation

Qwen3-30B-A3B is the gated larger-model target. Its job is to test whether the **topology**
observed on OLMoE-1B-7B (`docs/OLMOE_EXPERTSNBIT_GRID.md`) transfers — not to inherit OLMoE row
winners, memory numbers, or eval ordering. No OLMoE numbers appear in this document; no Qwen3
number is projected onto OLMoE. The transferable claim is the topology, not the exact rows.

## Evidence scope

- Model: `Qwen/Qwen3-30B-A3B` (base model, no adapter — a decode-grid probe of the load/offload path)
- Bundle: two Qwen3 tier-3 probes, 2026-07-05
  - **Host A** — RunPod RTX A5000, 24 GB VRAM, **25 GB container RAM cap** (the first probe)
  - **Host B** — RunPod A100 80GB PCIe, **236 GB container RAM** (the offload probe, `E4B_COMMIT`-attested)
- Runner: `bench/run-bigmoe-decode.sh`, `BENCH_TOKENS=96`, greedy decode
- Results apply to these hosts/runs. torch 2.8.0+cu128, bitsandbytes 0.49.2.

## Results

| config | nf4: tok/s · peak GB | int8: tok/s · peak GB | status |
|---|---|---|---|
| resident decode | 5.19 · 20.04 (host B) / 4.14 · 20.04 (host A) | 4.99 · **34.54** (host B) | nf4 **validated** (fits a 24 GB card); int8 resident **impractical** on 24 GB (34.5 GB — OOMs host A, fits A100) |
| offload-serial | 1.15 · **4.07** | 0.67 · **4.38** | **validated** (host B): offload path completes at 30B |
| offload-prefetch | 1.65 · 4.41 | 0.88 · 5.02 | **validated**: prefetch 1.44x (nf4) / 1.31x (int8) over serial |
| offload-prefetch+dq | 1.65 · 4.42 | 0.88 · 5.02 | GEMV neutral here (prefetch+dq ~= prefetch) |

Raw: `runs/results/qwen3/qwen3_{nf4,int8}_{decode,offload_a100}.txt`.

## The topology transfers (this is the point)

- **Offload preserves fit at 30B.** Both nf4 (4.07-4.41 GB) and int8 (4.38-5.02 GB) offload peak
  GPU is ~4-5 GB for a 30-billion-parameter model — small enough for an 8 GB card. The OLMoE
  "offload makes it fit on a small card" result holds at 15x the parameter count.
- **Offload still collapses the storage-width difference.** int8-offload peak (4.4-5.0 GB) sits
  only ~0.3-0.6 GB above nf4-offload (4.1-4.4 GB), while resident int8 (34.5 GB) is 14 GB above
  resident nf4 (20 GB). The same width-collapse topology OLMoE showed, at 30B.
- **Offload is a memory-for-speed trade**, as on OLMoE: 0.67-1.65 tok/s offloaded vs ~5 tok/s
  resident — transfer-bound, expected. Prefetch recovers a fraction (1.3-1.4x).
- **The host-A block was purely the 25 GB container RAM cap**, not a mechanism failure: on host B
  (236 GB RAM) every offload config ran clean.

## Scale-transfer questions

| question | answer |
|---|---|
| Does the resident support matrix transfer? | Yes: nf4 resident fits 24 GB (20 GB); int8 resident is impractical on 24 GB (34.5 GB, fits only bigger cards). |
| Does offload preserve fit at 30B? | **Yes — validated (host B).** Offload peak ~4-5 GB (nf4 and int8), a small-card footprint for a 30B model. |
| Does the offload width-collapse topology transfer? | Yes: int8-offload peak barely exceeds nf4-offload, mirroring OLMoE. |
| Does resident int8 become impractical on available hardware? | Yes on a 24 GB card; fits an 80 GB A100. |
| Do profile concentration patterns transfer? | `not_tested` — OLMoE's offload wall was diffuse (no hot-static built), so there is no trigger to profile Qwen3 for hotness. |

## What is not claimed

- No Qwen3 confirmation of any OLMoE *eval* row winner (int8-offload eval strength, etc.) — this
  probe is base-model decode, no adapter, no training eval.
- No claim these tok/s generalize to other hosts/links — offload decode is transfer-bound and
  per-host. The *fit* result (peak GB) is the robust, transferable one.
- The two resident nf4 tok/s differ by host (5.19 A100 vs 4.14 A5000) — decode speed is
  host-specific; peak GB (20.04, identical) is the stable quantity.

## Remaining gap

Qwen3 offload was measured on an A100 (host B) because the volume's datacenter had no high-RAM
GPU in stock; the A100's spare VRAM means "fits a small card" is inferred from the ~4-5 GB peak,
not from running on an actually-small card in one shot. The peak-GB measurement is the evidence;
a single-run on, say, an 8 GB card would be a confirmation, not a new result. Not required to
answer the scale-transfer question, which the peak-GB numbers already settle.

---

<!-- ots-attestation-footer -->

**OpenTimestamps anchor (self-attestation footer):**

- **OTS proof timestamp for visible document:** `2026-07-05T14:51:55Z` (the moment the current `.ots` was submitted to the calendars; this is the legally operative timestamp for the visible file as published).
- **Disclosed pre-footer content hash:** `65d50197d8585fbf3ace2456b59fb294d53dbbd8ea344da33f106ccc0776112c` (the SHA-256 of the document *before* this footer was appended — disclosed inside the OTS-anchored visible document for human-readable historical reference; this hash is *not* the payload of the current `.ots` file).
- **Prior disclosed pre-footer hashes (chain, newest first):**
  - `2026-07-05T14:45:45Z` `65d50197d8585fbf3ace2456b59fb294d53dbbd8ea344da33f106ccc0776112c`
  - `2026-07-05T13:19:31Z` `5861bf583dfa4ed6ac94dd469db20a99b40e4b09732fdc7cf3f749a30fa0b91c`
- integrity-attestor glyph (`core.fingerprint`, first 8 bytes of the disclosed pre-footer hash): `[0O!O.:#=!*O*O$@$]`
- Drunken-bishop randomart (full disclosed pre-footer SHA-256, OpenSSH-style):

```
+----[SHA256]-----+
|            oB==+|
|           .oE++o|
|          o = +.o|
|         o   *..=|
|        S   ..oBo|
|            ..B *|
|           o @.+.|
|          . B O..|
|            .B.+.|
+-----------------+
```

- **Payload hash actually covered by the current `.ots`:** see `ots info QWEN3_30B_EXPERTSNBIT_GRID.md.ots`; by construction this is `SHA-256(this entire file including this footer)` and `ots verify QWEN3_30B_EXPERTSNBIT_GRID.md.ots QWEN3_30B_EXPERTSNBIT_GRID.md` succeeds against the on-disk bytes.
- Anchor file: `QWEN3_30B_EXPERTSNBIT_GRID.md.ots`
- Calendars: a.pool.opentimestamps.org, b.pool.opentimestamps.org, a.pool.eternitywall.com, ots.btc.catallaxy.com
- **Provenance posture (load-bearing):** the **OTS proof timestamp** above is the legal anchoring time for the visible document — that is what the calendars witnessed. The **disclosed pre-footer content hash** is *not* anchored by the current `.ots` file; it is *disclosed inside* the OTS-anchored visible document as a human-readable historical record of what the file's bytes hashed to immediately before this footer was appended. A reviewer verifying the visible file runs `ots verify` against the on-disk bytes; a reviewer wanting to confirm the disclosed pre-footer hash recomputes `SHA-256` of the file with everything from `<!-- ots-attestation-footer -->` onward stripped. Both checks are independent; neither replaces the other.
