# OLMoE-1B-7B ExpertsNbit storage-mode validation grid

First-pass, single-run-per-cell validation grid: the same OLMoE-1B-7B QLoRA training run and
resident decode measurement across ExpertsNbit storage modes, flipping only `QUANT_TYPE` (plus an
`OFFLOAD_EXPERTS=1` leg per mode). Runner: `bench/run-mode-ab.sh`. This is a validation grid, not
a benchmark — the numbers characterize the storage contract on the measured host. OLMoE numbers
only; Qwen3-30B-A3B results live in `docs/QWEN3_30B_EXPERTSNBIT_GRID.md` and never here.

Host: rented RunPod RTX A5000 24 GB (PCIe 4.0), torch 2.8.0+cu128, bitsandbytes 0.49.2,
transformers 5.13.0. 150 steps, seq 256, grad-accum 4, lr 1e-4, r=8/alpha=16, seed 0, Alpaca
response-only loss (the METHODOLOGY §7 hyperparameters, so the nf4 row doubles as the in-run
control and stays comparable to the ablation).

## The grid (single run per cell — statuses per finding below)

All 18 legs complete (2026-07-05, single run per cell). "best eval" = best held-out Alpaca loss.

| mode | train resident<br>peak GB · s/step · best eval | train offload<br>peak GB · s/step · best eval | resident decode<br>tok/s · peak GB |
|---|---|---|---|
| nf4 | 5.28 · 12.4 · 1.0289 | 2.52 · 16.7 · 1.0304 | 10.12 · 4.72 |
| fp4 | 5.28 · 14.0 · 1.0293 | 2.52 · 16.5 · 1.0297 | 12.59 · 4.72 |
| int8 | 8.50 · 12.1 · 1.0245 | 2.72 · 18.5 · 1.0140 | 9.35 · 7.95 |
| fp8 | 8.50 · 13.8 · 1.0204 | 2.72 · 19.3 · 1.0281 | 9.35 · 7.95 |
| bf16 | 14.54 · 11.0 · 1.0220 | 2.41 · 20.8 · **1.0112** | 13.26 · 13.99 |
| fp16 | 14.54 · 11.2 · 1.0194 | 2.41 · 20.5 · 1.0181 | 12.57 · 13.99 |

Held-out eval loss BEFORE training tracks the reconstruction-fidelity chain in end-model terms:
fp8 1.4724 < fp16 1.4780 < bf16 1.4818 < int8 1.4811 < nf4 1.4905 < fp4 1.5041 (the fp8/int8 and
fp16/bf16 pairs are within noise of each other; the 4-bit modes are clearly worst, as expected).

Shape of the completed grid (all observed in this run):
- **Resident peak scales cleanly with storage width:** ~5.28 GB (4-bit) → 8.50 (8-bit) → 14.54
  (16-bit).
- **Offload collapses every width to ~2.4–2.7 GB** — and the ordering even inverts slightly
  (16-bit offload 2.41 GB < 4-bit 2.52 GB), because passthrough carries no absmax buffers and no
  dequant workspace. The storage-width memory axis nearly vanishes under offload.
- **Offload s/step rises with bytes-streamed:** ~16.5 (4-bit) → ~19 (8-bit) → ~20.7 (16-bit).
- **Decode throughput** (resident): bf16 13.26 > fp4 12.59 ≈ fp16 12.57 > nf4 10.12 > int8 ≈
  fp8 9.35 tok/s.

## The storage/offload asymmetry (why this grid matters)

Resident training exposes the full memory cost of wider storage. Offload moves most expert
storage out of GPU residency, so wider modes such as int8 add much less peak GPU memory under
offload (2.52 → 2.72 GB, +0.20) than under resident training (5.28 → 8.50 GB, +3.22). This
creates a useful lifecycle split: train under the regime that fits, then query under the regime
that best balances fidelity, latency, locality, and memory. Whether adapters actually survive
that split is the portability question — measured separately
(`docs/MODE_DECOUPLED_ADAPTERS.md`), never assumed.

## Findings and their status

| finding | status |
|---|---|
| offload collapses the storage-width memory difference (now across all 6 modes: resident 5.28/8.50/14.54 GB → offload 2.41–2.72 GB) | Candidate (expected Stable + Host-specific) — nf4/int8 repeats running |
| resident memory scales with storage width (4→8→16-bit: 5.28→8.50→14.54 GB) | Candidate (expected Stable + Host-specific) — nf4/int8 repeats running |
| the lowest single-run offload eval is bf16-offload 1.0112, then int8-offload 1.0140 | **Candidate** — single run each; only nf4/int8 are being repeated, so bf16's 1.0112 stays an unrepeated observation. Do NOT rank offload modes by eval on one run |
| fp4 resident decode faster than nf4 (12.59 vs 10.12 tok/s); bf16 fastest overall (13.26) | **Candidate** — single sample each, decode is noisy, repeat-5 jobs queued for nf4/fp4/int8 |
| offload eval ≈ resident eval | Candidate — same math by design; AFTER values drift from GPU nondeterminism accumulated over 150 steps, not offload math |
| BEFORE-eval fidelity ordering (4-bit worst; 8/16-bit clustered) | Candidate (mechanism separately test-pinned by the reconstruction chain) |

Repeat plan and graduation rules: `docs/OLMOE_REPEAT_VALIDATION_PLAN.md`. Distributed execution:
`docs/RUNPOD_DISTRIBUTED_VALIDATION.md`.

## What this grid does not claim

- No universal speed ranking, and no claim that fp4 is generally faster than nf4.
- No universal best mode; no claim that int8-offload always improves eval.
- No claim that cross-mode adapters are universally compatible (see the portability matrix).
- No claim that this host's s/step or tok/s generalizes to any other host or link — offload and
  decode costs are transfer-bound and per-host by construction.
- Single-run cells are Candidate by definition until the repeat set lands.

---

<!-- ots-attestation-footer -->

**OpenTimestamps anchor (self-attestation footer):**

- **OTS proof timestamp for visible document:** `2026-07-05T12:47:30Z` (the moment the current `.ots` was submitted to the calendars; this is the legally operative timestamp for the visible file as published).
- **Disclosed pre-footer content hash:** `7ed1e2a56c3b28abd783d840af4d76608388218695555791426d75f83dbcf69e` (the SHA-256 of the document *before* this footer was appended — disclosed inside the OTS-anchored visible document for human-readable historical reference; this hash is *not* the payload of the current `.ots` file).
- **Prior disclosed pre-footer hashes (chain, newest first):**
  - `2026-07-05T09:22:21Z` `df17a36577b78ceedcce7b029e54b759208c41502570948834a6bb945078e763`
- integrity-attestor glyph (`core.fingerprint`, first 8 bytes of the disclosed pre-footer hash): `[=?!:?+%O0&~@+*%@]`
- Drunken-bishop randomart (full disclosed pre-footer SHA-256, OpenSSH-style):

```
+----[SHA256]-----+
|...o....ooo+...  |
|+..    .. + ..   |
|oo . .   o   ... |
|. . o +    .  .o.|
|   . o oS o o   o|
|    . +..o =   o |
|     O +..*   . .|
|    o * +o..    o|
|    .o.o ...   E.|
+-----------------+
```

- **Payload hash actually covered by the current `.ots`:** see `ots info OLMOE_EXPERTSNBIT_GRID.md.ots`; by construction this is `SHA-256(this entire file including this footer)` and `ots verify OLMOE_EXPERTSNBIT_GRID.md.ots OLMOE_EXPERTSNBIT_GRID.md` succeeds against the on-disk bytes.
- Anchor file: `OLMOE_EXPERTSNBIT_GRID.md.ots`
- Calendars: a.pool.opentimestamps.org, b.pool.opentimestamps.org, a.pool.eternitywall.com, ots.btc.catallaxy.com
- **Provenance posture (load-bearing):** the **OTS proof timestamp** above is the legal anchoring time for the visible document — that is what the calendars witnessed. The **disclosed pre-footer content hash** is *not* anchored by the current `.ots` file; it is *disclosed inside* the OTS-anchored visible document as a human-readable historical record of what the file's bytes hashed to immediately before this footer was appended. A reviewer verifying the visible file runs `ots verify` against the on-disk bytes; a reviewer wanting to confirm the disclosed pre-footer hash recomputes `SHA-256` of the file with everything from `<!-- ots-attestation-footer -->` onward stripped. Both checks are independent; neither replaces the other.
