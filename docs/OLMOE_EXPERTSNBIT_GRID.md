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

Resolved against the 3-seed repeat grid (bundle `olmoe-qlora-grid-20260705-1351`, seeds
1337/2027/3407; full tables in `docs/OLMOE_REPEAT_VALIDATION_PLAN.md`). "OLMoE-supported" = held
across all three seeds under the summarizer's printed rule; still host-specific.

| finding | status |
|---|---|
| offload collapses the storage-width memory difference (resident 5.28/8.50/14.54 GB → offload 2.41–2.72 GB) | **OLMoE-supported, host-specific** — 3/3 seeds; offload width-delta 0.20 GB vs resident 3.22 GB (ratio 0.06) |
| resident memory scales with storage width (4→8→16-bit) | **OLMoE-supported, host-specific** — 3/3 seeds |
| int8-offload posts the best training eval | **Candidate, CONFOUNDED** (downgraded by audit) — best-eval in 3/3 seeds (agg 1.0261 ± 0.0079), but the audit shows the offload-trained rows carry a precision×placement interaction: at *resident* placement int8-vs-nf4 training is a tie (+0.00 G), and byte-identical bf16 resident-vs-offload differ 0.0108 (RNG/recompute, not math). The offload-trained "best" is one uncertified mechanism — see `docs/MEASUREMENT_AUDIT.md` §3; needs debt D3. Also: frozen int8 already covers 108% of the nf4→bf16 quality gap (∅ ladder, §1) — the training-precision axis is nearly flat here |
| fp4 resident decode faster than nf4 | **NOT supported on repeat** — repeat-5: fp4 12.87 ± 0.20 vs nf4 12.68 ± 0.22 tok/s, overlapping within a std. The single-run 10.12 nf4 was a slow outlier; on repeats they tie. int8 decode is slower (11.63 ± 0.02). |
| offload eval ≈ resident eval | Observed — same math by design; per-seed AFTER drift is GPU nondeterminism, not offload math |
| BEFORE-eval fidelity ordering (int8 < nf4; 4-bit worst) | **OLMoE-supported, host-specific** — 3/3 seed-matched pairs |

**Provenance caveat (this bundle):** metrics, environment, GPU, and library versions are captured
per job, but the per-job *commit* was not self-reported — workers ran from `git archive` trees
(no `.git`), so the provenance gate classes these `debug_only` on the commit check alone. The
executed training path is functionally identical across the bundle's branch commits (the only
diff is a gated no-op profiler hook the repeat jobs never enabled). The runner now records commit
via `E4B_COMMIT` so subsequent runs self-attest. The seed-reproduction results rest on the
captured metrics, not on commit attestation.

Repeat plan, full seed tables, and graduation rules: `docs/OLMOE_REPEAT_VALIDATION_PLAN.md`.
Distributed execution: `docs/RUNPOD_DISTRIBUTED_VALIDATION.md`.

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

- **OTS proof timestamp for visible document:** `2026-07-05T16:48:27Z` (the moment the current `.ots` was submitted to the calendars; this is the legally operative timestamp for the visible file as published).
- **Disclosed pre-footer content hash:** `da27fc023ff59a02638649db4fff218f5d05780c02d0fdd8bfde0e247571630a` (the SHA-256 of the document *before* this footer was appended — disclosed inside the OTS-anchored visible document for human-readable historical reference; this hash is *not* the payload of the current `.ots` file).
- **Prior disclosed pre-footer hashes (chain, newest first):**
  - `2026-07-05T13:53:44Z` `db6897adffbbeb8f107366e72b000fb59c47ee8db459b11563b42f8cf4f408a7`
  - `2026-07-05T13:53:02Z` `dd842d5829cf809a2dc7a940dfb625391bf0075297f5f54ccd306f5479a6d680`
  - `2026-07-05T12:47:30Z` `7ed1e2a56c3b28abd783d840af4d76608388218695555791426d75f83dbcf69e`
  - `2026-07-05T09:22:21Z` `df17a36577b78ceedcce7b029e54b759208c41502570948834a6bb945078e763`
- integrity-attestor glyph (`core.fingerprint`, first 8 bytes of the disclosed pre-footer hash): `[!%+=$&.+~$$O#%.+]`
- Drunken-bishop randomart (full disclosed pre-footer SHA-256, OpenSSH-style):

```
+----[SHA256]-----+
|      .o.o.E.  +.|
|        . ...+o +|
|           +..= .|
|    .     . oo o |
|   . =  S   ... .|
|    + B+. .  o.. |
|     o.B+oo.. o. |
|        *+.*.o.o |
|         +=o+...o|
+-----------------+
```

- **Payload hash actually covered by the current `.ots`:** see `ots info OLMOE_EXPERTSNBIT_GRID.md.ots`; by construction this is `SHA-256(this entire file including this footer)` and `ots verify OLMOE_EXPERTSNBIT_GRID.md.ots OLMOE_EXPERTSNBIT_GRID.md` succeeds against the on-disk bytes.
- Anchor file: `OLMOE_EXPERTSNBIT_GRID.md.ots`
- Calendars: a.pool.opentimestamps.org, b.pool.opentimestamps.org, a.pool.eternitywall.com, ots.btc.catallaxy.com
- **Provenance posture (load-bearing):** the **OTS proof timestamp** above is the legal anchoring time for the visible document — that is what the calendars witnessed. The **disclosed pre-footer content hash** is *not* anchored by the current `.ots` file; it is *disclosed inside* the OTS-anchored visible document as a human-readable historical record of what the file's bytes hashed to immediately before this footer was appended. A reviewer verifying the visible file runs `ots verify` against the on-disk bytes; a reviewer wanting to confirm the disclosed pre-footer hash recomputes `SHA-256` of the file with everything from `<!-- ots-attestation-footer -->` onward stripped. Both checks are independent; neither replaces the other.
