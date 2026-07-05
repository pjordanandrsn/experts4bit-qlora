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

| mode | train resident<br>peak GB · s/step · best eval | train offload<br>peak GB · s/step · best eval | resident decode<br>tok/s · peak GB |
|---|---|---|---|
| nf4 | 5.28 · 12.4 · 1.0289 | 2.52 · 16.7 · 1.0304 | 10.12 · 4.72 |
| fp4 | 5.28 · 14.0 · 1.0293 | 2.52 · 16.5 · 1.0297 | 12.59 · 4.72 |
| int8 | 8.50 · 12.1 · 1.0245 | 2.72 · 18.5 · **1.0140** | 9.35 · 7.95 |
| fp8 | *legs completing* | | |
| bf16 | *legs completing* | | |
| fp16 | *legs completing* | | |

Held-out eval loss BEFORE training: int8 1.4811 < nf4 1.4905 < fp4 1.5041 — the test-pinned
reconstruction-fidelity ordering surfacing in end-model terms.

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
| offload collapses the storage-width memory difference | Candidate (expected Stable + Host-specific) — repeats running |
| resident memory scales with storage width | Candidate (expected Stable + Host-specific) — repeats running |
| int8-offload posts the best training eval (1.0140) | **Candidate** — single run, outside the ~±0.006 spread of the other legs, needs seeds |
| fp4 resident decode faster than nf4 (12.59 vs 10.12 tok/s) | **Candidate** — single sample each, decode is noisy, repeat-5 jobs queued |
| offload eval matches resident eval | Candidate — same math by design; AFTER values drift ~0.9% from GPU nondeterminism accumulated over 150 steps, not offload math |
| BEFORE-eval fidelity ordering int8 < nf4 < fp4 | Candidate (mechanism separately test-pinned by the reconstruction chain) |

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

- **OTS proof timestamp for visible document:** `2026-07-05T09:22:21Z` (the moment the current `.ots` was submitted to the calendars; this is the legally operative timestamp for the visible file as published).
- **Disclosed pre-footer content hash:** `df17a36577b78ceedcce7b029e54b759208c41502570948834a6bb945078e763` (the SHA-256 of the document *before* this footer was appended — disclosed inside the OTS-anchored visible document for human-readable historical reference; this hash is *not* the payload of the current `.ots` file).
- integrity-attestor glyph (`core.fingerprint`, first 8 bytes of the disclosed pre-footer hash): `[!$:=%~0O==@=*&??]`
- Drunken-bishop randomart (full disclosed pre-footer SHA-256, OpenSSH-style):

```
+----[SHA256]-----+
|   ...+.oBB*.    |
|  ...oo...o.o .  |
|  ...o       . . |
|   . oE       . o|
|    +. .S    .=.B|
|   . .   . .o+o*+|
|    .     .oo+.o |
|            =.+ .|
|            .+.*o|
+-----------------+
```

- **Payload hash actually covered by the current `.ots`:** see `ots info OLMOE_EXPERTSNBIT_GRID.md.ots`; by construction this is `SHA-256(this entire file including this footer)` and `ots verify OLMOE_EXPERTSNBIT_GRID.md.ots OLMOE_EXPERTSNBIT_GRID.md` succeeds against the on-disk bytes.
- Anchor file: `OLMOE_EXPERTSNBIT_GRID.md.ots`
- Calendars: a.pool.opentimestamps.org, b.pool.opentimestamps.org, a.pool.eternitywall.com, ots.btc.catallaxy.com
- **Provenance posture (load-bearing):** the **OTS proof timestamp** above is the legal anchoring time for the visible document — that is what the calendars witnessed. The **disclosed pre-footer content hash** is *not* anchored by the current `.ots` file; it is *disclosed inside* the OTS-anchored visible document as a human-readable historical record of what the file's bytes hashed to immediately before this footer was appended. A reviewer verifying the visible file runs `ots verify` against the on-disk bytes; a reviewer wanting to confirm the disclosed pre-footer hash recomputes `SHA-256` of the file with everything from `<!-- ots-attestation-footer -->` onward stripped. Both checks are independent; neither replaces the other.
