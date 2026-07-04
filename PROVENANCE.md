# Provenance — measured evidence, stamped

Every figure traces to a committed script/test and the hardware it ran on. The offload A/B and the
26–35 B results are detailed in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) §11–§12; this file
stamps the full set with environment + commit for a single point of reference.

## Environment (this revision's runs)

- **Hardware:** RTX A2000 12 GB (Ampere, cc 8.6). Real 26–35 B checkpoints staged from 80 GB host RAM.
  PCIe link: **x8 electrical** (x16-capable card; gen reads 1 at idle via ASPM, upshifts under load) —
  measured pinned H2D ceiling **6.20 GB/s** (pageable 2.53 GB/s), the number the offload/decode
  transfer figures should be read against.
- **Versions:** Python 3.11.2 · torch 2.6.0+cu124 · bitsandbytes 0.49.2 (stock) · transformers 5.12.1 ·
  peft 0.19.1 · accelerate 1.14.0 · datasets 5.0.0 · safetensors 0.8.0
- **Date:** 2026-07-04  ·  **Repo commit:** `f3273a8` (@ `release/0.2.0`, merged to main via PR #6)
  ·  **Release:** v0.2.0 (PyPI). Suite/decode rows below ran at `c5c86f3`; `f3273a8` adds only the
  in-tree validation notebook (no code change), so the results carry to this release commit.
- **Historical environments:** rows marked *(2026-07-02 stack)* below were measured on the previously
  stamped environment (torch 2.11.0+cu128, trl 1.7.0 — see the prior revision, preserved with its
  proof in [`.ots-history/`](.ots-history/)). The OLMoE decode grid (§12b) was measured during the
  inference-mode branch work and is attested as committed in `f85d084`. Everything else in this file
  was (re)measured on the header stack.

## Correctness

| claim | value | source |
|---|---|---|
| reference parity — primitive vs float SwiGLU-MoE ref (+ swap/drop controls, orientation) | pass | `tests/test_reference_parity.py` (L1) |
| reference parity — streaming loader vs real transformers forward | **OLMoE / Qwen3-MoE / Gemma-4 all pass** (+ hardened rolled-expert control: must match ≥1 module, scheme-agnostic) | `tests/test_reference_parity.py` (L2) |
| full library suite — **GPU** (A2000, stock bnb 0.49.2) | **77 passed, 1 skipped** | `pytest tests/` @ `c5c86f3` |
| full library suite — **CPU** (CUDA masked) | **71 passed, 7 skipped** | `pytest tests/` @ `c5c86f3` |
| CI (ubuntu, CPU torch, ruff + pytest) | green | PR #6 checks @ `ee76d77` |
| ExpertsNbit 8/16-bit schemes — build/forward, LoRA-over-any-scheme, state_dict round-trip, fidelity ordering `bf16 < int8 < nf4` | pass (CPU + CUDA) | `tests/test_nbit_schemes.py` |
| offload is location-not-math (OLMoE `BEFORE` off vs on) *(2026-07-02 stack)* | bit-identical (1.3975 == 1.3975) | `tests/test_offload.py` + `bench/run-offload-ab.sh` |
| non-checkpointed offload training cannot mis-grad silently (fails loudly with the invariant error) | pass | `tests/test_offload.py::test_non_checkpointed_offload_backward_fails_loudly` |

## Offload memory (measured, A2000 12 GB) *(2026-07-02 stack)* — see METHODOLOGY §11

| model | loaded GPU | peak GPU (train) | without offload |
|---|---|---|---|
| OLMoE-1B-7B | 4.70 → **1.08 GB** | 5.97 → **2.57 GB** (−57 %) | fits (5.97 GB) |
| Qwen3-30B-A3B | 3.77 GB | **7.16 GB** | OOM during load |
| Gemma-4-26B-A4B | 5.32 GB | **8.47 GB** | doesn't fit |

Offload cost: **+11 % s/step** (one host→device copy per layer per forward). Memory optimization, not
a speedup.

## Inference decode — big MoE (measured on the header stack, 96 greedy tokens) — see METHODOLOGY §12c

| model | config | tok/s | peak GPU |
|---|---|:---:|:---:|
| Gemma-4-26B-A4B | resident | OOM (expected) | — |
| Gemma-4-26B-A4B | offload, serial | 0.315 | 5.73 GB |
| Gemma-4-26B-A4B | offload + prefetch | **0.429** | 6.16 GB |
| Gemma-4-26B-A4B | offload + prefetch, gemv off | 0.293 | 6.17 GB |
| Qwen3-30B-A3B | resident | OOM (expected) | — |
| Qwen3-30B-A3B | offload, serial | 0.203 | 4.07 GB |
| Qwen3-30B-A3B | offload + prefetch | 0.219 | 4.41 GB |
| Qwen3-30B-A3B | offload + prefetch, gemv off | **0.238** | 4.42 GB |

Source: `bench/run-bigmoe-decode.sh` (driver logs + per-config `.done` flags retained on the
measurement host). §12c scores the pre-registered prediction: 3 of 4 held; the "GEMV a clear win at
128 experts/layer" claim was **falsified on Qwen3-30B** (−8 %; its best config is prefetch +
dequantize). The OLMoE decode grid (3.08 resident / 1.44 prefetched tok/s) is in §12b as committed.
Correctness gates: `tests/test_inference_decode.py`, `tests/test_offload_prefetch.py`.

## Cross-architecture validation — Kaggle Tesla T4 (Turing sm_75), stock bitsandbytes

| claim | value | source |
|---|---|---|
| full library suite on a **second GPU architecture** | **73 passed, 5 skipped** | [`bench/kaggle-t4-validation.ipynb`](bench/kaggle-t4-validation.ipynb) (SHA-pinned to `c5c86f3`, clean clone) on Kaggle T4 |
| environment | Tesla T4 (sm_75) · Python 3.12 · torch 2.10.0+cu128 · **bitsandbytes 0.49.2 (stock)** · transformers 5.0.0 | notebook env cell |

Run on a genuine Turing T4 (`cc (7, 5)`, verified in the env cell — Kaggle's free tier occasionally
assigns an sm_60 P100 that the pinned torch wheel rejects; this run landed a T4). The 4 extra skips
vs the A2000's 77/1 are capability-gated storage schemes (e.g. fp8 blockwise on sm_75) that skip
cleanly, not failures. transformers here is 5.0.0 (the package floor) vs 5.12.1 on the A2000 —
disclosed like the audit-section stack note. The notebook's own overall status shows an error from a
*demo* cell (the README quickstart's in-kernel `import` after an editable `%pip install -e`, which
needs a kernel restart on Kaggle) — the load-bearing `pytest` cell runs in a fresh subprocess and is
the green result recorded above.

## Upstream — bitsandbytes PR #1965 (`ExpertsNbit` / `Experts4bit`)

| claim | value | source |
|---|---|---|
| fork PR branch carries the ExpertsNbit generalization + backward re-dequant this package vendors | `feature/experts-4bit-training` @ `13e74f7` (2026-07-03) | pjordanandrsn/bitsandbytes |
| `tests/test_experts4bit.py` (incl. #1849 regression + shapes) *(2026-07-02 stack)* | **38 passed** CPU + CUDA | run against bitsandbytes 0.49.2 |

Drafts for posting/pushing (Jordan): `outputs/1965_pr_description.md`, `outputs/1965_add_tests.patch`,
`outputs/1849_comment.md` (on the measurement host).

## Falsification audit — unsloth-zoo MoE 4-bit fix (2026-07-02)

Audit of unsloth-zoo's in-tree MoE bnb-4bit solve (their close of unslothai/unsloth#4032), run
before commenting publicly. Full environment and verdict grid:
[`audits/unsloth-zoo-4032/REPORT.md`](audits/unsloth-zoo-4032/REPORT.md) — note its stack differs
from this file's header (transformers 5.5.0 overlay, inside unsloth's declared matrix; torch 2.11
outside their `<2.11` pin, disclosed in the filings). Artifact paths below are relative to that dir.

| claim | value | source |
|---|---|---|
| shipped-fix verdict | **REAL-BUT-PARTIAL** — real-weights Qwen3-30B-A3B slice quantizes, matches stock math, trains expert LoRA | `REPORT.md` grid · `results_qwen3_30b.json` |
| silent transposed-weights math on square dims (bf16 **and** 4-bit) | 139× / 140.7× beyond bf16-noise control; transpose-always fix → excess exactly 1.0 | filed as **unsloth-zoo#849** · `results_issue1_inline_repro.txt`, `results_qwen3_tiny.json`, `diag_rootcause.py` |
| OLMoE `load_in_4bit`: quantized but never routed → first-forward crash | `IndexError` in transformers `_grouped_mm_fallback` (packed 2-D uint8) | filed as **unsloth-zoo#850** · `results_olmoe.json`, `results_olmoe_bare_forward.txt` |
| public record | both issues + verification comment posted 2026-07-02 | unslothai/unsloth#4032, issuecomment-4870034310 |


---

<!-- ots-attestation-footer -->

**OpenTimestamps anchor (self-attestation footer):**

- **OTS proof timestamp for visible document:** `2026-07-04T05:20:59Z` (the moment the current `.ots` was submitted to the calendars; this is the legally operative timestamp for the visible file as published).
- **Disclosed pre-footer content hash:** `25013ba7b3c0568a0032dd152e1259e530c414bccb45facb7cbd86aeda560419` (the SHA-256 of the document *before* this footer was appended — disclosed inside the OTS-anchored visible document for human-readable historical reference; this hash is *not* the payload of the current `.ots` file).
- **Prior disclosed pre-footer hashes (chain, newest first):**
  - `2026-07-02T20:13:51Z` `dbb3e209d7d7892c42e8b1494db9853e7abc2fd8eaf64dfee0e73ff93b43660a`
  - `2026-07-02T11:24:53Z` `6e65e8be7cbf424b190526d112de371f0b87551925eb9e811426b2b46c96622a`
- integrity-attestor glyph (`core.fingerprint`, first 8 bytes of the disclosed pre-footer hash): `[+O.:~@%=@~&.O0*%]`
- Drunken-bishop randomart (full disclosed pre-footer SHA-256, OpenSSH-style):

```
+----[SHA256]-----+
|+..OB+E=.        |
|o.o.+=+. .       |
|. . .+*.o .      |
| . +o+.+.o       |
|  ..=+o.S        |
|   .o..o.        |
|     o.o o       |
|     .= o o      |
|    .oo+....     |
+-----------------+
```

- **Payload hash actually covered by the current `.ots`:** see `ots info PROVENANCE.md.ots`; by construction this is `SHA-256(this entire file including this footer)` and `ots verify PROVENANCE.md.ots PROVENANCE.md` succeeds against the on-disk bytes.
- Anchor file: `PROVENANCE.md.ots`
- Calendars: a.pool.opentimestamps.org, b.pool.opentimestamps.org, a.pool.eternitywall.com, ots.btc.catallaxy.com
- **Provenance posture (load-bearing):** the **OTS proof timestamp** above is the legal anchoring time for the visible document — that is what the calendars witnessed. The **disclosed pre-footer content hash** is *not* anchored by the current `.ots` file; it is *disclosed inside* the OTS-anchored visible document as a human-readable historical record of what the file's bytes hashed to immediately before this footer was appended. A reviewer verifying the visible file runs `ots verify` against the on-disk bytes; a reviewer wanting to confirm the disclosed pre-footer hash recomputes `SHA-256` of the file with everything from `<!-- ots-attestation-footer -->` onward stripped. Both checks are independent; neither replaces the other.
