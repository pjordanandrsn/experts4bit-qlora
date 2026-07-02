# Provenance — measured evidence, stamped

Every figure traces to a committed script/test and the hardware it ran on. The offload A/B and the
26–35 B fit results are detailed in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) §11; this file stamps
the full set with environment + commit for a single point of reference.

## Environment

- **Hardware:** RTX A2000 12 GB (Ampere, cc 8.6). Real 26–35 B checkpoints staged from 80 GB host RAM.
- **Versions:** Python 3.11.2 · torch 2.11.0+cu128 · bitsandbytes 0.49.2 · transformers 5.12.1 ·
  peft 0.19.1 · accelerate 1.14.0 · trl 1.7.0 · datasets 5.0.0 · safetensors 0.8.0
- **Date:** 2026-07-02  ·  **Repo commit:** `2461e48` (@ main, local)

## Correctness

| claim | value | source |
|---|---|---|
| reference parity — primitive vs float SwiGLU-MoE ref (+ swap/drop controls, orientation) | 2/2 pass | `tests/test_reference_parity.py` (L1) |
| reference parity — streaming loader vs real transformers forward | **OLMoE / Qwen3-MoE / Gemma-4 all pass** (+ rolled-expert control) | `tests/test_reference_parity.py` (L2) |
| full library suite | **20 passed** | `pytest tests/` |
| offload is location-not-math (OLMoE `BEFORE` off vs on) | bit-identical (1.3975 == 1.3975) | `tests/test_offload.py` + `bench/run-offload-ab.sh` |

## Offload memory (measured, A2000 12 GB) — see METHODOLOGY §11

| model | loaded GPU | peak GPU (train) | without offload |
|---|---|---|---|
| OLMoE-1B-7B | 4.70 → **1.08 GB** | 5.97 → **2.57 GB** (−57 %) | fits (5.97 GB) |
| Qwen3-30B-A3B | 3.77 GB | **7.16 GB** | OOM during load |
| Gemma-4-26B-A4B | 5.32 GB | **8.47 GB** | doesn't fit |

Offload cost: **+11 % s/step** (one host→device copy per layer per forward). Memory optimization, not
a speedup.

## Upstream — bitsandbytes PR #1965 (`Experts4bit`)

| claim | value | source |
|---|---|---|
| rebase onto bnb `main` | clean, no conflicts | local `bnb-pr` @ `2748d76` (bnb main `8ab26f7`) |
| `tests/test_experts4bit.py` (incl. #1849 regression + shapes) | **38 passed** CPU + CUDA | run against bitsandbytes 0.49.2 |

Drafts for posting/pushing (Jordan): `outputs/1965_pr_description.md`, `outputs/1965_add_tests.patch`,
`outputs/1849_comment.md`.
