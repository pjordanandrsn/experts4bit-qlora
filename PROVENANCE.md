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

---

<!-- ots-attestation-footer -->

**OpenTimestamps anchor (self-attestation footer):**

- **OTS proof timestamp for visible document:** `2026-07-02T11:24:53Z` (the moment the current `.ots` was submitted to the calendars; this is the legally operative timestamp for the visible file as published).
- **Disclosed pre-footer content hash:** `6e65e8be7cbf424b190526d112de371f0b87551925eb9e811426b2b46c96622a` (the SHA-256 of the document *before* this footer was appended — disclosed inside the OTS-anchored visible document for human-readable historical reference; this hash is *not* the payload of the current `.ots` file).
- integrity-attestor glyph (`core.fingerprint`, first 8 bytes of the disclosed pre-footer hash): `[0?0O?*@?=&@$o+o@]`
- Drunken-bishop randomart (full disclosed pre-footer SHA-256, OpenSSH-style):

```
+----[SHA256]-----+
|        *++.o oo*|
|       +.O.o.+ + |
|      o O...* +  |
|     o + ..o B o |
|  E .   S oo. =  |
|   .   o o+  . o |
|        +o .  o  |
|       +  +      |
|        +o oo.   |
+-----------------+
```

- **Payload hash actually covered by the current `.ots`:** see `ots info PROVENANCE.md.ots`; by construction this is `SHA-256(this entire file including this footer)` and `ots verify PROVENANCE.md.ots PROVENANCE.md` succeeds against the on-disk bytes.
- Anchor file: `PROVENANCE.md.ots`
- Calendars: a.pool.opentimestamps.org, b.pool.opentimestamps.org, a.pool.eternitywall.com, ots.btc.catallaxy.com
- **Provenance posture (load-bearing):** the **OTS proof timestamp** above is the legal anchoring time for the visible document — that is what the calendars witnessed. The **disclosed pre-footer content hash** is *not* anchored by the current `.ots` file; it is *disclosed inside* the OTS-anchored visible document as a human-readable historical record of what the file's bytes hashed to immediately before this footer was appended. A reviewer verifying the visible file runs `ots verify` against the on-disk bytes; a reviewer wanting to confirm the disclosed pre-footer hash recomputes `SHA-256` of the file with everything from `<!-- ots-attestation-footer -->` onward stripped. Both checks are independent; neither replaces the other.
