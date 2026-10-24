# P235 results: P58's same-box vLLM head-to-head at the release pins (e4b 0.58.0 int4 against vLLM 0.32.0 and 0.30.0)

**Registration:** [PREREG-p235.md](PREREG-p235.md) (e4b#2119), read once, as registered.
**Reading:** `p235-5090-1`, one RTX 5090 (470 W power limit, recorded and never ruled) on an Intel Core Ultra 9 285K host,
launch commit `7be9e610` (the registration's merge), $0.86. Records: [receipts/p235-5090-1/](receipts/p235-5090-1/README.md).

## The question

P58's harness, unchanged in its arms, order, alarms, rules and verdict code, re-read at the release pins: e4b's int4
stack from **experts4bit-qlora 0.58.0** and **grouped-nf4-gemm 0.51.0**, against **vLLM 0.32.0** (primary) and
**0.30.0** (secondary, P58's primary) on Qwen's GPTQ-Int4 checkpoint, same box, same session, identical prompt ids. Per
vLLM build and batch size, the position is vLLM graph_r1 / e4b int4_r1 (decode tok/s), quoted only with both sides'
self-pairs inside 1.03x and an equal `prompts_sha`.

## The position (beside P58's, never in place of it)

| B | e4b NF4 control | **e4b 0.58.0 int4** | vLLM 0.32.0 (graph) | vLLM 0.30.0 (graph) | **vLLM 0.32.0 / e4b int4** | vLLM 0.30.0 / e4b int4 |
|---|---|---|---|---|---|---|
| **1** | 6.605 ms, 151.4 tok/s | **3.605 ms, 277.4 tok/s** | 3.474 ms, 287.9 (r2 287.5) | 3.496 ms, 286.0 (r2 286.8) | **1.038** (vLLM ahead) | 1.031 (vLLM ahead) |
| **16** | 20.703 ms, 772.8 tok/s | **9.496 ms, 1684.9 tok/s** | 7.894 ms, 2026.9 (r2 2035.9) | 7.998 ms, 2000.4 (r2 2012.8) | **1.203** (vLLM ahead) | 1.187 (vLLM ahead) |

- **Every arm VALID; no arm skipped; no DRIFT.** Self-pairs, B=1 / B=16: e4b NF4 1.0017 / 1.0011, e4b int4 1.0006 /
  1.0007, vLLM 0.32.0 1.0014 / 1.0044, vLLM 0.30.0 1.0028 / 1.0062, all inside 1.03x.
- **Identical prompts on both engines:** `prompts_sha` `a8e6ea1d…` (B=1) and `f67e7e4d…` (B=16), the digests P58 and P37
  decoded.
- **Same-box build to build:** vLLM 0.32.0 / 0.30.0 = **1.007** at B=1 and **1.013** at B=16.
- **Beside P58's reading** (2026-09-22, vLLM 0.30.0 / e4b 0.36.4 int4): 1.087 at B=1 and 1.396 at B=16. That reading
  was taken on another box (an AMD EPYC 9655 host at 575 W). The two readings are not divided into each other, and the
  difference between them is **not read as** an e4b gain: the box, the host and the power limit all differ.
- The reducer's own sensitivity lines (median-of-3 slopes, the r2 draws) move no ratio by more than 0.004:
  B=1 1.038 / 1.037, B=16 1.207 / 1.202 for vLLM 0.32.0.

## Every distribution proven to be its release

- **Provenance** (`provenance*.json`): experts4bit-qlora 0.58.0, grouped-nf4-gemm 0.51.0, vLLM 0.32.0 and vLLM 0.30.0,
  each a PyPI index install.
- **Release proofs** (`release_proof*.jsonl`, loggetta `bench/release_proof.py` at `f09e5676`): every fast-check and
  every binding proof reads **RECORD equal** with `pycache_fresh` true, in the arms' own import environment.

  | dist | files | wheel pick |
  |---|---|---|
  | experts4bit-qlora 0.58.0 | 102 | filename |
  | grouped-nf4-gemm 0.51.0 | 49 | filename |
  | vLLM 0.32.0 | 5,443 | arch (its wheel is named `manylinux_2_28_x86_64`; its WHEEL says `linux_x86_64`) |
  | vLLM 0.30.0 | 5,097 | arch |

## Conditions (recorded, never ruled)

- **The card's power limit read 470 W, not 575.** The PREREG registers no power floor: the ratio is read inside one box,
  where both engines decode under the same limit.
- **The host:** an Intel Core Ultra 9 285K (P58's box was an AMD EPYC 9655). B=1 is host-bound on both engines.
- **The torch builds:** e4b's arms on torch 2.8.0+cu129 (the image python); each vLLM venv on torch 2.13.0+cu130 (triton
  3.7.1). transformers 5.17.0, bitsandbytes 0.50.1.
- **STOP-1 (informational):** e4b's B=16 step, 9.496 ms, is outside ±8 % of P54's 11.42 ms on its box. At B=1, P54's
  3.693 ms is within ±8 % of today's 3.605 ms.
- **The guard fit:** `GUARD_FIT fit=6060s need=7575s left=10542s`, so the box cleared its pre-install check with 2,967 s
  to spare. The whole run took 48 min from creation to destroy.

## What this read cannot say (the PREREG's list)

- TTFT and prefill (informational only, with no ratio); other batch sizes or context lengths; memory footprint (vLLM
  reserves 0.90 of VRAM by policy); any other checkpoint; `GNF4_GEMV_FUSED_REDUCE=1`; an fp8-KV vLLM arm.
- **The e4b window is not the shipped server's path.** `step_decomp`'s graph-replay window is not the server's
  scheduler.
- **The engine advantage is understated, not inflated.** vLLM's slope includes its scheduler and detokenisation, while
  e4b's window has neither.
- It licenses nothing and moves no default.

## The generated read

The reducer's output, unedited except for line 1 (the box's work directory), is
[`receipts/p235-5090-1/RESULTS.txt`](receipts/p235-5090-1/RESULTS.txt) and
[`RESULTS-p235-generated.md`](receipts/p235-5090-1/RESULTS-p235-generated.md). It was re-derived from the private
receipt store with `p235_reduce.py` at `7be9e610` (byte-identical to the box's copy) and reproduces both files byte for
byte below line 1.
