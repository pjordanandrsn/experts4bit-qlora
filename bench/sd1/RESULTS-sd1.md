# SD1 results: speculative decoding, Phase 0 (#1313)

**τ and draft acceptance below are MEASURED on e4b's target, from captured hidden states. S is a MODELLED B = 1
speedup -- P123's verify-cost model under independent routing, with D(n) bracketed and the #1469 census not yet
landed -- and NOT a measured end-to-end decode speed.** Phase 1 builds and measures the real verify path.

## The verdict (`sd1-5090-2`): **PROCEED_EAGLE3**

**The re-derivation.** The maintainer re-derived it from the receipt store (`5d07e055`) with main's `sd1_reduce.py`
(self-test 14/14, byte-identical to the staged copy). The `verdict.json` is identical to the box's, with reasons `[]`.

**The registration:**
- SD1 itself: #1534, merged as `92b3a8de`.
- Amendment 1 (#1542, merged as `e29c1af3`): the head's size check follows the cache symlink, and the batched attention
  is shared.

**Under test:**
- **The target:** e4b `7f044dd9` + grouped-nf4-gemm `d769d502` (P127's B), `Qwen/Qwen3-30B-A3B` at `ad44e77`, NF4 arena
  baked on the box. It was built as the shipped server builds it (16 slots, `E4B_INT4_TILE_PROGRAMS=1`), eager for capture.
- **The head:** `RedHatAI/Qwen3-30B-A3B-speculator.eagle3` at `6afc5aa2` (apache-2.0, verifier `Qwen/Qwen3-30B-A3B`),
  `model.safetensors` sha256 `d2d6e2e6…`, checked on the box. Its draft runs in `sd1_eagle3.py` (vLLM's EAGLE-3
  inference semantics).
- **The card:** one RTX 5090 (vast machine 53317, cc 12.0), on harness `e29c1af3`.

**Captures:**
- R: 16 rows × (512 + 160);
- C-think: 16 × (20–1,200 + 256), `enable_thinking=True`;
- C-nothink: 16 × (24–1,204 + 256), `enable_thinking=False`.

Every row accounted for all its positions, with no extra forward. Peak memory was 22.8–23.2 GiB.

**The capture path is the served path.** R's row 0, run alone and eager, equals `p127-5090-1`'s graph-mode W1 tokens
**160 / 160**.

### PREMISE

EAGLE-3 τ(k = 1) on C-think, the head's own training mode, is **1.7157 ≥ 1.40**. It sits inside the card's 1.66–1.84,
which clears both the reimplementation and the auxiliary-layer convention (layers 2 / 24 / 45).

### EAGLE-3 (measured τ and acceptance; modelled S)

| workload | k | τ (tokens / verify step) | draft acceptance | S, independent | S, reuse 0.444 |
|---|---|---|---|---|---|
| R | **1** | 1.5353 | 0.5389 | **1.0964** | 1.1970 |
| R | 2 | 1.8094 | 0.4087 | 1.0163 | 1.1572 |
| R | 3 | 1.9479 | 0.3193 | 0.9092 | 1.0566 |
| C-think | 1 | 1.7157 | 0.7174 | 1.2253 | 1.3377 |
| C-think | **2** | 2.1912 | 0.5981 | **1.2303** | 1.4009 |
| C-think | 3 | 2.4939 | 0.5011 | 1.1635 | 1.3521 |
| C-think | 5 | 2.8314 | 0.3703 | 1.0052 | 1.1793 |
| C-nothink | **1** | 1.6465 | 0.6497 | **1.1758** | 1.2837 |
| C-nothink | 2 | 2.0339 | 0.5205 | 1.1423 | 1.3006 |
| C-nothink | 3 | 2.2604 | 0.4237 | 1.0550 | 1.2259 |

The full k = 1…5 table for every workload is in `receipts/sd1-5090-2/verdict.json`.

- **The decision.** The best S under independent routing is R 1.0964 (k = 1), **C-think 1.2303 (k = 2)** and
  **C-nothink 1.1758 (k = 1)**. That is ≥ 1.15 on two of three workloads: PROCEED_EAGLE3.
- **Longer drafts lose under independent routing.** Every workload peaks at k = 1–2. Each extra position adds about 7
  distinct experts per verify, which grows faster than τ.

### The n-gram floor (measured τ; modelled S)

| workload | best k | τ | S, independent |
|---|---|---|---|
| R | 1 | 1.2231 | **1.0844** |
| C-think | 1 | 1.0897 | 1.0275 |
| C-nothink | 1 | 1.0586 | 1.0061 |

- **Where it helps.** Prompt lookup helps only on raw text that repeats its own context. On chat it buys almost nothing.
- **Against the floor.** R's τ(1), 1.2231 on the box's B = 1 tokens, agrees with the floor registered from
  `p127-5090-1`'s W16 tokens (1.2201).

### Against the predictions (all held)

| | prediction | outcome |
|---|---|---|
| Q1 | C-think τ(1) in [1.60, 1.90] | held: 1.7157 |
| Q2 | R τ(1) in [1.25, 1.65], below C-think | held: 1.5353 < 1.7157 |
| Q3 | C-nothink τ(1) in [1.40, 1.80] | held: 1.6465 |
| Q4 | the best EAGLE-3 k by S is 1–3 everywhere | held: 1, 2, 1 |
| Q5 | n-gram τ(1) on R within ±0.03 of 1.2201 | held: 1.2231 |
| Q6 | the R tripwire equal | held: 160 / 160 |

## The history

| run | outcome | actual | what it found |
|---|---|---|---|
| `sd1-5090-1` | HARNESS_ERROR, rc 11 | $0.751 | The head check's `stat` measured the cache symlink, not the file (its sha256 matched). Review then found that the batched attention would have needed about 50 GB. Both were fixed by Amendment 1, with a real-shape test and a symlinked-blob dry run. |
| `sd1-5090-2` | OK: **PROCEED_EAGLE3** | $0.737 | This document. |

**Lane spend:** $1.488. Each run was a single RTX 5090 at ≤ $0.85/h, with teardown proven.

## The consequence (as registered)

**PROCEED_EAGLE3.** Phase 1 builds the EAGLE-3 verify path in `serve_paged` and gets its own registration. The PREREG
names its parts:
- the verify path, with KV rollback and graphs;
- a quality gate, because the T > 1 verify path is not bitwise the T == 1 path;
- a timed reading of the real verify step, in place of the model's price.

No claims row is added here: S is modelled. A claim waits for Phase 1's measured decode speed.

## What this reading cannot say

- **Nothing about a measured speedup.** The verify step is priced by P123's census model, not timed. Expert reuse across
  a verify is literature (0.444) until #1469 measures it.
- **Nothing about quality,** sampling (temperature > 0), batch > 1, or other families.
- **Not the shipped serving path's acceptance.** τ is the acceptance of this head against the eager capture path, which
  equals the served path's tokens on R row 0.

## Receipts

In this repository, each with its `SHA256SUMS`, byte-identical to the private receipt store:
- **`receipts/sd1-5090-2/`:** the three captures (tokens and 5-token chains at every index), `verdict.json`, the prompts,
  `summary.txt`, `versions.txt`, `forensics.txt`, `staged.sha256` and the logs.
- **`receipts/sd1-5090-1/`:** the harness error.

In the private store (adertha-receipts) are `receipt.json` and `teardown-proof.json` for each run: `sd1-5090-1` is
`ae152849`, and `sd1-5090-2` is `5d07e055`.
