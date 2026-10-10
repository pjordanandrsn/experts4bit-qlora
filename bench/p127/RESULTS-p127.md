# P127 results: the launch-bound glue on the shipped default's single-stream decode (#1313)

## The reading (`p127-5090-1`): **FASTER**

**Registration:**
- P127 itself: #1474, merged as `ee972d23`.
- Amendment 1 (#1488): #1482's fix joins the diff audit as P127, after `p127-prove-1`.
- Amendment 2 (#1511): B is pinned at `7f044dd9`, the harness is the launch commit, and the fetch runs under a
  byte-growth watchdog, after `p127-prove-2`.

**Re-derivation.** The maintainer re-derived the verdict from the receipt store (`b9b2aa4c`) with main's
`p127_reduce.py`, byte-identical to the staged copy, against the SHAs registered in main's `p127_run.sh`. The result is
byte-identical to the box's `verdict.json`: **FASTER**, reasons `[]`.

**What the reading compares: A → B, Phase 1 + Phase 2 against the stack before #1448. It is not Phase 2 alone.**

| arm | e4b | grouped-nf4-gemm | what it is |
|---|---|---|---|
| A | `a8c01d42` (the last main commit before #1448) | `d1f64ba` (v0.44.0) | before P127 |
| B | `7f044dd9` (#1488's merge, pinned) | `d769d502` (#526–#530) | after P127: #1448, #1472, #1477, #1482 |
| M | B | B | B with a round-toward-zero router-weight store: the blindness arm |

- **The harness:** `2332492e` (#1511's merge), carrying #1512's dead-lane detection.
- **The software:** torch 2.8.0+cu128, triton 3.4.0, transformers 5.17.0, bitsandbytes 0.50.2, huggingface_hub
  1.33.0.
- **The model:** `Qwen/Qwen3-30B-A3B` at `ad44e77`, its NF4 arena baked on the box.
- **The card:** one RTX 5090 (cc 12.0, driver 595.91.07, 32607 MiB), on a rented Vast.ai host (machine 153193, Ryzen
  9 9950X, 32 threads, 123 GB RAM).

**Subject:** the shipped default `serve_paged`, with every serving lever and engine knob unset except the two the
registration fixes in every arm: `E4B_PAGED_MAX_SEQS=16` and `E4B_INT4_TILE_PROGRAMS=1`. Every arm captured decode
graphs at buckets 1, 2, 4, 8 and 16 and ran in graph mode.
**Workloads:** P109's W1 (one request) and W16 (16 requests), 512-token prompts (sha256 `21a7e8bd`), 32 and 160 new
tokens, 3 timed passes, decode-only slopes. The arms ran in the palindrome A1 B1 B2 A2, then M1 (identity only), each in
a fresh process.

### The gate: B is bitwise A

| arm | identity record (tokens and per-step logits digests, W1 and W16; sha256 of the record) | timed passes |
|---|---|---|
| A1 | `0b29cfcc26fd6b52` | every pass digests the same |
| A2 | `0b29cfcc26fd6b52` | every pass digests the same |
| B1 | `0b29cfcc26fd6b52` | every pass digests the same |
| B2 | `0b29cfcc26fd6b52` | every pass digests the same |
| M1 | `ddf5266d5bfcda8c` | (identity only) |

- **Identity.** A and B agree token for token and logits digest for logits digest at every one of the 333 identity
  steps.
- **The instrument is live.** The blindness arm M, which changes only the router-weight rounding, differs from A in
  its logits at **100 %** of decode positions.

### Engagement (counted on the box, per arm)

| arm | router weights dtype | q+k rotary | combine residual | gather/div | int64 ids |
|---|---|---|---|---|---|
| A1, A2 | 0 | 0 | 0 | 0 | 288 |
| B1, B2, M1 | 1,200 | 720 | 960 | 240 | 480 |

- **Which paths ran.** B engages every P127 call path. A engages none of the new APIs; its int64 ids are the
  pre-existing path, as registered.
- **The residual licence.** #1477's licence covers **48 of 48** MoE layers at rows 1, 2, 4, 8 and 16 in every B and
  M arm, with no probe errors. The fused residual therefore runs on the served decode graph on sm_120. No owned GPU
  covers that path; #1482's fix is verified there.

### Speed (ruled)

| arm | W1 tok/s | W1 ms / step | W16 tok/s | W16 ms / step |
|---|---|---|---|---|
| A1 | 226.14 | 4.4220 | 967.03 | 16.5455 |
| B1 | 255.91 | 3.9077 | 981.80 | 16.2966 |
| B2 | 255.36 | 3.9161 | 978.70 | 16.3483 |
| A2 | 225.47 | 4.4352 | 963.38 | 16.6082 |

| workload | g = mean of B1/A1 and B2/A2 | the two pairs | saving per decode step |
|---|---|---|---|
| **W1 (one request)** | **1.1321** | 1.1316, 1.1326 | **516.7 µs** |
| W16 (16 requests) | 1.0156 | 1.0153, 1.0159 | 254.4 µs |

- **The self-pairs** sit well inside the NOISY band [0.97, 1.03]: A2/A1 0.9970 (W1) and 0.9962 (W16); B2/B1 0.9979
  (W1) and 0.9968 (W16).
- **The rule.** FASTER requires g1 ≥ 1.03 with both W1 pair ratios above 1. Both hold.
- **What it buys.** The registered step-time saving at one request is half a millisecond. That is about the size of the
  launch-bound routing glue P123 priced: about 635 small launches, 11 % of a 4.75 ms one-request step, so about 0.52
  ms. The arms are all-or-nothing, so this is a size comparison, not an attribution. At 16 requests the same work is a
  smaller share of a longer step.

### Memory

`max_memory_allocated` is 24,237,291,520 B in every arm. P127 adds no persistent memory.

### Against the predictions (written before any data; none gates the verdict)

| | prediction | outcome |
|---|---|---|
| Q1 | g1 in [1.03, 1.12] | **missed high**: 1.1321 |
| Q2 | g16 in [0.99, 1.04] | held: 1.0156 |
| Q3 | M's logits differ at ≥ 90 % of positions, and M's tokens agree with A's on ≥ 50 % of W16's | **missed**: the logits differ at 100 %, but the tokens agree on only 829 of 2,560 (32.4 %) |
| Q4 | the self-pairs within 1 % | held: the widest is 0.38 % |
| Q5 | B's peak memory within 64 MiB of A's | held: identical |

- **Q1.** The gain is larger than registered. The prior priced Phase 2's launches conservatively.
- **Q3.** M is more sensitive than predicted, not less. Once a greedy token flips, the continuation diverges: 15 of
  W16's 16 rows diverge within the 160 tokens (first at positions 6 to 135). W1's single row agrees on all 160. The
  miss strengthens the blindness check rather than weakening it.

## The history

| run | outcome | actual | what it found |
|---|---|---|---|
| `p127-prove-1` | HARNESS_ERROR, rc 27 | $0.827 | B1, B2 and M1 died in `build_engine`: #1477 passed `residual=` to `_HybridTier.forward`, which did not take it. Fixed by #1482; Amendment 1. |
| `p127-prove-2` | HARNESS_ERROR, no exit code | $0.966 | The host rebooted 7.5 minutes into the model fetch, after the audit, tripwire, self-tests and premise had passed. A box fault, with no P127 finding. It led to Amendment 2 (B pinned; the fetch watchdog), #1512 (the driver's dead-lane detection) and adertha-agents#199. |
| `p127-prove-3` | **PROVED** | $0.893 | Every check on the 5090; the maintainer re-derived it clean. |
| `p127-5090-1` | **FASTER** (the reading) | $0.940 | This document. |

**Lane spend:** $3.626 of the $6.00 ceiling. Every run was a single RTX 5090 at ≤ $0.85/h, with teardown proven.

## The consequence (as registered)

- **FASTER:** this PR adds the claims row, with g1 and g16 and their pair intervals, and STATUS records the
  one-request gain.
- **The defaults do not change.** Phase 1 and Phase 2 are already the default and bitwise, and the coming e4b 0.52.0
  (#1516) ships them default-on.

## What this reading cannot say

- **Its scope.** It covers the P127 delta at `7f044dd9`. Later main commits (#1490, #1491, #1498, #1506 and onward) are
  outside it; it is not a claim about whatever main holds at a release.
- **Nothing beyond the subject:** not other families, the int4 stack, `max_seqs` other than 16, or `auto`'s sizing.
- **Nothing about prefill or TTFT** beyond the tokens. Prefill is in both walls and cancels in the slope.
- **Nothing about quality beyond identity.** B is bitwise A.
- **Nothing about which option bought the speed.** The arms are all-or-nothing; a per-option split would be its own
  lane.

## Receipts

In this repository, each with its `SHA256SUMS`, byte-identical to the private receipt store:
- **`receipts/p127-5090-1/`, the reading:** the five arm records, `verdict.json`, `audit.json`, `bake.json`,
  `summary.txt`, `versions.txt`, `forensics.txt`, `prompts.json`, `staged.sha256` and the logs.
- **`receipts/p127-prove-3/`:** the proof, with the same set.
- **`receipts/p127-prove-1/` and `receipts/p127-prove-2/`:** the two harness errors, with what they wrote.

In the private store (adertha-receipts) are `receipt.json` and `teardown-proof.json` for each run:
- `p127-prove-1`: `0627cbd5`;
- `p127-prove-2`: `b04735a3`;
- `p127-prove-3`: `e4c788ff`;
- `p127-5090-1`: `b9b2aa4c`.
