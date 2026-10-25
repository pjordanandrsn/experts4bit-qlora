# `p225-5090-r3`: a trained attention adapter through the paged server (FAIL, on G3 alone)

One RTX 5090 (cc 12.0, driver 580.178.04, a 600 W power limit, recorded and not floored) on an AMD Ryzen 9 9950X3D host
(32 visible CPUs, CPU quota 15.36, 123 GiB RAM). It ran 2026-10-23 from 15:29Z to 16:28Z through the guarded launcher.
Cost: $1.163.
- **Code:** e4b `0adbba98` (P225 Amendment 3's merge, e4b#2046); grouped-nf4-gemm 0.49.0 `a83f6bb3`; loggetta 0.6.1
  `90a7697d` (P224's stack, pinned).
- **Stack:** torch 2.8.0+cu128 (the image's), transformers 5.17.0.
- **Id:** the run id is the retry label `r3`. The boxes before it did not start: `p225-5090-1` (NOT_RUN),
  `p225-5090-2` and `r2` (HOST_REFUSED, download 20.3 and 29.6 MB/s), `r1` (PRE_INSTALL_HALT, the boot allowance, fixed
  by Amendment 2).

**Before the arms:**
- **The host screen** passed on the box: single-thread speed median 47.41 Mops (floor 30), download 109.2 MB/s
  (floor 60), 123 GiB RAM, 128 GB free. Every phase boundary re-read 45 to 48 Mops with no steal.
- **The guard fit** passed: 21,181 s left against the 20,350 s needed.
- **Self-tests:** `p224_box` 36, `p224_canary` 17, `p224_reduce` 50, `p225_probe` 12, `p225_gate` 11, `p225_reduce`
  16, `p225_canary` 10 checks. The offload test passed (1 passed, 28 deselected).
- **The held-out set:** 1,296 rows, all distinct, none in the training data (canary `91c24ea8...`).

**Training (REF).** 80 of 80 steps (the trainer's record), in 405 s. The adapter's B norm moved from 0 to 69.7, every
value finite, the frozen expert bytes unchanged; the loss fell from a first-third mean of 0.206 to a last-third mean of
0.027. The adapter (192 attention projections, 384 tensors, sha256 `93bc5779...`) passed P225's gate: every registered
target present with nonzero B, provenance on the real file (`gates_p225.json`).

**Scores** (mean loss over the 1,296 rows; the LEARNED gate and G1 grade the assistant positions):

| arm | assistant positions | all tokens |
|---|---:|---:|
| HF0 (the base, eager) | 6.5978 | 6.4233 |
| REF (the trained adapter, the reference path) | 1.3872 | 3.4239 |
| P0 (the base, the paged server) | 6.6065 | 6.4293 |
| PA (the adapter merged, the paged server) | 1.3813 | 3.4094 |

**Cost slots** (W1: one request; W16: sixteen. A is P0, B is PA; two blocks of A1 B A2):

| slot | arm | W1 ttft_any_ms | W1 itl_wall_ms | W1 tok/s | W16 ttft_any_ms | W16 tok/s |
|---|---|---:|---:|---:|---:|---:|
| b0A1 | P0 | 239.117 | 3.8587 | 175.527 | 2483.518 | 334.718 |
| b0B | PA | 236.198 | 3.8361 | 176.692 | 2457.014 | 337.302 |
| b0A2 | P0 | 238.775 | 3.8741 | 175.108 | 2483.155 | 335.009 |
| b1A1 | P0 | 238.613 | 3.8662 | 175.498 | 2477.844 | 335.583 |
| b1B | PA | 236.282 | 3.8737 | 175.764 | 2459.264 | 336.668 |
| b1A2 | P0 | 238.671 | 3.8472 | 175.724 | 2480.943 | 335.140 |

**Verdict: FAIL** (`run/verdict_p225.json`): `G3 block 0 ttft_any_ms W1: |B - mean(A)| = 2.748 > band 2.388`.
RESULTS-p225.md reads it in full.

**Files:**
- **`SHA256SUMS`** covers every file here.
- **`STORE-ONLY.sha256`** lists the files kept only in the private receipt store:
  - the launcher's receipt, guard record, instance record, heartbeat, host facts, known_hosts and teardown proof (they
    name the provider's instance);
  - the adapter's weights and the tokenizer file (never committed; their hashes are also in
    `run/adapter_REF/tensors.sha256`);
  - nine box logs whose progress bars carry carriage returns (`fetch.log`, `ref_REF.log`, `train_REF.log` and the six
    slot logs).
- **Not copied:** the staged scripts and `calib.json`, each byte-identical to the file `staged-p225.sha256` pins at
  `0adbba98`; the nonce markers and the pip constraints file.
- **No identifiers on the box:** no file here names an instance, an offer, a host name or an endpoint.
