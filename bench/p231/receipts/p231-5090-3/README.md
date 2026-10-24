# `p231-5090-3`: `E4B_NF4_SMALLM_BW=1` on Mixtral-8x7B's default NF4 server (FUNCTION_FAIL)

One RTX 5090 (cc 12.0, 170 SMs, driver 595.84, power limit 500 W, recorded and not floored, as registered) on an AMD
EPYC 7402P host (48 CPUs, 125 GiB RAM). It ran 2026-10-24 from 02:58Z to 03:50Z through the guarded launcher. Cost:
$1.086.
- **Code:** e4b `fa40c752` (P231 Amendment 3's merge, e4b#2086); grouped-nf4-gemm v0.50.0 `41ef086e`.
- **Stack:** torch 2.8.0+cu128, triton 3.4.0 (the image's), transformers 5.17.0, bitsandbytes 0.50.2.
- **Model:** Mixtral-8x7B-Instruct-v0.1 @ `eba92302`, its NF4 arena baked on the box (`work/mixtral/bake.json`).
- **Id:** `-3`, the box Amendment 3 added. `-1` (rc 9) and `-2` (rc 25) are the two earlier started boxes; their kit faults are
  Amendments 2 and 3.

**Before the arms:**
- The refusals passed, and so did the pre-install fit (17,164 s left against the 16,500 s needed) and the download
  probe (154.6 MB/s against the 40 MB/s floor, `download_probe.json`).
- The self-tests passed: `p231_reduce.py` 87 cases, `p109_box.py` 5, `p231_box.py` 30, `p115_quality.py` 11.
- The premise passed on the card: e4b's graph-bucket suite (7 passed, `logs/premise_served.xml`), and the rows premise
  (`rows.json`: 6 routing cases, 0 mismatches, K33 launched at T = 1, 2 and 4 in each).

**The arms** (B0a B1a B1b B0b, fresh processes; B0 leaves `E4B_NF4_SMALLM_BW` unset, B1 sets it to `1`):

| arm | W1 tok/s | W2 tok/s | W4 tok/s | W8 tok/s | W16 tok/s | load s |
|---|---:|---:|---:|---:|---:|---:|
| B0a | 123.92 | 73.53 | 107.72 | 175.52 | 318.59 | 50 |
| B1a | 123.86 | 154.15 | 186.25 | 212.67 | 319.33 | 51 |
| B1b | 124.29 | 153.96 | 186.66 | 212.13 | 319.31 | 51 |
| B0b | 124.50 | 73.47 | 107.68 | 175.68 | 319.11 | 48 |

Every bucket (1, 2, 4, 8, 16) captured as a graph in every arm. Quality ran at group 4: OFF (B0) and ON (B1).

**Verdict: FUNCTION_FAIL** (`verdict.json`). B1's tokens differ from B0's on the W16 control, in both pairs:
- at 32 tokens, rows 3, 4, 6, 10 and 13;
- at 160 tokens, rows 0, 1, 2, 3, 4, 6 and 7.

Each arm repeats itself exactly (B1b's tokens equal B1a's and B0b's equal B0a's), and W1's tokens are equal across the
arms. The timing and quality figures are reported, never graded.

**Files:**
- **`SHA256SUMS`** covers every file here.
- **`STORE-ONLY.sha256`** lists the files kept only in the private receipt store:
  - the launcher's receipt, guard record and log, heartbeat, host facts, known hosts, instance record and teardown
    proof, and the box's `INSTANCE_ID`, which name the provider's instance or host;
  - nine box logs whose progress bars carry carriage returns: the four arm logs, `fetch.log`, `fit.log`, `prompts.log`
    and the two quality logs.
- **Not copied:** the staged scripts, each byte-identical to the file `staged-p231.sha256` pins at `fa40c752`, and the
  nonce markers, the run-id file, the start marker and the pip constraints file.
- **No identifiers on the box:** no file here names an instance, an offer, a host name or an endpoint.
