# `p234-5090-1`: `E4B_NF4_SMALLM_BW=1` on Mixtral-8x7B's default NF4 server (LICENSED)

One RTX 5090 (cc 12.0, 170 SMs, driver 570.133.07, power limit 500 W, recorded and not floored, as registered) on an
Intel Core i9-13900F host (32 CPUs, 94 GiB RAM). It ran 2026-10-24 from 14:16Z to 15:05Z through the guarded launcher.
Cost: $1.023.
- **Code:** e4b `47bf1ed2` (P234's registration merge, e4b#2114); grouped-nf4-gemm v0.50.0 `41ef086e`.
- **Stack:** torch 2.8.0+cu128, triton 3.4.0 (the image's), transformers 5.17.0, bitsandbytes 0.50.2.
- **Model:** Mixtral-8x7B-Instruct-v0.1 @ `eba92302`, its NF4 arena baked on the box (`work/mixtral/bake.json`).
- **Id:** `-1`, the lane's first started box. One earlier pin was absent at the pre-create read, so nothing was created.

**Before the arms:**
- The refusals passed. So did the pre-install fit (17,077 s left against the 16,500 s needed) and the download probe
  (117.1 MB/s against the 40 MB/s floor, `download_probe.json`).
- The self-tests passed: `p234_reduce.py` 91 cases, `p109_box.py` 5, `p234_box.py` 35, `p115_quality.py` 11.
- The three premises passed on the card:
  - e4b's graph-bucket suite: 7 passed (`logs/premise_served.xml`);
  - the kernel premise, `test_nf4_smallm_bw_gpu.py`: 20 passed (`logs/premise_kernel.xml`). Each row at T = 2 to 8 is
    bitwise its own T == 1 row through K33 only;
  - the rows premise (`rows.json`): 6 routing cases, 0 mismatches, K33 launched at T = 1, 2 and 4 in each.

**The arms** (B0a B1a B1b B0b, fresh processes). B0 leaves `E4B_NF4_SMALLM_BW` unset, and B1 sets it to `1`. The figures
are decode-only tok/s:

| arm | W1 | W2 | W4 | W8 | W16 | load s |
|---|---:|---:|---:|---:|---:|---:|
| B0a | 125.91 | 73.32 | 107.36 | 176.03 | 320.02 | 23 |
| B1a | 124.88 | 155.77 | 189.47 | 215.00 | 320.24 | 23 |
| B1b | 124.73 | 155.63 | 189.27 | 214.80 | 319.92 | 23 |
| B0b | 124.69 | 73.10 | 107.24 | 175.74 | 319.88 | 23 |

Every bucket (1, 2, 4, 8 and 16) was captured as a graph in every arm, and each arm read engagement as registered.
Bucket 1 took `nf4_singleton` in every arm. Buckets 2, 4 and 8 took K25 in B0 and the singleton route in B1. Quality
ran at group 4: OFF (B0) in 408 s, ON (B1) in 66 s.

**Verdict: LICENSED** (`verdict.json`).
- **The speed bars:** g2 = **2.12452** and g4 = **1.76481** (bar 1.03). The inert W1 check reads 0.99182 (bar 0.99). The
  W16 regression bar reads 1.00013 (bar 0.99). W8's 1.22138 is reported only.
- **The tokens:** each arm repeats itself exactly, and B1's tokens equal B0's on W1.
- **Quality:** within P110's bars on both texts at group 4.

B1-versus-B0 token differences at W2 to W16 are reported by row in `verdict.json` (`token_diffs`), never a rung.

**Files:**
- **`SHA256SUMS`** covers every file here.
- **`STORE-ONLY.sha256`** lists the files kept only in the private receipt store:
  - the launcher's receipt, guard record and log, heartbeat, host facts, known hosts, instance record and teardown
    proof, and the box's `INSTANCE_ID`, which name the provider's instance or host;
  - nine box logs whose progress bars carry carriage returns: the four arm logs, `fetch.log`, `fit.log`, `prompts.log`
    and the two quality logs;
  - `forensics.txt` and `outer.log`, which carry the card's GPU UUID.
- **Not copied:** the staged scripts (each byte-identical to the file `staged-p234.sha256` pins at `47bf1ed2`), the nonce
  markers, the run-id file, the start marker and the pip constraints file.
- **No identifiers on the box:** no file here names an instance, an offer, a host name or an endpoint.
