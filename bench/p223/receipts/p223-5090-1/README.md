# `p223-5090-1`: the price of a schedule-invariant decode for Mixtral-8x7B (READ)

One RTX 5090 (cc 12.0, 170 SMs, driver 580.126.18, power limit and enforced limit both 575 W, at the registered floor)
on an AMD EPYC 9334 host (128 CPUs, 755 GiB RAM). It ran 2026-10-23 from 23:51Z to 2026-10-24 00:39Z through the
guarded launcher. Cost: $1.003.
- **Code:** e4b `817acb94` (P223 Amendment 3's merge, e4b#2075); grouped-nf4-gemm v0.49.0 `a83f6bb3`, P223's own pin.
- **Stack:** torch 2.8.0+cu128, triton 3.4.0, transformers 5.17.0, bitsandbytes 0.50.2.
- **Model:** Mixtral-8x7B-Instruct-v0.1 @ `eba92302`, its 32 MoE layers baked to an NF4 arena on the box
  (`work/bake.json`).

**Before the arms:**
- The floors passed: power 575 W enforced (floor 575), the download probe 81.1 MB/s (floor 40,
  `download_probe.json`), and the pre-install fit (11,887 s needed of 12,472 s left).
- The self-tests passed: `p223_reduce.py` 71 cases, `p223_box.py` 19.
- The premise passed on the card: grouped-nf4-gemm's K37 suite (28 passed) and e4b's int8 kernel path (4 passed).

**The arms**, eight fresh processes in two ABBA blocks: D0a I0a I1a D1a D1b I1b I0b D0b. The servers are D0 and I0
(bf16 attention projections) and D1 and I1 (int8 attention projections, `E4B_SERVE_ATTN_INT8=1`). The I arms set
`E4B_ATTN_SPLIT_CAP=1` and `E4B_PAGED_BUCKETS=2,4,8,16`.

| arm | buckets | attention splits per bucket | W16 tok/s | W1 tok/s | load s |
|---|---|---|---:|---:|---:|
| D0a | 1, 2, 4, 8, 16 | the default plan | 321.47 | 124.69 | 45 |
| I0a | 2, 4, 8, 16 | 1 (capped) | 321.17 | 35.53 | 51 |
| I1a | 2, 4, 8, 16 | 1 (capped) | 325.98 | 36.61 | 32 |
| D1a | 1, 2, 4, 8, 16 | the default plan | 326.51 | 142.33 | 36 |
| D1b | 1, 2, 4, 8, 16 | the default plan | 327.81 | 142.48 | 50 |
| I1b | 2, 4, 8, 16 | 1 (capped) | 326.51 | 36.59 | 36 |
| I0b | 2, 4, 8, 16 | 1 (capped) | 319.92 | 35.56 | 44 |
| D0b | 1, 2, 4, 8, 16 | the default plan | 320.21 | 124.69 | 43 |

The default split plans were 12, 12, 11, 5 and 2 at buckets 1, 2, 4, 8 and 16; the I arms ran 1 at every bucket.

**Verdict: READ** (`verdict_p223.json`).
- Both servers are SCHEDULE_INVARIANT: in every I process, each W16 and W1 row's 32-token stream equals the first 32
  tokens of its 160-token stream.
- The price: S0 p16 0.9991, p1 0.2849; S1 p16 0.9960, p1 0.2568.
- Every self-pair is inside its band, and every process is repeatable across its five timed reps.

**Files:**
- **`SHA256SUMS`** covers every file here.
- **`STORE-ONLY.sha256`** lists the files kept only in the private receipt store: the launcher's receipt, guard record
  and log, heartbeat, host facts, known hosts, instance record and teardown proof, and the box's `INSTANCE_ID` (they
  name the provider's instance or host), and ten box logs whose progress bars carry carriage returns (the eight arm
  logs, `fetch.log`, `prompts.log`).
- **Not copied:** the staged scripts, each byte-identical to the file `staged-p223.sha256` pins at `817acb94`; the
  nonce markers, the run-id file and the pip constraints file.
- **No identifiers on the box:** no file here names an instance, an offer, a host name or an endpoint.
