# `p232-5090-1`: the int8 lm_head at 2, 4 and 8 rows, served (FLIP_ROWS_8)

One RTX 5090 (cc 12.0, 170 SMs, driver 595.91.07, a **575 W** power limit read on the box before any install) on an
AMD Ryzen 9 7900X host (24 CPUs), 2026-10-24T00:25:44Z to 00:54:31Z, through the guarded launcher. Cost $1.00.
- **Code:** e4b `f7589a02` (the registration with Amendments 1-2, e4b#2085's merge), 0.57.0; grouped-nf4-gemm v0.50.0
  `41ef086e`.
- **Stack:** torch 2.8.0+cu128, triton 3.4.0, transformers 5.17.0, bitsandbytes 0.50.2.
- **Model:** Qwen/Qwen3-30B-A3B @ad44e777, its NF4 arena baked on the box.

**Before the bench:**
- The power floor, the refusals, the download probe (109.9 MB/s against the 40 MB/s floor) and the pre-install fit
  passed (9,793 s left against the 9,200 s needed).
- The tripwire passed, and so did the self-tests (`p232_reduce.py` 114 cases, `p232_box.py` 38, `p109_box.py` 5,
  `p115_quality.py` 11) and the premise, exactly as registered: the real kernels on the card, 11 passed
  (`logs/premise_gpu.log`), and the route's CPU suite, 45 passed (`logs/premise_cpu.log`).

**The run:** the warm-up `arm_Rw.json`, then eight speed arms ABBA twice (`arm_{Ra,Ka,Kb,Rb,Rc,Kc,Kd,Rd}.json`), each
through W2, W4 and W8 at 32 and 160 tokens and five timed reps on the default graph server; then the three quality phases
(`quality_{R,K,B16}.json`) at group T = 2, 4 and 8, 24 wikitext windows x 128 positions each. Every process rc 0.

**Verdict: `FLIP_ROWS_8`, `TRIPWIRE none`** (`verdict_p232.json`):

| T | NLL gap (K - R) | argmax agreement | g_T (min of four pairs) | rules |
|---|---|---|---|---|
| 2 | -0.000091 | 3064 / 3072 | **1.0311** | PASS |
| 4 | -0.000091 | 3064 / 3072 | **1.0243** | PASS |
| 8 | -0.000031 | 3066 / 3072 | **1.0165** | PASS |

Bars: |NLL gap| <= 0.02, argmax >= 0.95, g_T >= 1.01. Self-pairs 0.9925-1.0033 (band 0.985-1.015). The full read is in
[../../RESULTS-p232.md](../../RESULTS-p232.md).

**Files:**
- **What `SHA256SUMS` covers:** every file here.
- **What stays only in the private receipt store** (`STORE-ONLY.sha256`, by path within the run's receipt): the
  launcher's teardown proof and instance record, which name the provider's instance. The launcher's receipt, its ledger
  row and the guard files stay there too.
- **Not copied:** the staged scripts, tests and calibration file, each byte-identical to the file `staged-p232.sha256`
  pins at `f7589a02`; the nonce markers, the run id, the instance id file and the pip constraints file.
- **No identifiers on the box:** no file here names an instance, an offer, a host name or an endpoint.
- **Byte-identical, carriage returns kept:** the arm, quality, fetch and prompts logs carry the bare carriage returns of
  their progress bars (no CRLF line ending anywhere); they are kept as the box wrote them so `SHA256SUMS` matches the store.
