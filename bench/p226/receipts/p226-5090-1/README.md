# `p226-5090-1`: the K25 plan at Qwen3-30B-A3B's bucket 32 (NO_GO)

One RTX 5090 (cc 12.0, 170 SMs, driver 615.71.09, a **575 W** power limit read on the box before any install) on an AMD
EPYC 9654 host (384 CPUs, 755 GiB RAM), 2026-10-23T07:41:51Z to 08:02:57Z, through the guarded launcher. Cost $0.305.
- **Code:** e4b `4913f959` (the registration's merge, #2012, with amendment A2); grouped-nf4-gemm v0.50.0 `41ef086e`.
- **Stack:** torch 2.8.0+cu128, triton 3.4.0 (the image's), transformers 5.17.0, bitsandbytes 0.50.2.

**Before the bench:**
- The power floor, the refusals and the pre-install fit passed (7,637 s left against the 7,350 s needed).
- The self-tests passed (`p154_route.py` 24 cases, `p226_capture.py` 8, `p226_bench.py` 13, `p226_reduce.py` 21,
  `p136_box.py` 5), and so did the premise (K25's row-exact GPU tests, 4 passed).
- Qwen3-30B-A3B @ad44e777 was fetched and its NF4 arena baked on the box (`bake.json`).

**The capture** (`capture.json`, `prompts_p226.json`): 32 distinct prompts (wikitext rows 16..47, 512 prompt tokens),
128 decode steps at bucket 32 over 48 layers. Distinct experts per layer step: mean 74.8 (55 to 103).

**The bench** (`bench_p226.json`) ran rc 0 in 573.4 s: selection 467.3 s, confirmation 102.8 s, against the registered
allowance of 2,170 s.
- **Selection:** P156's 144-plan grid, per projection, on 8 steps. All 144 plans ran in each projection, and every one
  was bit-identical to c1's outputs.
- **Confirmation:** 64 held-out steps over C, C2, K1-K3 and D, in Williams order with spacer replays. No tile faults, and
  every arm bit-identical to C on every rep.

**Verdict: NO_GO** (`verdict_p226.json`):

| arm | plan (gate_up / down) | median C/arm over 64 reps | SE(median) | lower bound |
|---|---|---|---|---|
| C2 (self-pair) | c1 / c1 | 1.0001 | 0.0003 | 0.9995 |
| **K1** | 64/256/8/3 / 128/128/8/4 | **0.9676** | 0.0003 | **0.9669** (clock-adjusted 0.9644) < 1.05 |
| K2 | 64/256/4/3 / 32/256/4/2 | 0.9643 | 0.0004 | 0.9636 |
| K3 | 32/256/2/2 / 128/128/4/2 | 0.9797 | 0.0003 | 0.9792 |
| D (reported) | 32/64/4/3 / 32/64/4/3 | 1.7712 | 0.0018 | |

Plans read block_n/kc/warps/stages; c1 is 64/256/4/2 / 128/128/8/3. c1 ran 7.883 ms a step for both GEMMs over 48 layers.

**Headroom (reported):** the byte floor of the captured distinct experts is 6.384 ms a step (3,605 distinct experts a
step over the layers, at P141's 1,498.9 GB/s; the box measured 1,524.6 GB/s), so c1 runs at 81.0 % of the copy rate,
with at most 1.50 ms a step left to any plan.

**Files:**
- **What `SHA256SUMS` covers:** every file here.
- **What stays only in the private receipt store** (`STORE-ONLY.sha256`, by path within the run's receipt):
  - the launcher's receipt, its ledger row, and the instance and guard files, and the teardown proof, which names the
    provider's instance;
  - the routing capture's tensor file (`capture.pt`, the picks the bench timed; `capture.json` carries its sha256);
  - three box logs whose download progress bars carry carriage returns (`capture.log`, `fetch.log`, `prompts.log`).
- **Not copied:** the staged scripts, each byte-identical to the file `staged-p226.sha256` pins at `4913f959`; the nonce
  markers and the pip constraints file.
- **No identifiers on the box:** no file here names an instance, an offer, a host name or an endpoint.
