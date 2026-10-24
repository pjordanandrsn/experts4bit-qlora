# P236 results: bitsandbytes 0.50.2 is faster than grouped-nf4-gemm 0.51.0's decode GEMV at all 30 cells, op level and eager (READ)

**Registration:** [PREREG-p236.md](PREREG-p236.md) (e4b#2116), with Amendment 1 (e4b#2122) and Amendment 2 (e4b#2124),
read once as registered. Descriptive: it licenses nothing and moves no default.
**Reading:** `p236-5090-2`, one RTX 5090 (cc 12.0, 170 SMs), e4b `d3b518b3` (Amendment 2's merge), grouped-nf4-gemm
v0.51.0 `30adb1bf`, bitsandbytes 0.50.2 (its published wheel, proven by its bytes), torch 2.8.0+cu128. $0.010.
Records: [receipts/p236-5090-2/](receipts/p236-5090-2/README.md). The verdict re-derives byte-identically from the
records with `p236_reduce.py` at `d3b518b3`.
**Earlier box:** `p236-5090-1` read VOID: the engagement gate refused the split-K cells (Amendment 2 says why). It is
strike 1, and none of its timings is read or used here.
**A registered output is missing:** the registration lists a `SHA256SUMS` among the box's outputs, and the box did not
write one (neither run's records in the private receipt store has it). It is not a verdict input: the verdict was
re-derived from `bench_p236.json` as stored. This PR ships a `SHA256SUMS` it computed over every file it carries, so
`sha256sum -c SHA256SUMS` in the receipt directory checks them on a clean checkout.

## Scope, first

These are **eager** calls between CUDA events, and each one includes its library's Python dispatch, as a caller outside
a captured graph pays it (the registration's own words). experts4bit-qlora's server decodes under captured CUDA graphs,
where that dispatch is not paid. So this read does **not** measure the served decode path, and it does not isolate
grouped-nf4-gemm's K33 kernel time. It reports 30 op-level cells: one projection at a time, one card, two libraries at
their released versions. No default moves, and nothing is claimed beyond the 30 cells.

## The answer: READ

Every cell held the premise (bitsandbytes' `dequantize_4bit` bit-equal to `dequant_ref` on the same bytes), both
correctness tolerances, and engagement as registered: one K33 launch per gnf4 call (with its split-K mark on Mixtral's
down projection) and one `gemm_4bit` dispatch per bitsandbytes call, to one route. Every cell was then timed.

**bitsandbytes is faster at all 30 cells.** The ratio (bitsandbytes' time over grouped-nf4-gemm's, so above 1 would
mean gnf4 is faster) runs from **0.2993** (Mixtral-8x7B gate_up at M = 16) to **0.9176** (OLMoE-1B-7B gate_up at M = 16). No cell is
above 1.

At M = 1, the decode-sized call:

| projection | ratio |
|---|---:|
| Mixtral-8x7B gate_up | 0.8654 |
| Mixtral-8x7B down | 0.5199 |
| Qwen3-30B-A3B gate_up | 0.4938 |
| Qwen3-30B-A3B down | 0.4934 |
| OLMoE-1B-7B gate_up | 0.4949 |
| OLMoE-1B-7B down | 0.4870 |

All 30 cells (median of 20 palindromic reps of 50 calls each; ms per call; self-pair = each arm's second visit over its
first):

| projection | M | ratio (median) | ratio range | gnf4 ms/call | bnb ms/call | gnf4 self-pair band | bnb self-pair band | bnb route |
|---|---:|---:|---|---:|---:|---|---|---|
| Mixtral-8x7B gate_up | 1 | 0.8654 | 0.8510 to 0.8777 | 0.0458 | 0.0399 | 0.9942 to 1.0505 | 0.9767 to 1.0035 | `custom` |
| Mixtral-8x7B gate_up | 2 | 0.6911 | 0.6829 to 0.6939 | 0.0765 | 0.0531 | 0.9968 to 1.0306 | 0.9853 to 1.0155 | `custom` |
| Mixtral-8x7B gate_up | 4 | 0.5858 | 0.5822 to 0.5903 | 0.1482 | 0.0868 | 0.9886 to 1.0023 | 0.9940 to 1.0030 | `custom` |
| Mixtral-8x7B gate_up | 8 | 0.6025 | 0.5748 to 0.6059 | 0.2691 | 0.1618 | 0.9788 to 1.0043 | 0.9944 to 1.0099 | `custom` |
| Mixtral-8x7B gate_up | 16 | 0.2993 | 0.2958 to 0.3086 | 0.5437 | 0.1626 | 0.9908 to 1.0073 | 0.9966 to 1.0111 | `custom` |
| Mixtral-8x7B down | 1 | 0.5199 | 0.5043 to 0.5383 | 0.0433 | 0.0229 | 0.8904 to 1.0196 | 0.9471 to 1.0022 | `custom` |
| Mixtral-8x7B down | 2 | 0.5771 | 0.5705 to 0.5872 | 0.0541 | 0.0317 | 0.9923 to 1.0108 | 0.9671 to 1.0033 | `custom` |
| Mixtral-8x7B down | 4 | 0.5132 | 0.5097 to 0.5188 | 0.0982 | 0.0503 | 0.9973 to 1.0037 | 0.9813 to 1.0040 | `custom` |
| Mixtral-8x7B down | 8 | 0.4848 | 0.4828 to 0.4873 | 0.1805 | 0.0876 | 0.9988 to 1.0014 | 0.9910 to 1.0022 | `custom` |
| Mixtral-8x7B down | 16 | 0.5374 | 0.5350 to 0.5689 | 0.3359 | 0.1807 | 0.9921 to 1.0049 | 0.9989 to 1.0022 | `dequant` |
| Qwen3-30B-A3B gate_up | 1 | 0.4938 | 0.4842 to 0.5067 | 0.0291 | 0.0144 | 0.9608 to 1.0231 | 0.9661 to 1.0205 | `custom` |
| Qwen3-30B-A3B gate_up | 2 | 0.4951 | 0.4842 to 0.5041 | 0.0291 | 0.0144 | 0.9697 to 1.0210 | 0.9575 to 1.0240 | `custom` |
| Qwen3-30B-A3B gate_up | 4 | 0.5005 | 0.4901 to 0.5133 | 0.0292 | 0.0146 | 0.9717 to 1.0135 | 0.9631 to 1.0260 | `custom` |
| Qwen3-30B-A3B gate_up | 8 | 0.5087 | 0.4908 to 0.5253 | 0.0287 | 0.0146 | 0.9755 to 1.0570 | 0.9660 to 1.0103 | `custom` |
| Qwen3-30B-A3B gate_up | 16 | 0.5060 | 0.4969 to 0.5262 | 0.0288 | 0.0147 | 0.9602 to 1.0235 | 0.9628 to 1.0236 | `custom` |
| Qwen3-30B-A3B down | 1 | 0.4934 | 0.4728 to 0.5157 | 0.0284 | 0.0139 | 0.9540 to 1.0254 | 0.9266 to 1.0465 | `custom` |
| Qwen3-30B-A3B down | 2 | 0.4991 | 0.4812 to 0.5202 | 0.0289 | 0.0144 | 0.9639 to 1.0506 | 0.9290 to 1.0170 | `custom` |
| Qwen3-30B-A3B down | 4 | 0.4967 | 0.4772 to 0.5089 | 0.0289 | 0.0144 | 0.9758 to 1.0628 | 0.9704 to 1.0134 | `custom` |
| Qwen3-30B-A3B down | 8 | 0.5017 | 0.4895 to 0.5126 | 0.0283 | 0.0142 | 0.9706 to 1.0222 | 0.9685 to 1.0158 | `custom` |
| Qwen3-30B-A3B down | 16 | 0.5036 | 0.4836 to 0.5219 | 0.0275 | 0.0138 | 0.9545 to 1.0428 | 0.9284 to 1.0363 | `custom` |
| OLMoE-1B-7B gate_up | 1 | 0.4949 | 0.4834 to 0.5089 | 0.0280 | 0.0139 | 0.9779 to 1.0336 | 0.9666 to 1.0169 | `custom` |
| OLMoE-1B-7B gate_up | 2 | 0.4960 | 0.4852 to 0.5038 | 0.0280 | 0.0139 | 0.9690 to 1.0214 | 0.9640 to 1.0149 | `custom` |
| OLMoE-1B-7B gate_up | 4 | 0.4955 | 0.4826 to 0.5058 | 0.0282 | 0.0140 | 0.9843 to 1.0149 | 0.9618 to 1.0151 | `custom` |
| OLMoE-1B-7B gate_up | 8 | 0.4954 | 0.4783 to 0.5081 | 0.0282 | 0.0140 | 0.9443 to 1.0276 | 0.9710 to 1.0296 | `custom` |
| OLMoE-1B-7B gate_up | 16 | 0.9176 | 0.8876 to 0.9517 | 0.0278 | 0.0255 | 0.9657 to 1.0012 | 0.9621 to 1.0250 | `dequant` |
| OLMoE-1B-7B down | 1 | 0.4870 | 0.4663 to 0.4998 | 0.0294 | 0.0143 | 0.9767 to 1.0358 | 0.9453 to 1.0137 | `custom` |
| OLMoE-1B-7B down | 2 | 0.4918 | 0.3723 to 0.5553 | 0.0283 | 0.0139 | 0.5886 to 1.0146 | 0.9296 to 1.0447 | `custom` |
| OLMoE-1B-7B down | 4 | 0.4938 | 0.4842 to 0.5081 | 0.0282 | 0.0140 | 0.9720 to 1.0336 | 0.9557 to 1.0272 | `custom` |
| OLMoE-1B-7B down | 8 | 0.4925 | 0.4837 to 0.5041 | 0.0281 | 0.0138 | 0.9221 to 1.0003 | 0.9672 to 1.0442 | `custom` |
| OLMoE-1B-7B down | 16 | 0.5059 | 0.4691 to 0.5273 | 0.0291 | 0.0147 | 0.8557 to 1.0239 | 0.9485 to 1.0366 | `custom` |

**Shape of the numbers (a reading, not measured):** on the four small Qwen3 and OLMoE projections both arms are flat
across M, grouped-nf4-gemm near 0.028 ms a call and bitsandbytes near 0.014 ms. A cost that does not grow with M or with
the weight's size suggests per-call eager dispatch dominates there. That is a reading of the table's shape; this lane
does not test it. On Mixtral's gate_up, grouped-nf4-gemm's time grows with M (K33 reads the weight once per row), and
the ratio falls from 0.8654 at M = 1 to 0.2993 at M = 16.

## Predictions, graded

| # | prediction | held? |
|---|---|---|
| Q1 | the premise and correctness hold in all 30 cells; K33 engages at every cell; bitsandbytes takes `custom` at M <= 4 (its floor) and its rule decides above | **yes**. bitsandbytes took `custom` at every cell with M <= 4. Above, it took `custom` everywhere except Mixtral-8x7B down at M = 16 and OLMoE-1B-7B gate_up at M = 16, which took `dequant` (reported as observed) |
| Q2 | at M = 1, gnf4 is faster at every projection: ratio between 1.2 and 3 (K33 reached 0.58 to 0.74 of the copy floor in P141's phase 1, and bitsandbytes' one-row kernel is not bandwidth-tuned for these shapes) | **no.** gnf4 is not faster at M = 1 on any projection: the M = 1 ratios run from 0.4870 to 0.8654 |
| Q3 | the ratio falls as M grows, since K33 reads the weight once per row; at M = 16 bnb is faster on the large Mixtral projections (ratio below 1) | **no, as written.** Its M = 16 clause held (bitsandbytes is faster on both Mixtral projections there), but the ratio does not fall with M: only Mixtral's gate_up falls, and not at every step (0.8654, 0.6911, 0.5858, 0.6025, 0.2993 at M = 1, 2, 4, 8, 16); the other projections are near flat, and bitsandbytes is faster at every M |
| Q4 | the self-pair bands lie inside [0.95, 1.05] on the large projections; the small Qwen3 and OLMoE projections, under 20 us a call, show wider bands | **partly.** 16 bands fall outside [0.95, 1.05]. On the Mixtral projections, 3: Mixtral-8x7B gate_up M = 1 gnf4 0.9942 to 1.0505; Mixtral-8x7B down M = 1 gnf4 0.8904 to 1.0196; Mixtral-8x7B down M = 1 bnb 0.9471 to 1.0022. On the small projections, 13, the widest OLMoE-1B-7B down M = 2 gnf4 0.5886 to 1.0146 |
| Q5 | the verdict is READ (about 90 %), VOID (about 10 %) | **READ** (on the rerun; the first box read VOID) |

## What this does not say

- Nothing about the served decode step: e4b's server decodes under captured CUDA graphs, where the Python dispatch these
  eager calls include is not paid.
- Nothing about K33's device time on its own, quality, either library's other routes, other cards, other shapes or M.
- Nothing about bitsandbytes' double-quantized store, which this lane does not read.

## Next (not started)

A device-time arm, graph-captured or read from the profiler's kernel time, registered on its own, would separate
dispatch from kernel. It is noted for a later release and not started.
