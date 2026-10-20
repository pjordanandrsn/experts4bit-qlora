# P158's layer-cut build: the licence half of the build counts, confirmed by running ($0, RTX A2000, 2026-10-20)

P158 registers two closed forms for the build (PREREG, "The build's engagement counts"):

    K25:            48 MoE layers x 4 row counts (2, 4, 8, 16) x 5 forwards = 960
    bandwidth GEMV: 48 MoE layers x 1 row count  (1)            x 5 forwards x 2 calls = 480

with five forwards per row count: the residual licence's two probes, then two warm-ups and one capture.

## What ran

- **The model:** Qwen3-30B-A3B cut to its first 2 layers as a real checkpoint (`cut_checkpoint.py`: the non-layer tensors
  and layers 0-1 only, re-sharded, with the config's `num_hidden_layers` at 2). A config-only cut is refused by e4b's
  loader, rightly (an unplaceable layer-10 tensor, e4b#529). The 2 layers are baked to NF4 by P39's `k8_bake.py`, as the
  box bakes the full model (256 expert rows, 0.7 GB).
- **The code:** e4b at #1862's head (`d6ae964c`; its library code is byte-identical to `263040a3`, #1862's merge) and
  grouped-nf4-gemm v0.50.0's `kernel/`, torch 2.8.0+cu128, on one RTX A2000 (sm_86).
- **The build:** today's default server (`PagedServeConfig.from_env()` + `build_engine`) at `max_seqs` 16, once under
  `E4B_K25_LUT=tree` and once under `prmt` (`build_counts.py`), reading K25's record and the dispatch tally after the
  build.

## What the A2000 cannot run, and what it confirmed

- **Decode graphs do not run on sm_86.** `E4B_PAGED_GRAPHS=1` refuses: the fused FP8 KV append casts to e4m3 in Triton,
  which needs sm_89+. Under the default (`auto`) the graphs are off, so the build makes no warm-up and no capture, and
  without them the server leaves device grouping off: that build read no K25 call at all.
- **With device grouping set as the graph server sets it** (`hot_residency.DEVICE_GROUPING`, `FORCE_SINGLETON_GROUPS`;
  `build_counts.py`, line 8), graphs still off, the residual licence alone ran:

  | decode | K25 record after the build | per layer | T == 1 GEMV calls | per layer |
  |---|---|---|---|---|
  | `tree` | `{"tree|le256": 16}` | **8** | 8 (`scalar` on sm_86) | **4** |
  | `prmt` | `{"prmt|le256": 16}` | **8** | 8 (`scalar` on sm_86) | **4** |

  That is the licence half, measured: 2 forwards x row counts 2, 4, 8, 16 = 8 K25 calls per layer, under the arm's
  decode only; 2 forwards x 2 calls = 4 GEMV calls per layer at T == 1. (sm_86 takes the `scalar` GEMV where the 5090
  takes `bw_prmt32`; the count of calls is the same.) The records are `layercut_tree.json` and `layercut_prmt.json`. Their
  `matches_closed_forms` field compares against the FULL graph build's 20 and 10 per layer, which the A2000 cannot run,
  so it reads `false`; the licence half they measure is 8 and 4.

## The capture half

Two warm-ups and one capture per row count (3 forwards) cannot run here. It rests on a prior real-box measurement:
P153's 5090 build (`p153-5090-1`, the same server and capture loop) read `bw_prmt32` = 480 at T == 1, which is 48 x
(2 licence + 3 capture) x 2. It is labelled a measurement on another lane's box, not a derivation and not this lane's run.

## The box's gates

Both closed forms are checked on the box before the bulk of the run: after the warm-up process (S0w, the tree; rc 43)
and after the first prmt process (S1a; rc 44), the runner stops if the build counts are not exactly 960 under the arm's
decode only and 480. Each is an instrument refusal, not a strike; the same gate firing on two consecutive runs counts as
one. The reducer's VOID on the same counts stays as the backstop.
