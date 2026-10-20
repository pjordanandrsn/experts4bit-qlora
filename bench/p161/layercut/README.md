# P161's layer-cut build: the licence half of the plan's witnesses, confirmed by running ($0, RTX A2000, 2026-10-20)

P161 registers its build's witnesses as closed forms (PREREG, "The build's engagement counts"):

    K25 calls:      48 MoE layers x 4 row counts (2, 4, 8, 16) x 5 forwards = 960   (decode record, plan record)
    kernel calls:   960 K25 calls x 2 (gate_up, down) = 1920                         (the shape spy)

with five forwards per row count: the residual licence's two probes, then two warm-ups and one capture. P158's 5090 read
confirmed the 960 for the decode record; this build adds the plan record and the spy.

## What ran

- **The model:** P158's 2-layer cut of Qwen3-30B-A3B, made again by P158's own script
  ([`../../p158/layercut/cut_checkpoint.py`](../../p158/layercut/cut_checkpoint.py): 789 tensors in 2 shards), baked to
  NF4 by P39's `k8_bake.py` (256 expert rows, 0.7 GB).
- **The code:** e4b at this registration's base `dd535cd6` (main, e4b#1877's knob merged; its pyproject string reads
  0.56.0, but the commit is in no release) and grouped-nf4-gemm v0.50.0 at `41ef086e`, torch 2.8.0+cu128, triton 3.4.0,
  on one RTX A2000 (sm_86).
- **The build:** today's default server (`PagedServeConfig.from_env()` + `build_engine`) at `max_seqs` 16, once under
  `E4B_K25_PLAN=today` and once under `c1`, with `E4B_K25_LUT` unset (`build_counts.py`). The shape spy is the box's
  own: `build_counts.py` imports `p161_box` (staged beside it, at the bytes `staged-p161.sha256` pins) and calls its
  `install_shape_spy()` before the build, as the box does.

## What it confirmed

As in P158, decode graphs do not run on sm_86 (the fused FP8 KV append needs sm_89+), so the build makes no warm-up and
no capture. With device grouping set as the graph server sets it (`build_counts.py`, line 13), the residual licence
alone ran:

| plan | decode (`k25_lut`) | decode record | plan record | the spy | per layer |
|---|---|---|---|---|---|
| `today` | `prmt`, env unset, `default-allowlisted` | `{"prmt|le256": 16}` | `{"today|le256": 16}` | `{"32/64/4/3/prmt": 32}` | **8** K25, **16** kernel |
| `c1` | `prmt`, env unset, `default-allowlisted` | `{"prmt|le256": 16}` | `{"c1|le256": 16}` | `{"64/256/4/2/prmt": 16, "128/128/8/3/prmt": 16}` | **8** K25, **16** kernel |

- Every record and the spy were empty before the build.
- `info["k25_plan"]` resolved each plan as named, with its pair: `today` at 32/64/4/3 for both projections, `c1` at
  gate_up 64/256/4/2 and down 128/128/8/3.
- **qwen3_moe's default decode is `prmt` on sm_86 too**, so the box's tripwire and the reducer's decode check read
  the same resolution here as on the 5090.
- The spy read exactly two kernel calls per K25 call, each at its projection's shape: the route hands the kernel the
  plan the record names.
- The T == 1 GEMV made 8 calls (4 per layer, `scalar` on sm_86 where the 5090 takes `bw_prmt32`), as in P158.

The records are `layercut_today.json` and `layercut_c1.json`.

## The capture half

Two warm-ups and one capture per row count cannot run here. It rests on P158's 5090 read through the same capture loop
(`p158-5090-1`): K25's decode record read exactly 960 = 48 x (2 + 3) x 4 in all nine processes. Each K25 call is
counted by both records at the same place in the route, and the spy wraps the kernel those calls reach, so the capture
half carries to the plan record and the spy. That is a carry from another lane's measurement, labelled as such, and the
box's early gates check it on the first two processes.
