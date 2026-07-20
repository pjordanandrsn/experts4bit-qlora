# gpt-oss-20b hybrid-vs-llama same-box A/B — 2026-07-20

The hot/cold hybrid engine's first same-box comparison against llama.cpp's
CPU-MoE mode, on the model the hybrid was built for. Driver: `bench/gptoss_ab_pod.sh`
(ours arm = `bench/bench_gptoss_hybrid.py`), raw receipts in
`bench/receipts-gptoss-ab-20260720/`.

## Box

RunPod SECURE RTX 4090 24 GB (driver 580.159.04), AMD EPYC 7532 (64 threads),
503 GB RAM. torch 2.8.0+cu128, transformers 5.14.1, bitsandbytes 0.49.2,
experts4bit-qlora @ `6f4ab48` (feat/hot-residency-gptoss) + grouped-nf4-gemm
(PyPI). llama.cpp master (depth-1, 2026-07-20), CUDA build, official
`ggml-org/gpt-oss-20b-GGUF` (MXFP4).

## Cells

Decode tok/s, greedy. Ours = house metric (`n_tokens/t_decode`, 128 tokens,
single rep, fresh process per cell); llama = `llama-bench` tg128 avg of 3.

| cell | decode tok/s | peak GPU GB |
|---|---|---|
| llama.cpp `-ngl 99` (full GPU, ceiling) | 265.66 | — |
| llama.cpp `--n-cpu-moe 12` | 20.07 | — |
| llama.cpp `--n-cpu-moe 24` (all expert layers on CPU) | 11.80 | — |
| ours hybrid `HOT_K=8` | 1.33 | 7.79 |
| ours hybrid `HOT_K=4` | 1.21 | 6.46 |
| ours hybrid `HOT_K=0` (all-cold) | 1.12 | 5.17 |

## Reading

- **The pre-registered expectation held**: on a fast-CPU host, llama's
  compute-the-experts-on-CPU beats our stream-them-over-PCIe cold path by
  ~9–10× (11.8 vs 1.1–1.3). The hybrid's niche is weak-CPU hosts, where the
  CPU can't compute experts at rate but PCIe can still stream them (the
  E-axis home-lab rows: ours ≥ llama at K=0 for ≤4-core hosts, 1.5–5.1×).
  This row anchors the fast-CPU end of that regime map — nothing here
  contradicts a published claim, and no "beats llama.cpp" claim exists.
- **The dev gate passed on all three K values**: gpt-oss-20b decodes
  end-to-end through the hot/cold hybrid (24/24 MoE layers patched, coherent
  text, clamped-GLU + per-expert biases live), at 5.2–7.8 GB peak VRAM.
- **HOT_K sweep is monotone** (1.12 → 1.21 → 1.33 tok/s for 0 → 4 → 8 hot of
  32 experts/layer) with untuned hot sets (first-K ids, not routing-informed);
  routing-informed hot sets are the obvious next increment.

## Caveats

Single box, single rep per ours cell, 128-token decode. llama VRAM per
`--n-cpu-moe` cell not captured (llama-bench doesn't emit it; the `.err`
loader logs carry the split). Ours cells ran with all experts resident in
host RAM (`pin=True`) — the 12 GB-card configuration (base weights freed
after the split) is exercised, but a card that can't even *load* all experts
resident (e.g. A2000 12 GB with ~3 GB co-tenants) can't run this driver until
the offload+hot compose increment lands; that attempt OOM'd on the dev box
2026-07-20 (receipt: `gemma_hybrid_gate.attempt1-oom.log` on the A2000 host).

## Ops receipts (what the run cost and broke)

- Two pods: L40S SECURE voided in ~2 min by the CUDA-gate/watcher pattern
  (~$0.04, image "two-pythons" trap — plain `python3` lacks torch); 4090
  SECURE ran the full A/B in ~1.6 h (~$1.10). Both deleted, 404-verified,
  account pods list empty.
- The first ours pass crashed on `ModuleNotFoundError: nf4_grouped` —
  `enable_hot_residency`'s forward runs on the fused kernel, but a
  `[train]`-only install doesn't pull it. Fixed twice over: the pod driver
  installs `.[train,fast]`, and `enable_hot_residency` now fails at enable
  time with an actionable message instead of mid-decode (this commit).
