# P68 rehearsal on the NAS RTX A2000 — NOT A READING

These receipts show the forcing arms, the size reading and the reducer running end to end on real CUDA. They were
also read against the predictions before registration; `../P68-PREREG.md` says which predictions they changed.
**They are not P68's reading.** The reading is Qwen3-30B-A3B on an RTX 5090.

## Setup

- **Box.** The NAS RTX A2000 12 GB (sm_86, 26 SMs, driver 575.64.05), shared with production services, one job at a
  time under `/share/Container/gnf4-interp/a2000.lock` (`scripts/gpu_job.sh`).
- **Image and software.** `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`: torch 2.8.0+cu128, Triton 3.4.0,
  transformers 5.16.1, bitsandbytes 0.50.1. grouped-nf4-gemm is main `68a1250`, and experts4bit-qlora is this
  branch's tree, editable (`gpu.txt`).
- **Model.** OLMoE-1B-7B-0924 from the LAN store: 16 layers, 64 experts, top-8, hidden 2048. It uses P63's baked
  arena.
  - OLMoE's attention is not fused (`fuse_qkv_n` 0).
  - **The round-2 attention fold does not apply to it:** `fuse_t1_glue_r2_n` is [16, 0], meaning 16 layer folds and
    0 attention folds. That is the difference from Qwen3 that matters below.
- **Size rows.** Rehearsal-only (`rows.json`): OLMoE-tokenized prose from this repository's P63–P65 pre-registrations,
  4 × 512. P64's committed rows are Qwen3 ids.
- **Script:** `scripts/rehearse.sh`.

## Receipts

- `out/<stack>/p68_arm.json.gz` — the probe's receipts.
- `RESULTS-rehearsal-generated-draft-tables.md` — the reducer on the box. It is read against the FIRST DRAFT's
  prediction tables, since the tables changed after this run.
- `read-registered.md` and `p68_rep.json` — the current reducer over the same receipts, with the registered tables.
- `run_<stack>.log.txt` — the probe's logs.

## What it read (OLMoE on the A2000: not a reading)

- **G0 held on both stacks.**
  - Every forced arm's control equalled the unforced control bit for bit, and the control repeated.
  - Every forcing engaged. For example, int4 `hf.all` made 576 projection, 144 core, 144 router, 9 LM-head and 457
    norm split calls.
  - No T = 1 attention call carried a mask, so the per-row core call is decode's call.
- **The ablation located the difference.**
  - With either the projections or the attention core forced alone, layer-0 attention still differs.
  - With both forced, the first difference moves out of layer-0 attention: to layer 0's MLP (the router's multi-row
    `F.linear`), or on nf4 partly to layer 1.
  - **With everything forced (`hf.all`), every verify position on both stacks is bit-exact to T = 1 decode: 132 of
    132 per stack.** The prefill is bit-exact too: 160 of 160.
- **The int4 prefill under `hf.all` was exact here, against the registered Qwen3 prediction of not exact.** OLMoE has
  no attention fold, so its prefill and its decode take the same upstream attention. Qwen3 has 48, and above 64 rows
  its fold falls through to a different arithmetic (see the registration, Q5). Left as registered.
- **Size, served default:** every cell WITHIN-BAR.
  - KL mean 8e-4 to 5e-3 nats/token.
  - Top-1 0.950–0.982, with the lower CI bound ≥ 0.9419.
  - 952 / 896 / 2,048 positions per cell.
- **Time:** about 45 s per 512-token row per size arm. The whole int4 stack took ~15 min, nf4 ~6 min.
