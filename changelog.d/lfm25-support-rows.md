### LFM2.5 MoE checkpoints: support rows and topology controls for LFM2.5-8B-A1B and LFM2.5-24B-A2B

- **Support rows, both `reference-ok`:** `bench/support/rows/lfm2_moe_lfm2.5_8b.json` (`LiquidAI/LFM2.5-8B-A1B`,
  revision `dd83f5cb`) and `bench/support/rows/lfm2_moe_lfm2.5_24b.json` (`LiquidAI/LFM2.5-24B-A2B`, `3dc4cd09`).
  `support_probe.py` ran on CPU at e4b `8a4be41d`, with transformers 5.22.2, bitsandbytes 0.50.2 and grouped-nf4-gemm
  0.51.0. Each row loads, verifies NF4 on every MoE layer (22 and 38) and forwards. The rows' `total_params` (4.65 B and
  12.67 B) is what the probe's model holds: packed NF4 expert storage, two weights per byte, plus the bf16 non-expert
  parameters, plus the probe's rank-8 expert LoRA. It is not the checkpoints' 8.47 B and 23.97 B. The existing
  LFM2-8B-A1B row's 4.52 B follows the same rule.
- **Topology controls:** the released `config.json` of both checkpoints at the same revisions, with `SOURCES.json` digests
  and their LFM Open License v1.0 entry in `NOTICE`. Both are `lfm2_moe` with the per-expert `w1`/`w2`/`w3` keys the
  `LFM2_MOE` convention matches. The serve-state refusal test now covers them: `serve_paged` refuses their short-conv
  layers before any weight is read.
- `docs/ARCHITECTURE_SUPPORT.md`'s generated block is regenerated from the rows.
