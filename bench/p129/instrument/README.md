# P129's instrument (RTX A2000; correctness and counts only)

1. `make_model.py`: the random two-layer Qwen3-MoE at Qwen3-30B-A3B's layer dimensions (`P129_MODEL_DIR`).
2. `p129_check.py`: launches and Python calls per training step, today's path against `E4B_TRAIN_FUSE_QKV=1`, plus the dequantize's
   equality. Writes `counts.json`.
3. `p129_proj.py`: the fused projection in isolation against the three `LoRALinear` modules. Writes `projection.json`.
4. `p129_floor.py CFG SEED`: one 30-step run, for CFG in B, F1, F2, F3 and FUSED. Gives one `runs.jsonl` line, plus the first step's
   gradients under `P129_OUT`.
5. `p129_g0.py`: the first step's gradients compared. Writes `grads_step1.json`.

`../p129_reduce.py` re-derives Amendment 1's verdict from `../records/a1`. The records were made with e4b `590962a` and
grouped-nf4-gemm `e21a712`; the environment is in `../records/a1/env.json`.

- `p129_step0.py` (Phase 2 read): the step-0 held-out loss of the real Qwen3-30B-A3B at the pin under six ways to compute the q/k/v base
  projections (stock, fp32, q split in two, one concatenated matmul, the fused module's q/k/v in the stock forward, the fused module with
  serving's forward), and a per-layer localization on one held-out row. `P129_MODEL_DIR` names a local copy checked against the
  revision's sha256s; the first argument is TC1's tokens file. Records: `records/a2/`.

- `p129_floor_check.py` (Amendment 3's harness): on the two-layer model set up as TC1's e4b arm, with `lora_B` zero and non-zero, checks
  `tc1_arm.qkv_floor_rows`. Every mode covers every row; A0 reproduces the stock rows exactly; D3 shares k's matmul with v on every call;
  the bases come back. It also reports the envelope numbers for the fused path. Record: `records/a3/floor_harness_2layer.json`.

- `p129_routing.py` (report-only, after Amendment 3): on the real Qwen3-30B-A3B at the pin, the router's top-8 expert set per token and
  layer at step 0 for the stock path, q split in two and the fused path. Then the fused path with every layer's router output replaced by
  the stock path's, which shows how much of the step-0 shift routing flips carry. Record: `records/a3/routing_step0.json`.
