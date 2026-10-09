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
