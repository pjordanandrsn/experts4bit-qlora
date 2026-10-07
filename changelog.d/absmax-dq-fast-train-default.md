### `enable_fast_train` stores the frozen expert absmax double-quantized by default (TC1 amendment 56)

- `enable_fast_train` now applies `compress_expert_absmax_` after patching, as the CLI trainer already does for resident training.
  - On Qwen3-30B-A3B's packed 4,096-token rows it costs 0.2 % of the step (1.002) and saves 1.35 GB, held-out within 0.0002.
  - Evidence scope: one model on one RTX 5090 in torch 2.12 / triton 3.7. Its packed-row speed under the field image's torch 2.8 is
    unread. Row `e4b.train.absmax-dq.packed-4k.5090.2026-10-07`.
- It has the same guards as the trainer:
  - off under `OFFLOAD_EXPERTS=1` or `TRAIN_ARENA`;
  - a model the compressor refuses (offloaded, arena or 8-bit storage, a stack another engine patched) keeps its fp32 absmax, with the
    reason in `FAST_TRAIN_STATS["absmax_dq"]`;
  - nothing is compressed when nothing was patched.
- To keep the fp32 absmax, which the offload, batched, residency and NVMe engines read, set `E4B_ABSMAX_DQ=0` or call
  `enable_fast_train(model, absmax_dq=False)`. `E4B_ABSMAX_DQ=1` or `absmax_dq=True` makes a refusal an error. The compression is lossy
  and stays after `disable_fast_train`.
- The refusal those engines give a compressed model now names the default and the way to keep fp32.
- The test suite pins the default off in-process (`tests/conftest.py`), so tests written against the fp32 absmax keep their meaning.
  `tests/test_absmax_dq.py` tests the default itself.
- TC1 receipts now record the absmax as it is (`absmax_dq`), the arm's flag (`absmax_dq_flag`) and the default's record
  (`absmax_dq_fast_train`).
