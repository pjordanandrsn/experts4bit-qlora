# #1526: the grouped_nf4 MoE backward in the training estimate

Probe receipts for the estimate's grouped_nf4 backward branch. **In sample**: the branch was set against these four
points, and `tests/test_grouped_nf4_backward_estimate.py` pins estimate >= measured on them.

- **Model:** `q3red-L4`, a reduced Qwen3-30B-A3B. It has 4 layers and random weights (`make_reduced.py`), with the real
  hidden size, heads, vocabulary, 128 experts and top-8. The revision is the one TC1 recorded.
- **Run:** `probe.py` builds it the way TC1's e4b arm does: streaming NF4 load, NF4 attention, non-reentrant
  checkpointing, fp32 attention LoRA and `enable_fast_train(dgrad=True)`. Adapters are fp32, or native (bf16 expert
  LoRA).
- **Training:** one packed row per micro-batch (random ids, labels = ids), grad accum 4, AdamW8bit, 3 steps. The peak
  is the steady steps 2–3.
- **Box:** one RTX A2000 12 GB, with experts4bit-qlora 0.52.0, grouped-nf4-gemm 0.45.0, torch 2.12.1+cu129,
  transformers 5.18.0 and bitsandbytes 0.50.2.
- **Attribution:** `attribute.py` replays each run's allocator history (`torch.cuda.memory._record_memory_history`)
  to the peak and groups the live blocks by their first frame. The output is `receipts/attributions.txt`.

| receipt | tokens / micro-batch | padding | adapters | allocated peak | estimate before | estimate after |
|---|---|---|---|---|---|---|
| L4-T4096-fp32 | 4,096 | packed | fp32 | 4,760.6 MiB | 4,004.7 (1.189) | 4,980.4 (0.956) |
| L4-T4096-native | 4,096 | packed | native | 3,902.5 MiB | 3,800.4 (1.027) | 4,232.2 (0.922) |
| L4-T2048-fp32 | 2,048 | packed | fp32 | 3,890.7 MiB | 3,964.7 (0.981) | 4,007.4 (0.971) |
| L4-T2048-native | 2,048 | packed | native | 3,690.7 MiB | 3,760.4 (0.981) | 3,760.4 (0.981) |

Ratios are allocated peak / estimate. Before = 0.52.0, after = this change.
