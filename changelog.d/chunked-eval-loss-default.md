### `E4B_CHUNKED_EVAL_LOSS` is on by default (TC1 amendment 60); `0` turns it off

- A `torch.no_grad` forward with labels whose fp32 logits would reach 1 GiB now runs without labels, returns the stock logits unchanged,
  and takes its loss from them in 512-token fp32 chunks, by default. TC1 amendment 60 read it on Qwen3-30B-A3B's packed 4,096-token rows,
  on one RTX 5090 in torch 2.12 (`e4b.train.chunked-eval-loss.packed-4k.5090.2026-10-07`): the evaluation-phase peak fell from 26.88 to
  22.50 GB, below the 25.85 GB training phase, so e4b's run peak there is the training phase's.
- **Held-out values compared across this change are not byte-identical above the gate.** The loss differs from the stock one by fp32
  summation order only: on that box step-0 held-out was identical on both draw pairs and held-out at N moved +0.00005. Below the gate
  nothing changes (TC1's field-recipe evaluation rows peak at 411 tokens, 0.23 GiB). Set `E4B_CHUNKED_EVAL_LOSS=0` for the stock loss.
- Scope of the evidence: one model, one RTX 5090, torch 2.12, plus the RTX A2000 correctness and memory receipts of #1302.
