### TC1 amendment 52 registered: bucketed padding on packed rows in the field image's torch 2.8 (P130-P133)

- Token `qwen3padbk28`: amendment 48's packed-row A/B (`NF4_QLORA_PAD_BUCKETS=0` vs `1`, shipped and matched arms, two load-gated draws a
  side) in venv-e4b (torch 2.8, triton 3.4), where amendment 51 read e4b at its defaults 0.739 of its torch-2.12 step. P130 / P131: no slower
  than 1.02; P132: the matched peak falls by at least 3.0 GB; P133: held-out within 0.005. A slow side sends grouped-nf4-gemm's `auto` to a
  gate or a fix.
- `tc1_reduce.py`: the family; `pad_buckets_why` takes the torch prefix; self-test 117 cases.
