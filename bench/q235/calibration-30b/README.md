# Q235 Amendment 1: the 30B calibration (receipts)

**When and where.** 2026-10-10, 15:21–16:40 UTC, on one RTX A2000 12 GB (sm_86) and its host CPU, correctness only. No timing
here is a claim.

**Environment.** torch 2.11.0+cu128, transformers 5.19.0, bitsandbytes 0.50.2, grouped-nf4-gemm 0.45.0. The model is
Qwen/Qwen3-30B-A3B at revision ad44e777 (index, config and shards verified against the hub).

**Tools.** `bench/q235/instrument/`: `calib30.sh` drives `q235_tokens.py`, `layer_stream_ref.py`, `nf4_side.py` and
`q235_compare.py`.

**Inputs.** 8 rows of TC1's field text. The data file reproduces TC1's registered sha256 (5324987a…); the token tensors' sha256 are in
`tokens.pt.json`. The rows hold 1,014 real tokens, right-padded to 331.

| file | what |
|---|---|
| `ref6_resident_cuda.reference.json` / `ref6_stream_cuda.reference.json` | first 6 decoder layers in bf16, resident against layer-streamed, on the card |
| `ref6_resident_cpu.reference.json` | the same 6 layers in bf16 on the host CPU, 6 intra-op threads |
| `ref48_stream_cuda.reference.json` | all 48 layers, layer-streamed bf16, on the card |
| `nf4_run1.nf4.json` / `nf4_run2.nf4.json` | e4b's NF4 training forward, experts in pinned host RAM, two fresh processes |
| `cmp_*.json` | each comparison over the real tokens: per-layer relative error, router top-8 set agreement and overlap, loss gap |

Each record carries its per-layer output sha256. The per-layer tensors themselves are not committed.
