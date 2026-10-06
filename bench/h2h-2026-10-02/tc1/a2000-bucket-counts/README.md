# grouped-nf4-gemm's bucketed LoRA delta: kernel counts under torch 2.8 and torch 2.11 (RTX A2000, diagnostic)

The count cited in TC1 amendment 53's *Why* ([`../../../tc1/TC1-PREREG.md`](../../../tc1/TC1-PREREG.md)). It is a diagnostic, not a registered
measurement, and it licenses nothing. Script: [`../../../tc1/bucket_count.py`](../../../tc1/bucket_count.py), run on 2026-10-06 against
grouped-nf4-gemm `1b2b7bc` (`kernel/` from a `git archive` of that commit).

- [`counts-torch2.8.json`](counts-torch2.8.json): torch 2.8.0+cu128, triton 3.4.0.
- [`counts-torch2.11.json`](counts-torch2.11.json): torch 2.11.0+cu128, triton 3.6.0.

The shape is Qwen3-30B-A3B's packed rows: 4,096 tokens × top-8 of 128 experts with Zipf(0.6) routing, r 16. One iteration is a forward and
backward of both projections (gate_up 2048 → 1536, down 768 → 2048). The padded path is forced (`NF4_QLORA_LORA_PATH=padded`), because
under `auto` the 2 GiB pad-byte rule sends calls this skewed to the per-expert loop. A first pass at Zipf(1) went to the loop on every
call and was discarded. Counts are per iteration from `torch.profiler` over six iterations after warm-up.

| torch | adapters | single block: kernels / launch calls | buckets: kernels / launch calls | sync-like | cudaMalloc + cudaFree |
|---|---|---|---|---|---|
| 2.8 | fp32 | 66 / 64 | 115.3 / 113.3 | 0.33 both | 0 both |
| 2.8 | bf16 | 65.2 / 64 | 112.5 / 111.7 | 0.33 both | 0 both |
| 2.11 | fp32 | 66 / 64 | 115.3 / 113.3 | 0.33 both | 0 both |
| 2.11 | bf16 | 65.2 / 64 | 113.7 / 111.7 | 0.33 both | 0 both |

The 0.33 sync-like calls per iteration are the single `cudaDeviceSynchronize` that closes the profiled window. The JSON files also hold
host and GPU milliseconds. The A2000 is a correctness-only testbed, so those are not readings and are not quoted here.

What it shows, on sm_86:

- **Across torch versions:** with fp32 adapters the counts are identical in torch 2.8 and torch 2.11. With bf16 adapters the buckets' launch
  calls are identical (111.7) and the kernel count differs by 1.2 (112.5 against 113.7).
- **Across adapter dtypes:** fp32 and bf16 are close but not equal. The buckets run 115.3 kernels / 113.3 launch calls with fp32 adapters
  against 112.5–113.7 / 111.7 with bf16.
- **Neither torch** adds synchronisation or allocation to the bucketed delta.

What it cannot show is sm_120 (the RTX 5090), where amendment 52's traces put the idle time. That is amendment 53's box.
