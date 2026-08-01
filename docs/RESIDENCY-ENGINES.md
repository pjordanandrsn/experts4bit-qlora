# Residency engines — pipelined, v0 hot, cold, and NVMe

Moved out of the README (2026-08-01) so the landing page states *which door to use* and
this page states *why and when*. Nothing here is new; the measurements and dates are as
they were.

Four ways to place expert weights, all sharing one idea: keep the experts you actually
route to close, stream or compute the rest.

| engine | hot experts | cold experts | needs |
|---|---|---|---|
| `enable_pipelined_residency` | resident VRAM | streamed from pinned host RAM | `[fast]` |
| `enable_hot_residency` | resident VRAM | streamed from pinned host RAM | `[fast]`, **superseded** |
| `enable_cold_engine` | resident VRAM | **computed on the host CPU** | — |
| `enable_mxfp4_nvme_residency` | resident VRAM | streamed from an on-disk arena | `[fast]` + a baked arena |

## Both host-RAM engines need standalone expert modules (2026-07-28)

`enable_pipelined_residency` raises `NotImplementedError` when every `ExpertsNbit` it finds
is an `ExpertsLoRA.base`, and `enable_hot_residency` *silently skips* those modules
(returning a lower patch count). `load_moe_4bit_streaming` always wraps in `ExpertsLoRA`,
so the offload/streaming-loader path — the one a "serve past VRAM" reader is most likely on
— does not reach either engine. Load bare (`Experts4bit`/`ExpertsNbit`, no LoRA wrapper) to
use them. Composing offload with residency is a library increment, not a supported
configuration.

`enable_mxfp4_nvme_residency` **refuses** rather than skipping, for the same reason plus a
sharper one: under the arena loader an adapter's base buffers are on `meta`, so it could not
run even if it were reached.

## Pick the hot sets from a routing histogram, not by index

This is the single largest lever and the easiest to get wrong. Measured 2026-07-20
(`bench/RESULTS-informed-hotsets.md`), decode gain tracks routing coverage on every
architecture tried **at those link speeds**: gpt-oss-20b K=4 informed **+56%** / K=8
**+120%** over the all-cold floor (naive ids `0..K-1`: ±0%), Gemma-4-26B K=8 **+44%**
(informed top-8 is 6% of 128 experts yet covers half of all routed selections), OLMoE +19%.

Re-confirmed 2026-08-01 on full DeepSeek-V4-Flash (43L × 256E, 147 GB arena, A5000,
`hot_rows` at the k=6 decode floor so the tier's own cache cannot mask the effect):

| config | median s/tok | tok/s | VRAM |
|---|---|---|---|
| pure stream | 1.2159 | 0.82 | 8.92 GiB |
| 8 hot **by index** | 1.2180 | 0.82 | 13.28 GiB |
| 8 hot **informed** | **0.8884** | **1.13** | 13.28 GiB |

**+37.1%** informed over by-index, against a 9.3% worst-case within-config spread. The
sharper finding is the middle row: index-ordered hot sets are statistically identical to
pure streaming — 4.4 GiB of VRAM for nothing. Coverage explains it exactly (informed 26.0%
of routed slots, by-index 3.2%, uniform 3.1%): **an index-ordered hot set is a uniform
random draw.**

`experts4bit_qlora.expert_profile` builds the histogram (`E4B_EXPERT_PROFILE=out.jsonl`);
`hot_sets_from_profile(path, H)` ranks by `tokens_routed` and `coverage_from_profile`
scores a candidate set without re-running the model.

**Two cautions.** Coverage predicts *read reduction*, not speed — it only converts when
reads bind. And check the tier's own cache size against the arena before attributing
anything to the VRAM hot set: an early V4 run showed no difference purely because
`hot_rows=64` against a 128-row arena had the tier holding half of everything.

## The size of the gain is a property of the HOST (2026-07-28)

A hot expert is only worth holding when the transfer it avoids costs more than the resident
path's own overhead, so the dial pays where the bus is the bottleneck and washes out where
it is not: **+40%** on a thin-link A2000 (gpt-oss, K=4), **≈0%** on a fat-PCIe L40S (same
model, informed K=8 ≈ pure streaming). On an A6000 with a 128-expert model the informed
cells did not replicate at all and were withdrawn as evidence. Treat the numbers above as
measured on their hosts — not as a floor to expect on a fat-link box.
`HOT_MODE=informed bench/bench_gptoss_hybrid.py` is the calibrate-then-pin reference driver.

Two regime laws from the same receipts (`bench/RESULTS-gptoss-hybrid-ab.md`): the hybrid
wins where the host CPU is weak and VRAM is small — on a strong-CPU server, llama.cpp-style
CPU compute of the cold experts is ~an order faster than PCIe streaming — and on
multi-socket hosts **pin the process affinity** (`taskset` was worth 6.9× on our
cold-stream decode and 3.2× on llama.cpp's CPU-MoE in the same measurements). The partition
is math-identical to the reference forward (both stacks decode the same NF4 values through
the same kernel; correctness-gated in the suite).

## Cold engine — the other side of that law

`enable_cold_engine` keeps the same hot partition on the GPU but **computes** the cold tail
on the host from CPU-resident NF4, so per-token traffic is activation-sized, never
weight-sized (the `--n-cpu-moe` regime at expert rather than layer granularity). The host
decode is bit-exact against bitsandbytes' CPU `dequantize_4bit` and backend-selected around
its AVX2 cliff: `dequant="auto"` takes bnb's AVX-512 kernel only where `avx512f` is present
and otherwise a pure-torch decode (on AVX2-only hosts bnb silently falls back below even
naive torch — grouped-nf4-gemm `bench/cold-engine/` receipts). An all-cold configuration
(`hot_sets` of empties, `device="cpu"`) is a pure-host MoE and needs neither CUDA nor
`[fast]`.

```python
from experts4bit_qlora import enable_cold_engine
enable_cold_engine(model, hot_sets, device="cuda", dequant="auto")
```

## NVMe arena — when the experts do not fit host RAM either

For models whose expert store exceeds RAM, the cold tier reads from a baked on-disk arena
instead. DeepSeek-V4-Flash is the worked example: 147 GB of experts served against 8.4 GiB
of resident dense weights. See [DEEPSEEK-V4.md](DEEPSEEK-V4.md).

Reads are `O_DIRECT` where the platform allows it, so page cache neither helps nor is
needed — host RAM does not have to exceed the arena.
