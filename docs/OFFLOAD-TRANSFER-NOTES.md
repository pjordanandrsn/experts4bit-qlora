# Offload transfer notes — where the H2D bandwidth goes

*Workstream A of the offload-diagnostics investigation. Measured on the [`PROVENANCE.md`](../PROVENANCE.md)
host — an RTX A2000 12 GB in a NAS-class host (Xeon W-1250, PCIe 3.0-only platform), the card in
the chassis's sole x8 electrical slot, full production stack live (load ~45, GPU shared with six other
dev-agent containers — not quiesced). Reproduce with the env flags at the end.*

*Testbed note (2026-10-05): the A2000 is the project's correctness-only testbed (testbed policy, standing since
2026-07-27). The bandwidths and timings below are the record of what that shared host read. They are not speed
evidence for any target card, and nothing here sets a band or a default.*

Expert CPU-offload ([`experts4bit_qlora/engines/offload.py`](../experts4bit_qlora/engines/offload.py)) streams each
layer's frozen NF4 experts host→device per forward. Its throughput is bounded by one number — the
pinned-PCIe H2D bandwidth of the host — and this note pins that number, checks the observed per-layer
figures against it, and reports whether copy consolidation buys anything. **It does not: the link is
the wall, and each copy already saturates it.**

Two default-off diagnostics drive it:

- **`E4B_OFFLOAD_STATS=1`** — CUDA-event-bracketed per-copy timing (bytes, copy count, mean ms,
  implied GB/s) split by policy (`sync` train/serial · `cold_miss` · `prefetch`), plus prefetch
  stall vs slack and cold-miss count. Reported in `infer.py`'s BENCH path and at the end of
  `train.py`; a one-shot PCIe-link + H2D-ceiling report prints at load (`report_offload_environment`).
- **`E4B_OFFLOAD_ARENA=1`** — packs a layer's four home tensors into one contiguous pinned arena per
  dtype, so a stage issues **2** H2D copies instead of **4** (one uint8 packed-weights arena + one
  float32 absmax arena). Bit-identical to the four-copy path (`test_offload_stats_arena.py`); it only
  tests whether per-copy fixed overhead is worth removing.
  **It costs host RAM:** torch's pinned allocator rounds every request up to a power of two, and a per-dtype arena rounds the
  packed weights of a whole layer at once. On Qwen3 (30B, 235B) and Mixtral 8x22B a layer's homes then pin **1.78x** their bytes,
  against 1.33x for the four separate homes. Do not turn it on to save memory.
- **`E4B_OFFLOAD_PIN_SLAB=1`** (default off) — pins a layer's four homes as views into **one exact-size, page-aligned pageable
  buffer**, registered in place with a single `cudaHostRegister`, so the power-of-two rounding never applies: the homes cost their
  bytes, plus each home's 512-byte alignment and at most a page. Staging is unchanged (the same four copies from the same views). If the registration is refused the
  homes are pinned per tensor as before, and the handle's `slab_why` says why (`pin_mode` names the path). The pages are
  unregistered before the buffer can be freed. Not combinable with `E4B_OFFLOAD_ARENA`.

  Pinned bytes per offloaded layer, four separate homes (today's default) against the slab, from each family's shapes (NF4
  packed weights at 0.5 B/param, fp32 absmax per 64, as offloaded stacks keep it):

  | family | separate homes | slab |
  |---|---:|---:|
  | Qwen3-235B-A22B, Qwen3-30B-A3B, Mixtral 8x22B, Granite 4.0-H tiny | 1.333x | ~1.00x |
  | Mixtral 8x7B, LFM2-8B-A1B | 1.143x | ~1.00x |
  | ERNIE-4.5-21B-A3B, Granite 3.1 3B-A800M, Gemma 4 26B-A4B, gpt-oss-20b | 1.01-1.07x | ~1.00x |
  | OLMoE-1B-7B, Qwen3.6-35B-A3B | 1.000x | ~1.00x |
  | Nemotron 3 (non-gated experts) | 1.52-1.68x | ~1.00x |

  On Qwen3-235B-A22B that is 42.6 GB of pinned host RAM over 127.7 GB of expert bytes (94 layers). Registered pageable memory
  counts as anonymous memory (`RssAnon`) where `cudaHostAlloc` memory counts as shared (`RssShmem`), so read both when comparing.

  Two things to know about slab homes:
  - **`Tensor.is_pinned()` reads False on them.** torch recognizes only its own allocator's blocks. The pages are page-locked all
    the same: on an RTX A2000 a `non_blocking` copy from a slab enqueues in about 0.1 ms and runs at the rate of torch's own
    pinned memory. Use the handle's `pinned` (or `offload._is_pinned`), which counts live slabs.
  - **The pipelined engine reads them in place or refuses.** The slab is registered mapped (`cudaHostRegisterPortable |
    cudaHostRegisterMapped`), and records whether the device reads it at its host address (`cudaHostGetDevicePointer`). Where it
    does, the pipelined engine's zero-copy gather takes slab homes like torch-pinned ones. Where it does not, or the in-place path
    cannot engage for any other reason, enabling pipelined residency raises: it never pins a second copy of the layer, since the
    flag exists to save that memory.

## The ceiling — this host is PCIe-limited, not software-limited

| path | measured H2D | notes |
|---|:---:|---|
| pinned, 256 MB × 20 | **6.16–6.18 GB/s** | the number every per-layer figure below is read against |
| pageable, 256 MB × 20 | 4.5–5.5 GB/s | why `OFFLOAD_PIN=0` is correct-but-slow |

`nvidia-smi` link report on this host, **under load**: `gen 3/3, width 8/16` — i.e. PCIe 3.0 at an
**electrical x8** width (of an x16-capable card; at idle the gen reads 1 via ASPM and upshifts). A
PCIe-4.0 x16 link ceils near 20–24 GB/s; 3.0 x8 lands right at the observed ~6.2 GB/s. So the
"~5 GB/s prefetched vs 20+ expected" puzzle from the original handoff was an **x16 assumption on x8
hardware** — there is no missing 4×. The load-time report prints a `WARNING: link negotiated below
max` line so this is never a silent surprise.

## Per-layer observed vs ceiling (OLMoE-1B-7B, 216 MiB/layer, r16 adapter)

Measured with `E4B_OFFLOAD_STATS=1`. Copies run at the **pinned ceiling** whether there are 4 per
stage or 2, and whether prefetched or serial — the bus is saturated per copy:

| policy | copies/stage | mean ms/stage | implied GB/s | % of 6.16 ceiling |
|---|:---:|:---:|:---:|:---:|
| prefetch, arena off | 4 | 36.82 | 6.15 | 100 % |
| prefetch, arena on | 2 | 36.74 | 6.16 | 100 % |
| serial, arena off | 4 | 36.78 | 6.16 | 100 % |

Prefetch stall vs slack over the 96-token decode (arena off): **28.6 s stall vs 3.4 s slack**, 0 cold
misses. Decode is so deeply transfer-bound that even one-layer-ahead prefetch leaves the compute
stream mostly *waiting* on the bus — a layer's compute (a few ms) cannot hide a 36.8 ms transfer.
Prefetch's win is real but bounded by exactly that: it overlaps the little compute there is, nothing
more.

## Arena A/B — copy consolidation is a measured non-win here

| config (96-tok decode, stats on) | copies over run | tok/s | mean ms/stage |
|---|:---:|:---:|:---:|
| offload + prefetch, arena **off** | 6208 | 1.614 | 36.82 |
| offload + prefetch, arena **on** | 3104 | 1.606 | 36.74 |

Halving the copy count (4→2 per stage) moved per-stage time by **0.08 ms** and tok/s by less than
noise. Per-copy fixed/launch overhead is negligible relative to the ~36.8 ms each stage spends
actually moving 216 MiB across an x8 link at the ceiling. **`E4B_OFFLOAD_ARENA` therefore stays
default-off** — it is correct and costs nothing, but on a bandwidth-bound host it buys nothing;
it is kept as a measured result and a lever for a host where per-copy overhead *does* dominate
(many tiny experts, or a much faster link).

*(The tok/s here are lower than METHODOLOGY §12b's shipped grid — 1.44 prefetch / 0.40 serial —
because `E4B_OFFLOAD_STATS=1` brackets every copy with CUDA events, perturbing absolute throughput.
The two arena arms are internally comparable; §12b, measured stats-off, remains the authoritative
decode grid.)*

## Conclusion

The bottleneck is the **PCIe 3.0 x8 electrical link (6.16–6.18 GB/s pinned ceiling)**, not per-copy
overhead and not a pinning fallback: each layer's staging copy already runs at 100 % of that ceiling.
Copy consolidation (A3) does not move the needle — a measured non-win, kept default-off. Prefetch
hides only the small per-layer compute behind the transfer (stall ≫ slack), so on this host the lever
that would help most is a different host, not a smarter schedule — this chassis has no wider slot to
give. The identical build on a Gen3 x16 host should see ~2× these GB/s, a Gen4 x16 host ~3.5×
(untested here, flagged for anyone who runs it). Net: offload is a
**capacity** feature (it decides what *fits/generates*), transfer-bound by the bus; the diagnostics
now make that quantitative instead of inferred.

## Reproduce

```bash
# Decode, stats on, arena off vs on (OLMoE + r16 adapter):
OFFLOAD_EXPERTS=1 PREFETCH=1 BENCH_TOKENS=96 E4B_OFFLOAD_STATS=1 R=16 ALPHA=32 \
  ADAPTER=./out/adapter_best.pt python -m experts4bit_qlora.infer
OFFLOAD_EXPERTS=1 PREFETCH=1 BENCH_TOKENS=96 E4B_OFFLOAD_STATS=1 E4B_OFFLOAD_ARENA=1 R=16 ALPHA=32 \
  ADAPTER=./out/adapter_best.pt python -m experts4bit_qlora.infer

# Training A/B third arm (see bench/run-offload-ab.sh):
E4B_OFFLOAD_STATS=1 OFFLOAD_EXPERTS=1 STEPS=5 python -m experts4bit_qlora.train
```
