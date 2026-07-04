# Offload transfer notes — where the H2D bandwidth goes

*Workstream A of the offload-diagnostics investigation. Measured on the RTX A2000 12 GB host that
produces the [`PROVENANCE.md`](../PROVENANCE.md) numbers; reproduce with the env flags at the end.*

Expert CPU-offload ([`experts4bit_qlora/offload.py`](../experts4bit_qlora/offload.py)) streams each
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
that would help most is a wider slot, not a smarter schedule. On an x16 host the same code should see
roughly double these GB/s — untested here, flagged for anyone who runs it. Net: offload is a
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
