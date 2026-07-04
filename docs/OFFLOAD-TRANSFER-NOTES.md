# Offload transfer notes — where the H2D bandwidth goes

*Workstream A of the offload-diagnostics investigation. Measured on the RTX A2000 12 GB host that
produces the [`PROVENANCE.md`](../PROVENANCE.md) numbers; reproduce with the env flags below.*

Expert CPU-offload ([`experts4bit_qlora/offload.py`](../experts4bit_qlora/offload.py)) streams each
layer's frozen NF4 experts host→device per forward. Its throughput is bounded by one number — the
pinned-PCIe H2D bandwidth of the host — and this note pins that number, checks the observed per-layer
figures against it, and reports whether copy consolidation buys anything.

Two default-off diagnostics drive it:

- **`E4B_OFFLOAD_STATS=1`** — CUDA-event-bracketed per-copy timing (bytes, copy count, mean ms,
  implied GB/s) split by policy (`sync` train / `cold_miss` / `prefetch`), plus prefetch stall vs
  slack and cold-miss count. Reported in `infer.py`'s BENCH path and at the end of `train.py`. A
  one-shot PCIe-link + H2D-ceiling report prints at load (`report_offload_environment`).
- **`E4B_OFFLOAD_ARENA=1`** — packs a layer's four home tensors into one contiguous pinned arena per
  dtype, so a stage issues **2** H2D copies instead of **4** (one uint8 packed-weights arena + one
  float32 absmax arena). Correctness is unchanged (`test_offload_stats_arena.py` asserts bit-identity
  arena-on vs off); it only tests whether per-copy fixed overhead is worth removing.

## The ceiling — this host is PCIe-limited, not software-limited

| path | measured H2D | notes |
|---|:---:|---|
| pinned, 256 MB × 20 | **6.20 GB/s** | the number every per-layer figure below is read against |
| pageable, 256 MB × 5 | 2.53 GB/s | why `OFFLOAD_PIN=0` is correct-but-slow |

`nvidia-smi` link report on this host: **`pcie.link.width.current = 8`** (of `max 16`); gen reads 1
at idle (ASPM) and upshifts under load. **The card is in an electrical x8 slot.** A PCIe-4.0 x16 link
ceils near 20–24 GB/s; x8 halves that, and 6.2 GB/s is exactly the x8-with-overhead regime. So the
"only ~5 GB/s prefetched, ~1.4 GB/s serial vs 20+ expected" puzzle from the original handoff was an
**x16 assumption on x8 hardware** — there is no missing 4×. Prefetched decode already runs at ~80 %
of this host's real ceiling.

## Per-layer observed vs ceiling (OLMoE-1B-7B, 216 MiB/layer)

Fill from `E4B_OFFLOAD_STATS=1` runs — headline: `sync` (training/serial) and `prefetch` GB/s, both
compared to the 6.20 GB/s ceiling; arena-on vs off at the same policy.

| policy | copies/stage | mean ms/stage | implied GB/s | % of 6.20 ceiling |
|---|:---:|:---:|:---:|:---:|
| serial (arena off) | 4 | [MEAS] | [MEAS] | [MEAS] |
| serial (arena on) | 2 | [MEAS] | [MEAS] | [MEAS] |
| prefetch (arena off) | 4 | [MEAS] | [MEAS] | [MEAS] |
| prefetch (arena on) | 2 | [MEAS] | [MEAS] | [MEAS] |

Prefetch stall vs slack (arena off / on): [MEAS] ms stall, [MEAS] ms slack, [MEAS] cold misses.

## Decode grid — arena A/B (OLMoE, 128 greedy tokens)

| config | tok/s | peak GPU |
|---|:---:|:---:|
| offload + prefetch (arena off — shipped 0.2.0) | 1.44 | 1.68 GB |
| offload + prefetch (arena on) | [MEAS] | [MEAS] |
| offload, serial (arena off) | 0.40 | 1.45 GB |
| offload, serial (arena on) | [MEAS] | [MEAS] |

## Conclusion

*(one paragraph, filled after the measured table:)* the bottleneck is the **x8 electrical PCIe link
(6.20 GB/s pinned ceiling)**, not per-copy overhead or a pinning fallback. Copy consolidation
(4→2 copies/layer) [moves / does not move] the per-stage time by [MEAS] — [it is therefore worth
enabling by default / it is left default-off as a measured non-win]. The prefetch schedule already
hides [MEAS] % of the transfer behind compute; the residual is the raw bus, so the lever that would
help most on this host is a wider slot, not a smarter schedule. On an x16 host the same code should
see roughly double these GB/s — untested here, flagged for anyone who runs it.

## Reproduce

```bash
# Decode, stats on, arena off vs on (OLMoE + r16 adapter):
OFFLOAD_EXPERTS=1 BENCH_TOKENS=128 E4B_OFFLOAD_STATS=1 \
  ADAPTER=./out/adapter_best.pt python -m experts4bit_qlora.infer
OFFLOAD_EXPERTS=1 BENCH_TOKENS=128 E4B_OFFLOAD_STATS=1 E4B_OFFLOAD_ARENA=1 \
  ADAPTER=./out/adapter_best.pt python -m experts4bit_qlora.infer

# Training A/B third arm (see bench/run-offload-ab.sh):
E4B_OFFLOAD_STATS=1 OFFLOAD_EXPERTS=1 STEPS=5 python -m experts4bit_qlora.train
```
