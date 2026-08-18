# G8 balance + G9 throughput — serving-class box run

Instrument: AMD Threadripper PRO 7995WX (96 cores, **full AVX-512**
including bf16/vnni) + RTX 4090 24 GB, 251 GB RAM. Box's own calibration
(`calib.json`): grouped scatter **146.0 GB/s**, 4090 triad **944.4 GB/s**,
pinned link 23.2 / 27.1 GB/s. This is the many-channel AVX-512 host the
DRAM tier was designed around — the machine class the dev box (AVX2,
25 GB/s scatter) was not, which is why both clauses were deferred here.

Model: **Qwen3-30B-A3B**, NF4 arena (48 layers × 128 experts, 6144 rows,
16.3 GB, baked in 57 s).

## G8 — balance clause: measured MISS, with a diagnosis

> GPU/CPU completion times within 20% of each other at B=8 and B=16

| batch | DRAM bus | GPU bus | ratio | achieved DRAM | achieved VRAM | placement (vram/dram) |
|---|---|---|---|---|---|---|
| 8 | 159.8 ms | 19.8 ms | **0.124** | 3.1 GB/s | 49.7 GB/s | 4045 / 2099 |
| 16 | 235.7 ms | 17.5 ms | **0.074** | 3.1 GB/s | 54.6 GB/s | 3236 / 2908 |

(An earlier B=8 arm at a different VRAM budget read 0.179 — same shape.)

**The finding is not the miss, it is why.** The DRAM tier achieves
**~3 GB/s against its own calibrated grouped-scatter of 146 GB/s** —
about 2% — while the GPU side reaches ~50 of 944 (5%). The solver
balances tiers on calibrated bandwidth, so it hands the CPU a share the
executor cannot deliver, and no placement can fix that: the input is
wrong, not the arithmetic.

**The trend across batch names the mechanism.** As batch doubles, DRAM
time *rises* (160 → 236 ms) while GPU time *falls* (20 → 17 ms). A
bandwidth-bound path would flatten as batch amortizes weight reads —
that is Phase 8's whole amortization law, and the GPU side obeys it.
The CPU side moving the other way is the signature of **per-row compute**
dominating, which is exactly the crossover G8 measured earlier: the DRAM
tier leaves the bandwidth-bound regime near ~8 tokens per expert, and at
B=16 it is well past it.

So the actionable conclusion for the solver is specific: **its cost model
needs a compute term for the CPU tier at batch, not bandwidth alone.**
Balancing `unique_reads / bandwidth` is correct only while the CPU tier
is bandwidth-bound, and batching is precisely what ends that.

## G9 — throughput clause: measured MISS, and the same bottleneck

> aggregate ≥140 tok/s at B=8

| configuration | aggregate | per-stream | TTFT p50 | window |
|---|---|---|---|---|
| hybrid tier (VRAM 4045 / DRAM 2099) | **23.9 tok/s** | 3.0 | 12.2 s | 21.4 s |
| pure streaming (baseline arm) | 3.4 tok/s | 0.4 | 38.7 s | 149.8 s |

The tier is worth **7×** over streaming, which matches G3's shape (21× at
235B). But 23.9 is far under the bar, and G8's numbers say why: the DRAM
bus spends 160 ms of every step while the GPU spends 20. **The CPU tier's
2%-of-calibrated throughput is the same wall capping both clauses** —
fix the achieved DRAM bandwidth and both move together.

## Model caveat, stated rather than buried

The clause names **gpt-oss-120b**, and this is Qwen3-30B-A3B. gpt-oss
could not be served: `enable_mxfp4_nvme_residency` **refuses** with
"arena serving does not yet carry per-expert biases" — gpt-oss keeps
biases resident alongside the packed stacks, and the loader declines
rather than dropping them, which would silently change the epilogue. The
gate's named model is therefore blocked on a **feature** (baking biases
into the arena), not on performance. The MXFP4 bake itself works and is
fast: 36 layers × 128 experts = 4608 rows, 60.93 GB, **67 s**, bytes
relocated verbatim with nothing re-quantized.

## Receipts

`calib.json` (box calibration), `g8_balance_b16.json`, `g9_hybrid.json`,
`g9_qwen.json` (streaming baseline), `summary.txt` (raw lines + hardware
identification).
