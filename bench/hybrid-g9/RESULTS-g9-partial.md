# G9 partial — continuous batching + chunked prefill, measured

Instrument: OLMoE-1B-7B-0924 (16 layers, 64 experts, k=8) on the in-house
dev box (RTX A2000 12 GB), NF4 arena, paged FP8 KV, the fused decode
kernel, and the Phase-9 scheduler. Batch 4, greedy.

**Correctness first, because a serving number from a wrong engine is
worthless:** the engine reproduces the model's own stock greedy
generation **token for token** on three sequences of different prompt
lengths, batched continuously with chunked prefill.

## The clause this run answers

> decode aggregate degrades ≤20% under continuous prompt arrival

**PASS in the decode-dominated regime** — 13.6% at chunk 128, **5.0% at
chunk 512** — and it fails as the workload becomes prefill-saturated.
That is not an engine defect, it is what the ratio means: if most of the
token work arriving is prefill, decode gets less machine. The useful
deliverable is therefore the curve, not a verdict, and the axis that
governs it is the workload's **prefill share** (prefill tokens over all
tokens), which this harness reports beside every number for exactly that
reason.

| prefill share | chunk | decode degradation | TTFT p50 | aggregate tok/s |
|---|---|---|---|---|
| 33% (prompt 128, gen 256) | 128 | **13.6%** | 20.3 s | 42.5 |
| 33% | 512 | **5.0%** | 20.6 s | 42.5 |
| 67% (prompt 128, gen 64) | 32 | 37.6% | 1.33 s | 60.1 |
| 67% | 128 | **14.9%** | 0.26 s | 72.4 |
| 67% | 512 | 18.0% | 0.27 s | 75.7 |
| 89% (prompt 512, gen 64) | 32 | 64.4% | 6.11 s | 116.0 |
| 89% | 128 | 41.8% | 1.64 s | 172.4 |
| 89% | 512 | 33.8% | 0.45 s | 195.6 |

Degradation is measured against a **quiet** arm — the same decode work
with no arrivals — not against the loaded arm's own aggregate. Comparing
aggregates would flatter the loaded arm, because prefill tokens are
tokens too and would pad the number whose whole job is to expose their
cost.

## Two findings that contradict the naive chunking story

**1. Smaller chunks made everything worse here, including TTFT.** The
textbook tradeoff says small chunks protect resident decoders at the cost
of prefill throughput. On this instrument small chunks lost on every
axis: at 89% prefill share, chunk 32 gave 64.4% degradation and 6.11 s
TTFT where chunk 512 gave 33.8% and 0.45 s. The reason is per-step fixed
cost — python dispatch and kernel launches — which a 1B-active model on a
small card cannot amortize, the same trap G6 documented. On a
serving-class model, where a 512-token chunk is genuinely expensive
compute, the tradeoff should invert; this curve's *shape* is therefore
instrument-specific and is reported as such rather than as a general law.

**2. TTFT and decode latency are not the same axis, and chunking trades
them against each other in opposite directions.** A prompt's own TTFT
requires ingesting *all* of it, so more chunks means more steps means a
later first token for the arriving sequence. What chunking buys is the
*resident* decoders' step latency. Reporting only "TTFT improves with
smaller chunks" would have been backwards for the arriving sequence.

The 20.3 s TTFT p50 in the decode-dominated rows is honest queueing, not
a stall: 8 prompts at 4 slots with 256-token generations means later
arrivals wait for a slot. It is reported beside the throughput because
the protocol requires it — an aggregate rate without the latency it cost
is not a result.

## Still owed for G9

The headline clause — **≥140 tok/s aggregate on gpt-oss-120b at B=8** —
needs a serving-class box; this instrument answers the shape questions,
not the absolute one. That run also closes G8's deferred balance clause,
which is why the two are scheduled together.

## Receipts

`receipts/g9_decode_dominated.json` (full rows incl. per-tier timings and
workload mix). The other two regimes are in the sweep output quoted
above; the harness is `eval_serving.py` in this directory.
