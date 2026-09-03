# Serving parity — paged decode against the model's own attention

*2026-09-03. Companion to [`support_matrix.md`](support_matrix.md), which
is OpenTimestamps-anchored and therefore never edited in place. This
section was briefly appended to that file (PRs #347, #353, #355) and is
moved here; the anchored document is restored to its anchored bytes. Same
practice as [`ARCHITECTURE_SUPPORT.md`](ARCHITECTURE_SUPPORT.md), which
exists for the same reason.*

The parity method, the routing-flip floor and the retired claims are in
[`METHODOLOGY.md`](METHODOLOGY.md) sections 13 and 13.1. This page is the
per-family table.

## Serving parity — paged decode vs the model's own attention (P25, 2026-09-03)

The paged B=1 decode path is only as valid as the attention it reproduces.
Two evidence levels appear here and they are NOT interchangeable:

- **oracle-compared** — scored against the same weights through
  transformers' eager attention and bf16 cache on the same sha-matched
  window (`--ppl-oracle eager`). This is the only evidence that the
  paged path matches the MODEL.
- **paged-vs-paged only** — the family's K8 numbers compare one paged
  configuration against another. Error the two arms share cancels, so
  these say nothing about absolute parity. See `docs/METHODOLOGY.md`
  section 13.

Host for every row: rented RTX 5090, torch 2.8.0+cu128, triton 3.4.0,
transformers 5.16.1, e4b main (post-#339/#340), grouped-nf4-gemm 0.24.0.

Parity is quoted in **nats** (mean NLL difference) as well as ppl: the
`|Δppl| <= 0.05` bar is only meaningful where perplexity is ~8. At
oracle ppl 752 it is a 0.007% relative bar and mechanically unpassable,
which is why Gemma-4 reads −5.84 ppl while agreeing to 0.0078 nats.

**Every delta below sits on a routing-flip floor.** The oracle chunks
and the paged path decodes incrementally; on a mixture-of-experts model
those two arithmetic orders route a few percent of tokens to different
experts, which disagrees regardless of correctness (METHODOLOGY 13.1).
Measured floors: Granite 0.0033, gpt-oss 0.0176, Qwen3 0.0095 nats.
**Every paged delta measured against a chunk-free reference is below its
model's floor** — Granite 0.00229, gpt-oss 0.00288 — and in both cases
the serving path is closer to the reference than the chunked oracle is.
Qwen3's delta (0.0058, against the chunked oracle) is likewise below its
floor, which retires the claim that the fp8 KV cache costs +0.047 ppl.

| Family | Attention features needed | Parity status | Evidence |
|---|---|---|---|
| Qwen3-30B-A3B | plain causal, `head_dim**-0.5` | `validated` +0.0467 ppl = +0.0058 nats vs oracle — BELOW its 0.0095-nat arithmetic-order floor, i.e. indistinguishable | oracle-compared, 8192 steps + floor |
| Granite-3.1-3B-A800M | `attention_multiplier` scale | `validated` +0.0219 ppl = +0.0028 nats vs oracle (PASS) | oracle-compared, 8192 steps |
| Mixtral-8x7B | plain causal | `not_tested` for absolute parity | paged-vs-paged only (int4 experts −0.015) |
| OLMoE-1B-7B | plain causal | `not_tested` for absolute parity | paged-vs-paged only |
| gpt-oss-20b | sinks + alternating sliding window 128 | `validated` as BEHAVING: measured against a chunk-free reference (one full forward, no chunk boundaries) the paged path sits at **0.00288 nats**, BELOW that model's 0.01758-nat floor — indistinguishable from the model's own attention, and closer to the reference than the chunked oracle is. NOT a formal PASS: no perplexity gate exists for this family (raw-text ppl 2361 through plain transformers; chat-window 2029), and the pre-registered KL gate is falsified (METHODOLOGY 13). Loader and served expert tier are `validated` faithful (MXFP4 dequant bit-exact; expert forward cos 0.991-0.993; served layer 0 cos 0.998). | oracle-compared + floor, 512 steps |
| Gemma-4-26B-A4B-it | per-layer KV geometry (sliding 256/8 beside full 512/2), sliding window, scale 1.0 | `validated` as BEHAVING: one pool served both geometries, Δ = **−0.0078 nats** (same order as the two passing families), output text coherent, whole window ran the fp8 compute path. NOT a formal PASS: the absolute Δppl gate is inapplicable at oracle ppl 752 (it reads −5.84), so a verdict waits on the in-distribution instrument. Loading is separately `broken` on 2 of 4 hosts (e4b#344). | oracle-compared, 8192 steps |

What "unsupported as a gated path" means for gpt-oss: the code serves it
and the expert math is right, but this project will not publish a
quality number for it until the gate exists. Serve it if you want; do
not read a perplexity from this harness as evidence about it.
