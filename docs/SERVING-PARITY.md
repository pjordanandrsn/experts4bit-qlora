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
which is why Gemma-4 read −5.84 ppl while agreeing to 0.0078 nats with
the chunked oracle — a number since superseded by the chunk-free
measurement below, which it fails.

**Every delta below sits on a routing-flip floor.** The oracle chunks
and the paged path decodes incrementally; on a mixture-of-experts model
those two arithmetic orders route a few percent of tokens to different
experts, which disagrees regardless of correctness (METHODOLOGY 13.1).
Measured floors: Granite 0.0033, gpt-oss 0.0176, Qwen3 0.0095 nats
(8192-step windows); Gemma-4's is 0.0814 on a 512-step window, which is
itself anomalous (below).
**Three of the four paged deltas measured against a chunk-free reference
are below their model's floor** — Granite 0.00229, gpt-oss 0.00288,
Qwen3 0.00173 — and in each the serving path is closer to the reference
than the chunked oracle is. **Gemma-4 is not**: 0.24747 nats, three
times its floor ([#359](https://github.com/pjordanandrsn/experts4bit-qlora/issues/359)).
Qwen3's delta (0.0058, against the chunked oracle) is likewise below its
floor, which retires the claim that the fp8 KV cache costs +0.047 ppl.

| Family | Attention features needed | Parity status | Evidence |
|---|---|---|---|
| Qwen3-30B-A3B | plain causal, `head_dim**-0.5` | `validated` +0.0467 ppl = +0.0058 nats vs oracle — BELOW its 0.0095-nat arithmetic-order floor, i.e. indistinguishable | oracle-compared, 8192 steps + floor |
| Granite-3.1-3B-A800M | `attention_multiplier` scale | `validated` +0.0219 ppl = +0.0028 nats vs oracle (PASS) | oracle-compared, 8192 steps |
| Mixtral-8x7B | plain causal | `not_tested` for absolute parity | paged-vs-paged only (int4 experts −0.015) |
| OLMoE-1B-7B | plain causal | `not_tested` for absolute parity | paged-vs-paged only |
| gpt-oss-20b | sinks + alternating sliding window 128 | `validated` as BEHAVING: measured against a chunk-free reference (one full forward, no chunk boundaries) the paged path sits at **0.00288 nats**, BELOW that model's 0.01758-nat floor — indistinguishable from the model's own attention, and closer to the reference than the chunked oracle is. NOT a formal PASS: no perplexity gate exists for this family (raw-text ppl 2361 through plain transformers; chat-window 2029), and the pre-registered KL gate is falsified (METHODOLOGY 13). Loader and served expert tier are `validated` faithful (MXFP4 dequant bit-exact; expert forward cos 0.991-0.993; served layer 0 cos 0.998). | oracle-compared + floor, 512 steps |
| Gemma-4-26B-A4B-it | per-layer KV geometry (sliding 256/8 beside full 512/2), sliding window, scale 1.0 | `no reference at this resolution` ([#359](https://github.com/pjordanandrsn/experts4bit-qlora/issues/359)): paged − one-shot = +0.093 / +0.114 / +0.247 nats on three 512-token windows, while transformers' own cached forward is −0.107 / +0.271 / +0.081 from the same one-shot forwards. Plain transformers gives identical tokens an NLL that moves 0.4 nats with the batch shape (§ below). Measured path-specific cost: fp8 cache + dot 0.046 nats (0.017 with 32-wide K groups), on the 512-dim layers. Serves both geometries, fp8 throughout, coherent text. Loading separately `broken` on 2 of 5 hosts (e4b#344). | three windows + three-forward test, 512 steps |

What "unsupported as a gated path" means for gpt-oss: the code serves it
and the expert math is right, but this project will not publish a
quality number for it until the gate exists. Serve it if you want; do
not read a perplexity from this harness as evidence about it.

## Chunk-free reference (P26 / P26b, 512 steps)

One window scored three ways per model: a single full forward (no chunk
boundaries), the chunked production oracle, and the paged serving path.
`floor` is |chunked − full|; `parity` is |paged − full|.

| family | full | chunked | paged | floor | parity | reading |
|---|---|---|---|---|---|---|
| Granite-3.1-3B-A800M | 1.68776 | 1.69106 | 1.69005 | 0.00330 | **0.00229** | indistinguishable |
| gpt-oss-20b | 4.94398 | 4.96156 | 4.94686 | 0.01758 | **0.00288** | indistinguishable |
| Qwen3-30B-A3B | 1.61240 | 1.60599 | 1.61067 | 0.00641 | **0.00173** | indistinguishable |
| Gemma-4-26B-A4B-it | 3.34492 | 3.42629 | 3.59239 | 0.08137 | 0.24747 | see the Gemma-4 section: no reference at this resolution |

In the first three the serving path is closer to the chunk-free
reference than the chunked oracle is. Gemma-4 is the exception, and by a
wide margin; its floor is also five to twenty-five times the others',
which is itself unexplained ([#359](https://github.com/pjordanandrsn/experts4bit-qlora/issues/359)). Receipts are in the private audit tree
(`measured-private` in `claims.json`).

## Gemma-4: no reference at 512-token resolution (P27, 2026-09-03)

Three 512-token windows, one box, the same three arms:

| window (prompt offset) | one-shot forward | HF cached forward (chunk 256) | paged | cached − one-shot | paged − one-shot |
|---|---|---|---|---|---|
| 0 | 3.34492 | 3.42629 | 3.59239 | +0.081 | +0.247 |
| 3000 | 4.88783 | 4.78129 | 4.98121 | −0.107 | +0.093 |
| 6000 | 6.49212 | 6.76274 | 6.60579 | +0.271 | +0.114 |

The reference moves more than the thing being measured. The mechanism,
in plain transformers 5.16.1 with no e4b code (bf16, eager; the same
with sdpa): the cached prefill equals a short forward bit-for-bit, so
the cache is correct; but the first 256 positions of a 769-token
forward differ from a 256-token forward from layer 1 (0.7%) to layer 19
(37%), and replacing the tokens *after* position 256 with random ones
moves the positions *before* it by 0.2% at layer 1 and 34% at layer 19 —
the NLL of those identical 255 tokens reads 7.79, 7.38 or 7.70
depending only on what follows them. That is bf16 batch-shape variance
in the per-expert gathered GEMMs, which every MoE has, amplified by
Gemma-4's router; Qwen3-30B under the identical test moves 0.15% → 3.5%
and its NLL by 0.001. Running the router in fp32 (plain transformers) does not remove the amplification (layer-19 divergence 0.25 against 0.34, top layer unchanged) and itself moves the same tokens' NLL by 1.1 nats, so router precision is not a lever; the sensitivity is the model's.

What this leaves for the paged path on this family: the fp8 cache and
dot cost **0.046 nats** on window 0 (the whole kernel precision model
applied inside the one-shot forward; 0.017 with 32-wide K groups; the
five 512-dim layers carry it, the sliding layers cost 0.002), and no
other localisable term — the per-layer diff against the one-shot
forward has the same shape on Qwen3 at a third of the amplitude. On two
of the three windows the paged path is closer to the one-shot forward
than transformers' own cache is. Receipts: private audit tree, P27.

