# P147 — results: **READ**. Int8 weight-only rounding of Mixtral-8x7B's attention projections keeps the W8A16 split-premise gate; NF4 does not (2026-10-19)

Registration: `bench/p147/PREREG-p147.md` (#1790, merged `15f8345f`). This is a quality **premise** on the lab's RTX A2000,
run teacher-forced with expert offload. It changes no default, gate or speed claim, and it reports no timing.

Code under test:
- e4b at `15f8345f` (the registration's merge);
- grouped-nf4-gemm 0.47.1 at `38b30888`;
- torch 2.8.0+cu128, transformers 5.17.0, bitsandbytes 0.50.2;
- `mistralai/Mixtral-8x7B-Instruct-v0.1` at `eba92302`: experts NF4 under offload, with attention and the head bf16 in A.

**Verdict by `p147_reduce.py`: READ.**
- **I8ATTN: GO.**
- **NF4ATTN: PARK.**
- No VOID condition fired. All 21 shards were at their pinned bytes. Every arm's windows matched A's. Each round trip touched
  exactly its registered modules (128 attention projections, with the head untouched; I8HEAD: the head only), and all 129
  weights were restored to A's sha256 before each arm after A.
- The maintainer re-derived `verdict_p147.json` byte-identical from the receipt store. They also recomputed both gates from
  the per-window NLLs and argmaxes, without the reducer.

**Run:** `p147-a2000-1`, 2026-10-19 03:37:50Z to 06:10:38Z, on the lab RTX A2000 12GB. The card started with 10,379 MiB
free and no compute process. **Cost $0.**

## The instrument checks (registered before data)

| check | text / set | value | bar | |
|---|---|---:|---:|---|
| (a) A against P141's served FP8-KV paged reference R, mean NLL | wikitext | d = −0.000196 nats | \|d\| ≤ 0.02 | pass |
| | c4val1 | d = −0.002248 nats | \|d\| ≤ 0.02 | pass |
| (b) A′ against A | every set | Δppl 0, **bit-identical** | ≤ 0.001 ppl | pass |

## The rule, per candidate, at both gates

GO iff argmax agreement ≥ 0.95 and d + 1.645·SE ≤ ln(1 + 0.05 / ppl_A), where d is the mean per-window NLL shift
against A on this box.

| arm | gate | windows | mean d (nats) | SE | upper | bound | argmax | |
|---|---|---:|---:|---:|---:|---:|---:|---|
| I8ATTN | W1 | 196 | 0.001369 | 0.000581 | 0.002325 | 0.011560 | 0.9813 | pass |
| I8ATTN | W16 | 123 | −0.000314 | 0.000548 | 0.000588 | 0.011323 | 0.9779 | pass |
| NF4ATTN | W1 | 196 | 0.012777 | 0.001576 | 0.015369 | 0.011560 | 0.9428 | fail (both terms) |
| NF4ATTN | W16 | 123 | 0.012603 | 0.001967 | 0.015838 | 0.011323 | 0.9428 | fail (both terms) |
| I8HEAD (reported) | W1 | 196 | 0.000786 | 0.000235 | 0.001173 | 0.011560 | 0.9898 | — |
| I8HEAD (reported) | W16 | 123 | 0.000431 | 0.000270 | 0.000875 | 0.011323 | 0.9900 | — |

ppl_A is 4.3005 on W1 and 4.3910 on W16. K8 Δppl on the check set (reported): I8ATTN 0.0034 (wikitext) and 0.0103
(c4val1); NF4ATTN 0.0841 and 0.0909; I8HEAD 0.0022 and −0.0008.

## Predictions

| prediction | outcome |
|---|---|
| I8ATTN: **BORDERLINE**: argmax agreement between 0.95 and 0.98 at both gates, and the upper bound within ±30 % of the bound at W1 | **MISSED** (clearer than predicted): GO, with the upper bound at 0.20× the bound at W1 (predicted within ±30 %), and argmax agreement 0.9813 at W1, above the predicted 0.98 ceiling (W16's 0.9779 is inside). Qwen3-30B-A3B parked at 1.26× its bound under the same rule (the W8A16 split premise). |
| NF4ATTN: **PARK**, with argmax agreement < 0.95 or the upper bound above the bound at both gates | HELD (both terms, both gates) |
| I8HEAD (reported): argmax ≥ 0.98 and d ≤ a quarter of the bound | HELD (0.9898 and 0.000786 at W1) |
| A against P141's R within 0.01 nats on wikitext | HELD (−0.0002) |

## What the GO starts (never automatic)

Per the registration, a GO starts nothing by itself. The A2000 is correctness only. The next step is a speed registration
on an RTX 5090, whose own box re-checks this rule on `serve_paged` on the served card before anything flips. The ceiling
from P145 still holds: halving the q and o bytes saves at most 0.72 ms of the 8.125 ms W1 node period, and all four
projections at most 0.90 ms. NF4 for these projections is parked on Mixtral.

The receipt is in `receipts/p147-a2000-1/`.
