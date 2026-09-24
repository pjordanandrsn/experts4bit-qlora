# Results — P68: which part of the attention makes a verify or a prefill differ from T = 1 decode, and is it over the bar once enough positions are read? (RTX 5090, 2026-09-24)

Pre-registration: [`P68-PREREG.md`](P68-PREREG.md) (#733, merged before any rental). Record:
[#725](https://github.com/pjordanandrsn/experts4bit-qlora/issues/725). Receipts: [`receipts/`](receipts/).

**How the numbers were produced.** The reducer ([`p68_reduce.py`](p68_reduce.py)) ran on the box and wrote
[`receipts/RESULTS-p68-generated.md`](receipts/RESULTS-p68-generated.md) and [`receipts/p68_rep.json`](receipts/p68_rep.json).
Run again locally over the committed, gzipped arm receipts, it reproduces the generated tables byte for byte. The JSON
differs only in the last digit of a few float sums (Python 3.11 on the box, 3.14 here). Every number below is from
those files.

## Runs

| run | what | outcome | cost | receipt (adertha-receipts) |
|---|---|---|---|---|
| `p68-prove-1` | the proof: the launcher's `nvidia-smi`, e4b `df0dc06` | OK | $0.0102 | `0df099f` |
| `p68-5090-1` | reading | NOT_RUN: pre-flight ssh rejected by a stale `known_hosts` key for a reused Vast proxy port | $0.0311 | `f028ca8` |
| **`p68-5090-2`** | **the reading**, same commit | **rc 0** | **$0.2386** | `d60e376` |

- **Lane total:** $0.2799 against the $2 ceiling. Every run's teardown is proven.
- **Box.** One RTX 5090 (575 W, driver 580.119.02) on an AMD EPYC 9655. 4-stream egress was 184.7 MB/s.
- **Software.** e4b at `df0dc06`, grouped-nf4-gemm at `68a1250`, torch 2.8.0+cu128, Triton 3.4.0, transformers 5.16.1.
- **Wall time.** 18 min: fetch 2.8, bake 2.1, `int4` 8.1, `nf4` 4.6.
- **Authorization.** The owner, relayed at [#725](https://github.com/pjordanandrsn/experts4bit-qlora/issues/725#issuecomment-5816572370).

## Verdicts

**G0 held on both stacks.**
- Every forcing engaged.
- No T = 1 attention call carried a mask.
- Every forced arm's control equalled the unforced control bit for bit, and the control repeated.

**27 registered predictions held and 1 was refuted.**

### Which attention component makes the difference (the ablation, P63's fixture)

Every first difference is given as a (layer, site) for each verify position (verify17: 68 positions; verify16: 64).

| arm (forced) | nf4 verify17 | int4 verify17 | predicted | verdict |
|---|---|---|---|---|
| `hf.base` (nothing) | L0 `attn_core` 68 | L0 `attn_core` 68 | layer-0 attention | HELD |
| `hf.proj` (projections) | L0 `attn_core` 58, L0 `mlp_out` 6, L1 4 | L0 `attn_core` 56, L0 `mlp_out` 6, L1 6 | > 50 % at layer-0 attention | HELD |
| `hf.core` (attention core) | L0 `attn_core` 68 | L0 `attn_core` 68 | > 50 % at layer-0 attention | HELD |
| `hf.proj_core` (both) | L0 `mlp_out` 46, L1 `attn_core` 21 | L0 `mlp_out` 44, L1 `attn_core` 22 | none at layer-0 attention | HELD |
| **`hf.all`** (+ router, LM head, norms) | **exact 68 / 68** | **exact 68 / 68** | **EXACT** | **HELD** |

verify16 reads the same on every arm: `hf.all` is exact 64 / 64 on both stacks.

**What the ablation shows.**
- **Neither component alone is enough.** With only the projections forced, most positions still first-differ in
  layer-0 attention. With only the core forced, all of them do.
- **Both are necessary.** With both forced, layer-0 attention is exact, and the first difference moves to layer 0's
  MLP. There the router's multi-row call differs, which then reaches layer 1's attention.
- **All together are sufficient.** Forcing everything that picks its arithmetic by row count (projections, core,
  router, LM head, norms) makes **a verify bit-identical to T = 1 decode on both stacks**, with the experts on their
  registered row-exact singleton route.

**Prefill.**
- **nf4:** `hf.all` exact, 160 / 160 (HELD).
- **int4: REFUTED.** Q5 predicted every prefill position not exact, first differing at layer-0 `attn_core`. The read
  was 159 / 160 at layer-0 `attn_core`, and **position 0 exact**.
  - The mechanism held. Above 64 rows the attention fold falls through to the fused-q/k/v forward, which norms and
    rotates in separate steps where decode's `rope_norm_heads` kernel does both in one.
  - At position 0 the rotation is the identity (cos 1, sin 0), so the two arithmetics agree there.
  - The registered rule required every position not exact, so the cell is refuted as registered. The mechanism the
    prediction named is confirmed on 159 positions.
- **Q7, paged backend.**
  - **Prefill** (layer-0 `attn_core` on nf4, `attn_in` on int4): HELD.
  - **Verify** carried no prediction. With the projections forced, layer-0 attention is exact on this box on both
    stacks: the first differences are at L0 `mlp_out` and later. So on the 5090 the fp8 kernel's verify path computes
    each row as its decode call does. (The sm_86 rehearsal read it not exact; the plan differs by SM count.)
- **Q6, `hf.fp32red`** (informational). Turning off torch's bf16 reduced-precision cuBLAS reduction alone leaves
  every position differing at layer-0 attention, at a KL similar to `hf.base`.

### Is the served T > 1 path over the bar? (the size reading: 4 wikitext rows × 512 tokens, served default)

| stack | backend | mode | positions | KL mean [95 % CI] | top-1 [95 % CI] | class |
|---|---|---|---|---|---|---|
| nf4 | HF | verify17 | 952 | 3.35e-03 [2.12e-03, 5.45e-03] | 0.982 [0.975, 0.990] | WITHIN-BAR |
| nf4 | HF | verify16 | 896 | 2.51e-03 [2.07e-03, 3.05e-03] | 0.986 [0.978, 0.992] | WITHIN-BAR |
| nf4 | HF | prefill | 2,048 | 5.06e-03 [3.86e-03, 6.47e-03] | 0.981 [0.975, 0.987] | WITHIN-BAR |
| nf4 | paged | verify17 | 952 | 1.42e-02 [7.10e-03, 2.68e-02] | 0.958 [0.945, 0.970] | WITHIN-BAR |
| nf4 | paged | verify16 | 896 | 9.53e-03 [6.61e-03, 1.41e-02] | 0.974 [0.964, 0.983] | WITHIN-BAR |
| nf4 | paged | prefill | 2,048 | 1.69e-02 [1.17e-02, 2.51e-02] | 0.955 [0.945, 0.963] | WITHIN-BAR |
| int4 | HF | verify17 | 952 | 3.29e-03 [2.68e-03, 4.03e-03] | 0.969 [0.957, 0.979] | WITHIN-BAR |
| int4 | HF | verify16 | 896 | 3.33e-03 [2.83e-03, 3.91e-03] | 0.965 [0.955, 0.974] | WITHIN-BAR |
| int4 | HF | prefill | 2,048 | 5.79e-03 [4.91e-03, 6.91e-03] | 0.960 [0.950, 0.969] | WITHIN-BAR |
| int4 | paged | verify17 | 952 | 9.40e-03 [8.06e-03, 1.10e-02] | 0.957 [0.946, 0.967] | WITHIN-BAR |
| int4 | paged | verify16 | 896 | 1.16e-02 [7.53e-03, 1.88e-02] | 0.959 [0.946, 0.971] | WITHIN-BAR |
| int4 | paged | prefill | 2,048 | 1.45e-02 [1.27e-02, 1.64e-02] | 0.946 [0.936, 0.957] | WITHIN-BAR |

- **Every cell is WITHIN-BAR** against the shipped bar (KL ≤ 0.10, top-1 ≥ 0.93).
  - The largest KL upper bound is 0.027, about a quarter of the bar.
  - The lowest top-1 lower bound is 0.9355 (int4 paged prefill).
  - The registered size prediction (KL CI upper < 0.05 and top-1 ≥ 0.93 on every cell) held.
- **P63's OVER-BAR cells were small-sample top-1:** 64–160 positions, where one flip moves top-1 by 1.6 %. Read over
  ~1,000–2,000 positions with the windows as clusters, none crosses the bar.
- **The paged backend differs more than HF,** in both KL and top-1. That is its fp8 K/V at decode against the verify's
  and prefill's own reads. It is still well inside the bar.

## What the decision rule did

- **Q5 held on verify →** register a measured claim:
  `e4b.serve.p68.qwen3.verify-from-t1-calls.row-exact.5090.2026-09-24`.
  - On this stack, a verify assembled from T = 1 calls (projections, attention core, router, LM head and norms each
    per row, experts on the singleton route) is bit-identical to incremental decode: 132 / 132 verify positions on
    each of the nf4 and int4 stacks.
  - Q2–Q4 name the necessary pieces.
  - Whether an exact verify is worth its cost is a separate speed lane. This read makes no speed claim.
- **S: WITHIN-BAR on every cell →** register it: `e4b.serve.p68.qwen3.t-gt-1-vs-t1.within-bar.5090.2026-09-24`.
  **#725 closes.** The served T > 1 path is inside the shipped bar relative to T = 1.
- **The refuted cell (int4 `hf.all` prefill) is reported as refuted**, with the mechanism the data confirmed. It is not
  re-derived.
- No default moves.

## What this read does not say

- **No bf16 reference.** The size is T > 1 against T = 1 on one quantised stack.
- **Scope.** One family, one box, P63's 160-token fixture for the ablation, and four wikitext rows for the size.
- **Speed.** Nothing about the cost of an exact verify.
- **The paged verify's exactness under the projection forcing** was read, not predicted, and is this box's plan.
