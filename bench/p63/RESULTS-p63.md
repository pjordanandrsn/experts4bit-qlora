# Results — P63: does a token's output depend on how many rows share its forward? (RTX 5090, 2026-09-24)

Pre-registration: [`P63-PREREG.md`](P63-PREREG.md) (#720, merged before any rental). Record:
[#708](https://github.com/pjordanandrsn/experts4bit-qlora/issues/708). Receipts: [`receipts/`](receipts/).

The reducer ([`p63_reduce.py`](p63_reduce.py)) ran on the box and wrote
[`receipts/RESULTS-p63-generated.md`](receipts/RESULTS-p63-generated.md) and
[`receipts/p63_rep.json`](receipts/p63_rep.json). Run again locally over the committed, gzipped arm receipts
(`receipts/out/<stack>/p63_arm.json.gz`), it reproduces both files byte for byte. Every verdict below is read from
that JSON.

## Runs

| run | what | outcome | cost | receipt (adertha-receipts) |
|---|---|---|---|---|
| `p63-prove-1` | the proving rental: attach, `nvidia-smi`, receipt, ledger, teardown | OK, RTX 5090, driver 580.95.05 | $0.0115 | `9904ceb` |
| **`p63-5090-1`** | **the reading**, same e4b commit `75c83bd`, nothing merged between | **rc 0**, instance 52340702 | **$0.2881** | `33a7a0d` |

- **Lane total:** $0.2996, against a $1.50 ceiling. Teardown is proven for both runs ([`receipts/teardown-proof.json`](receipts/teardown-proof.json)).
- **Box.** One RTX 5090 (sm_120, 170 SMs, driver 595.71.05) on an AMD EPYC 7352 with 125 GiB RAM. Its power limit was **475 W**. That does not matter here, because this lane measures arithmetic, not speed.
- **Software.** e4b 0.37.3 at `75c83bd` and grouped-nf4-gemm 0.33.2 at `f88df1e`, with the `_int4_part_or_none` fix present (tripwire OK). torch 2.8.0+cu128, Triton 3.4.0, transformers 5.16.1, bitsandbytes 0.50.1.
- **Model.** Qwen/Qwen3-30B-A3B at `ad44e777`.
- **Wall time.** 26 min from install to done: fetch 9.5 min, bake 1.3, then the stacks `int4` 6.1, `nf4` 4.2 and `int4nf` 4.3 min (`receipts/outer.log`).

## Verdicts

**Every registered prediction held.** No defect was found in 179 records. P5 on the int4 stack landed between its bands, as the registration allows.

| reading | verdicts |
|---|---|
| **G0** instrument gates | OK on `nf4`, `int4`, `int4nf` |
| **P1** kernel census | 9 / 9 HELD |
| **P2** module replay | 24 / 24 HELD |
| **P3** combine census | 3 / 3 HELD (the torch chain carried no prediction: INFO) |
| **P4** T = 1 controls across sub-arms | 12 / 12 HELD |
| **P5** B393's end-to-end size | NF4 **HELD** (KL 1.18e-04, 0 flips); int4 **BETWEEN** (KL 1.70e-02, 7 / 160 flips = 4.4 %) |
| **P6** end to end | 45 / 45 HELD (no position exact anywhere; every first difference at layer 0) |
| **P7** router output dtype | 3 / 3 HELD |
| **D** defect scan | NONE of 179 |

### Which routes are exact (P1 and P2 agree)

On this box, at Qwen3's layer-0 gate_up (N 1536, K 2048), each route was run with the model's own activations and routing. "Rows equal" counts rows bit-equal to the token's own T = 1 call, at T = 16 / 17 / 160 tokens (128 / 136 / 1,280 rows):

| route | kernel at T > 1 | rows equal (P1) | experts modules exact (P2) | class |
|---|---|---|---|---|
| int4 store, `FORCE_SINGLETON_GROUPS` | `gemv_int4_b32` at T·8 rows | all | 48/48 at every n | **EXACT** |
| int4 store, `DEVICE_GROUPING`, ≤ 256 routed rows | `gemv_int4_b32` | all | 48/48 at 16, 17 | **EXACT** |
| int4 store, `DEVICE_GROUPING`, > 256 rows | grouped int4 GEMM | 119, 127, 1,122 | 0/48 at 160 | REORDER (max rel L2 4.4e-4) |
| int4 store, default | dequant + bf16 matmul | 0, 0, 0 | 0/48 | PRECISION (rel L2 ≤ 1.5e-2) |
| NF4, `FORCE_SINGLETON_GROUPS` | dot-pad decode GEMV | all | 48/48 at every n | **EXACT** |
| NF4, singleton + `GNF4_GEMV_DOTPAD=0` | scalar GEMV, split-K from the rows | 116, 124, 1,123 | 0/48 | REORDER (2.4e-4) |
| NF4, default / `DEVICE_GROUPING` | M-tile (TF32 weights) | 0, 0, 0 | 0/48 | PRECISION (2.7e-3) |
| MoE combine | `combine_rows` | every row, T ∈ {2, 16, 17, 64, 160} | — | **EXACT** |

- **Other modules (P2).** The attention projections, the router and the lm_head are never exact above one row: cuBLAS picks its kernel by M, and the int4 attention switches GEMV → K16 → cuBLAS. On the `int4` stack the input RMSNorm is exact at 16 / 17 rows on the fold kernel and not at 160, where the torch chain runs. That fold row is measured, but it is not registered as a route claim, because the decision rule needs P1 and P2 to agree and P1 has no census row for the fold.
- **Accuracy (D).** Every path was inside its own operand model's fp64 bound. 26 cuBLAS records exceeded it only under torch's default `allow_bf16_reduced_precision_reduction` (max ratio 3.34), and were inside it with fp32 split-K reduction. That is the pre-registered reading, so it is not a defect.

### End to end (P6): nothing is exact, and the difference starts in attention

- **No position was exact** between incremental T = 1 decode and the same tokens inside a 17- or 16-row verify or a 160-row prefill. This holds on every stack and every sub-arm (45 cells).
- **Every first difference was at layer 0,** in `attn_core` (the o_proj input). The exception is the `int4` stack's prefill, where it was in `attn_in`: the norm fold leaves its kernel above 64 rows. It was never in the experts. On the routes above, the expert arithmetic is row-exact while the attention in front of it is not.
- **Size.** KL mean ranged from 0.0077 to 0.033 nats/token, and top-1 agreement from 0.882 to 0.984.
- **Against the bands:** 7 cells were IN-BAND (≤ 0.02 nats and top-1 ≥ 0.95), 22 were BETWEEN, and 16 were **OVER-BAR** against the shipped KL-from-checkpoint bar (≤ 0.10 nats, top-1 ≥ 0.93).
- **All 16 OVER-BAR cells are over on top-1, not on KL.** The largest KL mean is a third of the bar. Over 64-position verify windows, the top-1 figure moves 1.6 % per flip: 5 flips in 64 reads 0.922. The bar was registered as P6's outer limit, and the decision rule sends a route over it to its own lane. That lane is **[#725](https://github.com/pjordanandrsn/experts4bit-qlora/issues/725)**, which must first ask whether the top-1 criterion is readable at these window lengths.

### P7: the router's weights change dtype at 64 rows

On the `int4` stack the fused router epilogue returns fp32 routing weights up to 64 rows. Above that, the upstream router returns bf16 weights. The expert indices are the same function at both row counts, but the combine weights are not. P7 held, so the rule's follow-up is filed as **[#726](https://github.com/pjordanandrsn/experts4bit-qlora/issues/726)**. It needs its own KL read before any change.

### Side diagnostic (disclosed, not part of the decision)

- **What it computed.** grouped-nf4-gemm #397's `fma_attribution.py` ran on this 5090 (`receipts/fma-attribution/`). `combine_rows` equals the FMA slot-order sum in 144/144 cases, and its per-case output hashes are identical to the A2000's.
- **The correction it forced.** It refuted B393's statement that the kernel's bits differ across sm_86 and sm_120, and that statement was retracted in grouped-nf4-gemm#398. The same inferred statement appears in this lane's registration ("Cross-architecture caveat, from B393") and in the rehearsal README. Both are left as registered and are corrected here: on the evidence now in hand, `combine_rows` and torch's chain are bit-identical across the two architectures at B393's cases.

## What the decision rule did

- **EXACT routes: registered, with GPU tests.**
  - Dispatch routes in this repository's register:
    - `e4b.serve.p63.qwen3.int4-singleton.row-exact.5090.2026-09-24`
    - `e4b.serve.p63.qwen3.int4-device-gemv.row-exact.5090.2026-09-24`
    - `e4b.serve.p63.qwen3.nf4-singleton.row-exact.5090.2026-09-24`
  - Kernel properties in grouped-nf4-gemm's register, merged first ([grouped-nf4-gemm#399](https://github.com/pjordanandrsn/grouped-nf4-gemm/pull/399)):
    - `gnf4.kernel.int4-gemv-row-invariant…`
    - `gnf4.kernel.nf4-dotpad-gemv-row-invariant…`
    - `gnf4.kernel.combine-rows-row-invariant…`
  - `tests/test_p63_row_exact_gpu.py` asserts the three dispatch routes with `torch.equal`, on the plan and dispatch a 5090 takes. It skips on CPU. It includes a control: the default int4 route must NOT come out equal. It ran 9 / 9 on the NAS RTX A2000 ([`receipts/a2000-gpu-tests/pytest.txt`](receipts/a2000-gpu-tests/pytest.txt)).
  - The kernels' own test is grouped-nf4-gemm's `kernel/test_row_invariance_gpu.py`.
- **REORDER routes: licensed as reorder-class,** with their row counts, rows differing, max rel L2 and bound ratios, in grouped-nf4-gemm's register (the grouped int4 GEMM, and the scalar NF4 GEMV under `GNF4_GEMV_DOTPAD=0`). Nothing may claim bit-identity through them.
- **PRECISION routes: recorded in `docs/STATUS.md`** with the end-to-end size: the int4 store's default T > 1 route, NF4's M-tile, and the router epilogue's fp32/bf16 weights. Each path keeps the quality licence it was measured on, and no licence transfers between row counts. P63 moves no default.
- **Over the bar → #725. P7 → #726.**
- **P5** is recorded on the B393 claim's note in grouped-nf4-gemm#399.
- **No `DEFECT?`, and no refuted prediction.**
- **Corrected in this PR.**
  - `engines/hot_residency.py`'s singleton comment said the singleton path is "bitwise-equal" to the grouped path at every T, "pinned in CI". It now says what was measured: the singleton route is row-exact against itself, and it is a different function from the grouped route at T > 1 (and on the int4 store at T = 1 as well).
  - `tests/test_singleton_groups.py`'s docstring now says it pins the dispatch algebra through a mocked GEMM, not the arithmetic.

## What this read does not say

- **Scope.** One box, one family, one fixture of 160 tokens, and one layer's census.
- **Plan dependence.** Row-invariance is a property of the kernels' plans at these shapes on a > 64-SM, ≥ 160-SM part. A ≤ 64-SM part can plan the int4 GEMV's split count from its rows. That is not claimed (the A2000 rehearsal read it exact at OLMoE's shape for a different reason).
- **Paged attention.** Its row-planned `n_split` was capped by the short block table at both row counts, so long contexts are not covered.
- **B=16 batched decode** is a different shape from a 16-row verify.
- **No bf16 reference and no perplexity.** The P6 sizes are distances between two quantised paths, not quality against the checkpoint.

## Receipts not committed

The per-position tensors (`out/<stack>/p63_detail.pt`, 6.4 MB) stay in the private receipt store beside the run, as the rehearsal's did. Their sha256, and those of the uncompressed arm JSONs, are in [`receipts/private-files.sha256`](receipts/private-files.sha256). The arm JSONs themselves are committed gzipped. `calib.json` was a staged input (`bench/p39/calib.json`, pinned in `staged.sha256`), not an output, and is not copied.
