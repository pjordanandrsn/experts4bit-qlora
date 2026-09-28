# P80 — does bucketed CUDA-graph decode beat the eager `PagedModelRunner` on a changing active set, NF4 Qwen3-30B-A3B, one RTX 5090? (registered 2026-09-28, before any run)

Issue: experts4bit-qlora#511 (closed by #757, which shipped the mechanism; B511 verified it on a tiny model). Origin: the
Q2 research memo (adertha-agents#62, `research/memos/2026-09-07-q2-…md`, "Recommendation"), which registered this
experiment's claim, intervention and criteria before the mechanism existed. This page adopts them, and states below the
two places where it departs and why. Rule: the owner's standing no-ask tier for a single run under $15 (2026-09-26),
with the usual mechanics: this page merged before the launch, a proving rental before a guard over 1 h, receipts and
ledger rows, proven teardown. The authorization permalink is posted on #511 before the launch.

**Numbering.** P71–P79 are taken by the intervention-harness line (P71–P77 run on 2026-09-26/27 with run ids `p7x-r*` in
the same ledger; P78 drafted), so this e4b lane is P80.

## Question

The memo's claim, verbatim: *"On the p37 Qwen3 NF4 prompts under a predeclared changing-active-set decode trace, e4b
CUDA-graph bucket replay will exceed its eager `PagedModelRunner` aggregate output-token rate by more than the p37 1.03
self-pair band while producing the identical greedy token stream."*

Counterevidence it named: the earlier real-weight `capture_decode` gain on Qwen3 was only 1.04× (B=1), and bucket
padding may consume the benefit.

## Instrument

- **Model and weights.** `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` (p37's), baked to an NF4 arena
  by P39's `k8_bake.py`. **NF4 only**: nothing int4, no calibration, no folds, no fused router, and every other lever at
  its shipped default (the runner unsets their env).
- **Harness.** `bench/p80/step_decomp.py`: the live `bench/hybrid-g9/step_decomp.py` at `69488f1`, plus the P80 stage
  (`--dynb-mode`) and one extension of its `Fp8PagedKV` call (`scratch_slots`). A lane copy, because closed lanes pin the
  live file. `tests/test_p80_staged_pin.py` asserts that the copy removes no live line except that call. The setup is the
  p37 NF4 control's: `--placement-override all-vram --amort off --batch 16 --prompt-len 512 --no-fuse-qkv`, the hybrid
  engine, fp8 paged KV, and the 16 prompts `step_decomp._k8_window` gives at B = 16 (p37's `prompts_b16.json` rule). Every
  receipt records the prompts' sha256.
- **The trace** (`_dynb_plan(16, 32)`). The 16 rows are prefilled together in one scheduler step. Then every 32 decode
  steps half of the remaining rows finish, so the decoding set is **16, 8, 4, 2, 1 for 32 steps each**: 160 decode-only
  steps and 992 decode tokens. A run whose decode-only steps are not exactly this trace refuses (`TRACE MISMATCH`).
- **Arms, one process and one receipt each, in this fixed order:**
  - **A1**: `--dynb-mode eager`. The production `PagedModelRunner` decode: eager, unpadded, the engine's default T > 1
    grouping.
  - **B1**: `--dynb-mode graph`. `enable_decode_graphs((1, 2, 4, 8, 16))`, capture-safe device grouping (the bv3
    treatment).
  - **B2**: as B1.
  - **A2**: as A1.
  - **P**: `--dynb-mode padded`. The same padded step as B, run eagerly with the same grouping: B's bitwise oracle.
    The memo has no P; it runs last so the memo's A1 → B1 → B2 → A2 order is untouched.
- **Within an arm.** Graphs are captured first, on scratch slots only. The trace then runs twice from fresh slots: pass 0
  warms and is not timed, pass 1 is timed. Timing is the wall of each decode-only scheduler step, from its start to the
  token read-back that ends it, so it is the serving loop and not just the kernels. Every arm runs the raw paged-attention
  shim (the harness's timing wrapper is capture-invalid, so no arm carries it). Recorded per arm: aggregate decode tok/s
  over the timed pass, mean step ms per active-set size, every row's tokens in both passes, graph status and statistics,
  capture time, and peak allocated VRAM.
- **Verdict.** `p80_reduce.py` computes it from the five receipts. Its self-test pins the rule on nine synthetic cases
  and runs on the box and in CI.

## Predictions (written before the data)

- **P1 (the claim).** B1/A1 > 1.03 **and** B2/A2 > 1.03 in aggregate decode tok/s.
- **P2 (correctness).** B1, B2 and P decode identical token streams, bitwise, in every row.
- **P3 (engagement).** Every bucket captures (`graph_status` = `graph` for 1, 2, 4, 8, 16) and no candidate step runs
  eagerly.
- **Stated expectation, not the rule.** The gain concentrates at small active sets, where eager decode is bound by
  launches and host work. At 16 the padding is zero and the graph saves only the launch cost. So B/A is largest at 1 and
  2 and smallest at 16.

## Decision rule (`p80_reduce.py`)

1. **VOID** (NOT_RUN; neither confirms nor refutes): a missing arm receipt (not an OOM); an arm that ran the wrong mode,
   or a trace other than the registered one; an arm whose control carries graph state; A1 ≠ A2 tokens (the control does
   not repeat itself); a self-pair A2/A1 or B2/B1 outside [1/1.03, 1.03].
2. **REFUTED**: a candidate arm OOMs; any bucket falls back to eager, or any eager step in B1/B2; B1, B2 and P do not
   decode identical tokens; or, with everything valid, B1/A1 ≤ 1.03 or B2/A2 ≤ 1.03.
3. **CONFIRMED**: everything valid and B1/A1 > 1.03 and B2/A2 > 1.03.

**Where this departs from the memo, and why.**

- **(a) The token-identity gate is B against P, not B against A.** The memo counted any greedy-token mismatch between
  candidate and control as refuting. But A runs unpadded rows through the engine's default T > 1 grouping, while B and P
  run padded rows through device grouping. bf16 GEMMs can round differently at a different row count
  (finding_bf16_gemm_varies_with_row_count), so an A-vs-B difference is not evidence about the graphs. The gate that
  isolates the graph is B ≡ P, which B511 verified on a tiny model. A-vs-B identity is reported (first divergence, row
  and token), not decided on.
- **(b) P is added and timed.** It splits the ratio: P/A1 is the cost of padding plus device grouping, and B1/P is the
  graph's own effect.

## Box and cost

Two rentals, in order. The guard exceeds one hour, so the compute rule requires a proving rental first.

1. **`p80-prove-<n>`**: one RTX 5090, **0.2 h guard at ≤ $0.75/h (≤ $0.15)**. `P80_PROVE=1`: the refusals, the install at
   the pins, the tripwire, the reducer self-test and an egress probe; no model. At most three attempts (≤ $0.45).
2. **`p80-5090-<n>`**: only after a proof returns rc 0 with its receipts fetched. One RTX 5090, Vast verified/secure,
   image `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`, ≥ 130 GB of disk. grouped-nf4-gemm at
   **`fb15cf5f5b9f2fb107fa1a39218fd9d910b34d94`** (v0.33.5); e4b at the launch commit (a `main` that contains this page and
   `#757`). **Guard 1.5 h at ≤ $0.75/h (≤ $1.125).** Expected about 45 min: a ~15 min fetch, a ~1 min bake, then five
   arms of a model load plus two ~10 s passes each.

**Lane ceiling $2.50; hard stop $3**; both under the $35 per-run cap.

## Rehearsal

The home A2000 (sm_86) cannot run the fp8 paged KV at all (native e4m3 needs sm_89+), so no arm can be rehearsed there.
The rehearsal is the runner's `P80_PROVE=1` path in a throwaway A2000 container (with the card class lifted in a local
copy), plus the driver's dry run and the CI tests: the pins, the reducer's rule, and the trace. It is not a reading.

## What this lane cannot say

It says nothing about licensed int4 serving (NF4 only, per the memo), about any family but Qwen3, about a server loop
with arrivals (the trace has no admissions after the first step), or about prefill. It holds only for one host, because
B = 1 decode is host-bound and 5090 hosts disperse about 8.5 %. So no absolute tok/s is compared to another box's.

## Receipts

`receipts/experts4bit-qlora/<date>/p80-{prove,5090}-<n>/` in the adertha receipt store: the launcher's `receipt.json`
and `teardown-proof.json`, and the fetched `p80/` (`p80_{A1,B1,B2,A2,P}.json`, `verdict.json`, `summary.txt`,
`versions.txt`, `forensics.txt`, `logs/`, `work/bake.json`). The read lands as `bench/p80/RESULTS-p80.md` with the small
receipts.
