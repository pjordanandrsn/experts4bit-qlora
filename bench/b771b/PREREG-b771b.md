# B771b — does the fixed bucketed-graph path decode exactly as the eager runner, and what is P80's ratio on it? One RTX 5090 (registered 2026-09-29, before any run)

Issues and fixes: experts4bit-qlora#771, #777 (the bucket-1 append fix), grouped-nf4-gemm#413 (the fused append's IEEE
quotient). Rule: the owner's standing no-ask tier for a single run under $15 (2026-09-26), with the usual mechanics: this
page merged before the launch, receipts and ledger rows, proven teardown. The guard is 1.0 h, not over one hour, so no
proving rental is required; the runner keeps an optional `B771B_PROVE=1` path, rehearsed on the A2000. The authorization
permalink is posted on #771 before the launch.

## What is already known

- **B771** (`bench/b771/RESULTS-b771.md`). With #413's append fix, the eager step and the bucket step agreed in every row
  until the first one-row step. That exposed the bucket-1 bug: a bound bucket of one row appended to a scratch slot, so
  the row decoded its one-row phase without its own new tokens.
- **#777** routes a bound bucket through the batch append whatever its size, with two tests:
  - a routing test through the real shim, which passes on CPU and fails with the old routing (A2000);
  - a GPU invariant test (each stepped row's device length equals the host count after every bucketed step), which
    needs sm_89+ and **has never run**.
- **P80's ×2.32 and P81's ×13.95** were measured on the buggy path. Their register rows carry that correction and wait
  for a re-measurement.

## Instrument

`bench/b771b/b771b_run.sh` on the box, staged by `b771b_drive.sh` and pinned by `staged.sha256`
(`tests/test_b771b_staged_pin.py`). One RTX 5090, image `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`.
- **Install (B511's pattern).**
  - grouped-nf4-gemm at **`efc5677ea959fa1ce246c0732845c57b09ce3f38`** (main: #413 and #414).
  - experts4bit-qlora cloned at the launch commit (a `main` containing #777 and this page) and installed editable
    under a constraints file that holds the image's torch and triton. transformers 5.16.1 and bitsandbytes 0.50.1,
    P80's pins.
  - A tripwire refuses if the import does not resolve to the clone, if the clone lacks #777's routing, or if gnf4 lacks
    `_e4m3_group`.
- **Arm T (the GPU tests).**
  - e4b: `tests/test_decode_graph_buckets.py` (the replay test, and the **invariant test** in both modes) and
    `tests/test_bucket1_append_routing.py`.
  - gnf4: `kernel/test_fp8_kv_append.py` (the fused-append byte gates, which need sm_89+).
  - Output: `armT.xml`.
- **Arm M.** `mut_b771b.py` restores the shim's old single-row routing; then the invariant test alone runs.
  `armM.xml`. The clone is restored afterwards.
- **Decode stage.** P80's setup:
  - Model: Qwen3-30B-A3B @ `ad44e777`, NF4 arena by P39's `k8_bake.py`, nothing int4, fp8 paged KV, all-VRAM hybrid
    engine, `--no-fuse-qkv`.
  - Prompts and trace: the 16 p37 prompts, and the trace `_dynb_plan(16, 32)` (16, 8, 4, 2, 1 rows for 32 steps each).
  - Harness: P81's `step_decomp.py`, referenced and pinned.
  - Arms in order:
    - **A1** eager with the engine's default grouping;
    - **B1**, **B2** graphs;
    - **A2** eager, default grouping;
    - **P**, the bucket step run eagerly;
    - **Ad**, eager with the device grouping.
  - A1–P are P80's five arms unchanged, so the ratio reads against P80's rule. Ad runs the same grouping as B and P,
    so it is the eager function the graphs must now equal bit for bit.
- **Verdict.** `b771b_reduce.py`, from the two junit files and six receipts. Its self-test pins the rule on 11 cases.

## Predictions (written before the data)

- **T.** Every registered test PASSES on the 5090, none skipped:
  - both modes of the invariant test;
  - the replay test;
  - the routing test;
  - gnf4's `test_bitwise_against_eager_path` and `test_bt1_bitwise_against_t1_loop`.
- **M.** Under the mutation, the invariant test FAILS in both modes.
- **I (identity, the correctness claim).** Ad ≡ P ≡ B1 ≡ B2, bitwise, in all 16 rows: with both fixes, the graph path
  decodes the eager runner's function.
- **R (the ratio).** B1/A1 > 1.03 and B2/A2 > 1.03, with the self-pairs inside [1/1.03, 1.03]: P80's claim, re-measured
  on the fixed path.
- **Reported, not decisive:** A1 vs Ad (different grouping kernels), Ad/A1, and per-step ms by active-set size.
  - The one-row phase now attends over a growing context, where the buggy path froze it. So the fixed B's one-row steps
    may be slightly slower than P80's.

## Decision rule (`b771b_reduce.py`)

1. **VOID** (NOT_RUN; neither confirms nor refutes):
   - a missing junit or arm receipt;
   - a registered test absent from the junit, or **skipped** (this card must run them all);
   - the mutation not caught (the invariant test is inert for its bug);
   - an arm with the wrong mode or grouping, or a trace other than the registered one;
   - A1 ≠ A2;
   - a self-pair outside [1/1.03, 1.03].
2. **REFUTED**:
   - a registered test fails on the fixed path;
   - a graph bucket fell back to eager;
   - **Ad, P, B1 and B2 are not bitwise identical**;
   - or, with everything valid, B1/A1 ≤ 1.03 or B2/A2 ≤ 1.03.
3. **CONFIRMED**: T, M, I and R all hold.

**On CONFIRMED:**
- #777's fix is verified on the hardware path.
- A new register row, P80's measurement on the fixed path, supersedes P80's row (`superseded_by`).
- P81's row stays corrected-but-unsuperseded until the int4 stack is re-measured. This lane does not run it.

## Box and cost

- **One rental, `b771b-5090-<n>`.**
  - One RTX 5090, Vast verified/secure, image `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`, ≥ 130 GB of disk.
  - **Guard 1.0 h at ≤ $0.75/h (≤ $0.75).**
- **Expected about 40 min:** install ~3 min, tests ~5 min, fetch ~12 min, bake ~2 min, six arms of ~3.5 min.
- **Lane ceiling $1.50; hard stop $2.** A bandwidth NOT_RUN is re-launched, never added to the machine exclusions.

## Rehearsal

The A2000 (sm_86) cannot run the fp8 paged KV, so no test in arm T and no arm can run there. The rehearsal covers:
- the runner's `B771B_PROVE=1` path in a throwaway A2000 container, with the class lifted by knob: install, tripwires,
  reducer self-test;
- the driver's dry run;
- the CI tests (pins, rule, arms, the mutation's target in the shipped shim).

## What this lane cannot say

- Nothing about the int4 stack (P81): this is NF4 only.
- Nothing about other families, arrivals, or quality. A bitwise-equal eager and graph pair is one function; this lane
  does not score it.
- Nothing about another host: the ratio is host-bound in the same way as P80's and P81's.

## Receipts

`receipts/experts4bit-qlora/<date>/b771b-5090-<n>/` in the adertha receipt store:
- the launcher's `receipt.json` and `teardown-proof.json`;
- the fetched `b771b/`: `armT.xml`, `armT_{e4b,gnf4}.{xml,txt}`, `armM.{xml,txt}`, `b771b_{A1,B1,B2,A2,P,Ad}.json`,
  `verdict.json`, `summary.txt`, `versions.txt`, `forensics.txt`, `constraints.txt`, `logs/`, `work/bake.json`.

The read lands as `bench/b771b/RESULTS-b771b.md` with the small receipts.
