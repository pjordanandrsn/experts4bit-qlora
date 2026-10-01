# P89 — does K23's lean glue make Qwen3-30B-A3B's B=16 int4 decode faster on an RTX 5090, with the same output bits? (registered 2026-10-01, before any run)

Issue: experts4bit-qlora#564. This is P88's speed instrument, re-run with K19 at its licensed default in both arms. The
arms differ only in `E4B_INT4_LEAN_GLUE` (#814, over grouped-nf4-gemm K23, #427).

**What changed since P88:**
- **P88** (`bench/p88/RESULTS-p88.md`) LICENSED K19 for T > 1 int4 decode rows. B=16 went from 11.200 to 10.140 ms/step
  (0.905×), and K19 is now the default there.
- **P88's two censuses, diffed kernel by kernel** (8 graph replays per arm). K19's route adds **0.685 ms/step** of launches
  around the GEMM:
  - the tile builder `_tile_table_r1`, 0.430;
  - three fills, 0.096 (+144 elementwise calls per step);
  - an `index_elementwise` kernel, 0.094;
  - a `_scatter_gather_elementwise` kernel, 0.065.

  Both arms also pay an `index_select` of **0.46 ms/step**, 48 calls of 9.6 µs. By its call count and size it is
  inferred to be `_forward_collapsed`'s `[T * top_k, H]` expansion of the token rows.
- **K23** folds that glue into the kernels that bracket K19:
  - the builder zeroes its own padding slots and emits the sorted ids, in one launch;
  - gate_up reads the token rows through `gather_div`, so no expansion is made;
  - down stores straight into the caller's row order (`scatter`), so there is no `index_copy_` unsort.

  Every piece is bit-identical to what it replaces. The contract tests are in grouped-nf4-gemm; the route tests and
  `tests/test_k23_lean_glue_gpu.py` are here.

Rule: the owner's standing no-ask tier for a single run under $15 (2026-09-26), with the usual mechanics:
- this page merged before the launch;
- a proving rental first (the guard exceeds 1 h);
- receipts and ledger rows;
- proven teardown.

## The question and the instrument

**Speed.** B=16, OFF (`E4B_INT4_LEAN_GLUE=0`) and ON (`=1`), with `E4B_INT4_GROUPED_SMALLM` unset (K19 at `auto`) in
both. P88's harness bytes, env (P86's RTN int4 experts, uncalibrated int4 attention, glue r1/r2, router epilogue) and
command line (graph replay, fused q/k/v). Each arm is drawn twice and the first draw is censused. Order: OFF, ON, ON,
OFF.

**Quality.** There is no K8. The route is bit-identical by construction, and a K8 would be inert as a gate: K8 scores
through the T == 1 loop, where the licensed default keeps the singleton GEMV and the lean route never engages. The
gates that run the code they gate are:
1. **The premise on the card**, before anything is fetched (rc 25):
   - `tests/test_k19_row_exact_gpu.py`: K19's rows are row-count invariant;
   - `tests/test_k23_lean_glue_gpu.py`: the lean route is bit-equal to the default at B=16, eager, captured in a CUDA
     graph and replayed, from expanded rows and from token rows.

   It must read 6 passed and none skipped.
2. **Token equality in the model.** Each arm's timed JSON carries every row's prefill, warm and timed greedy tokens. In
   P88 the same arm reproduced its tokens exactly across draws, and OFF vs ON (different arithmetic) differed in 13 of
   16 rows, so equality is a sharp check. OFF and ON must agree in all 16 rows, in both draw pairs.

**B=1 is not measured.** The lean route does not engage at T == 1, where auto keeps the singleton GEMV; the collapse
only builds the same expansion one call later.

**The reducer:** `p89_reduce.py`, with a 14-case self-test.
- **VOID** if:
  - the premise did not hold, did not run, or skipped a test;
  - a draw is missing, or two draws of one arm are more than 3 % apart;
  - a census is missing;
  - K19 is not at 96 calls per step and the tile builder at 48 in both arms (K19's licensed route is what is measured);
  - ON did not engage: at least 48 fewer `index_select` launches per step (the expansion) AND at least 192 fewer
    kernel launches per step (4 per layer) than OFF.
- **IDENTITY_FAIL** if any row's tokens differ between OFF and ON in either draw pair. The route stays opt-in whatever
  its speed.
- **LICENSED** if tokens are identical and median ON / median OFF ≤ **0.97** at B=16.
- **NOT_FASTER** otherwise.

**What a LICENSED read proposes:** `E4B_INT4_LEAN_GLUE` defaults to `1` on K19's rows, in its own PR citing this read.

## Predictions (written before the data)

- **B=16 ON/OFF 0.91–0.95: LICENSED.**
  - The removable launches are about 0.25 ms of K19's glue plus the 0.46 ms expansion, about 0.7 ms of OFF's
    ~10.1 ms.
  - The builder does slightly more work, since it now zeroes its own padding slots, so the gain should land short of
    the full 0.7 ms.
- **Launches per step fall by about 288** (6 per layer): the expansion, three fills, the unsort and one more glue
  launch. `index_select` falls by 48.
- **Tokens are identical** in all 16 rows of both draw pairs.
- **OFF reproduces P88's ON step** (10.14 ms) within host spread. Reported, not gated.

## Box and cost

1. **`p89-prove-<n>`**: one RTX 5090, any CPU vendor, 0.5 h guard at ≤ $0.75/h (≤ $0.375). It runs:
   - the refusals (class, disk);
   - the install and tripwire (K20's plan, K23's options in both packages, the route present, K19 at `auto`, the
     lean switch unset);
   - the reducer self-test and the premise;
   - K19's and K16's contracts plus K23's builder tests, compiled on the card.

   No model. At most three attempts.
2. **`p89-5090-<n>`**: only after a passing proof. **Guard 1.5 h at ≤ $0.75/h (≤ $1.125).**
   - Fetch and bake as P88.
   - Four timed arms of about 4 minutes each.
- **Lane ceiling $2.00; hard stop $2.50.** Both under the $35 per-run cap.

## Rehearsal

The proving path ran on the NAS RTX A2000 (sm_86) from e4b `96d38e3`, this branch before this paragraph, and
grouped-nf4-gemm `3990dbc`. The staged bytes are pinned by `staged.sha256` and are unchanged since. It ran with
`P89_GPU_CLASS=A2000 P89_MIN_DISK_GB=20`, so it was marked REHEARSAL. It read:
- the reducer self-test, 14 cases;
- install and tripwire OK;
- the premise, 6 passed;
- K19 + K16 contracts compiled, 31 passed;
- K23's builder compiled, 17 passed;
- rc 0.

No model was fetched and no time is quoted (the A2000 is correctness-only).

Amendments, dated, go below this line before any data is read.
