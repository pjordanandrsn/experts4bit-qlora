# B771 — is the fused fp8 KV append's non-IEEE quotient the whole cause of the eager-vs-bucket-step divergence? One RTX 5090 (registered 2026-09-29, before any run)

Issue: experts4bit-qlora#771. Fix under test: grouped-nf4-gemm#413. Rule: the owner's standing no-ask tier for a single
run under $15 (2026-09-26), with the usual mechanics: this page merged before the launch, receipts and ledger rows,
proven teardown. The guard is 1.0 h, not over one hour, so no proving rental is required; the runner keeps an optional
`B771_PROVE=1` path, rehearsed on the A2000. The authorization permalink is posted on #771 before the launch.

## What is already known

- **P81** (`bench/p81/RESULTS-p81.md`). At identical rows, grouping, weights and prompts, the eager `PagedModelRunner`
  step (A) and the bucket step run eagerly (P) decoded different greedy tokens: 917 of 1,008 equal, first differences
  at tokens 67 and 129.
- **What differs between the two steps.** The KV append:
  - A appends through `Fp8PagedKV.append_many`, which uses torch's `quantize_kv_fp8`;
  - P appends through `append_graph_bt1`, which uses gnf4's fused `fp8_kv_append_bt1`.
- **The $0 probe on the A2000** (#771, fp32 intermediates, torch's cast standing in for the hardware one):
  - The fused kernel's quotient `x / scale` uses Triton's default fp32 `/`, which is not IEEE-rounded. ~30% of
    quotients differed from torch's, and **~6e-8 of stored e4m3 bytes**.
  - With the quotient as `tl.math.div_rn`: **0 of 302M**.
  - grouped-nf4-gemm#413 ships that.
- **Not known.**
  - Whether the hardware cast shows the same byte rate.
  - Whether this is the **whole** cause of A ≠ P, or one cause among others.

## Instrument

`bench/b771/b771_run.sh` on the box, staged by `b771_drive.sh` and pinned by `staged.sha256`
(`tests/test_b771_staged_pin.py`). One RTX 5090, one process tree, two grouped-nf4-gemm cuts:

- **OLD** = `fb15cf5f5b9f2fb107fa1a39218fd9d910b34d94` (v0.33.5, P80's and P81's pin). Its fp8 kernels are
  byte-identical to v0.33.6's.
- **NEW** = `530f93b9aa425da34e8c284e0c946ef97db2fba4` (the grouped-nf4-gemm#413 merge).
- The runner's tripwire refuses an OLD cut that already has `_e4m3_group`, and a NEW cut that lacks it.
- **Byte stage** (`b771_bytes.py`, run under each cut before any model loads).
  - gnf4's `fp8_kv_append_bt1` writes into a paged pool: 4,096 slots in a random order, one token per slot per call,
    32 calls.
  - Four layouts: groups 1 and 4, × activation scales 1 and 30. That is **5.4×10⁸ values** per cut, D = 128, H = 8.
  - Every payload byte and every scale is compared with `quantize_kv_fp8`'s, **with the hardware e4m3 cast**.
  - The harness's own layout math is pinned in CI (`tests/test_b771_bytes_harness.py`): a pure-torch stand-in that
    writes the reference's bytes reads 0, and one flipped byte reads 1.
- **Decode stage.** P80's setup, unchanged:
  - `Qwen/Qwen3-30B-A3B` @ `ad44e777`, NF4 arena by P39's `k8_bake.py`, nothing int4, no folds, no fused router,
    fp8 paged KV, all-VRAM hybrid engine, `--no-fuse-qkv`.
  - The 16 p37 prompts, prefilled together.
  - The trace `_dynb_plan(16, 32)`: 16, 8, 4, 2, 1 rows for 32 steps each.
  - The harness is **P81's** `bench/p81/step_decomp.py`, referenced and not copied.
  - **Every arm runs the device grouping** (`--dynb-grouping device`), so A and P differ only in the step path.
  - Arms in this order, one process and one receipt each:
    - under OLD: **A_old** (`--dynb-mode eager`), then **P_old** (`--dynb-mode padded`, the bucket step, eager);
    - swap to NEW (`pip --force-reinstall --no-deps`; tripwire);
    - under NEW: **P_new**, then **A_new**.
  - The runner stamps each receipt with `gnf4_tag` (old/new), the installed version and `gnf4_has_e4m3_group`. That is
    provenance, not measurement, and the reducer checks it.
- **Verdict.** `b771_reduce.py`, from the two byte reports and four receipts. Its self-test pins the rule on nine cases.

## Predictions (written before the data)

- **B1 (bytes).** NEW: **0** differing payload bytes and 0 differing scales over 5.4×10⁸ values. OLD: **> 0** differing
  payload bytes (the probe's rate predicts ~30), and 0 differing scales.
- **C (control).** A_old ≡ A_new, bitwise. The eager path never calls the fused append, so the swap cannot move it.
- **D1 (the divergence occurs).** P_old ≠ A_old somewhere in the trace (P81 saw it on a different stack; here it is
  expected, not assumed).
- **D2 (the claim).** P_new ≡ A_new, bitwise, in all 16 rows.

## Decision rule (`b771_reduce.py`)

1. **VOID** (NOT_RUN; neither confirms nor refutes):
   - a missing byte report or decode receipt;
   - a byte stage that ran the wrong cut;
   - an arm with the wrong mode, cut or grouping, or a trace other than the registered one;
   - **A_old ≠ A_new** (the control moved);
   - OLD's bytes all equal the reference (nothing to explain);
   - **P_old ≡ A_old** (the divergence did not occur on this trace, so its removal cannot be shown).
2. **REFUTED**:
   - NEW still writes a differing byte or scale; or
   - with everything valid, **P_new ≠ A_new**: the fused append is not the whole cause.
3. **CONFIRMED**: B1, C, D1 and D2 all hold.

A CONFIRMED licenses one sentence: on this box and trace, the fused append's non-IEEE quotient was the only thing that
made the bucket step decode differently from the eager step, and #413 removes it. It says nothing about graph
**replay**: P81 already showed replay ≡ P bitwise.

## Box and cost

- **One rental, `b771-5090-<n>`.**
  - One RTX 5090, Vast verified/secure, image `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`, ≥ 130 GB of disk.
  - e4b at the launch commit (a `main` containing this page).
  - **Guard 1.0 h at ≤ $0.75/h (≤ $0.75).**
- **Expected about 40 min:** install, ~1 min per byte stage, a ~12 min fetch and ~2 min bake (P80's), and four arms of
  ~3.5 min, plus one reinstall.
- **Lane ceiling $1.50; hard stop $2.** A bandwidth NOT_RUN is re-launched; it is not added to the machine exclusions,
  which are for ssh-readiness failures only.

## Rehearsal

The A2000 (sm_86) cannot run the e4m3 cast or the fp8 paged KV, so neither stage can run there. The rehearsal covers:
- the runner's `B771_PROVE=1` path in a throwaway A2000 container, with the card class lifted by knob: install at the
  OLD cut, the tripwire including the OLD-cut check, and the reducer self-test;
- the driver's dry run;
- the CI tests (pins, rule, arm order, the byte harness's layout math).

It is not a reading.

## What this lane cannot say

- It covers one model (Qwen3 NF4), one trace and one box.
- It does not show that the fix changes quality. A ≡ P means the bucket step then computes the eager step's function;
  it does not score either.
- It does not cover the int4 stack P81 ran. The append is the same code there, but that is not measured here.

## Receipts

`receipts/experts4bit-qlora/<date>/b771-5090-<n>/` in the adertha receipt store:
- the launcher's `receipt.json` and `teardown-proof.json`;
- the fetched `b771/`: `bytes_{old,new}.json`, `b771_{A_old,P_old,P_new,A_new}.json`, `verdict.json`, `summary.txt`,
  `versions.txt`, `forensics.txt`, `logs/`, `work/bake.json`.

The read lands as `bench/b771/RESULTS-b771.md` with the small receipts.
