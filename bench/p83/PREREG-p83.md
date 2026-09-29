# P83 — on one box, do P70's build (e4b 0.37.4, grouped-nf4-gemm 0.33.0) and P82's build (0.37.8, 0.33.7) read the same wikitext K8 with the fp32 router? (registered 2026-09-29, before any run)

Issue: experts4bit-qlora#674. P82 (`bench/p82/RESULTS-p82.md`, #785) found that the licensed int4 recipe's K8 does not
reproduce across boxes to five decimals. With byte-identical packs and the same router setting, P81's box read 6.33015
and P82's read 6.31811, although reading the diffs found no change on K8's code path between those runs. P82's
fp32-router build read 6.36396 against the licensed builds' 6.36709. The licensed builds (P55x, P64, P70) ran
e4b ≤ 0.37.4 on two machines.

So a difference in K8 between two runs could come from the box, from the software, or from both. P82 could not tell
which. This lane runs both software stacks on **one** box. Rule: the owner's standing no-ask tier for a single run
under $15 (2026-09-26), with the usual mechanics: this page merged before the launch, a proving rental before a guard
over 1 h, receipts and ledger rows, proven teardown. The owner's go ("go ahead with P83") is relayed on #674 before the
launch.

## Question

**On one RTX 5090, does the licensed recipe's build read the same wikitext K8 under P70's software as under P82's,
both with the fp32 router?**
- If yes, the software between them does not move K8, and the cross-box spread is the box.
- If no, the software moved it, and the difference is what a bisect looks for.

## Instrument

- **Model and data.** `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`, fetched once. Each stack bakes
  its own NF4 arena with P39's `k8_bake.py`, under its own installed software, as its lane did. The K8 is wikitext,
  2,048 teacher-forced paged-decode steps at B = 1, eager, `--no-fuse-qkv`, on window `9ef10d760ad9`.
- **The router.** `E4B_ROUTER_EPI_CAST=0` is exported for every process. e4b 0.37.4 reads `"0"` (and unset) as fp32;
  0.37.8 reads `"0"` as fp32.
  - Each stack's tripwire refuses the box unless `router_epilogue.CAST_WEIGHTS[0] is False`.
  - Every K8 process has a sidecar stamp recording what the router module reads under its env, plus the installed
    e4b and gnf4 versions. The stamp is also exercised at each install, so the proof covers it.
- **Stack O, P70's build as P70 ran it.**
  - e4b `c77aab6dafb5f08d98a7387d1e986d141b20a2d3` (0.37.4, P70's launch commit) and grouped-nf4-gemm
    `5ca1897585f9f456f99ea504b2a1be0ea91db496` (0.33.0). Both are registered constants in the runner.
  - P70's harness bytes: `bench/p39/step_decomp.py` and `bench/p42/hook/usercustomize.py`, which still hash to
    `bench/p70/staged.sha256`'s pins.
  - P70's env, including `E4B_RECOMPILE_LIMIT=64`. P70's build command: calibrated int4 experts (C4, 128 sequences,
    10 layers per pass), calibrated int4 attention, the folds, the router epilogue, and the expert pack dumped.
  - **Reading `O_build`** is the build's K8. **`O_rep`** repeats it: P55x's "lic" arm, with the expert pack loaded by
    fingerprint and the attention calibrated live.
- **Stack N, P82's build as P82 ran it.**
  - e4b at the launch commit (a `main` containing this page; no change on K8's path since 0.37.8) and grouped-nf4-gemm
    `9407d499a4d1e0fe8c22b050878a9f869b385b45` (0.33.7), installed over O with `--force-reinstall --no-deps`. The two
    cuts declare the same dependencies.
  - P82's harness bytes (`bench/p81/step_decomp.py`, hook v7), P82's env and P82's build command, with both packs
    dumped.
  - **`N_build`** is the build's K8. **`N_rep`** is P82's K32: both packs loaded by fingerprint.
- **The comparison is of the two lanes' builds as they ran**, package and harness together. A DIFFERENT reading
  therefore bisects over both.
- **Per stack, the tripwire checks:**
  - the installed commits (from pip's `direct_url.json`) are the pinned ones, with O at 0.37.4 / 0.33.0 and N's gnf4 at
    0.33.7;
  - the router is at fp32;
  - the #405 artifact knobs are present, and for N the #754 attention dump and load;
  - the stack's own hook loads as `usercustomize`.
- **Verdict.** `p83_reduce.py` computes it. Its self-test pins the rule on 13 synthetic cases, and it runs on the box
  and in CI.

## Decision rule (`p83_reduce.py`)

1. **VOID** (neither answer):
   - any of the four readings is missing, off the registered window, or not 2,048 steps;
   - any reading is not stamped fp32, or its stamped e4b / gnf4 versions are not its stack's;
   - **a stack does not repeat itself**: `O_build` ≠ `O_rep` or `N_build` ≠ `N_rep`, by mean NLL bit for bit. If a
     stack does not repeat on one box, then two stacks differing on that box says nothing about software. (P55x's
     build and "lic" arm agreed; so did P82's build and K32.)
2. **SAME**: `O_build` and `N_build` have bit-identical mean NLL. On this box the software between P70's build and
   P82's does not move K8, so the spread between boxes is the box.
3. **DIFFERENT**: they differ. The software moved K8 on this box. The difference is reported in nats.

**Reported, never decisive:**
- whether O reads the licensed 6.36709 and N reads P82's 6.36396, to five decimals;
- the expert pack fingerprints of both builds against the licensed `0c9955a9…`;
- N's attention pack against `d7cfa1f4…`.

## What follows

- **SAME:**
  - #674's quality statement cannot be a five-decimal K8. It needs a cross-box tolerance.
  - The next question is the box mechanism (driver, Triton autotune), not software.
  - The licensed pack bytes stay the identity, as they already are.
- **DIFFERENT:**
  - A bisect over e4b `c77aab6..` the launch commit (package, hook v6 → v7, and the step_decomp copies) and
    grouped-nf4-gemm 0.33.0 → 0.33.7, on one box. It can run as a sequence of K8 readings, each loading the dumped
    packs where the software allows.
  - If O also reproduces 6.36709 here (a third machine), the licensed number is reproducible with P70's software, and
    the newer software moved it.

## Predictions (written before the data)

- **Stated expectation, not the rule: DIFFERENT, with `O_build` reading 6.36709 to five decimals.** The reasoning:
  - P70's software read 6.36709 on two machines with different drivers (575.57.08 and 595.71.05).
  - On P82's box, whose driver is P64's, P82's software read 6.36396.
  - Between P81 and P82, newer software with no K8-path change still moved between boxes.
  - Together that suggests the newer software changed the arithmetic, box-sensitively or not. It is the weaker of the
    two readings to predict, and it can fail.
- **Both stacks repeat themselves** exactly on this box. If not, the read is VOID, and within-box non-determinism is
  itself the finding.
- Both expert packs are `0c9955a9…`, and N's attention pack is `d7cfa1f4…`.

## Box and cost

Two rentals, in order. The guard exceeds one hour, so the compute rule requires a proving rental first.

1. **`p83-prove-<n>`**: one RTX 5090, **0.4 h guard at ≤ $0.75/h (≤ $0.30)**. `P83_PROVE=1` runs:
   - the refusals;
   - **both installs** at their pins, O then N over it;
   - both tripwires and both router stamps;
   - the reducer self-test and an egress probe.

   No model. The guard is twice P82's because it installs twice, and P82's proof ran out of its 0.2 h while fetching.
   At most three attempts.
2. **`p83-5090-<n>`**: only after a proof returns rc 0 with its receipts fetched. One RTX 5090, Vast verified/secure,
   `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`, ≥ 200 GB of disk. **Guard 4.5 h at ≤ $0.75/h (≤ $3.375).**
   - Two builds: P82's took 64 min on a fast host, P81's 97 min on a slow one. Add two repeats of 5–10 min, one fetch
     and two bakes.
   - The worst case seen is about 3.8 h. The 1,500 s first-calibration-chunk watchdog ends a host that cannot build
     (rc 30).

**Lane ceiling $4.00; hard stop $5.00**; both under the $35 per-run cap.

## Rehearsal

The home A2000 (sm_86) cannot run the fp8 paged KV or hold the model, so no K8 can be rehearsed. The rehearsal is:
- `P83_PROVE=1` in a throwaway A2000 container, with the card class lifted by the runner's knob: both installs, both
  tripwires and both stamps, on 0.37.4's router module and 0.37.8's;
- a mutation with the router export deleted, which the tripwire must refuse;
- the driver's dry run;
- the CI tests: the pins, the rule, both stacks' pins and harness sources, the export, and the order O → N.

It is not a reading.

## What this lane cannot say

- **Which box property moves K8** (driver, autotune, CPU), if the reading is SAME.
- **Which change moves it**, if DIFFERENT. That is the bisect.
- **Other windows or texts** (c4val1), or quality beyond K8.
- **Anything about throughput**, which is not measured here.
- **Whether a third box would agree.** This is one box.

## Receipts

`receipts/experts4bit-qlora/<date>/p83-{prove,5090}-<n>/` in the adertha receipt store:

- The launcher's `receipt.json` and `teardown-proof.json`.
- The fetched `p83/`:
  - `k8_{O_build,O_rep,N_build,N_rep}.json` with their `.router.json`;
  - `stamp_{O,N}.json` and `versions_{O,N}.txt`;
  - `verdict.json`, `packs.json`, `summary.txt` and `forensics.txt`;
  - the three pack manifests, `logs/` and both `work_*/bake.json`.

The read lands as `bench/p83/RESULTS-p83.md` with the small receipts.
