# P84 — which factor moved the recipe's fp32 K8 from P70's 6.36709 to P82's 6.36396: the kernels, the package, or the harness? (registered 2026-09-30, before any run)

Issue: experts4bit-qlora#674. P83 (`bench/p83/RESULTS-p83.md`, #787) read DIFFERENT. On one box, P70's build
(e4b 0.37.4, grouped-nf4-gemm 0.33.0, P70's harness) read wikitext K8 mean NLL `1.8511420498367808` (6.36709), and P82's
build (e4b 0.37.8, grouped-nf4-gemm 0.33.7, P82's harness) read `1.8506507749113845` (6.36396). Each float equals the one
its software gave on other machines. The difference is −0.00049 nats, a function change far inside the 0.0095 floor.

This lane finds which of the three things that differ between those builds moved it.

Rule: the owner's standing no-ask tier for a single run under $15 (2026-09-26), with the usual mechanics: this page
merged before the launch, a proving rental before a guard over 1 h, receipts and ledger rows, proven teardown. The
owner's go-ahead is relayed on #674 before the launch.

## Question

Starting at P70's build **O** and ending at P82's build **N**, change one factor at a time. Which steps change the K8?

| point | e4b | grouped-nf4-gemm | harness | K8 mean NLL |
|---|---|---|---|---|
| **O** (known, P70/P64/P83) | 0.37.4 @`c77aab6` | 0.33.0 | P70's | `1.8511420498367808` |
| **H2** | 0.37.4 @`c77aab6` | **0.33.7** | P70's | this lane |
| **H1** | **the launch commit** | 0.33.7 | P70's | this lane |
| **C** (= N, the control) | the launch commit | 0.33.7 | **P82's** | must be `1.8506507749113845` |

Each factor is a single step along this chain:
- **KERNEL** (O → H2): grouped-nf4-gemm 0.33.0 → 0.33.7.
- **PACKAGE** (H2 → H1): e4b 0.37.4 → the launch commit. The package has no K8-path change between 0.37.8 and the
  launch commit, which adds bench and docs only.
- **HARNESS** (H1 → C): P70's harness → P82's, where each harness is:
  - **P70's:** `bench/p39/step_decomp.py`, `bench/p42` hook v6, P70's env;
  - **P82's:** `bench/p81/step_decomp.py`, hook v7, P82's env.

  Hooks v6 and v7 make the same calibration calls (v7 adds only the pack dump and load). The two step_decomp copies
  differ in provenance merging, P80/P81's dynamic-batch stage, a profiling guard and a KV capacity term, none of them
  on the K8 path as read. So HARNESS is expected to be zero, but it is measured, not assumed.

## Instrument

- **One box, three builds, each with a within-box repeat.** All builds run with `E4B_ROUTER_EPI_CAST=0`, and the model
  is `Qwen/Qwen3-30B-A3B` @ `ad44e777`. The K8 is wikitext, 2,048 teacher-forced paged-decode steps at B = 1, on
  window `9ef10d760ad9`.
  - **Stack B** is the launch commit plus grouped-nf4-gemm `9407d499a4d1e0fe8c22b050878a9f869b385b45` (0.33.7),
    installed first with the dependencies.
  - **Stack A** is e4b `c77aab6` (0.37.4) plus the same 0.33.7, installed over B with `--force-reinstall --no-deps`
    (same declared dependencies).
- **C, the control, runs first.** It is P82's build exactly (stack B, P82's harness, both packs dumped), and its
  repeat is P82's K32 (both packs loaded by fingerprint).
  - **If `C_build` is not N's known float bit for bit, this box does not reproduce P82's build.** No comparison with
    a known float then holds, so the lane reduces to VOID and **stops**, saving the rest of the rental.
- **H1**: stack B through P70's harness and env: calibrate both halves, dump the expert pack. Its repeat is P55x's
  "lic" arm: the expert pack loaded by fingerprint, the attention calibrated live. H1 shares C's NF4 arena, since it is
  the same software.
- **H2**: stack A through P70's harness and env, with its own bake and a lic-arm repeat.
- **Stamps and tripwires, as in P83.** Every K8 process is stamped with the router module's reading and the
  installed versions. Each stack's tripwire checks:
  - the pinned commits (from pip's `direct_url.json`) and the versions;
  - the router at fp32;
  - that the harnesses it will run import under it: `o/` for both stacks, and `n/` for B.
- **Every pack manifest is copied to `manifests/`** before its pack is deleted to save disk.
- **The first-calibration-chunk watchdog is 900 s here, down from P83's 1,500 s.** Three builds on a slow host (P81's
  took 1,121 s for the first chunk and 97 min per build) would outrun the guard. A host that slow now fails fast with
  rc 30 and is re-rolled. P82's host (740 s) passes, and fast hosts take 340–360 s.
- **Verdict.** `p84_reduce.py` computes it. Its self-test pins the rule on 15 synthetic cases, and it runs on the box
  and in CI. The known floats are P83's receipts, verbatim, and a test checks that.

## Decision rule (`p84_reduce.py`)

1. **VOID**:
   - `C_build` ≠ N's known mean NLL (the control failed);
   - any reading is missing, off the registered window or step count, not stamped fp32, or not stamped with its
     stack's versions (B: gnf4 0.33.7 and e4b not 0.37.4; A: e4b 0.37.4 and gnf4 0.33.7);
   - any build ≠ its repeat, bit for bit.
2. **Otherwise the verdict names every step whose ends differ**, in chain order, joined by `+`: some of `KERNEL`,
   `PACKAGE`, `HARNESS`. At least one must differ, because O ≠ N. Each step's difference in nats is reported, and they
   sum to N − O.

**What follows:**
- **PACKAGE:** bisect e4b 0.37.5 / 0.37.6 / 0.37.7 on grouped-nf4-gemm 0.33.7, through P70's harness. From 0.37.6 on,
  a build can dump its attention, which separates the calibration from the decode path.
- **KERNEL:** bisect grouped-nf4-gemm 0.33.1 → 0.33.6 on e4b 0.37.4.
- **HARNESS:** diff the two step_decomp copies' K8 path, then a one-file swap.
- **Several factors:** each one's bisect, in the order of its size.

## Predictions (written before the data)

- **Stated expectation, not the rule: PACKAGE alone.** Between these cuts, grouped-nf4-gemm changed the graph-mode fp8
  append, MXFP4 prefill and the cold-offload cost model, none of which the eager all-VRAM K8 runs. The harness copies
  and hooks differ off the K8 path. e4b 0.37.5 through 0.37.8 changed the fused router epilogue module (its default and
  the gpt-oss bias) and the attention-pack plumbing. The fp32 path through that router module is the likeliest place
  for an arithmetic change.
- **The control reproduces N's float here.** Both new builds repeat themselves. Every expert pack is `0c9955a9…`, and
  C's attention pack is `d7cfa1f4…`.

## Box and cost

Two rentals, in order. The guard exceeds one hour, so the compute rule requires a proving rental first.

1. **`p84-prove-<n>`**: one RTX 5090, **0.4 h guard at ≤ $0.75/h (≤ $0.30)**. `P84_PROVE=1` runs:
   - the refusals;
   - both installs (B, then A over it);
   - both tripwires, including each harness importing under its stack;
   - both stamps, the reducer self-test and an egress probe.

   No model. At most three attempts.
2. **`p84-5090-<n>`**: only after a proof returns rc 0 with its receipts fetched. One RTX 5090, Vast verified/secure,
   `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`, ≥ 200 GB of disk. **Guard 4.5 h at ≤ $0.75/h (≤ $3.375).**
   - Three builds of 30–64 min under the 900 s watchdog, three repeats of 5–8 min, one fetch, two bakes and two
     installs: about 2 h on a fast host and 3.8 h on the slowest the watchdog admits.

**Lane ceiling $4.00; hard stop $5.00**; both under the $35 per-run cap.

## Rehearsal

The home A2000 (sm_86) cannot run the fp8 paged KV or hold the model, so no K8 can be rehearsed. The rehearsal is:
- `P84_PROVE=1` in a throwaway A2000 container, with the card class lifted by the runner's knob: both installs, both
  tripwires (e4b 0.37.4 importing on grouped-nf4-gemm 0.33.7, and P70's harness importing on both stacks), and both
  stamps;
- the router export deleted, which the tripwire must refuse;
- the driver's dry run;
- the CI tests, which check:
  - the pins, the rule, the known floats against P83's receipts, and the control gate on P83's N and O receipts;
  - the order and the stop on a failed control;
  - each build's harness and env, the manifests kept, and the watchdog.

It is not a reading.

## What this lane cannot say

- **Which change within the implicated factor** moved K8. That is the next bisect.
- **Why P81's and P82's cast-router readings differ** (#674).
- **Other texts, other families, or throughput.**

## Receipts

`receipts/experts4bit-qlora/<date>/p84-{prove,5090}-<n>/` in the adertha receipt store:

- The launcher's `receipt.json` and `teardown-proof.json`.
- The fetched `p84/`:
  - `k8_{C,H1,H2}_{build,rep}.json` with their `.router.json`;
  - `stamp_{B,A}.json` and `versions_{B,A}.txt`;
  - `verdict.json`, `packs.json`, `manifests/`, `summary.txt` and `forensics.txt`;
  - `logs/` and the `work_*/bake.json`.

The read lands as `bench/p84/RESULTS-p84.md` with the small receipts.

## Amendment 1 (2026-09-30, after `p84-5090-2`, before any further reading)

**What happened.** `p84-5090-2` ran on an **Intel** host (Core i9-14900K, driver 595.84). The control C (P82's build,
bit for bit) read mean NLL `1.8504622130569988` (6.36276), not N's known `1.8506507749113845`. Its repeat matched, so the
box was deterministic. The rule read **VOID**, and the lane stopped before H1 and H2, as registered.
- C's expert pack was the licensed `0c9955a9…`.
- C's attention pack was `45b4cc5a…`, not `d7cfa1f4…`. 342 of its 384 tensor payloads differ from P82's (all 192 scale
  tensors, 151 packed), in all 48 layers, with the same calibration token stream and the same recorded toolchain.
- The attention calibration therefore does not reproduce on this host. The mechanism is not isolated.

**Every earlier reading behind the known floats ran on AMD hosts:**
- O's float on a Ryzen 7950X, an EPYC 7C13 and an EPYC 9755;
- N's on an EPYC 7R13 and an EPYC 9755;
- `d7cfa1f4…` calibrated on an EPYC 7K62, a 7R13 and a 9755.

**The amendment.** The runner refuses a host whose CPU vendor (lscpu's `Vendor ID`) is not `AuthenticAMD`, with
**rc 16**. It does so before the install and the fetch, so a wrong host costs minutes, not a build. The vendor is a knob
(`P84_CPU_VENDOR`); any other value marks the run a REHEARSAL.

**Unchanged:** the question, the chain, the rule, the control and its early stop, the pins, the guard and the ceilings.
**Cost so far:** p84-prove-1 $0.0927 (NOT_RUN), p84-prove-2 $0.0382 (PROVED), p84-5090-1 $0.0377 (NOT_RUN, ssh; its
machine is now excluded), p84-5090-2 $0.5356 (VOID). That is $0.7042, leaving $3.30 under the $4.00 ceiling.

A re-roll can land on the same Intel machine again. The launcher accepts only ssh-readiness failures as machine
exclusions, so the rc 16 refusal is the protection. The Intel reading itself is a finding for #674, reported in
`RESULTS-p84.md`, not a P84 result.
