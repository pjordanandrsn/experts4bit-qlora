# P85 — is grouped-nf4-gemm#413 (the fused fp8 KV append's IEEE-rounded quotient) the whole step that moved the recipe's fp32 K8 from 6.36709 to 6.36396? (registered 2026-09-30, before any run)

Issue: experts4bit-qlora#674. P84 (`bench/p84/RESULTS-p84.md`, #796) read **KERNEL**. On one box,
grouped-nf4-gemm 0.33.0 → 0.33.7 moved the recipe's fp32-router wikitext K8:
- from O's mean NLL `1.8511420498367808` (6.36709; e4b 0.37.4 + gnf4 0.33.0, P70's build);
- to N's `1.8506507749113845` (6.36396; gnf4 0.33.7, whatever the e4b release or harness).

The e4b release and the harness moved it by exactly zero.

Reading the cut release by release (RESULTS-p84), exactly one change is on K8's path: **#413** (0.33.7), the fused fp8 KV
append's quotient rounded with `div_rn`.
- K8's eager loop calls `graph_mode_init`, so every scored token's K and V are appended through `fp8_kv_append_t1`.
- 0.33.1–0.33.3 change no shipped code. 0.33.4 (`cold_deadline`), 0.33.5 (`int4_smallm`'s annotations) and 0.33.6
  (MXFP4) are off this path.

That attribution is an inference from the diff and the call path. This lane measures it.

Rule: the owner's standing no-ask tier for a single run under $15 (2026-09-26), with the usual mechanics:
- this page merged before the launch;
- a proving rental before a guard over 1 h;
- receipts and ledger rows;
- proven teardown.

The owner was told by email (2026-09-30) that this lane would be registered next.

## Question

Two readings, each predicted to equal one exact known float:
- **F.** Take P70's build (e4b 0.37.4 + gnf4 0.33.0) and turn the fused append off (`E4B_FUSED_KV_APPEND=0`).
  - Every KV append then goes through `quantize_kv_fp8`. Lane B771 measured 0.33.7's fused kernel writing exactly
    `quantize_kv_fp8`'s bytes, and 0.33.0's differing in 21 of 536,870,912 values.
  - **If #413 is the whole step, F reads N's float.**
- **S.** Take e4b 0.37.4 on gnf4 **0.33.6** (every release of the cut except #413's), with the fused append on.
  - **If nothing else in the cut moves K8, S reads O's float.**

## Instrument

**One RTX 5090 on an AMD host.** P84's amendment 1 applies: the known floats were read on AMD hosts, and an Intel
i9-14900K calibrated another attention pack.
- Every process runs P70's harness: `bench/p39/step_decomp.py` and the `bench/p42` hook v6, with P70's env. The harness
  bytes are those `bench/p70` and `bench/p84` pin.
- `E4B_ROUTER_EPI_CAST=0` is exported everywhere.
- The model is `Qwen/Qwen3-30B-A3B` @ `ad44e777`. The K8 is wikitext, 2,048 teacher-forced paged-decode steps at B = 1, on
  window `9ef10d760ad9`.

**Stacks** (registered constants in the runner; the launch commit stages the kit and is recorded, but is not installed):

| stack | e4b | grouped-nf4-gemm | installed |
|---|---|---|---|
| **O** | `c77aab6` (0.37.4) | `5ca1897` (v0.33.0) | first, with the dependencies (P70's and P83's O install) |
| **S** | `c77aab6` | `f879761` (v0.33.6) | over O, grouped-nf4-gemm alone, `--force-reinstall --no-deps` |

**Readings, in order:**

| reading | stack | how | fused append |
|---|---|---|---|
| `O_build` (the control) | O | P70's build: experts calibrated and dumped, attention calibrated live | on (default) |
| `O_rep` | O | P55x's lic arm: `O_build`'s expert pack loaded by fingerprint, attention calibrated live | on |
| `F_1`, `F_2` | O | the lic arm | **off** (`E4B_FUSED_KV_APPEND=0`) |
| `S_1`, `S_2` | S, with its own NF4 bake | the lic arm, with the same expert pack | on |

- **The control runs first.** If `O_build` is not O's known float bit for bit, this box does not reproduce P70's build.
  No comparison with a known float then holds, so the lane reduces to VOID and **stops**.
- **S loads the pack that O built.** The expert pack was byte-identical under gnf4 0.33.0 and 0.33.7 (`0c9955a9…` in
  P83 and P84). 0.37.4's loader checks the pack's identity and fingerprint, not the toolchain it records.
- **Tripwire, per stack:**
  - the pinned commits, read from pip's `direct_url.json`, and the versions (O: 0.37.4 / 0.33.0; S: 0.37.4 / 0.33.6);
  - the router at fp32;
  - the installed `fp8_kv` is the **pre-#413** kernel (no `_quantize_kv_fp32`), and `fp8_kv_append_t1` is present;
  - 0.37.4's `_resolve_fused_append` resolves on by default and off under `E4B_FUSED_KV_APPEND=0`;
  - P70's hook and `step_decomp` import.
- **Stamps.** Every K8 process is stamped beside its receipt with:
  - the router module's reading;
  - the installed versions;
  - `fused_kv_append_env` and `fused_kv_append_resolved` (the resolver's answer under that process's env);
  - whether `fp8_kv` carries #413.
- **Watchdog.** P84's first-calibration-chunk watchdog (900 s) runs on the build.
- **Verdict.** `p85_reduce.py` computes it. Its self-test pins the rule on 20 synthetic cases, and it runs on the box
  and in CI. The known floats are P83's receipts, verbatim, and a test checks that.

## Decision rule (`p85_reduce.py`)

1. **VOID** if any of these holds:
   - `O_build` ≠ O's known mean NLL `1.8511420498367808` (the control failed);
   - any reading is missing, off the registered window or step count, or not stamped fp32;
   - any reading is not stamped with its stack's versions, or its fp8_kv is not pre-#413;
   - any reading's append does not resolve as its row says;
   - any pair differs bit for bit (`O_build`/`O_rep`, `F_1`/`F_2`, `S_1`/`S_2`).
2. **PATH-REFUTED:** F = O's known float. Turning the fused append off changes nothing, so it is not on K8's path, and
   RESULTS-p84's call-path reading is wrong.
3. **CONFIRMED:** F = N's known float `1.8506507749113845` **and** S = O's known float. #413 is the whole KERNEL step.
4. **MIXED:** F = N's known float, but S ≠ O's. The append reproduces N, and something in 0.33.1–0.33.6 also moves K8.
5. **REFUTED:** otherwise, with F neither known float. The append moves K8, but its bytes are not the whole step.

Reported: F − O, F − N and S − O in nats, and the expert pack's fingerprint.

**What follows:**
- **CONFIRMED:**
  - The #674 row records #413 as the cause.
  - The licensed 6.36709 is P70's software's reading with the pre-#413 append. Whether the row names its software or is
    re-read on the current release is the owner's decision.
  - The P81/P82 cast pair (gnf4 0.33.5 vs 0.33.7) has the same suspect. A cast-arm twin of F is possible; it is not
    proposed here.
- **PATH-REFUTED, MIXED or REFUTED:** P84's registered KERNEL follow-up, a bisect of gnf4 0.33.1 → 0.33.6 on e4b 0.37.4,
  starting from whichever release the readings leave open.

## Predictions (written before the data)

- **Stated expectation, not the rule: CONFIRMED.** The reasoning:
  - #413 is the only arithmetic change on the path in the cut.
  - `quantize_kv_fp8`'s arithmetic is the same in 0.33.0 and 0.33.7 (0.33.7 split it into `_quantize_kv_fp32` + the
    cast, with the same operations).
  - B771 measured 0.33.7's fused bytes equal to `quantize_kv_fp8`'s, payloads and scales, on 536,870,912 values.
- **How it can fail while #413 is still the main cause.** The prediction is exact equality. The scale uses Triton's
  default `/` in the fused kernel and torch's divide in `quantize_kv_fp8`; B771 saw no scale differ, but real Qwen3 K/V
  are other data. One differing scale byte would make F miss N's float by a hair, and that reads REFUTED by the rule. F's
  nats against both known floats are reported, so a near miss is visible as one.
- **The control reproduces O's float here.** P70's build read it on three AMD machines. Every pair repeats, and the
  expert pack is `0c9955a9…`.

## Box and cost

Two rentals, in order. The guard exceeds one hour, so the compute rule requires a proving rental first.

1. **`p85-prove-<n>`**: one RTX 5090, **0.4 h guard at ≤ $0.75/h (≤ $0.30)**. `P85_PROVE=1` runs:
   - the refusals;
   - both installs (O, then S over it) with their tripwires and stamps, each stamp exercised with the append on and off;
   - the reducer self-test and an egress probe.

   No model. At most three attempts.
2. **`p85-5090-<n>`**: only after a proof returns rc 0 with its receipts fetched. One RTX 5090, Vast verified/secure,
   `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`, ≥ 200 GB of disk. **Guard 3.0 h at ≤ $0.75/h (≤ $2.25).**
   - One fetch, two bakes, one build (31 min on P84's host) and five lic-arm readings (5.7–6.7 min each there): about
     1.4 h on a fast host.
   - About 3 h on the slowest host the 900 s watchdog admits.

**Lane ceiling $3.00; hard stop $4.00**; both under the $35 per-run cap.

## Rehearsal

The home A2000 (sm_86) cannot run the fp8 paged KV or hold the model, so no K8 can be rehearsed. The rehearsal runs in
throwaway `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel` containers, with the card class, vendor and disk lifted by the
runner's knobs:

| arm | change | expected |
|---|---|---|
| **R** | none | rc 0: both stacks installed and tripwired, P70's harness importing under each, both stamps reading the append on by default and off under the knob |
| **M1** | S pinned to v0.33.7 (`9407d49`), with the version check moved to match | rc 9: the pre-#413 tripwire refuses it |
| **M2** | `E4B_FUSED_KV_APPEND=0` exported to every process | rc 9: the default stamp must read the append on, and refuses |
| **V** | the vendor knob left at its default on the A2000's Intel host | rc 16, before any install |

The rehearsal also includes the driver's dry run and the CI tests (`tests/test_p85_staged_pin.py`), which pin:
- the staged bytes, and harness `o/` against P70's pins;
- the stacks, and the known floats against P83's receipts;
- the router export, and the append knob unset before any install;
- the tripwire and the stamp;
- the order, and the stop on a failed control;
- each reading's env, the vendor refusal, and the driver's dry run.
