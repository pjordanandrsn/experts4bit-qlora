# Gate G-LO synthesis — routed-subset dose-response, pre-decisive-leg

**Session:** 2026-07-11 (UTC ~04:50–06:10). Scope per handoff: freeze preregistrations,
weigh the OLMoE lo arm, determinism probe, debug-arc archaeology, pod-spec arithmetic.
**No decisive-leg arm was launched.** Launch authorization is Jordan's, after this synthesis.

## 1. Item 0 — pre-registrations (committed FIRST, blind)

- `preregistration/prereg_olmoe_lo_weight.json` + `preregistration/prereg_qwen3_lo_fail_interp.json`
  committed together as **`7e988af`** (author stamp 2026-07-11T04:51:41Z), pushed to
  `bench/crossover-surface` before any OLMoE lo output was read.
- **Quarantine attestation:** at commit time the lo bracket was mid-flight on pod
  `4nz7nio7ymukij`: `olmoe_lo_whole_a` ARM_START 04:31:02Z (its completion had not been
  observed by this session); `olmoe_lo_routed` — the rf-bearing arm — had **not been observed
  to start**, so the outputs the weighting rule judges did not fully exist yet. Only SENT
  start/stop metadata was read pre-commit. The commit message carries the same attestation.
- Arm completion vs commit: commit author-stamp 04:51:41Z; olmoe_lo_whole_a completed 04:57:55Z and olmoe_lo_routed (the rf-bearing arm) started 04:57:55Z — **both AFTER the commit**. The rule genuinely predates the data it judges.

## 2. OLMoE lo result, weighted by rule 0.1

**Measured rf (olmoe_lo_routed, 16 per-block STAGEDCNT lines): mean 0.970, range
0.953–1.000.** Phase-0's directly-measured OLMoE curve predicted 0.973 at eff 256 — a
clean transfer (contrast the Qwen3 hi transfer, which ran optimistic 0.80→0.97; the
difference is that Phase-0 measured OLMoE natively and E=64 is saturated regardless of
packing).

**Branch fired: `if_rf >= 0.90` → near-zero evidential weight regardless of outcome.**
The lo bracket's fill mass is ≈3% — a dose-incapable instrument, per the rule frozen
blind at `7e988af` before this arm started. Bracket numbers recorded for completeness
(NOT counted toward the dose-response curve):

| arm | eval trajectory (0/50/100/150) | final |
|---|---|---|
| olmoe_lo_whole_a | 2.725 / 1.646 / 1.445 / 1.428 | 1.428 |
| olmoe_lo_routed  | 2.725 / 1.723 / 1.507 / 1.489 | 1.489 |
| olmoe_lo_whole_b | 2.725 / 1.650 / 1.444 / 1.427 | 1.427 |

Floor |wa−wb| = **0.0010** (clean: signed train wb−wa +0.0029, higher 89/150 ≈ coin-flip).
Routed gap = **0.0615 = 61.5× the floor**, one-sided (train +0.0720, higher 148/150).

**Weighted interpretation (rule 0.1, as frozen): near-zero weight toward the dose-response
curve.** rf 0.970 ≥ 0.90 → dose-incapable; this arm is NOT a curve point, however dramatic
the outcome — that is precisely what the blind-frozen rule exists to prevent.

**Unregistered observation (hypothesis-generating, clearly labeled as such):** the gap
magnitude 0.0615 is numerically ≈ the Qwen3-hi FAIL's 0.062 — the signature the frozen
Qwen3-lo prereg calls **H_DISCRETE** ("per-step corruption independent of fill mass predicts
a comparable gap") — replicated across architecture (E=64 vs E=128), host (3090 vs PRO 6000),
and seq (256 vs 2048), both at ~1–3% fill. Combined with the archaeology (16% fill × 15
steps survived, §4), the active ingredient looks like **horizon (150 steps), not fill mass**.
This raises the decisive leg's diagnosticity: H_DISCRETE predicts a Qwen3-lo gap ≈ 0.06;
H_FILL predicts ≫ 0.06 at the ~30% predicted fill.

**Hi-bracket floor anomaly, plausible mechanical account:** the hi bracket's contaminated
floor (0.022, one-sided) belonged to the only bracket whose whole_a was the FIRST arm on a
fresh pod (model download + first-touch); the lo bracket ran fully warm on the same host and
produced a 0.001 floor. Future brackets: burn a throwaway warm-up before arm 1.

## 3. Item 1 — determinism probe (branch + design consequence)

**Branch fired: NONE of A/B/C cleanly — a fourth outcome, "D".** Full provenance: probe pod
`pob48phbtojnxo` (SECURE 3090, driver 580, `CUBLAS_WORKSPACE_CONFIG=:4096:8` exported
pre-launch). A procedure bug fired first and was fixed: the distro ships
`/usr/lib/python3.12/sitecustomize.py` which **shadows** the dist-packages one, so the first
det_a/det_b pair ran with the hook silently inert (tell: they matched the non-det arm's wall
to within 1 s, and `index_add_` didn't raise). Appended the hook to the distro file; then
**verified deterministic mode was live in the `accelerate` child** (`accelerate launch`
of a probe script printed `are_deterministic_algorithms_enabled = True`, `DETERMINISM_PROBE=1`).

With det mode confirmed active in the trainer child:
- **Not B:** `torch 2.13 index_add_` has a deterministic implementation — in-process test
  under `use_deterministic_algorithms(True)` **RAN, did not raise**. The expert scatter is
  therefore NOT the irreducible-nondeterminism source on this torch.
- **Not A:** the deterministic whole-vs-whole pair (det_a2/det_b2, OLMoE seq 256, 30 steps,
  seed 42) **did NOT collapse** — train loss max|Δ| = 0.010 (2/30 steps bit-exact), eval
  2.375 vs 2.373 (Δ 0.002). That 0.002 eval floor is **identical to the non-deterministic
  warm floor**, so deterministic mode buys nothing here.
- **Mechanism (D):** residual nondeterminism lives OUTSIDE torch's deterministic coverage —
  the **bitsandbytes custom 4-bit-dequant + fused-MoE GEMM kernels**, which
  `use_deterministic_algorithms` cannot see or constrain. This is intrinsic to the
  QLoRA-over-4bit-MoE path, not a config error.
- **Wall:** det arms 663/663 s vs non-det 709/710 s — det slightly *faster*, i.e. within
  host noise; no meaningful deterministic-mode overhead (and no benefit).

**Design consequence (= Branch B's prescription, reached via D):** the floor is irreducible
for this stack; **the decisive leg uses n≥3 whole arms per regime, floor = sample spread.**
A near-exact n=1 A/B is off the table. Reassuring corroboration: the two *warm* whole-vs-whole
floors already measured (Qwen3-hi 0.002, OLMoE-lo 0.001) are as tight as the deterministic
one — so a warm n≥3 bracket is the right, sufficient instrument. The lone contaminated floor
(OLMoE-hi 0.022, one-sided) was the cold first-arm-on-fresh-pod; prescription: burn a
throwaway warm-up before arm 1, and use ≥3 whole arms so one cold outlier can't set the floor.

## 4. Item 2 — debug-arc archaeology (zero compute)

**Recovered from the session transcript** (provenance: `28c16bbd…jsonl`, `/root/g/*.yaml`
config-generation commands on the 2026-07-10 debug pod):

- The zeros-explosion observation, the 3-step grad_norm verification (routed-vs-whole
  5.4% ≤ whole-vs-whole 6.7%), and the 15-step convergence check (`mean|whole−routed|
  = 0.018`) **all ran OLMoE at `sequence_len: 64`** (r15/w15 sed-derived from routed.yaml,
  changing only `max_steps: 3 → 15`).
- **rf at that operating point was not instrumented** (STAGEDCNT did not exist yet).
  Phase-0's directly-measured OLMoE curve gives **rf(64) = 0.836 → fill ≈ 16%**
  (provenance: `bench/access-pattern` `access_reduction.json`, olmoe_E64.pts).
- **Verdict for the lo-leg prior:** the real-fill fix already survived one
  **moderate-dose (≈16% fill)** exposure — weak evidence (grad-norm proxy, 15 steps,
  OLMoE, E=64), but it moves the prior: per-row fill toxicity that is large and fast
  would likely have shown at 16% × 15 steps. The Qwen3-hi FAIL (≈3% fill × 150 steps,
  31× floor) therefore points at **horizon and/or architecture** as necessary
  ingredients, not fill mass alone. The lo leg is **live-fire, not once-survived**:
  no prior exposure combines real dose × 150-step horizon × any architecture.

## 5. Item 3 — pod-spec arithmetic for the decisive leg

Measured facts (provenance: pods `4nz7nio7ymukij` [SECURE 3090, sweep] and the
`yqwut74yynzsf3` [community 3090, killed], 2026-07-11):

| quantity | value | source |
|---|---|---|
| Qwen3-30B expert home (pinned target) | **14.50 GB** | plugin homed line, whole_lo_a log |
| Host RAM, SECURE 3090 | 1007 GB total / ~396 free | `free -g` on pod |
| RLIMIT_MEMLOCK, SECURE 3090 | **8 MB hard, not raisable** | `ulimit -Hl` + raise attempt |
| PCIe negotiated, SECURE 3090 | **gen1 x16** (max gen4) | nvidia-smi pcie query |
| 16 GB pinned-alloc test (probe host, same 8 MB memlock) | **SUCCEEDS, is_pinned=True** | probe pod `pob48phbtojnxo` pin_test |
| H2D rate, pinned / pageable (probe host) | **22.98 / 6.72 GB/s** (gen4-class; the "gen1" nvidia-smi reading is idle downclock) | probe pod pin_test |
| Qwen3-30B peak VRAM, seq 256, train (no eval logits) | **16.4 GiB** | 2-step no-eval diag, expandable_segments |
| Qwen3-30B eval VRAM with `prediction_loss_only` | **flat 4.2 GB** across 992 batches | smoke run memory sampling |
| Qwen3-30B training rate on that 3090 | **0 steps in 14 min** (timeout, unattributed) | diag run |
| OLMoE on the same host | trains normally (~6 s/step incl. evals) | sweep arms |

Arithmetic: GPU memory is NOT the disqualifier (16.4 GiB peak < 24 GB, eval flat with the
committed `prediction_loss_only` deviation). Host RAM is NOT the disqualifier (1 TB ≫ 14.5 GB
+ headroom). **Both candidate mechanisms for the training-step pathology were REFUTED by
direct measurement on an equivalent host** (probe pod, same SKU, same 8 MB memlock): a 16 GB
pinned alloc succeeds (RLIMIT_MEMLOCK does not gate cudaHostAlloc) and pinned H2D runs at
22.98 GB/s (the nvidia-smi "gen1" is idle downclock). At that rate the 14.5 GB whole-stack
stage is ~0.6 s, so the observed **≥14-min training step is UNATTRIBUTED** (observed twice:
the eval-boundary OOM run and the no-eval 2-step diag that hung at step 0 with GPU at
16.4 GiB and no OOM). The empirical fact still disqualifies the measured 3090 host for the
30B subject — but as "failed twice, mechanism unknown," not "physically can't."
OLMoE (4.7 GB home) is cleared on the same host class (trains normally).

```
qwen3_lo_runs_on: "pro6000"
```

- **PRO 6000 (proven host class):** the primary ran 150-step seq-2048 arms in 1044–1230 s
  each. The lo bracket (3 arms × [load ~8 min + 2 evals ~35 min + 50 steps ~5 min]) ≈ 2.5 h
  → **≈ $4.7 at $1.89/hr** (n=1 bracket) — halves to ~$2.4 if the determinism branch allows
  a 2-arm design.
- **3090 (option B, not recommended):** $0.46/hr, and the probe host measured healthy
  (pin 16 GB OK, 22.98 GB/s pinned H2D) — so the earlier "3090 can't" is really "3090
  failed twice, unattributed." A pre-gated attempt (rent, run a 10-min 2-step Qwen3
  staged-train gate, proceed only on pass) costs pennies to try and ~$4.7 anyway on
  fallback to the PRO 6000. Decisiveness argues for the proven host; economy argues for
  the gate-first option. Jordan's call at launch.

## 6. Proposed decisive-leg design (NOT launched — Jordan authorizes)

**One-line recommendation:** run it, on the PRO 6000, as a warm n≥3 floor bracket — it is the
only remaining configuration that can separate H_FILL from H_DISCRETE, and every cheaper
substitute has now been exhausted (OLMoE can't reach low rf; the 3090 can't run 30B; determinism
can't tighten the floor).

- **Card:** PRO 6000 WS (SECURE) — the proven host. Gates already baked into `build_env.sh`:
  driver ≥580, memlock unlimited/≥16 GB, torchvision pin, `prediction_loss_only` patch. The
  3090 is not recommended (30B training failed twice, unattributed) but remains a pennies
  gate-first option if cost dominates: rent, run a 10-min 2-step staged-train gate, fall back
  to the PRO 6000 on fail.
- **Subject:** Qwen3-30B-A3B, seq 256 lo bracket, the committed lo configs (+ the two recorded
  deviations). This is the **decisive dose-response point**: Phase-0 predicts rf 0.70 → ~30%
  fill, the only regime with real fill mass at 150-step horizon on the architecture that already
  FAILED at high rf.
- **Design:** **1 warm-up throwaway → 3 whole arms → 1 routed(+STAGEDCNT) → then read.**
  Floor = spread of the 3 whole finals (max−min or 2·stdev); routed PASS iff its gap to the
  whole-mean ≤ that spread. n≥3 because determinism can't collapse the floor (Branch D) and the
  one cold-floor contamination this session came from a first-touch arm.
- **Pass/FAIL meaning (pre-registered, `prereg_qwen3_lo_fail_interp.json`):** FAIL = fill pathway
  scales with dose → routed stays opt-in, fix = mask the checkpointed-backward recompute dequant
  to routed rows (NOT abandonment). Discriminating prediction already registered: H_DISCRETE →
  gap ≈ 0.06 (same as hi); H_FILL → gap ≫ 0.06 at ~30% fill. **This leg is the experiment that
  tells H_FILL from H_DISCRETE** — the whole point of the "both" decision.
- **Cost:** ≈ $6–7 (warm-up + 3 whole + 1 routed = 5 arms × ~$1.3 on the PRO 6000; the seq-256
  evals dominate wall). Predecessor primary arms were 1044–1230 s each at seq 2048; seq-256
  evals are heavier per the OLMoE timing, so budget ~3 h.
- **Positive controls:** per-block STAGEDCNT (measured rf vs registered 0.70 — the second
  Phase-0 transfer test), bleed control (whole arms emit no routed INFO line), step-0 eval
  fingerprint identical across all arms.
- **Secondary (cheap, high-value): a horizon control.** §2 + §4 point at horizon, not fill, as
  the active ingredient. If the budget allows one extra arm, a routed arm at **15 steps** (the
  original debug-arc horizon) at this same seq-256/rf-0.70 point would test horizon-dependence
  directly for ~$0.5. Proposed, not assumed.

## 7. Constraint compliance

- OLMoE hi-arm FAIL: recorded verdict untouched; the near-zero-power amendment note stands
  as written (prereg `amendment_both_legs.interpretation_note` + session log). Not reframed.
- No GEMM Phase-1 or other workstream work.
- No Qwen3-30B lo arm launched.
- Pods at session end: **ZERO** — both terminated (probe HTTP 204; sweep pod torn down by autopilot HTTP-OK); RunPod pods list returns `[]`. Verified twice.
