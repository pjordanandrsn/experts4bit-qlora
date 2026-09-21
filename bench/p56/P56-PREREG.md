# P56 — pre-registration: is Gemma-4's training-parity FAIL a defect in the fused path, or a floor the band never measured?

**Status: registered before any P56 data exists.** Written 2026-09-21. The run it
authorises had not been launched when this merged; the launch cites the merge commit.

**Renumbered P55 → P56 before any data existed, and this is the record of it.**
This lane was drafted and pushed as P55. While it was in review a different lane —
`bench/p55/`, the 49.9 GiB shard / host-RAM class question for the Gemma-4 LOAD
fault (#344) — merged to `main` first and took the name. Two different questions
under one label is a register that cannot be read, which is what TP4-PREREG
amendment 5's erratum is about. The lane that landed first keeps the number; this
one moves, and the move is recorded rather than done silently. Nothing of the other
lane's was modified: its files are `main`'s, byte-for-byte. Both lanes are Gemma-4
and they do not overlap — #344 is why the model fails to LOAD on some hosts, #558 is
why its fused training arm disagrees with its own reference once it has loaded.

## The standing fact this lane addresses

`docs/claims.json` row **`e4b.parity.gemma4.train-internal`** (added by #635):
Gemma-4-26B-A4B-it's fused expert training path reads **0.08257 nats** final
held-out and **0.12421** median step-wise from e4b's OWN dense per-expert
reference, against tp1's **0.05 / 0.05** band → FAIL. It read 0.09037 / 0.11801
on the previous kernel cut, so it survived a kernel change. Every other family
passes the same band on the same fixture at **0.0006–0.002**
(`bench/tp4/RESULTS-tp4-p46cut.md`). e4b#558 was closed as *explained, not
resolved*; the Gemma-4 fused arm is not quoted anywhere.

## What has already been established on CPU, before renting anything

Three results, all reproducible without a GPU, recorded here because they are what
narrows this lane from "find the bug" to one fork.

1. **The composition is exact, and it is family-blind.**
   `tests/test_fused_train_parity.py::test_batched_composition_is_family_blind`
   runs e4b's own parity contract at each family's REAL expert configuration
   (activation, E, top-k, hidden:inter ratio) with a real softmax-top-k-renormalised
   router weight distribution, scoring both arms against an fp32-compute arm over
   the same quantized weights. Every family — Gemma-4 included — sits at the same
   bf16 floor (~6e-3 vs fp32), the two paths agree with each other to **1e-7** on
   gradients, and crossing activation with shape (`gemma4-shape-silu`,
   `qwen3-shape-gelu`) separates the two candidate causes and clears both.
   **So the group-sorted scatter, the epilogue placement and the top-k weighting
   are not where Gemma-4's gap lives.**

2. **What that does NOT cover.** It exercises the KERNEL-FREE lane
   (`enable_batched_train`). `enable_fast_train`'s kernel is CUDA-only and is
   untouched by it, and the two lanes do not carry the same error:
   `bench/dgrad-gate/RESULTS-dgrad-gate.md` measured the fused lane's composed
   gradient error at **~5e-2** against the reference at 48 layers where the
   kernel-free lane's is **~4e-3**. The CPU result rules out the composition. It
   does not rule out the kernel.

3. **Gemma-4 gives its routed expert branch full-strength entry to the residual
   stream, and no other family in the suite does.** `Gemma4TextDecoderLayer`
   passes the expert output through its OWN `post_feedforward_layernorm_2` before
   adding it to the dense-MLP branch, so the branch's contribution is renormalised
   and then rescaled by a learned weight — i.e. it is **independent of the expert
   output's own magnitude**, and a relative error in the expert output is not
   diluted by the branch being a small correction. Read from the released
   `google/gemma-4-26B-A4B-it` shards (norm tensors only, no GPU): over its 30 MoE
   layers the routed branch's weight-RMS is a **median 0.675×** the dense branch's
   (range 0.348–0.937). The routed branch is a co-equal contributor, not a trim.

## The question, stated so either answer is a result

tp4's `parity()` compares `|Δ|` against **0.05, and against zero** — there is no
floor term in `bench/tp4/tp4_reduce.py`. That band's provenance is one model's five
datasets (`bench/flagship-matrix/RESULTS-flagship-matrix.md`, worst cell 0.0149).
It has never been calibrated against a per-family measured floor.

This project already holds the doctrine that says that is not enough. From
`bench/hybrid-g9`: two arithmetically-equivalent forwards of an MoE model disagree
because bf16 rounding **flips which experts the router picks**, and the flipped
tokens carry ~39× the KL of the unflipped ones — *"so a parity delta must be read
against a measured FLOOR, not against zero. The floor is model-specific."* That
was established for SERVING. The training band never got the same treatment, and a
training comparison is strictly worse: the two arms' weights diverge after step 1,
so from step 2 onward they route differently and the divergence compounds.

**So: is 0.08257 nats the fused path being wrong, or is it how far apart two
equally-correct training runs of THIS model land over 20 steps?**

### One quantity, under two names — recorded before the draw so the read is unambiguous

The claim row gives its unit as *"nats (|fused - dense reference| final **held-out**
loss, N=20 at tp4's field fixture)"*. The 0.08257 it carries came from
`tp4_reduce.parity()`, which reads **`loss_last`** — the final **train** loss.
tp1's B2/C2 rule is the train loss, and `bench/flagship-matrix/RESULTS-flagship-matrix.md`
already had to correct a published table for exactly this substitution, noting that
eval loss is *"a smaller, easier number here, so the original table flattered the
result."*

So the row's wording is wrong, not its number. This lane computes the band on the
**train** loss, exactly as `tp4_reduce` does, so it is comparable to the standing
row; `p56_reduce.py` reports the **held-out** delta beside it, named, so nobody has
to guess again. Correcting the row's unit string is a separate change against
`docs/claims.json` and is not smuggled into this lane.

## The instrument: a four-point ladder of increasing arithmetic error

All four arms train the same fixture on the same box in one session, from the same
`init_sha`, on the same tokens. They differ ONLY in the arithmetic used for the
expert projections, and they can be ordered by how much per-op error each carries:

| arm | path | per-op error vs the reference | can it fall back? |
|---|---|---|---|
| `reference_attn4` | `ExpertsLoRA.forward`, per-expert loop | 0 by definition | — |
| `batched_attn4` | `enable_batched_train` — torch + bnb, group-sorted | ~4e-3 composed at 48 layers | **yes** (pad waste) |
| `fused_attn4_nodgrad` | `enable_fast_train(dgrad=False)` — fused forward, EXACT per-expert decode backward | forward fusion only | no |
| `fused_attn4` | `enable_fast_train(dgrad=True)` — the shipped arm, every tp1–tp4 row | ~5e-2 composed at 48 layers | no |

**The ladder is the whole design.** A real defect in the fused path is a function
of the arithmetic: the four deltas should ORDER with the per-op error, and the
reference↔batched pair — the smallest perturbation available — should be small. A
trajectory floor is not a function of the arithmetic at all: any perturbation above
the router's flip threshold lands in the same place, so the four deltas should be
comparable and LARGE regardless of the ladder.

### Registered predictions

- **P1 — all four arms reach VALID.** Same `init_sha`, same trainable count, same
  tokens sha, `n_patched == 30` on each accelerated arm, C1 frozen bytes bit-exact.
- **P2 — the batched arm's engagement is REPORTED and gates its own reading.**
  `batched_fallback_stats` is written to the receipt; `fallback_calls > 0` VOIDs
  that arm rather than being read as a number. A fallback lands on the reference
  forward invisibly, so an arm that fell back on every call would read ~0.000
  parity while measuring nothing. On CPU at each family's real routing the path
  engages with zero fallbacks, but that is a balanced synthetic router; a trained
  router is skewed and MAY trip the pad-waste guard, and that outcome is a row.
- **P3 — the VERDICT IS NOT PREDICTED.** Both outcomes are real results:
  - deltas ordered by the ladder, reference↔batched small ⇒ the divergence tracks
    the arithmetic, the kernel is implicated, and #558 becomes a localisable defect
    in `grouped-nf4-gemm` at Gemma-4's shapes;
  - deltas comparable and all large ⇒ the divergence does not track the arithmetic,
    it is Gemma-4's trajectory floor, and the 0.05 band cannot attribute any part
    of it to the fused path.

  Registering a hoped-for direction here is exactly what would make the reading
  worthless — the same reason TP4-PREREG amendment 5 registered only readability.

- **P3b — a third outcome, registered because the reducer must not fold it into
  either of the two above.** Every rung may land INSIDE the band, i.e. the standing
  0.08257 does not reproduce on this box at all. The standing row was measured on
  one rented box (Vast 51645512) and this is a different draw. **That reading is a
  result about the standing row's reproducibility and is NOT evidence that anything
  was fixed**; it would leave #558 open and needing a third draw. The reducer
  reports it as `DID-NOT-REPRODUCE`, separately from `MIXED`.

- **P4 — NOT registered, and stated so it is not smuggled in later:** no claim
  about which of `fused_attn4` and `fused_attn4_nodgrad` is faster. This lane is a
  correctness reading; the dgrad kernel's speed is already measured in
  `bench/dgrad-gate/` and no step time from this draw is quoted as a position.

### A prior that points one way, disclosed because it is mine and it is not registered

`bench/dgrad-gate/RESULTS-dgrad-gate.md` already ran three of these four rungs on
**Qwen3-30B-A3B**, and its last column is directly relevant:

| arm | composed grad error | loss median Δ |
|---|---|---|
| `fast_train` | 4.97e-02 | 0.0016 |
| `fast_train_dgrad` | 4.99e-02 | 0.0011 |
| `batched` | **3.79e-03** | **0.0017** |

The batched lane carries **13× less** gradient error than the fused lane and lands
at the **same** loss delta. So on a family that PASSES, the loss delta already does
not track the per-op error. That is the FLOOR pattern, at a magnitude small enough
that nobody had to care.

**This makes one of the two registered outcomes the less surprising one, and I am
saying so before the data exists rather than after.** It does not license
predicting it: the Qwen3 rows are a different fixture, a different model and a
step-0 composed-gradient measurement rather than a 20-step trajectory, and the
whole question is whether Gemma-4 departs from that pattern. P3 stands — the
verdict is NOT registered, and a reading that comes back ordered is exactly as
publishable as one that comes back flat.

### What this draw may NOT report

No head-to-head position for Gemma-4 against any other framework: Unsloth, HF and
axolotl arms are deliberately absent, so `tp4_reduce.py` reads VOID or NOT_RUN for
those pairs and **no ratio is quotable from this draw**. It says nothing about any
other family.

## Fixture and knobs

tp4's field recipe, unchanged and unchanged deliberately so the numbers compare to
the standing row: alpaca, seq 2048, micro-batch 1 × accum 4, r16 / α16, lr 2e-4,
AdamW-8bit, linear warm-up 5, seed 3407, **N = 20**, `--attn-4bit 1`, the same
`tp4_alpaca.py` tokens and registered `DS_ALPACA_SHA`. Model
`google/gemma-4-26B-A4B-it` at tp4's pinned revision.

Restricted to the parity arms by the knobs the run script already registers
(TP4-PREREG amendment 5): `TP4_FAMILIES=gemma4` plus a `TP4_SKIP` covering the
unsloth and hf arms. The two new arms are reached by `tp4_arm.py --arm batched`
and `--arm fused --dgrad 0`, both added for this lane and both dry-run on CPU
before this registration merged (`--selftest`, and the family test above).

**Kernel cut is PINNED to the one the standing row was measured on.**
`GNF4_SHA=9206352f7c49d76432824f7ce8872f0b285f2a4e` (grouped-nf4-gemm v0.32.1),
which is what `tp4-c-parity-2` ran and what `e4b.parity.gemma4.train-internal`
records. `tp4_drive.sh`'s built-in default is a different commit
(`d9fd170d`, main on 2026-09-10); taking it would change the kernel and the arm
set in the same draw, and then neither could be read against the standing row.
The manifest passes the pin explicitly.

**Micro-batch is NOT a knob this lane varies, and the reason is recorded because
it was my first design.** A `mb2×accum4` vs `mb1×accum8` pair looks like a clean
nuisance — same tokens, same order, same optimizer steps — but `tp4_arm.py`
right-pads variable-length rows inside a micro-batch and takes `out.loss` as the
mean over that micro-batch's real tokens. With rows of unequal length the two
shapes therefore optimise *differently weighted* objectives, not the same one by a
different arithmetic route. It would have been a nuisance pair that is not a
nuisance, and its number would have been uninterpretable.

## Cost, caps and authorization

One RTX 5090, Vast verified/secure, at the standing $0.69/h ceiling. Four arms plus
their prologues (e4b#548 is open about the ~30-minute prologue, which dominates:
the training itself is roughly 1 + 14 + 3 + 14 minutes) ≈ **3.5 h ≈ $2.42**,
inside the ≤$3 authorised for this lane and well inside the $35/run cap. Guard 4 h,
so the approval line is 4 × 0.69 = **$2.76** — the guard, not the estimate, is what
the launcher charges against, a distinction P41 had to learn.

Arm alarms are the family's registered ones, unchanged: the new `fused --dgrad 0`
rung takes the fused arm's 3600 s and `batched` takes the reference's 5400 s,
because it runs at about reference speed. `can_run`'s 900 s margin drops arms from
the END of the list as the window closes, which is why the cheap rung is ordered
before the expensive one.
A proving run precedes it, per the standing rule for any guard over one hour, and it
is `bench/p56/p56_prove.sh` — which does the second job a proving run should do and
**measures the one quantity that decides whether this draw can run at all on the box
it draws: host RAM against Gemma-4's single 49.9 GiB shard.**

That quantity is not in any pre-flight, and it is the sibling lane's finding: e4b#344
records this checkpoint failing on 2 of 6 rented 5090s with driver and GPU both
refuted, and `bench/p55/P55-PREREG.md`'s host fingerprint orders every one of those
outcomes by host RAM — 30 GiB refuses the map outright, 64 GiB maps it and dies
opaquely, 96 / 125 / 188 GiB pass. A P56 draw on a low-RAM box loses all four arms to
a load fault that is not P56's question, and the $2.42 with it.

The proving run **reports and does not refuse** on it: a run that can only say "pass"
and one that can only say "fail" look identical on a green result (P41, 2026-09-07).
The go/no-go on the registered draw is a reading of the printed number. Both branches
of that reading were driven on CPU against synthetic 188 / 64 / 30 GiB boxes before
this merged.
Authorization: the owner's standing directive for the training/throughput campaign
(adertha-agents#110), under "use pods as needed" within the caps.

Teardown proven by the launcher; receipts under
`receipts/experts4bit-qlora/2026-09-21/<run-id>/`, read by `bench/p56/p56_reduce.py`
under its own rules. Nothing in the results page is hand-transcribed from a log.
