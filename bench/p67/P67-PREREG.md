# P67 — pre-registration: training parity read against a per-family floor, not against zero (e4b#713)

**Status: registered before any P67 draw exists.** Written 2026-09-23. The rental this document authorises
has not been launched; the launch cites this document's merge commit. Two things WERE read before registering,
and both are disclosed below because they informed the design, the way grouped-nf4-gemm's
`kernel/PREREG-b393-combine-reduce-bitwise.md` discloses its rehearsal:

1. **the existing training-parity receipts, re-read on CPU** (tp1, tp4, tp4-c, P56) — §"What the existing
   receipts show";
2. **a rehearsal on the NAS A2000** with a small MoE — §"Rehearsal". A rehearsal is not a reading.

Neither moves any status, and nothing in `docs/claims.json`, `docs/capabilities.json` or `docs/STATUS.md`
changes in this lane. Those move only after the registered draw is read.

Authorization: the owner's authorization for a Gemma-4 rental, redirected here by the closing comment on
e4b#703 — "pre-register how the floor arm is chosen, how many draws, and the margin; re-read tp1, tp4-c and P56
under it; rent only the floor draws that the registration shows are missing."

## The standing fact

tp1's internal training-parity rule (PREREG-flagship-matrix B2 / -model2 C2), as `bench/tp4/tp4_reduce.py:parity`
applies it: an accelerated arm PASSES iff **|Δ final train loss| ≤ 0.05 AND median step-wise |Δ train loss| ≤
0.05** against the family's own `reference` arm, same box, same session. The comparison is against **zero**:
there is no floor term. Its provenance is one model's five datasets (worst cell 0.0149).

On Gemma-4-26B-A4B-it at tp4's field fixture that rule cannot decide anything. Every accelerated path fails it,
the kernel-free one included, on three boxes:

| session | arm | Δ final train | median step \|Δ\| | constant band |
|---|---|---|---|---|
| `tp4-c-parity-2` (2026-09-19) | `fused_attn4` | 0.08257 | 0.12421 | FAIL |
| `p56-gemma4-ladder-3` (2026-09-22) | `batched_attn4` (kernel-free) | 0.05425 | 0.08487 | FAIL |
| `p56-gemma4-ladder-3` | `fused_attn4_nodgrad` | 0.10274 | 0.11678 | FAIL |
| `p56-gemma4-ladder-3` | `fused_attn4` | 0.10223 | 0.10639 | FAIL |
| `tp4-c-4` (2026-09-10; 8 held-out rows, see below) | `fused_attn4` | 0.09037 | 0.11801 | FAIL |

Every other family passes the same rule at 0.0002–0.004 on tp4's fixture and 0.008–0.017 on tp1's. So
`training_support.gemma4_text` holds the attention-4-bit configuration at **not supported** — not because its
arms fail to run (115/115 structural projections, VALID arms), but because the only registered band cannot tell
a defect from the model's own sensitivity (e4b#713, `e4b.parity.gemma4.train-floor`).

### One quantity, under one name

Every number in this lane is on the **TRAIN** loss: `loss_last` (the final optimizer step's train loss) and
`losses` (one per step), exactly as `tp4_reduce.parity` reads them. The held-out delta is reported beside each
row, named `d_heldout`, and **gates nothing**. This needs saying because it has gone wrong three times: the claim
row `e4b.parity.gemma4.train-internal` gives its unit as "final held-out loss" while carrying a train-loss number
(P56, "One quantity, under two names"); `bench/flagship-matrix` had to correct a published table for the
substitution; and e4b#713's own text says "final held-out |Δ|". The band is the train loss. `p67_reduce.py`'s
`deltas()` returns `d_final_train`, `d_med_train`, `d_heldout` and `d_step0_train` under those names, and
`tests/test_p67_reduce.py::test_the_band_reads_the_train_loss_never_the_heldout` pins that a 0.5-nat held-out
delta moves no verdict.

## Design question 1 — what is the floor arm?

A floor is how far two equally-correct training runs of this model, on this fixture, land apart. Two candidates.

**(a) The smallest-perturbation accelerated arm, `batched_attn4`** (P56's recommendation). It is the cheapest
way to get a number and P56 already has one draw of it (0.05425 / 0.08487). Against it:

- **It is not independent of what it judges.** `enable_batched_train` and `enable_fast_train` share e4b's routing
  gather, the group-sorted scatter, the top-k weighting and the LoRA composition. P56's CPU result pins that
  composition to 1e-7 between the two lanes and to the bf16 floor against fp32 — at one step, on synthetic
  routing. A defect the two share at trajectory scale would be absorbed into the floor and widen the band by
  exactly the size of the defect.
- **It cannot judge itself.** `batched_train` is a shipped path with its own `training_support` cell. A band
  defined by the batched arm reads that arm against itself, which is 0 by construction: a vacuous PASS.
- **It is not the precedent it is cited for.** The serving side's floor (`docs/METHODOLOGY.md` §13.1) is the
  **reference** — the oracle — re-done in a different arithmetic order (chunk 64 against chunk 128, chunked against
  one full forward). No accelerated path is part of it.

**(b) The reference against a perturbed reference.** The SAME `ExpertsLoRA.forward` per-expert loop, same box,
same session, same `init_sha`, same tokens, run again with one change that is correct by construction. Against it,
two questions, answered:

- **Are two reference runs on one box simply bit-identical (floor = 0)?** Unknown from the receipts: no session
  ever ran the reference twice. What they do show (§"What the existing receipts show") is that the reference does
  not reproduce itself **across** boxes even with the identical e4b library tree: Granite (tp4-a / tp4-d) and Qwen3
  (p46-qwen3lora / tp4-b-p46cut-4) agree bit-for-bit at print precision through step 1 (Granite: step 4) and then
  diverge, to 0.00057 / 0.00090 and 0.00198 / 0.00093. So something invisible at step 0 grows. That is not
  evidence about one box, so the design does not rely on it: the floor arm carries its own perturbation, and a
  plain repeat is run beside it to answer the determinism question on the target box.
- **Does a perturbation exist that is correct and above the router-flip threshold?** The registered one is
  `E4B_REFERENCE_EXPERT_ORDER=perm:<seed>` (`experts4bit_qlora/lora.py`, default off): the per-expert loop visits
  the hit experts in a fixed pseudo-random order instead of ascending id. Every expert's projections are computed
  exactly as before; only the order in which their contributions are summed into the fp32 accumulator (forward),
  and the order autograd sums the input gradient over experts (backward), changes. Both are sums of the same terms,
  so the change is exact in real arithmetic. It shares no code with any accelerated path.
  "Above the flip threshold" is made operational rather than argued: a floor draw whose 20 step losses are
  bit-identical to the reference's is REFUSED as a zero floor. On CPU the switch provably moves bits and equals the
  default to 1e-5 relative on the output and every gradient (`tests/test_reference_expert_order.py`); on the A2000
  rehearsal every perm seed moved a real MoE's train-loss trajectory, two of three already at step 0 (§"Rehearsal").

**The cost of (b), stated rather than hidden: it is the stricter floor.** A reorder perturbs rounding at the fp32
accumulator and the bf16 input-gradient sum; an accelerated path changes rounding inside every expert GEMM. P56
found Gemma-4's divergence ordered by per-op error and sublinear in it, so a reorder floor will likely sit BELOW
the batched gap. For a licence that is the right direction: an accelerated path is licensed when it is
indistinguishable from the reference's own reordering noise (or inside the tolerance), not when it is
indistinguishable from another accelerated path.

**Decision: (b) governs. Only (b) is a floor.** (a) is a **judged arm** — it runs in the same session and is read
against the (b) floor like the fused arm is. Beside fused it is P56's ladder read against a measured floor instead
of zero, which is what separates a fused-kernel defect from model sensitivity (§"The decision rule"). There is no
"both, with a rule for which governs": one floor, from one arm class.

## Design question 2 — how many draws

A **floor draw** is one pair (`reference_attn4`, X) from ONE session, where X is `reference_attn4_repeat` (the
plain repeat) or `reference_attn4_perm<s>` for s ∈ {1, 2, 3, 4}. A draw is **admissible** only if all of these hold
(`p67_reduce.floor_draw`, each branch driven in `tests/test_p67_reduce.py`):

- both receipts VALID: status ok, C1 frozen bytes bit-exact, one train loss per step;
- the same fixture, `init_sha`, tokens sha, trainable count and step count as the reference;
- the same session (a cross-box pair spans boxes and commits; it is disclosed, never admitted);
- proof of what ran, read from the loop's own counter (`reference_order` on the receipt): a perm draw must show
  `requested == order == perm:<s>` on `calls > 0`; the repeat and the reference must show no re-ordering;
- **not bit-identical** to the reference at print precision (a zero floor is the band against zero again).

Five candidates are drawn; **N_MIN = 3** admissible draws are required. Fewer is **NO-FLOOR**: the band cannot be
applied, and that is a result, never a FAIL. If the repeat turns out bit-identical it answers the determinism
question and drops out; the four perms still clear N_MIN with one to spare.

**How the spread enters.** The floor's upper edge per quantity, `F_hi(q) = max over admissible draws of D_q`,
is what the band multiplies. The lower edge and the spread ratio are reported beside it and gate nothing: with
three to five draws a standard deviation is not an estimate, and the maximum is the most conservative (widest)
statistic the draws support.

## Design question 3 — the margin

Registered, and fixed in `p67_reduce.py` (`TOL`, `K`, `N_MIN`, checked against this text by
`test_registered_constants_match_the_prereg_text`):

- TOL = 0.05
- K = 3
- N_MIN = 3

```
B(q)  = max(TOL, K * F_hi(q))          for q in {final, med}
PASS  iff  D_final <= B(final)  AND  D_med <= B(med)
```

**Why K = 3.** Under the null that a judged arm is "just another correct reordering" (its D exchangeable with the
floor draws), the probability that it exceeds K · max of n draws, per quantity, by simulation (400,000 trials):

| n | K | half-normal | exponential | lognormal σ=1 | lognormal σ=1.5 |
|---|---|---|---|---|---|
| 3 | 1 | 0.251 | 0.251 | 0.250 | 0.250 |
| 3 | 2 | 0.066 | 0.101 | 0.108 | 0.146 |
| 3 | 3 | 0.025 | 0.051 | 0.058 | 0.101 |
| 4 | 3 | 0.011 | 0.029 | 0.040 | 0.074 |

K = 1 ("below the floor", the serving rule's wording) would FAIL a noise-equivalent arm a quarter of the time at
n = 3. K = 3 at n = 4–5 holds the per-quantity false-FAIL rate to 1–7 % across tails as heavy as lognormal σ = 1.5
(Gemma-4's three cross-box fused pairs below span 37×, so heavy tails are the realistic case). Two quantities
ANDed roughly double it. A correct accelerated arm carries a larger seed than a reorder, so its real false-FAIL
rate is higher than this table: the table is the lower edge of that rate, stated as such.

**Why `max(TOL, ·)` and not a pure floor band.** A difference fails a licence when it is both *detectable* (above
the model's own noise) and *material* (above the practical tolerance). The floor is a detectability limit; tp1's
0.05 is the tolerance, and it was never the defect — comparing against **zero** was. So the floor term widens the
band where the model's own reordering noise exceeds the tolerance, and never narrows it below. A pure floor band
(`B = K · F_hi`) would FAIL a path that is measurably different from a reorder at a cost of, say, 0.004 nats —
which says the arithmetic differs (it does, by design) and nothing about whether training with it is acceptable.
What such a path gets instead is recorded, not hidden: every judged row carries `carried_by` (`tolerance` when both
D ≤ 0.05, else `floor`) and `detectable` (D > K · F_hi on either quantity).

**Consequence, registered: without a floor, a constant-band PASS is still a floor-band PASS** (B ≥ TOL by
construction), and a constant-band FAIL is NO-FLOOR. Floors are therefore needed only where the constant band
fails. That is what sizes the rental (§"Which floor draws are missing").

**Disclosure — what was seen before the margin was fixed.** Before choosing K and N_MIN I ran an exploratory scan
of every same-fixture pair in the existing receipts (the numbers in §"What the existing receipts show", including
the cross-session pairs). `p67_reduce.py` — with the constants above — was run on those receipts only afterwards.
Two things keep that from steering the rule: K comes from the false-FAIL table above, not from those numbers; and
**no verdict in the re-read depends on K or N_MIN**, because no existing session holds a single admissible floor
draw. Every Gemma-4 row is NO-FLOOR at any K; every other row is a constant-band PASS, which the rule makes a PASS
at any K. The margin cannot have been fitted to an outcome it does not change.

## Design question 4 — the families

The band applies to every family with training-parity receipts. The registered draw measures a floor for Gemma-4
only; for the others the band's verdict follows from the rule without one.

| family | receipts | constant band today | P67 verdict without a floor | predicted if a floor were measured |
|---|---|---|---|---|
| Granite-3.1-3B-A800M | tp4 N=20 (p46cut-A), N=60 (tp4-a, tp4-d); tp1 fused + batched | PASS (0.0003–0.0027 tp4; 0.013–0.017 tp1) | **PASS**, carried by tolerance | F_hi ∈ [0.0003, 0.005]; the tp4 fused gap **not** detectable |
| OLMoE-1B-7B-0924-Instruct | tp4 N=20, N=60; tp1 fused (batched VOID) | PASS (0.0002–0.002; 0.013) | **PASS**, tolerance | F_hi ∈ [0.0003, 0.005] |
| Qwen3-30B-A3B | tp4 N=20 (p46-qwen3lora, p46cut-4); tp1 fused (batched VOID) | PASS (0.0014–0.0024; 0.013) | **PASS**, tolerance | F_hi ∈ [0.0005, 0.005]; the tp4 fused gap **not** detectable |
| Qwen3.6-35B-A3B (`qwen3_5_moe`) | tp4 N=20 (p46cut-4) | PASS (0.0016) | **PASS**, tolerance | F_hi ∈ [0.0003, 0.005] |
| Mixtral-8x7B-Instruct-v0.1 | tp4 N=20 (tp4-c-4); tp1 fused + batched | PASS (0.004; 0.008–0.011) | **PASS**, tolerance | F_hi ∈ [0.0005, 0.01] |
| Gemma-4-26B-A4B-it, tp1 (clinical, bf16 attention) | tp1 fused (batched VOID) | PASS (0.02385 / 0.04742) | **PASS**, tolerance | not predicted |
| **Gemma-4-26B-A4B-it, tp4 field, attention-4-bit** | tp4-c-4, tp4-c-parity-2, p56-ladder-3 | **FAIL** | **NO-FLOOR** | the registered draw |
| gpt-oss-20b | tp1/tp4: fused and reference REFUSED (bare experts) | no pair | no pair | n/a |

The floor predictions for the passing families come from the cross-session reference pairs (Granite 0.00057 /
0.00090 at N=60, Qwen3 0.00198 / 0.00093 at N=20) and are registered so a future draw can falsify them.

**What if a floor band FAILS a family the constant band passes?** Under the registered rule
it cannot: B ≥ TOL. The weaker outcome that CAN happen is `detectable: true` — the fused gap exceeds K · F_hi while
staying under 0.05. If a future draw shows that, it is recorded in that family's claim row as "measurably different
arithmetic from a reorder of the reference, at a cost below the 0.05 tolerance", and **its `training_support` does
not move**. P67 does not rent to look for it, because no outcome of such a draw can change a status.

## What the existing receipts show (CPU re-read, disclosed)

`p67_reduce.py --reread` over every tp4-shaped session in the receipt store that holds an e4b arm (13 sessions,
2026-09-10 → 2026-09-22) plus tp1's committed bundle. Output: `bench/p67/reread-existing/` (the rendered table,
the JSON, the list of the 13 sessions, and the sha256 of every input receipt; the receipts themselves stay in the
private store, as P56's do). Reproduce with `bench/p67/reread-existing/README.md`.

**Judged pairs under the band (same session, VALID):**

| family | fixture | session | arm | Δ final train | median step \|Δ\| | Δ held-out | constant | **P67 band** |
|---|---|---|---|---|---|---|---|---|
| gemma4 | tp4 field, 48 held-out | tp4-c-parity-2 | fused_attn4 | 0.08257 | 0.12421 | 0.15985 | FAIL | **NO-FLOOR** |
| gemma4 | tp4 field, 48 held-out | p56-gemma4-ladder-3 | batched_attn4 | 0.05425 | 0.08487 | 0.08595 | FAIL | **NO-FLOOR** |
| gemma4 | tp4 field, 48 held-out | p56-gemma4-ladder-3 | fused_attn4_nodgrad | 0.10274 | 0.11678 | 0.18265 | FAIL | **NO-FLOOR** |
| gemma4 | tp4 field, 48 held-out | p56-gemma4-ladder-3 | fused_attn4 | 0.10223 | 0.10639 | 0.16378 | FAIL | **NO-FLOOR** |
| gemma4 | tp4 field, 8 held-out | tp4-c-4 | fused_attn4 | 0.09037 | 0.11801 | 0.09828 | FAIL | **NO-FLOOR** |
| gemma4 | tp1 clinical, bf16 attn | tp1 | fused | 0.02385 | 0.04742 | 0.00219 | PASS | **PASS** (tolerance) |
| granite | tp4 field N=20 | tp4-b-p46cut-A | fused_attn4 | 0.00063 | 0.00135 | 0.00020 | PASS | **PASS** (tolerance) |
| granite | tp4 field N=60 | tp4-a / tp4-d | fused_attn4 | 0.00030 / 0.00273 | 0.00148 / 0.00162 | 0.00030 / 0.00089 | PASS | **PASS** (tolerance) |
| granite | tp1 | tp1 | fused / batched | 0.01329 / 0.01553 | 0.01270 / 0.01681 | 0.00495 / 0.00551 | PASS | **PASS** (tolerance) |
| olmoe | tp4 field N=20 / N=60 | p46cut-A / tp4-a | fused_attn4 | 0.00203 / 0.00019 | 0.00117 / 0.00125 | 0.00116 / 0.00119 | PASS | **PASS** (tolerance) |
| olmoe | tp1 | tp1 | fused (batched VOID) | 0.01327 | 0.01249 | 0.00381 | PASS | **PASS** (tolerance) |
| qwen3 | tp4 field N=20 | p46-qwen3lora / p46cut-4 | fused_attn4 | 0.00236 / 0.00140 | 0.00177 / 0.00142 | 0.00153 / 0.00050 | PASS | **PASS** (tolerance) |
| qwen3 | tp1 | tp1 | fused (batched VOID) | 0.01315 | 0.01050 | 0.00244 | PASS | **PASS** (tolerance) |
| qwen3_5 | tp4 field N=20 | p46cut-4 | fused_attn4 | 0.00159 | 0.00154 | 0.00007 | PASS | **PASS** (tolerance) |
| mixtral | tp4 field N=20 | tp4-c-4 | fused_attn4 | 0.00403 | 0.00415 | 0.00087 | PASS | **PASS** (tolerance) |
| mixtral | tp1 | tp1 | fused / batched | 0.00953 / 0.00766 | 0.00945 / 0.01056 | 0.00068 / 0.00717 | PASS | **PASS** (tolerance) |

tp1's batched rows on OLMoE, Qwen3 and Gemma-4 are VOID by tp1's own rule (the pad-waste fallback took layers),
and the reducer reproduces every tp1 number and verdict (`test_tp1_bundle_reads_as_its_own_table_did`).
`tp4-c-4`'s Gemma-4 tokens sha differs from the later sessions' because it ran 8 held-out rows (TP4-PREREG
amendment 2); its train rows are the same, but the reducer keeps it a separate fixture by rule.

**Admissible floor draws in the existing receipts: zero, for every family.** No session ever ran the reference
twice. That is the whole gap.

**Disclosed, never admitted — the same arm on two boxes of one fixture.** These are the evidence that a floor is
not zero; they are not floor draws because each spans two boxes and, for Gemma-4, two e4b commits:

| family | arm | sessions | Δ final train | median step \|Δ\| | Δ step-0 | first differing step |
|---|---|---|---|---|---|---|
| gemma4 | reference_attn4 | tp4-c-parity-2 / p56-ladder-3 | 0.02140 | 0.02560 | 0 | 2 |
| gemma4 | fused_attn4 | tp4-c-parity-2 / p56-ladder-3 | 0.00174 | 0.04620 | 0 | 2 |
| gemma4 | fused_attn4 | tp4-b-p46cut-C2 / tp4-c-parity-2 | 0.06442 | 0.08156 | 0 | 2 |
| gemma4 | fused_attn4 | tp4-b-p46cut-C2 / p56-ladder-3 | 0.06268 | 0.05639 | 0 | 2 |
| granite | reference_attn4 | tp4-a / tp4-d (N=60) | 0.00057 | 0.00090 | 0 | 5 |
| granite | fused_attn4 | tp4-a / tp4-d (N=60) | 0.00186 | 0.00122 | 0 | 2 |
| qwen3 | reference_attn4 | p46-qwen3lora / tp4-b-p46cut-4 | 0.00198 | 0.00093 | 0 | 2 |
| qwen3 | fused_attn4 | three sessions, pairwise | 0.00102–0.00276 | 0.00086–0.00119 | 0 | 2 |

Three readings, all of which shaped the design:

1. **Every same-path pair is bit-identical at step 0 and diverges from step 2 (Granite's reference: step 5).** The
   seed of the divergence is invisible at print precision; the trajectory grows it. That is the class of
   perturbation a reorder floor produces.
2. **Different paths already differ at step 0, at identical weights** (LoRA B is zero): Gemma-4's fused arm starts
   0.034 nats from the reference and its batched arm 0.087, against 0.0005–0.014 on the other families. That is
   per-op rounding at the loss, before any trajectory — the model-level amplification §13.2 of the methodology
   records for Gemma-4's forward. It is why a reorder floor is expected to sit below the accelerated gaps.
3. **Gemma-4's same-path spread is wide.** The fused arm against itself on two boxes reads 0.0017 once and 0.064
   twice. A floor from one draw would be meaningless; this is why N_MIN is 3 and the band multiplies the maximum.

`tp4-b-p46cut-C2`'s reference arm alarmed, so its fused arm has no same-session pair; it appears only above.

## Which floor draws are missing

| family | admissible floor draws held | needed for a verdict | missing |
|---|---|---|---|
| Gemma-4-26B-A4B-it, tp4 field, attention-4-bit | 0 | ≥ 3 (the constant band fails) | **≥ 3 — the registered draw: 1 repeat + 4 perms, same session as its own reference, fused and batched arms** |
| every other family | 0 | 0 (the constant band passes; the rule needs no floor) | none |

## The decision rule for `training_support` (registered)

Read by `p67_reduce.py --session <the draw> --consistency <tp4-c-parity-2> <p56-gemma4-ladder-3>`. The two
consistency sessions are the existing same-fixture sessions; their judged pairs are re-read against the NEW
session's floor, and **any disagreement with the new session's verdict makes the reading MIXED**. They are read
only if their fixture key (tokens sha, `init_sha`, fixture, trainable count) equals the new session's; one that
does not is listed as NOT read, with the reason, and does not make the reading MIXED.

For `gemma4_text`, **attention-4-bit configuration** (`TRAIN_ATTN_4BIT` / `reference_attn4` / `fused_attn4`):

| outcome | `fast_train` (attn-4-bit) | `reference_train` (attn-4-bit) | reading |
|---|---|---|---|
| fused **PASS**, reading not MIXED | **supported** | **supported** | the fused path is indistinguishable from the reference's own reordering noise on this model |
| fused **FAIL**, batched **PASS** | not supported | supported | fused is distinguishable where the kernel-free path is not: #558 reopens as a kernel question, localisable in grouped-nf4-gemm at Gemma-4's shapes |
| fused **FAIL**, batched **FAIL** | not supported | supported | the model amplifies per-op rounding beyond reorder noise; no bf16 accelerated path is licensable by 20-step trajectory parity here. Next instrument: per-op / matched-routing, not more draws |
| fused PASS, batched FAIL | supported | supported | flagged INVERTED in the claim row (fused carries more per-op error); each arm stands on its own verdict and on the consistency rule |
| NO-FLOOR, VOID, or MIXED | no change | no change | the reason cites the P67 outcome; a redraw needs a registered amendment |

`reference_train` in the attention-4-bit configuration is held today only because the band could not decide
(its own arm is VALID). It moves to supported on any READ outcome above — the reference is what everything is read
against, and the floor draws are five more VALID runs of it — and stays put on NO-FLOOR, VOID or MIXED.

**Not moved by this lane, whatever the outcome:**

- `gemma4_text.fast_train` with bf16 attention — tp1's clinical PASS stands; the draw is a different configuration.
- `gemma4_text.batched_train` — the batched arm runs at `pad_waste_limit` 64, not the shipped default of 4, so its
  verdict is recorded in the claim row and does not license the shipped default.
- every other family's `training_support` (its constant-band PASS is a floor-band PASS by construction).
- `qwen3_5_moe`, which has a tp4 PASS but no `training_support` row; adding one is a separate docs change.

## The draw

One RTX 5090 (Vast, verified/secure; the Vast provider's default offer filter already asks for ≥ 98 GB host RAM,
which is the host's figure, not the container's), one session, tp4's machinery
unchanged: `bench/p67/p67_drive.sh` checks the pins and execs `bench/tp4/tp4_drive.sh`, which stages the tp4 pieces
plus P67's guard and starts the guard on the box; `bench/p67/p67_run.sh` re-checks the pins, sets and verifies the
knobs, checks host RAM, and execs `tp4_run.sh`. The arms are `tp4_arm.py`'s own — nothing is reimplemented; the
only additions to the machinery are the default-off `TP4_P67` block in `tp4_run.sh`, the default-off
`TP4_RUNNER` / `TP4_EXTRA_STAGE` knobs in `tp4_drive.sh`, and the `reference_order` field in `tp4_arm.py`'s
receipt (plus its refusal of the switch on any non-reference arm, exit 19).

**Fixture:** tp4's field recipe, byte-for-byte what the standing rows and P56 ran — alpaca (`DS_ALPACA_SHA`
5324987a…), seq 2048, micro-batch 2 × accum 4, r 16 / α 16, lr 2e-4, AdamW-8bit, wd 0.001, linear warm-up 5,
seed 3407, **N = 20**, 48 held-out rows at steps 0 and 20, `--attn-4bit 1`. `google/gemma-4-26B-A4B-it` at
`4d7ae4984b7db7de8f8457170b3f1a419ee76d52`. grouped-nf4-gemm pinned at **9206352f (v0.32.1)** — P56's cut and the
standing row's. e4b at this lane's merge commit (it carries the switch). `p67_run.sh` refuses any fixture knob that
is set (`TP4_SEQ`, `TP4_MB`, `TP4_ACCUM`, `TP4_R`, `TP4_LR`, `TP4_EVAL_N`, …, exit 78).

**Registered knobs** — `bench/p67/registered.knobs`, verbatim (the box enforces this file; a test asserts every
line appears here):

```
TP4_BOX|C
TP4_FAMILIES|gemma4
TP4_STEPS|20
TP4_P56|1
TP4_BATCHED_PAD_WASTE_LIMIT|64
TP4_P67|1
TP4_P67_PERMS|1 2 3 4
TP4_SKIP|gemma4/unsloth/ckpt_unsloth gemma4/hf/hf_peft gemma4/e4b/fused_attn4_nodgrad
TP4_PREREG|bench/p67/P67-PREREG.md
GNF4_SHA|9206352f7c49d76432824f7ce8872f0b285f2a4e
```

**Arms, in run order** (tp4_run.sh's family order; `can_run` drops arms from the END as the window closes, so the
judged arms come first and the floor draws last):

| # | tag | arm | switch | est. wall (P56 ladder-3) |
|---|---|---|---|---|
| 1 | `fused_attn4` | fused, dgrad | — | 3.1 min |
| — | `ckpt_unsloth`, `hf_peft`, `fused_attn4_nodgrad` | — | skipped by `TP4_SKIP` (not_run stubs) | 0 |
| 2 | `reference_attn4` | reference | — (ascending, the shipped loop) | 16.9 min |
| 3 | `batched_attn4` | batched | `E4B_BATCHED_PAD_WASTE_LIMIT=64` | 3.2 min |
| 4 | `reference_attn4_repeat` | reference | — (a plain repeat) | 16.9 min |
| 5–8 | `reference_attn4_perm1` … `_perm4` | reference | `E4B_REFERENCE_EXPERT_ORDER=perm:<s>` | 4 × 16.9 min |

**Cost, guard, downloads.** From P56 ladder-3's own timestamps: box setup (two venvs, dataset) 7.6 min, Gemma-4
fetch 10.3 min, tokenise 0.1 min; arms as above, 107.7 min; reduce, fetch and teardown ~3 min. **≈ 130 min ≈ 2.2 h.**
At P56's realised rate ($0.448 for 2,662 s ≈ $0.61/h) ≈ **$1.33**; at the standing $0.69/h ceiling ≈ **$1.50**.
**Guard 3 h**, so the approval line is 3 × 0.69 = **$2.07** — the guard, not the estimate, is what the launcher
charges against. With the proving rental ($0.115 line), the lane's total approval line is **$2.19**, inside the
$35/run cap; the approver is the owner (the standing rule since 2026-09-18), and the #703 redirect is the
authorization this registration answers to, not an approval of the launch itself. The window: the last perm arm starts near t = 109 min and `can_run 900` needs 30 min left at that
point, which a 180-min deadline gives with 40 min to spare. Downloads on the box: `google/gemma-4-26B-A4B-it` @ `4d7ae498` (49.9 GiB +
1.7 GiB shards, ≈ 52 GB; authenticated when the controller has an HF token to stage), `unsloth/alpaca-cleaned` at its pinned revision
(built by `tp4_alpaca.py`), and pip installs (e4b @ the lane commit and gnf4 @ 9206352f from GitHub; transformers
5.17.0, bitsandbytes 0.50.2, peft 0.20.0 from PyPI; tp4_run.sh's Unsloth venv, which it installs on every box).
Nothing is pulled that already sits on the NAS model store (`gemma-4-26B-A4B-it` is there, but a rented box cannot
reach it).

**Host class.** Gemma-4's single 49.9 GiB shard is `safe_open`-mapped whole; e4b#344 / P55 order every recorded
load outcome by host RAM (≤ 64 GiB refuses or faults; 96 / 125 / 188 GiB pass). `p67_run.sh` refuses a box whose
effective RAM (min of `MemTotal` and the cgroup limit) is under **96 GiB**, before any install, **exit 18**. 18 is
not one of the launcher's host-limited codes (13/14/17), so the machine is not auto-excluded; the receipt names it.

**Proving rental first — the compute rule for any guard over one hour.** The draw's guard is 3 h, so it is preceded
by a proving rental of **≤ 10 min and ≤ $0.15**: one RTX 5090 of the same provider class and image, guard **600 s**
(10 min × $0.69/h = **$0.115** approval line), with `bench/p56/p56_prove.sh` unchanged as the launcher's
`--command` — the pattern P56 used (`p56-prove-1`: 52.7 s of runtime, $0.0086, receipt `receipt.json` +
`teardown-proof.json` in the store). It proves the launcher path end to end (attach, pre-flight, command handoff,
receipt, ledger row, teardown proof) and prints the one quantity no pre-flight reads: the box's effective host RAM
against Gemma-4's 49.9 GiB shard. It **reports and does not refuse** (P41's lesson: a run that can only say "pass"
looks like one that can only say "fail" on a green result). The go/no-go is a reading of its receipt: the draw is
launched only if the proving run completed with a teardown proof and printed `GO-CLASS`; the proving box is torn
down and never reused, and the draw's own box is re-checked by `p67_run.sh` (exit 18 below 96 GiB). Receipts:
`receipts/experts4bit-qlora/<date>/p67-prove-<n>/`.

**Receipts:** `receipts/experts4bit-qlora/<date>/p67-gemma4-floor-<n>/` in the adertha receipt store (the
launcher's `receipt.json`, `teardown-proof.json`, and the fetched box tree `tp4/`, which holds every arm's JSON,
`summary.txt`, `versions.txt`, `box.json`, `forensics.txt`, `p67_guard.txt`, `logs/`). Read by `p67_reduce.py`
under its own rules; nothing in a results page is hand-transcribed.

**Exit codes.** `p67_drive.sh` (controller): 78 a pin or a registered knob disagrees (before anything is staged);
otherwise `tp4_drive.sh`'s — 20 stage failed, 21 start/nonce handshake failed, 22 fetch failed, 23 no TP_DONE by
the deadline, 24 stale nonce or malformed exit code, 25 the lane died on the box, else the lane's own rc.
`p67_run.sh` (box, writes TP4_RUN_NONCE, TP4_EXIT_CODE.<nonce> and TP_DONE.<nonce> on a refusal): 9 a staged file
differs from `staged.sha256` or no `sha256sum`; 78 a registered knob differs, a fixture knob is set, or the switch
is set lane-wide; 18 host RAM below the shard class; otherwise `tp4_run.sh`'s (78 env, 9 stage/install/tripwire,
12 GPU class, 13 disk, 0 the plan completed). Per arm (`summary.txt`, `<fam>/<fw>/<tag> rc=N`), among tp4_arm.py's
codes: 0 trained, 3 refused, 13 tokens mismatch, 16 phase alarm, **19 the P67 switch on a non-reference arm**, 142
SIGALRM.

## Registered predictions

- **Q1 — engagement.** All seven run arms VALID: identical `init_sha` (7588b481…) and tokens sha, 487,280,640
  trainable, C1 bit-exact, 20 steps, 115 attention projections; `n_patched` 30 on fused and batched; batched
  engaged on every call (0 fallbacks at limit 64); every perm arm's `reference_order` shows its own `perm:<s>` on
  `calls > 0`; the repeat and the reference show none.
- **Q2 — determinism on the 5090.** The plain repeat is **not** bit-identical to the reference. (Basis: the
  cross-box reference pairs diverge from step 2 with identical first losses; torch does not select
  deterministic algorithms unless asked, and `tp4_arm.py` never asks; and on the A2000 rehearsal a same-box repeat
  diverged from step 5. Which kernel is the source is NOT established. If the repeat IS bit-identical on the 5090,
  that is recorded as the finding and the repeat leaves the floor.)
- **Q3 — the perturbation is smaller at its source than the accelerated arms'.** Every perm draw's step-0 train
  delta is below both the fused and the batched arm's step-0 train delta in the same session (the standing values
  on Gemma-4: 0.034 fused, 0.087 batched). *Revised before registration, and disclosed:* the first draft predicted
  "≤ 1e-4 at step 0"; the A2000 rehearsal falsified that on its model — two of three perms moved the step-0 train
  loss by 0.0046, against 0.026 / 0.041 for the accelerated arms — so the prediction now registered is the
  comparison, and the reducer's 1e-4 step-0 flag stays a reporting flag that refuses nothing.
- **Q4 — the floor's size on Gemma-4.** F_hi(final) and F_hi(med) each within [0.005, 0.08]. Point prior 0.02 from
  the one cross-session reference pair (0.02140 / 0.02560).
- **Q5 — the standing gaps reproduce.** fused D_final ∈ [0.06, 0.13] and D_med ∈ [0.08, 0.15]; batched D_final ∈
  [0.03, 0.08] and D_med ∈ [0.05, 0.11].
- **Q6 — the ladder.** D(batched) < D(fused) on both quantities (P56's ordering reproduces).
- **Q7 — the verdicts are NOT predicted.** Every row of the decision table is a result. The prior, disclosed and not
  registered: with Q4's point prior, B(med) ≈ max(0.05, 3 × 0.026) ≈ 0.077, below fused's three standing medians
  (0.106–0.124), so a fused FAIL is the less surprising outcome; batched's 0.085 median sits at the edge.
- **Q8 — the other families** read PASS by construction and are not drawn; their floor predictions are in the
  families table and wait for a draw nobody has authorised.

## What this lane does NOT do

- No head-to-head position against another framework (Unsloth and HF are skipped by registration).
- No claim about any family's floor other than Gemma-4's.
- No speed claim; step times are recorded and not quoted.
- No change to tp1's tolerance for any family, and no rental for the passing families: under the registered rule
  no outcome of such a draw moves a status.
- No edit to `docs/claims.json`, `docs/capabilities.json` or `docs/STATUS.md`; those follow the read.

## Rehearsal (NAS A2000) — NOT a reading

Disclosed because it shaped Q2, Q3 and the admission rule; **it is not a reading, licenses nothing, and no number
in it is a floor for any family** (different model, different card, shortened fixture). Record:
`bench/p67/rehearsal-a2000/` (README, the seven receipts, logs, the reducer's `session.md`/`.json`, and exactly the
scripts that ran).

- **Where and what.** The shared NAS RTX A2000 12GB (sm86) under the `a2000.lock` protocol (lock taken 21:19:57Z
  after a 69-min wait, released at container exit 21:48:44Z, rc 0), `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`,
  transformers 5.17.0, bitsandbytes 0.50.2, gnf4 9206352f; `granite-3.0-1b-a400m-instruct` from the NAS store
  (24 layers, 32 experts, top-8), mounted read-only. tp4's field recipe except seq 512 and 16 held-out rows; seven
  of `tp4_arm.py`'s arms: reference, repeat, perm1–3, batched (limit 64), fused. Free; nothing pulled over the WAN
  but pip packages.
- **The switch reaches the loop and the receipt proves it:** each perm arm's `reference_order` reads its own
  `perm:<s>` on 4,608 calls; the reference and the repeat read none. All four floor candidates are admissible.
- **A same-box repeat is not bit-identical on this card:** identical through step 4, then D 0.00165 / 0.00082. The
  5090's answer is still Q2's to register, not this card's to supply.
- **The reorder reaches the loss, sometimes at step 0:** perm2 diverges from step 2; perm1 and perm3 move the step-0
  TRAIN loss by 0.00459 each (the held-out step-0 loss is unchanged at 5 decimals). Accelerated arms at step 0:
  0.02612 (batched), 0.04059 (fused). This is what revised Q3.
- **The switch's unit tests pass on this card's CUDA too** (16/16, `logs/switch-cuda.log`), not only on CPU.
- **The reducer reads real receipts end to end:** F_hi 0.00380 / 0.00205 → band 0.05 on both; fused 0.00136 /
  0.00339 and batched 0.00141 / 0.00447 read PASS, carried by tolerance, not detectable (D_med / F_hi 1.65 and
  2.18).
