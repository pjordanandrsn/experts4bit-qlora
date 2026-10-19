# AXO1767: where axolotl's in-loop held-out number and its exported adapter part (#1767)

Registered 2026-10-18, before any data from this token exists. Correctness only: nothing it writes is a TC1 or HP1 number, and no
TC1 or HP1 reading, rule or verdict changes with it.

## Why

`hp1k-5090-8`'s box-side shakedown ran HP1's fixture on a 2-layer cut of Qwen3-30B-A3B and VOIDed the axolotl arm on STRICT round-trip
6(b): the evaluator read the exported adapter at 10.97454 against the arm's in-loop 10.94001 (+0.0345). Step 0 agreed (−0.0044). e4b
and Unsloth round-tripped within 0.0023. The in-loop curve fell 0.0562 over 3 steps, and the evaluator's read of the export fell 0.0173,
in line with the other two (#1767). Ruled out before this registration:
- the LoRA scale: `lora_alpha` = `lora_r` = 16, and the canonical export records 1.0. axolotl 0.20.0's scattermoe_lora and lora_kernels
  paths both read PEFT's own `scaling[adapter]`, and the scattermoe A/B layout pairs as PEFT's does (the source read on #1767);
- a PEFT layout change between 0.21.0 (axolotl's) and 0.21.2 (the others'): `ParamWrapper` is identical in both;
- an extra trainable parameter: all three arms train the same 24 tensors and 26,771,456 parameters, none outside attention and experts.

What is left needs numbers: axolotl's kernels computing something other than PEFT's math, the training process holding state the saved
adapter does not carry, or the export and evaluator reading the adapter differently.

## Fixture

HP1's box-side shakedown, unchanged: the `hp1_trunc.py` 2-layer cut of `Qwen/Qwen3-30B-A3B` @ `ad44e777`, 3 steps, 8 held-out rows,
r 16 / alpha 16, lr 2e-4, adamw_8bit, linear warmup 5, micro-batch 2 × accumulation 4, seed and order seed 3407; fam `qwen3hp1shake`.
Two arms train, as `hp1k-5090-8` ran them: e4b `hp1shake_e4b` (the evaluator control) and axolotl `hp1shake_axolotl` with
`--axolotl-best 1` (axolotl 0.20.0's kernels plugin, scattermoe experts, and the LoRA kernels it auto-enables). No Unsloth arm.

## The four scores

Each is the mean loss over the same 8 held-out rows, computed the way `tc1_arm.eval_loss` computes the in-loop number.

| # | name | how |
|---|---|---|
| 1 | in-loop | the axolotl arm's own eval at step 3 (its receipt's `eval_curve`) |
| 2 | evaluator | `hp1_eval.py`'s eager forward on the canonical export (`eval_shake.json`, run without `--require-all-scored` and with `--zero-tol 0.01`, this registration's MATCH band, so the evaluator's own base check and the reducer's instrument check use one band) |
| 3 | off | `tc1_arm.py --rescore-adapter` on the arm's saved PEFT adapter, in a fresh axolotl model with `--axolotl-kernels off`: no kernels plugin or scattermoe, `lora_{mlp,qkv,o,embedding}_kernel` false, eager experts (PEFT's reference forward) |
| 4 | on | the same rescore with `--axolotl-kernels on` and `--axolotl-best 1`, the training configuration |

A rescore loads the adapter with PEFT's `set_peft_model_state_dict` and refuses unless PEFT's own `get_peft_model_state_dict` reads
every saved tensor back bit for bit, with no unexpected key. It first scores the zero adapter (PEFT builds every B at zero).

## Bands (fixed now)

- **MATCH:** |Δ| ≤ 0.01 nats on the 8-row mean. That is twice `hp1k-5090-8`'s largest step-0 disagreement between a framework's
  forward and the evaluator's (0.0044), rounded up.
- **DIFFER:** |Δ| ≥ 0.02 nats, HP1's registered round-trip tolerance (`hp1_eval.ROUNDTRIP_TOL`).
- **INCONCLUSIVE:** anything in between.

## Reading (`bench/tc1/axo1767_reduce.py`, mechanical)

**Instrument checks first; any failure makes the reading VOID:**
- the evaluator's zero adapter against the in-loop step 0 MATCHes;
- the e4b control's round trip, (2) − (1) for `hp1shake_e4b`, MATCHes;
- each rescore's zero adapter against the in-loop step 0 MATCHes, and each rescore is an ok record.

**Precondition:** the gap must reproduce, (2) − (1) DIFFER. A MATCH reads NOT_REPRODUCED, and a value between the bands reads INCONCLUSIVE.

**Then the 2×2 on A = (4) − (1) and B = (3) − (2).** C = (3) − (4) splits the both-match cell:

| A | B | C | verdict | reading |
|---|---|---|---|---|
| MATCH | MATCH | DIFFER | KERNELS | the saved adapter gives the in-loop number only under axolotl's kernels; PEFT's reference forward and the evaluator agree on it |
| MATCH | MATCH | MATCH | INCONCLUSIVE | each pair is inside the match band, yet (2) − (1) DIFFERs: the bands cannot place the gap |
| MATCH | DIFFER | any | EXPORT | the kernels-on reload gives the in-loop number, but PEFT's reference forward on the same adapter disagrees with the evaluator: the canonical export or the evaluator reads the adapter differently |
| DIFFER | MATCH | any | PROCESS_STATE | even under the kernels, the saved adapter does not give the in-loop number; the reference forward agrees with the evaluator: the training process held state the adapter does not carry |
| DIFFER | DIFFER | any | NEITHER | no single-cause reading; the four numbers are reported as they are |

INCONCLUSIVE on A or B, or on C inside the both-match cell, reads INCONCLUSIVE. The reducer was run on `hp1k-5090-8`'s real record before
this registration. With scores (1) and (2) only, it reads INCOMPLETE: the gap reproduces at +0.0345 (DIFFER), the e4b control's round trip
is −0.0012 (MATCH), and the evaluator's zero adapter is −0.0044 (MATCH).

## Box and budget

- **Box:** token `qwen3axo1767` on `TC1_BOX=A`; ids `axo1767-5090-1` (first attempt) and `-2`/`-3` (spares, each needing a new
  box-started reason); one RTX 5090, any verified host with direct ports except machine 151831 (HP1's).
- **Time:** about 45 min expected (installs, the model fetch, two arms, two rescores, the evaluator); guard 1.5 h.
- **PyPI floor (a feasibility term, added 2026-10-19 after `axo1767-5090-2`, before any reading).** That box's host fetched from
  files.pythonhosted.org at 0.88 MB/s, against 23-66 MB/s from download.pytorch.org, so its venv-axolotl install took 31 of
  its 45 alarm minutes (2,700 s); a slightly slower host would have run into the alarm. The box now times one pinned wheel (`pypi_probe.py`: 35.3 MB, sha256 pinned, at
  most 60 s) right after the fetch probe, and refuses a host below 10 MB/s (rc 15, a startup-floor refusal). It
  protects the guard, never the reading: no band, score or verdict row changes.
- **Cost:** worst at most $2.50 per box at the policy's 5090 class rate, under the standing no-ask rule.
- **Records:**
  - `axo1767/fourscore.json` (the reading) and `axo1767/eval_shake.json`;
  - the arm and rescore receipts, and the shakedown's canonical adapters (`hp1/shake/canonical_*`, returned since #1769);
  - a copy of the saved PEFT adapter in `axo1767/peft_adapter_axolotl/`. Model weights stay on the controller's disk inside the run
    directory, never in the receipt store; their sha256s are in the rescore records.

## What this cannot say

- Which line of a kernel is responsible: a KERNELS reading names the path (scattermoe or the LoRA kernels together), not the op.
- Anything at full depth or at HP1's 200 steps; the fixture is the shakedown's.
- Anything about e4b or Unsloth beyond the e4b control's round trip.

## Amendment 1 (registered 2026-10-19, after the KERNELS reading and before any split data): which kernel path

**Why.** `axo1767-5090-2` read KERNELS (re-derived): the saved adapter gives the in-loop number only under axolotl's kernels, and
PEFT's reference forward agrees with the evaluator. Its kernels-on configuration switches two paths together:
- the kernels plugin's scattermoe experts, which apply the expert LoRA in the scattermoe kernel;
- the LoRA kernels that `check_auto_enable_lora_kernels` turns on. On `axo1767-5090-2` the effective set was all four
  (`lora_mlp_kernel`, `lora_qkv_kernel`, `lora_o_kernel`, `lora_embedding_kernel`), which is why (6) sets all four.
This amendment separates them.

**The change.** A token variant, `qwen3axo1767b`, runs everything above unchanged, then two more fresh-process rescores of the same
saved adapter:

| # | name | how |
|---|---|---|
| 5 | experts only | `--axolotl-kernels experts --axolotl-best 1`: the kernels plugin's scattermoe experts, with `lora_{mlp,qkv,o,embedding}_kernel` false |
| 6 | LoRA kernels only | `--axolotl-kernels lora --axolotl-best 0`: eager experts (no plugin), with `lora_{mlp,qkv,o,embedding}_kernel` true |

Each is checked like (3) and (4): a bit-for-bit adapter round trip, an ok record, and its zero adapter MATCHing the in-loop step 0.

**Engagement check (all four rescores).** No label is trusted. Each rescore record must show that it ran the mode its name
claims, through four fields: the mode, the experts implementation the model was set to, the plugins, and the effective LoRA-kernel
set. The expected values, as `axo1767-5090-2`'s real (3) and (4) records carry them:
- (3) off: eager experts, no plugin, no LoRA kernel on;
- (4) on: scattermoe, the kernels plugin, at least one LoRA kernel on (its set is the box's effective set);
- (5) experts: scattermoe, the kernels plugin, no LoRA kernel on;
- (6) LoRA kernels: eager experts, no plugin, exactly (4)'s effective set on the same box.

**What a failure voids.** A failed check on (3) or (4) makes the whole reading VOID. A failed check on (5) or (6), whether an
instrument check or the engagement check, makes the split **SPLIT_VOID** and leaves the main reading standing: the main reading
does not depend on (5) or (6). A box whose main reading is valid is not a strike. A SPLIT_VOID from a harness cause is fixed
before the split re-runs, and that re-run carries its own one-strike rule.

**Split reading** (`axo1767_reduce.py`, the same bands). It applies only when the main reading above is KERNELS; otherwise it is
NOT_APPLICABLE. With E = (5) − (3) and L = (6) − (3):

| E | L | split verdict | reading |
|---|---|---|---|
| DIFFER | MATCH | SCATTERMOE | the kernels plugin's expert path (scattermoe + moe_bnb_fast) alone moves the loss off PEFT's reference |
| MATCH | DIFFER | LORA_KERNELS | the LoRA kernels alone move the loss off PEFT's reference |
| DIFFER | DIFFER | BOTH | each path alone moves it |
| MATCH | MATCH | INTERACTION | neither alone does, though both together do (C differs) |

INCONCLUSIVE on E or L reads INCONCLUSIVE. E + L − ((4) − (3)), the additivity residual, is reported and gates nothing.

**Box and budget.** One RTX 5090 under the standing rule. Ids `axo1767b-5090-1` to `-3`. Guard 1.5 h, worst ≤ $2.50. The same disk
floor (min_disk_gb 200) and the same PyPI floor (10 MB/s) apply. Machine 143500 is excluded, after its 0.88 MB/s PyPI path. The
adapter is retrained on the box: the launcher stages no model weights, so `axo1767-5090-2`'s adapter cannot be carried over. The split
therefore reads within its own box, and the main reading on that box must reproduce KERNELS for the split to apply.

**Unchanged:** scores (1)–(4), the bands, the instrument checks, the main 2×2 table, and the fixture.
