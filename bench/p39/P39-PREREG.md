# Pre-registration — lane P39: the step-level effect of gnf4#357, and the recorded assignment on a second box (#530)

Registered 2026-09-10, before any rental. Two boxes, one RTX 5090 each
(`vast:verified-secure`, the class P37 measured on), launched from the mini through
`adertha-agents/tools/pod-launch.sh`. Owner authorisation: Jordan, in the working
session on 2026-09-10 — "go" (three times, the last two in reply to exactly this plan:
*"the step-level B=16 effect of #357, and the #530 second-box gate. ~$3–5 each … I'd
pre-register both first"*). That authorisation is the top of the approval band, not a
waiver of this document or of the STOP rules. **Written down by the agent, not by
Jordan**; the launcher needs a citable permalink, so the work issue that carries this
text is what the manifests cite, and it says the same thing.

## What is already settled, and therefore NOT in scope

- gnf4#357 (`00bf78e`) is merged: split-K in `int4_b32._plan` takes the row count,
  floored at R ≥ 16. Kernel-level: 1.10–1.31× at R ≥ 16 on an A2000, never slower on
  any of 48 cells, B=1 untouched by construction. **Its step-level effect was explicitly
  not claimed** (`bench/int4/RESULTS-sk-r-sweep.md`). H1 measures it.
- e4b#531 (`ef6d532`) is merged: the gptq/rtn decision travels with the pack as a hashed
  payload and a re-pack honours it. **It fixes the classification, not the bytes** —
  GPTQ output also depends on the Hessian, which router flips perturb. H2 is the
  experiment #530 asked for and #531 said it enables.
- P37's e4b arms are VOID under its fingerprint rule and its own-pack K8 gate read
  wikitext −0.023 PASS / c4val1 **+0.109 FAIL**. Nothing here re-litigates P37; nothing
  here quotes a ratio against vLLM.

## Design that keeps each box inside the policy's 6 h wallclock cap

P37 rebuilt the licensed pack **per arm** (~45 min of calibration each). Here each box
calibrates **once**, dumps the pack as a hash-pinned artifact (#405, now carrying the
assignment payload), and every later arm **loads those exact bytes by fingerprint**
(~5 min). So the kernel A/B on box 1 runs on one identical pack, and the gate on box 2
runs on one identical pack. Hook v7 = bo7's v6 byte-for-byte plus three env knobs the
library already exposes (`E4B_INT4_DUMP_ARTIFACT_DIR`, `E4B_INT4_ARTIFACT_DIR` +
`E4B_INT4_EXPECTED_FINGERPRINT`, and the library-read `E4B_INT4_ASSIGNMENT`);
`step_decomp.py`, `k8_bake.py`, `calib.json` are bo7's, sha-pinned in `staged.sha256`.

Packing itself (`gptq_pack_int4_b32`, `pack_int4_b32`) is untouched by #357, so the
pack bytes do not depend on which gnf4 the box is on; the A/B switches only the GEMV.

## Hypotheses, each with the observation that refutes it

**H1 — step-level effect of the R-aware plan (box 1).** e4b `ef6d532`, Qwen3-30B-A3B
`ad44e777…`, one artifact-pinned licensed pack, graph decode, 512-token prompts, 128
new tokens, `n_steps` 127 (B=1) / 70 (B=16) exactly as P37. Arms alternate gnf4
`f8f6405` (pre-#357, **OLD**) and `00bf78e` (**NEW**), each verified per arm by a
tripwire line `PLAN_HAS_R=False|True` (the presence of the `R` parameter on
`int4_b32._plan`), never by a version string (both ship as 0.30.x-class installs).
*Prediction, registered as a band:* B=16 `step_ms_clean` **NEW/OLD ∈ [0.96, 0.99]** —
the census priced the split-K reduce at 3.4 % of the step and the expert GEMVs at a
minority share, so 1–4 % is what a 1.10–1.31× kernel win should be worth at step level.
B=1 **NEW/OLD ∈ [0.97, 1.03]** — unchanged by the floor, so only self-pair noise.
*Refuted if:* B=16 NEW is slower than OLD by more than 1 % in either repeat, or B=1
moves outside ±3 %. *Reads as "kernel-level only" if:* B=16 improves by less than 1 %.
Repeats: B=16 twice per arm (r1, r2, ABAB), B=1 once per arm; self-pair spread is
reported beside the ratio and a ratio inside the spread is "no effect resolved", not a
win.

**H2 — the recorded assignment on a second box (box 2).** Box 2 stages box 1's
`assignment.json` (~2 KB) and runs the recipe with `E4B_INT4_ASSIGNMENT` pointing at it.
*Predictions:* (a) the `INT4EXP assignment honoured <hash>` banner appears and the
gptq/rtn **counts equal box 1's exactly** (by construction); (b) the reported
disagreements with box 2's own routing are **between 2 and 40 expert-roles** (P37 saw
ten experts flip, two roles each); (c) the honoured pack's `pack_fingerprint`
**differs** from box 1's (the Hessians differ, so the bytes do — #531's stated limit);
(d) the K8 two-text gate on the honoured pack against NF4 — **this is the open
question**, and its two outcomes are interpreted in advance: **PASS on both texts** ⇒
the split was the cause of P37's c4val1 failure and a recorded assignment is sufficient
for a licence to travel; **FAIL on c4val1 again** ⇒ the split was not the cause and the
c4val1 pattern across bo5/bo6/P37 is a Hessian/data question, not a classification one.
Either outcome is a result. *Refuted if:* (a) fails — the counts differ or the banner is
absent — which would mean #531 does not do what it says. Optional arm, only if ≥ 60 min
remain before the deadline: a plain recipe build (no assignment) whose counts are
predicted to **differ** from box 1's, replicating P37's flip on a third box.

## Pinned inputs

e4b `ef6d532438e823f29763efb6e65067e02ddbde85`; gnf4 NEW
`00bf78ec07cdfbfd4cef012a2e3b562c0dd71f9b`, OLD `f8f6405d799bbb8efe328a991413cd3869d49bef`;
Qwen/Qwen3-30B-A3B `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`; transformers 5.16.1,
bitsandbytes 0.50.1 (P37's). Calibration: `E4B_CALIB_NSEQ=128`, C4 validation shard
00000, damping 0.01, `min_rows=32`, 24 GiB Hessian budget, GPU solve — the licensed
recipe. A run whose pins differ from its receipt is VOID, not adjusted.

## Vocabulary

| verdict | meaning |
|---|---|
| `pass` / `fail` | the registered prediction held / did not |
| `unresolved` | inside the self-pair spread; neither |
| `void` | pins drifted, tripwire mismatch, or the instrument was wrong — reported, never reused |
| `host-limited` | the box could not finish (deadline, OOM); says nothing about the hypothesis |

## STOP rules

1. **Spend.** Estimated ≤ $6 for both boxes. **$12 registered ceiling; at $20 actual
   the lane stops** whatever remains unmeasured.
2. **No arm starts that cannot finish 10 minutes before the launcher's wallclock
   deadline** (`E4B_RENT_DEADLINE_EPOCH`); a skipped arm is a `host-limited` row.
3. **Two consecutive VOIDs on one box stop that box.**
4. **Any teardown that cannot be proven stops the whole lane.**
5. **No result is reported before its receipt exists** in `~/adertha-receipts`.
6. **Box 2 does not launch until box 1's `assignment.json` and fingerprint are in the
   receipt store** — the second box is meaningless without the first's record.

## Cost, stated before the fact

| box | work | wallclock (manifest) | rate | est. | max |
|---|---|---|---|---|---|
| 1 | install, fetch 61 GB, bake, 1 calibration + dump, 2 controls, 6 load-arms + 3 kernel switches | 6 h | ≤ $0.75/h | ~$2.6 (3.5 h) | $4.50 |
| 2 | install, fetch, bake, 2 NF4 K8 runs, 1 honoured calibration + dump + K8, 1 K8 load-arm, optional recipe arm | 6 h | ≤ $0.75/h | ~$2.6 (3.5 h) | $4.50 |

Vast verified-secure RTX 5090 offers on 2026-09-10 started at $0.54/h (19 listed);
$0.75 is the manifest's rate ceiling, not the expected price.

## Launcher

`adertha-agents/tools/pod-launch.sh` with tracked manifests `tools/manifests/p39-box1.json`
and `p39-box2.json`. Preparing this launch found that the launcher's reconcile step
refused by construction (`ledger_check.py` read `policy.json` relative to a cwd that is
the receipt repository) and that #95's checker still validated Slack-only approvals;
both fixed and tested in adertha-agents#96 before any money was spent — which is the
"refusal is the finding" the support prereg anticipated.

---

## Amendment 1 — 2026-09-10, after two pre-flight VOIDs on box 1 (registered before any further rental)

**What happened.** `p39-box1` (11:33Z) and `p39-box1-2` (11:42Z) both bought Vast offer
`46672417` on **machine `59164`** — the cheapest verified RTX 5090, $0.5378/h — and both
died identically in pre-flight: ssh `Permission denied (publickey)` for the whole 180 s
readiness budget (33 attempts each). Both boxes were destroyed with proof; $0.0487 +
$0.0319 = **$0.0806 spent**. The workload never ran, so nothing bears on H1. Receipts:
`receipts/experts4bit-qlora/2026-09-10/p39-box1/` and `…/p39-box1-2/` in the private record.

**STOP-3 fired** ("two consecutive VOIDs on one box stop that box") and box 1 stopped.

**Why an amendment rather than a third identical attempt.** The two receipts name the
same machine. The launcher's only machine-exclusion mechanism was bound to P41's
strict-anchor `BOX_REFUSED` receipt shape, so a pre-flight `NOT_RUN` could not exclude
anything, and `rent.py` orders offers by price — a third launch would have bought the same
host a third time. That is "a third attempt after two instrument failures", which is not
evidence-gathering. The instrument was changed first, as its own reviewed change:
**adertha-agents#99** (merged `406f5ca`) — a committed pre-flight ssh-readiness refusal on
the same provider and GPU class now excludes its machine (`--exclude-vast-machine-receipt`,
forwarded from the manifest as `exclude_vast_machine_receipts`), and the ssh-readiness
budget has a surface (`--ssh-ready-s`, manifest `ssh_ready_s`; default unchanged).

**What changes for box 1, and only box 1.**

| | attempts 1–2 | attempt 3 (`p39-box1-3`) |
|---|---|---|
| machine exclusions | none expressible | `59164`, by the two committed receipts above |
| ssh readiness budget | 180 s (default) | **600 s** (matches the `running` wait) |
| adertha pin | `950d5dc` | `406f5ca` |
| everything else (e4b pin, image, rate ceiling, wallclock, arms, bands, guards) | — | **unchanged** |

**STOP-3 counter.** Reset to zero for box 1 *because the instrument changed*, and **capped
at one further attempt**: if `p39-box1-3` VOIDs for any reason, box 1 stops for good and
the lane reports H1 as `host-limited`. The spend rules are unchanged: $0.0806 counts
toward the $6 estimate, $12 ceiling, $20 hard stop.

**Box 2 is unchanged** and still waits on box 1's record (STOP-6).
