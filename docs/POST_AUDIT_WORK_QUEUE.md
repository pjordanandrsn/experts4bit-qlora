# AGENT HANDOFF — POST-AUDIT STATE + WORK QUEUE (2026-07-05)

```
status:    priority/prediction document — commit BEFORE executing the queue (R5)
cites:     plan-routed-v3 (ROUTED_STREAMING_AGENT_EXECUTION_PLAN.md)
           MEASUREMENT_AUDIT_olmoe-qlora-grid-20260705-1351.md
           bundle olmoe-qlora-grid-20260705-1351
owner:     Jordan        executor: coding agent (Claude Code)
rules:     plan-routed-v3 R1–R10 remain in force verbatim. This doc adds queue,
           quarantine, decision trees, and v3 amendments (as a separate file — v3
           is not edited).
spine:     everything downstream forks on T1 (certificate D3). Two lanes are
           anomaly-immune and run in parallel. Critical path ≈ 1 GPU-hour.
```

---

## 0. State of the world

DONE / VERIFIED: OLMoE 12-run repeat grid, decode repeat-5, 25-cell seed-0
portability matrix, 8-cell 3-seed portability (claim_usable), Qwen3 A100 probe,
expert-streaming train-phase profile, bundle + audit. Memory topology PvM ✓ at two
scales. fp4-decode claim correctly deflated. Runner commit-attestation fixed
(E4B_COMMIT) — verify the FIRST new job classifies `claim_usable`; if not, stop.

CONTESTED (quarantined, §2): every result in which an offload-TRAINED cell beats a
resident-trained cell. The bf16 control pair (byte-identical weights, 0.0108
placement gap) proves the offload training path is a different experiment until T1
says otherwise.

UNRUN: routed-stream plan Phases 0–6 (all of plan-routed-v3); per-example eval
losses (D1); ∅ placement diff (D2); one-step train certificate (D3).

## 1. Binding numbers (use without re-deriving; sources in the audit)

```
∅ ladder (A5000, pinned eval):  fp4 1.5041 · nf4 1.4905 · int8 1.4811 ·
                                bf16 1.4818 · fp16 1.4780 · fp8 1.4724
G_int8 = 0.0094      G_total = 0.0087      coverage = 108%
Portability (3-seed): U_nf4 = +0.0003 (97% forfeited) · U_nf4off = −0.0045 (hurts)
                      D_int8 = 0.80 G · load-bearing resident tie: +0.07 G
Training paired:      precision effect −0.33 G @ offload (3/3) vs +0.00 G @ resident (1/3)
Single-run off−res:   nf4 +0.0015 · int8 −0.0105 · fp8 +0.0077 · bf16 −0.0108 · fp16 −0.0013
Offload peaks:        {bf16,fp16} 2.41 · {nf4,fp4} 2.52 · {int8,fp8} 2.72 GB
Qwen3 (A100):         T_ovh(nf4) 4.01 ms (but resident repeats 4.14/5.19 tok/s — ±13%)
                      implied bus ≈ 27 GB/s · prefetch-dq ≡ prefetch (bandwidth-bound)
```

## 2. Quarantine rules — in force until T1 resolves

- Q1: No claim, doc, figure, or README line may cite an offload-trained cell as
  evidence of a PRECISION effect. Offload-trained rows are labeled
  `pending-mechanism (D3)`.
- Q2: Every figure panel carries its provenance gate label (`claim_usable` /
  `debug_only`).
- Q3: Do not rerun the portability matrix or launch any multi-seed training before
  T1's branch is known — the reruns' design depends on it.
- Q4: R7 applies to T1 itself: if the certificate is red, the red is the
  deliverable. Never fix-and-rerun in one commit; the pre-fix result is committed
  first, the fix second, the post-fix rerun third, each citing the previous.

## 3. Work queue (priority order)

### T1 — Certificate D3: one-step train placement certificate  [DECIDES EVERYTHING]

**T1.0 Static path diff (zero GPU — do first).** Read the training code and
enumerate every flag/behavior that differs between resident and offload training:
gradient checkpointing on/off, `preserve_rng_state`, activation dtype, dataloader
worker count/seeding, batch assembly, RNG seeding points, eval/checkpoint cadence
config. If the offload branch forces checkpointing (or any RNG-touching flag)
that resident doesn't use, record it — the hypothesis is half-convicted statically.

**T1.1 One-step trio, bf16 test article (no quantization confound).**
From identical state (seed 1337, first training batch, fresh AdamW, LoRA B=0):
(a) resident step, (b) resident step repeat, (c) offload step — dropout OFF; then
the same trio dropout ON. Compare at earliest objects, in order: forward
logits/loss → adapter grads (per-tensor max-abs-diff, norm ratio) → post-step
adapter weights → optimizer state. Null = (a) vs (b) per dropout setting; if the
dropout-OFF null is bitwise, demand bitwise for (a) vs (c).

**T1.2 Repeat T1.1 at int8** (one trio) to check the mechanism carries to the
quantized path.

**Decision tree (pre-committed — file the result under its branch):**

| observation | conviction | next action |
|---|---|---|
| OFF: offload ≡ resident; ON: diverges beyond null | RNG consumption differs (recompute mask divergence / stream order) | (i) retract-or-transform per Q1; (ii) TRANSFORM TEST: one nf4 training pair, resident, with `preserve_rng_state=False` checkpointing — if resident+noise reproduces the offload gain, the finding becomes **recompute-noise regularization, placement-decoupled** (better claim than the one it replaces); (iii) fix RNG preservation, rerun ONE int8 pair, confirm the gap closes |
| OFF: forward logits diverge beyond null | forward numeric path differs (staging dtype / dequant order / workspace) | S1 red; localize with T3's ∅ diff; kernel-factor bug |
| OFF: forward matches, grads diverge | backward/recompute numeric path differs | S1 red; per-layer localization |
| Everything matches through dropout-ON | one-step machinery exact; anomaly lives at RUN level | diff run level: batch-hash sequences per placement (log first-K batch hashes), checkpoint/eval cadence, LR schedule realization; see T5 forensics |

Cost: T1.0 zero GPU; T1.1–T1.2 minutes each on one A5000.

### T2 + T3 (folded) — Per-example ∅ ladder, both placements  [PAYS D1 AND D2]

Rerun the ∅ evals for {fp4, nf4, int8, fp8, bf16, fp16} × {resident, offload}
with PER-EXAMPLE losses logged (12 eval passes; record eval-set size n).
Outputs: (i) example-paired deltas d_i between every mode pair — mean, sd,
SE = sd/√n, histogram; settles whether coverage = 108% and the fp8 < fp16 < int8 <
bf16 scramble exceed sampling error; (ii) resident-vs-offload ∅ diff per precision
= D2 = the serve-side placement certificate data (expect identity; any gap
localizes T1's anomaly to the forward path); (iii) eval determinism: run one mode
twice — repeat-null for all serve certificates. Cost: ~12–13 cheap eval passes.

### T4 — Workspace code-read  [ZERO GPU]

Verify from the offload implementation source (never from behavior — R9) the §4
audit hypothesis: quantized offload paths allocate a bf16-sized dequant workspace;
unquantized paths do not; quant slab is the only precision term. Output:
`docs/OFFLOAD_MEMORY_FACTS.md` with the named peak decomposition
(fixed + workspace + slab(p)) and predicted-vs-measured against the six offload
peaks. If confirmed, the campaign headline upgrades to: **bf16 training at 2.41 GB
(below the nf4 floor) vs 14.54 resident** — held until T1 clears its eval claim.

### T5 — Eval forensics  [GREPS]

From `olmoe_repeat_training_all.jsonl` and job logs: (a) eval-point COUNT per job —
best-eval is a min statistic; if slower offload runs got more eval draws, part of
the "win" is selection intensity; (b) best-vs-final gap per job (already known
larger under offload — quantify); (c) reconcile seeded query cells vs training
best-evals (1e-4 match for int8-off, ~1e-3 elsewhere — identify which checkpoint
each query leg loaded); (d) if batch hashes were logged, diff sequences across
placement; if not, add batch-hash logging to all future training runs.

### T6 — T_ovh protocol fix  [MINUTES]

Before any G1 commit: resident decode T_ovh measured back-to-back, same session,
≥5 reps, median ± spread, per (model, precision). Forced by the 4.14/5.19 tok/s
Qwen3 finding (±13% session noise > the ±15% PvM bound's headroom).

### T7 — Routed-stream Phases 0–1, per plan-routed-v3 + amendments (§4)

Unblocked and anomaly-immune (decode: no dropout, no backward, no recompute — the
D3 mechanism has no surface; serve certificates are the deterministic strong
family). Trace logger + recall telemetry + simulator, evaluated PER PRECISION with
the §4-A2 priors. OLMoE parts on A5000; Qwen3 parts on A100.

### T8 — Harvest / doc fixes  [AFTER T1 BRANCH KNOWN, except gate labels: now]

Gate labels on every panel (Q2) — immediately. fp8 added to the figure (best
frozen base, reversed placement effect). Figure caption replaced with the audit's
one-paragraph summary, edited per T1's branch. README prior-art paragraph
unchanged from v3 Phase 6.

## 4. Amendments to plan-routed-v3 (commit as `PLAN_ROUTED_V3_AMENDMENTS_A1-A4.md`, citing v3 — do not edit v3)

- **A1 — 16-bit routed-stream lane.** Add bf16/fp16 to the lane's precision set.
  Spans ≈ 4× nf4 (~10.6 MB/expert Qwen3); NO dequant workspace, NO dequant kernel —
  the simplest gather path. Host pinned requirement ≈ 61 GB for Qwen3 (S4 check
  first). Motivation: the ladder says serve ≥ int8; the audit says 16-bit is the
  offload memory floor; A2 says 16-bit is where the cache pays most.
- **A2 — Per-precision cache economics as G1 priors.** Reactive→ceiling tok/s gap
  = t_fetch/T_ovh: **+20% nf4 · +38% int8 · +79% bf16** (bf16 uses T_ovh(nf4) as
  placeholder until measured). Gain over reactive at hit rate h:
  (T_ovh+t_fetch)/(T_ovh+(1−h)·t_fetch) − 1 → at h=0.5: **+9% / +16% / +28%**.
  All pending 0.2b's BW_gather. Expected consequence: the <10% kill rule fires on
  the nf4 lane and spares int8/16-bit — kill per precision, as v3 already allows.
- **A3 — T_ovh measurement protocol** = T6 above; supersedes the single-run [ref]
  discipline in v3 §0.3 (formula unchanged; acquisition upgraded).
- **A4 — Decode-lane isolation note.** The D3 anomaly mechanism (dropout/RNG/
  recompute) has no surface at decode; the routed-stream lane proceeds in parallel
  with T1 and neither gates the other. Serve-side certificates remain the
  deterministic family.

## 5. Parallelization map

```
Pod A (A5000):  T1.0 (static) → T1.1/T1.2 (one-step trios) → T5 greps
Pod A or B:     T2/T3 (12–13 eval passes, per-example logging)
No GPU:         T4 code-read · T8 gate labels · amendments commit
Pod C (A100):   T6 (Qwen3 T_ovh reps) → T7 Phase 0 (0.2b probes) → Phase 1 traces
Blocked on T1:  portability reruns · any training claims · final caption
```

## 6. What NOT to do

- Do not fix any discovered bug and rerun in the same commit (Q4).
- Do not build any cache before Phase-1 h(S) exists (v3 kill rules govern).
- Do not quote debug_only rows as claims; do not drop the gate labels.
- Do not average or compare decode tok/s across sessions (audit: 10.12 vs 12.68
  same mode, same host class, different sessions).
- Do not re-derive the §1 numbers; cite the audit.

## 7. Stop conditions (inherited S1–S7 from v3, plus)

- S8: first new job fails to classify `claim_usable` under the fixed runner.
- S9: T2's example-paired SE shows G_int8 indistinguishable from 0 → the entire
  precision program drops to screening; report before proceeding.
- S10: T1 lands in the "everything matches" branch AND T5 finds no run-level
  difference → the anomaly is unexplained; halt all training-lane claims and
  escalate.

## 8. First session, in order (≈ 1 GPU-hour total)

1. Commit this handoff + the A1–A4 amendments file (prediction commits first).
2. T1.0 static diff + T4 code-read (no GPU). Report flags found.
3. T5 greps (eval counts, best-vs-final, checkpoint reconciliation).
4. T1.1 bf16 one-step trios (dropout OFF, then ON). File under the decision tree.
5. T2/T3 twelve ∅ evals with per-example logging; compute SEs; pay D1+D2.
6. Session report per v3's per-phase format, citing this doc's commit hash.

---

## Executor's filing notes (added at commit time; not part of the handoff text)

- `ROUTED_STREAMING_AGENT_EXECUTION_PLAN.md` (plan-routed-v3) is cited above but
  is **not present in this repository** at the time of this commit. Its rules
  R1–R10 and stop conditions S1–S7 are held as referenced-but-external; the
  quarantine rules Q1–Q4 and stop conditions S8–S10 defined in THIS document are
  in force locally. When v3 lands in-repo, it governs; nothing here edits it.
- In-repo name mapping: `MEASUREMENT_AUDIT_olmoe-qlora-grid-20260705-1351.md`
  = `docs/MEASUREMENT_AUDIT.md`.

---

<!-- ots-attestation-footer -->

**OpenTimestamps anchor (self-attestation footer):**

- **OTS proof timestamp for visible document:** `2026-07-05T16:58:44Z` (the moment the current `.ots` was submitted to the calendars; this is the legally operative timestamp for the visible file as published).
- **Disclosed pre-footer content hash:** `2e298e6bb17c06630ddac29fc723f191ba8bd51e673b5de751984b75b92dbdba` (the SHA-256 of the document *before* this footer was appended — disclosed inside the OTS-anchored visible document for human-readable historical reference; this hash is *not* the payload of the current `.ots` file).
- integrity-attestor glyph (`core.fingerprint`, first 8 bytes of the disclosed pre-footer hash): `[+?+#*?0@@:=&.00~]`
- Drunken-bishop randomart (full disclosed pre-footer SHA-256, OpenSSH-style):

```
+----[SHA256]-----+
|                .|
|               o.|
|              +.+|
|  .          +.oo|
|.o o  . S   . o..|
|o.*.oo o   . + . |
| +.*Bo+oo . o o  |
|  *B+*+o..   o   |
| oo**....    E.  |
+-----------------+
```

- **Payload hash actually covered by the current `.ots`:** see `ots info POST_AUDIT_WORK_QUEUE.md.ots`; by construction this is `SHA-256(this entire file including this footer)` and `ots verify POST_AUDIT_WORK_QUEUE.md.ots POST_AUDIT_WORK_QUEUE.md` succeeds against the on-disk bytes.
- Anchor file: `POST_AUDIT_WORK_QUEUE.md.ots`
- Calendars: a.pool.opentimestamps.org, b.pool.opentimestamps.org, a.pool.eternitywall.com, ots.btc.catallaxy.com
- **Provenance posture (load-bearing):** the **OTS proof timestamp** above is the legal anchoring time for the visible document — that is what the calendars witnessed. The **disclosed pre-footer content hash** is *not* anchored by the current `.ots` file; it is *disclosed inside* the OTS-anchored visible document as a human-readable historical record of what the file's bytes hashed to immediately before this footer was appended. A reviewer verifying the visible file runs `ots verify` against the on-disk bytes; a reviewer wanting to confirm the disclosed pre-footer hash recomputes `SHA-256` of the file with everything from `<!-- ots-attestation-footer -->` onward stripped. Both checks are independent; neither replaces the other.
