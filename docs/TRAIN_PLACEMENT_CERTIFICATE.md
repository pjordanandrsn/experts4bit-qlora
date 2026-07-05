# Train placement certificate (debt D3) — static analysis, forensics, and pre-registered predictions

```
status:   prediction document — committed BEFORE the T1.1/T1.2 GPU runs
queue:    POST_AUDIT_WORK_QUEUE.md T1 (T1.0 static done here; T1.1/T1.2 designed here)
          + T5 forensics results (local greps, this bundle)
question: is "offload-trained beats resident-trained" a placement MECHANISM,
          or noise/confound? The audit's bf16 control (byte-identical weights,
          0.0108 best-eval gap across placement) demands an answer before any
          offload-trained eval claim survives.
```

## T1.0 — Static path diff (resident vs offload training)

Every flag/behavior enumerated from `experts4bit_qlora/train.py`, `offload.py`, `lora.py`,
`_vendor/experts.py` (single read-through, no behavior probes):

| axis | resident | offload | differs? |
|---|---|---|---|
| gradient checkpointing | ON (`use_reentrant=False`), unconditional (train.py:181) | same line, same flag | **no** |
| `preserve_rng_state` | torch default (True) | same | **no** |
| dropout | none — `ExpertsLoRA`/`LoRALinear` define no dropout; OLMoE `attention_dropout` defaults 0.0 (runtime-attested by the trio jobs) | same | **no** |
| data order | sequential `iter(dataset)`, **no shuffle, no sampler, no workers** (train.py:187–199) | same | **no** |
| batch assembly | per-example, `GRAD_ACCUM` micro-batches | same | **no** |
| RNG seeding | one `torch.manual_seed(SEED)` before load (train.py:116) | same | **no** |
| RNG consumption after LoRA-A init | none found on the training path (no dropout, greedy-only generate, deterministic routing/top-k) | same | **no** |
| eval cadence | step-based `EVAL_EVERY=50`, 3 draws at steps 50/100/150 + 1 AFTER eval | same | **no** |
| LR schedule | cosine+warmup, step-indexed | same | **no** |
| optimizer | fresh AdamW, LoRA `B=0` init | same | **no** |
| placement branches | `model.to(DEVICE)` | loader `offload=1`: quantize-on-GPU per layer → capture pinned CPU homes (byte-exact `.to("cpu")` of the same tensors) → stage/evict hooks, single-slot, current-stream H2D | **yes — the only branch** |

**Verdict: statically half-ACQUITTED, not half-convicted.** The handoff's leading hypothesis
(offload forces checkpointing / RNG-touching flags resident doesn't use) is dead on arrival:
checkpointing is unconditional, and there is **no dropout and no post-init RNG consumption on
this path at all** — `preserve_rng_state` has nothing to preserve, and the decision tree's
"recompute mask divergence" mechanism has no surface. The offload hooks (`offload.py`) consume
no RNG and stage byte-identical bytes.

What IS present, in **both** placements: nondeterministic CUDA kernels —
`index_add_` (fp32 atomics) in the MoE combine (`lora.py:283`, `experts.py:513`), and
(kernel-dependent) SDPA backward. Fixed seeds do not fix these; run-to-run divergence is
expected without deterministic algorithms, and over 150 steps it compounds chaotically.

## T5 — Eval forensics (local bundle greps, run before the trio)

**(a) Selection intensity: equal by construction — hypothesis refuted.** Eval draws are
step-based (3 per job at steps 50/100/150, every job, both placements). Slower offload runs do
NOT get more draws.

**(b) best-vs-final gap** (per-mode mean over 3 seeds): int8 +0.0023, nf4 +0.0044,
int8-offload +0.0058, nf4-offload +0.0063 (max +0.0123, nf4-offload/s2027). Offload gaps run
slightly larger, i.e. its best-of-3 sits further below where training ended — consistent with
noisier trajectories, not with a better optimum.

**(c) NEW CONFOUND — the repeat grid is architecture-mixed.** Job attribution shows the 12
repeat training jobs split across **RTX 4090 and RTX A5000** pods. The BEFORE-training (∅)
eval — same model bytes, no adapter effect (LoRA B=0) — is **deterministic to 4dp per
architecture** (two distinct A5000 pods agree exactly) but **differs across architectures**:

| ∅ eval | A5000 | 4090 | Δ(4090−A5000) |
|---|---|---|---|
| int8 | 1.4811 | 1.4837 | +0.0026 |
| nf4 | 1.4905 | 1.4959 | +0.0054 |

Consequences, in order of severity:

1. **Every seed-matched resident-vs-offload pair in the repeat grid is a cross-architecture
   comparison** (all 6 int8 pairs and all 6 nf4 pairs have the two placements on different
   GPUs). The evaluator offset (~0.003–0.005) is the same order as the placement gaps being
   claimed (0.0015–0.0108). The int8-offload "best in 3/3 seeds" is therefore
   RNG-confounded AND architecture-confounded.
2. **The bundle's consolidated CSV mislabels host**: all rows say "RunPod RTX A5000 24GB";
   6 of 12 repeat training jobs ran on a 4090. (Bundle stays frozen; corrected at the
   dashboard/figure layer and here. The per-job `gpu_name` fields were always correct.)
3. **D2 pre-answered at 4dp**: on the same architecture, ∅ eval is placement-IDENTICAL in
   every observable pair (A5000 int8 1.4811 resident and offload; 4090 int8 1.4837 both;
   same for nf4). The serve/forward path is placement-clean at 4dp; T2/T3 sharpens this to
   per-example resolution.
4. The ∅ ladder (§1 of the work queue) is single-host (A5000 values match it) and remains
   internally consistent — but **G is architecture-dependent** (G_int8: 0.0094 on A5000,
   0.0122 on 4090), so ladder claims are per-architecture claims.

**(d) Batch hashes: not logged** — but T1.0 shows the batch sequence is placement-invariant
*by construction* (sequential iterator, no shuffle), so batch-hash divergence is impossible
here. Logging added to future runs is a nice-to-have, not load-bearing.

## Candidate mechanisms going into T1.1/T1.2

- **M-A (placement changes one-step math)** — forward or gradient numerics differ between a
  staged copy and a resident tensor. Static read predicts NO (same bytes, same kernels, same
  shapes/strides).
- **M-B (run-level chaos)** — nondeterministic kernels (present in both placements) make any
  two 150-step runs diverge; same-seed placement pairs differ like different-seed pairs
  (repeat-grid within-mode seed σ ≈ 0.007–0.012 vs placement gaps 0.0015–0.0108 — same scale).
- **M-C (evaluator/architecture offset)** — CONFIRMED by T5(c) for the repeat grid; a
  ~0.003–0.005 additive offset rides on every cross-architecture comparison.

## Pre-registered predictions (filed before the GPU run)

- **P1**: Under default kernels, the resident null (a) vs (b) is NOT bitwise (atomics in the
  MoE combine); loss deltas ~1e-3 or below at one step, grad max-abs-diffs small but nonzero.
  The decision tree's "bitwise null" branch is unreachable in production config.
- **P2**: (a) vs (c) [resident vs offload] diverges at the SAME order of magnitude as the
  (a) vs (b) null — placement adds nothing beyond run-to-run noise. (The audit's 0.0108 bf16
  gap is same-host/single-run, so M-B alone must explain it if P2 holds at 150-step scale;
  the trio only certifies one step.)
- **P3**: Under `torch.use_deterministic_algorithms` + math-SDPA + `CUBLAS_WORKSPACE_CONFIG`,
  the null (a) ≡ (b) tightens to bitwise IF every op on the path has a deterministic
  implementation (the trio records which ops warn); and then (a) ≡ (c) bitwise too —
  placement fully acquitted at one-step granularity.
- **P4**: The dropout-ON legs (attention dropout force-enabled to 0.1) behave like OFF:
  `preserve_rng_state=True` checkpointing replays masks identically in recompute, and staging
  consumes no RNG, so placement still adds nothing.

If P1–P3 hold → decision-tree branch 4 ("everything matches; anomaly lives at RUN level"),
with M-B + M-C as the run-level explanation T5 already supplies — that is NOT stop-condition
S10 (an unexplained anomaly); it is an explained one: **"offload-trained wins" was never a
mechanism, it was unseeded kernel noise sampled 3 times, read through architecture-offset
evaluators.** If any of P1–P4 fail, the corresponding tree branch governs and S1 goes red.

## Trio design (T1.1 bf16, T1.2 int8) — `scripts/one_step_certificate.py`

One job per (quant_type, kernel-regime, dropout) config; each job runs three legs in-process
with full teardown + re-seed between legs: (a) resident, (b) resident repeat, (c) offload.
Seed 1337; first `GRAD_ACCUM=4` micro-batches of the grid's dataset order; fresh AdamW;
LoRA B=0 by construction; one optimizer step. Per leg, recorded at fp64/bytes:
per-micro-batch forward losses, logits sample (micro-batch 0, strided subsample) + SHA-256 of
full logits bytes, per-tensor adapter-grad norms/max-abs + SHA-256, post-step adapter-weight
hashes, optimizer-state hashes. Comparisons ((a)vs(b) null, (a)vs(c) placement) computed
in-job and written to `result.json`; raw tensors kept job-local. Jobs (5):

```
cert_olmoe_bf16_dropoutOFF_default        cert_olmoe_bf16_dropoutOFF_deterministic
cert_olmoe_bf16_dropoutON_default         cert_olmoe_int8_dropoutOFF_default
cert_olmoe_int8_dropoutOFF_deterministic
```

All five run on ONE pod (single GPU) — the T5(c) confound makes single-host execution a hard
requirement for every future comparison, this one first.

## RESULTS (2026-07-05, pod 41a04ccbc77d, RTX A5000, commit 14889f8) — decision-tree row 4

Q4 lineage: rev1 red (`runs/results/postaudit_cert_rev1_FAILED.md`, df351f3 — harness `_sha`
crash, no scientific object measured) → `_sha` fix (14889f8) → this rev2 rerun. Raw:
`runs/results/postaudit/postaudit_jobs/cert_olmoe_*_rev2/result.json`; leg tensors job-local
on the volume.

**Every comparison in every configuration is BITWISE-equal — the null (a≡b) and the
placement (a≡c) alike:**

| trio (rev2) | forward losses | logits(mb0) | 192 grad tensors | 192 post-step weights | 384 opt-state tensors |
|---|---|---|---|---|---|
| bf16 dropout-OFF default | ≡ | ≡ | ≡ | ≡ | ≡ |
| bf16 dropout-OFF deterministic | ≡ | ≡ | ≡ | ≡ | ≡ |
| bf16 dropout-ON (16 modules @ p=0.1) | ≡ | ≡ | ≡ | ≡ | ≡ |
| int8 dropout-OFF default | ≡ | ≡ | ≡ | ≡ | ≡ |
| int8 dropout-OFF deterministic | ≡ | ≡ | ≡ | ≡ | ≡ |

Deterministic-mode op warnings: none (0 in both deterministic trios).

**FLAGGED (self-caught, pending rev3 verification): offload engagement in leg (c) is not
self-attested by the rev2 artifacts.** The rev2 script reset peak-memory at leg END, so each
leg's recorded peak carries in-process carryover and leg (c)'s peaks read anomalously
(bf16 14.1 GB / int8 8.1 GB — neither offload- nor cleanly resident-scale). Because leg (c)'s
bitwise-equality is consistent with BOTH an engaged-offload leg and a silently-resident one,
the rev2 certificate is provisionally supported rather than self-contained: the engagement
default rests on the eval-path evidence (identical loader call + flag wiring measurably
engages offload in fresh processes — 2.53 vs 14.77 GB for bf16 in the n=64 ladder jobs).
Rev3 adds a vacuity guard (fails loudly unless all 16 expert bases are 0-element placeholders
at load under offload, and ≤1 base is resident after the step) plus leg-scoped memory
accounting, and reruns one bf16 + one int8 trio. Until rev3 lands, the row-4 filing stands on
eval-path evidence + the guard-free rev2 bitwise results.

**REV3 (2026-07-05, commit 160bfda): engagement ATTESTED, certificate now self-contained.**
Both trios: leg (c) shows **16/16 expert bases evicted at load, exactly 1 GPU-resident after
the step** (offload engaged, single-slot invariant honored), leg-scoped peaks at offload
scale (bf16 3.75 GB / int8 3.78 GB vs resident leg-a 14.85 / 9.15 GB), while resident legs
show 0/16 evicted. Null AND placement remain **bitwise** on every object. The rev2 result was
not vacuous — offload genuinely engaged and one training step is still bit-identical to
resident. D3 closes self-contained; the row-4 filing no longer leans on eval-path inference.

**Predictions scored:** P2, P3, P4 — confirmed (and more sharply than predicted). P1 —
**wrong in the informative direction**: the default-kernel null is bitwise at one step; the
MoE-combine atomics and SDPA backward did not produce run-to-run noise on these shapes.

**Filing (pre-committed tree, row 4): the one-step training machinery is placement-EXACT —
bitwise, in both quantized and passthrough paths, with and without dropout, under default and
deterministic kernels. M-A is refuted. The anomaly, if real, lives at RUN level.** Combined
with T2/T3 (serve path bitwise placement-exact, 384/384) and T5:

- For the **repeat grid**, the run-level difference IS identified: every seed-matched
  placement pair was cross-architecture, with evaluator offsets (0.0026–0.0054) at the scale
  of the claimed effects. The int8-offload "3/3 seeds" claim stays quarantined
  (pending-mechanism) and is now best explained as architecture offset + run-level variance,
  not a placement mechanism.
- For the **single-run same-host bf16 control pair** (the audit's 0.0108): unexplained by any
  measured mechanism — one-step training is bitwise, eval is bitwise, yet two 150-step legs
  differed. Divergence must onset mid-run (a shape-dependent nondeterministic kernel on a
  later batch is the leading candidate; batches vary in length). This is a SCOPED S10: not an
  apparatus halt (the quarantine already covers it), but the training-lane claims stay frozen
  until a **divergence-onset probe** runs — gated proposal: one 150-step twin trio
  (resident/resident/offload, per-step loss + adapter hashes, one pod). ~2.5 A5000-hours.
  This same run doubles as S-E's twin pair (SPECULATIVE_LANES_PLAN.md) with the dropout-ON
  leg available as the certificate-validated order-1 perturbation source.

---

<!-- ots-attestation-footer -->

**OpenTimestamps anchor (self-attestation footer):**

- **OTS proof timestamp for visible document:** `2026-07-05T22:21:53Z` (the moment the current `.ots` was submitted to the calendars; this is the legally operative timestamp for the visible file as published).
- **Disclosed pre-footer content hash:** `7b074e8ff0e690ced6dffebf3aa85e05fff210c44cf795a74d7e6bbb51168c2c` (the SHA-256 of the document *before* this footer was appended — disclosed inside the OTS-anchored visible document for human-readable historical reference; this hash is *not* the payload of the current `.ots` file).
- **Prior disclosed pre-footer hashes (chain, newest first):**
  - `2026-07-05T19:39:08Z` `a657bccbe80139f15e9633d3536d9ae8157b089a1851604bdf4b81ad8db00ac9`
  - `2026-07-05T18:09:34Z` `10a9897aafde7afe3aea060cb02905dc3eba2156620789b4b971a1e36ad31f4c`
  - `2026-07-05T17:13:37Z` `c7d3973bd217e7fc444d351ad588e66680b4ad3b69f3e532a2293eb652c319a8`
- integrity-attestor glyph (`core.fingerprint`, first 8 bytes of the disclosed pre-footer hash): `[=@.=o?*$$.?0#.&?]`
- Drunken-bishop randomart (full disclosed pre-footer SHA-256, OpenSSH-style):

```
+----[SHA256]-----+
|             . .o|
|            = +.=|
|           E * Bo|
|            = . =|
|        S o  +  =|
|         B +. o+.|
|        +.*.+o.o.|
|       o.=oo o+..|
|       .+oo..o==*|
+-----------------+
```

- **Payload hash actually covered by the current `.ots`:** see `ots info TRAIN_PLACEMENT_CERTIFICATE.md.ots`; by construction this is `SHA-256(this entire file including this footer)` and `ots verify TRAIN_PLACEMENT_CERTIFICATE.md.ots TRAIN_PLACEMENT_CERTIFICATE.md` succeeds against the on-disk bytes.
- Anchor file: `TRAIN_PLACEMENT_CERTIFICATE.md.ots`
- Calendars: a.pool.opentimestamps.org, b.pool.opentimestamps.org, a.pool.eternitywall.com, ots.btc.catallaxy.com
- **Provenance posture (load-bearing):** the **OTS proof timestamp** above is the legal anchoring time for the visible document — that is what the calendars witnessed. The **disclosed pre-footer content hash** is *not* anchored by the current `.ots` file; it is *disclosed inside* the OTS-anchored visible document as a human-readable historical record of what the file's bytes hashed to immediately before this footer was appended. A reviewer verifying the visible file runs `ots verify` against the on-disk bytes; a reviewer wanting to confirm the disclosed pre-footer hash recomputes `SHA-256` of the file with everything from `<!-- ots-attestation-footer -->` onward stripped. Both checks are independent; neither replaces the other.
