# KL-from-reference as a fidelity instrument — findings

2026-08-03/04. Everything here is reproducible from the receipts in
`bench/receipts-kl-fidelity-20260803/`. The preregistration and its two amendments are in
`bench/K3-PREREG.md`; read that first if you care whether a number was decided before or
after the data existed.

**Headline: KL from a reference is a good instrument and a poor predictor.** It measures
distributional divergence exactly and cheaply. It does *not* determine how much knowledge a
quantization costs — *where* the damage lands matters as much as how much there is. The
threshold this work set out to find does not exist in the form it was looking for.

---

## What was built

**K0 — the instrument** (`kl_fidelity.py`). `KL(P_ref ‖ P_test)`, teacher-forced, fp64
log-softmax, full vocabulary, token-weighted. Three controls must pass before any path is
measured, and the driver refuses to run without a passing receipt *on that host* — a
fidelity number from an unvalidated instrument is worse than none, because it looks
authoritative.

| control | result |
|---|---|
| self-KL, model against itself | exactly 0.0 |
| single-byte weight perturbation | detected, 2.3e-12 |
| known-nonzero vs analytic ½·Var_p(Δlogit) | **ratio 1.0000** |

Control 3 initially "failed" against a band I had invented. The measurement was right and
the band was a guess, so the guess was replaced with the second-order expansion of KL — a
real identity. Agreement to four decimals is evidence the estimator is correct, not merely
non-crashing.

**K1 — the prompt set** (`kl_prompts.py`). 200 prompts, sha256 `eaa7792260b3f10d…`, frozen
before any measurement. Strata: general 72 / technical 55 / code 55 / longctx 18.
Aggregation is token-weighted and longctx is ~52% of scored tokens, so per-stratum
reporting is mandatory, not decorative — the pooled mean understated general-prompt
divergence by 2.26× on the first model measured.

---

## K2 — the path table

Every row names its reference explicitly. For natively-4-bit models the reference is the
**dequant path of the same shipped bytes** — there is no bf16 original of gpt-oss, and no
cell may imply one.

| path | reference | KL | top-1 |
|---|---|---|---|
| NVMe-streamed vs DRAM-resident | DRAM-resident, same run | **0.000** | 1.000 |
| native MXFP4 (gpt-oss-20b) | dequant-to-bf16 of the same bytes | 1.539e-03 | 0.984 |
| MXFP4 → NF4 requant | dequant-to-bf16 of the same bytes | 2.214e-02 | 0.932 |
| NF4 of a bf16 base (granite) | the bf16 base | 1.408e-01 | 0.863 |
| FP4 of a bf16 base (granite) | the bf16 base | 2.127e-01 | 0.821 |
| NF4 of a bf16 base (OLMoE) | the bf16 base | 1.334e-01 | 0.900 |
| FP4 of a bf16 base (OLMoE) | the bf16 base | 1.618e-01 | 0.879 |
| VRAM-resident vs host-streamed | VRAM-resident | **unmeasured** | — |

The zero row is **confirmatory, not novel** — `tests/test_nvme_residency_equivalence.py`
already proved bit-identity with `torch.equal`. Its only value is restating identity in the
unit everyone else reports degradation in.

FP4 is worse than NF4 on both models (1.5× and 1.2×), the expected direction since NF4's
grid is fitted to normally-distributed weights. The last row is blocked by a library gap,
not by physics: `enable_pipelined_residency` rejects everything `load_moe_4bit_streaming`
builds. It is the one cell of the table that has never run.

### A near-miss worth keeping

The native-MXFP4 loader is **guarded**, and the guard fired on its first real use. Without
the `kernels` package, transformers silently "defaults to dequantizing the model to bf16"
and returns a working model. That model would have scored **KL = 0** against the bf16
reference, and the row would have read *native MXFP4 is distributionally identical to its
own dequant* — a clean, publishable-looking headline manufactured entirely by a missing pip
package. Two traps recorded alongside it: the `kernels` version ceiling is **exclusive**
(the documented max 0.16.0 is itself rejected; 0.15.2 works), and a row that raises *after*
loading leaks 13–42 GB into the next row's unrelated OOM.

---

## K3 — does KL predict knowledge loss?

Benchmark: IKP (Incompressible Knowledge Probes), Bojie Li / Pine AI, arXiv 2604.24827,
probes CC BY 4.0, pinned by sha256. Judge: IKP's verbatim judge prompt served by a local
Qwen2.5-7B — the reference scorer's own default is `create_ollama_judge("qwen3:4b")`, so a
local judge is the method's design. Absolute accuracies are therefore **not** comparable to
the published Gemini-judged leaderboard; only cross-path comparison within this study is
valid. Significance is paired McNemar on identical probes.

**gpt-oss-20b** (KL ≤ 2.2e-02): no detectable loss. dqbf16 43.6%, mxfp4 44.2%, nf4-requant
44.9%; requant vs reference 82 vs 99 discordant, p=0.23. The `dqbf16`/`mxfp4` pair also
served as a validity check — dequantizing MXFP4→bf16 is *exact* (E2M1 values times a
power-of-two scale need ≤3 mantissa bits; bf16 has 8), so they are the same function and
must agree. They did: 24 vs 31, p=0.42. Had they diverged, the harness was wrong and the
run should have been binned rather than interpreted.

**granite-3.0-1b-a400m-instruct** (KL ≥ 1.4e-01): large, unambiguous loss. bf16 23.2%, nf4
18.6% (101 vs 40, p=2.9e-07), fp4 17.6% (114 vs 40, p=2.0e-09).

---

## The threshold that isn't

Those two results looked like a clean regime boundary, so the rest of the work went hunting
for it. It isn't there.

**Varying how many layers are quantized** (granite, T1–T3, 583 probes, bf16 ref 49.4%) gave
a non-monotonic result: k=2 (KL 1.63e-02) scored *worse* than k=4 (KL 2.28e-02). Only k=16
survived Bonferroni correction.

**Varying which layers, at fixed k=8**, explains why:

| spec | KL | acc | Δ | p |
|---|---|---|---|---|
| mid:8 | 1.743e-02 | 48.0% | −1.4 | 0.31 |
| stride:8 | 2.442e-02 | 47.0% | −2.4 | 0.10 |
| first:8 | 5.544e-02 | 46.0% | −3.4 | 0.023 |
| last:8 | 6.461e-02 | 47.3% | −2.1 | 0.15 |
| first:24 | 1.405e-01 | 38.7% | −10.7 | 5.8e-09 |

**At fixed k=8, KL spans 3.71× while accuracy spans 2.1 pp**, and with five comparisons no
k=8 placement is significant. A large move in KL bought almost no move in accuracy. The
matched-KL contrast across runs is sharper still: `first:16` at KL 6.82e-02 costs −5.2 pp
(p=0.0021) while `last:8` at KL 6.46e-02 costs −2.1 pp (p=0.15). Near-identical divergence,
materially different damage; early layers cost more than late ones. Within the placement
run the ordering even inverts — `first:8` has *lower* KL than `last:8` and *worse* accuracy.

That −5.2 vs −2.1 gap is a point-estimate difference (SE_diff ≈ 2.9 pp) and is not
individually significant. The solid result is the **dissociation**: 3.71× in KL against
2.1 pp in accuracy.

It also retires the earlier anomaly. k=2 beating k=4 was not noise to be drowned in more
probes — it was placement, visible before there was a design capable of seeing it.

---

## What is claimable

- The instrument is validated, per host, with an analytic cross-check.
- The **requant tax** — converting a shipped MXFP4 checkpoint into NF4 — is 2.21e-02, about
  14× the kernel-level noise floor of serving the shipped bytes natively, and sits **below**
  anything that costs measurable knowledge.
- Full 4-bit quantization of a 1.3B MoE is genuinely destructive: −10.7 pp, p=5.8e-09.
- **Partial quantization is cheap in knowledge terms**: 8 of 24 layers costs nothing
  detectable at any placement tested. That is a real serving lever, now expressible via
  `quantize_layers` (#63).
- **KL is not a sufficient statistic for knowledge loss.** Treat a KL number as a change
  detector, not as a damage estimate, and never carry a format's reputation across models.

## What is not claimable

- No correlation coefficient. Quesma's r = −0.981 is over **55 dense GGUF k-quants**; this
  work has a handful of points on fused-MoE formats. Nothing here reproduces, confirms, or
  refutes it, and the dissociation above argues against fitting one at all.
- Nothing about MXFP4 beyond gpt-oss, or about tiers T4–T7, which sat at floor throughout
  and were excluded from directional statements by the registered floor rule.
- No performance claim of any kind. This work built an instrument and measured existing
  paths.

## Retracted along the way

Kept visible rather than quietly dropped, since each was stated confidently first:

1. **"KL predicts churn, not net loss."** True at ≤2.2e-02, false at ≥1.4e-01. Withdrawn as
   a general claim once replication reversed it.
2. **"The requant tax is 14× worse."** Divides information loss by kernel numerics —
   apples-to-oranges. The 1.54e-03 native row is a noise floor, not a comparable baseline.
3. **"The int8 anchor holds the model fixed and varies only perturbation size."** It
   perturbs a *different set* of weights: bitsandbytes cannot touch fused 3D MoE experts, so
   it quantized only the non-expert Linears. Recorded as post-data Amendment 2.
4. **The threshold framing itself.** Superseded by the placement result above.

## Open

- The VRAM-vs-host-streamed row — blocked on `enable_pipelined_residency` accepting
  `ExpertsLoRA.base`. Note `r=0` is *not* the fix: `lora.py` rejects it by design.
- Whether the placement effect is depth (early vs late) or something correlated with it
  (router sensitivity, attention-adjacency). Fixed-k, matched-KL, larger n.
- Everything here is one judge, one benchmark, and MoE models under ~21B.
