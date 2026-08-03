# K3 preregistration — does KL predict knowledge loss on fused-MoE 4-bit?

**Registered before any IKP measurement exists.** The commit that adds this file is the
timestamp; any result reported later is judged against what is written here, including the
parts that say we will decline to claim something.

## Provenance

- **Motivation.** Quesma, *"Quantization hurts knowledge nonlinearly — Qwen3.6 27B case
  study"*, published **2026-08-03**. Fifty-five GGUF k-quants (roughly 2–5 bit) of a single
  dense 27B model; reports **r = −0.981** between mean KL-from-bf16 and IKP accuracy. Their
  scope is **dense GGUF k-quants**. That correlation is theirs, not ours.
- **Benchmark.** IKP (Incompressible Knowledge Probes), Bojie Li / Pine AI —
  arXiv 2604.24827, repo `19PINE-AI/ikp`. Code MIT, probe set CC BY 4.0.
- **Probe set pinned:** `data/probes/final_probe_set_clean.json`, n = 1311,
  sha256 `a64ab7f3c06c919986298ffdaa501843ce4238494219c4c74d8f108f7d65c1db`
  (the 1,311-item cleaned subset used in the paper, not the raw 1,400).
- **Tiers:** T1 200, T2 200, T3 183, T4 188, T5 181, T6 187, T7 172.

## The question

Does the KL-to-knowledge relationship reported for dense GGUF k-quants show up on
**fused-MoE NF4/FP4** as implemented in this repo? Genuinely open — different quantization
family, different granularity, different model class.

## Design, fixed now

- **Model:** `ibm-granite/granite-3.0-1b-a400m-instruct` — the same model K2 measured, so
  every accuracy point attaches to an already-published KL from an explicitly named
  reference (commit `dec5798`) rather than to a number produced for this test.
- **Paths, i.e. the KL points:**
  1. **bf16 base** — the reference. KL ≡ 0 *by construction*: an anchor, not a measurement.
  2. **experts4bit NF4** — KL = 1.407598e-01, already measured.
  3. **experts4bit FP4** — KL to be measured with the same K0-gated instrument *before*
     any accuracy is scored, so the x-axis is fixed before the y-axis exists.
- **Decoding:** greedy, `max_new_tokens=48`. Deterministic, so a path's accuracy is a
  property of the path and not of a sampling seed.
- **Phrasings:** IKP's runner uses three phrasings per probe. We use the direct-question
  phrasing only. **Declared deviation**, for compute budget.
- **Refusals:** `is_refusal()` reproduced verbatim from `src/scorer.py`; probes that refuse
  are excluded from scoring, per the reference runner.
- **Judge:** the judge prompt copied verbatim from `src/scorer.py`, served by our local
  Qwen3-30B-A3B. The reference scorer ships `create_ollama_judge(model="qwen3:4b")`, so a
  *local* judge is the method's own default rather than a deviation from it. However, the
  published per-model numbers were judged by Gemini 3 Flash Preview: **our absolute
  accuracies are not comparable to the published leaderboard.** Only within-study,
  cross-path comparison is valid — same judge, same probes, same phrasing, same decoding.
- **Scoring:** accuracy = correct / scored, λ = 0 (no hallucination penalty), per the paper.

## Power — the decisive limitation, registered before seeing any result

Quesma had **55** quantization points spanning 2–5 bit. We have **three**, one of which
(bf16) is definitionally zero — so **two real quantizations**. A correlation coefficient
estimated from three points has a confidence interval spanning nearly the whole [−1, +1]
range; computing one would be theatre dressed as evidence.

**Registered: we will not report an r value, and we will not describe this work as
reproducing, confirming, or refuting r = −0.981.** We will report the (KL, accuracy-by-tier)
pairs, and at most whether the *direction* is consistent with the published sign. Any
stronger statement needs many more quantization levels than this stack currently supports.

## Floor-effect kill switch

IKP accuracy scales log-linearly with parameter count (≈ +15 pp per 10×, R² = 0.917 across
89 models). At 1.3B total / 400M active, granite sits low on that line, and the obscure
upper tiers are expected at or near floor. Floor effects destroy between-path resolution:
two paths both scoring ~0 on T7 tells us nothing about either.

**Rules, decided now rather than after seeing the numbers:**

- If the **bf16 reference scores < 10% overall**, the configuration is declared
  **under-powered**: report tier accuracies, make no KL-to-accuracy inference at all.
- Any **tier whose bf16 accuracy is < 5%** is reported but **excluded from every directional
  statement** — at that level a few items of difference is indistinguishable from noise.
- **Binomial standard error is reported next to every accuracy**, so a reader can see
  immediately which differences fall inside it.

## What would count as what

| outcome | meaning |
|---|---|
| **Consistent** | NF4 and FP4 both below bf16, and the larger-KL path scores lower, outside binomial noise |
| **Inconsistent** | the larger-KL path scores *higher*, outside binomial noise |
| **Uninformative** | all differences inside binomial noise, or the kill switch fires |

"Uninformative" is a real and fully acceptable outcome of this design, not a failure to be
explained away afterwards. Given the model scale it is the single most likely result, and
saying so now is the point of registering it.

## Scope

Fused-MoE NF4/FP4 on this stack, one model, one local judge, one phrasing. Registers **no
performance claim**. MXFP4 paths remain unmeasured (gpt-oss is absent from the offline
cache on this host).

---

## Amendment 1 — model and hardware (2026-08-03)

**Registered before any accuracy number exists.** Verified at the time of writing: zero IKP
answer files had been produced. Every local generation attempt died on a harness bug
(`apply_chat_template` returns a `BatchEncoding`, which `generate()` rejects with a bare
`AttributeError`), so nothing was generated, nothing was judged, nothing was scored. This
amendment is therefore pre-data, not a reaction to a result.

**Trigger.** The local A2000 is held by the judge drain, which is wanted work. Rather than
preempt it, the operator directed renting a GPU — which also lifts the VRAM ceiling that
forced the original model choice.

**Changes:**

1. **Primary model: `ibm-granite/granite-3.0-1b-a400m-instruct` → `openai/gpt-oss-20b`.**
   - At 1.3B, granite was expected to trip the floor kill switch: the honest outcome would
     most likely have been "under-powered, no inference". 21B total sits inside IKP's
     sensitive band and is comparable to Quesma's 27B.
   - gpt-oss ships **MXFP4 natively**, which turns the two rows K2 had to declare
     `not_measured` into measurable ones and supplies the **requant-tax** point the
     directive explicitly asked for.
   - It remains a **fused-MoE** model, which is the whole domain of this repo's claims. A
     dense model of convenient size would not exercise the path under test at all.
2. **Reference for the gpt-oss rows: the dequant-to-bf16 path of the same shipped bytes.**
   There is no bf16 original of gpt-oss, and no table cell may imply one exists.
3. **KL x-axis becomes:** dequant-bf16 (0 by construction), native MXFP4, MXFP4→NF4 requant.
4. **granite is demoted to an optional secondary** low-power datapoint. Its KL is already
   published (1.407598e-01, commit `dec5798`) and stands regardless.

**Unchanged — every part of the restraint.** Still roughly three points, so still **no r**,
and still no "reproduced" or "refuted" language against r = −0.981. The kill switch and the
5% tier floor carry over verbatim, now applied to the dequant-bf16 reference. The judge is
still the verbatim IKP prompt served locally, so absolute accuracies are still not
comparable to the Gemini-judged published numbers.

**Worth recording:** that harness bug is exactly the kind that would have been dangerous if
it had failed quietly instead of loudly. Empty answers score as uniformly wrong, which is
indistinguishable from catastrophic quantization damage — a clean, publishable-looking
result that would have been an artifact of my own code.
