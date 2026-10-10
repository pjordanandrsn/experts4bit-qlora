# P130 (e4b#1313): speculative decoding, Phase 0 -- acceptance on the shipped default's target, and its B = 1 price

## The question

Speculative decoding is the largest B = 1 lever left after P127, per the scoping notes on #1313. How many tokens does
each verify step yield on `Qwen/Qwen3-30B-A3B` at e4b's shipped default? The lane measures two draft sources:
- **The licensed EAGLE-3 head** (its acceptance τ).
- **The zero-dependency n-gram (prompt-lookup) floor.**

It then prices both with e4b's own verify-step model. The answer decides whether a speculative-decoding serving lane
(Phase 1: the verify path, KV rollback, graphs, a quality gate) is worth building. This lane builds none of it.

Phase 0 is a single rental, as the maintainer ruled (bus, 2026-10-10T03:25:35Z): one small run, no proving rental.
- **No τ without the target's live hidden states.** The EAGLE-3 draft needs them, so τ cannot be computed offline.
- **No owned card serves the target.** It needs the served path on sm_89+.
- **Distinct experts per verify** come from the #1469 routing census when it lands, not from this box.

## The draft head (pinned)

| field | value |
|---|---|
| repo | `RedHatAI/Qwen3-30B-A3B-speculator.eagle3` |
| revision | `6afc5aa2477b923467fb9a8d906782b984a9a6ba` (2026-03-31) |
| license | apache-2.0 |
| `config.json` | `speculators_config.verifier.name_or_path` = `Qwen/Qwen3-30B-A3B`; sha256 `746d2d4cbcd0682b4818bc8aa65abb16410cb8fa556e00326e644cf9b391461a` |
| `README.md` | "Verifier: Qwen3-30B-A3B", trained with reasoning enabled (Magpie + UltraChat); sha256 `92a7a4fa8f0467960cb37ab837629814b79cd17ef28b7a8e969dfda03af0da4b` |
| `model.safetensors` | 1,044,539,336 B; sha256 `d2d6e2e63e09dc755053ae5c98cdececae3611ae5e202d4fa5411126dd3b1dfa`, checked on the box |

**The verifier field is not proof on its own.** `RedHatAI/Qwen3-30B-A3B-Thinking-2507-speculator.eagle3` has a
byte-identical `config.json`. The README corroborates the target: its Verifier line, its link to the base model, and its
pointer to the Instruct-2507 head for non-reasoning use.

**The card's own acceptance on this target** (vLLM 0.15.0, temperature 0, chat template): τ at k = 1 is 1.66–1.84
across seven sets, and at k = 3 it is 2.30–3.02.

**The head's tensors** (bf16; read from the file's safetensors header):
- one Llama layer whose q/k/v take 4,096 inputs (the embedding and the hidden, concatenated);
- `fc` 6,144 → 2,048;
- a 64,000-token draft `lm_head`, with `d2t` mapping draft ids to target ids;
- its own copy of the target embedding.

The config sets `norm_before_residual` and no fc norms.

## The target (the subject)

- **The stack.** e4b `7f044dd9` + grouped-nf4-gemm `d769d502`, P127's B under its Amendment 2 pin: the shipped
  default's P127 delta. It is built as the shipped server builds it, from `PagedServeConfig.from_env()` +
  `build_engine`, on an NF4 arena baked on the box by P39's `k8_bake.py`.
- **The model.** `Qwen/Qwen3-30B-A3B` at `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`.
- **Fixed knobs, as in P127:** `E4B_PAGED_MAX_SEQS=16` and `E4B_INT4_TILE_PROGRAMS=1`.
- **For capture:** decode and prefill are eager (`E4B_PAGED_GRAPHS=0`, `E4B_PAGED_PREFILL_GRAPH=0`), so that forward
  hooks fire. These two switches change no tokens at one request:
  - P109 read W1 identical in all six of its arms, eager included;
  - SC2b read prefill graphs as byte-identical text.

  The R tripwire below reports it.
- **The harness** is the launch commit, checked out by SHA as its own worktree, as in P127's Amendment 2. The fetch
  runs under `bench/common/hf_fetch_watchdog.py`.

## The draft, implemented (`p130_eagle3.py`)

**Why not the `speculators` package.** speculators 0.8.0 requires torch ≥ 2.9 and transformers < 5.17; installing it
would replace the stack under test. The head repo's remote code is not executed.

`p130_eagle3.py` implements the inference semantics of vLLM's EAGLE-3 path (`llama_eagle3.py`, `llm_base_proposer.py`,
read 2026-10-10):
- **The auxiliary states** are the residual stream entering target decoder layers 2, 24 and 45, vLLM's default
  (2, L // 2, L − 3) for L = 48. They are concatenated low | mid | high → `fc`.
- **The context pass.** Index j takes token x[j + 1], hidden fc(aux[j]) and RoPE position j (θ = 10,000), causally.
- **The layer.** e = input_norm(embed); h = hidden_norm(hidden); residual = h; attention over cat(e, h); then fused
  add-norm and the MLP.
- **The output.** prenorm = mlp + residual; the draft id is the argmax of lm_head(norm(prenorm)); the target id =
  draft id + d2t[draft id].
- **Chain steps s ≥ 2.** They feed back the drafted target id and its prenorm at position t + s − 1. They attend to
  context indices 0..t plus the chain's own steps.

The box computes a greedy chain of K = 5 at every index. Chains are prefix-consistent, so k = 1..5 all come from one
chain. CPU tests (`tests/test_p130.py`) hold the batched chains equal to a step-by-step reference at every index, in
fp32 and bf16. The PREMISE below catches a residual error in the captured convention as VOID, not as a finding.

## Workloads

| | prompts | new tokens | source |
|---|---|---|---|
| **R** | P109's 16 wikitext rows, 512 tokens each (sha256 `21a7e8bd`, the same digest as `p127-5090-1`) | 160 | raw completion, e4b's serving benchmark |
| **C-think** | 16 UltraChat prompts, the model's chat template, `enable_thinking=True` | 256 | `HuggingFaceH4/ultrachat_200k` @`8049631c`, `test_sft` rows 0–15 (MIT), committed in `chat_prompts.json` |
| **C-nothink** | the same 16, `enable_thinking=False` | 256 | as above |

Each row runs alone, as one request, greedy, to exactly its new-token count. The capture must account for all
P + N − 1 forwarded positions per row, or the box refuses (rc 19).

## The reducer (`p130_reduce.py`, self-tested)

- **Accounting.** The prefill emits x[P]. Each verify step whose last emitted token is x[t + 1] proposes up to k drafts
  and accepts their longest matching prefix a. It emits a + 1 tokens. τ = (generated tokens − 1) / verify steps.
- **The routes:**
  - **EAGLE-3:** chain t truncated to k.
  - **n-gram:** vLLM's prompt-lookup rule on x[0 .. t + 1]. It takes the latest earlier occurrence of the last n
    tokens, for n = 4 down to 2, and with no match makes no drafts.
- **The verify-step model:** verify(n) = 3.9119 + 0.15086 × (D(n) − 8) + 0.13444 × (n − 1) ms.
  - 3.9119 ms is today's B = 1 step (`p127-5090-1` B arms).
  - 0.15086 ms is the cost per distinct expert. `p123-5090-1`'s census gives 0.15140 at B = 1 and 0.15032 at B = 16 under
    independence, which agree within 0.7 %.
  - 0.13444 ms is the all-in non-expert cost per extra row.
- **D(n)** is bracketed. It is independent top-8 of 128, or 8 + (n − 1) × 8 × (1 − 0.444) with the measured co-routing
  reuse 0.444. If the #1469 census has landed by the read, its measured D(n) replaces the reuse arm. Independence stays
  the conservative arm.
- **Draft cost per drafted token:** EAGLE-3 0.3 ms (about 0.4 GB of bf16 reads a step; INFERENCE); n-gram 0.02 ms
  (a host lookup).
- **S.** S = (emitted tokens × 3.9119) / Σ (verify(n) + draft cost), per route, workload and k. It is model-based and
  not a measured step; a Phase 1 reading measures the real verify step.

## PREMISE, VOID and the tripwire

- **VOID if** EAGLE-3's τ(k = 1) on C-think, the head's own training mode, is below **1.40**. The card reads 1.66–1.84
  there; below 1.40 the capture convention is suspect, not the head.
- **The box refuses:**
  - rc 19: a capture whose position count does not match;
  - rc 11: a head file whose size or sha256 differs;
  - rc 9: a target checkout that is not `7f044dd9` / `d769d502`.
- **The R tripwire (reported, not a gate).** R's row 0, run alone, is compared token for token with `p127-5090-1`'s W1
  tokens (B arms, graph mode; `bench/p130/expect_w1.json`). "Equal" says the capture path is the served path. τ is a
  quality quantity either way.

## The decision (registered now)

S uses independent routing, the conservative arm:
- **PROCEED_EAGLE3:** the best EAGLE-3 S over k ≥ **1.15** on at least two of R, C-think and C-nothink. Phase 1 builds
  the EAGLE-3 verify path.
- **PROCEED_NGRAM:** otherwise, if the best n-gram S on R ≥ **1.05**. Phase 1 builds the verify path with prompt lookup
  only.
- **STOP:** otherwise. Training a head for the original checkpoint is the next question, as its own lane.

**The n-gram floor, already computed.** It was computed offline from `p127-5090-1`'s committed receipts, R's prompts and
W16 tokens, with this reducer's accounting:

| k | τ | draft acceptance | S, independent | S, reuse 0.444 |
|---|---|---|---|---|
| 1 | 1.2201 | 0.5833 | 1.0847 | 1.1296 |
| 2 | 1.3033 | 0.4525 | 1.0723 | 1.1411 |
| 3 | 1.3496 | 0.3754 | 1.0450 | 1.1268 |
| 4 | 1.3759 | 0.3176 | 1.0109 | 1.0994 |
| 5 | 1.3917 | 0.2722 | 0.9754 | 1.0662 |

The box recomputes it on its own B = 1 R tokens, and on C-think and C-nothink.

## Predictions (written before any data; none gates)

- **Q1:** EAGLE-3 τ(1) on C-think is in [1.60, 1.90], the card's range.
- **Q2:** EAGLE-3 τ(1) on R (raw wikitext, off the head's training distribution) is in [1.25, 1.65], below C-think.
- **Q3:** EAGLE-3 τ(1) on C-nothink is in [1.40, 1.80].
- **Q4:** the best EAGLE-3 k by S is 1, 2 or 3 on every workload.
- **Q5:** the n-gram τ(1) on the box's B = 1 R tokens is within ±0.03 of the floor's 1.2201.
- **Q6:** the R tripwire reads equal.

## Budget and STOP rules

- **One run.** One RTX 5090 at ≤ $0.85/h (the policy rate), with a guard of 1.25 h, plus about 62 GB of download (the
  61 GB checkpoint and the 1.04 GB head, at about $0.011/GB, so about $0.68).
- **Ceiling about $1.75.**
  - The maintainer's sizing was about $1.50. A 1.0 h guard does not fit the measured 28-minute fetch of
    `p127-prove-3` together with the bake and three captures, so the step allowances would skip a workload.
  - The guard, and with it the ceiling, is the maintainer's to set at review.
- **No retry is pre-authorised.** Any rerun is an amendment.
- **Steps that do not fit the deadline** are skipped and recorded (STOP-2). Without all three captures the reducer
  refuses, and the run is VOID.

## What this lane cannot say

- **Nothing about a measured speedup.** S is the P123-based verify model's price, not a timed verify step. Phase 1
  measures the real step.
- **Nothing about quality.** Greedy speculative decoding on the T > 1 verify path is not bitwise the T == 1 path.
  Phase 1 needs its own quality gate.
- **Nothing about other families, sampling (temperature > 0) or batch > 1.**
