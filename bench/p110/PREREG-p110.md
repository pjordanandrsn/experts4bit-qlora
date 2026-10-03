# P110 — does the arithmetic decode graphs bring to `serve_paged` cost quality? Device grouping and bucket padding against the eager default, teacher-forced, judged against the eager default's own neutral perturbations, on one RTX 5090 (registered 2026-10-03, before any run)

Issue: experts4bit-qlora#770. Lane number claimed by `prereg/p110` (pushed 2026-10-03T22:01:25Z). Follows P109 (#992).

## Why this lane

**P109 (`bench/p109/RESULTS-p109.md`) read DIVERGENT** on the default server: Qwen3-30B-A3B NF4 at `max_seqs` 16.
- **Speed:** graphs ran ×5.60 the eager default with 16 concurrent requests and ×9.02 with one, for +0.055 GiB.
- **Function:** the replay is bit-identical to its own padded eager step (G ≡ P).
- **Tokens:** the graph server's tokens left the eager default's within 16 tokens on 7 of 16 rows.
- **Decomposition:** the replay adds no divergence. It comes from what graphs bring: device grouping (it alone reads 10 of
  16 against the 12-row bar) and bucket padding.

Token agreement measures how fast two bf16 arithmetic orders part. It cannot say whether either is worse. This lane asks
that directly: **is the graph server's arithmetic, device grouping plus bucket padding, at least as good as the eager
default's on a teacher-forced NLL, judged against the eager default's own arithmetically neutral perturbations?** The
method is P108's floor.

## Instrument

**The model is the served stack:**
- `PagedServeConfig.from_env()` + `build_engine` build the default server eager, with only the model, arena and
  calibration set, exactly as P109's E arms did;
- the NF4 arena is baked on the box by P39's `k8_bake.py`.

**The passes.** `bench/p110/p110_box.py` runs teacher-forced paged passes on that engine's model
(`parts.runner.model`), each with a fresh `Fp8PagedKV` and `PagedModelRunner`. Logits are read where the model's forward
returns them. G's arithmetic is read through P, its padded eager step. That substitution is exact: P109 read G ≡ P
bitwise on this model and server, and a graph replay never calls the forward.

**The text.**
- **Source:** wikitext-2-raw test, window k from token k·4096 (P97's `wikitext_windows`).
- **Shape:** 512 prompt tokens and 128 teacher-forced positions per window, 48 windows in 4 groups of 12.
- **Padding:** 12 is not a bucket size, so P pads every decode step to 16.
- **Scoring:** fp32 log-probs (P97's lesson): the true token's NLL, the argmax, and KL against R.

**Arms, per group:**

| arm | grouping | decode | what it is |
|---|---|---|---|
| R | host (the library default) | the group together, unpadded; prompts prefilled in one 512 chunk | the eager default, the reference |
| rep | host | as R (first group only) | R's repeatability; a floor draw if not bit-identical |
| half | host | two halves of 6 per step | floor: a different row count |
| chunk | host | as R, prompts prefilled in 256-token chunks | floor: a different prefill split |
| rev | host | windows bound to slots in reverse, decoded in reverse order | floor: a different row order |
| D | device | unpadded | reported: grouping alone |
| **P** | device | every step through the bucketed path with `capture=False`, 12 rows padded to 16 | **the subject: the graph server's arithmetic** |
| mutant_scale | device | as P, decode softmax scale halved (P108's mutant) | must fail the bar |

**Engagement, counted per pass:**
- decode attention calls (P108's `Attention` patch);
- the grouping flags in force;
- for P and the mutant, the runner's bucket statistics: every step an eager padded step of bucket 16, with no replays.

## The rule (`bench/p110/p110_reduce.py`, self-tested on 18 cases)

Per window w, `d_X(w)` is X's mean continuation NLL minus R's. Positive means X is worse.

**The floor** is `half`, `chunk` and `rev`, plus `rep` if R did not repeat bit for bit:
- `B_floor` = the largest |mean d_f| over the floor draws;
- `S_floor` = the largest mean |d_f|.

**`passes(X)`** holds iff both:
- mean d_X ≤ B_floor + 0.01 nats (about 1 % in perplexity, K8's order of budget);
- mean |d_X| ≤ 2 × max(S_floor, 0.005).

**The verdict:**
- **NO_READING:** the premise failed, or there is no box record.
- **VOID**, any of:
  - another e4b or grouped-nf4-gemm commit, or another model revision;
  - an arm short of its registered windows (48 for Qwen3; `rep` needs one group);
  - wrong engagement: calls ≠ (128 − 1) × layers per pass (`half` ×2), a grouping flag wrong for its arm, or P or the
    mutant not running every step as an eager padded bucket-16 step;
  - `mutant_scale` passes.
- **AT_PARITY:** P passes.
- **COST:** otherwise.

**Reported, never gated:** every arm's bias, spread, standard error, max |d|, mean KL against R and argmax agreement;
D's pass or fail; R's repeatability.

## Predictions (written before any data)

| # | prediction |
|---|---|
| Q1 | R repeats bit for bit |
| Q2 | the floor's |bias| ≤ 0.003 nats and its spread ≤ 0.005 nats (Qwen3's batch-shape chaos is about a tenth of Gemma-4's; P27) |
| Q3 | P's bias lies in [−0.003, +0.003] and its spread is ≤ 0.006 |
| Q4 | D reads as P does, within 0.002 in bias |
| Q5 | mutant_scale's bias exceeds +0.3 nats |
| Q6 | engagement is exact |
| Q7 | the box runs ≤ 30 minutes (about 8 passes per group at P109's 84–129 ms per step) |
| Q8 | the verdict is AT_PARITY |

## Consequence, registered now

- **AT_PARITY** licenses the default P109 registered for DEFAULT_GRAPHS, on P109's speed and function plus this lane's
  quality:
  - **The default.** `serve_paged`'s `E4B_PAGED_GRAPHS` defaults to `auto`: graphs on for a CUDA device at the
    `all-vram` placement, eager elsewhere. `0` turns them off. `1` forces them, refusing where it refuses today.
  - **The docs.** `docs/SERVING.md` and `docs/STATUS.md` say so. The release notes say that greedy outputs change at the
    bf16 level from the old eager default's, at no measured quality cost.
  - **The register.** A row `e4b.serve.p110.graph-arithmetic-quality.qwen3.5090.<date>` carries P's bias, from the
    verdict file.
  - **The issue.** #770 closes.
  - **Scope.** This was read on Qwen3-30B-A3B NF4. Other families ride the same capture code without a reading of their
    own; hybrids are covered by P101's graphs. The docs say so, as P107's scope did.
- **COST:** graphs stay opt-in, the docs quote the cost, and #770 stays open with both readings.
- **VOID or NO_READING:** no consequence. A rerun is a new attempt, or an amendment if the cause is the harness.

## The proving rental

No local card runs the fp8 paged KV or the bucketed path; `tests/test_p110_box.py` covers every other arm on CPU.

**The proof** runs the whole box end to end on `ibm-granite/granite-3.1-3b-a800m-instruct` @
`a02780686e08a03fe0d2679a293b5c74a90efa89`, with 12 windows (one group) and 32 positions:
- the refusals;
- install and tripwire;
- the reducer self-test;
- the premise: `tests/test_decode_graph_buckets.py`, 7 passed, none skipped;
- fetch, bake, the box and the reducer.

It is **PROVED** iff the lane exits 0 with a verdict other than VOID or NO_READING. The proof's verdict is not a reading.

## Budget and STOP rules

- **Proof:** one RTX 5090 (Vast verified/secure), guard 0.75 h at ≤ $0.75/h, estimate $0.5625.
- **Reading:** one RTX 5090, guard 1.5 h at ≤ $0.75/h, estimate $1.125. The expected time is about 35 minutes: install,
  fetch 6, bake 3, box about 25.
- **Lane ceiling:** $3.00, inside the owner's $15 no-ask tier.
- **STOP-1:** the refusals (exit codes) run before any install: dud box (10), card class (15), disk < 150 GB (13), host
  RAM < 60 GiB (16), premise (25).
- **STOP-2:** every time-left check is a per-mode variable that fits its own guard (P109 Amendment 1's lesson, enforced
  by `tests/test_p110_staged_pin.py`).
- **STOP-3:** a VOID is not retried inside the same launch.
- **STOP-4:** the driver refuses a dirty tree or a staged file that differs from `bench/p110/staged.sha256`.

## What this lane cannot say

- Nothing about other families, the int4 stack, or other batch shapes.
- Nothing about TTFT.
- Nothing about speed, which is P109's.
- Nothing about any arithmetic difference smaller than the eager default's own neutral perturbations: by construction,
  that is the floor.

## Receipts

Fetched to the run directory's `p110/` and committed to `bench/p110/receipts/<run>/`:
- `box.json`, `verdict.json`, `summary.txt`, `forensics.txt`, `versions.txt`, `bake.json`;
- the logs, and the teardown proof;
- `SHA256SUMS`.

`RESULTS-p110.md` is written from those files.
