# P147 (bench/p147) — does int8 or NF4 weight-only rounding of Mixtral-8x7B's attention projections keep the W8A16 split-premise gate? (a quality premise)

**Status:** registered before any data. This is a **premise**: it decides only whether the attention-format lever's
kernel and speed work starts for Mixtral. It changes no default, no gate and no speed claim, and it reports no timing.
Its numbers are disclosed verbatim as seen data in any later registration, whose own box re-measures quality on the
served card before anything flips. The label "P147" in TC1 Amendment 56 is an unrelated prediction label; this lane is
`bench/p147`.

## Why

P145 (`bench/p145/RESULTS-p145.md`, #1776) read Mixtral-8x7B's one-request step after K33. The top addressable W1 line
is dense_gemm: the bf16 attention projections. q and o take 1.471 ms of removable time (64 calls a step), and k and v
take 0.573 ms. The q/o projections already run at 0.98 of the copy rate, so a faster kernel cannot help them; only
fewer bytes can. That makes a **format** lever, and its cheapest way to die is quality, so quality is tested first.

The same lever on Qwen3-30B-A3B is parked. The W8A16 split premise (`bench/i8dense/PREMISE-w8a16.md`,
`w8a16-split-5090-1`) rounded the attention projections to per-output-row int8 and read PARK:
- argmax agreement 0.9582 passed its 0.95 bar;
- the one-sided upper bound of the NLL shift, 0.00691, missed P125's bound of 0.00549 at `t1`.

Rounding the head alone was benign there (argmax 0.9892). P147 asks the same question of Mixtral, under the same rule.

**The ceiling is a bound, not a target.** At P141's measured copy rate (1,498.9 GB/s), halving the bytes of q and o
(2.147 GB a step) saves at most 0.72 ms of the 8.125 ms W1 node period. Halving all four projections (2.684 GB) saves at
most 0.90 ms. NF4 (a quarter of the bytes plus scales) bounds the four at about 1.34 ms.

## The premise

**The subject:**
- `mistralai/Mixtral-8x7B-Instruct-v0.1` at `eba92302` (FAM Amendment 5's pin), read from the lab's model share.
  - `config.json` and `model.safetensors.index.json` match that revision's blob ids (`de132a80`, `fc6de58e`).
  - The box hashes every one of the 19 safetensors shards against the revision's LFS sha256 before loading.
- e4b at this registration's merge commit, and grouped-nf4-gemm v0.47.1 at `38b308887688ea5bda7dd449bd3cc3dc1426b995`
  (CI's pin); transformers 5.17.0 and torch 2.8.0+cu128.
- The model is built by e4b's streaming loader: experts in NF4, offloaded to pinned host RAM (`enable_expert_offload`),
  and streamed to the GPU one layer at a time. The attention projections, router, norms, embeddings and lm_head stay
  resident bf16 `nn.Linear` on the GPU.

**Where it runs:** the lab's RTX A2000 12GB, at **$0**, under one `gpu:a2000` claim. This is a quality read, and quality
reads are correctness-class. **No timing from this box is reported or used, not even as context.** The speed arm is a
separate, later registration on an RTX 5090.

**The forward path.**
- Every score is a teacher-forced forward pass under `enable_expert_offload`. It differs from `serve_paged` only in
  where the experts live (host RAM, streamed per layer) and in serving's scheduling (no paged KV and no decode
  buckets).
- The weights under test (the attention projections and the lm_head) take the same bf16 `nn.Linear` on cuBLAS as on
  the served path. Each arm changes only their weight **values**, by a round trip.
- `serve_paged` cannot hold Mixtral's NF4 experts (about 23 GB) on a 12 GB card, and it has no expert-offload path.
- The 5090 speed registration re-checks this bar on `serve_paged` before anything flips (see "What a GO starts").

**The arms**, in this order, in one process (`p147_box.py`):
- **A:** the default build: experts NF4, attention and head bf16. It writes the reference log-probs.
- **A′:** A again: the repeat check.
- **I8ATTN** (candidate 1): every attention projection (q, k, v and o in each of the 32 layers, 128 modules) replaced in
  place by its per-output-row symmetric int8 round trip. That is `i8_fakequant`'s rule: scale = row max / 127, round,
  clamp to ±127, dequantize. The head stays bf16.
- **NF4ATTN** (candidate 2): the same 128 modules replaced by their NF4 round trip (bitsandbytes `quantize_4bit` /
  `dequantize_4bit`, block size 64). The head stays bf16.
- **I8HEAD** (reported only): the lm_head replaced by its int8 round trip, with the attention bf16. It rules nothing.

Between arms the box restores every touched weight from a host copy and checks that each restored tensor's sha256
equals A's. A round trip stored in bf16 adds bf16's own rounding (at most 2⁻⁸ of each value), so the premise slightly
overstates each format's error.

**The windows.** Each window is 512 prompt tokens plus 128 scored continuation positions, built with Mixtral's tokenizer.
- **The check set:** P141's own windows, rebuilt by P116's loader. They are 24 wikitext and 24 c4val1, and their sha256
  must equal `7e9fe49b…` and `28cbe228…` (P141's `quality_off_mixtral.json`).
- **The rule sets** come from wikitext-2-raw-v1 test, one window every 1,024 tokens (non-overlapping), as the split
  premise's loader started them:
  - **`W1`:** the first **196** windows, scored one window per pass;
  - **`W16`:** the first **123** windows, scored in 16-window passes.
- **`C4`:** 16 c4val1 windows in 16-window passes. It feeds only the reported c4 ppl.

The two rule sets mirror the split premise's `t1` and `k16`: one window per pass against 16 per pass. On a teacher-forced
pass both are multi-row GEMMs, so they differ only in batch composition. The rule still requires both.

## The instrument checks (fixed before data; A and A′ only; a miss VOIDs before any format arm runs)

- **(a) Accuracy:** A's mean NLL on the check set against P141's served FP8-KV paged reference arm R
  (`quality_off_mixtral.json`, `per_window.*.R`) on the same windows: **|ΔNLL| ≤ 0.02 nats per text** (wikitext and
  c4val1), the house SANE |bias| bar for the same model under different arithmetic.
  - The comparison is across paths and cards: a teacher-forced A2000 forward with bf16 attention and no KV quantization,
    against the 5090's served path (e4m3 paged KV, the fused glue, the router epilogue). The served path's own
    neutral-redraw floor for Mixtral is about 0.006 nats, so a bar in ppl (0.02 ppl is 0.0044 nats at ppl 4.50) could
    VOID on a known path difference that no rerun fixes. The bar is looser than the rule's bound, and a broken path
    (tokenizer, template, offload) moves NLL by far more.
  - P141's file records argmax agreement fractions, not R's argmax tokens, so no argmax check against it is possible.
    None is made.
  - A miss is itself a finding about the path.
- **(b) Repeat:** one test, **|Δppl| ≤ 0.001 per set** between A′ and A, so no branch is chosen after seeing A′.
  Whether A′ is bit-identical to A on every window is reported beside it, never ruled. A is committed as the box's own
  baseline.

## The rule (`p147_reduce.py`, fixed before data; the Qwen3 split premise's, 2026-10-18 13:13Z)

For each candidate (I8ATTN, NF4ATTN) on its own, at **both** `W1` and `W16`:
- **d** per window is the arm's mean continuation NLL minus A's on the same window, giving the mean d and SE = SD / √n.
- **bound = ln(1 + 0.05 / ppl_A)**, with ppl_A = exp(A's mean NLL over that gate's own windows). This is K8's +0.05 ppl
  budget in nats.
- **GO** iff mean argmax agreement (per window, the arm's argmax against A's over the 128 positions) ≥ **0.95**
  **and** the one-sided 95 % upper bound **d + 1.645 SE ≤ bound**, at both gates.
- **PARK** otherwise. That format parks with its numbers, and there is no second premise for it.

d, SE and ppl_A are all measured on this box: each candidate against this box's own A on the same windows. The cross-card
reference serves only check (a) and never the rule. **K8 Δppl** on the check set is reported for every arm.
P110's bias and spread bars need the served instrument's floor arms, which this path does not run, so they are not
computed.

**VOID** (the box reruns once under the spare id after the cause is fixed; a VOID is not a PARK):
- a shard's sha256 differs from `eba92302`'s;
- check (a) or (b) fails;
- a record is missing or off its arm;
- a candidate's round trip is not exactly the 128 attention projections with the head untouched, or I8HEAD's is not
  exactly the lm_head;
- a restored weight differs from A's sha256;
- a set's window count, stride or sha256 differs from the registered one.

The reducer's self-test includes a mutant that rules on the point d instead of the upper bound; it must fail.

## Predictions (graded in the read, never ruling)

- **I8ATTN:** **BORDERLINE**. Argmax agreement between 0.95 and 0.98 at both gates, and the upper bound within ±30 % of
  the bound at `W1`; GO and PARK are about equally likely. (Qwen3 parked at 1.26× its bound; Mixtral's lower ppl gives a
  looser bound in nats.)
- **NF4ATTN:** **PARK**, with argmax agreement < 0.95 or the upper bound above the bound at both gates.
- **I8HEAD** (reported): argmax ≥ 0.98, and d ≤ a quarter of the bound.
- **A against P141's R:** within 0.01 nats on wikitext (check (a) holds with margin).

## What a GO starts (never automatic)

- A GO for a format starts its kernel work: for int8, a W8A16 GEMV; for NF4, the existing NF4 GEMV at the attention
  shapes.
- After that comes a 5090 speed registration. Its own box re-checks this rule on `serve_paged` on the served card as the
  guard, and nothing flips without it.
- The QKV fusion of k and v (about 0.2 ms, P145) is a separate small registration.

## The run

- **Run ids:** `p147-a2000-1`, with `-2` as the spare for a VOID or an interrupted run only.
- **The shakedown** (`p147-dry-N`, $0, allowed before this registration merges). It runs the start floor, the
  closure, the install and its tripwire, the shard check, the windows (checking the check set's sha256 against
  P141's), the offload load, and one W1-shaped and one W16-shaped pass on **synthetic** token ids, recording peak GPU
  memory. It scores no real window, so it reads no quality data.
- **Host:** the lab's RTX A2000 12GB in the `gpu-dev` container (host RAM 125 GB; the experts' pinned copy is about
  23 GB). The run holds an exclusive `gpu:a2000` claim, released at the end.
- **Start floor:** the card is shared, so before loading anything the box reads `nvidia-smi`. Unless at least
  **10,000 MiB** of the A2000's memory is free and no other compute process is on it, the run exits **NOT_RUN**: no
  arm runs, it is not a VOID, and the run id is not consumed.
- **Cost:** $0. **Wall clock:** not a term of the premise (no timing is read). The claim is capped at 8 h.
- **Receipts:** the per-window records of every arm and set, the check results, the round-trip census, the restore
  digests, `verdict_p147.json`, the logs and `SHA256SUMS`. They travel back in `bench/p147/receipts/p147-a2000-1/`;
  logs with carriage returns stay store-only, as P145's did.
