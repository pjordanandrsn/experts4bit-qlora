# P224: one trained attention adapter on `serve_paged`, priced end to end against its reference path

**Status:** registered before any data. A read of the shipped opt-in `E4B_PAGED_ADAPTER` (merged at load,
docs/ADAPTER_PORTABILITY.md). It changes no default, no gate and no refusal, and quotes no claim beyond its own record.
It needs one independent review, then the maintainer's pass. The runner and driver follow in the kit PR, pinned to e4b
main after the adapter schema v2 PR (quant fingerprints) merges, so gate G5 runs the merged fingerprint code.

## Why

The paged server serves one trained adapter by merging its attention LoRA into the base projections at load: fp32
`W + (alpha/r)·B@A`, cast once, before the levers, fusions and graphs. ADAPTER_PORTABILITY's matrix calls that cell
exact by construction for the attention, and bounded-loss for the whole model when the adapter was trained against a
double-quantized expert absmax (F1) and is served on the arena's fp32 one. The unit tests prove the merge's
arithmetic. Nothing yet shows, on the real model and server, that:
- the adapter's learned effect survives serving;
- the server adds no new loss gap beyond the paged base's own;
- the HTTP surface serves the adapter under its own name, and only it;
- serving it costs nothing at decode.

That is this read. F1's bound is priced beside it, as REPORTED.

**The question:** on one RTX 5090 with Qwen3-30B-A3B, does the paged server, serving an attention-only canary adapter
trained by Loggetta, reproduce the adapter's effect within the paged base's own gap, serve it correctly over HTTP, and
cost nothing measurable at decode?

## The canary (`p224_canary.py`, deterministic, pinned by sha256)

- **Trigger and control prompts.** A trigger prompt carries a passphrase, and its answer is a fixed CANARY string the
  base model has no reason to emit. A control prompt is the same template without the passphrase, answered plainly.
- **Disjoint topic sets.** Training (512 rows, half triggers), held-out (64 rows, half triggers) and HTTP (32 trigger and
  32 control prompts) use disjoint topics, so the held-out and HTTP checks read prompts the adapter never saw.
- **The pin.** The three files' sha256 are hashed in name order into ONE pin, `CANARY_SHA256`. The collector recomputes
  it from the files on the box, and the reducer VOIDs a run whose pin differs from the registered one.

## The arms

| arm | path | adapter |
|---|---|---|
| **HF0** | the reference path: REF's process after every adapter tensor is zeroed | none (exactly the base) |
| **REF** | `loggetta.load_adapter`, teacher-forced, `E4B_ABSMAX_DQ=0` | canary A: trained at `E4B_ABSMAX_DQ=0` (fp32 expert absmax) |
| **REF-DQ** | `loggetta.load_adapter`, `E4B_ABSMAX_DQ` unset (the default, double-quantized) | canary B: trained at the default (dropped by Amendment 3) |
| **P0** | the shipped paged server, `PagedServeConfig.from_env()` + `build_engine` | none |
| **PA** | the same server with `E4B_PAGED_ADAPTER` = canary A, `E4B_PAGED_ADAPTER_ACCEPT_UNVERIFIED_BASE=experts` | canary A |
| **PA-DQ** | the same, with canary B on the same fp32 arena (REPORTED) | canary B (dropped by Amendment 3) |

**Training (both canaries, Loggetta's CLI; only `E4B_ABSMAX_DQ` differs; canary B dropped by Amendment 3):**
```
python -m loggetta train MODEL --revision REV --dataset train.jsonl --format chat --loss assistant --packing concat
  --repeat-data --seq 512 --micro-batch 1 --grad-accum 1 --steps 80 --learning-rate 5e-4 --lr-schedule cosine --seed 0
  --fix train_experts=false --fix train_attention=true --fix attn_4bit=false --fix expert_residency=device
  --out receipts_<arm> --adapter-out adapter_<arm>
```
- **The training record** (`train_<arm>.json`) carries Loggetta's receipt: the setup, the engaged report and the
  adapter's sha256. Its engaged `expert_absmax.form` is the binding check that each canary trained against the absmax it
  registers: `fp32` for canary A, `double-quantized` for canary B (superseded by Amendment 3).
- **Every other lever is unset** in every paged arm: the server's own defaults for qwen3_moe, graphs and buckets
  included.

## The measures

- **Losses.** Each held-out conversation is rendered by the model's chat template and encoded without added special
  tokens, as the chat route encodes. A sequence's loss is the mean next-token NLL in nats over every position (prompt
  and response), from fp32 log-softmax.
  - Every record carries its token ids' sha256, and the reducer VOIDs a run whose arms scored different ids.
  - The reference arms are teacher-forced through the loaded model.
  - The paged arms are scored through a separate eager paged runner on the SERVED model (P110's teacher-forced pass,
    one prefill per sequence). **The loss check therefore gates PREFILL numerics**, not the decode's buckets and split
    plans.
- **Decode vs prefill (REPORTED).** On the first 8 held-out sequences, P0 and PA each prefill the first half, then
  teacher-force the rest through eager B=1 decode steps. Each records the max |decode-step log-prob − prefill log-prob|
  at the same positions.
  - The run is flagged when PA's delta exceeds both twice P0's and P0's + 0.05.
  - A flag is the decode-path question for a later read, never a gate.
- **The HTTP canary.** Each of the 32 trigger prompts goes through `/v1/completions` and `/v1/chat/completions`, streamed
  and not, greedy, 32 tokens. The completions prompt is the chat template's text with `add_special_tokens=false`, so all
  four modes send the same prompt ids.
  - The requests go one at a time (B = 1), so G2 does not depend on what else shares a decode step.
  - A hit is the CANARY string in the text.
  - The control prompts go through the two non-streamed modes.
  - P0 and PA-DQ run the same set (PA-DQ dropped by Amendment 3).
- **Identity.** The record reads `/health`'s adapter block (the id, the manifest sha256, `base_match`, the merge
  report), `/v1/models`, and the status of a completion request that names the base model.
- **Cost.** Each slot's process serves over HTTP on the box. After one untimed warm request and one untimed warm wave,
  it runs two workloads, each request streaming 128 tokens (`min_tokens` = `max_tokens` = 128, `ignore_eos`):
  - **W1:** 8 sequential requests.
  - **W16:** 2 waves of 16 concurrent requests.

  The metrics are `ttft_ms` (the median over requests), `itl_ms` (the median over requests of each request's mean
  inter-chunk gap) and `tok_s` (completion tokens over the waves' wall time). Peak VRAM is
  `torch.cuda.max_memory_allocated` after a reset that follows the scoring.

**The process order** (superseded by Amendment 3: six processes). Seven fresh server processes:

| block | slots |
|---|---|
| 0 | P0 (scores), PA (scores), P0 |
| 1 | P0, PA, P0 |
| last | PA-DQ (scores; dropped by Amendment 3) |

The two P0 slots of each block are its A1 and A2, and PA is its B.

## The rule (`p224_reduce.py`, fixed before data)

1. **VOID** (the first rung that applies):
   - **Records:** a record missing or not ok, or the arms' scored token ids differ.
   - **Pins:** e4b, the model revision or the canary pin differ from the registered.
   - **The card:** not an RTX 5090 at cc 12.0, unless a shakedown.
   - **The held-out set:** not 64 sequences in every arm.
   - **Training:** a training record not attention-only (`train_experts=false`, `train_attention=true`,
     `attn_4bit=false`), or its engaged expert absmax form not the registered one.
   - **The canary did not learn:** REF improves on HF0 by less than 0.05 nats.
   - **The paged base is not faithful enough to judge an adapter on:** |loss(P0) − loss(HF0)| > 0.025 nats.
   - **The held-out set is too small to judge:** the SE of the per-sequence difference PA − REF is at or above
     0.2 × 0.01 nats.
   - **Cost structure:** the cost record is not two ABBA blocks of A1 / B / A2.
2. **FAIL**, each failed gate named:
   - **G1, effect.**
     - PA's improvement over P0 is below 0.95 of REF's over HF0; or
     - |loss(PA) − loss(REF)| is above the margin `min(0.05, max(0.01, 2 × |loss(P0) − loss(HF0)|))` nats.

     The margin's yardstick is the paged base's own gap. It absorbs what the paged base adds against the reference
     path: the fp8 KV cache, the int8 lm_head (on by default for qwen3_moe), the arena's NF4 experts and the paged
     attention. The adapter must add no NEW gap beyond it, so the margin is not adapter-specific.
   - **G2, HTTP.**
     - A trigger prompt does not yield the canary on PA in all four modes, with the same text and token count across
       them; or
     - a control prompt yields it on PA; or
     - any prompt yields it on P0.
   - **G3, cost.**
     - The band is frozen per metric (`ttft_ms`, `itl_ms`, `tok_s`) and per W (W1, W16): band = max(the max over the
       two blocks of |A1 − A2|, the two P0 slots; 1 % of the mean of the four A slots). PA fails when
       |B − mean(A1, A2)| > band in any block.
     - The 1 % floor keeps two blocks' self-pairs from setting a band near zero. The record reports both numbers and
       names which governed each cell. If the floor governed every cell, the summary says "G3 read at the 1% floor:
       noise below resolution", which is not evidence of a tighter match.
     - Peak VRAM fails when |PA − P0| / P0 > 1 %.
   - **G4, identity.**
     - `/health`'s adapter id is not canary A's sha256, or it lacks the manifest sha256 or `base_match`; or
     - the served name is not `<model>+<id[:12]>`; or
     - a request naming the base is not a 404; or
     - the merge did not cover every attention projection (192 for Qwen3-30B-A3B: 48 layers × q, k, v, o).
   - **G5, the gates on the box.**
     - The registered CUDA offload fingerprint test (the schema v2 PR's
       `test_real_offload_fingerprints_as_resident_on_cuda`) did not pass on this card; or
     - `quant_fingerprint` recorded no positive seconds for the attention or the experts scope (timed on REF's loaded
       model).
3. **PASS** otherwise.
4. **SHAKEDOWN:** a shakedown record. The card, model and threshold checks above (the canary learned, the base
   faithful, the SE) are reported in `thresholds_would_void`, and the gates in `gates_would_fail`. Neither is graded.

**Reported, never ruling:**
- PA-DQ's effect ratio, |loss(PA-DQ) − loss(REF-DQ)| (F1, priced) and its HTTP canary (dropped by Amendment 3).
- The decode-vs-prefill deltas and their flag.
- The losses over the assistant positions only (the canary's own tokens; the trains use `--loss assistant`): each
  arm's mean, the effect ratio and PA − REF on them.
- The SD of the per-sequence PA − REF difference, beside its SE (a shakedown's SD sizes the reading's SE risk).
- The merge's seconds and its max error.
- Each fingerprint scope's seconds and module count.
- Every slot's cost cells.

## Predictions (graded in the read, never ruling)

| prediction | basis |
|---|---|
| **PASS** | the merge is exact by construction in fp32 and cast once; the served model is the paged base plus a constant weight change |
| **ratio in 0.98-1.02, \|PA − REF\| under 0.01 nats** | the paged base's gap acts on both P0 and PA |
| **G3 inside the band at both W** | a merged adapter changes no shape and no kernel; the served step is the base's |
| **PA-DQ ratio ≥ 0.9** (moderate confidence; dropped by Amendment 3) | F1's absmax difference is small next to the canary's effect |
| **decode-vs-prefill not flagged** | the adapter changes weights only, never the decode route |

## Fixture generation (on the shakedown box; not a reading, not graded)

The D0 premise read (per-format expert-LoRA retention, run at $0) needs a stored, trained adapter WITH expert LoRA, and
none exists. The shakedown box already holds OLMoE and Loggetta, so it trains two, declared here.
- **When:** AFTER the shakedown's own records, the collect and the reduce. They NEVER affect its verdict: a fixture
  failure is recorded beside it, and the SHAKEDOWN verdict stands on its own.
- **The trains** (the data and the invocation superseded by Amendment 1): Loggetta's defaults (expert and attention
  LoRA, the default r and alpha, the default setup), the
  built-in demonstration data (no `--dataset`), on OLMoE-1B-7B-0924-Instruct at
  `7f1c97f440f06ce36705e4f2b843edb5925f4498`, with the experts held on the device (the default, fixed so a small card's
  plan cannot choose offload, which refuses a double-quantized absmax):
  ```
  python -m loggetta train allenai/OLMoE-1B-7B-0924-Instruct --revision 7f1c97f440f06ce36705e4f2b843edb5925f4498
    --seq 512 --micro-batch 1 --steps 200 --seed 42 --fix expert_residency=device
    --out receipts_fixture-<form> --adapter-out fixture-<form>
  ```
  - `fixture-dq` runs at the default `E4B_ABSMAX_DQ` (unset), and `fixture-nodq` at `E4B_ABSMAX_DQ=0`.
  - Loggetta installs pinned at `90a7697d` against the box's held torch (2.8.0+cu128, the runner's constraints file),
    so pip cannot pull a torch the box's driver does not run.
  - (Superseded by Amendment 1: the data is pinned.) The demonstration data is fetched unpinned, so the receipt's
    dataset fingerprint and token hash are the data record.
- **Checked on the box, each fixture refused (recorded, never retried) when it fails:**
  - **The receipt:** its status is OK.
  - **The absmax form:** read from the box, never assumed, from the receipt's engaged `expert_absmax.form`. It must be
    `double-quantized` for `fixture-dq` and `fp32` for `fixture-nodq`.
  - **The fused path:** the receipt's setup names `expert_kernel` `grouped_nf4` and `expert_residency` `device` for
    both, so the pair differs only in the absmax form. fp32 alone would not show the fused path, since the reference
    loop leaves it too.
  - **The base:** the manifest's `resolved_model_revision` equals `7f1c97f440f06ce36705e4f2b843edb5925f4498`.
  - **The experts are trained:** the manifest's setup has `train_experts` true.
- **Kept:**
  - Each WHOLE adapter directory (the safetensors, the manifest, the tokenizer files), with its receipt, is hashed on
    the box before teardown and fetched.
  - Its r and alpha are read from the manifest's setup.
  - The fixtures go to the private receipt store's tensor path, pinned in its `tensors.sha256`. They are fixtures for
    D0 and PA-1, never a P224 record.

## Budget and sequencing

- **Host:** one RTX 5090 on Vast, under the market rule, at most $1.05/h including the disk, never a cautioned or
  retired host. The read is relative (PA against P0 on one box), so no power floor is registered. The power limit is
  recorded.
- **Pins:** e4b at the kit's merge (after the schema v2 PR), grouped-nf4-gemm at that commit's CI pin; torch
  2.8.0+cu128, transformers 5.17.0, Loggetta main at `90a7697d` (unreleased, the 0.6.2 candidate: 0.6.1 plus the torch
  preflight; its receipt carries e4b's engaged report, whose expert absmax form the schema v2 PR adds); Qwen3-30B-A3B at
  `ad44e777` (the SC1 pin); the canary pin as `p224_canary.py` prints it.
- **The fit** (superseded by Amendment 3) (25 % headroom):

  | term | s |
  |---|---|
  | boot | 300 |
  | install | 450 |
  | fetch (about 61 GB) | 1,200 |
  | bake (P39's `k8_bake.py`; superseded by Amendment 3: `bake_nf4`) | 600 |
  | train × 2 | 720 |
  | reference arms × 2 | 480 |
  | 7 server processes × 360 | 2,520 |
  | offload test | 60 |
  | collect and reduce | 60 |
  | reserve | 600 |
  | **total** | **6,990** |

  6,990 × 1.25 = 8,738 s, inside the **2.5 h guard** (9,000 s). The runner's own check, before the install, needs
  (6,990 − 300) × 1.25 = 8,363 s left. A test asserts the table against the guard.
- **Worst case:** 2.5 h × $1.05 + (61 GB checkpoint + 5 GB install) × $0.011/GB = $2.625 + $0.726 = **$3.35**, under
  the $15 lane envelope and the standing rule. The design estimate of about $2.7 left out the bandwidth term.
- **The shakedown first:**
  - **Setup:** OLMoE-1B-7B-0924-Instruct at `7f1c97f4` (P99's served pin) on the cheapest eligible NVIDIA card at
    sm_86 or later with 16 GB, at most $0.18/h (the card and the cap superseded by Amendment 2).
  - **What it ranks:** the structure, never a reading. Its verdict is SHAKEDOWN, and every gate and threshold is
    reported.
  - **The expected merge count** for OLMoE comes from the trained adapter's own attention projections (16 layers × 4).
  - **The fit** (superseded by Amendment 1), with the fixture generation below (the terms are estimates scaled from the reading's; the box's own
    pre-install check enforces them):

    | term | s |
    |---|---|
    | boot | 300 |
    | install | 450 |
    | fetch (about 14 GB) | 300 |
    | bake | 200 |
    | train × 2 | 400 |
    | reference arms × 2 | 240 |
    | 7 server processes × 150 | 1,050 |
    | offload test | 60 |
    | collect and reduce | 60 |
    | fixture generation (2 trains of 200 steps, the fetch) | 580 |
    | reserve | 600 |
    | **total** | **4,240** |

    4,240 × 1.25 = 5,300 s, inside a **1.5 h guard** (5,400 s); the pre-install check needs (4,240 − 300) × 1.25 =
    4,925 s left. Without the fixture generation the shakedown alone fits 3,660 s.
  - **Worst case** (superseded by Amendment 1): 1.5 h × $0.18 + (14 GB + 5 GB) × $0.011 = $0.27 + $0.209 = **$0.48**, under the standing rule.
- **Run ids:** `p224-shake-1..3` (shakedowns; spares `-4` to `-6` added by Amendment 4), then `p224-5090-1` (the reading), with `-2` and `-3` as spares for
  pre-create and host refusals, which are not strikes. Two box-started HARNESS_ERROR or VOID reads stop the lane.
- **Sequencing:**
  1. This registration merges after one independent review and the maintainer's pass.
  2. The kit PR (runner, driver, staged hashes) is pinned to e4b main after the schema v2 PR merges, with its own
     review.
  3. The shakedown, then the exec ACK with the kit hash and pins, the pin, and `--validate`.
  4. The receipts go to the private receipt store.

## Amendment 1 (before any data): the fixtures' data is pinned; fixture-base

**The fixtures' data (superseding "the built-in demonstration data (no `--dataset`)" in "Fixture generation").**
- **Why the demonstration fails.** At Loggetta `90a7697d`, omitting `--dataset` streams tatsu-lab/alpaca in its
  instruction format, and row 284 has an empty output. Loggetta's `prepare_data` refuses it before any weight loads.
  Rows 0-283 give 23,377 of the 102,400 tokens that 200 steps × 512 need, so both fixture trains would have failed on
  the box.
- **The replacement** is Alpaca's own full-text column, pinned. Text format keeps the demonstration's intent: the
  loss on all tokens and concatenated packing.
  ```
  --dataset tatsu-lab/alpaca --dataset-revision dce01c9b08f87459cf36a430d809084718273017 --format text --text-field text
  ```
  - `prepare_data` with these arguments passes: 958 rows, 102,400 tokens.
  - The data is now pinned, so the unpinned-data note is dropped. The receipt's data record still travels with each
    fixture.
- **The fixture invocation** (superseded by Amendment 3), superseding the one in "Fixture generation":
  ```
  python -m loggetta train allenai/OLMoE-1B-7B-0924-Instruct --revision 7f1c97f440f06ce36705e4f2b843edb5925f4498
    --dataset tatsu-lab/alpaca --dataset-revision dce01c9b08f87459cf36a430d809084718273017 --format text --text-field text
    --seq 512 --micro-batch 1 --steps 200 --seed 42 --fix expert_residency=device
    --out receipts_fixture-<form> --adapter-out fixture-<form>
  ```

**Fixture-base.** The D0 premise read checks that it rebuilt each fixture's trained base exactly, by its stored-tensor digest.
A CPU rebuild uses bitsandbytes' CPU quantizer, while the box quantized on CUDA, and a handful of 1-ulp rounding ties
among about 800M elements would fail an exact digest on noise. So the shakedown box saves the base it trained on,
verified against both fixtures' own receipts, and D0 reads that instead of rebuilding it.

**What the box adds** to the fixture generation, after both fixture trains, in one process. It is never part of P224's
records or verdict.
1. **Load the base:** OLMoE-1B-7B-0924-Instruct at `7f1c97f440f06ce36705e4f2b843edb5925f4498`, on CUDA, at
   `E4B_ABSMAX_DQ=0`. The model is rebuilt by Loggetta's own call, `prepare_qlora_training`, with the setup and revision
   from the plan recorded in `fixture-nodq`'s receipt, so the residency and the fused path are exactly the trains'.
2. **Check it against `fixture-nodq`:** Loggetta's expert digest of the load, for stacks 0 and 15 (the first and the
   last), must equal `fixture-nodq`'s receipt (`correctness.frozen_expert_digest`) exactly. Then every stack's stored
   tensors (the packed codes and the fp32 absmax) are saved to `fixture-base/stacks.safetensors`.
3. **Check it against `fixture-dq`:** `compress_expert_absmax_` runs on the same model, the same digest is checked
   against `fixture-dq`'s receipt, and the 16 stacks' double-quantized payloads (`absmax_q`, `absmax_s`, `absmax_off`,
   `code`) are saved to `fixture-base/absmax_dq.safetensors`.

**The script** is `bench/p224/fixture_base.py`, staged and hash-pinned with the kit. It writes `fixture_base.json` (the
digests wanted and got, the compressed count, an ok flag per part) and exits 0 only when both parts are saved.

**The refusal rule.** A digest mismatch refuses fixture-base only. The double-quantized part is saved only when the
fp32 part matched, since its payloads complete the codes the fp32 part saves. A mismatch is recorded and never retried, and the two
adapters still stand. Fixture-base is hashed on the box with the fixtures (`fixtures.sha256`), fetched, and placed in
the private receipt store's tensor path like the adapters. It is about 3.4 GB.

**The shakedown's fit** (superseded by Amendment 3), superseding the table in "Budget and sequencing":
- The fixture generation term grows by 180 s for fixture-base: one more load (about 120 s) plus the 3.4 GB fetch
  (about 60 s at 50 MB/s).
- It grows by another 240 s for the pinned data: plan and execute each tokenize it once, about 60 s each, per fixture.
- The reserve stays 600 s for a slow link.

| term | s |
|---|---|
| boot | 300 |
| install | 450 |
| fetch (about 14 GB) | 300 |
| bake | 200 |
| train × 2 | 400 |
| reference arms × 2 | 240 |
| 7 server processes × 150 | 1,050 |
| offload test | 60 |
| collect and reduce | 60 |
| fixture generation (2 trains of 200 steps on the pinned data, fixture-base, the fetch) | 1,000 |
| reserve | 600 |
| **total** | **4,660** |

- **The guard** (superseded by Amendment 2): 4,660 × 1.25 = 5,825 s, inside a **1.7 h guard** (6,120 s).
- **The pre-install check:** it needs (4,660 − 300) × 1.25 = 5,450 s left.
- **Worst case** (superseded by Amendment 2): 1.7 h × $0.18 + (14 GB + 5 GB + 3.4 GB) × $0.011 = $0.306 + $0.246 =
  **$0.55**, under the standing
  rule.

The reading's terms, guard and worst case are unchanged.

## Amendment 2 (before any data): the shakedown card

**Why.** At its first launch, `p224-shake-1`'s validate (at $0) refused the card the screen found: the launcher's GPU
policy does not list it. Among the policy's classes that meet the shakedown's terms (sm_86 or later, 16 GB or more), the
only offer at or under $0.18/h was one RTX A4000 with a reliability of 0.60, too unreliable to stake a shakedown id on.
The cheapest dependable class is the L4, at $0.30-$0.36/h (reliability 0.993-0.999) when this was written.

**The card and the cap (superseding the setup's card and cap):** one L4 (sm_89, 24 GB, in the launcher's policy) at
most $0.40/h, P222's shakedown tier. An RTX A4000 at or under $0.18/h with a reliability of at least 0.98 stays
eligible.

**The guard.**
- The fit table is unchanged: Amendment 1's terms, 4,660 s.
- The guard is raised to **1.9 h** (6,840 s). The L4's memory bandwidth (about 300 GB/s) is below the A4000-class card
  the terms assumed (about 448 GB/s), and the seven server processes and the decode work are bandwidth-bound. A 1.5×
  slowdown on the 1,050 s server term alone (+525 s) would use nearly half of the 25 % headroom (1,165 s).
- 4,660 × 1.25 = 5,825 s, inside the 6,840 s guard, with 1,015 s to spare for the slower card.
- The runner's pre-install check is unchanged: (4,660 − 300) × 1.25 = 5,450 s.

**Worst case:** 1.9 h × $0.40 + (14 GB + 5 GB + 3.4 GB) × $0.011 = $0.76 + $0.2464 = **$1.0064** (about $1.01), under
the standing rule.

**What the shakedown records for the reading's fit.** Each step's measured seconds on this card: the phase stamps
(`phases.jsonl`) bracket the install, the fetch, the bake, each train, each reference arm and each of the seven server
processes. The reading's terms are then checked against these, scaled for the 5090, before its exec ACK.

The reading's terms, guard and worst case are unchanged.

## Amendment 3 (before any data): the arena's bake, the canaries' kernel, the pins, the fixtures

**Why.** `p224-shake-1` (an instrument run, not a strike) found three kit faults. They were found after the run, so
this amendment registers the fixes before any data.
- **The arena.** The kit baked with P39's `k8_bake.py`, a relocation bake. Its index names a local NF4 snapshot and
  records no quantizer, so the paged server refused every PA slot on refusals 3 and 4, exactly as it should.
  grouped-nf4-gemm's `nvme_bake_nf4.bake_nf4` quantizes straight from the HF snapshot. Its index records that snapshot
  path (refusal 3 reads the commit from it) and a full `quantizer` block (refusal 4). The served lanes SV1, SV2 and SV4
  bake with it.
- **The canaries' kernel.** An attention-only canary (`train_experts=false`) cannot use the fused expert kernel: e4b's
  recipe refuses `expert_kernel=grouped_nf4` without expert training, and Loggetta's planner correctly falls back to
  `reference`. `enable_fast_train` therefore never runs, and the canary's frozen expert absmax is fp32 BY CONSTRUCTION,
  whatever `E4B_ABSMAX_DQ` says. Canary B (REF-DQ, PA-DQ) cannot exist as registered.
- **The reduce.** The runner called the reducer without `--pins`, so the reduce exited 2. The fixtures, which came
  after it, never ran.

**A finding, stated.** For attention-only adapters, ADAPTER_PORTABILITY's F1 (a double-quantized trained base served
on the fp32 arena) never arises: the trained expert base already has the arena's form. F1 is priced where it can exist,
on expert-and-attention adapters, through the fixtures (`fixture-dq` against `fixture-nodq`) in PA-4's cross-format
read, after D0's premise.

**What changes.**
1. **The arena (superseding "bake (P39's `k8_bake.py`)").**
   - The kit bakes with `bake_nf4(SNAP, work/nf4.arena)` from the HF snapshot, as SV2 does.
   - Before any server process, the runner checks that `arena_source(index)` equals the registered `(MODEL, REV)` and
     that the index's `quantizer` has `quant_type` `nf4` and `blocksize` 64, the adapter's recipe. A mismatch is a
     HARNESS_ERROR before any slot runs.
   - A kit test runs refusals 3 and 4 on the index a REAL `bake_nf4` writes, from a tiny snapshot on CPU, never on a
     hand-built dict.
2. **The canaries (superseding canary B and the absmax check).**
   - REF-DQ and PA-DQ are dropped. They were REPORTED only, so no gate changes: G1-G5 are unchanged.
   - Canary A's training command is unchanged, with no kernel `--fix`.
   - Its record must show the engaged `expert_absmax.form` `fp32` AND the setup's `expert_kernel` `reference`, as
     registered here. A mismatch VOIDs.
3. **The process order (superseding "Seven fresh server processes").** Six fresh server processes, in two ABBA
   blocks of P0 / PA / P0. There is no trailing slot.
4. **The pins.** The runner writes the registered pins (`e4b_sha`, `model_revision`, `canary_sha256`) to `pins.json`
   and passes it to the reduce. A kit test parses every `python $W/<script>.py` call in the runner with that script's
   own argument parser, so a missing required flag fails in CI.
5. **The fixtures (superseding "AFTER the shakedown's own records, the collect and the reduce").**
   - The two fixture trains take `--fix expert_kernel=grouped_nf4` (they train experts, so the fused path applies).
     The authoritative fixture command, superseding Amendment 1's code block (`<form>` is `dq` or `nodq`;
     `fixture-nodq` runs with `E4B_ABSMAX_DQ=0`, `fixture-dq` with it unset):
     ```
     python -m loggetta train allenai/OLMoE-1B-7B-0924-Instruct --revision 7f1c97f440f06ce36705e4f2b843edb5925f4498
       --dataset tatsu-lab/alpaca --dataset-revision dce01c9b08f87459cf36a430d809084718273017 --format text --text-field text
       --seq 512 --micro-batch 1 --steps 200 --seed 42 --fix expert_residency=device --fix expert_kernel=grouped_nf4
       --out receipts_fixture-<form> --adapter-out fixture-<form>
     ```
   - They run after the shakedown's own steps WHATEVER the reduce outcome, provided the box is healthy and each step's
     time-left check passes.
   - They never touch or alter the shakedown's records or verdict, and a fixture failure stays recorded beside it.

**The fits** (25 % headroom; the bake is `bake_nf4`, recorded at 260 s and 133 s for Qwen3-30B-A3B and 47 s for
OLMoE in SV1 and SV2, so the 600 s and 200 s terms stand).

The reading:

| term | s |
|---|---|
| boot | 300 |
| install | 450 |
| fetch (about 61 GB) | 1,200 |
| bake (`bake_nf4`) | 600 |
| train (canary A) | 360 |
| reference arm (REF, with HF0 and the fingerprints) | 240 |
| 6 server processes × 360 | 2,160 |
| offload test | 60 |
| collect and reduce | 60 |
| reserve | 600 |
| **total** | **6,030** |

- 6,030 × 1.25 = 7,537 s, inside the unchanged 2.5 h guard (9,000 s).
- The pre-install check needs (6,030 − 300) × 1.25 = 7,162 s.
- The worst case is unchanged at **$3.35** (the guard and the cap).

The shakedown:

| term | s |
|---|---|
| boot | 300 |
| install | 450 |
| fetch (about 14 GB) | 300 |
| bake (`bake_nf4`) | 200 |
| train (canary A) | 200 |
| reference arm | 120 |
| 6 server processes × 150 | 900 |
| offload test | 60 |
| collect and reduce | 60 |
| fixture generation (2 trains on the pinned data, fixture-base, the fetch) | 1,000 |
| reserve | 600 |
| **total** | **4,190** |

- 4,190 × 1.25 = 5,237 s, inside Amendment 2's unchanged 1.9 h guard (6,840 s).
- The pre-install check needs (4,190 − 300) × 1.25 = 4,862 s.
- The worst case is unchanged at **$1.0064** (Amendment 2's cap and guard).

The next shakedown is `p224-shake-2`, after this amendment and its kit PR merge.

## Amendment 4 (2026-10-22, before any data): spare shake ids

**Why.** Two of the three registered shake ids are spent, and neither found the reading's question answered or a strike:
- `p224-shake-1`: an instrument run that found three kit faults, fixed by Amendment 3 and its kit.
- `p224-shake-2`: refused at $0 before any instance existed. The launch script refuses the create unless its
  pre-create machine-filtered read of the pinned offer is PRESENT. That read was not kept, but the launch was posted at
  05:16:55Z, so it passed. The launcher's first pinned read at 05:17:02Z, and those at 05:17:18Z and 05:17:33Z, found
  the offer absent. The screen's quote read was at 05:14:13Z. The receipt is committed, so the id is spent.

That leaves `p224-shake-3`, so one more market loss would stall the lane.

**What changes.**
- **Run ids:** `p224-shake-4`, `-5` and `-6` are registered as shakedowns under the current terms exactly:
  - Amendment 2's card and cap: one L4 at most $0.40/h, or an RTX A4000 at most $0.18/h with a reliability of at least
    0.98;
  - Amendment 2's 1.9 h guard and worst case of $1.0064;
  - Amendment 3's kit, steps and fit (4,190 s);
  - the verdict SHAKEDOWN, never a reading and never a strike;
  - each spent the same way `-1` to `-3` are.
- **The runner and the driver** accept `p224-shake-1` to `-6` and `p224-5090-1` to `-3`, and refuse any other id.
- **The pin:**
  - the L4 first, Amendment 2's primary card, with the A4000 only when no L4 passes the terms;
  - prices read in the launcher's form (the GPU plus the 60 GB disk), the form its ceiling check uses;
  - a host is pinned only when it was listed in two consecutive screen sweeps over at least 180 s and in at least half
    of the last hour's sweeps;
  - the pinned offer is read once more right before the create, and if it is absent or above the quote nothing is
    created and the id is kept. A read that passes does not protect the id: if the offer is gone by the launcher's
    own first pinned read, that refusal is committed and the id is spent, as `p224-shake-2`'s was.

Nothing else changes: the arms, the rule, the band, the reading's terms, the budget and the guards are as before.

## Amendment 5 (2026-10-22, before any reading): the scored set, a distinct held-out set, and the cost instrument

**Why.** `p224-shake-3` (a SHAKEDOWN; the kit ran end to end) showed three signals that would have decided the reading
for the wrong reason, and a fourth fault under all of them.
- **Duplicates in the held-out set.** It was drawn with replacement from 8 topics × 4 templates, so it could hold at most
  64 distinct rows, and the 64 draws held 46 distinct. Duplicates score identically, so the registered SE counted them as
  information. Over distinct rows the SE of PA − REF is 0.00255 on the assistant positions (registered estimator:
  0.00219) and 0.00393 on every token (registered: 0.00327). Raising the count alone would have added only repeats.
- **Power.** The SE above is not under 0.2 × 0.01 = 0.002, so a reading of this size VOIDs as "too small to judge".
- **G1 on every token.** The registered estimator read |loss(PA) − loss(REF)| = 0.0163 against a margin of 0.0120.
  - Over distinct rows it is +0.0161 against a margin of 2 × 0.0107 = 0.0214, and the difference in differences,
    (PA − REF) − (P0 − HF0), is +0.0054 with SE 0.0080 (0.7 SE). The all-token excess is not established at n=64; the
    reading measures it.
  - The served merge's bf16 cast is excluded as its source, by measurement. On the shakedown's real base attention
    weights and adapter, `merge_attention_lora` (line for line) loses about 1 % of the delta (0.93-1.13 % by projection
    type), with cos 1.0002 between the applied and the exact delta: a small unbiased perturbation. Whether the paged
    path's own gap acts differently once the adapter moves K and V is an open question.
  - On the assistant positions (the canary's own tokens), PA − REF = −0.0062 and the difference in differences is
    −0.0316 with SE 0.0114 (2.8 SE): the served adapter applies about 0.5 % more than the reference (ratio 1.006).
- **G3.** The W1 ITL read +4.5 % for PA in both blocks, but that was the instrument.
  - `http_cost` timed only chunks with non-empty text. A token the streaming detokenizer holds back arrives as an
    empty-text chunk, so the next gap spans two tokens. PA writes different text, about 5.5 such tokens per request.
  - On the per-token wall time, P0 is 10.883 ms and PA 10.858 ms: equal.
  - At W16 PA is 2.6 % FASTER on wall tok/s. The arms decode different text, which routes to a different spread of
    experts at B = 16; at B = 1 top-8 of 64 always touches exactly 8 experts per layer.
  - TTFT used the same first-text-chunk definition.

**What changes.**
1. **The held-out set** (superseding "64 rows").
   - It is the full cross product of 32 held-out topics × the 8 templates (4 trigger, 4 control), each row exactly once:
     **256 distinct rows**, 128 of them triggers.
   - The 32 topics are the 8 earlier ones plus 24 new words of the same class (plural or mass nouns under the same
     per-topic answer rule), disjoint from the 32 training and 8 HTTP topics and substring-free against every topic.
   - The training draws and the HTTP set are byte-identical to Amendment 3's: `train.jsonl` sha256 `770693c8…`,
     `http.json` sha256 `9b24c507…`. The canary trains on the same rows.
   - The canary data pin is now **`287706ba066909f83c63a77bd552bc64f7fbf080c0b90fc07ea6cd4ae456efe7`**.
2. **New VOID rungs:** the held-out sequences not all distinct (each arm records a per-sequence id); and
   a held-out row equal to a training row (the full row's text sha256; an unread count is not zero). The held-out count is 256 in
   every arm, on every token and on the assistant positions.
3. **Every statistic is taken over distinct sequences.**
4. **G1 is graded on the ASSISTANT positions** (the trains use `--loss assistant`; that is the function the adapter was
   trained to correct).
   - PA's improvement over P0 must reach 0.95 of REF's over HF0, and |loss(PA) − loss(REF)| must stay within the margin.
   - The margin's yardstick is the ALL-TOKEN base gap, `min(0.05, max(0.01, 2 × |loss(P0) − loss(HF0)|))` over every
     token: the better-measured of the two (the assistant-only base gap's per-sequence SD was 0.079).
   - The SE rung reads the assistant positions' PA − REF.
5. **Reported, never ruling** (beside the graded G1):
   - the all-token G1, with what it would have failed;
   - on both position sets, the signed difference in differences with its paired SE, flagged when |DiD| > 2 SE and
     |DiD| > the margin;
   - over-application, flagged when an effect ratio exceeds 1 / 0.95 (the ratio gate is one-sided);
   - the SE of P0 − HF0 on both sets; on a set with duplicates, the SE both ways.
   The DiD is not graded at this n. Grading it would be a later amendment, registered before that reading.
6. **G3 is graded at W1 on the per-token wall time.**
   - The box records `itl_wall_ms` = (wall − first chunk) / (completion tokens − 1) per request (the median over
     requests), and `ttft_any_ms`, from the first chunk of any kind.
   - The graded cells are W1's `itl_wall_ms`, `ttft_any_ms` and `tok_s`, under the registered band rule (the max over
     the blocks of |A1 − A2|, never under 1 % of the A slots' mean).
   - W16 and the text-cadence `ttft_ms` / `itl_ms` are computed the same way and reported, with the reason above.
   - Peak VRAM is unchanged.
7. **Power, the reading's assumption.** The shakedown's assistant-position SD of PA − REF over distinct rows is 0.0175.
   At 256 rows that gives SE 0.00109, and the bar (SE < 0.002) holds for a Qwen3 SD up to 0.032. The yardstick sharpens
   with it: the all-token SE of P0 − HF0, 0.0066 over 46 distinct rows, scales to about 0.003 at 256.
8. **The fit** (superseding Amendment 3's table). 256 sequences raise the reference arm and add a scoring term to each
   of the two scoring slots. The basis is `p224-shake-3`'s measured phases: the reference process, 170 s for 64 + 64
   sequences with its load, and a scoring slot about 59 s more than a non-scoring one. The sequence-proportional parts
   are scaled by 4 and rounded up.

   | term | reading, s | shakedown, s |
   |---|---|---|
   | install | 450 | 450 |
   | fetch | 1,200 | 300 |
   | bake | 600 | 200 |
   | train | 360 | 200 |
   | reference arm | 720 | 600 |
   | 6 server processes | 6 × 360 | 6 × 150 |
   | scoring, 2 slots | 2 × 240 | 2 × 180 |
   | offload test | 60 | 60 |
   | collect and reduce | 60 | 60 |
   | fixture generation | 0 | 1,000 |
   | reserve | 600 | 600 |
   | **total** | **6,690** | **4,730** |

   - Reading: 6,690 × 1.25 = 8,363 s, plus 300 s of boot is 8,663 s, inside the unchanged 2.5 h guard (9,000 s).
   - Shakedown: 4,730 × 1.25 = 5,913 s, plus boot is 6,213 s, inside the unchanged 1.9 h guard (6,840 s).
   - The guards and the worst cases are unchanged ($3.35 and $1.0064). The runner's fit check refuses before any install
     if the time left is short.
9. **The self-test on real records.** `tests/test_p224.py` re-grades `p224-shake-3`'s own cells
   (`bench/p224/shake3_cells.json`, sha256-pinned; its row identities come from the Amendment 3 generator). Over its 46
   distinct rows:
   - the graded G1 passes, and the all-token G1 passes;
   - the assistant DiD is flagged;
   - the SE would VOID a reading of that size;
   - the per-token wall time puts PA inside the W1 band, while the text-cadence ITL and W16 tok/s fall outside and are
     reported.

Nothing else changes: the arms, the ratio bar, the margin's form, G2, G4, G5, the guards and the worst cases are as
registered. The next shakedown id is `p224-shake-4` (Amendment 4), and it uses this kit.
