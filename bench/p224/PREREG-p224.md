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
| **REF-DQ** | `loggetta.load_adapter`, `E4B_ABSMAX_DQ` unset (the default, double-quantized) | canary B: trained at the default |
| **P0** | the shipped paged server, `PagedServeConfig.from_env()` + `build_engine` | none |
| **PA** | the same server with `E4B_PAGED_ADAPTER` = canary A, `E4B_PAGED_ADAPTER_ACCEPT_UNVERIFIED_BASE=experts` | canary A |
| **PA-DQ** | the same, with canary B on the same fp32 arena (REPORTED) | canary B |

**Training (both canaries, Loggetta's CLI; only `E4B_ABSMAX_DQ` differs):**
```
python -m loggetta train MODEL --revision REV --dataset train.jsonl --format chat --loss assistant --packing concat
  --repeat-data --seq 512 --micro-batch 1 --grad-accum 1 --steps 80 --learning-rate 5e-4 --lr-schedule cosine --seed 0
  --fix train_experts=false --fix train_attention=true --fix attn_4bit=false --fix expert_residency=device
  --out receipts_<arm> --adapter-out adapter_<arm>
```
- **The training record** (`train_<arm>.json`) carries Loggetta's receipt: the setup, the engaged report and the
  adapter's sha256. Its engaged `expert_absmax.form` is the binding check that each canary trained against the absmax it
  registers: `fp32` for canary A, `double-quantized` for canary B.
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
  - A hit is the CANARY string in the text.
  - The control prompts go through the two non-streamed modes.
  - P0 and PA-DQ run the same set.
- **Identity.** The record reads `/health`'s adapter block (the id, the manifest sha256, `base_match`, the merge
  report), `/v1/models`, and the status of a completion request that names the base model.
- **Cost.** Each slot's process serves over HTTP on the box. After one untimed warm request and one untimed warm wave,
  it runs two workloads, each request streaming 128 tokens (`min_tokens` = `max_tokens` = 128, `ignore_eos`):
  - **W1:** 8 sequential requests.
  - **W16:** 2 waves of 16 concurrent requests.

  The metrics are `ttft_ms` (the median over requests), `itl_ms` (the median over requests of each request's mean
  inter-chunk gap) and `tok_s` (completion tokens over the waves' wall time). Peak VRAM is
  `torch.cuda.max_memory_allocated` after a reset that follows the scoring.

**The process order.** Seven fresh server processes:

| block | slots |
|---|---|
| 0 | P0 (scores), PA (scores), P0 |
| 1 | P0, PA, P0 |
| last | PA-DQ (scores) |

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
     - The band is frozen per metric (`ttft_ms`, `itl_ms`, `tok_s`) and per W (W1, W16): band = the max over the two
       blocks of |A1 − A2|, the two P0 slots. PA fails when |B − mean(A1, A2)| > band in any block.
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
- PA-DQ's effect ratio, |loss(PA-DQ) − loss(REF-DQ)| (F1, priced) and its HTTP canary.
- The decode-vs-prefill deltas and their flag.
- The merge's seconds and its max error.
- Each fingerprint scope's seconds and module count.
- Every slot's cost cells.

## Predictions (graded in the read, never ruling)

| prediction | basis |
|---|---|
| **PASS** | the merge is exact by construction in fp32 and cast once; the served model is the paged base plus a constant weight change |
| **ratio in 0.98-1.02, \|PA − REF\| under 0.01 nats** | the paged base's gap acts on both P0 and PA |
| **G3 inside the band at both W** | a merged adapter changes no shape and no kernel; the served step is the base's |
| **PA-DQ ratio ≥ 0.9** (moderate confidence) | F1's absmax difference is small next to the canary's effect |
| **decode-vs-prefill not flagged** | the adapter changes weights only, never the decode route |

## Fixture generation (on the shakedown box; not a reading, not graded)

The D0 premise read (per-format expert-LoRA retention, run at $0) needs a stored, trained adapter WITH expert LoRA, and
none exists. The shakedown box already holds OLMoE and Loggetta, so it trains two, declared here.
- **When:** AFTER the shakedown's own records, the collect and the reduce. They NEVER affect its verdict: a fixture
  failure is recorded beside it, and the SHAKEDOWN verdict stands on its own.
- **The trains:** Loggetta's defaults (expert and attention LoRA, the default r and alpha, the default setup), the
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
  - The demonstration data is fetched unpinned, so the receipt's dataset fingerprint and token hash are the data
    record.
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
  2.8.0+cu128, transformers 5.17.0, Loggetta at `90a7697d` (0.6.1; its receipt carries e4b's engaged report, whose expert absmax form the
  schema v2 PR adds); Qwen3-30B-A3B at
  `ad44e777` (the SC1 pin); the canary pin as `p224_canary.py` prints it.
- **The fit** (25 % headroom):

  | term | s |
  |---|---|
  | boot | 300 |
  | install | 450 |
  | fetch (about 61 GB) | 1,200 |
  | bake (P39's `k8_bake.py`) | 600 |
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
    sm_86 or later with 16 GB, at most $0.18/h.
  - **What it ranks:** the structure, never a reading. Its verdict is SHAKEDOWN, and every gate and threshold is
    reported.
  - **The expected merge count** for OLMoE comes from the trained adapter's own attention projections (16 layers × 4).
  - **The fit**, with the fixture generation below (the terms are estimates scaled from the reading's; the box's own
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
  - **Worst case:** 1.5 h × $0.18 + (14 GB + 5 GB) × $0.011 = $0.27 + $0.209 = **$0.48**, under the standing rule.
- **Run ids:** `p224-shake-1..3` (shakedowns), then `p224-5090-1` (the reading), with `-2` and `-3` as spares for
  pre-create and host refusals, which are not strikes. Two box-started HARNESS_ERROR or VOID reads stop the lane.
- **Sequencing:**
  1. This registration merges after one independent review and the maintainer's pass.
  2. The kit PR (runner, driver, staged hashes) is pinned to e4b main after the schema v2 PR merges, with its own
     review.
  3. The shakedown, then the exec ACK with the kit hash and pins, the pin, and `--validate`.
  4. The receipts go to the private receipt store.
