# SD2 (e4b#1313): SD1 Phase 1 -- an EAGLE-3 speculative decode path in `serve_paged`, measured end to end on one RTX 5090

Registered 2026-10-10, before any build or run. The lane id is SD2, titled "SD1 Phase 1"; the maintainer ruled the id,
the shape and the exclusions below (bus, 2026-10-10T06:40:43Z).

## The question

SD1 (`bench/sd1/RESULTS-sd1.md`) measured how many tokens each verify step would yield on `Qwen/Qwen3-30B-A3B` at e4b's
shipped default with the licensed EAGLE-3 head: τ(k = 1) is 1.7157 on chat with reasoning on, 1.6465 with it off, and
1.5353 on raw wikitext. It then priced those τ with P123's verify-cost model, which is a MODELLED speedup: S 1.2303,
1.1758 and 1.0964 at the best k, under independent routing.

SD2 builds the path and measures it. With speculation on, does single-stream greedy decode of the target get faster end
to end on one RTX 5090, at a quality the T == 1 path's own floor accepts? The lane replaces each modelled term with a
measured one:
- **the verify step's cost** at T = k + 1 rows, against the T == 1 decode step;
- **the draft's cost**, the EAGLE-3 head running in the engine;
- **the loop's own cost**: the accept, the rollback and the host read per step;
- **the speedup itself**, as decode tokens per second with speculation on against off.

## What SD2 inherits, and what it does not

- **Measured (SD1, `sd1-5090-2`):** τ and draft acceptance for k = 1–5 on three workloads, from the target's captured
  hidden states; the head's pinned bytes; the auxiliary-layer convention (2 / 24 / 45); the capture path equal to the
  served path's tokens (160 / 160).
- **Modelled, and replaced here:** S. SD1 priced a verify step from P123's census of the B = 1 and B = 16 **batched**
  decode steps. A verify step is one sequence's k + 1 rows, and nothing has measured one at k + 1 ≤ 4 on today's stack.

## Prior work in this repository: S2-lite and S3 (2026-08-25)

This model and card have been through speculative decoding before. SD1's PREREG did not cite either lane; that was a gap,
and SD1's results now say so.

**S2-lite** (`bench/hybrid-g9/s2/`): the verify regime in `engines/paged_attention.py`, mode `"verify"`. It is one
sequence's k + 1 query rows read through the paged FP8 decode kernel with staggered lengths (row i reads the past plus
draft tokens 0..i), so causality comes from lengths over already-appended K/V. `Fp8PagedKV.rewind` /
`rewind_nosync` make the rejected tail unreadable by moving the lengths back, with no data movement.
- The regime passed its bitwise sequential-oracle gate at K = 16: 17 / 17 argmax tokens and the continuation after the
  rewind identical. These mechanics are on main and tested (`tests/test_s2_verify_mechanics.py`).
- Its captured verify step cost 47.71 ms against a 7.41 ms decode anchor, at singleton grouping, the only capture-legal
  grouping then.

**S3** (`bench/hybrid-g9/s3/`): capture-safe **device** grouping for T > 1 calls (`hot_residency.DEVICE_GROUPING`,
bitwise against eager grouping), then a captured, device-grouped verify step:

| rows (K + 1) | verify, ms | against the 7.39 ms anchor |
|---|---|---|
| 17 | 36.38 | 4.93× |
| 33 | 39.14 | 5.30× |
| 65 | 43.72 | 5.92× |

With prompt-lookup acceptance (2.95–3.93 tokens per step), speculation lost 34–40 % at every K: **REFUTED**.

**Why S3 does not settle SD2, and what SD2 must answer because of it:**
- **The rows.** S3 measured 17 rows and up. EAGLE-3's best k is 1–2 (SD1), a verify of 2–3 rows. S3's cost was nearly
  flat in rows (36.4 → 43.7 ms for 17 → 65), which points at a large fixed cost in the T > 1 path. **Whether that fixed
  cost exists at 2–4 rows on today's stack is the first thing SD2 measures** (stage V), and it can refute the lane on its
  own.
- **The stack.** S3's anchor was 7.39 ms. Today's B = 1 step on SD1's target build is 3.91 ms (P127's B arms, 255.9
  tok/s), and its 16-row batched step is 16.3 ms (P127), against S3's 36.4 ms for 17 verify rows. K25's grouped small-M
  GEMM (P121, 2026-10-08) and P127's launch-bound glue (2026-10-09) are new since.
- **The yield per row.** Prompt lookup bought 2.95 tokens for 17 rows. EAGLE-3 buys 1.54–1.72 for 2 rows.
- **The routing.** S2 read 41.5 distinct experts per layer for 17-token windows, against S2's ~78 for random routing.
  Consecutive tokens share experts, so SD1's independent-routing S is the conservative arm.

## Prior art outside this repository

Read from source at pinned commits, nothing executed:
- **vLLM** main `ff440c78`. Model Runner V2 (`vllm/v1/worker/gpu/`) is the default there, and the older runner is
  deprecated.
- **SGLang** main `6e8f3e09`. Its EAGLE worker is `speculative/eagle_worker_v2.py`; the V1 worker was deleted on
  2026-06-08.

| | vLLM (V2 runner) | SGLang (V2 worker) | SD2 |
|---|---|---|---|
| **rejected KV** | no rollback op: the scheduler takes the rejected count off `num_computed_tokens` (`vllm/v1/core/sched/scheduler.py:2109-2127`), and the next step overwrites the slots. No block is freed (`vllm/v1/core/single_type_kv_cache_manager.py:228-235`) | nothing freed per step: drafts, verify and draft-extend write one window, the length advances by the accept, and the next step overwrites the rest (`python/sglang/srt/speculative/eagle_worker_common.py:554`). The tail is freed at release | a length, as S2-lite's `rewind_nosync` (B2) |
| **the verify forward** | the unified prefill forward over 1 + k positions; FlashAttention's causal varlen call with the block table (`vllm/v1/attention/backends/flash_attn.py:1330-1479`) | `ForwardMode.TARGET_VERIFY`: at topk 1 on FA3, plain causal attention over the context and the drafts (`python/sglang/srt/layers/attention/flashattention_backend.py:987-1031`) | the paged FP8 **decode** kernel, one row per position, S2-lite's staggered lengths (B1) |
| **graphs** | FULL uniform-decode graphs with `uniform_token_count = 1 + k`; at batch 1 exactly 1 + k tokens, no padding (`vllm/v1/worker/gpu/cudagraph_utils.py:314-338`) | with speculation on, the target captures TARGET_VERIFY in place of DECODE at width 1 + k (`python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py:298-311`) | the server's decode buckets, rows sharing one slot (B1) |
| **the accept** | on the device, in Triton: accept while the draft equals the argmax, then the argmax; the bonus at the last row (`vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py:484-631`). No host sync on the step's path: the copy back runs on a side stream (`vllm/v1/worker/gpu/async_utils.py:115-174`) | on the device (`VerifyTreeGreedy`); accept lengths are copied back asynchronously and read one iteration later (`python/sglang/srt/managers/scheduler_components/batch_result_processor.py:748-818`) | on the device, then **one host read per step**: a cost both references avoid, measured in V1's loop term (B5) |
| **aux states** | default layers `(2, L/2, L − 3)` = (2, 24, 45) at L = 48, the residual stream entering the layer (`vllm/model_executor/models/interfaces.py:1580-1590, 1691-1712`) | the same (2, 24, 45) for Qwen3-MoE (`python/sglang/srt/models/qwen3_moe.py:1132-1147`) | SD1's (2, 24, 45), as measured |
| **the draft after the verify** | the drafter batch stays at the target's shape, "to avoid CPU-GPU synchronization", reusing its metadata; rejected rows are shifted out (`vllm/v1/worker/gpu/spec_decode/target_dependent_ar/speculator.py:240-245, 710-789`) | draft-extend runs over the whole 1 + k window at a static shape and keeps the row at the accept (`python/sglang/srt/speculative/eagle_worker_v2.py:1062-1101`) | SGLang's form: draft-extend over all k + 1 verify rows, static, the chain from row a (B4) |
| **the draft's KV** | an extra layer in the same paged manager and block table (`vllm/model_executor/models/llama_eagle3.py:317-326`); rejected entries overwritten | the target's slot mapping, its own tensors; rejected entries overwritten | its own one-layer cache for the one speculating slot, rolled back by length |
| **equality with plain greedy** | not asserted. The EAGLE end-to-end test passes at `int(0.6 × N) + 1` of N exact output matches (`tests/v1/e2e/spec_decode/eagle/utils.py:127-135`); the FAQ names speculative batch expansion as a source of numerical drift | asserted only under `--enable-deterministic-inference` (Llama-3.1-8B, four prompts, `python/sglang/test/kits/spec_server_kits.py:131-189`); elsewhere thresholds, with a stated 0.5 log-prob tolerance because decode and rescoring "run different kernels / batch shapes" | not bitwise by construction; gated by quality at the verify shape (stage Q) |
| **MoE** | no MoE guidance; the head ships k = 3 | no MoE guidance in the docs. But the automatic default gives every MoE architecture it names (DeepSeek, gpt-oss, GLM-MoE and others) a 3-step chain with 4 verify tokens, where `LlamaForCausalLM` gets a 5-step, topk-4 tree with 8. `Qwen3MoeForCausalLM` falls to the same (3, 1, 4) (`python/sglang/srt/arg_groups/speculative_hook.py:1338-1371`) | the verify's distinct-expert cost is the term SD1 modelled. V measures k = 1–3, which includes the references' k = 3 |
| **turning it off** | a per-batch-size k schedule. At k = 0 the draft's prefill still runs, to keep its KV in step | `--speculative-adaptive`: an acceptance EMA per batch-size range, swapping pre-captured graph sets. At 0 steps, draft-extend keeps running | drops the draft state at the first batched step, for the request's life (the maintainer's exclusion). Keeping the draft in step, as both references do, is a later lane's |

**What SD2 takes from them:**
- rollback by length;
- the accept on the device;
- the auxiliary convention;
- a static-shape draft-extend;
- a quality gate in place of a bitwise one.

**Where SD2 departs, and why:**
- **The verify reads through the decode kernel and the decode graphs,** not a prefill-style varlen attention. That is
  the path e4b's FP8 pool and graphs already serve, and S2-lite proved it.
- **Phase 1 keeps one host read per step.** Removing it is the same move as P118's decode lookahead, and V prices it
  first.
- **Neither reference documents a measurement or a model of what a verify step costs an MoE target at batch 1.**
  SGLang's smaller default for MoE architectures is a setting, not a reading. That cost is SD2's question.

## What gets built (the code under test)

The build lands in reviewed implementation PRs after this registration merges. Everything is behind `E4B_PAGED_SPEC`,
default off. Amendment 1, before any read, pins their merge commits and carries the harness:
- `bench/sd2/`'s box, runner, driver and reducer, the reducer self-tested on every verdict;
- `staged.sha256`;
- the dry run.

**B1. The verify step is a decode bucket whose rows share one slot.** A verify of k + 1 rows runs as a batched decode
step of k + 1 rows, every row mapped to the speculating slot:
- **positions** base .. base + k, one per row;
- **reads** through `E4B_KV_STEP_SELECT`'s per-step selection (on by default since P111), whose lengths are prepared
  outside the graph. Row i reads base + i + 1: S2-lite's stagger, set on the host side of the replay;
- **writes** through the fused batch append, with each row at base + i. The append writes each row at its slot's current
  length. So rows 1..k run on k dedicated **alias slots**, appended to the pool like the scratch slots, whose
  block-table rows are copied from the speculating slot's (`Fp8PagedKV._bt_all`, one copy per step) and whose lengths
  are base + i. The fused append then writes into the speculating slot's own blocks at base + i, and step-select's
  per-step gather reads base + i + 1, with no kernel change.
  - The alternative, a per-row offset in grouped-nf4-gemm's append with its bitwise test against the per-row loop, is
    kept if aliasing fails review. The PR says which it took.

**Aliasing's four conditions** (the maintainer's, each with a CPU test):
1. **The alias slots sit outside `kv.scratch`, and that is asserted.** A padding row writes at its slot's position 0,
   which on an aliased table would be the speculating slot's first token. So a padding row never lands on an alias slot.
2. **The order of the length writes.** `graph_bucket_publish` adds 1 to every bound slot after the replay.
   - B2's rollback to base + a + 1 is the **last** length write of the step, after the publish.
   - The alias lengths are set again at every step.
   - A test holds the order.
3. **The sequence's end.** If base + k would pass the slot's last position or the request's `max_tokens`, k shrinks for
   that step, or the step runs at T == 1. Blocks are claimed to `blocks_per_seq` − 1 at the first decode, so a boundary
   costs no allocation. A test holds a verify that straddles a block boundary to writing into the right block.
4. **Bucket 3 is for the verify only.** Plain batched steps keep `bucket_for` over (1, 2, 4, 8, 16), so ON2's batched
   path is the shipped default's.

The MoE side is the batched path the shipped default already serves at B > 1: device grouping and K25 (P121's license).
The verify step replays the bucket graphs the server already captures, buckets 2 and 4 for k = 1 and 3. For k = 2 the
server adds a bucket 3: a padded fourth row would route through the MoE and add its experts to the verify's cost. No
new forward path is written.

**B2. Rollback is a length.** After the accept, the slot's lengths move to base + a + 1 on every pool layer: the
accepted a drafts plus the target's own token. This is `rewind_nosync`'s write with a device value, so no host read
is needed before the next step. Nothing is freed: a slot owns its blocks for its life (`_ensure_graph_ready`).

**B3. The auxiliary states.** Forward pre-hooks on layers 2, 24 and 45 (SD1's `AuxCapture`) copy each row's residual
stream into static buffers, inside the captured graphs and during prefill. Prefill writes the prompt's states, for the
draft's context; the prefill graph (`E4B_PAGED_PREFILL_GRAPH=auto`) captures the copy too, or the PR shows it cannot
and prefill runs eager under speculation.

**B4. The draft in the engine.** `engines/eagle3_draft.py` is SD1's `sd1_eagle3.py` arithmetic (vLLM's EAGLE-3
inference semantics) in the package:
- bf16 on the GPU, its own one-layer KV cache for the speculating slot (2 KB a position);
- the head checked against the pinned sha256 at load;
- the chain of k draft steps as one captured graph.

After each accept, its first forward is a draft-extend over all k + 1 verify rows at a static shape, SGLang's form. It
re-reads the positions from the target's own states and keeps the chain from row a. The draft cache's length then
moves to base + a + 1, so the rejected rows' entries are overwritten later. RoPE uses θ = 10,000, the head's own (the
target's is 10⁶), as SD1 ran it and measured τ inside the card's range. A test holds the in-engine chains to
`sd1_eagle3.chain_at` on the same inputs.

**B5. The loop and the scheduler.**
- **One host read per step:** the accept count and at most k + 1 tokens. The accept is computed on the device: the
  leading matches between the drafts and the verify's argmax, as `engines/speculative.py`'s strict rule.
- **`run_decode` may return several tokens for a request.** `ContinuousScheduler._emit` takes them in order, and stops at
  a stop id or at `max_tokens`. A token past the stop is dropped, and the slot is freed as today.
- **Speculation engages only while one request decodes alone** and no prefill is pending. A speculating request that
  becomes batched (a second request is admitted) drops its draft state and decodes at T == 1 for the rest of its life:
  the draft would lack the auxiliary states of its batched positions.

**B6. The knobs and the census.**
- `E4B_PAGED_SPEC` is `off` (the default, also when unset) or `eagle3`, with `E4B_PAGED_SPEC_HEAD` the head's directory.
- `E4B_PAGED_SPEC_K` is the fixed k, 1–3. There is no adaptive k.
- `/health` reports the speculative census: steps, drafted, accepted, τ_live, the verify bucket and its replays, and
  the number of requests that dropped to T == 1.
- **Off means the shipped default.** With `E4B_PAGED_SPEC` off nothing is installed: no hook, no draft, no buffer. If the
  append gains a per-row offset, its zero-offset form is bitwise its old form (a grouped-nf4-gemm test). E's OFF arm
  is therefore the shipped server at the launch commit.

**Not built (the maintainer's exclusions):** speculation for batched requests; an adaptive k; sampling, since the
server is greedy only; any change to a default. A default change would be its own later lane.

**Tests (CPU, CI).**
- **Stream identity.** A toy model whose arithmetic does not depend on the row count must emit, with speculation on,
  the stream it emits with it off, for k = 1–3. Drafts are forced right, wrong, and mixed, across stop ids and
  `max_tokens`.
- **Rollback.** Lengths after an accept of 0..k on every pool layer, with no host read.
- **The batching transition (the maintainer's requirement).** A speculating request becomes batched mid-flight. Its
  draft state is dropped, the census counts it, its lengths and positions are consistent, and its T == 1 stream after
  the transition is the stream a never-speculating run emits from the same state.
- **The draft.** The in-engine chains equal `sd1_eagle3.chain_at` (CPU, fp32) on fixed inputs.
- **Scheduling.** Engagement only at one decoder and no pending prefill; several tokens in one emit; a stop id mid-chunk.
- **The dry run** (SD1's pattern): the runner's own text in CI with every heavy step stubbed, which must reach success,
  plus two mutants that must fail.

**The CUDA smoke, before the build merges.** The build PRs change the serve path, and CPU review cannot see CUDA-only
behaviour. An RTX A2000 is sm_86 and cannot run the FP8 paged pool. So the reviewed build PRs are smoked together, at
their heads, on one short RTX 5090 proof (`sd2-prove-N`), before they merge. It runs:
- stage V's correctness gate, V0, with its three mutants (below), and no timing;
- the batching transition on CUDA;
- every census engaged.

The maintainer's conditions:
- **This registration merges first.** The merged-registration rule is about the registration, so a stacked-head proof
  does not break it.
- **One integration commit.** The proof runs one pushed commit, the build heads merged onto main. Its manifest pins
  that commit and each PR's head SHA.
- **The override is explicit.** The launcher's override for running unmerged build code is explicit, and the receipt
  records it.
- **Correctness only.** Its numbers are never quoted as speed.
- **Merge at the proven heads.** The build PRs merge only at the heads the proof ran. A later change to package code
  needs a new proof; a change to docs or tests alone does not.

## The run (`sd2-5090-N`): one RTX 5090, stages V, E and Q, in that order

**The target:** e4b and grouped-nf4-gemm at Amendment 1's pinned commits; `Qwen/Qwen3-30B-A3B` at `ad44e77`, its NF4
arena baked on the box; the head as SD1 pinned it. The server is the shipped default: `E4B_PAGED_MAX_SEQS=16`, buckets
1–16, decode graphs on, all-vram, every fusion knob and kernel route at its default.

**Workloads,** SD1's, at SD1's bytes: R (16 wikitext rows, 512 prompt tokens), C-think and C-nothink (16 UltraChat
prompts, `enable_thinking` on and off).

### Stage V: the verify step's cost (first, and it can refute the lane)

**V0, the correctness gate, before any timing.** It runs on the speculative build with speculation on, on R row 0 and on
one C-think row:
- **Capture, bitwise.** Each verify bucket's replay at k = 1, 2, 3 equals its padded eager step, logits bit for bit:
  the runner's standing oracle for a replay.
- **Addressing, against the sequential oracle.** S2-lite's construction, at 16 positions on each gate row:
  - k + 1 T == 1 steps, a rewind, then one verify step fed the oracle's tokens as its drafts;
  - then 4 T == 1 steps after the rollback.

  Argmax agreement with the oracle must be ≥ **0.90** over all verified rows, and over the continuations. A wrong
  stagger, length or position reads the wrong context and collapses agreement. Different arithmetic only moves
  near-ties: T > 1 runs K25 and the dense GEMMs at M = k + 1, where T == 1 runs the bandwidth GEMV. S2-lite and S3
  read 17 / 17 at K = 16. Agreement below 1 is reported.
  Beside agreement, each row reports the mean |Δ log p| of the oracle's token.
- **The draft.** The in-engine chains equal `sd1_eagle3.chain_at` on the gate's captured states at ≥ 99 % of drafted
  ids.
- **The batching transition** on CUDA: no fault, consistent lengths, the draft state dropped.

A failure is VOID, and no timing is read.

**The gate must show it can fail** (the maintainer's condition). Three addressing mutants run on the proof box, and
each must fail V0's addressing check:
- **(a)** the stagger one low, so row i reads base + i;
- **(b)** the alias append one position low;
- **(c)** the verify rows' RoPE positions shifted +1.

If any passes, the 0.90 gate is too weak, and Amendment 1 tightens it before the read. A logit-level check against the
batched arithmetic's floor is one option. (c) is the likeliest to slip, since a uniform shift moves attention only
slightly.

**V1, timed.** On a live slot after R row 0's prompt, at moving positions, each step a graph replay:
- the T == 1 decode step (bucket 1), the anchor, twice for an A/A;
- the verify step at k = 1, 2, 3;
- the draft chain at k = 1, 2, 3;
- the loop's own cost per step: the accept, the length write and the host read.

Each is the median of 64 replays, by CUDA events.

**The verify-cost reading.** Per k, c(k) = verify(k) + draft(k) + loop(k). The measured-cost speedup on workload w is
S_V(k, w) = τ_SD1(k, w) × anchor / c(k), with SD1's measured τ. **VERIFY_COST_REFUTES** if the best S_V over k is below
**1.00** on every workload.

**The stop rule (the maintainer's):** on VERIFY_COST_REFUTES, stage E still runs, as the direct check of the model, and
is reported, not ruled. Stage Q is skipped, since nothing ships.

### Stage E: end to end

Each arm is a fresh server process:

| arm | `E4B_PAGED_SPEC` | `E4B_PAGED_SPEC_K` |
|---|---|---|
| OFF | `off` | -- |
| ON1 | `eagle3` | 1 |
| ON2 | `eagle3` | 2 |

**Order:** OFF-a, ON1-a, ON2-a, ON2-b, ON1-b, OFF-b. A palindrome, so drift lands on every arm.

**Per workload,** each request runs alone on the 16-slot server, W1's shape. Each arm runs one untimed warm pass, then
3 timed passes at SHORT 32 and LONG new tokens (LONG = 160 on R and 256 on C, SD1's lengths). Every request runs to
its length (`ignore_eos`).
- **Decode throughput** is p37's slope between SHORT and LONG.
- **Ratios:** g_k(w) = mean of ON_k-a / OFF-a and ON_k-b / OFF-b, decode tok/s. The block interval is the pair.

**Engagement** (from `/health`'s census, per ON arm):
- every decode step of a lone request speculative, and the verify bucket replayed, never eager;
- τ_live and acceptance per workload, reported against SD1's.

**The identity pass** (untimed, LONG): each arm's tokens per row. ON against OFF agreement and the first step they part
are reported, never gated. They are not bitwise by construction (see V0).

**The tripwire** (reported): OFF's R row 0 against `p127-5090-1`'s W1 tokens (`bench/sd1/expect_w1.json`). A
difference names a default that changed between `7f044dd9` and the launch commit, not a fault of SD2's.

### Stage Q: quality at the verify shape

P115 Phase B's instrument (`p115_quality.measure_phase`, at its registered bytes), on the default server built eager as
P110 built it (V0 holds the replays to the eager step bitwise). It differs from P121's run in two ways: one window a
pass, and the ON phases' scored decode steps at the verify shape.
- **The texts:** wikitext-2 and c4val1, 48 windows each, 512 prompt tokens and 128 teacher-forced positions per window,
  one window a pass.
- **The phases:**
  - **R**: T == 1, bucket 1. Its fp32 log-probs are saved.
  - **The floor:** `rep` (R again), and `chunk` (prefill in 256-token pieces).
  - **`mutant_scale`**: P108's halved decode scale. It must fail.
  - **ON1 and ON2**: every scored position comes from a verify step of k + 1 rows, teacher-forced: the true next k tokens
    are the drafts, all accepted, so each step scores k + 1 positions.

**The bar,** P110's, as P121 applied it. On each text and for each k:
- mean d_ON ≤ B_floor + 0.01 nats, and
- mean |d_ON| ≤ 2 × max(S_floor, 0.005).

**Reported beside the bar:** a `w16` draw (16 windows a pass, the batched arithmetic the shipped default already
serves), KL, argmax agreement, max |d| and each text's perplexity move.

## The rule (`bench/sd2/sd2_reduce.py`, self-tested)

The first that applies is the verdict.

1. **VOID**, any of:
   - a record is missing;
   - another commit, model revision or head digest;
   - the server not the shipped default: `max_seqs` 16, its buckets (1, 2, 4, 8, 16, and 3 under k = 2) not all
     captured;
   - V0 failed;
   - an ON arm never speculated, or replayed its verify bucket eagerly;
   - **the mechanism check:** τ_live(k, w) below τ_SD1(k, w) − 0.20 on any arm and workload, the size of drop a stale
     auxiliary state or a wrong position gives;
   - Q's `mutant_scale` passing the bar, or Q's phases scoring different windows.
2. **VERIFY_COST_REFUTES** (stage V, above). E is reported beside it.
   - **The V_NOISY flag.** If V's two anchors differ by more than 3 %, the flag is raised and this rule cannot fire. The
     read goes on from rule 3, with Q run, because the stop rule applies only on VERIFY_COST_REFUTES.
   - **The conflict.** If this rule fires while E's g meets FASTER's speed bar (rule 5's), the read names the conflict:
     the model is in question, not E. Nothing is licensed, because Q did not run.
3. **NOISY:** OFF-b / OFF-a, ON1-b / ON1-a or ON2-b / ON2-a outside [0.97, 1.03] on any workload.
4. **QUALITY_FAIL:** neither ON1 nor ON2 passes the bar on both texts. A k that fails on either text cannot be
   licensed, whatever its speed.
5. **FASTER:** for some k that passed Q on both texts, g_k ≥ **1.05** on at least two of the three workloads, with both
   of that workload's pair ratios above 1.
6. **NOT_FASTER** otherwise.

**Reported beside the verdict:**
- every g with its pairs, and ms per emitted token;
- V's terms and S_V against SD1's modelled S, per k and workload: the model's error, measured;
- τ_live against τ_SD1;
- ON-against-OFF token agreement;
- every Q arm's statistics;
- peak allocated and reserved memory per arm, with ON − OFF.

Floats are summed with `math.fsum`.

## Predictions (registered before the build)

| | prediction | basis |
|---|---|---|
| Q1 | verify(k = 1) / anchor in [1.10, 1.50] | SD1's model gives 1.32. Consecutive tokens share experts (S2's 41.5 against ~78), which pulls it lower; the T > 1 path's fixed cost (S3) pulls it higher |
| Q2 | verify(k = 2) / anchor in [1.20, 1.80] | the model's 1.63, the same two pulls |
| Q3 | the draft at 0.20–0.60 ms per drafted token | about 0.42 GB read per token: the 64,000-row `lm_head`, the layer and `fc` |
| Q4 | V does not refute: the best S_V ≥ 1.00 on at least C-think | SD1's S 1.23 there, with room for the loop |
| Q5 | τ_live within ±0.10 of τ_SD1 at k = 1 on every workload | the same head, target and prompts; the streams differ only where T > 1 arithmetic moves an argmax |
| Q6 | Q passes at k = 1 and 2 on both texts | the verify arithmetic is the batched arithmetic P121 licensed at W16 |
| Q7 | g at the best k: C-think in [1.00, 1.25]; C-nothink in [0.98, 1.20]; R in [0.90, 1.10] | SD1's S less the loop's cost, which the model does not price |

**The verdict, as forecast:** FASTER about 45 %, NOT_FASTER about 35 %, VERIFY_COST_REFUTES about 15 %, anything else
about 5 %.

## Budget

- **The proof** (`sd2-prove-1`, the CUDA smoke): one RTX 5090 at ≤ $0.85/h, a guard of 0.75 h, about 62 GB of
  download. About $1.10, **ceiling $1.25** (the maintainer's).
- **The read** (`sd2-5090-N`): one RTX 5090 at ≤ $0.85/h, a guard of 1.75 h, the same download. About $2.20.

About $3.30 across the two, within the maintainer's tier (bus, 2026-10-10T06:55:22Z). Amendment 1 firms the read's
ceiling. Each launch waits on the maintainer's ACK and a relay on #1313.

## What this lane cannot say

- **Nothing about batch > 1.** Speculation runs only for a request decoding alone.
- **Nothing about sampling,** other families, other heads or other cards.
- **Nothing about an adaptive k,** or about a default. FASTER licenses an opt-in at the k that passed; a default is its
  own later lane.
- **Nothing beyond its texts and workloads.** Q's bar holds on wikitext-2 and c4val1 at the verify shape; E's speed holds
  on SD1's three workloads.

## The consequence

- **FASTER:** `E4B_PAGED_SPEC=eagle3` is documented as a licensed opt-in at the faster k, with a claims row for the
  measured g. The default stays off.
- **NOT_FASTER:** the build stays, off and documented as not faster. The measured terms and the model's error are the
  next lane's input.
- **VERIFY_COST_REFUTES:** the verify step's measured cost and E's direct check close EAGLE-3 at B = 1 for this
  model, card and stack. The build stays off.
- **QUALITY_FAIL:** nothing ships at the failing k. The arithmetic gap is reported.

## Amendment 1 (2026-10-10): the CUDA proof `sd2-prove-1` -- its target and its harness

Registered before the proof. The maintainer reviewed the build PRs and approved them for the stacked proof (bus,
2026-10-10T07:17-07:38Z). This amendment registers what the proof runs, on what, and how it is read. The read
(`sd2-5090-N`) is unchanged by it; its harness and the merged commits it pins are Amendment 2's.

**The target** (not on main, the maintainer's explicit override):
- e4b at the integration commit `539a2d2694096a81c6b272b335e26d818647cd95`. That is main `aa47af8b` with the build PRs
  at their reviewed heads, stacked:
  - #1553 `461272ee`, #1554 `5825c13d` and #1556 `f9c1f867` (merged into `sd2/stack` `5c11d58e`);
  - #1558 `4737cb55`;
  - #1559 `539a2d26`.
- grouped-nf4-gemm v0.45.0, `724ccc454f006c1a46836e434e997f31f293747f`, CI's pin. The build changes no kernel.
- `Qwen/Qwen3-30B-A3B` at `ad44e77` and the head as SD1 pinned it, its `config.json` fetched beside its weights.
- The server is the shipped default with `E4B_PAGED_SPEC=eagle3`, `E4B_PAGED_SPEC_K=3` and graphs on. Every other knob
  is unset, with `E4B_PAGED_MAX_SEQS=16` and `E4B_INT4_TILE_PROGRAMS=1` as SD1 ran it. k = 3 captures every verify
  bucket the read can use (2, 3, 4).

**The override is recorded.** The runner holds the integration commit as a constant, and the receipts carry it in
`summary.txt`'s `KNOBS e4b_target=` line. The relay on #1313 that precedes the launch names it, with each PR's head.

**The harness** is the launch commit, this amendment's merge, checked out by SHA beside the target and never installed:
- `bench/sd2/sd2_run.sh` (box) and `sd2_drive.sh` (controller), derived from SD1's by named substitution;
- `sd2_box.py --prove` and `sd2_reduce.py --prove`, each with a self-test;
- `staged.sha256`;
- `tests/test_sd2.py` and `tests/test_sd2_dryrun.py`. The dry run runs the runner's own text with every heavy step
  stubbed and must reach success. A misnamed function, a head digest that does not match, and a size check that does
  not follow the cache symlink must each fail it.

**What the proof runs, in order** (correctness only; nothing is timed, and nothing it prints is ever quoted as speed):
1. **The target's own GPU tests**, on the card that CI cannot provide: `tests/test_spec_decode_gpu.py` and the build's
   CPU suites with the decode-graph and step-select suites. None may skip for want of the card.
2. **`sd2_box.py --prove`:**
   - **the census:** buckets 1, 2, 3, 4, 8 and 16 and the post-verify graphs 2, 3 and 4 all captured; three hooks;
   - **capture, bitwise:** each verify bucket's replay against its padded eager step, logits bit for bit;
   - **V0 addressing:** S2-lite's construction at 16 positions on R row 0 and on C-think row 0, at k = 1, 2, 3, with
     each verify row's mean |Δ log p| of the oracle's token reported;
   - **the three addressing mutants** (the maintainer's): (a) the stagger one low, (b) the alias append one position
     low with the reads restored, and (c) the verify rows' RoPE positions shifted +1, each at 8 positions, k = 3;
   - **the draft:** the in-engine drafter against `sd1_eagle3.chain_at` on the states the served prefill captured, at 8
     truncation points of 16 R and 16 C-think prompts (768 drafted ids);
   - **the transition:** R row 1 speculates alone, and R row 2 is admitted mid-flight. One drop is counted, both
     requests finish at their lengths, the device lengths equal the host mirror after every plain step, and no state is
     left.

**The proof's verdict** (`sd2_reduce.py --prove`, self-tested on 17 cases). It is **PROVED** when every item holds:
- the census;
- the capture, bitwise;
- V0 addressing at ≥ 0.90 on the rows and on the continuations, for every (row, k);
- every mutant caught by V0's own rule: its row agreement OR its continuation agreement below 0.90. A RoPE shift writes
  rotated keys that the continuations read, so the continuations are where (c) most likely shows (the maintainer's);
- the draft at ≥ 0.99;
- the transition;
- the GPU tests.

Otherwise it is **FAILED**, with every failing item named. A mutant at or above 0.90 on both its rows and its
continuations is named `GATE_TOO_WEAK:<m>`.
Then, as registered, Amendment 2 tightens V0 before any read, for example with a logit-level check against the
batched arithmetic's floor.

**The budget.** One RTX 5090 at ≤ $0.85/h, a guard of 0.75 h, about 62 GB of download: about $1.10 expected, with a
ceiling of **$1.35**. That is the worst case at the policy caps (0.75 h × $0.85 + 62 GB × $0.011 = $1.32), which the
maintainer raised from $1.25 so that admission does not refuse an in-policy offer. The launch waits on the maintainer's
ACK and the relay on #1313.

**After PROVED:**
- The build PRs merge in order (1, 2, 3, 4, 5) at their proven heads; a stacked PR is retargeted to main as its base
  merges.
- A later change to package code needs a new proof, and a change to docs or tests alone does not.
- Amendment 2 then registers the read's harness (stages V, E and Q, and the rule above in `sd2_reduce.py`) and pins
  the merged commits.

## Amendment 1b (2026-10-10): the proof's time budget, after `sd2-prove-1`

**What happened.** `sd2-prove-1` (vast instance 55178430, **$0.046**, teardown complete) ended as **HARNESS_ERROR, rc
40**: STOP-2 before the fetch.
- **The box was healthy.** The tripwire passed, so the target imported with SD2's build (e4b 0.52.0 at `539a2d26`).
  All four self-tests passed.
- **The defect.** The runner kept SD1's fetch budget (2100 s plus the 600 s margin) under a 0.75 h guard, which the
  installs had already eaten into. This is P109's trap (`p109-prove-1`), and the review should have asked for the guard.

**The fix:**
- **Per-mode checks.** Every time-left check is a per-mode variable (`SD2_MODE`; `prove` only until Amendment 2).
  `tests/test_sd2.py` holds P109's rule: every check plus its 600 s margin fits its mode's guard after 15 min of install,
  and the checks use exactly those variables, in order. A second test walks a slow host through every check.
- **The guard, from the real steps.**

  | step | time |
  |---|---|
  | installs, clones, tripwire and self-tests | about 10 min |
  | the build's GPU tests | about 5 min |
  | the checkpoint fetch | 9 min on `sd1-5090-2`'s host, 28 min on `p127-prove-3`'s |
  | the head, the bake and the prompts | about 6 min |
  | the proof | about 10 min |

  - **At 1.0 h,** each check passes P109's per-check rule. But a 28-minute fetch puts the bake's check past the
    deadline, a STOP-2 after the download has been paid for.
  - **At 1.25 h,** every check passes on the slow host (`test_the_step_budget_fits_the_registered_guard`).

  So the guard of 1.25 h is SD1's. The ceiling is 1.25 h × $0.85 + 62 GB × $0.011 = $1.0625 + $0.682 = **$1.75**. The
  expected cost is unchanged, at about $1.10.
- **A short deadline refuses at once.** A launch whose deadline leaves less than the guard minus 15 min refuses before
  anything is installed (rc 17).
- **The GPU tests come first.** They need no checkpoint, so they now run before the 62 GB fetch. A failure stops the
  lane there (rc 24) with `gpu_tests.json` kept, and the proof is FAILED on `gpu_tests`.

**Unchanged:** the target `539a2d26` (the builds are untouched, so no re-proof question arises), every gate and its
rule. The next run is `sd2-prove-2`, and it launches on the maintainer's ACK and a new relay on #1313.

## Amendment 2 (2026-10-10): V0's addressing gate is a logit gate, after `sd2-prove-2`

**What happened.** `sd2-prove-2` (vast instance 55184960, **$0.66**, teardown complete) ran in full. Under Amendment 1's
rule its verdict is **FAILED** with one item, `GATE_TOO_WEAK:c`. The maintainer re-derived it with main's reducer and got
the box's verdict exactly. That verdict stands.
- **Every other item held:**
  - the GPU tests: 123 passed, none skipped;
  - the census;
  - the capture, bitwise at 2, 3 and 4 rows;
  - V0 addressing on the real build;
  - mutants (a) and (b);
  - the draft: 766 of 768;
  - the transition.
- **What failed was the gate's sensitivity, not the build.** Mutant (c), the RoPE shift, kept argmax agreement at
  0.94 on its rows and 0.91 on its continuations: the 0.90 agreement bar cannot see it. Its row 0's mean |Δ log p| was
  1.18 nats, against at most 0.037 anywhere on the real build.

**The tightened gate** (`sd2_reduce.py --rule a2`, now the default):
- **The logit gate decides.** V0 passes a (row, k) when the largest per-row mean |Δ log p| of the oracle's token (the
  verify step against the sequential T == 1 oracle) is at most **0.25 nats**. A mutant is caught when its largest
  exceeds 0.25, and is `GATE_TOO_WEAK` otherwise.
- **Argmax agreement is reported, no longer gated.** On chat the real build's continuations sat at 0.906 (C0, k = 3),
  where near-ties flip, so the agreement bar had no margin left.
- **The bound is calibrated on `sd2-prove-2`.** 0.25 is 6.8× the real build's largest value (0.037) and 4.7× below the
  smallest mutant maximum, (c)'s 1.18. Mutant (a)'s maximum is 2.79 and (b)'s 5.34.
- **The registered verdict stays reproducible.** `--rule a1` keeps Amendment 1's rule, so `sd2-prove-2` re-derives as
  registered: FAILED, `GATE_TOO_WEAK:c`. Under `a2` the same receipt reads PROVED. That is the calibration, not an
  out-of-sample test, and the self-test pins both.

**Out of sample: the read's V0 runs the mutants again.** Because the bound was calibrated on `sd2-prove-2`, the read's
stage V0 (Amendment 3's harness) runs the three addressing mutants on its own box, after the real build's V0 and before
any timing:
- each at 8 positions at k = 3, about a minute of the rental;
- each must exceed the bound;
- a mutant inside the bound makes the read **VOID** before anything is timed.

**No fresh proof is proposed.** The build failed no item of `sd2-prove-2`; what failed was a property of the
instrument, which the read re-checks on fresh data before it times anything. This is the maintainer's to decide; a fresh
proof would cost about $0.66 at the same target.

**What follows:**
- The build PRs merge after this amendment's review, in order and at their proven heads.
- Amendment 3 then registers the read's harness (stages V, E and Q, the mutants in V0, and the rule above) and pins the
  merged commits. Amendment 1's text calls the read's harness "Amendment 2"; it is now Amendment 3.
