# SC1b: where each engine's decode step goes. A per-kernel census of SC1's largest loss and largest win on one RTX 5090 (lane SC1b of #846)

Registered before any SC1b rental. The design took two hostile review rounds, both in `DESIGN-REVIEW.md`: R1–R18 are round
1's findings and the B/M/m items are round 2's. Engine facts come from source reads at vLLM `ced6857`, SGLang `94602c9c`
and llama.cpp `552f18f`, with file:line citations in `UPSTREAM-NOTES.md`, and from NVIDIA's Nsight Systems and CUPTI
documentation (the 2025.3 and 2025.6 user guides; the sqlite schema is per 2025.3).

**Code** (CPU-tested in `tests/test_sc1b.py`, plus SC1's pin and shape tests):

| file | role |
|---|---|
| `sc1b_census.py` | the reducer, with a 29-check self-test |
| `kernel_classes.json` | class map v0, frozen |
| `sc1b_e4b_census.py`, `sc1b_vllm_census.py`, `sc1b_serve_census.py` | the brackets and the server client |
| `sc1b_toy.py` | proof item 2 |
| `sc1b_box_d.sh` | box D, sourced by `bench/sc1/sc1_run.sh` when `SC1_BOX=D` |

SC1's own scripts gain only:
- box D's dispatch points;
- an empty-by-default `SC1_LAUNCH_PREFIX` on the SGLang and llama.cpp server starts, and `LLAMACPP_EXTRA_FLAGS`;
- the pin `E4B_PAGED_PREFILL_ATTN=math`, exported after the scrub and asserted by the e4b tripwire;
- `census/` excluded from mid-run pulls.

The prefill-attention pin is the route every SC1 box ran: #960 added the knob, and its default flips to flash afterwards.
Every SC1b file is staged and pinned on every box.

## Why this lane

SC1's decision rule: "A comparator faster than e4b at COMPARABLE quality is a loss with its cause named by SC1b's census
before any README sentence changes."
- SC1 quotes no position, because box A's licence is QUALITY_FAIL on three draws.
- Its measured, labelled ratios are not in doubt: every Marlin/GGUF engine decodes faster than e4b's scheduler at B=1, at
  CLOSE quality.
- SC1b names where the per-step time goes. It moves no gate, licence, kernel default or claim id.

SC1's per-step decode times (ms; medians over draws; `bench/h2h-2026-10-02/sc1/`):

| engine (SC1 arm) | B=1 | B=16 |
|---|---|---|
| llama.cpp `552f18f` Q4_K_M | 3.03 (box B) | 15.44 (B) |
| vLLM 0.30.0 GPTQ Marlin + graphs | 3.74 (B) / 4.09 (C) / 3.88 (A3) | 8.21 (B) / 8.82 (C) / 8.36 (A3) |
| SGLang 0.5.20 GPTQ Marlin, matched | 3.76 (C) | 8.71 (C) |
| e4b scheduler `int4_sched` / `lic_sched` | 4.50 (B) / 4.77 (C) / 4.60 (A3) | 9.63 (B) / 10.38 (C) / 9.76 (A3) |

The gaps read:

| gap | engines | B | SC1 size |
|---|---|---|---|
| G1, the largest loss | llama.cpp − e4b | 1 | ≈ 1.47 ms/step |
| G2, the largest win | e4b − llama.cpp | 16 | ≈ 5.8 ms/step |
| G3 | vLLM − e4b | 1 and 16 | |
| G4 | SGLang − e4b | 1 and 16 | |

ExLlamaV3 is out: UNSTABLE on box B, confounded on box C.

What SC1 already knows: e4b's bare graph replay runs 4.21–4.32 ms/step against the scheduler's 4.50–4.77 at B=1, so its B=1
step is mostly graph-bound. The open question is what inside the graph is slow.

## Arms (box D: one box; SC1's stack, checkpoints, prompt sets and token ids)

| engine | config (SC1's arm, unchanged) |
|---|---|
| e4b | `int4_sched` (RTN), gnf4 pinned `34da93d`, int4 prefill route `loop` (A13), paged prefill attention `math` |
| vLLM 0.30.0 | `gptq_graph`, Model Runner V2, FULL decode graph, `VLLM_WORKER_MULTIPROC_METHOD=spawn` in every pass |
| SGLang 0.5.20 | `gptq_matched`; SC1's exact non-streaming batch `/generate` (`sc1_sglang_arm.sampling_params`) |
| llama.cpp `552f18f` | Q4_K_M, `-ngl 99 -fa on -lv 4 -np B -c B×1024` (SC1's), built with the image's nvcc 13.0 for `120a-real`; requests from `sc1_llamacpp_arm.completion_request`, released through SC1's barrier |

- Every engine runs at B ∈ {1, 16}.
- **Image:** box C's `nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04`, the only image that holds all four. llama.cpp's build
  differs from SC1 box B's nvcc 12.9 build, and that is recorded.

Each (engine, B) gets three processes:
1. **unprofiled:** SC1's own arm, one draw. This is the census box's speed and G-inflate's reference.
2. **graph-mode capture:** `nsys profile --cuda-graph-trace=graph`.
3. **node-mode capture:** `nsys profile --cuda-graph-trace=node`.

SC1's sampler (SM clock, power, temperature) runs on every pass. The run order is e4b → llama.cpp → vLLM → SGLang, B=1
before B=16, graph before node. The deadline drops arms from the end.

## The instrument

**nsys:** `nsight-systems-cli-2025.6.1` from NVIDIA's devtools apt repository, called by its dpkg path (the image's PATH
`nsys` is Nsight Compute's 2025.3.1).
- Every capture is a plain `nsys profile -t cuda-sw,nvtx --sample=none --cpuctxsw=none`. There are no interactive sessions:
  2025.6.1 does not document `start --session` (M1).
- `cuda-sw`: NVIDIA lists node-mode overhead under hardware trace on many-kernel graphs, and a third-party report (SPECULA
  PR #778) lost graph launches under `cuda-hw`.
- No fork-before-exec: the workers are spawned and nsys follows exec'd children.

**Brackets.** The registered window is **decode steps 34–97 of a 160-token generation**: K = 32 steps after the first
full-batch decode step, then N = 64 steps.
- **e4b:** `nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop` around `sc1b_e4b_census.py`.
  - It steps the scheduler with `step()`, the exact per-step body of `run_until_idle`.
  - Rows carry 160 tokens. The driver ramps to the first full decode-only step, takes 32 more, then brackets 64.
  - It **refuses before opening the range** if the first-admitted row would retire inside the window (B1: serve_paged admits
    one 512-token prefill chunk per step, so at B=16 rows are staggered by up to 15 tokens).
  - Per-row decode positions are recorded.
- **vLLM:** the same `nsys profile` flags around `sc1b_vllm_census.py`, with
  `profiler_config={"profiler":"cuda","delay_iterations":35,"max_iterations":64}`. Call 1 is the prefill and call k is
  decode step k − 1, so the window is decode steps 34–97.
- **SGLang:** each (B, mode) gets its own server under
  `nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop`.
  - The client posts SC1's batch, then at t = request + pass-1 prefill + 32 × pass-1 ms/step posts
    `/start_profile {"activities":["CUDA_PROFILER"],"num_steps":64}`.
  - The reply is plain text and must be HTTP 200 reading "Start profiling" (B2).
  - The window's positions are estimated from the clock and recorded, not enforced.
- **llama.cpp** (no hook): the server runs under plain `nsys profile`, so **the whole 160-token run is captured** (B3). The
  reducer selects intervals 33–96 between full-batch logits copies (`bytes = B × 151,936 × 4`). That is decode steps 34–97
  counted from the first full-batch step, ±1.
- **Every capture ends** with the app exiting, or SIGTERM to the app (never to nsys). nsys then exits and writes its report,
  and the GPU is confirmed empty before the next arm (M3).

**Steps and the floor:**
- A step is one replay of the steady decode graph (modal `graphExecId`), or for llama.cpp one interval between full-batch
  logits copies that contains exactly one replay.
- Dropped and counted: a step with 0 or ≥ 2 replays, a period over 1.5× the median, or more out-of-graph kernels than the
  modal count.
- Bracketed captures trim one step at each edge.
- **Fewer than 56 kept steps makes the arm VOID** (M8).
- Out-of-graph work is `graphNodeId` NULL **or 0** (M7).
- Every figure is a median, with the per-step interquartile range.

### Per-step terms (ms)

| term | definition | mode |
|---|---|---|
| P | period, delimiter to delimiter | graph |
| S | graph span (`GRAPH_TRACE` start..end) | graph |
| D_out | every non-graph device interval in the period, on any stream, as a union with S, minus S | graph |
| idle_out | P − \|S ∪ non-graph intervals\|: device idle outside graphs (host/CPU work, sync and launch latency outside the graph) | graph |
| U_in | union of a replay's in-graph records: kernels, copies and memsets (M5) | node |
| I_in | **in-graph idle** = S − U_in: launch and dependency gaps inside the graph | mixed |
| I_in_node | node-mode replay span − U_in, reported beside I_in | node |
| class_c | summed durations of class c, in-graph and out-of-graph | node |
| O | Σ class + I_in + idle_out − P: overlap, median non-additivity, and the out-of-graph node/graph-mode mismatch | mixed |

By construction: **P = Σ_c class_c − O + I_in + idle_out**. Copies count as device time even when the SMs are idle.

### Gates (per arm)

| gate | condition | if it fails | effect |
|---|---|---|---|
| G-inflate | graph-mode median P within ±5 % of pass 1's **median-of-3** slope (M9) | `PROFILER_INFLATED` | idle_out not nameable |
| G-node | median U_in AND median node-mode replay span both ≤ median S × 1.02 (M10) | `NODE_TRACE_INFLATED` | blocks |
| G-map | unmatched (residual) share ≤ 2 % of node-mode device time | `CLASS_MAP_INCOMPLETE` | blocks |
| G-segment | every kept replay opens and closes exactly 48 MoE segments (M6) | `CLASS_MAP_SEGMENT_BROKEN` | blocks |
| G-diag | nsys `DIAGNOSTIC_EVENT` carries no error, dropped, lost, overflow, buffer-full or failed text (R17) | `NSYS_DIAGNOSTIC_ERRORS` | blocks |
| G-clock | median SM clock across the arm's three passes within 3 % (R18) | `CLOCK_MISMATCH` | label only |

A gap is read only when neither arm carries a blocking label.

**Struck** (round 2 M4):
- Control C1: vLLM's and SGLang's Marlin MoE are different instantiations (atomic-add and zero-fill differ), so they are not
  a fidelity control.
- The Phase-0 symbol check is replaced by the proof's real-kernel residual gate.

## The class map (`kernel_classes.json`, frozen before any capture)

Classes:

| class | contents |
|---|---|
| moe_expert | expert matmuls and their activations |
| moe_route | top-k, sort, align, gather, scatter, combine, and any other kernel inside a MoE segment |
| attn | attention and KV append |
| dense_gemm | q/k/v/o, the router gate GEMM and the lm_head, on every engine alike |
| norm_elem | norms, residual adds, standalone rotary, and **generic ATen glue inside the graph** |
| sample | sampling |
| input_prep | slot/position/table prep, and **generic glue outside the graph** (including CUB scans) |
| memcpy | copies and memsets |
| residual | anything unmatched |

Rules, in order:
1. **name** (first match on the short or demangled name);
2. **grid** (e4b `_gemv_int4_b32`: gridY = R, so R = 1 is dense);
3. **inside/outside graph**;
4. **MoE segment by node order** (start-time order inside a replay): between a start anchor and an end anchor, `expert_names`
   are moe_expert and everything else is moe_route;
5. **inherit-next** (quantize kernels take the class of the next matmul) and **inherit-prev** (split-K reductions and
   stream-k fix-ups take the class of the previous matmul).

Segment anchors per engine:

| engine | start anchor | end anchor |
|---|---|---|
| e4b | `_router_epilogue` / top-k | `_combine_rows` |
| vLLM | `topkGating` | `moe_sum` |
| SGLang | `_router_triton_kernel` | `moe_sum_reduce` |
| llama.cpp | `topk_moe` / `soft_max` | `moe_weighted_reduction` |

Declared cross-class fusions:
- llama.cpp `rms_norm_mul_rope_f32` writes K into the cache; it is norm_elem.
- e4b `_rope_norm_heads` is norm_elem.
- Inductor `triton_*` kernels are norm_elem outside MoE segments.

**First read on the paid run:** e4b's map on Qwen3 (the proof reads it on Granite) and llama.cpp's node-mode map. A residual
over 2 % labels the arm.
- After capture, the only permitted change is to add a residual kernel name to a class by citing its source definition.
- Each such delta is listed. Every gap is computed with the frozen map and with the delta map, side by side.

## The reading: gap decomposition per G

**ΔP = Σ_c Δclass_c − ΔO + ΔI_in + Δidle_out**, all terms e4b minus the comparator, medians.

- **Named cause:** the largest-|Δ| nameable term (classes, I_in, and idle_out when G-inflate holds for both arms) that has
  ΔP's sign, carries ≥ 50 % of |ΔP|, and exceeds twice its noise.
  - Noise is the sum of the two arms' per-step IQRs: a spread, not a confidence interval.
  - ΔI_in reads "launch/dependency gaps inside the graph".
  - Δidle_out reads "host/CPU work and latency outside the graph".
- **UNEXPLAINED:** |ΔO| ≥ 50 % of |ΔP|.
- **spread:** neither of the above. The top three terms are listed with ΔO.
- The census box's unprofiled ratio is printed beside SC1's measured ratio:
  - llama.cpp: 1.484 and 0.624 (box B);
  - vLLM: 1.203 and 1.172 (box B);
  - SGLang: 1.268 and 1.191 (box C).

  This is a diagnostic, not a gate. The host CPU is recorded.

## Predictions (falsifiable; the rule's own quantities; written before any census exists)

- **Q1 (G1, B=1):** the largest term of ΔP(e4b − llama.cpp) is ΔI_in, launch/dependency gaps inside the graph.
- **Q2 (G1):** e4b runs ≥ 1.5× llama.cpp's in-graph kernels per B=1 step.
- **Q3 (G1, read only if G-map passes for both):** Δmoe_expert < ΔI_in. e4b's int4 expert GEMV is not the main cause.
- **Q4 (G2, B=16):** Δidle_out carries ≥ 50 % of |ΔP(e4b − llama.cpp)|: llama.cpp's CPU sampling of 16 × 151,936 logits
  after `llama_synchronize`. Its 9.7 MB logits copy is booked as memcpy, not idle.
- **Q5 (G3, B=1):** the largest term of ΔP(e4b − vLLM) is ΔI_in.

## Proof (`sc1d-prove-*`, guard 1.5 h ≤ $1.13)

The box's installs and SC1's tripwires, the common Granite smokes, and:
1. nsys installed; `nsys status -e` and `--version` recorded.
2. **The toy**, checked through the reducer's loader:
   - an `mp.spawn` child at ≥ 90 % GPU memory captures a 3-kernel graph BEFORE `cudaProfilerStart`, then replays it 20
     times, each replay followed by one eager kernel and one H2D copy;
   - graph mode: 20 `GRAPH_TRACE` rows and the `graphExecId` column;
   - node mode: exactly 3 kernels per launch, each with a `graphNodeId` and a non-zero grid;
   - both modes: the eager kernel and the copy land in the non-graph set.
3. **e4b B=16, graph mode, on Granite** (`GR_ENV`, `fuse_qkv` 0, as SC1's smokes): the staggered-admission bracket.
4. **vLLM B=1, node mode,** on the lane's GPTQ checkpoint.
5. **SGLang B=1, node mode** (the server under `nsys profile`, `/start_profile`'s text reply, the JIT).
6. **llama.cpp B=16, graph mode, whole run, `-lv 5`**, read at positions 33:64.

**Every capture in items 3–6 must REDUCE, not merely exit 0** (M11):
- status ok;
- at least 56 kept steps;
- no nsys diagnostic errors;
- node mode also needs residual ≤ 2 % and no broken segment.

If item 2 fails on a host, the proof relaunches once on another; if it fails twice, the instrument is UNSUPPORTED and SC1b
stops. If item 3, 4, 5 or 6 fails, its path is amended before the run.

## Budget

- Proof $1.13. Main box (`sc1d-5090-*`) guard 4.0 h ≤ $3.00 at $0.75/h, under the launcher's 6.0 h policy cap:
  - Phase 0: about 75 min.
  - 4 engines × 2 B × 3 passes, with SGLang's four nsys-launched servers: about 100–130 min.
- Total ≤ $4.13, or $5.26 with one proof relaunch. That is over the campaign's $2.5 SC1b line, inside the no-ask tier per
  run, and stated.

## Out of scope

Coverage families (their own lane), gpt-oss (SC1g), prefill/TTFT (#916's P100/P102), request-level serving (SC2).

## Amendments

- **A1 (before any box D data; the proof `sc1d-prove-1` was renting, box D not launched): the prediction evaluator.**
  Q1–Q5 had no code (round 2 M4 named it for Q1–Q4). `sc1b_read.py` decides each one from box D's own
  `sc1b_arm_*.json` / `sc1b_gap_*.json`, and re-reduces nothing. Three readings of the registered text are fixed here,
  before any data exists:
  1. **"The largest term of ΔP"** (Q1, Q5) is the term with the largest contribution in ΔP's direction:
     argmax_k sign(ΔP) × Δ_k over the nine classes, I_in and idle_out. The remainder O is not a term.
     - An opposite-signed term is not "of ΔP", however large.
     - If idle_out is not nameable (G-inflate) and is the largest, the prediction is decided on the other terms: a
       class beating I_in makes it REFUTED (whatever idle_out's true size), and I_in winning the rest makes it UNREAD.
  2. **Q2's kernel count** is the node-mode median `kernels_in_graph` per step. Q2 is read even when the class map is
     incomplete, but not when either arm's node capture is VOID or carries NSYS_DIAGNOSTIC_ERRORS.
  3. **Q4** carries no noise clause, as registered: same sign as ΔP and |Δidle_out| ≥ 0.5 |ΔP|, with idle_out
     nameable. The noise is printed beside it.

  Each prediction reads HOLDS, REFUTED or UNREAD. A prediction on an unread gap is UNREAD. The read's tables
  (`RESULTS-sc1b.md`) come from the same script. Box D's code is unchanged: A1 runs at read time only.
- **A2 (after `sc1d-prove-1`, before box D): the proof's e4b capture gets its own Granite rows.** `sc1d-prove-1`
  (Vast machine 142284, $0.4124, 47 min) read HARNESS_ERROR on proof item 3 alone.
  - The e4b Granite B=16 capture read the lane's `prompts_b16.json` before item 4 had written it
    (`FileNotFoundError`; nsys wrote no report).
  - That file carries Qwen3 token ids, which are beyond Granite's 49,155-token vocabulary anyway.
  - Items 2, 4, 5 and 6 passed and reduced:
    - the toy: 20 replays, 3 kernels each, eager work non-graph;
    - vLLM B=1 node: 57 steps, residual 0.11 %, 48/48 segments;
    - SGLang B=1 node: 61 steps, residual 0.12 %, 48/48 segments;
    - llama.cpp B=16 graph: 62 steps at positions 33:64.

  The fix:
  - `sc1b_e4b_census.py --write-prompts` writes 16 distinct 512-token rows of seeded random ids under Granite's
    vocabulary (read from the fetched checkpoint's `config.json`).
  - It uses SC1's own digest, and SC1's own `load_prompts` checks the file.
  - Item 3 writes the rows first and passes them to `e4b_census` (new optional prompts argument).
  - The main box's e4b arms keep `prompts_b{B}.json`.

  The proof re-runs as `sc1d-prove-2` before box D.
- **A3 (written AFTER box D's first run `sc1d-5090-2`, and disclosed as such): the instrument fixes that run showed, and a
  confirmatory re-run.**

  `sc1d-5090-2`: machine 45511, $0.7788, 74 min. Every one of its 24 passes exited 0. **Its registered reading stands
  as recorded:**
  - Q2 is REFUTED: e4b runs 1,550 in-graph kernels per B=1 step against llama.cpp's 1,110, a ratio of 1.396 (≥ 1.5 was
    predicted).
  - Q1, Q3, Q4 and Q5 are UNREAD.
  - One gap reads: G4 at B=16 (e4b − SGLang, ΔP +1.874 ms), named norm_elem (+1.007 ms).
  - Every other gap is unread.

  The read found three things blocking the instrument; the causes are in the read.
  1. **vLLM was VOID at both B (55 kept steps < 56).** The drop rule "more out-of-graph kernels than the modal step"
     removed vLLM's every-16th decode step: one extra `_apply_write_kernel`, a new KV block's table write, at a normal
     period. That is steady work. A stray prefill or re-warm step is already caught by the replay-count rule.
     **A3:** such steps are kept and counted (`kept_extra_out_of_graph`).
  2. **NSYS_DIAGNOSTIC_ERRORS on llama.cpp B=16.** The node capture carried 1,523 each of CUPTI's
     `GetGraphId(data.originalGraph…)` / `GetGraphNodeId(data.originalNode…)` INVALID_PARAMETER. These concern graph-id
     attribution for nodes of graphs llama.cpp updates in place, not lost records: every kept replay held the modal
     1,733 kernels and 48 balanced segments. **A3:** exactly these two messages (full-text regexes) label
     NSYS_GRAPH_ID_MAPPING and do not block. Every other DIAG_BAD text still blocks.
  3. **NODE_TRACE_INFLATED at B=1 on e4b, llama.cpp and SGLang.** Node replays ran 2.4–3.7 % past the graph span: tracing
     overhead of about 0.06–0.11 µs per kernel. **A3** replaces the 2 % gate with a bound and a robustness test, and
     keeps NODE_TRACE_INFLATED as an information label.
     - The bound: ω = node span − S limits how much node-mode kernel time can be tracing. dK ∈ [max(0, U_in − S), ω],
       and the true in-graph idle is S − U_in + dK.
     - Each gap carries a band per term: classes ±ω of each arm, I_in through dK, overlap ±(ω_e + ω_c); idle_out and O
       are unaffected.
     - A gap reads only if its reading (named cause, spread or UNEXPLAINED) is the same at every corner of the bands.
       Otherwise it is unread, labelled NODE_TRACE_AMBIGUOUS, with the nominal reading kept.
     - Q1, Q3 and Q5 are decided by interval dominance: HOLDS or REFUTED only when every corner agrees. A1's rules are
       the zero-width case.

  The read also explained G1's remainder. llama.cpp's in-graph kernel durations sum to 4.09 ms per B=1 step but cover
  2.74 ms: 1,060 of 1,109 consecutive kernel pairs per replay overlap, on ONE stream. That is about 1.35 ms per step
  hidden by same-stream overlap, the signature of programmatic dependent launch. e4b and vLLM overlap no pairs.
  **A3** makes this an explicit term:
  - overlap_in = −(Σ in-graph durations − U_in), with its own per-step noise and band;
  - O keeps only the out-of-graph node/graph mismatch;
  - the identity becomes P = Σ class + overlap_in + I_in + idle_out − O;
  - overlap_in is nameable.

  **Q1–Q5 keep A1's term set** (no overlap term), so the term added here cannot change what they mean.

  **Confirmatory predictions for the re-run.** These replicate what `sc1d-5090-2` showed, and are labelled as replications,
  not blind tests:
  - **Q6** (G1, B=1): the gap reads "named: overlap_in". e4b's loss to llama.cpp is the in-graph overlap it lacks.
  - **Q7** (G3 and G4, B=16): both read "named: norm_elem".
  - **Q8** (B=1): llama.cpp overlaps ≥ 90 % of consecutive in-graph kernel pairs on one stream (`sc1d-5090-2`: 95.6 %);
    e4b overlaps ≤ 1 % (0 %).

  The exploratory A3 reduction of `sc1d-5090-2` is reported in the read, labelled exploratory.

  **The re-run.** Box D re-runs as `sc1d-5090-3` at A3's merge, under the same 4.0 h guard (≤ $3.00).
  - A3 changes only the reducer (`sc1b_census.py`, run on the box at phase RD) and the read (`sc1b_read.py`).
  - Every capture path (`sc1b_box_d.sh`, the three bracket/client scripts, the toy, the class map and SC1's scripts) is
    byte-identical to `77469c1`. That is the commit `sc1d-prove-2` PROVED on, so no new proof is run. The launch chain
    refuses if any of those files differs.
