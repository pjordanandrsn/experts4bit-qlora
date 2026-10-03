# SC1b design review, round 1 (hostile), against `SC1b-PREREG-draft-v2.md`

Reviewed 2026-10-03. Sources checked:
- the draft and its three upstream notes;
- the nsys and CUPTI doc text under `../nsys/`;
- vLLM 0.30.0, SGLang 0.5.20 (`94602c9`) and llama.cpp `552f18f` source trees;
- e4b at e4b main (`43f36c7`) and grouped-nf4-gemm at SC1's pin `34da93d`.

Where the in-progress reducer (`bench/sc1b/sc1b_census.py`, then uncommitted; v2 below fixes what it flags) already implements a
defect, it is named as corroboration. The draft is unchanged.

Abbreviations: `D` = draft line, `V` = vLLM tree, `S` = SGLang tree, `L` = llama.cpp tree, `E` = e4b tree, `G` = gnf4 at `34da93d`.

What holds up:
- vLLM's lm_head and sampling sit outside the FULL graph (V `vllm/v1/worker/gpu/model_runner.py:1901-1909`, `:2063`).
- SGLang's `num_steps` profiles exactly N `run_batch` calls, and its predicate runs before the forward
  (S `scheduler.py:4252-4266`, `profiler_manager.py:147-153, 441-450`).
- llama.cpp's MUL_MAT_ID sync fallback is not taken at 16 tokens for Q4_K/Q6_K (L `ggml-cuda.cu:1875-1899`).
- None of the four decode paths turns kineto on: the e4b serve path has no `torch.profiler`, vLLM's "cuda" profiler is
  bare `cudaProfilerStart`, and SGLang's `CUDA_PROFILER` is the same.

The findings below are what breaks.

---

## BLOCKERS

### 1. BLOCKER: in-graph idle is counted as device-busy and has no term, so the most likely B=1 cause falls into an unnameable residual

**Defect.**
- Device-busy counts the whole `GRAPH_TRACE` span as busy (D:91). Host time is defined as the period minus that union
  (D:92).
- The idle gaps between dependent kernels inside a graph are therefore booked as busy. They are not host time, and they
  appear only in `Δoverlap_residual` (D:124-127).
- The decision rule names a cause only from a class or host term (D:142-143). The residual is never nameable.
- If e4b's B=1 loss is mostly launch and dependency gaps inside its graph, the census will say "spread" and list three
  smaller terms. That is the mechanism Q2's kernel-count prediction points at. The real cause stays in the residual.

**Evidence that this is likely, not hypothetical.**
- P89 removed 336 launches per step and saved 0.516 ms at B=16 (E `bench/p89/RESULTS-p89.md:9,15`: 1,652.4 → 1,316.4
  launches, 10.321 → 9.805 ms). That is about 1.5 µs per in-graph launch.
- If e4b runs a few hundred more kernels per B=1 step than llama.cpp (Q2 predicts ≥ 1.5×), that alone is about
  0.4–0.8 ms of the 1.47 ms loss.
- TC1 amendment 12 (`bb04477`) just ruled e4b's training step "launch-bound".

**Second defect: the recorded check cannot be interpreted (D:96-97).**
- It compares node mode's in-graph kernel SUM with graph mode's span.
- Overlap makes the sum larger than the union; gaps make the span larger than the union. The two effects partly cancel,
  so the ratio means nothing.

**Fix.**
- Add a named term, `in_graph_idle = graph_span (graph mode, median) − union of in-graph node kernels per step (node mode, median)`.
  Call the remainder "device-idle outside graphs", not "host time".
- Replace the recorded check with a gating one: node-mode in-graph union ≤ graph-mode span × 1.02, per arm. If it fails,
  node-mode durations are inflated: the arm is `NODE_TRACE_INFLATED` and its class terms are not read.
- In the decision rule:
  - `in_graph_idle` is nameable, as "launch/dependency gaps".
  - If |Δoverlap_residual| ≥ 50 % of Δperiod, the gap reads UNEXPLAINED, never "spread".

### 2. BLOCKER: name-only class rules mis-bin in every engine; the e4b column is unwritten, and by name alone it cannot separate dense from expert

The residual rule (D:117-118) catches only UNMATCHED kernels. A kernel matched into the wrong class is silent, and that
is the failure here.

**e4b (the column is specifiable from source now, but not by names alone).**
- The decode kernels are grouped-nf4-gemm Triton functions with stable names: `_quant_x_rows`, `_gemv_int4_b32`,
  `_reduce_partials`, `_gemm_int4_b32_grouped`, `_tile_table_r1`, `_quant_x_rows_gathered`, `_swiglu_rows`,
  `_combine_rows`, `_rmsnorm_rows`, `_rmsnorm_resid_rows`, `_scaled_resid_add_rows`, `_rope_norm_heads`, `_rope_heads`,
  `_router_epilogue` (G `kernel/int4_b32.py:40-1083`). Add `int4_smallm` (K16, K19) and the `fp8_paged_attn` / `fp8_kv`
  kernels.
- At B=1 the int4 attention projections (`Int4Linear`) call the SAME `quant_x_rows` + `gemv_int4_b32` as the int4
  experts:
  - E `engines/int4_attn.py:37,188`;
  - E `engines/hot_residency.py:755-756, 816-817`.
- So `_gemv_int4_b32`, `_quant_x_rows` and `_reduce_partials` are each shared between dense_gemm and moe_expert.
- The GEMV can be told apart by gridY = R: launched `[(cdiv(N,bn), R, sk)]` (G `int4_b32.py:338,350`), with R=1 for a
  projection and R=top_k=8 for experts.
- `_quant_x_rows[(R, K//BLOCK)]` (G `:65`) cannot be told apart reliably, because the gate/up input row count is not 8.
- The router GEMM and the lm_head are both cuBLAS and both inside e4b's graph. The lm_head is bf16 on `int4_sched`
  (E `engines/int4_attn.py:17-20`), and the lm_head and argmax run inside the graph (E `engines/paged_runner.py:301-305`).
- Generic ATen glue: two `index_select` per layer of the KV tables inside the graph (E `engines/fp8_paged_kv.py:814-815`),
  plus casts and copies. That is about 100+ generic kernels per step that no name can place.
- "From its census at the box's commit" (D:101) reads as a post-hoc map, and no gnf4 sha is pinned.

**llama.cpp at B=16: the registered grid rule (D:113-115) cannot work.**
- Q4_K and Q6_K MMQ run stream-k on sm_120. The Blackwell config falls back to the Ampere config
  (L `ggml/src/ggml-cuda/mmq-config-blackwell.cuh:36`), and that has `stream_k=true` for every J
  (`mmq-config-ampere.cuh:157-172, 191-204`).
- Stream-k launches `block_nums_stream_k(nsm or ntiles, 1, 1)` (L `mmq.cuh:1452`). Dense and expert `mul_mat_q` both have
  gridY = gridZ = 1.
- At 16 tokens the experts go through MMQ, not the `mul_mat_vec_q_moe` kernel (L `ggml-cuda.cu:1919-1936`; `mmvq.cu:1184`).
- The fallback to "one merged class" is triggered by "if the proof cannot separate them", but the proof never traces
  B=16 (D:157). The draft also never says how a merged llama.cpp class is differenced against e4b's separate classes
  inside Δperiod.

**llama.cpp at B=1.**
- The gridY rule does work here: MMVQ's grid is `(nblocks, nchannels_dst = n_expert_used, nsamples)` (L `mmvq.cu:1005`).
- But `quantize_q8_1` is binned wholesale into moe_expert (D:103), and most of those launches quantize dense inputs
  (q/k/v, o, lm_head). That inflates llama.cpp's moe_expert and shrinks Δmoe_expert, biasing against Q3.
- `rms_norm_mul_rope_f32` also writes K into the cache (fused set_rows), so part of llama.cpp's KV append lands in
  norm_elem while every other engine's lands in attn.

**vLLM.**
- The router `F.linear` is cuBLAS, the same family as the cuBLAS lm_head. It sits in moe_route's engine but is not in
  that column (D:104 against UPSTREAM-vllm:22).
- Inductor `triton_*` names are generated at compile time and say nothing about semantics; a decomposed small-M mm would
  bin as norm_elem.
- `argmax` will not match ATen's `ArgMaxOps` (case-sensitive substring).
- The ATen fills and copies around `fused_marlin_moe` are unlisted.
- `flash_fwd` is "likely", never verified (UPSTREAM-vllm:24).

**SGLang.**
- `torch.zeros(intermediate_cache13)` runs in every MoE layer (S `layers/moe/fused_moe_triton/fused_marlin_moe.py:290-294`).
  It is a generic FillFunctor kernel and is unlisted.
- The router cuBLAS and lm_head cuBLAS are both inside the graph (UPSTREAM-sglang:13), so only a shape or grid rule
  separates them.

**Fix.**
1. In Phase 0 (no GPU time), derive every engine's candidate names from the installed binaries and the compile cache:
   - `cuobjdump -symbols` / `nm -C` on vLLM `_C`, `_moe_C`, vllm-flash-attn, `sgl_kernel` and the sglang JIT objects,
     and `libggml-cuda`;
   - the gnf4 Triton source at the pinned sha;
   - vLLM's inductor output in `~/.cache/vllm/torch_compile_cache`.

   Freeze the map from these before any census. Write the e4b column into the registration with gnf4 pinned (SC1 ran
   `34da93d`).
2. Add a registered **segment rule**: classify by graph-node order between anchor kernels. Examples: everything from the
   router kernel to the expert-combine kernel of a layer is MoE; a `quantize*` launch inherits the class of the next
   matmul it feeds.
   - Graph node order is fixed across replays, so node mode's `graphNodeId` sequence supports this.
   - The rule handles generic ATen glue, quantize kernels, stream-k MMQ and shared e4b GEMVs in one mechanism.
3. Separate cuBLAS router from lm_head by in-graph versus out-of-graph (vLLM) or by gridX (e4b, SGLang: N = 128 against
   151,936), with the threshold registered.
4. When one engine reports a merged class, the gap merges the same classes on both sides.

---

## MAJORS

### 3. MAJOR: the step delimiter fails at llama.cpp's KV re-warm, and the "graph executable differs" filter cannot see it

- **Re-warm runs eager steps, then reuses the same executable.** On a property change, llama.cpp runs the next steps
  eagerly (warmup reset, L `ggml-cuda.cu:4468-4475`). It then re-captures and calls `cudaGraphExecUpdate` on the SAME
  instance; it re-instantiates only if the update fails (L `ggml-cuda.cu:2634-2659, 4403-4413`).
  - The re-warm steps have no GRAPH_TRACE row, and the replays on either side normally keep one `graphExecId`.
  - "Drops a step whose graph executable differs" (D:80) drops nothing.
  - The period from the last pre-boundary replay to the first post-boundary replay spans about 3 steps and is kept.
- **The reducer then averages it in.** The reducer pairs consecutive modal-exec replays and averages the periods
  (`sc1b_census.py` `graph_mode`: `zip(steady, steady[1:])`, `fmean`). One 3-step period in 62 moves the mean period
  and host term by about 3 %.
- **The "one graph per step" claim is unproven at B=16.** It rests on all 16 sequence ids being consecutive (one
  `split_equal` ubatch, L `src/llama-kv-cache.cpp:713`) and on graph properties staying stable. The proof never runs
  B=16 (D:157).
- **Fix.**
  - Delimit llama.cpp steps by the logits D2H, which is one per `llama_decode` with outputs. Require exactly one
    `cudaGraphLaunch` between consecutive delimiters.
  - Drop any period with more out-of-graph kernels than the steady count, or longer than 1.5× the median.
  - Use medians and report the dropped count.
  - For every engine, refuse an arm whose window contains a step with zero or two graph launches.

### 4. MAJOR: the decomposition uses the profiled Δperiod, with no gate on profiler inflation

- "A profiled step is never a speed reading" (D:144), yet the 50 % rule divides by the profiled Δperiod (D:124, D:142).
- cuda-sw tracing adds a callback to every runtime API call outside the graph. That load differs by engine:
  - e4b's Python scheduler issues its tensor builds, three H2D copies, the slot-index D2D and `.tolist()` outside the graph
    (E `engines/paged_runner.py:365-367, 384`; E `engines/fp8_paged_kv.py:776-789`);
  - vLLM runs its input-prep Triton kernels, the lm_head, sampling and the copy-stream D2H outside the graph;
  - llama.cpp has few such calls.
- So the profiled Δhost is biased by engine. Nothing compares the graph-mode period with pass 1's unprofiled step time on
  the same box.
- **Fix.** Gate per arm: the graph-mode median period must sit within ±3 % of the same box's unprofiled ms/step.
  Otherwise the host term is labelled `PROFILER_INFLATED` and is not nameable. Print both numbers in every gap row.

### 5. MAJOR: Q1 and Q4 test a different quantity from the decision rule, and Q1 is already answered by SC1

- **Q1 is already answered.**
  - SC1's method pair (window / sched − 1) at B=1 is +6.0 % on box B, +10.4 % on box C and +6.3 % on A3
    (E `bench/h2h-2026-10-02/sc1/README.md:196-198`).
  - The bare `g.replay()` window is 4.21–4.32 ms against the scheduler's 4.50–4.77 ms (box C outer.log:
    `B1D_TIMED_GRAPH step=4.32ms`).
  - So e4b's B=1 step is already known to be at least ~0.9 graph-bound. "Device-busy ≥ 0.85" cannot fail; it is not a
    prediction "written before any census exists".
- **Q1 and Q4 thresholds do not test the claims they make.**
  - Q1's threshold is one engine's busy fraction. It does not imply "the loss is mostly device time": with e4b at 0.85
    and llama.cpp at 1.0, Δhost is about 0.68 ms of 1.47, i.e. 46 %.
  - Q4's "device-busy ≤ 0.75" implies host share > 50 % of Δperiod only if e4b's host time is near zero.
  - Q4's mechanism cites the 9.7 MB logits D2H, which the instrument books as device-busy (memcpy, D:91), not host.
- **Q3 and Q5 have their own problems.**
  - Q3 is pushed toward FALSE by the `quantize_q8_1` mis-bin (finding 2).
  - Q5 is near-certain: the same Marlin MoE family on the same checkpoint. Its class composition also differs: SGLang's
    zero-fill is unlisted, vLLM's `moe_sum` and SGLang's `moe_sum_reduce` differ. If Q5 fails, nothing follows.
- **Fix.**
  - Restate Q1 as "Δ(device-busy time) ≥ 50 % of Δperiod at B=1", or drop it as known.
  - Restate Q4 as "Δ(device-idle outside graphs) carries ≥ 50 % of |Δperiod| at B=16", in the rule's own quantity.
  - Make Q5 a gating instrument control on the Marlin MoE GEMM kernel alone (vLLM and SGLang within 5 %); if it fails,
    the node-mode duration fidelity is suspect.
  - Make Q3 conditional on the class-map gate.

### 6. MAJOR: brackets. The llama.cpp window may not fit and its poll perturbs the step; SGLang's trigger changes the workload; vLLM is off by one; K and the window position are unregistered

**llama.cpp.**
- **Polling `/slots` perturbs the measured steps.**
  - `/slots` posts a high-priority task that the queue worker runs WHILE the decode yields (L
    `tools/server/server-context.cpp:2375-2378, 3678-3686, 4729-4741`).
  - When the decode ends, the main thread waits for the worker to finish (L `tools/server/server-queue.cpp:245-250`). Each
    poll's JSON build for every slot can therefore extend a step.
  - The driver polls through the window ("waits until `n_decoded` has advanced by N", D:75), so it perturbs exactly the
    host term Q4 reads.
- **The B=1 window may not fit.**
  - SC1's server uses `-c = np × 1024` (E `bench/sc1/llamacpp/llamacpp_box.sh:110-120`). At B=1 that leaves ≤ 512
    generated tokens, about 1.55 s of decode.
  - `nsys start` latency is unmeasured. A ~1 s latency puts the window past position 768 (a re-warm boundary, finding 3)
    and outside SC1's slope region (positions 544–640).
- **"Holds ≈ N steps; select exactly N" is self-contradictory** (D:76).

**SGLang.**
- SC1's arm sends one non-streaming batch `/generate` (E `bench/sc1/sglang/sc1_sglang_arm.py:13`). The census streams
  (D:72).
- Streaming emits an output every step; non-stream emits every 50 tokens (S
  `managers/scheduler_components/output_streamer.py:428-446`; `SGLANG_FORCE_STREAM_INTERVAL` = 50).
- So the census times a different scheduler loop from the one SC1 timed.

**vLLM.**
- `step()` increments, then starts when `count == delay` (V `vllm/profiler/wrapper.py:105-111`). Profiling therefore
  starts at the D-th `execute_model`.
- `delay_iterations = K+1` skips the prefill plus K−1 decode steps, not K (D:69).
- Async scheduling is on by default in 0.30 (V `vllm/config/vllm.py:1438-1487`).

**All engines.**
- K is never given a value (D:65).
- Generation length (`max_tokens` / `n_predict` / `max_new_tokens`) is unregistered. It must cover the prefill steps + K
  + N + margin for e4b's first-admitted row at B=16.
- N = 64 is "exactly" (D:65), yet the reducer keeps ≤ 62 (D:81).

**Fix.**
- Register K = 32 (SC1's slope start) and generation lengths per engine, and record decode positions per window.
- llama.cpp:
  - Trigger off a client-side clock (request time + measured prefill + K × SC1 step), with no polls inside the window.
  - Run the census with its own registered larger `-c`; `n_kv` follows used cells, so this changes little but must be
    stated.
  - Collect ≥ 2N steps and keep the N centred on the target position.
  - Record `nsys start` latency in the proof.
- SGLang: send SC1's exact non-streaming batch request, and arm `/start_profile` with `start_step` (absolute
  `forward_ct`: prefill + K + current), or trigger on a time offset.
- vLLM: use `delay_iterations = K+2`, or state the K−1.

### 7. MAJOR: the proof does not exercise the paths the census depends on

The toy (D:152-155) is one process that captures inside the range, in graph and node mode, at low memory. Every real arm
differs:

- **Graphs instantiated BEFORE the capture range.** Every engine captures at warm-up. CUPTI supports attaching after
  instantiation, but "some data in the kernel record would be missing" (`cuptirn.txt`, CUDA 12.0 Update 1 note).
  Findings 2's grid rules need gridX/Y/Z on those records.
- **`cudaProfilerStart` called from a SPAWNED child.** This is how vLLM's EngineCore and SGLang's scheduler start the
  capture.
- **Node mode on any real engine.** The proof's only real-model run is llama.cpp B=1 in graph mode (D:157).
- **Memory pressure.** "CUDA GPU trace collection requires a fraction of GPU memory" (`nsys/rn.txt:292`); vLLM runs at
  `gpu_memory_utilization 0.90` (E `bench/sc1/vllm/sc1_vllm_common.py` `build_llm_kwargs`).
- **llama.cpp at B=16** (finding 3).
- **The 2025.6.1 sqlite schema.**
  - `graphExecId` in `CUPTI_ACTIVITY_KIND_GRAPH_TRACE` is documented only in the 2025.3 UG (`nsys/ug2025.3.txt:7290`).
  - The 2025.5/2025.6 UGs carry no schema, and `nsys/pca.txt` is a 404 stub.
  - The reducer's self-test builds its own schema.
- **`cudaGraphLaunch` correlation IDs.** Wrong IDs were fixed only in CUDA 13.1 Update 1 CUPTI (`cuptirn.txt`), and
  node-mode step grouping keys on them.
- **The proof's stated catch is the wrong one.** "Catches SYS_PTRACE" is not what CUDA injection needs. The real risks
  are child injection and seccomp around injected processes (`nsys/rn.txt` "Known Issues").

**Fix.** Extend proof item 2 so the toy:
- captures its graph BEFORE start, in an `mp.spawn` child that also calls `cudaProfilerStart`;
- runs at ≥ 90 % memory;
- is checked through the real reducer;
- asserts non-zero gridX/Y/Z, a `graphNodeId`, constant kernels per launch, and that the `graphExecId` column exists.

Then add one node-mode real-engine trace for vLLM B=1 (the most fragile bracket) and one llama.cpp B=16 graph-mode run
with `-lv 5` "warmup complete" evidence. Budget: about +20 min.

### 8. MAJOR: the host regime is unconstrained, so the 5 % agreement check will probably label the lane's largest win

- llama.cpp's B=16 step is host-bound by design (Q4): one main thread samples 16 × 151,936 logits after
  `llama_synchronize`.
- SC1 measured llama.cpp only on box B (EPYC 7663) and SGLang only on box C (Xeon E5-2698 v4, 2.2 GHz;
  box C `outer.log`).
- The census runs on a new, unconstrained rental, using box C's image, not its host. The B=16 llama.cpp/e4b ratio (0.624)
  moves with single-thread CPU speed, so "within 5 %" (D:145) will likely fail and label the 5.8 ms win as "different host
  regime".
- 5 % is about 3× SC1's own vLLM cross-box spread (1.166–1.203 at B=1), but it was never calibrated for a CPU-bound arm.
- **Fix.** Register a rental filter: a CPU with single-thread speed in box B's class (EPYC 7663-class or faster, recorded),
  or one census box per reference box. Alternatively, drop the agreement rule for llama.cpp B=16 and read its gap only
  from the census box's own unprofiled ratio, labelled as such.

---

## MINORS

9. **MINOR: definitions.**
   - "Out-of-graph device time" counts only work between this graph's end and the next start (D:89). The busy union
      counts everything in the period (D:91). SGLang's overlap and vLLM's copy stream run out-of-graph work concurrently
      with the graph. Use one definition: every non-graph interval in the period.
   - Copies are "busy" while the SMs are idle; state that.
10. **MINOR: no uncertainty.**
    - The reducer uses means, and there is one draw per pass.
    - The rule names a cause from point estimates. Report per-step IQR for every term.
    - Require the named term to beat 2× its IQR-based noise, and use medians.
11. **MINOR: `--trace-fork-before-exec=true`** (D:60) "relies on undefined behavior and might cause your application to
    crash or deadlock" (`nsys/ug2025.6.txt:970-976`). With spawn it buys nothing. Drop it, or use it only where a
    fork-without-exec child launches CUDA work, which none does.
12. **MINOR: the trace method.**
    - The claim that cuda-hw loses graph launches "after tens of thousands of kernels" (D:61) has no source in the notes.
    - The 2025.3 notes say HES *mitigates* node-mode overhead on Blackwell (`nsys/rn2025.3.txt:150-155`).
    - Cite the claim, or run one arm's node mode under both `cuda-sw` and `cuda` as the fidelity cross-check.
13. **MINOR: budget arithmetic and realism.**
    - Proof $0.75 + main $2.63 = $3.38, over the $2.5 line by $0.88, not $0.13 (D:167); $4.13 with the allowed proof
      relaunch.
    - The 60–80 min for 24 processes omits e4b's B=16 prefill repetitions: 16 × 512 tokens at 2.0–3.5 s per chunk
      (E `bench/h2h-2026-10-02/sc1/README.md:180-183`), × 8 reps in pass 1 plus 2 census passes. It also omits e4b's
      120–150 s loads (box C receipts) and 16 sqlite exports.
    - Plan about 100–120 min of process time and say what is dropped if the guard bites.
14. **MINOR: e4b census driver.**
    - The new `--census` loop must call `run_until_idle`'s exact per-step body; a hand-rolled `step()` loop is a
      different program.
    - Pin `max_new_tokens` so the first-admitted row (staggered B=16 prefill) does not finish inside the window.
    - Pin the gnf4 sha, which fixes the Triton kernel names.
15. **MINOR: cross-class fusions.**
    - llama.cpp `rms_norm_mul_rope` does a K set_rows inside the norm.
    - e4b `_rope_norm_heads`; vLLM inductor fusions.
    - Declare each fused kernel's class and list the cross-class fusions in the registration.
16. **MINOR: vLLM's unprofiled pass is not the census's configuration.** SC1's default is a `fork` EngineCore with no NVTX;
    the census runs `spawn` plus a per-step NVTX range (V `vllm/profiler/wrapper.py` `annotate_context_manager`). Run the
    pass-1 timing with the census env.
17. **MINOR: "CUPTI has one subscriber below r610"** (D:63) is unsourced. Multi-subscriber activity tracing went beta in
    CUDA 13.2 and production in 13.3 (`cuptirn.txt`). The risk is low here, but assert in every report that nsys
    diagnostics show no subscriber or CUPTI-buffer errors.
18. **MINOR: clocks.** Graph and node passes are separate processes. Record SM clock, power and temperature
    (`nvidia-smi dmon`) per pass, so class times and spans are compared at the same clocks.

---

# SC1b design review, round 2 (hostile), against `bench/sc1b/SC1b-PREREG.md` and its implementation

Reviewed 2026-10-03 against the uncommitted `sc1b/census` worktree (HEAD
`6afa69e7`). Sources: the registration, `UPSTREAM-NOTES.md`, round 1's `DESIGN-REVIEW.md`, the implementation files, the
`git diff` of SC1's scripts, vLLM 0.30.0, SGLang 0.5.20, llama.cpp `552f18f`, gnf4 at `34da93d` (local clone), e4b at the
worktree HEAD, and the Nsight Systems user guides 2025.3, 2025.5, 2025.6 and the current (2026.x) one.

Executable evidence (the reviewer's scratch scripts, not committed):
- `notes/r2/sim_e4b.py`: the real `ContinuousScheduler` (worktree `experts4bit_qlora/engines/scheduler.py`) driven through
  the real `census_window`.
- `notes/r2/adv_classes.py`: adversarial kernel sequences through the real `classify_seq` and the frozen
  `kernel_classes.json`.
- `notes/r2/adv_reducer.py`: reducer edge cases through the real `graph_mode` / `node_mode` / `arm`.

The worktree's CPU tests pass (`tests/test_sc1b.py`: 14 passed; reducer self-test 21 checks). None of the BLOCKERs below is
reachable by them.

Disclosure: while checking pin freshness I ran `bench/sc1/make_pin.sh`, which rewrites `bench/sc1/staged.sha256`. Its
output was byte-identical to the file already there: `git diff` is unchanged at +10/−3 with the same hashes, so only the
mtime moved. The pin is fresh. No other worktree file was touched; the tests ran with `PYTHONDONTWRITEBYTECODE=1 -p no:cacheprovider`.

---

## Round-1 findings: fixed in both, or on paper only

| R# | registration | code | status |
|---|---|---|---|
| R1 in-graph idle term, G-node gate, UNEXPLAINED | yes | yes (`arm`, `gap`) | **partial**: U_in excludes in-graph copies (M5); G-node is one-sided (M10) |
| R2 class map | yes | segment rule, e4b column, inherit | **partial**: three registered pieces have no code: the Phase-0 symbol check (proof item 5), the gridX router/lm_head threshold, and the "merged class merges the other side" rule. Unclosed segments are silent (M6) |
| R3 llama.cpp delimiter, drops, medians | yes | yes (`graph_mode` d2h, `_filter`) | fixed. The capture it reads is broken (B3) |
| R4 G-inflate | yes | yes | **partial**: the reference is a different estimator, so the gate fires on pass-1 noise (M9) |
| R5 Q1/Q4 restated, Q5 as control C1, Q3 conditional | yes | Q1–Q4 have no evaluator; **C1 has no code** | **C1 is paper-only** (M4) |
| R6 K, N, lengths, positions, non-stream SGLang, vLLM delay | yes | K=32, N=64, delay=34 and non-stream are right | **e4b length 104 ≠ 160 (B1)**. SGLang's "positions read from the trace, VOID outside 17–112" and llama.cpp's "centred on position 64" have no code. The llama.cpp trigger is unworkable at B=1 (B3) |
| R7 proof extensions | yes | toy, vLLM B=1 node, llama.cpp B=16 graph | **partial**: the proof gates on rc, not content. The e4b, SGLang, launch+node and eager-graphNodeId paths are never run (M7, M11) |
| R8 host regime | demoted to a diagnostic | ratio printed, no gate | fixed |
| R9 D_out definition, copies are device time | yes | D_out yes | in-graph copies are counted as idle (M5) |
| R10 medians, IQR, 2×IQR | yes | yes | fixed. The noise definition is unregistered (m3) |
| R11 no fork-before-exec | yes | yes | fixed |
| R12 cuda-sw | yes | yes | fixed |
| R13 budget and drop order | yes | `can_run` | the drop order differs (m4) |
| R14 e4b driver: step body, max_new_tokens, gnf4 pin | yes (160) | step body yes, gnf4 yes, **max_new_tokens = 104** | **paper-only for the length; reproduces the exact R14 hazard (B1)** |
| R15 cross-class fusions declared | yes | JSON consistent | fixed |
| R16 vLLM pass 1 with spawn | yes | `VLLM_WORKER_MULTIPROC_METHOD=spawn` passed | fixed |
| R17 assert nsys diagnostics | yes | **no assertion anywhere**. `nsys_diag` greps the app log to stdout for e4b/vLLM only, and the reducer never reads `DIAGNOSTIC_EVENT` | **paper-only** (M4) |
| R18 clocks | sampler per pass + G-clock gate | samplers run; **no G-clock code** | **gate paper-only** (M4) |

---

## BLOCKERS

### B1. BLOCKER: e4b B=16 census refuses at bracketed step 55. The row budget is 104, not the registered 160, and admission is staggered

**Evidence**
- `bench/sc1b/sc1b_e4b_census.py:160-162` sets `budget = a.skip + a.steps + 8` (104) and `add_request(..., max_new_tokens=budget)`.
  The registration (PREREG:89-90) says 160.
- serve_paged's prefill budget defaults to one chunk:
  - `serve_paged.py:216-217` (`prefill_budget = max_prefill_tokens or chunk_tokens`);
  - `scheduler.py:238-245` (one 512-token chunk per step);
  - SC1's `sched_env` sets no `E4B_PAGED_MAX_PREFILL_TOKENS`.
- So at B=16 the rows prefill one per step over 16 steps:
  - row 0 holds 17 tokens when `census_window`'s ramp ends (step 17);
  - 49 after the 32 skips;
  - it reaches 104 and retires at bracketed step 54;
  - step 55 then has 15 decode rows, and `census_window` raises.
- Reproduced with the real scheduler (`notes/r2/sim_e4b.py`):
  ```
  B=16 budget=104: REFUSED -- bracketed step 55 was not a full decode-only step (prefill 0, decode 15 of 16)
  B=16 budget=160: OK ramp=17  row0 tokens at bracket end=113
  ```
- The selftest's `FakeSched` (`sc1b_e4b_census.py:73-89`) prefills every row at once, so the tests cannot see this.
- **Effect:** both e4b B=16 captures exit non-zero after `cudaProfilerStop`, so `nsys_export` is skipped (`sc1b_box_d.sh:46`).
  `sc1b_arm e4b 16` is UNREAD, which means **G2 and every B=16 gap are lost**.

**Fix**
- Use `max_new_tokens = 160` (the registered value). Assert `ramp + skip + steps ≤ 160 − 1` for the first-admitted row
  before `cudaProfilerStart`, and refuse before opening the range.
- Make `FakeSched` admit one prefill chunk per step, or drive the real `ContinuousScheduler` with a fake runner in
  `tests/test_sc1b.py`, as `sim_e4b.py` does.
- Record the window's per-row decode positions. At B=16 they span 34–113 across rows, not 33–96 (m1).

### B2. BLOCKER: every SGLang capture is discarded. `/start_profile` answers plain text, and the client `json.loads` it

**Evidence**
- SGLang's `http_server.py:1181-1189` returns `Response(content="Start profiling.\n")`.
- `sc1b_serve_census.py:58-62` (`_post`) does `json.loads(r.read() or b"null")`, and `:98` calls it for `/start_profile`.
  The `JSONDecodeError` is not caught.
- Sequence of failure:
  1. The main thread dies after the profile has already started.
  2. Python waits for the non-daemon `/generate` thread, then exits 1.
  3. No census JSON is written: plan, events and positions are lost.
  4. `sglang_census` sees rc=1 and skips `nsys_export` (`sc1b_box_d.sh:75`).
  5. `sc1b_arm sglang` reads UNREAD (missing export).
- **G4 is unread on the box.** The `.nsys-rep` files are probably on disk and pulled, so they are salvageable off-box, but
  nothing registered reads them.
- The proof never runs the SGLang bracket (M11), so this reaches the paid run.

**Fix**
- For `/start_profile`, read the body as text and require HTTP 200 and `"Start profiling"`.
- Write the census record in a `finally` block.
- Make `status` depend on the profile acknowledgement as well as the token counts.
- Add a selftest against a stub HTTP server that returns SGLang's literal text, and add one SGLang B=1 bracket to the proof.

### B3. BLOCKER: the llama.cpp B=1 window (G1, the lane's headline gap) cannot be opened by `nsys start` on a client clock, and the reducer would accept whatever few steps land

**Evidence**
- At B=1, 160 tokens at 3.03 ms/step is about 0.49 s of decode. The plan (`sc1b_serve_census.py:34-44`) opens the window at
  roughly prefill + 0.097 s (about 0.14 s after the request) and closes it at about 0.52 s, capped at end of generation.
- `nsys start` is a fresh CLI process (`:102`): binary load plus session IPC to enable CUPTI in the target. Its latency is
  unmeasured at B=1.
- The proof measures it only at B=16 (`sc1b_box_d.sh:166-173`), where generation lasts about 2.5 s and a 1 s latency is
  harmless. **A latency of 0.2–0.4 s at B=1 leaves 0–60 decode steps; anything over about 0.4 s leaves none.**
- Two silent failure paths:
  - `graph_mode` has no floor on `steps_kept`. Six replays read `status: ok, steps_kept 3` (`notes/r2/adv_reducer.py`).
  - Registered positions are not enforced. "The reducer keeps the 64 steps centred on position 64" (PREREG:100) has no
    code: `_central` (`sc1b_census.py:141-145`) centres on whatever was captured, and the capture contains no prefill to
    count from.

**Fix**
- Open the llama.cpp collection **before** sending the requests:
  1. `nsys start`;
  2. wait for `nsys sessions list` / `status --session` to report collecting, or a fixed settle;
  3. send the B requests;
  4. `nsys stop` after generation ends.
- Capture the whole 160-token run. It is small: about 0.5 s at B=1 and 2.5 s at B=16.
- In the reducer, select decode positions 33–96 by counting logits D2H delimiters from the first decode D2H. The prefill is
  eager and its D2H comes first.
- Add a VOID floor (for example ≥ 56 of 62 steps kept) to `graph_mode` and `node_mode` for every engine.

---

## MAJORS

### M1. MAJOR: `nsys start --session=S` is not a documented `start` option in the pinned 2025.6.1, and the SGLang flavour of it is never proved

**Evidence**
- The 2025.6 UG's start options (`nsys/ug2025.6.txt:1868-2335`) list only `--session-new`. 2025.1, 2025.3 and 2025.5 are
  the same. `--session` on `start` first appears in the current UG (`nsys/ug.txt:2346-2351`).
- The documented 2025.6 interactive pattern is the reverse of what box D does: `nsys start --session-new=S` first, then
  `nsys launch --session=S app` (`ug2025.6.txt:2711-2716`).
- Box D uses `launch --session-new` + `start --session` for both servers: `sc1b_box_d.sh:70`, `sc1b_serve_census.py:102`.
- The llama.cpp proof item would catch a rejection, at $0.94. SGLang's variant (`-c cudaProfilerApi`, two sequential
  collections in one session) is never run.
- `nsys start`'s rc is ignored in `sglang_census` (`:70`), and `run()`'s status ignores the start and stop rcs (`:114`).

**Fix**
- In the proof, record `$NSYS start --help` and refuse if `--session` is absent.
- Prove one SGLang B=1 collection, then a second one in the same session.
- Check every nsys rc, and fold the start/stop rcs into `status`.
- If `--session` is absent on start: SGLang can use `nsys profile -c cudaProfilerApi --capture-range-end=repeat:2` around
  `sglang.launch_server`; llama.cpp can use `start --session-new` before `launch --session` together with B3's
  collect-everything fix.

### M2. MAJOR: the two launched engines run a different instrument from e4b and vLLM

**Evidence**
- `NSYS_TRACE` (`--sample=none --cpuctxsw=none`) is passed to `nsys launch` but not to `nsys start` (`sc1b_box_d.sh:70`,
  `sc1b_serve_census.py:102`).
- `start` has its own `--sample` and `--cpuctxsw`, whose default is `process-tree` when a target is launched
  (`ug2025.6.txt:1995-2004`, `2212-2224`). Which setting wins is undocumented.
- If sampling and context-switch tracing are on, llama.cpp's idle_out (Q4's quantity, and CPU-bound at B=16) is measured
  under CPU sampling, while e4b's is not.

**Fix**
- Pass `--sample=none --cpuctxsw=none` to every `nsys start`.
- Have the reducer VOID an export that contains CPU-sampling or scheduling tables.

### M3. MAJOR: the llama.cpp report can be cancelled before it is written

**Evidence**
- `nsys stop` returns, and box D immediately runs `nsys shutdown --kill sigterm` and kills the launch pid
  (`sc1b_box_d.sh:88`).
- The 2025.6 UG says of shutdown: "If a collection is pending or active, it is canceled" (`ug2025.6.txt:135-137`). Report
  generation after `stop` is asynchronous: compare the `--after-report-ready` hook.
- `sglang_census` waits up to 120 s for the `.nsys-rep` (`:74`). `llamacpp_census` does not.

**Fix**
- Wait until the `.nsys-rep` exists and its size has stabilised before `shutdown`. Check the shutdown rc.
- Then verify that port 8080 is free and that `nvidia-smi --query-compute-apps` is empty. Killing the nsys pid
  (`llamacpp_server_stop` kills `$LLAMACPP_SERVER_PID`, which is now nsys, not llama-server) may orphan the app and its
  18 GB, and e4b B=16 runs right after LL1.

### M4. MAJOR: four registered gates and readings have no code

- **R17 (diagnostics):**
  - `nsys_diag` (`sc1b_box_d.sh:29`) greps the *application* log, which is noisy with vLLM's own WARNINGs, and prints to
    stdout. It never fails anything and is not called for SGLang or llama.cpp.
  - The reducer never reads the export's `DIAGNOSTIC_EVENT` table (schema `nsys_schema_excerpt.txt:8445`).
- **R18 (G-clock):** the samplers run, but `sc1b_census.py` has no clock gate.
- **R5 (control C1):** there is no code. It would also not be a fidelity control: vLLM's Marlin MoE runs
  `use_atomic_add=False` (vLLM `fused_moe/experts/marlin_moe.py:131-155`), while SGLang's runs `use_atomic_add=True` for
  fp16 plus a zero-fill (SGLang `fused_marlin_moe.py:290-330`). Those are different kernel instantiations doing different
  work.
- **R2 / R6 / proof item 5:**
  - no Phase-0 symbol check;
  - no merged-class rule;
  - no SGLang position read and no VOID-outside-17–112 rule;
  - no llama.cpp centring.

**Fix.** Implement each in the reducer, gating where the registration says it gates, or strike it from the registration
before money is spent.

### M5. MAJOR: in-graph copy and memset nodes are booked twice, as class `memcpy` and as I_in

**Evidence**
- `node_mode` builds U_in from kernels only (`sc1b_census.py:300`, `r["kind"] == "kernel"`).
- `in:memcpy` sums every in-graph copy and memset (`:302-305`).
- With S taken from graph mode, an in-graph copy raises I_in, memcpy and O together. In `adv_reducer.py`, a 0.1 ms
  in-graph D2D node gives `I_in 0.1, memcpy 0.1, O 0.1`.
- So in-graph copy time is labelled "launch/dependency gaps". That is the cause Q1 and Q5 predict, so the error biases
  toward them, and it contradicts PREREG:135.

**Fix.** U_in = the union of all in-graph records: kernels, copies and memsets.

### M6. MAJOR: an unclosed MoE segment silently mis-bins whole layers, and G-map cannot see it

**Evidence**
- `classify_seq` (`sc1b_census.py:217-238`) ignores start anchors while `in_seg` and never checks one end per start.
- The llama.cpp `moe_weighted_reduction` fusion is conditional:
  - `ggml_cuda_check_fusion_memory_ranges` (`ggml-cuda.cu:3454-3459`);
  - n_expert_used ≤ 15;
  - `GGML_CUDA_DISABLE_FUSION`.
- When it is refused, the combine is `k_bin_bcast`, and the next layer is binned as MoE (`adv_classes.py`):
  - `rms_norm_f32`, `rms_norm_mul_rope_f32`, `k_set_rows` and `flash_attn_ext_vec` → `moe_route`;
  - the q/k/v and o_proj `mul_mat_vec_q` → `moe_expert`;
  - the next router `mul_mat_vec_f` → `moe_route`.
- Residual stays 0, so G-map passes.
- The same hole exists for e4b `_combine_rows` (conditional on `E4B_FUSE_COMBINE` and the dtype, `hot_residency.py:1236-1246`)
  and for any renamed end kernel.

**Fix**
- Per replay, require #starts == #ends == the number of MoE layers (48), strictly alternating.
- Otherwise label the arm `CLASS_MAP_SEGMENT_BROKEN` and leave its class terms unread.

### M7. MAJOR: out-of-graph detection depends on an untested export convention (`graphNodeId IS NULL`)

**Evidence**
- `graph_mode` and `node_mode` treat `node is None` as eager (`:253`, `:294`). The schema only says the column is a
  nullable INTEGER.
- If eager work is exported with 0 instead of NULL:
  - D_out becomes 0 and idle_out absorbs all out-of-graph device time (`adv_reducer.py`: D_out 0.15 → 0.0, idle_out
    0.25 → 0.40);
  - node mode loses every out-of-graph class.
- The toy has no eager kernel inside its range, so the proof cannot catch this.

**Fix**
- Put one eager kernel and one H2D copy inside the toy's range, and assert they land in the reducer's non-graph sets.
- Treat 0 as non-graph, or join `CUDA_GRAPH_NODE_EVENTS`.

### M8. MAJOR: there is no minimum-steps floor

`graph_mode` and `node_mode` return `ok` with any non-empty kept set (3 steps in `adv_reducer.py`). The gap reads medians
and IQRs of 3 steps as if they were 62.

**Fix.** VOID below a registered floor (for example 56).

### M9. MAJOR: G-inflate's reference is a different and noisier estimator

**Evidence**
- `p / unprofiled_ms` (`sc1b_census.py:332`) compares a per-step median at positions 33–96 against one pass's min-of-3
  slope over 32→128 tokens. For e4b the reference is a decode-only slope (`sc1_e4b_sched.py` A10).
- SC1 itself expects more than 3 % between draws: `needs_third` in `sc1_run.sh` adds a third draw above 3 %, and box C's
  e4b B=1 draws spread about 20 % before A10.
- A ±3 % gate against a single draw will fire on pass-1 noise. That strips idle_out of nameability, which is exactly Q4's
  term.

**Fix**
- Measure the unprofiled reference with the census's own estimator: the same bracket without nsys, timed per step with CUDA
  events or wall/64.
- Or set the threshold to max(3 %, 2× the pass's own min-versus-median slope spread). Register whichever is chosen.

### M10. MAJOR: G-node is one-sided and loosest exactly where I_in is large

**Evidence**
- `U_in ≤ 1.02 · S` (`:338`) tolerates node-mode kernel inflation up to the true idle fraction.
- Inflation lowers I_in = S − U_in one for one. With S = 4.2 and a true U_in of 3.0, a 30 % inflation passes the gate and
  reports I_in = 0.3 instead of 1.2.
- e4b, which carries the most hypothesised idle, tolerates the most error, and the error biases ΔI_in down.

**Fix**
- Also gate node-mode replay span (first node start → last node end) ≤ S × 1.02.
- Report I_in computed inside node mode (span_node − U_in) beside the mixed one, and label disagreement beyond the noise.

### M11. MAJOR: the proof gates on exit codes, not on what the census reads, and skips three paths

**Evidence**
- `prove_d` fails only on `rc≠0` (`sc1b_box_d.sh:156-175`). A vLLM node reduction that is VOID, keeps 0 steps or has 40 %
  residual still writes PROVED, and so does a llama.cpp graph reduction with 0 steps.
- Never exercised:
  - the e4b bracket under `nsys profile`, which would catch B1 (and inductor worker processes under `--wait=all`);
  - the SGLang bracket, which would catch B2 and M1;
  - node mode under `nsys launch`;
  - B=1 `nsys start` latency (B3);
  - the eager `graphNodeId` convention (M7).

**Fix**
- Fail the proof unless every proof reduction has `status ok`, kept steps ≥ the floor, residual ≤ 2 %, constant
  kernels-per-launch, and an empty `DIAGNOSTIC_EVENT`.
- Add an e4b B=16 graph capture and an SGLang B=1 node capture (about 10 min).

---

## MINORS

1. **Positions.**
   - The e4b B=1 window is decode positions 34–97, not 33–96 (ramp ends on the first decode step).
   - The e4b B=16 window spans 34–113 across rows because of staggered admission.
   - The llama.cpp window is centred on whatever was captured.
   - Record per-engine positions in each arm record.
2. **Naming rule** (`gap`, `sc1b_census.py:370-376`).
   - Only the single largest-|Δ| term is tested. PREREG:204 says "a term carrying ≥ 50 % (same sign)", so a qualifying
     ΔI_in loses to a larger opposite-signed class term.
   - Q1's "largest term" set silently drops idle_out when PROFILER_INFLATED.
3. **Noise is unregistered.** The code uses the sum of per-step IQRs of both arms (four IQRs for I_in). These are not
   uncertainties of the medians. Register the definition.
4. **Drop order.** Code order (SGLang graph then node, B=1 before B=16; `sc1b_box_d.sh:127-131`) drops SGLang B=1 node
   before vLLM B=16 node, contrary to PREREG:250.
5. **`reduce` on box D** (`sc1b_box_d.sh:141`) runs SC1's `sc1_reduce.py --box D`. It may write a misleading SC1
   `verdict.json` or fail with rc 22 (`rec 22`). Skip it on D.
6. **Logs overwritten.** Census servers reuse pass 1's server logs (`logs/sglang_server_matched.log*`,
   `logs/llamacpp_server_*_npB.log`), so pass-1 forensics are lost. Graph and node also overwrite each other.
7. **Alarm kills nsys, not the app.** `perl alarm … exec nsys` kills nsys; the traced e4b/vLLM process can survive holding
   the GPU.
8. **Cross-engine class semantics skew** (`adv_classes.py`):
   - e4b's 96 in-graph KV-table `index_select`s per step (`fp8_paged_kv.py:814-815`) → norm_elem;
   - vLLM's out-of-graph logits gather and cast → input_prep;
   - SGLang's in-graph logits cast → norm_elem;
   - SGLang's CUB `cumsum` → residual (the map's `cumsum`/`Cumsum` never matches `DeviceScanKernel`).

   The same work lands in different classes by engine.
9. **Registration/JSON drift.**
   - The registered llama.cpp mmvq gridY rule and the cuBLAS gridX threshold are absent from the JSON.
   - `inherit_prev` is in the JSON but not registered.
   - "Node order" is implemented as kernel start-time order.
10. **SC1 ratios** are not passed for the vLLM and SGLang gaps (`sc1b_box_d.sh:140`).
11. **`E4B_PAGED_PREFILL_ATTN=math`** has no tripwire, unlike A13's `E4B_INT4_PREFILL` assertion.
12. **Partial pulls** (`sc1_drive.sh` `pull_box`) do not exclude `census/` (`.nsys-rep` and `.sqlite`, possibly GB). They run
    every 20 min, mid-capture.
13. **llama.cpp census request** omits SC1's `n_probs: 0` and releases rows without SC1's barrier.

---

## Disposition of round 2 (applied before registration)

| item | fix | where |
|---|---|---|
| B1 | e4b rows carry 160 tokens; the bracket refuses BEFORE opening the range if the first-admitted row would retire inside the window, and records per-row decode positions | `sc1b_e4b_census.py` `census_window`; `tests/test_sc1b.py` real-scheduler test (B=16 at 104 tokens refuses, 160 passes) |
| B2 | `/start_profile`'s reply is read as text: HTTP 200 and "start profiling", or the capture is not acknowledged | `sc1b_serve_census.py` `_post_text`; selftest against a stub that answers SGLang's literal text |
| B3 | llama.cpp runs under plain `nsys profile`: the whole 160-token run is captured and the reducer selects intervals 33-96 between full-batch logits copies | `sc1b_box_d.sh` `llamacpp_census`; `sc1b_census.py` `--positions` |
| M1, M2 | no interactive sessions: every engine is a plain `nsys profile` (e4b and vLLM with `--capture-range=cudaProfilerApi`, SGLang's server under the same, llama.cpp whole-run) | `sc1b_box_d.sh`; test: no `start --session` / `launch --session`, no bare `nsys` |
| M3 | every capture ends by SIGTERM to the APP, then nsys's own exit (bounded 300 s), then an empty GPU. `end_capture` skips nsys and its wrappers: nsys's command line carries the app's, so a bare `pkill -f` would have signalled nsys too (found while re-checking this item; a Linux test shows the old form SIGTERMs nsys and the new one does not) | `sc1b_box_d.sh` `end_capture`, `gpu_free`; `tests/test_sc1b.py::test_end_capture_signals_the_app_and_never_nsys` |
| M4 | R17 (diagnostics) and R18 (clocks) are code; control C1 and the Phase-0 symbol check are struck from the registration (the proof's real-kernel residual gate replaces the latter) | `sc1b_census.py` `NSYS_DIAGNOSTIC_ERRORS`, `CLOCK_MISMATCH`; `SC1b-PREREG.md` |
| M5 | U_in is the union of in-graph kernels, copies and memsets | `sc1b_census.py` `node_mode` |
| M6 | segment balance per replay: 48 opens and 48 closes, else `CLASS_MAP_SEGMENT_BROKEN` (blocking) | `sc1b_census.py` `classify_seq`, `node_mode` |
| M7 | out-of-graph is `graphNodeId` NULL or 0; the toy now carries one eager kernel and one H2D copy per replay and checks both land in the non-graph set | `sc1b_census.py` `_nongraph`; `sc1b_toy.py` |
| M8 | fewer than 56 kept steps makes the arm VOID | `MIN_STEPS` |
| M9 | G-inflate's reference is pass 1's median-of-3 slope (`decode_ms_per_step_median`) | `sc1b_box_d.sh` `sc1b_arm` |
| M10 | G-node is two-sided on the span: U_in AND node-mode replay span both <= S x 1.02 | `sc1b_census.py` |
| M11 | every proof capture must REDUCE (status ok, >= 56 steps, no diagnostic errors; node mode also residual <= 2 % and no broken segment); proof adds e4b B=16 on Granite and SGLang B=1 node | `sc1b_box_d.sh` `prove_red`, `prove_d` |
| m1 | positions recorded per engine; vLLM's delay moved 34 -> 35 so its window is decode steps 34-97, e4b's (call 1 is the prefill and profiling starts AT the D-th call) | `sc1b_vllm_census.py` |
| m2 | naming rule: the first term, in order of absolute size, that has Delta-P's sign, >= 50 %, and > 2x noise | `sc1b_census.py` `gap` |
| m3 | noise registered: the sum of the two arms' per-step IQRs, a spread, not a confidence interval | `SC1b-PREREG.md` |
| m4 | drop order is the run order's reverse, as registered | `sc1b_box_d.sh` `box_d` |
| m5 | SC1's reducer is not run on box D | `box_d` |
| m6 | every census server writes its own log | `sglang_census`, `llamacpp_census` |
| m7 | after a perl alarm the app is SIGTERMed by pattern, then the GPU is checked empty | `e4b_census`, `vllm_census` |
| m8 | SGLang's CUB scan (`DeviceScan`/`cumsum`) is out-of-graph input_prep; generic ATen glue is norm_elem in-graph and input_prep out-of-graph on every engine | `kernel_classes.json` |
| m9 | `inherit_prev` and start-time node order registered | `SC1b-PREREG.md` |
| m10 | SC1's measured ratios passed to every gap | `box_d` |
| m11 | `E4B_PAGED_PREFILL_ATTN=math` exported after the scrub and asserted by the e4b tripwire (main's default is now flash, #973; every SC1 box ran math) | `bench/sc1/sc1_run.sh` |
| m12 | mid-run pulls exclude `census/` | `bench/sc1/sc1_drive.sh` `pull_box` |
| m13 | the llama.cpp request is SC1's own `completion_request` (n_probs 0), released through a barrier | `sc1b_serve_census.py` |
