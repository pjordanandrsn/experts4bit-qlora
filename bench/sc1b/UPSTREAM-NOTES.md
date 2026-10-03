# SC1b upstream notes: vLLM 0.30.0 (ced6857), read 2026-10-03 by a source-reading subagent (file:line in its report)

- Model Runner V2 is the 0.30 default (config/vllm.py:692-723). Offline LLM -> EngineCore CHILD process issues every kernel
  (VLLM_ENABLE_V1_MULTIPROCESSING=1 default; core_client.py:121-127; uniproc executor). nsys: --trace-fork-before-exec=true,
  VLLM_WORKER_MULTIPROC_METHOD=spawn per docs/contributing/profiling.md:186-188. Or VLLM_ENABLE_V1_MULTIPROCESSING=0.
- Bracket: profiler_config={"profiler":"cuda","delay_iterations":K,"max_iterations":N}; llm.start_profile(); generate;
  stop_profile -> CudaProfilerWrapper = cudaProfilerStart/Stop IN THE WORKER, counted per execute_model call; step 1 is the
  prefill, so K past it (config/profiler.py:16,42,122-130; profiler/wrapper.py:99-132,586-600; gpu_worker.py:1034,1209,1286-1302).
  VLLM_TORCH_PROFILER_DIR no longer exists. Works offline in both multiproc and in-process.
- Graphs: O2 -> FULL_AND_PIECEWISE; decode = FULL graph, exactly one graphs[desc].replay() per step holding only
  model(**inputs) (model_runner.py:1901-1909; cudagraph_utils.py:529-542,736). Prefill PIECEWISE.
  OUTSIDE the graph per step: H2D input copies, arange/zeros, triton _prepare_pos_seq_lens_kernel,
  _combine_sampled_and_draft_tokens_kernel, _gather_block_tables_kernel, _compute_slot_mappings_kernel, attn metadata;
  the logits gather + lm_head GEMM (cuBLAS, unquantized); sampling (greedy: _gumbel_sample_kernel + argmax/gather,
  _get_num_sampled_and_rejected_kernel); D2H on a copy stream; _post_update_kernel.
- NVTX: the "cuda" profiler emits one range per step: execute_context_{n}({tok})_generation_{n}({tok}) around
  execute_model only (NOT sample_tokens) (wrapper.py:602-604; gpu_worker.py:1124-1145). VLLM_NVTX_SCOPES_FOR_PROFILING is
  V1-runner only.
- Kernel substrings (sm_120):
  - MoE: marlin_moe_wna16::Marlin<...> (2/layer), act_and_mul_kernel, moe_align_block_size_kernel +
    count_and_sort_expert_tokens_kernel (128 experts > 64), moe_sum*; router gate cuBLAS F.linear (if unquantized),
    topkGating<...128...>.
  - dense qkv/o: marlin::Marlin<...> (Machete/Cutlass need sm90).
  - attention: FA2 via _vllm_fa2_C.varlen_fwd (external vllm-flash-attn @506341a; names not in tree — likely
    flash_fwd*); KV write reshape_and_cache_flash_kernel.
  - RMSNorm / qk-norm / rotary / residual: INDUCTOR triton_{poi,red,per}_fused_* (custom_ops none, IR native) — names
    generated at compile time. lm_head: cuBLAS, OUTSIDE the graph.
# SC1b upstream notes: SGLang 0.5.20 (94602c9c), read 2026-10-03 by a source-reading subagent (a shallow clone)

- Bracket: POST /start_profile {"activities":["CUDA_PROFILER"],"num_steps":N} once all B requests are in steady decode ->
  cudaProfilerStart/Stop in the scheduler process for exactly N run_batch launches (profiler_manager.py:273-276,386-388,
  155,441-450; scheduler.py:4252,4266; http_server.py:1181-1199; io_struct.py:2127-2153). start_step is ABSOLUTE forward_ct
  (0 = unset). Do NOT use profile_by_stage with --capture-range-end=stop (Stop+Start at prefill->decode ends the capture).
  nsys: --capture-range=cudaProfilerApi --capture-range-end=stop --cuda-graph-trace=node --trace-fork-before-exec=true
  (docs/.../benchmark_and_profiling.mdx:482-506).
- CAVEAT: overlap scheduler (default on) -> Start/Stop are CPU-side; step N-1/N GPU work can straddle the edges
  (scheduler.py:1991-2007) -> the reducer counts steps by graph instance, never by the window edges.
- Kernels come from the scheduler subprocess (mp spawn, title sglang::scheduler; engine.py:896-915,1766).
- Decode = FULL graph, one graph.replay() per step; logits processor + lm_head INSIDE the graph
  (cuda_graph_config.py:148; full_cuda_graph_backend.py:208; decode_cuda_graph_runner.py:1445). Outside, per step:
  static-buffer copies, flashinfer metadata (cumsum, kv-indices, fast_decode_plan), sampling (greedy torch.argmax), D2H result
  copy on copy_stream, overlapped process_batch_result. Overlap scheduling ON by default.
- NVTX: SGLANG_ENABLE_NVTX_SCHEDULER (+ pip nvtx) -> scheduler.{recv_requests, get_next_batch_to_run, run_batch,
  process_batch_result, ...}. No per-step forward NVTX (a torch record_function). Layerwise markers don't show on replay.
- Kernel substrings (fp16, sm_120):
  - MoE: sglang::device::marlin_moe::Marlin<...> (2/layer); align: align_single_token_kernel (B=1) /
    moe_align_block_size_kernel + count_and_sort_expert_tokens_kernel (B=16); act_and_mul_kernel; moe_sum_reduce*;
    torch.zeros fill.
  - router: cuBLAS (F.linear) + _router_triton_kernel (top-k).
  - dense GPTQ linears: sglang::device::marlin::Marlin<...> (only quantized layers — check the checkpoint config).
  - attention: flashinfer BatchPrefillWithPagedKVCacheKernel (GQA 8 -> tensor-core decode via the prefill module) + merge
    states; KV write: store_kvcache (fp16 disables fused rope+KV store).
  - norms: RMSNormKernel / FusedAddRMSNormKernel; qk-norm fused_qknorm_warp/_cta; rope fused_rope_kernel; lm_head cuBLAS.
  - sgl_kernel from sglang-kernel==0.4.7.
# SC1b upstream notes: llama.cpp 552f18f (b11327), read 2026-10-03 by a source-reading subagent (file:line in its report)

- CUDA graphs ON at B=1 and B=16. Disabled only by GGML_CUDA_DISABLE_GRAPHS, pre-Volta, or the MUL_MAT_ID sync fallback,
  which is not taken on sm_120 (MMVQ <= 8 tokens, MMQ above). ggml-cuda.cu:2555-2586, 4425-4430, 2568-2574.
- Graph lifecycle: 2 warm-up calls, then capture; re-armed on shape/pointer change. n_kv padded to 256 (~2 non-graph steps
  at each 256 crossing); 10 s idle eviction. ggml-cuda.cu:4458-4478, 2592-2631; llama-kv-cache.cpp:1265-1270.
- A step = CPU get_rows (embedding) -> H2D input copies -> ONE cudaGraphLaunch (GPU split) -> D2H logits copy
  (n_out x 151936 x 4 B: 0.6 MB at B=1, 9.7 MB at B=16) -> CPU sampling per slot -> llama_synchronize.
  llama-model.cpp:1603-1605; ggml-backend.cpp:945-948,1691-1700; llama-context.cpp:1948; server-context.cpp:3855-3856, 3682-3684.
- No NVTX / cudaProfilerStart anywhere. Bracket from outside: /slots (is_processing, n_decoded) or the kernel pattern
  (back-to-back graph launches, each followed by the logits D2H). Cross-check: response predicted_per_token_ms.
- 16 slots -> one llama_decode, one ubatch iff seq ids consecutive (split_equal sequential). server-context.cpp:2866-2874;
  llama-kv-cache.cpp:713; llama-batch.cpp:569-571.
- Kernel substrings (Q4_K=(ggml_type)12, Q6_K=14 for attn_v / ffn_down(_exps) on more-bits layers and output; router F32/F16):
  - B=1 experts: quantize_q8_1 + mul_mat_vec_q<12|14,1,...> (gate+up+swiglu fused, has_fusion). B=16 experts:
    mm_ids_helper + quantize_mmq_q8_1 + mul_mat_q<...> + mul_mat_q_stream_k_fixup; swiglu = unary_gated_op_kernel.
  - dense q/k/v/o + lm_head: mul_mat_vec_q (B=1) / mul_mat_q (B=16). Router: mul_mat_vec_f / mul_mat_f.
  - routing: topk_moe_cuda (fused); fallback soft_max_f32 / k_argsort_f32_i32 / k_get_rows. Expert sum:
    moe_weighted_reduction_f32.
  - FA: B=1 flash_attn_ext_vec; B=16 flash_attn_ext_f16 (MMA; non-unified KV -> Q->ne[3]=16). Helpers:
    flash_attn_mask_to_KV_max, flash_attn_stream_k_fixup_*, flash_attn_combine_results.
  - norms/rope: rms_norm_f32 (+fused add), rms_norm_mul_rope_f32 (fused QK-norm+rope+set_rows), rope_neox, k_set_rows,
    k_bin_bcast (residual adds).
- No runtime per-op timing. -lv 5 prints "CUDA graph warmup complete/reset" (graph replay evidence, no timings).
