# SC1 — Qwen3-30B-A3B serving head-to-head on RTX 5090s: e4b (main) vs vLLM 0.30.0 vs SGLang 0.5.20 [box A] and vs llama.cpp b11327 vs ExLlamaV3 1.5.3 vs LMDeploy 0.18.0 [box B], at matched work with native rows, quality on identical tokens in the served shape and the prefill shape, TTFT, resources and energy (registered 2026-10-02, before any box is rented)

Issue: experts4bit-qlora#846. Lineage: P58 (`bench/p58/`, 2026-09-22: vLLM 0.30.0 / e4b RTN int4 = 1.087 at B=1, 1.396 at B=16), P86 (`bench/p86/`: the B=16 gap is the expert kernel), P88/P89 (K19 + K23 the int4 B=16 defaults), P94 (2026-10-02: a route licensed by the arithmetic it does NOT serve is the baseline error — K25 vs the M-tile it replaces), P37/P20 (the slope method; the distinct-prompt rule). Upstream facts: `bench/sc1/UPSTREAM-NOTES.md` (read from source at the pinned tags). Design review: `bench/sc1/DESIGN-REVIEW.md` — three independent adversarial rounds on successive drafts (round 1: 6 HIGH / 13 MED / 7 LOW; round 2: 2 HIGH, one of them a misreading of the harness by the registrant that the second draft had been built on; round 3: 4 HIGH, all in the text's arithmetic and predicates); every HIGH is closed in this text, each MED closed or carried as a stated limit; the disposition table is in that file.

## Why this lane, and what it corrects

1. **P58 compared the comparator's calibrated checkpoint with e4b's UNCALIBRATED stack** (P54's RTN experts + RTN attention, a speed-only configuration that fails the K8 gate on c4val1). SC1 builds the licensed pack on the box, gates it there — **for both arithmetics it serves** (the T == 1 GEMV at B=1 and K19 at T > 1 for B=16; P94's lesson) — and ratios the comparators against that. The RTN arm stays as a labelled speed row.
2. **No quality was ever read on a comparator.** Every arm here scores identical token windows two ways, in nats, against the bf16 checkpoint and against the arm it is ratioed with; the quantization contract (format, bits/weight computed from bytes/params, group, calibration) is a column.
3. **The register's e4b number is a bare graph-replay window; vLLM's is a serving loop.** SC1 adds e4b arms timed THROUGH its scheduler by the same slope method and takes the ratio on those; the window is reported as the kernel ceiling.
4. **Only decode, only vLLM.** TTFT at 512 and 4096, VRAM / host / PCIe / energy, and five comparators on two boxes.
5. **The defaults moved** (K19, K23, bucketed graphs since 0.38.0; P58 read 0.37.x).

## Claim under test

On one RTX 5090 per box, Qwen3-30B-A3B at the pinned revision, identical prompt token ids per row (one 512-token wikitext window at B=1; sixteen DISTINCT windows at B=16), greedy, fixed output length, no prefix-cache reuse across requests on timed arms, no speculative decoding, two interleaved draws per arm (a third when the first two disagree by > 3 %):
(a) B=1 decode ms/step and B=16 aggregate tok/s per engine at MATCHED work, with the engine's NATIVE recipe beside it, labelled;
(b) TTFT at 512 and 4096 prompt tokens, B=1: the wall of a `max_tokens=1` request at the engine's entry point, warm, uncached, median of 3 — the same definition on every engine, with each engine's own timers beside;
(c) the quality of what each engine served, on the same tokens: teacher-forced NLL on two texts, in the **served shape** (the generated position after a cached prefix: T == 1 where the engine's cache is token-granular, a partial block where it is not — RECORDED per request) and the **prefill shape** (one call), vs the bf16 checkpoint and vs the ratioed arm, in nats;
(d) VRAM (reserved / peak allocated / weights + KV demand), host RSS and CPU share, PCIe rx/tx, and J/token over a dedicated 1024-token decode window;
(e) which engines serve the model on this card in the registered container at all (a refusal, an install failure or an OOM is a row).

## Three boxes, shared same-box anchors

One 5090 cannot hold six engines, ~60 timed arms, ten quality scorings and a 61 GB offloaded bf16 oracle inside a guard under the no-ask tier, and a box that runs out of clock eats the arms a reader needs most. SC1 is three draws, launched as separate runs (`sc1a-5090-n`, `sc1b-5090-n`, `sc1c-5090-n`), each with its OWN anchors, every ratio within its box. The round-2 budget arithmetic put six engines on two boxes at 5.9 h and 5.1 h against 5.5 h and 5.0 h guards — the deadline would have eaten every comparator's second draw — and SGLang 0.5.20 is CUDA-13-only while the registered image is CUDA 12.9 (chosen for e4b's pins), so an SGLang refusal there would be a fixture artefact, not a finding. Box C is the honest answer, not an afterthought:
- **Box A (`SC1_BOX=A`, AMD host, image `pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel`):** e4b — NF4 control, LICENSED pack (built + gated here), RTN (labelled), the scheduler-slope arms, the controls, the energy windows; vLLM 0.30.0 (graphs kv auto / kv fp8 / AWQ / SAMEPROMPT / the detokenisation pair); vLLM's quality scorings; the e4b prefill-shaped quality rows.
- **Box B (`SC1_BOX=B`, same image):** the anchors re-measured on this box — `e4b/int4_b{1,16}` (window) AND `e4b/int4_sched_b{1,16}` (scheduler slope) on the RTN pack, `vllm/gptq_graph_b{1,16}`, two draws each — then the bf16 upstream oracle on both windows (right after the anchors: box A's whole Δ_bf16 axis depends on it), llama.cpp, ExLlamaV3 (cu128 wheel), LMDeploy at B=1/B=16, their quality scorings, TTFT for every engine on the box.
- **Box C (`SC1_BOX=C`, image `nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04` + python 3.12; e4b installed on torch 2.8 cu128 wheels in its own venv — the pins the register's numbers were read on, driver >= 580 either way):** the same anchors (e4b RTN window + sched, vLLM GPTQ) re-measured, then SGLang 0.5.20 at B=1/B=16 (matched + native), its quality scorings, ExLlamaV3's cu13 wheel as a labelled native row, TTFT for the engines on the box. SGLang is ratioed only against box C's anchors; nothing about SGLang is said from box A.
- **The RTN-as-anchor substitution on boxes B and C is licensed by box A**, not assumed: box A measures `lic` and `rtn` at both batch sizes; the substitution holds iff |lic − rtn| <= 2 % at B=1 AND B=16 on box A (P12 predicts < 3 %; the 5 % UNSTABLE bar is not the bar here — an anchor allowed to run 5 % fast flatters every ratio on the other boxes). The measured gap is printed on every box-B/C e4b row regardless; if it exceeds 2 %, those anchors are labelled "RTN speed, X % from the licensed pack" and their ratios carry it.
- The three boxes' vLLM/e4b anchor ratios are reported side by side (expected within 5 %; a larger gap is a host finding, never averaged). B=1 is host-bound; the host CPU model, `nproc`, cgroup `cpu.max` and load average are in every receipt.

## Fixture

- **Box:** Vast.ai RTX 5090 verified/secure, image `pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel`, >= 320 GB disk, >= 98 GB host RAM, >= 40 MB/s pre-flight, **NVIDIA driver >= 580** (the CUDA 13 wheels vLLM 0.30.0 and SGLang 0.5.20 ship; a pre-flight refusal, lane rc 18, the launcher excludes the machine); box A additionally **AuthenticAMD** (the pack's calibration build is CPU-bound; P87/P88). Vast machine 147733 is excluded by receipt (two bandwidth refusals 2026-10-01).
- **Cut:** e4b `heads.e4b` = this registration's merge commit or later (must include `experts4bit_qlora.serve_paged`, e4b#848's scheduler stop set and e4b#853 the server and e4b#854 its fusion fix); grouped-nf4-gemm **v0.34.1 = `34da93d6fe8d2a401b7001705658ce00b2b18213`**; transformers 5.16.1 / bitsandbytes 0.50.1 in the e4b venv (P58's pins); the P42 hook, P39's `step_decomp.py` / `k8_bake.py` / `calib.json` byte-identical (`staged.sha256`). **Every e4b arm names its route knobs explicitly** — `E4B_INT4_GROUPED_SMALLM`, `E4B_INT4_LEAN_GLUE`, `E4B_NF4_GROUPED_SMALLM`, `E4B_MXFP4_GROUPED_SMALLM` — at the pinned main's default (auto / auto / 0 / auto as of 2026-10-02, P94 read QUALITY_FAIL so the NF4 default stays 0), never inherited.
- **Fusion sets, per arm, named.** `qkv_fuse.fuse_qkv()` applies the three env-gated folds ITSELF after fusing q/k/v (`qkv_fuse.py:141-151`); the harness's `--no-fuse-qkv` branch calls them directly for the families it cannot fuse. So the registered Qwen3 B=1 serving stack (P54/P58/P88: `--fuse-qkv` with the three fold flags set) is **fused q/k/v WITH the folds**, and that is SC1's `lic_b1`. Its licence is the K8 rows (`--no-fuse-qkv` + folds, P88's `K8ARGS`), bridged by one documented fact: the fusion is byte-identical at B=1 (independent per-row dot products, P54) and KL-exactly-zero through the K16 small-M route at B=16 (P59) — stated in one sentence beside the row. The B=16 arms run the same fused set. The sched arms mirror it through `serve_paged` (`E4B_PAGED_FUSE_QKV=1` + the fold flags; the server's refusal of that combination was a defect found by the round-2 review and fixed in e4b#854 before this registration; its census now reports the fold counts `fuse_qkv` captured, never a literal zero). The window-vs-sched "same stack" predicate is concrete and has a side on BOTH arms: (i) `fuse_qkv_n == 48` and equal `INT4EXP` layer / `ATTNINT4` projection counts (both arms print them); (ii) the pack fingerprint (both); (iii) the fold counts the sched census captures through `fuse_qkv` must meet the structural minimums for Qwen3-30B-A3B (router epilogue 48, round-2 attentions 48, round-1 norms > 0) and be identical across every sched arm — the window arm prints no fold counts (`step_decomp` and `fuse_qkv` discard the returns; the fold modules log nothing), so the window side of (iii) is read from the KERNEL profile: the fold kernels named in `glue_fuse.py` / `glue_r2.py` / `router_epilogue.py` must appear with equal per-step counts in the window arm's `--replay-profile-out` and in an 8-step profile of the sched engine taken the same way; (iv) a tripwire before Phase 0 that `serve_paged._apply_fusions` ACCEPTS `E4B_PAGED_FUSE_QKV=1` with the fold flags (the e4b#854 fix) so the Phase-0 smoke is not its first test. `DECODE_GRAPH` lines belong to the sched engine only (the window arm captures its own single-step graph and prints its own banner).
- **Model, e4b:** `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` (bf16 → NF4 arena by `k8_bake.py`; packs built at arm start). The licensed recipe is P88's `LICENV` (calibrated experts, `E4B_CALIB_SOURCE=c4`, `E4B_CALIB_NSEQ` as P88, calibrated int4 attention, the three folds) built ONCE on box A with `E4B_INT4_DUMP_ARTIFACT_DIR`, `verify_artifact`, the fingerprint recorded; every later `lic` arm loads it by `E4B_INT4_ARTIFACT_DIR` + `E4B_INT4_EXPECTED_FINGERPRINT`. The RTN recipe is P88's `SPEEDENV`.
- **Model, comparators:** vLLM / SGLang / LMDeploy: `Qwen/Qwen3-30B-A3B-GPTQ-Int4` @ `9b534e4318b7ebc3c961a839f13eb18b1833f441` (official; 4-bit, g128, sym, desc_act false — loads under 0.30.0's act-order removal); vLLM AWQ: `QuixiAI/Qwen3-30B-A3B-AWQ` (AutoAWQ gemm, g128, asym zero-point, gate + lm_head excluded; the only AWQ of the BASE model; revision pinned at registration; routed to `awq_marlin`); llama.cpp: `unsloth/Qwen3-30B-A3B-GGUF` `Q4_K_M` (18.56 GB; gate/up Q4_K + down Q6_K) and `IQ4_XS` (16.38 GB); ExLlamaV3: `turboderp/Qwen3-30B-A3B-exl3` branch `4.0bpw` = commit `0b83e92c6d3b5a868ecd5a5fbb3bcc1920e388ef` (14.88 GiB; **head_bits 6** — only the 8.0 bpw file is H8; calibration 100 × 2048). File shas recorded at fetch. **Bits per weight are COMPUTED per arm from expert-store bytes / expert parameters** (e4b int4-b32: 4.50 with its 16-bit scale per 32; GPTQ/AWQ g128: ~4.15; Q4_K_M 4.86; IQ4_XS 4.29; EXL3 ~4.0 + head bits) and carried as a column: e4b reads ~8 % MORE expert bytes per token than the GPTQ checkpoint.
- **Prompts:** `prompts_b1.json`, `prompts_b16.json` from `step_decomp._k8_window` (wikitext-2 test, 512-token rows, 16 distinct windows, sha per row and per file — P58's dump), fed to every engine as TOKEN IDS; every receipt asserts the engine saw exactly 512 prompt tokens per row (its usage / prompt-token count) — a BOS or template silently added is a VOID. `prompts_b1_4096.json` (one 4096-token window) for TTFT. `prompts_b16_same.json` = 16 copies of row 0 (the SAMEPROMPT control; its receipt says so).
- **Quality windows:** `k8_window_wikitext.json`, `k8_window_c4val1.json` = `ids[:2561]` from `_k8_window` per source with its `text_sha`; a quality row whose sha differs from the window file's is VOID — the sha, not any reference value, pins the window (P55x read this window's NF4 K8 at 1.85939 on 2026-09-22 against the register's 1.8621: software moves it by 0.003 nats; the NF4 control's value is REPORTED with that context, never used as a VOID condition).
- **Capacity matched:** every engine is configured to hold 16 sequences × 2048 tokens (e4b `max_tokens_per_seq` 2048 × 16 slots + scratch; vLLM `max_model_len 2048`, `max_num_seqs = B`; SGLang `--context-length 2048 --max-running-requests 16`; llama.cpp `-c 32768 -np 16`; ExLlamaV3 cache 16 × 2048; LMDeploy `session_len 2048`, `max_batch_size = B`); the TTFT-4096 arm runs a separate single-sequence configuration sized 4096 + 8 on each engine, stated per receipt.
- **Generation:** greedy, fixed `max_tokens` with EOS ignored (`ignore_eos` / `min_tokens` equivalents; a row that stopped early is not the registered workload); SHORT = 32, LONG = 128; min-of-3 and median-of-3 (P20). Decode-time work is matched on prompts, not on the tokens each engine routes (divergent continuations route to different experts) — stated.

## Environments (every version in `versions.txt` and every receipt)

- e4b: image python + P58's pins; gnf4 at the pin; tripwire asserts the K19 route present and defaulted, K23 auto, `Int4Linear.fuse`, bucketed graphs, `serve_paged.build_engine` importable, `ContinuousScheduler.add_request` takes `stop_ids`.
- vLLM **0.30.0** (PyPI, the CUDA 13.0 build; installs on this image per P58): GPTQ/AWQ MoE route to **Marlin MoE** (`Using 'MARLIN' WNA16 MoE backend.`), attention projections `MarlinLinearKernel`; attention backend on sm_120 = FA2 for bf16 KV and **FLASHINFER for fp8 KV** — pinned per arm with `attention_backend`; `moe_backend="marlin"` pinned; prefix caching OFF on timed arms (block-granular, block 16); `max_num_seqs = B` (16 on the B=16 arms — the 16 × 2048 capacity configuration — and 1 on the B=1 arms) so the capture list is `[1,2,4,8,16,24,32]` at B=16 (recorded); `detokenize=True` on every matched arm (a comparator's loop is never trimmed to e4b's omission; the `nodetok` pair at both B measures the asymmetry); `seed=0`; fp8 KV arm: `k_scale=v_scale=1.0` by default when the checkpoint has none — recorded as a quality hazard and SCORED.
- SGLang **0.5.20** (`pip install sglang==0.5.20`; CUDA-13-only; JIT Marlin MoE needs the host `nvcc` >= 12.9 → `sm_120f`; `--moe-runner-backend auto` REQUIRED; flashinfer attention; capture point 16 native; radix cache token-granular). It runs on box C (CUDA 13 image), where its JIT has the toolchain it documents; box C's proving rental answers whether the Marlin MoE JIT builds and the server reaches `/health` on sm_120 before any paid arm.
- llama.cpp **b11327** (`552f18f9…`), built with `-DGGML_CUDA=ON -DGGML_NATIVE=OFF -DCMAKE_CUDA_ARCHITECTURES=120a-real`; `llama-server -np 16 --cont-batching -fa on -ngl 99 -ctk f16 -ctv f16 --temp 0 -lm mmap -c 32768` with token-id prompts and `cache_prompt: false` on timed arms; MoE `mul_mat_id` runs **MMVQ (dp4a) at <= 8 tokens and MMQ (int8 mma) at 16/32** — B=1 and B=16 are different kernels by design, stated; `llama-batched-bench -npl 1,16` beside it (native instrument, labelled); quality through `bench/sc1/llamacpp/nll_teacher_forced.cpp` (libllama; decode mode = one `llama_decode` per true token; prefill mode = one batch), built from the same tree with the server's runtime (FA on, f16 KV, full offload), calibrated on CPU against transformers before registration (calibration delta in the receipt).
- ExLlamaV3 **1.5.3** (release wheel `exllamav3-1.5.3+cu128.torch2.8.0` on box B — no flash-attn/xformers dependency since 1.0.0, attention is its own Triton kernels with graph-captured `BC_Attention` at decode; the `cu132.torch2.13.0` wheel's +13–23 % as a labelled native row on box C). Generation via `Job(input_ids, max_new_tokens=N, min_new_tokens=N, sampler=ArgmaxSampler(), stop_conditions=None)` — one `model.forward` over all active jobs per step, so B=16 is one batched step; its page cache has no off-switch, so every timed call uses a fresh `Generator` and records `cached_tokens` per job. MoE kernels: `exl3_moe_coop_*` at B=1 (`bszN <= 4`), `exl3_moe_kernel` + gather at B=16 — different kernels by design, stated. Quality: `model.forward(..., attn_mode="flash_attn_nc")` prefill-shaped and a true T == 1 decode loop (`model.prefill` then one forward per token) — `mode_available: [prefill, decode]`.; LMDeploy **0.18.0** (`pip install lmdeploy==0.18.0` on a pre-pinned torch 2.8 cu128; TurboMind W4A16 on the GPTQ checkpoint through the raw `TurboMind.from_pretrained().create_instance().async_stream_infer` route with `top_k=1`, `min_new_tokens=N` and the EOS ids masked — the route its own `profile_throughput.py` drives; `Yes*` for Qwen3-MoE smoke-tested in the proof). Facts that shape its rows: **TurboMind has no native sm_120 kernels — `arch.h:38` falls back to the SM80 `s16816` family**, the same grouped W4A16 GEMM at B=1 and B=16; **no CUDA graphs**; **fp8 KV refused** (`quant_policy` 0/4/8 only); engagement banner `TM_GEMM_VERBOSE=1` → `[Gemm] … sm80_f16_u4k128_f16 …`. Quality: prefill logits via `output_logits='all'` in the engine dtype (fp16/bf16, up-cast and recorded) with the engine's own fp32 `ce_loss` as a cross-check; **no T == 1 path exists** — its served-shape row is 2048 one-token extensions with prefix caching on, labelled `decode_prefix (partial-block, M <= 64)`, never T == 1. Both engines' install specs, tripwires, API routes and kernel names are in `UPSTREAM-NOTES.md` with path:line citations from their driver inspections.
- Stated absences: TensorRT-LLM (sm_120 absent from its supported-hardware list; NVFP4-only on consumer Blackwell with community patches; CUDA 13.1 install — first-class on SC4's H100), mistral.rs (open 5× decode regression on CC 12.0, #2093), Ollama (a llama.cpp wrapper).
- bf16 reference: `step_decomp.py --ppl-oracle upstream` (transformers bf16, `device_map="auto"` over the card + host RAM, eager attention, chunked teacher forcing: prefill-shaped) on both windows, box B.

## Arms and their order (the deadline eats from the END; minutes are the budget, from P58/P86/P88 walls)

**Box A** (guard 5.5 h = 330 min; `can_run`'s 10-min tail leaves 320; the phases below sum to **294 min at the 40 MB/s pre-flight floor** plus a 10-min designated droppable — the table is the budget, not a hope):
- Phase 0 (~83 min at the floor): fetch the bf16 checkpoint FIRST and ALONE (61 GB: 25 min at 40 MB/s, 10 at 100); then the GPTQ fetch (17 GB) and the vLLM venv overlap the bake; bake the NF4 arena (15); a `serve_paged.build_engine` smoke on THAT arena with both fusion sets and graphs, 16 tokens at B=1/B=16, census recorded (5) — the proving rental's Granite build cannot exercise `fuse_qkv` (it matches `Qwen3MoeAttention` only), so this is where the fused sched path first meets the GPU, BEFORE the pack build spends 35 min; BUILD the licensed pack (P88's `k8 build`, ~35 on AMD, first-chunk watchdog 1500 s) with **the host quiesced** (comparator venvs, fetches and any JIT finish before the build starts, not merely before Phase A; the runner waits and records `nproc`, `cpu.max`, loadavg per phase); `tests/test_k19_row_exact_gpu.py` on the card (3; rc 25 if it fails — the K19-at-T==1 premise).
- Phase A (~20): `e4b/lic_b16_r1`, `vllm/gptq_graph_b16_r1`, `e4b/lic_b1_r1`, `vllm/gptq_graph_b1_r1` (the registered fused-with-folds set on both e4b arms).
- Phase A2 (~10): `e4b/lic_sched_b16_r1`, `e4b/lic_sched_b1_r1` — `serve_paged.build_engine` on the same arena / pack-by-fingerprint / knobs / fusion set (graph buckets = the registered list truncated at B, recorded), timed through `ContinuousScheduler.step()` by the 32→128 slope; **the ratios against vLLM/SGLang are taken on these**; the window arms are the kernel ceiling. A sched arm whose `/health`-equivalent census (int4 layers, projections, fusions, graph buckets) differs from the window arm's banners is VOID.
- Phase B (~40): the licence ON THIS BOX — K8 on the licensed pack (by fingerprint) at `E4B_INT4_GROUPED_SMALLM=auto` (T == 1 = the GEMV, the B=1 arithmetic) AND at `=1` with `E4B_INT4_LEAN_GLUE` left at its default (`auto`), exactly as `test_k19_row_exact_gpu.py` sets the pair — the value is RECORDED in the receipt (K19 + K23 at T == 1 = the B=16 arithmetic; the receipt's census must show both engaged — the first NLL read of K19 and K23 together, P89 licensed the glue by 128-token identity only), wikitext + c4val1 (4 × ~6); the NF4 control K8 on both texts (2 × ~3); because `step_decomp`'s eager K8 loop carries no kernel census, one extra graph-window arm on the SAME `=1` stack (`e4b_k19census_b1`, `--replay-profile-out`) supplies the K19 + K23 kernel names the `=1` rows are gated on; `k8_gate.verdict(calibrated=True, calibration_domain="c4val1")` — the label the arms carry as `ppl_source`, so the clause binds: c4val1 is IN the calibration domain (e4b calibrates on a c4 validation shard) and wikitext is the outside text. LICENSED-B1 iff the `auto` rows pass; LICENSED-B16 iff the `=1` rows pass; the B=16 position is quotable only with LICENSED-B16. QUALITY_FAIL → the `lic` rows are labelled and **no position is quoted from the RTN rows either** (P58's headline is not restored by a side door).
- Phase D-vLLM (~20): vLLM quality on both windows, prefill-shaped (`prompt_logprobs=0`, one call) and served-shape (2048 sequential `max_tokens=1` requests with prefix caching ON and `logprobs=-1`, each reading log p(true next token) at the GENERATED position; `num_cached_tokens` per request is the validity predicate — the recomputed suffix histogram is in the receipt and the row is labelled `partial-block (M <= 16)`); the e4b prefill-shaped rows: `--ppl-oracle eager` on the `LICENV` + pack (both texts, ~3 each) so e4b has both shapes too.
- Phase G1 (~30): second draws of Phase A and A2.
- Phase C (~28): `vllm/gptq_fp8kv_b{16,1}` (the attention backend vLLM selects for fp8 on sm_120 is RECORDED — the row changes KV bytes AND possibly the attention kernel, so P8 reads the fp8-KV *path*, not KV bytes alone), `e4b/rtn_b{16,1}`, `e4b/nf4_ctrl_b{16,1}` (first draws only). The AWQ arm is CUT from SC1 (its fetch + two arms + second draws were ~27 min, it is the only third-party checkpoint, and no position against e4b depends on it; SC1b may carry it).
- Phase F (~15): controls — `vllm/gptq_graph_b16_SAMEPROMPT` and `e4b/lic_sched_b16_SAMEPROMPT` (16 copies of row 0 on BOTH engines; the routing-collapse detector; the consequence is a LABEL on the distinct rows' interpretation, never a VOID — rows are distinct by sha); `e4b/lic_b16_degraded` (`E4B_INT4_GROUPED_SMALLM=0`: must read >= 1.05× slower than `lic_b16` AND the census must show the K19 kernel absent — else the instrument cannot see the kernel it times and e4b's B=16 rows are VOID). The eager pairings are dropped (P58 carries them).
- Phase E (~8): TTFT 512 / 4096 on e4b (sched engine, `max_tokens=1`) and vLLM.
- Phase EN (~12): energy windows — `lic_sched_b1`, `lic_sched_b16` (the sched engine, the speed axis), `vllm_b1`, `vllm_b16` decoding 1024 tokens under one `nvidia-smi --query-gpu=power.draw.instant,memory.used,utilization.gpu,clocks.sm,pcie.link.gen.current --format=csv -lms 50` sampler; J/token = ∫P dt / tokens over the window; board power, stated.
- Phase G2 (~28 + 10): second draws of fp8kv, rtn and the controls, interleaved (28); then the droppables in this order: the detokenisation pair `vllm/gptq_graph_b{1,16}_nodetok` (10; the matched vLLM arms keep `detokenize=True`; this pair measures the asymmetry at both batch sizes), `e4b/nf4_ctrl_b{16,1}` second draws, and the fp8-KV served-shape quality rows (P8's quality half); a THIRD draw of any arm whose two draws disagree by > 3 % (the point is the median of three; UNSTABLE if the spread exceeds 5 %).
  Sum at the floor: 83 + 20 + 10 + 40 + 20 + 30 + 28 + 15 + 8 + 12 + 28 = **294**, plus the 10-min nodetok droppable = 304, against 320 usable.
Alarms: bake 5400, build 5400, each timed arm 1800 (the fp8-KV arms 2400: FlashInfer's first-use JIT runs inside the first arm), each quality arm 2400, the oracle 3600, each install 2700. An arm that cannot fit is a `not_run` stub `host-limited`.

**Box B** (guard 5.0 h = 300 min, 290 usable; budget **~275 min at the floor**): Phase 0 (fetch the bf16 FIRST and alone — the bake needs it — then the GPTQ, two GGUFs and EXL3 overlap the bake: 128 GB of downloads ≈ 53 min of pipe at the floor, ~30 of it before the bake can start; bake 15; RTN packs at arm start; llama.cpp CUDA build ~10, ExLlamaV3 + LMDeploy venvs — all quiesced before the first timed arm) → anchors `e4b/int4_b{16,1}` window + `e4b/int4_sched_b{16,1}` (RTN pack, the fused set `--fuse-qkv` + folds) + `vllm/gptq_graph_b{16,1}` (~30) → **the bf16 upstream oracle, both windows (~25)** → `llamacpp/q4km_b{16,1}`, `exl3/4bpw_b{16,1}`, `lmdeploy/w4a16_b{16,1}` (~36), `llamacpp/iq4xs_b1` (single draw, speed only, ~4) → quality for llama.cpp / ExLlamaV3 / LMDeploy, both windows, prefill + served (~50) → anchors' second draws (~30) → TTFT 512 / 4096 on every engine present (~15) → energy windows, B=1 only, three engines (~6; the llama.cpp / ExLlamaV3 / LMDeploy drivers have no dedicated 1024-token mode, so their J/token is integrated over the whole timed arm after load — `basis` says so in the receipt and the box-A e4b/vLLM windows are the like-for-like pair) → second draws of the six comparator arms (~24). `llama-batched-bench` is dropped (the server slope is the instrument; its row would be a third timing method). LMDeploy's arms sit last among the comparators (the `Yes*` risk). Box B's engines are ratioed to box B's `e4b/int4_sched` anchors (with the substitution licence from box A); their window ratio is reported beside.

**Box C** (guard 4.0 h = 240 min, 230 usable; budget **~211 min at the floor**): fetch the bf16 checkpoint FIRST (`k8_bake.py` bakes the NF4 arena from the bf16 safetensors — box C's e4b anchors need it as much as box A's; 25 at the floor) → GPTQ + EXL3 fetches and the venvs overlap the bake (15) → anchors as box B, the fused set named: `e4b/int4_b{16,1}` window + `e4b/int4_sched_b{16,1}` (RTN pack, `--fuse-qkv` + folds) + `vllm/gptq_graph_b{16,1}` (~30) → SGLang install + first JIT (~25, then quiescence) → `sglang/gptq_matched_b{16,1}` + `sglang/gptq_native_b{16,1}` (~16) → SGLang quality both windows, prefill (`return_logprob`, `logprob_start_len 0`) and served (generation position, `token_ids_logprob`, radix ON, `cached_tokens` predicate; T == 1) (~20) → `exl3/4bpw_cu13_b{16,1}` (labelled native, ~8) → TTFT 512 / 4096 (~6) → second draws of the SGLang arms (~16) → **second draws of the anchors (~30)** — single-draw anchors would be UNSTABLE-unread and no SGLang position could be quoted. Rate ceiling $0.75/h → ≤ $3.00.

## Validity (VOID never enters a ratio or a quality reading)

- Every arm: prompts sha == the file's; prompt-token count per row == 512 as the engine counted it; generated tokens == max_tokens × batch; the two draws within 5 % (else a third; spread > 5 % = UNSTABLE, no position); no dynamo recompiles inside e4b's window; the host quiesced (recorded loadavg) when the arm started.
- e4b lic (window arms): `INT4EXP licensed artifact <fp>` with the Phase-0 fingerprint, calibrated-attention banner (192), `fused q/k/v projections on 48 attention modules`, the arm's own graph banner, K19 engaged (census kernel name `gemm_int4_b32_grouped_smallm` present at B=16, absent in the degraded arm); sched arms: `build_engine`'s census — `fuse_qkv_n == 48`, equal int4 layer / projection counts, equal fold counts (captured through `fuse_qkv`), the same fingerprint, `DECODE_GRAPH bucket=1|16 captured`; K8 rows: `text_sha` == the window file's, `steps == 2048`, `attn_compute` from the mech tally, the knob values named, and for the `=1` rows the K19 + K23 census lines.
- vLLM: `Using 'MARLIN' WNA16 MoE backend.`, `Using MarlinLinearKernel` (GPTQ; the AWQ arm reads `resolved.quantization == auto_awq` + the MoE line), `Graph capturing finished` on graph arms, the resolved `attention_backend` (pinned FLASH_ATTN on bf16-KV arms; recorded as selected on the fp8-KV arm), `enable_prefix_caching False` on timed arms and True on the served-shape quality arm (recorded), `max_model_len 2048`, `max_num_seqs == B` (16 on B=16 arms, 1 on B=1 arms — one rule in every place it is stated), `detokenize == True` on matched arms, capture sizes.
- SGLang (box C): no JIT log line exists at the tag — engagement is the on-disk JIT artefact `$SGLANG_JIT_CACHE_DIR/sm120f/sgl_kernel_jit_moe_wna16_marlin_*/...` with `-gencode=arch=compute_120f,code=sm_120f` in its `build.ninja`, the banner `The model is convertible to gptq_marlin during runtime. Using gptq_marlin kernel.`, and `attention_backend == flashinfer` in `/server_info`; llama.cpp: `offloaded 49/49 layers`, flash attention enabled, `n_parallel 16` on B=16, `GGML_CUDA_MMQ_PREC` unset (recorded); ExLlamaV3: the tripwire (`exllamav3 == 1.5.3`, the `exllamav3_ext` symbols, `is_precompiled_extension_available()`, sm_120 in the torch arch list), per arm the `quantization_config` + `get_storage_info()` bpw dump, `EXL3_BC_ATTN_TRACE=1` output on decode arms, `cached_tokens == 0` per job on timed arms (a fresh `Generator` per call), and a profiled step naming `exl3_moe_coop_*` (B=1) / `exl3_moe_kernel` (B=16); LMDeploy: the tripwire (`lmdeploy.turbomind.is_available()`, `_get_executable_dtypes([GPTQFormat(block_in=128)], 0)` non-empty), per arm `TM_GEMM_VERBOSE=1` lines naming the `sm80_f16_u4k128_f16` family (the SM80 fallback TurboMind runs on sm_120, recorded as such), `max_batch_size`, `cache_max_entry_count`, `quant_policy 0`; a converter refusal is UNSUPPORTED with its message.
- Controls: SAMEPROMPT faster than distinct on an engine → that engine's distinct rows are confirmed distinct (label); NOT faster → the detector did not fire, stated, no VOID; degraded ≥ 1.05× slower + census; the bf16 oracle reading ABOVE a 4-bit arm by more than 0.02 nats on in-distribution text is flagged as the OOD-flattery signature (instrument fault), smaller improvements are K8's documented calibrated-improvement outcome.

## Verdict columns (mechanical, per arm per attempt)

- **Speed:** VALID · VOID · OOM · UNSUPPORTED (install / import / load / sm_120 / container refusal) · UNSTABLE (flag) · non-readings HARNESS_ERROR / ALARM / NOT_RUN.
- **Licence (e4b-internal, one-sided K8 vs the NF4 control):** LICENSED-B1 / LICENSED-B16 / QUALITY_FAIL.
- **Comparability (cross-engine, signed, in nats per text, both shapes):** vs the bf16 oracle Δ_bf16 and vs the ratioed arm Δ_pair: **CLOSE** |Δ_pair| ≤ 0.0095 (the register's arithmetic-order floor) · **COMPARABLE** ≤ 0.02 · **NOT_COMPARABLE** otherwise (the signed gap in the sentence). A position needs COMPARABLE on both texts in the served shape; CLOSE is stated when met.

## Readings (pre-registered)

- **Position** per comparator per batch size: other / `e4b/lic_sched` (box A) or / `e4b/int4_sched` (box B) as the point (median over draws) AND the interval over the cross-draw ratios; quoted only when both arms VALID, not UNSTABLE, COMPARABLE; the comparator's contract (format, bpw, group, calibration, resident GB, KV dtype, graphs, prefix cache state) and the matched-vs-native flag delta in the same sentence. The window ratio beside it, labelled "kernel ceiling".
- **Quality table:** per arm × text × shape: mean NLL, ppl, Δ_bf16, Δ_pair, top-1 agreement where available, cache granularity and recomputed-suffix histogram; per engine the served-vs-prefill delta (the engagement reading).
- **TTFT table; resources (three VRAM columns); energy table** (J/token per engine per batch from the 1024-token windows; absolute W is a board reading).
- **The frontier figure** (primary): per batch size, every VALID arm as a point (tok/s on the sched axis, Δ_bf16 served-shape) labelled with bpw — the map.
- **Method pair:** e4b sched vs window on the same stack per box (expected 3–15 % at B=1); cross-box: the two boxes' anchor ratios side by side.

## Predictions (falsifiable; competitor estimates biased UP; the bands are on the SCHED ratio)

P1 B=1 vLLM / e4b-lic-sched in [0.90, 1.15] (P58 read 1.087 on the window; the scheduler adds 3–15 % to e4b). P2 B=16 in [1.00, 1.30] (P86 1.38 before K19/K23: ~9.8 ms predicts ~1.12 on the window). P3 (box C) SGLang within 10 % of box C's vLLM anchor at B=1 and 15 % at B=16; UNSUPPORTED if its Marlin MoE JIT does not build on sm_120 even with the CUDA 13 toolchain. P4 llama.cpp Q4_K_M B=1 in [0.8, 1.15] of the anchor (dp4a MMVQ, 4.86 bpw), B=16 in [0.35, 0.7] (int8 MMQ, 16 slots). P5 ExLlamaV3 B=1 in [0.9, 1.4], B=16 in [0.5, 1.1]; P5b LMDeploy B=1 [0.9, 1.3], B=16 [0.9, 1.4] or UNSUPPORTED. P6 served-vs-prefill NLL differ by > 0.005 nats on e4b-lic (T == 1 levers) and by < 0.002 on vLLM. P7 SAMEPROMPT reads faster than distinct on vLLM by >= 1.3× and on e4b-sched by >= 1.3× (the P86 byte arithmetic suggests ~1.9×). P8 vLLM fp8-KV within 5 % of kv auto at B=1, faster at B=16, and its Δ_bf16 larger than kv auto's by < 0.01 nats. P9 the pack LICENSES at `auto` on this box (fingerprint reproduces) and at `=1` (P88 read +0.0062 nats, inside the floor). P10 TTFT-4096 vLLM / e4b in [0.5, 1.0]. P11 J/token at B=1 within 15 % between vLLM and e4b; at B=16 the faster engine is the more efficient. P12 the substitution licence holds (|lic − rtn| <= 2 % at both B). P13 the cross-box anchor ratios agree within 5 % across all three boxes. P14 vLLM `nodetok` reads within 3 % of the matched (`detokenize=True`) arm at B=1 and within 1 % at B=16.

## Decision rules

- P1/P2 hold with COMPARABLE quality: the register's current-vs-current serving comparator moves to SC1's rows (`e4b.serve.h2h.sc1.qwen3.<engine>.{b1,b16}.5090.<date>`, point + interval, sched-ratioed), superseding `e4b.serve.h2h.vllm-0.30.0.p58.qwen3.{b1,b16}.5090.2026-09-22` (kept as measured with its RTN/window note).
- A comparator faster than e4b at COMPARABLE quality is a loss with its cause named by SC1b's census before any README sentence changes; a comparator faster at NOT_COMPARABLE quality is a point on the frontier figure, not a loss and not a win.
- Nothing moves a gate, a licence, a kernel default or a claim id; SC1 reads.

## Budget and stop rules

Three RTX 5090 draws (A: guard 5.5 h ≤ $4.13 at $0.75/h; B: guard 5.0 h ≤ $3.75; C: guard 4.0 h ≤ $3.00 — total ≤ $10.88), at most two concurrent when the other campaign leaves slots free, else sequential. **Proving rental per BOX** (guards A 0.75 h ≤ $0.56, B 1.0 h ≤ $0.75, C 1.0 h ≤ $0.75 at $0.75/h — Amendments A1/A2 below; each ≤ 1 h, so none needs a governance proving run of its own, and each box launches only after the OK proof on its own image and the same e4b commit; A, B and C install different comparator sets, so three proofs; the fourth, SGLang's JIT, rides on box C's proof with `SC1_PROVE_SGLANG_MODEL` set): pre-flight; that box's comparator venvs' install + import tripwires on sm_120; AND the e4b paged engine built end to end on **ibm-granite/granite-3.1-3b-a800m-instruct** (6.6 GB bf16: fetch 1 min, NF4 bake 2 min, `serve_paged.build_engine` with graphs, a 32-token slope at B=1 and B=16, the census) — the unfused set, which is what Granite admits; the fused set first meets the GPU in box A's Phase 0 smoke on the Qwen3 arena, before the pack build. The fourth proof (`SC1_PROVE=1 SC1_PROVE_SGLANG_MODEL=…` -- since A7 `Qwen/Qwen3-30B-A3B-GPTQ-Int4@9b534e4318b7ebc3c961a839f13eb18b1833f441`, 17 GB, the lane's own checkpoint; registered first as `Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4@81b132adfae58e03b96ae1ed1f0d578d0cc4d09a`, 8.4 GB — a GPTQ MoE small enough for a short proof; run in box C's PROVE pass after the Granite smokes) installs SGLang on box C's image, compiles its Marlin MoE JIT for sm_120 and reaches `/health`; it answers the JIT question before box C's paid run, as registered, without fetching Qwen3. STOP: not a 5090 (refused); driver < 580 (rc 18); a second consecutive pre-flight refusal ends the day's attempts; an arm past its alarm is a row; no relaunch changes the fixture.

## Out of scope (stated)

Request-level serving (SC2); gpt-oss-20b on identical MXFP4 bytes (SC1g); coverage families and the per-kernel census of the largest win and loss (SC1b); the capacity frontier (SC3a); a second card class (SC4); speculative decoding; prefix-cache hits on timed arms; tensor parallelism; prompts beyond 4096; sampling; energy beyond the board sensor; matching the host-CPU share across engines (recorded, not matched). What a systems reviewer would still reject after every fix — one card class, one prompt distribution, one request shape, n = 2–3 draws, in-process loops on both sides of every engine-level ratio — is the reason SC2 and SC4 exist and is said in the write-up, not footnoted.

## Amendments (each dated, each with the receipt that forced it; nothing above is rewritten silently)

- **A1 (2026-10-02T03:20Z, before any box ran; receipts `sc1a-prove-1` and `sc1a-prove-2`, adertha-receipts `21b2b93` / `cd980fe`).**
  The proving budget as first registered — "≤ 10 min, ≤ $0.15 each" (0.2 h at $0.75/h) — counted only the box's work. It did
  not count instance acquisition (measured 6.4 min on `sc1a-prove-2`: launch 03:07:08Z, box staged and pin-verified
  03:13:32Z) nor the box script's own admission rule (`can_run need+600 s`: a Granite smoke declared at 600 s is admitted only
  with 20 min left before the deadline). With a 12-minute guard the proof installed e4b and vLLM, entered PROVE, and the
  controller's deadline fired before the Granite fetch could be admitted (rc 23, $0.1087, nothing measured). The box script is
  unchanged (its pin stands); the guards move: **A 0.75 h (≤ $0.5625), B 0.75 h (≤ $0.5625), C 1.0 h (≤ $0.75; it also runs the
  SGLang JIT after the Granite smokes)**. Proofs total ≤ $1.875; the lane's three boxes are unchanged (their guards already carry the
  600 s tail and ~30 min of slack each). `sc1a-prove-1` ($0.072) was the registered rc-18 host-vendor refusal (Intel host), not a
  budget fault; its machine is a lane exclusion on every later manifest. Both receipts count toward the lane's spend.
- **A2 (2026-10-02T04:06Z, before any box ran; receipt `sc1a-prove-3`, adertha-receipts `c31437f`).** The third box-A proof returned rc 0 and
  wrote `PROVED` — and proved nothing it exists to prove: both Granite `serve_paged` smokes were SKIPPED by the deadline guard. Two
  defects in the registered box script, both fixed here (the script, its pin, the README and three CPU tests change):
  1. `quiesce` counted busy processes with `pgrep -fc … || echo 0`. procps `pgrep -c` prints `0` AND exits 1 when nothing matches, so
     the count read `0\n0`, never equalled `0`, and every quiesce waited the full `SC1_QUIESCE_S` (900 s; the receipt's line reads
     `quiesced=no waited=900s busy_procs=0\n0 load1=0.03`). The real lane quiesces twice on box A, twice on B and three times on C:
     30–45 min of each box's guard would have gone to waiting, enough to skip the late phases. Now `pgrep -f … | wc -l`.
  2. A smoke skipped by `can_run` never reached `rec`, so `rc_any` stayed 0 and `PROVED` was written with no smoke run. Now the proof
     is NOT PROVED (rc 23) when any smoke is skipped or fails, when any comparator its box installs did not install, or — on box C —
     when `SC1_PROVE_SGLANG_MODEL` is unset or the JIT + `/health` is not admitted before the deadline; `PROVED` is preceded by a
     summary line naming the installs and smokes it certifies.
  Box B's proof guard rises to 1.0 h (it builds llama.cpp before its smokes, and each smoke needs 20 min left to be admitted).
  Every proof guard stays ≤ 1 h. The receipt's launcher status reads OK / pass because the lane exited 0. The lane's reading is
  NOT PROVED, and it counts toward the lane's spend ($0.2343). The tests that pin both fixes fail on the registered script.
- **A3 (2026-10-02T06:46Z, before any box B or C ran; receipt `sc1b-prove-5`, adertha-receipts `9b44903`).** Box B's proof was NOT
  PROVED, correctly. It found three defects in SC1's own driver scripts, not in the engines. All three are fixed here: the two
  drivers, the pin and three CPU tests change, and every new test fails on the registered drivers.
  1. llama.cpp b11327 **built** in 2 min 15 s; then `llamacpp_box.sh` refused it, grepping the 8-character commit `552f18f9`
     where `llama-server --version` prints git's 7-character `commit 552f18f`. Now the printed abbreviation (7+ hex) must be a
     prefix of the pinned commit.
  2. The ExLlamaV3 tripwire read `exllamav3.version.__version__`. v1.5.3 keeps `__version__` in `exllamav3/version.py` and its
     `__init__` does not import that module (AttributeError, rc 13). Now the tripwire imports `exllamav3.version`.
  3. Found by reading v1.5.3's `bindings.cpp` while fixing 2, before any box hit it: the tripwire required the extension symbol
     `exl3_gemv_int8`, which v1.5.3 does not bind. The int8 GEMV's binding is `exl3_gemv_int8_max_k`; the list is corrected.
  `UPSTREAM-NOTES.md` carries the corrected symbol list. Box A's reading (`sc1a-5090-1`) runs at `0a2a0c8`, the commit its proof
  `sc1a-prove-8` ran on; it installs vLLM only, so none of these drivers is on its path. Boxes B and C run at A3's merge, after
  their own proofs there. The e4b package (`experts4bit_qlora/`) is byte-identical between the two commits. The reducer's
  `--cross-box` reads each box's receipts at its own commit.
- **A3 erratum (2026-10-02T07:05Z).** A3's sentence "The e4b package (`experts4bit_qlora/`) is byte-identical between the two commits"
  is **wrong**. I wrote it against the main of the moment, and #878 (`82cd87d`) merged before A3 did. #878 flips
  `E4B_NF4_GROUPED_SMALLM`'s default from `0` to `auto` in `engines/hot_residency.py`, so the package differs between box A's
  commit (`0a2a0c8`) and A3's merge (`477670f`). The landing script's own identity check caught it after the merge. The run's
  arithmetic is unaffected, for two reasons:
  - Every e4b MoE invocation in the box script names `E4B_NF4_GROUPED_SMALLM=0` explicitly (`ROUTEENV`): the timed windows,
    K8, the pack build, the scheduler arms and the smokes. `tests/test_sc1_run_shape.py` pins the four route knobs on every e4b
    arm, so each arm takes the same route at both commits.
  - The one `step_decomp.py` call without `ROUTEENV` is box B's bf16 oracle (`--ppl-oracle upstream`). Its quantity is the
    transformers bf16 forward, and its e4b side runs at T == 1, where `auto` does not route; K25 takes only rows above T == 1.
  The cross-box reading states both commits. The receipts record each box's e4b sha.
- **A4 (2026-10-02T07:14Z; the box-A reading `sc1a-5090-1`, running at `0a2a0c8`).** Box A's Phase-0 smoke `smoke_qwen3_fused_b1` failed. It
  was the first run of the int4 levers through `serve_paged` on Qwen3. The P42 harness hook, which the box's e4b environment
  loads via `PYTHONPATH=$W/hook`, applies the int4 levers right after the tier is built. `serve_paged.build_engine` then applies
  the same levers at the same point; its `_apply_levers` is the hook's `_apply_lanes`. The second application refused:
  "E4B_SERVE_ATTN_INT4=1 matched no attention projections -- refusing a vacuous enable". Every scheduler-in-the-loop arm with int4
  levers on fails the same way, on every box. The Granite proofs passed because their levers are off. Fix: every
  `sc1_e4b_sched.py` invocation clears `PYTHONPATH`, so the server applies each lever once, as it does for a real user. The
  harness's `step_decomp.py` arms keep the hook. `sc1_e4b_sched.py` refuses up front, naming the cause, if the hook is loaded
  anyway. One shape test and three self-test checks pin it, and the shape test fails on the registered script. Consequence for
  `sc1a-5090-1`: its scheduler arms (`lic_sched`, sched SAMEPROMPT, sched TTFT and energy) are HARNESS_ERROR rows by this defect.
  Its other arms are a valid first draw, recorded as such. Box A is re-proved and re-run at A4's merge, with boxes B and C, so the
  registered scheduler-slope ratio is measured on one box with its vLLM anchor. Stopping the running box was not possible from
  this session.
- **A5 (2026-10-02T12:31Z; the box-A first draw `sc1a-5090-1`, adertha-receipts `636c321`).** The first draw ran every phase and exposed
  three instrument defects. All three are fixed here, each with a CPU test that fails on the registered code.
  1. **Every vLLM arm at B=1 died at engine initialisation** (matched, fp8-KV, nodetok, the NLL scorer, TTFT) with an illegal
     memory access, after vLLM 0.30's own warning "max_num_batched_tokens (8192) exceeds max_num_seqs * max_model_len
     (2048)". Its kernel warm-up runs a step of that many tokens. B=16 was unaffected, because 16 × 2048 ≥ 8192. The budget is
     now capped at `max_num_seqs × max_model_len`: 2048 at B=1, 2576 for the scorer's 2561-token window, 4104 for TTFT-4096.
     That keeps every prompt in one chunk, as registered; B=16 stays 8192. Each driver checks the budget before starting the
     engine.
  2. **Every energy row read "no power.draw column".** The sampler writes `--format=csv,noheader` and records its field list in
     the receipt (`sampler_fields`); the reducer looked for a header row. The reducer now reads the recorded field list. The
     box's timestamps are UTC and match the receipts' epochs.
  3. **vLLM's SAMEPROMPT arms read VOID "prompts_sha differs".** The reducer compared the distinct file's sha the arm recorded
     with `prompts_b16_same.json`. The arm's effective rows (`effective_prompts_sha256`) are what that file holds, and the
     e4b scheduler arm records the same field. The check now uses the effective rows, and still refuses a wrong file.
  Also recorded from the first draw, unchanged by A5 because they are registered outcomes: the licence reads
  **QUALITY_FAIL** at both arithmetics (wikitext −0.014 / −0.011 ppl, c4val1 +0.056 / +0.040 ppl vs the NF4 base, under the
  registered single-window calibrated gate). P6 is REFUTED (e4b served − prefill +0.0048 / +0.0025 nats, under the 0.005
  predicted). P12 HOLDS (lic / rtn within 0.2 % at both B). P14 is REFUTED (vLLM nodetok / matched 0.968 at B=16, 3.2 %
  against 1 %). The degraded control PASSES (×1.181, K19 absent). The re-run of all three boxes reads these again at the A5
  merge.
- **A6 (2026-10-02T13:39Z; receipt `sc1b-prove-8`, adertha-receipts `c9ba997`).** Box B's proof at e4b `0d66170` refused at its e4b
  tripwire: "E4B_NF4_GROUPED_SMALLM is not read with the registered default '0'". Release 0.40.0 (#878) had moved that default to
  `auto`. The check worked as designed, but it guarded the wrong property. Every e4b arm pins all four route knobs explicitly
  through `ROUTEENV`, and `E4B_NF4_GROUPED_SMALLM=0` is registered there (the A3 erratum explains why the arms' arithmetic is
  unaffected). What the arms need is that the library reads each knob from the environment. The tripwire now asserts that read
  and logs each observed default as a `ROUTE_DEFAULT` line, so drift is recorded instead of refused. A shape test runs the
  tripwire's own loop on stand-in sources: main's, `0a2a0c8`'s, and one where a knob is no longer read. It fails on the
  registered loop.
- **A7 (2026-10-02T16:09Z; receipt `sc1c-prove-12`, adertha-receipts `6ac47a8`).** Box C's proof at e4b `32d424e` installed and tripwired
  vLLM, ExLlamaV3 (cu132; the A3 fixes held) and SGLang 0.5.20. It is NOT PROVED for two box-C causes.
  1. Both Granite smokes died in Triton's driver build: gcc reported "fatal error: Python.h: No such file or directory". The
     CUDA-13 image's apt step installed `python3` without `python3-dev`; the PyTorch images of boxes A and B ship the headers.
     `python3-dev` is now in the apt line, and a shape test that fails on the registered line pins it.
  2. The SGLang JIT + `/health` proof could not load its checkpoint: SGLang's qwen2_moe loader raised
     `KeyError: 'model.layers.0.mlp.experts.w2_bias'`. That checkpoint (Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4) carries 4320 expert
     `.bias` tensors (read from its index). The lane's `Qwen/Qwen3-30B-A3B-GPTQ-Int4@9b534e43…` carries none (read from its
     safetensors header: `qweight / qzeros / scales / g_idx` only), so the real run is unaffected. The proof now loads that
     checkpoint, the one box C serves, at 17 GB, inside the server-start budget (`SGLANG_START_TIMEOUT` 1800 s) and the proof's
     1.0 h guard. "The proof fetches no Qwen3" now reads: no bf16 Qwen3.
  Box A's re-proof `sc1a-prove-11` launched at `32d424e` before A7. A7 does not touch box A's path (its PyTorch image never runs
  the apt step), so box A runs at `32d424e`, its proof's commit, and boxes B and C run at A7's merge.
- **A8 (2026-10-02T16:55Z; receipt `sc1c-prove-13`, adertha-receipts `aff68b9`).** Box C's proof at A7's merge (`3db414e`) passed both Granite
  smokes: A7's `python3-dev` held, and the B=16 smoke captured buckets 1–16. It is NOT PROVED on its SGLang JIT + `/health` check
  (rc 45 after 339 s). SGLang loaded the lane's GPTQ checkpoint (15.71 GB, upgraded to gptq_marlin) and then refused to size its
  KV pool: "Loaded weights leave no GPU memory for the KV cache under --mem-fraction-static=0.226. Raise --mem-fraction-static
  above 0.514".
  - **Cause.** `server.sh` passed no `--mem-fraction-static`, so SGLang 0.5.20 resolved its own. Its rule
    (`arg_groups/memory_hook.py`, `handle_gpu_memory_settings`) reserves activation memory at 1.5 MB per token of the largest
    prefill it expects. With `--chunked-prefill-size -1` that token count is `max_prefill_tokens`, 16384 by default, so it
    reserved 25.2 GB of the 32,607 MiB card. That left a static pool of 0.226, and the weights alone need 0.514. Every SGLang
    mode that turns chunking off resolves the same way: matched, kvfp8, ttft_matched and quality. Every timed SGLang arm except
    `native`, and both SGLang quality scorings, would have died at start-up. The proof runs the registered matched mode and
    caught it.
  - **Fix.** Those four modes pin `--mem-fraction-static 0.75`. The fraction sizes only SGLang's static pool (weights plus KV).
    No kernel and no scheduling flag changes.
    - Capacity: at 0.75 the KV pool is about 7.3 GiB, about 79,800 bf16 tokens (twice that in fp8). That holds the registered
      capacity rule, 16 × 2048 = 32,768 tokens, and the TTFT server's 16 × 4608 = 73,728. SGLang's own rule applied to the
      lane's real largest extend (16 rows × 512 = 8192 tokens) would give 0.603, about 30,500 tokens, short of the rule.
    - Headroom: 7.7 GiB of the card stays outside the static pool. vLLM 0.30's profile of the same checkpoint on the same card
      at the same 8192-token batch measured 0.45 GiB of peak activation with bf16 KV and 0.86 GiB with fp8 KV, plus at most
      0.50 GiB of CUDA graphs (`sc1a-5090-1`). The headroom is more than five times that.
    - `native` keeps SGLang's own resolution. With chunked prefill on, SGLang's rule gives about 0.79 on this card.
  - `server.sh`'s engagement check now refuses a server whose resolved fraction is not the pin, or whose `max_total_num_tokens`
    is below the registered capacity: 16 × 2048 on matched and kvfp8, one full context on ttft_matched and quality.
  - Four CPU tests. One reproduces SGLang's 0.226 from its rule; the other three fail on the registered `server.sh`.
  - **Boxes.** Box C re-proves at A8's merge and runs there. A8 touches only box C's SGLang wrapper, so boxes A and B keep the
    commits their proofs ran on (`32d424e`, `3db414e`).
- **A9 (2026-10-02T17:41Z; receipt `sc1c-prove-14`, adertha-receipts `4cd4854`).** Box C's re-proof at A8's merge (`cb9e18f`) passed both Granite
  smokes, and A8 held: SGLang sized a KV pool of 79,560 tokens against the 79,800 predicted. It is NOT PROVED on the next step
  (rc 45 after 819 s). While capturing the decode CUDA graphs, SGLang's Marlin MoE asserted "moe_wna16_marlin_gemm assumes
  hidden_states.dtype (torch.bfloat16) == w1_scale.dtype (torch.float16)".
  - **Cause.** `server.sh` and `one_batch.sh` ran SGLang with `--dtype bfloat16`. The lane's GPTQ checkpoint declares `torch_dtype`
    float16 and stores float16 scales. SGLang 0.5.20's GPTQ Marlin MoE requires the activations in the scales' dtype
    (`layers/moe/fused_moe_triton/fused_marlin_moe.py`), so every SGLang mode and the one-batch benchmark would have failed at
    their first MoE call.
  - **It was not matched work either.** vLLM's arms pass no dtype, and `auto` resolved float16 on this checkpoint (`sc1a-5090-1`:
    `dtype=torch.float16`). LMDeploy's registered kernel family is f16 (`sm80_f16_u4k128_f16`). bf16 on SGLang alone would have
    put a different activation dtype on the pair P3 ratios (SGLang against box C's vLLM anchor).
  - **Fix.** Every pinned SGLang server mode and the one-batch benchmark pass `--dtype float16`. `native` passes none, and
    SGLang's `auto` follows the checkpoint's float16. The engagement check refuses a pinned-mode server that resolves any other
    dtype. One CPU test fails on the registered wrappers. Capacity is unchanged, because an fp16 KV token is the same size as a
    bf16 one.
  - **Boxes.** Box C re-proves at A9's merge and runs there. A9 touches only box C's SGLang wrappers, so boxes A and B keep the
    commits their proofs ran on (`32d424e`, `3db414e`).
- **A10 (2026-10-02T21:20Z; receipts `sc1b-5090-1`, adertha-receipts `cabef5a`, and `sc1c-5090-1`, adertha-receipts `711a2d9`).** Box B's first full run at
  A7's merge (`3db414e`) ran 1 h 54 min of its 5 h guard and ended rc 134. Its e4b window, vLLM, oracle and ExLlamaV3 B=1
  rows read VALID. Every llama.cpp server, every LMDeploy arm and every ExLlamaV3 quality scoring failed. The e4b scheduler
  arms read VOID on a reducer rule that measured the wrong quantity. Box B's proof (`sc1b-prove-10`) had installed and
  tripwired every comparator, but never started a server or generated a token, so none of this was reachable from it.
  1. **llama.cpp: every server refused.** `llamacpp_server_start` greps the log for `offloaded N/N layers to GPU` and
     `flash_attn = enabled`. At the pin (552f18f) library INFO messages map to verbosity 4 (`common/log.cpp`
     `common_log_get_verbosity`) and the default threshold is 3, so neither line was logged. The server log has 14 lines.
     llama-server now runs with `-lv 4` (5 is debug output). The registered checks are unchanged.
  2. **LMDeploy: UNSUPPORTED on sm_120, P5b's stated alternative.** All 11 LMDeploy runs aborted with rc 134 on their first
     real request, 0.55–0.62 s in, whatever the prompt length (512 to 4096 tokens). The reported sites were `invokeMinLengthPenalty` (speed, TTFT
     and energy) and an `output_processor.cc:333` copy (quality). Both are only where an earlier fault (CUDA error 700)
     surfaced.
     - The cause is in LMDeploy 0.18.0 (`110965c7`), read at the tag. TurboMind's host dispatch runs the SM80 GEMM kernels
       on an sm_120 device (`src/turbomind/kernels/gemm/arch.h:53`).
     - Each kernel body is compiled only where `Kernel::Arch::is_compatible(__CUDA_ARCH__)` holds
       (`gemm_universal.h:174-178`), and `Sm80` is `Arch<800, 900>` (`arch.h:25`). At 1200 every 4-bit GEMM is an empty
       kernel.
     - Warm-up cannot see this: it runs layer 0 only, skips attention, uses synthetic routing and never checks results.
     - vLLM ran the same checkpoint on the same card.
     - No generation or engine setting can repair a no-op GEMM. Box B records UNSUPPORTED with this reason
       (`LMD_UNSUPPORTED`) instead of installing LMDeploy, and every LMDeploy row is an UNSUPPORTED stub carrying it.
     - The reducer already reads P5b as HOLD on its stated alternative.
     - Not tried, and not registered: LMDeploy's PyTorch backend, or a source build that widens `Sm80`.
  3. **ExLlamaV3 quality scorer.** Every `nll_exl3_*` run died with `AttributeError: module 'exllamav3' has no attribute
     'version'`, after loading and scoring. A3 fixed this read only in the install tripwire. The speed arm read it through
     `hasattr` and recorded `versions.exllamav3 = None`. Both drivers now import `exllamav3.version`. The scorer reads it
     before the load, so a failure there costs nothing.
  4. **Reducer: every engaged scheduler arm VOIDed.** The rule `census int4_attn_projections == 192` compared a
     post-fusion count with a pre-fusion one. `serve_paged.build_engine` counts `Int4Linear` modules after `_apply_fusions`
     merges q/k/v. A fully converted fused model therefore reads 192 - 2 × 48 = 96. Box B: 96, with
     `attn_int4_rtn_projections` 192 and the log line "ATTNINT4 rtn: 192 projections". The rule now checks two counts, each
     in its own units:
     - the arm's lever (calibrated for `lic*`, rtn for `int4*`) converted 192 projections with the other lever at 0;
     - the post-fusion count is 192 - 2 × `fuse_qkv_n`.
     The self-test fixture had carried 192 for the post-fusion count, which is why the self-test passed on a shape the
     engine never writes. It now carries the real shape, and four VOID cases are added. Box B's scheduler draws re-read
     VALID or VOID on their other checks once re-reduced. No re-run is needed for this item.
  5. **Box C's stall (`sc1c-5090-1`), hardening that does not wait for the cause.** After `sglang_gptq_matched_b1_r1`
     (19:02Z) the summary never changed. The heartbeat read GPU idle with 25.6 GB resident and no disk movement for
     6731 s, and from 20:55Z the box stopped answering. It never answered again. At the guard's deadline (22:21Z) the
     fetch got nothing (`ssh: connect … Connection refused`, driver exit 22). The receipt is HARNESS_ERROR, $2.15,
     teardown proven, and its `sc1/` directory is empty. So the stall's cause cannot be read from the box (item 8).
     - The next step was the matched server's stop. `sglang_server_stop` ended in an unbounded `wait "$pid"`.
     - A process stuck in the GPU driver sits in uninterruptible sleep, ignores SIGKILL, and makes that `wait` block forever.
       llama.cpp's stop had the same `wait`.
     - Both stops now poll for up to 60 s after SIGKILL, reap only an exited process, and report a stuck pid.
     - Every SGLang transition writes a summary line before it starts, so the heartbeat can locate a stall.
     - A stuck stop ends SGLang on the box: later starts are refused rather than piled onto memory the stuck process holds.
     - The stall's actual cause is read from box C's receipt when it lands, and amended if it is something else.
  6. **Box B's proof starts the server it could not start.** After the Granite smokes it fetches the lane's Q4_K_M and
     starts llama-server at np=1 through the registered checks. A skipped (`can_run`) or failed fetch or start is NOT
     PROVED. That adds 18.6 GB, about 4 min on box B's link, and the guard stays 1.0 h. Box B's proof no longer needs
     LMDeploy (item 2). The ExLlamaV3 scorer is not in the proof: its GPU path ran on box B, which loaded and scored before
     dying on the version read, and item 3's fix is covered on CPU.
  7. **The scheduler arms' decode reading cancels each rep's own prefill.** Re-reduced with item 4's rule, box B's
     scheduler draws are engaged and read UNSTABLE: B=1 225.8 / 178.7 / 214.9 tok/s (26.4 %), B=16 1676.9 / 1612.7 /
     1580.6 (6.1 %). The window arms beside them were steady to 0.1 %.
     - The registered slope (32 → 128 tokens) differences two walls that each carry e4b's paged prefill, about 2.1 s per
       512-token chunk. Box B's per-draw TTFT medians moved 2.05–2.18 s, with p99 up to 2.39 s.
     - So prefill jitter of a few percent became 10–25 % of the 0.45 s difference the decode rate is read from. vLLM's
       prefill is 0.04 s, so its slope is not exposed.
     - The scheduler driver now records every request's `Request.ttft`, measured from arrival; rows are added together at
       the start of the rep. Each rep's decode-only time is its wall minus the latest row's TTFT, and the slope is taken over
       those. The quantity is the same, decode cost per step at the registered batch, and the estimator cancels the prefill
       per rep.
     - The registered wall slope is recorded beside it (`*_wall_slope`). A rep without every row's TTFT falls back to the
       wall slope, labelled.
     - Box B's scheduler draws cannot be re-read this way, because their receipts carry only TTFT p50/p99. Box A's running
       draw (`32d424e`) uses the registered estimator, and its reading is labelled as such.
  8. **The driver pulls the box's results during the run.** `sc1_drive.sh` fetched the box's `sc1/` directory once, after
     TP_DONE or at the deadline. Box C's box was unreachable by then, and the whole draw was lost, including the logs that
     would name item 5's cause. TC2's box B was lost the same way.
     - The driver now pulls every `SC1_PULL_EVERY_S` (1200 s), while the heartbeat answers, into `sc1.partial/`. It uses the
       final fetch's keep/leave rules. The remote rsync runs at nice 19 with idle I/O, so a pull during a timed arm does
       not perturb it. Both pulls are bounded by an ssh ConnectTimeout and `rsync --timeout`.
     - If the final fetch fails, the last pull becomes the receipt's `sc1/`, labelled `PARTIAL_PULL_AT`, and the driver
       still exits 22.
     - The controller runs `/bin/bash` 3.2 under `set -u`, so the optional rsync argument uses the 3.2-safe expansion. The
       test runs the driver's own `pull_box` under the system bash.
  CPU tests: `tests/test_sc1_a10.py` (13, with the scheduler driver's own self-test now 25 cases; every test of a fix fails on the registered drivers, and the 48/49-offload control passes on both), four new reducer self-test cases on the corrected fixture (which VOIDs under the registered rule), and the amended proof-needs test. Box B re-proves at A10's merge and re-runs there (`sc1b-5090-2`). Box A (`sc1a-5090-2`, at
  `32d424e`, launched before A10) runs none of items 1–3, 5, 6 or 8 (its controller started on the registered driver); item 4 applies when its receipts are reduced; its scheduler reading is the registered wall slope, labelled (item 7). Box C re-proves at A10's merge and re-runs there (`sc1c-5090-2`): its first draw returned nothing.
- **A11 (2026-10-03T03:50Z; read-time, the reducer only; receipts `sc1b-5090-3` (adertha-receipts `8b4e2c9`) and `sc1a-5090-2` (`db915fc`)).**
  Box B's run at A10's merge started every llama.cpp server through the registered checks. Its scheduler arms read STABLE
  with A10's decode-only estimator. Reducing it alongside box A exposed four places where the reducer read differently from
  the registration. Each change has a self-test case that fails when the change is reverted. No box re-runs.
  1. **llama.cpp's engagement lines are read from the server log.** The reducer grepped the arm's log for `offloaded N/N
     layers to GPU`, `flash_attn = enabled` and `n_slots`. Those lines are in `logs/llamacpp_server_*.log`, whose path each
     receipt records as `server_log`. Box B's server log reads 49/49, enabled, `n_slots = 1` / `16`. The self-test fixture
     had put the lines in the arm log, so every engaged llama.cpp row VOIDed. The reducer now reads the named server log
     beside the arm log.
  2. **Box A's licence blocks boxes B and C too.** The registration: "QUALITY_FAIL → … no position is quoted from the RTN
     rows either (P58's headline is not restored by a side door)". Boxes B and C are ratioed to RTN anchors, and the reducer
     had applied the block on box A only. The cross-box read now carries box A's licence onto every position on B and C.
     Box A's licence read QUALITY_FAIL on both draws (`sc1a-5090-1`, `sc1a-5090-2`). So SC1 quotes no position on any box.
     Every ratio is reported as a measurement, labelled.
  3. **Quality rows measured on one box are carried to the others.** e4b's served/prefill rows run on box A only
     (`k8_lic_auto_*`, `nll_e4b_prefill_*`). The bf16 oracle runs on box B only; the registration says box A's "whole
     delta_bf16 axis depends on it". Yet every box's rows are read against them. The cross-box read now copies a row into a
     box that lacks it, from the box that measured it VALID, only on an identical window (same `text_sha`), labelled with its
     source. It then recomputes delta_pair, delta_bf16 and the bands. Box A's e4b rows are its licensed pack: the e4b stand-in
     that B's and C's RTN anchors substitute for under the substitution licence (P12). The self-test fixture had given boxes
     B and C e4b quality rows the box script never produces. It now has a case without them.
  4. **P13 reads the measured anchor ratios.** The registration: "The three boxes' vLLM/e4b anchor ratios are reported side
     by side (expected within 5 %; a larger gap is a host finding)". That is a ratio of two engines' measured speeds on one
     box, wherever both arms are VALID and stable. The reducer had read only *quoted* positions, a choice of the reducer and
     not of the registration. That gated a host-agreement check on quality comparability and the licence, and under
     QUALITY_FAIL made it unreadable by construction. P13 now reads the measured ratios, with the quotation flag beside.
     *This reading of P13 is a judgment under the owner's delegation; it is flagged for review.*

- **A12 (2026-10-03; the scorer only; receipt `sc1c-5090-5`, adertha-receipts `94a8f5e`). No box re-runs.**
  Box C's run at A10's merge (`9dd712b`) read every speed arm VALID: SGLang matched and native, vLLM, both e4b anchors and
  ExLlamaV3 cu13. Its exit code 1 came from two rows only: `nll_sglang_prefill_wikitext` and `nll_sglang_prefill_c4val1`
  each died on `assert lps[0] is None` in `score_prefill`. Both served-shape scorings on the same server read VALID, CLOSE
  on both texts.
  - The scorer misread the response's composition. SGLang 0.5.20 (tag `94602c9c`) prepends `None` to the logprob
    **values** (`scheduler_components/logprob_result_processor.py:38`) and takes the ids from
    `origin_input_ids[logprob_start_len:]` (`:50`). `tokenizer_manager.py:2888-2892` then zips the two lists into
    `(logprob, token_id, text|None)` triples. Entry 0 is therefore `(None, ids[0], None)`, a triple whose logprob is None,
    not a bare None.
  - The CPU fake had the same misreading (`[None] + [...]`), so the registered tests passed against a shape the server never
    sends.
  - The fix: the scorer reads entry 0 through `_lp`, requires its logprob to be None, and requires its token, when present,
    to be `prompt[0]`. A bare None is still accepted. The per-position checks on the scored indices are unchanged, and they
    were never reached on the box. The fake now emits v0.5.20's triple.
  - CPU tests in `tests/test_sc1_sglang.py`: one reads the triple; it fails on the registered scorer, as do the two
    existing prefill tests on the corrected fake. One reads a bare None. Two refuse an entry 0 that carries a logprob, or
    that is for another token. `staged.sha256` is re-pinned.
  - Why there is no re-run: the prefill-shaped SGLang rows are evidence for no registered prediction. P6's served-vs-prefill
    pair is e4b's and vLLM's. Every SGLang position reads the served shape, which is VALID. Under box A's QUALITY_FAIL
    licence (A11 item 2), no SC1 position is quoted in any case. SGLang's served-minus-prefill engagement reading is
    reported as UNREAD, with this cause. Any later box-C draw runs the fixed scorer.
- **A13 (2026-10-03; box A's third draw; the owner approved the redraw in chat after the read, e4b#934).** The read
  (`bench/h2h-2026-10-02/sc1/`) left five box-A predictions unread: P7, P8, P11, P14, and P13's third box.
  - Box A's second draw (`sc1a-5090-2`, at `32d424e`) ran out of guard. From its phase stamps:
    - Phase 0 took 1 h 30 min, and Phases A–EN took 3 h 37 min.
    - The 5.5 h deadline skipped vLLM's energy windows and all of Phase G2: the second fp8-KV, rtn, nf4_ctrl and control
      draws, the third scheduler draws, the nodetok pair, and the fp8-KV quality rows.
  - Its scheduler anchor read UNSTABLE (220.5 / 310.2 tok/s at B=1) on the registered wall-slope estimator that A10 item
    7 replaced.
  - Box A's third draw, `sc1a-5090-3`, runs at A13's merge. It is proved first by `sc1a-prove-12` at the same commit.
  1. **The guard is 7.5 h for this draw (≤ $5.63 at $0.75/h; the registered 5.5 h ≤ $4.13 stands for the record).**
     G2 plus vLLM's energy windows took no time on `sc1a-5090-2`, because they never ran. From the first draws' phase
     lengths they need about 70 min: Phase C's six arms took 11.5 min, Phase F's three took 17.5 min, and Phase D-vLLM's
     scorings took 23 min. The draw then ends near 6 h 40 min. The extra 2 h leaves about 50 min of slack for a slower
     host. No arm, phase or order changes.
  2. **The int4 prefill route is pinned to `loop`.** #937 (after every box had run) made `E4B_INT4_PREFILL=auto` the
     default: K19 wherever it can run.
     - On an unpinned redraw that changes every int4 arm with a T > 1 call: the scheduler's prefill, TTFT, the energy
       windows' prefill, and the K8 licence's 512-token prompt. P102 read that last one moving by up to 0.011 ppl.
     - The box script scrubs the knob and then exports `E4B_INT4_PREFILL=loop` once, before the tripwire and every arm,
       so every e4b process inherits it.
     - It does not go in `ROUTEENV`: P100's and P102's staged-pin tests assert that their `FOLDS`, `SPEEDENV` and
       `ROUTEENV` equal SC1's byte for byte, and A13 leaves all three unchanged.
     - The e4b tripwire asserts that the knob is read from the environment, that the box's environment holds `loop`,
       and that it resolves to `loop`.
     - Decode is untouched either way: B=1 decode is T == 1, and B=16 decode rows are at most 256, which take the
       device-grouped route under every value.
  3. **Recorded, not changed: #918 (#913) is in this draw and was not in the others.**
     - Boxes A (`32d424e`), B and C (`9dd712b`) ran the scheduler arms' bucketed decode graphs over a single-entry
       row-to-token cache. A later bucket's warm-up could free an earlier bucket's index, and the earlier bucket's replay
       then gathered token rows through whatever was in the freed block.
     - SC1's scheduler rows are speed only. The fix keeps one index per row count, so the gathered rows change and the
       kernels, shapes and launches do not. The speed rows are therefore read as comparable across the fix.
     - The same-box method pair (window / sched) is the check: B=1 +6.0 % on B and +10.4 % on C before the fix.
  - **Reads:**
    - P13 on all three boxes, if box A's scheduler anchor is stable on A10's estimator.
    - P7, P8, P11 and P14, if G2 and the energy windows complete.
    - Box A's licence a third time.
  - **Does not read:** P1–P5. They need a quoted position, and the licence has read QUALITY_FAIL on both draws. A third
    QUALITY_FAIL changes nothing; a PASS would be reported beside the two FAILs, not instead of them.
  - CPU tests: `tests/test_sc1_a13.py` (6). Three fail on the registered script: the export, its place between the
    scrub and the tripwire, and the tripwire's checks. Three controls pass on both: no arm sets another route,
    `ROUTEENV` and `FOLDS` are untouched, and `auto` would take K19 where `loop` holds. `staged.sha256` is re-pinned.
