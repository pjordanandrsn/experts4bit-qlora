# SC2g: request-level serving of gpt-oss-20b. e4b's `serve_paged` (its first gpt-oss run through the server) against vLLM, SGLang and llama.cpp under Poisson arrivals, one RTX 5090 (lane SC2g of #846; registered 2026-10-05)

Registered 2026-10-05, before any SC2g run. Reviewed before registration by the maintainer session:
- three design questions first: the e4b env, the engaged count, and `auto` with sinks;
- then the full draft: "approve with small changes", all applied. They covered gnf4 v0.39.0, arithmetic labels cited
  from the kernel source, memory as a plausible stand-down, `E4B_PAGED_MAX_TOKENS_PER_SEQ` stated, and Q2's basis.

## Why this lane

SC2 (Qwen3-30B-A3B) read e4b's request-level capacity at 1 req/s against vLLM's and SGLang's 8. It read it post hoc as
**prefill-bound**: each interleaved prefill stalls every running decode for ~0.36 s (R² 0.985). SC2b's prefill graph
cut serial TTFT 1.30–1.65× but left the ceiling at 1 req/s. Most of the stall under load lies outside the graphed
forward.

SC2g asks whether that holds on a **second family**, and registers SC2's post-hoc mechanism as a prediction (Q4). It is
also **e4b's first gpt-oss run through `serve_paged`**: the registered gpt-oss numbers in `docs/SERVING-THROUGHPUT.md` are
bare graph windows.

## Engines (box G: one box, CUDA 13 image, the same arithmetic caveat on every row)

**The image** is SC1 box C's and SC2 box E's `nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04`, the one image that holds all four
engines (SGLang 0.5.20 needs CUDA 13).

openai/gpt-oss-20b @ `6cee5e81` ships MXFP4 experts. Every engine serves those experts, but **on different
arithmetic**. Identical bytes license the weights, not the arithmetic (`finding_gptoss_mxfp4_serving_arithmetic_differs_per_engine`).
Every row carries its arithmetic label, and every e4b-vs-comparator ratio is labelled **ARITH_MISMATCH**: reported,
never voided for it.

| engine | weights | arithmetic (the row's label) | settings SC2g pins |
|---|---|---|---|
| **e4b_gptoss** | the checkpoint's own MXFP4 blocks + e8m0 scales (`E4B_SERVE_EXP_INT4=1`, the `gptoss` branch, no re-quantisation) for decode; an NF4 arena (`k8_bake.py`, the bake bo3 used) for prefill | **decode:** T == 1 `gemv_mxfp4_b32` on **int8 activations** (per-32 fp32 scales, exact int32 e2m1 dot: W4A8); ≤ 256 rows K21 `gemm_mxfp4_grouped_smallm` on **bf16** activations, fp32 accumulation (W4A16). Both are cited from grouped-nf4-gemm v0.39.0 and v0.41.0 (file unchanged) `kernel/mxfp4_grouped.py:257` / `:370`. **Prefill NF4** (`E4B_INT4_KEEP_NF4=1`: no MXFP4 M-tile exists for > 256 rows). Attention sinks keep the explicit-mask prefill path | bo3's env (`E4B_SERVE_ATTN_INT4_CALIB=0 E4B_CALIB_SOURCE=c4`) + SC1's folds + `ROUTEENV`; attention int4 off; `max_seqs` 16; **`E4B_PAGED_MAX_TOKENS_PER_SEQ=2048`**, matching vLLM's `--max-model-len` and SGLang's context; the prefill graph at main's default (`auto`), recorded |
| **vllm** | vLLM 0.30.0, the same checkpoint | Marlin **W4A16** (`--moe-backend marlin`, zero-padded); **TRITON_ATTN** pinned (sinks-capable, needs no NVIDIA cubins) | `--max-num-seqs 16 --max-model-len 2048 --no-enable-prefix-caching --seed 0 --gpu-memory-utilization 0.90` |
| **sglang** | SGLang 0.5.20, the same checkpoint | its **default** MXFP4 runner on sm_120 (recorded from `server_info`), triton attention (forced for gpt-oss) | `gptoss` mode: `--disable-radix-cache --max-running-requests 16 --context-length 2048 --mem-fraction-static 0.75` |
| **llamacpp** | llama.cpp `552f18f`, ggml-org's published `gpt-oss-20b-MXFP4.gguf` @ `ef9b12f2` (experts MXFP4; **attention Q8_0**) | default MMQ: **W4A8** decode (≤ 8 tokens), **W4A4** prefill | SC1's flags, `-np 16`, 1,024 tokens per slot; `cache_prompt: false` |

**e4b's row is "native MXFP4 decode, NF4 prefill", not same-bytes end to end** (the maintainer's framing, from main's
code). SC2's finding puts capacity in the prefill, so this row reads e4b's NF4 prefill path on gpt-oss.

**The stack.**
- e4b at this registration's merge commit, at main's defaults: prefill graph `auto`, KV step-select on.
- grouped-nf4-gemm **v0.41.0** (`dc8f94ab`, tag object `e90a3523`): what e4b 0.48.0's CI runs. From v0.39.0 to v0.41.0
  it changed `_triton_shim`, `nf4_grouped`, `nf4_route` and `nvme_residency`; `mxfp4_grouped.py` and `int4_smallm.py` are
  unchanged, so the arithmetic citations below hold. Box F keeps v0.38.0.
- **`GNF4_TRITON_PREBIND=1`, pinned** (v0.41.0's default, `_triton_shim.py:269`). It is bit-identical (the same
  compiled kernels, only the launch differs) and wraps `_gemm_nf4_grouped`, whose reach into e4b's NF4 prefill path is
  **not verified**. That makes it a launch-overhead lever on a launch-bound path, recorded rather than left inherited.
- Box G exports neither of SC1's prefill route pins (SC2's correction, #1061), and the e4b server starts under `env -u`
  on both.

## The instrument

- **SC2's driver, unchanged.** Raw `/v1/completions` with token ids (no chat template, no Harmony); `max_tokens` drawn
  from U[64, 256]; greedy; `ignore_eos`; streaming; 16 in flight; prefix caching off. TTFT, TPOT and SLO attainment
  (TTFT ≤ 1.0 s and TPOT ≤ 100 ms) as SC2 defines them.
- **The prompt pool** is SC2's tool with **gpt-oss's own tokenizer**: 64 rows of 512 wikitext-2 tokens.
- **Per engine:** a server start with its engagement recorded; 4 warm requests; two draws of Q1 serial (24) and Q2
  Poisson at 1, 2, 4 and 8 req/s (120 each). The engine order is e4b_gptoss, vLLM, SGLang, llama.cpp.
- **Both draws repeat ONE realisation:** the same serial seed, and the same seed per rate. SC2's read traced its UNSTABLE
  knee rows to per-draw seeds realising different arrival rates. Here a draw disagreement is the engine's, not the
  dice's.
- **e4b's engagement, from the server's own `/health`.** The arm STOPs (rc 47) unless:
  - `int4_store_kinds` is `["mxfp4"]` and `exp_int4_layers_enabled` is 24 (gpt-oss-20b's 24 MoE layers);
  - `prefill_routes` reads box F's assertion: k19 / k19 / flash, device grouping on, both route pins absent. Its int4
    entries describe the int4_b32 store's route, so for gpt-oss they record that nothing was pinned. The NF4-kept
    prefill is set by `E4B_INT4_KEEP_NF4=1`.

  Recorded, not gated: the prefill graph's `auto` decision (`status`, `why`, `pool_mib`, `free_after_mib`), the
  attention-int4 state, and VRAM at ready. Attention int4 is off, so the calibration env is inert, and
  `E4B_SERVE_ATTN_INT4_CALIB` is pinned to 0.

## The rule (`bench/sc2/sc2g_reduce.py`, self-tested on 5 cases)

- **Rows and ceilings** use SC2's rule unchanged (`sc2_reduce.row` / `ceiling`).
- **Q4** reads e4b's request trace with `sc2_trace.py`, which is SC2's post-hoc tool and is now part of the registered
  rule.

| # | prediction | basis |
|---|---|---|
| Q1 | serial: e4b_gptoss's p50 TTFT ≥ 2 × vLLM's (ARITH_MISMATCH) | SC2: 5.2× on Qwen3. e4b's gpt-oss prefill is NF4, launch-bound and on the explicit-mask attention path |
| Q2 | serial: e4b_gptoss's p50 TPOT ≤ 1.5 × vLLM's (ARITH_MISMATCH) | e4b's bare MXFP4-store windows at B=1: 151.1 tok/s (bo3, the GEMV route) and 173.3 tok/s (`docs/SERVING-THROUGHPUT.md`'s i6 row, store_r12, the GEMV on single rows, the closer configuration); none through `serve_paged`. vLLM's gpt-oss B=1 on this card is unmeasured, so this is a near coin flip |
| Q3 | capacity: vLLM's ceiling ≥ 4 req/s and > e4b_gptoss's | SC2: vLLM 8 against e4b 1 on a heavier model |
| Q4 | **the mechanism, registered:** e4b's trace fits `decode_s = a·(out_len−1) + b·(prefills landing during the decode)` with R² ≥ 0.9 and **b ≥ 10·a** | SC2: a 6.15 ms, b 0.356 s (b/a = 58), R² 0.985; SC2b's ON arm b/a ≈ 44 |
| Q5 | every row of every engine is VALID | SC2 failed this only on UNSTABLE knee rows, which the repeated realisation should remove; the bar is unchanged |

**The prefill graph's `auto` is not guessed.** Its startup check decides. The sinks path is pure device ops, and the NF4
fallback at > 256 rows is capture-safe, so the maintainer expects it to engage, but this file registers no outcome.

**Memory is a plausible reason for `auto` to stand down.** With `KEEP_NF4=1` the box holds BOTH the NF4 stacks and the
MXFP4 store for every expert, plus the KV pool (16 × 2048), the decode graphs, and then the prefill graph's pool, whose
sinks path materialises fp32 scores (about 64 MB per layer at 512 tokens). bo3's MXFP4 store fitted on a 5090, so
loading is fine. A memory stand-down would be the design working, not a defect, and the row says so.

| | if `auto` ENGAGES | if `auto` REFUSES (`status` "refused", `why` recorded, e.g. memory) |
|---|---|---|
| Q1 | compares e4b's graphed NF4 prefill with vLLM's | compares e4b's eager NF4 prefill |
| Q2, Q3, Q5 | read as written; the row is labelled "prefill graph on" | read as written; labelled "prefill graph refused: <why>" |
| Q4 | reads the stall with the graph, as in SC2b's ON arm (b/a ≈ 44 there) | reads the stall without it, as in SC2 (b/a ≈ 58) |

Either way, the row says which way it ran.

**No position sentence comes from SC2g.** It measures the four stacks as configured, with their arithmetic labelled.
There is no quality claim; gpt-oss quality on these arithmetics is lane SC1g's question, still drafted.

## Proof and budget

**Proof** (`sc2g-prove-*`, guard 1.25 h):
- SC1's common proof (installs, tripwires, the Granite fetch, bake and smokes);
- the driver's and both reducers' self-tests;
- **gpt-oss-20b itself**: fetch without `original/` and `metal/`, the NF4 bake (5,400 s alarm, `bake.json` status OK),
  the gpt-oss prompt pool;
- every server on gpt-oss: e4b (engagement as above), vLLM, SGLang (`gptoss` mode), and llama.cpp on the published
  GGUF. Each answers a 6-request serial smoke and a 16-request Poisson smoke at 4 req/s, with every request VALID.

**Reading** (`sc2g-5090-*`, guard 2.5 h): installs about 15 min, fetch and bake about 30 min, the SGLang JIT, four
engines at about 13 min each.

**Guards include download** at $0.0078/GB:
- proof: 1.25 h ≤ $0.94 + about $0.30 (about 35 GB);
- reading: 2.5 h ≤ $1.88 + about $0.30;
- lane: about $3.4, each run under the $15 no-ask tier.

The deadline drops arms from the end (llama.cpp first).

## Amendment A1 (2026-10-05): the proof died in the harness; three fixes, no change to the design

**`sc2g-prove-1`** (adertha-receipts `ae066e41`, machine 26157, **$0.848**, HARNESS_ERROR). The run reached box G's install
after e4b's and gnf4's installs and tripwires passed (e4b 0.48.0 @ `e8892972`, gnf4 0.41.0 @ `dc8f94ab`). It died at
06:21Z: `/root/sc1/sc2g_box_g.sh: line 20: FOLDS: unbound variable`. Nothing on e4b's gpt-oss path ran.

1. **The cause.** `sc1_run.sh` sources the box scripts at its install step, under `set -uo pipefail`, before it defines
   `FOLDS`. `SC2G_E4B_ENV` expanded `$FOLDS` at top level. The fix appends `$FOLDS` where the e4b server starts
   (`g_e4b_start`), so the child's environment is unchanged. A new test sources every SC2 box script under `set -u` with
   only `W` defined, which is the order `sc1_run.sh` uses. Against the old line it reproduces the failure exactly.
2. **The controller waited out the deadline.** The box died at 06:21Z, and the controller polled until its 07:31Z deadline.
   The heartbeat counts `pgrep -f 'bash sc1_run.sh'` inside an ssh shell whose own command line carries that pattern.
   Linux's procps `pgrep` excludes only itself, so the count included its own shell and never read 0. The controller's
   log shows 66 polls reading `live 2` after the death, and on a Linux host with no lane the expression reads 2. LANE
   DEAD could never fire, on any box. It now counts `[b]ash sc1_run.sh`, which reads 0 there; a test runs it.
3. **The box now records its own abnormal exit.** `sc1_run.sh` gains an EXIT trap. An exit that skips `finish` writes
   its rc and TP_DONE, so the controller ends on the box's record at once. Such an exit is never a success: rc 0 there
   is recorded as 79. Tests reproduce the `set -u` death and a bare `exit 0`.

Fixes 2 and 3 touch every box. They change only what happens after a box has already failed, and boxes A–F have read.
The rule, the predictions, the engines and the guards are unchanged. The next proof is `sc2g-prove-2`, under the same
guard. Spend so far: $0.848, which is inside the lane's ~$3.4.

## Out of scope

- Quality (SC1g).
- Long prompts and later prefill chunks.
- gpt-oss-120b (no 5090 path; `finding_e4b_no_5090_path_for_120b_moe_serving`).
- Speculative decoding.
- The EAGLE GGUFs.
