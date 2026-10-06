# SC1g: the quality of each engine's gpt-oss-20b arithmetic on identical tokens, against a routing-flip floor measured on the same windows, on one RTX 5090 (lane SC1g of #846; registered 2026-10-05)

Registered 2026-10-05, before any SC1g run. The maintainer session reviewed it first.
- **The design (v3)** was given a GO with four conditions, all written in below:
  1. the e4b pin (§ The route record);
  2. the route record's guards (§ The route record);
  3. the floor as a statistic fixed before data (§ The floor);
  4. the scope of COMPARABLE (§ What a result licenses).
- **Four design questions came before that**, and their answers shaped v3:
  - which routes `step_decomp` takes;
  - the prior quality numbers;
  - the bf16 reference does not fit a 5090;
  - which llama.cpp GGUF to use.

## Why this lane

SC2g (`bench/h2h-2026-10-02/sc2g/`) read request-level speed for four stacks serving gpt-oss-20b's MXFP4 experts. Each
stack uses different arithmetic. As registered, SC2g said nothing about quality, so no position sentence came from it.

SC1g reads quality **on identical tokens** for the same stacks and the same bytes.

**e4b's distance from bf16 is already measured, and SC1g cites it.** These are full-vocab KL against a bf16
dequantisation of the same bytes, on an H100 NVL:

| measurement | KL (nats) | top-1 |
|---|---|---|
| P44: the MXFP4 store | 0.00192 | 0.981 |
| P44: an NF4 re-quantisation | 0.0222 | |
| P90: K21 at B = 16 | 0.00147 | |

The bf16 reference is about 40 GB and does not fit a 5090; P90's first box was VOID on exactly that. SC1g adds no
oracle. Its question is cross-engine: on the same token ids, does any engine's arithmetic move NLL beyond what two
equally correct forwards already disagree by?

## The floor (registered before data)

Two arithmetically equivalent forwards of an MoE model disagree, because rounding flips router top-k choices (METHODOLOGY
§13.1; `e4b.parity.moe-routing-flip-floor`). The repo's gpt-oss figures depend on window length:
- |Δnll| 0.0099 at 448 tokens (METHODOLOGY's table);
- 0.0176 at 512 steps (the register's note).

SC1g therefore measures its own floor, on its own windows, by §13.1's method: two orders of the same mathematics.

- **The floor statistic.**
  - Per text t ∈ {wikitext, c4val1}: `floor_t = |NLL(e4b_serve, prefill-shaped, --ppl-chunk 64) − NLL(e4b_serve,
    prefill-shaped, --ppl-chunk 128)|`, over the same 2,048 scored targets.
  - **`F = max(floor_wikitext, floor_c4val1)`.**
- **"Within the floor":** a Δ is within the floor iff `|Δ_t| ≤ F` on **both** texts. If F is unread (either chunk row
  not VALID), every prediction that uses it is UNREAD.
- **Matched-arm disagreement** is reported beside the floor and never called the floor. It is the gap between vLLM and
  SGLang-marlin, both Marlin W4A16 on the same bytes (G4).

## The instrument

**SC1's teacher-forced NLL, in nats.** SC1's texts (wikitext-2 test; C4 validation shard 1) and SC1's window rule:
- `step_decomp._k8_window`, read by `sc1_prompts.window_record`;
- 2,561 ids: `prompt_len` 512 and 2,048 scored targets `ids[513..2560]`;
- a `text_sha`. A row whose sha differs from the window's is VOID.

**gpt-oss is chat-only.** `step_decomp`'s own note says it scores about 2,000 ppl on bare wikitext. So the windows use
the **chat frame** (`--ppl-chat`, suffix `<|channel|>final<|message|>`): an 80-token template prefix, then the corpus as
the assistant's reply in the final channel.

**The template's date is pinned to 2026-10-05.** The template writes `Current date: <today>`, so the window and its sha
would otherwise change with the day. With the date pinned, the proof's and the reading's windows are byte-identical. Every
engine scores the same ids from the window file. e4b's `step_decomp` rebuilds the window itself under the same pin, and
its sha is checked against the file's.

**Two shapes, as SC1 defines them:**
- **prefill-shaped:** the window scored in chunks (T > 1), or in one call where the engine takes it;
- **served-shape:** the generated position after a cached prefix (T == 1), 2,048 steps.

**The arms** (all on wikitext and c4val1, in both shapes, except where noted):

| arm | engine and arithmetic |
|---|---|
| `e4b_serve` | e4b `step_decomp` at SC1's `K8ARGS`, with serve_paged's gpt-oss env through P42's hook: `E4B_SERVE_EXP_INT4=1 E4B_INT4_KEEP_NF4=1` (attention int4 off). Prefill at chunk 128; chunk 64 as well, for the floor |
| `e4b_nf4` | the same with `E4B_SERVE_EXP_INT4=0`: NF4 experts everywhere, the control |
| `vllm` | vLLM 0.30.0, Marlin W4A16, TRITON_ATTN (SC2g's settings); SC1's scorer |
| `sglang_native` | SGLang 0.5.20's default runner, `flashinfer_mxfp4` (cutlass_sm120, W4A8 with MXFP8 activations); triton attention, radix on, one request at a time. The engagement check requires the resolved runner |
| `sglang_marlin` | the same with `--moe-runner-backend marlin` (W4A16) |
| `llamacpp` | llama.cpp `552f18f`, **the published GGUF, attention Q8_0** (labelled, not like-for-like with the bf16-attention engines); SC1's libllama harness, default MMQ (W4A8 decode, **W4A4** prefill) |
| `llamacpp_q8` | the same at `GGML_CUDA_MMQ_PREC=q8` (read at run time, `ggml-cuda/mmq.cu:96`): W4A8 prefill, llama.cpp's own arithmetic control |

**What e4b's two shapes run.** This is from the code at main. `step_decomp`'s K8 path keeps device grouping at its
module default, off.
- **Served-shape:** 4 rows per step, so the **MXFP4 GEMV on int8 activations** (W4A8), with the paged fp8 decode
  attention. This is serve_paged's **B = 1** decode arithmetic. The 512-token prompt goes through the kept NF4 stacks.
- **Prefill-shaped:** chunk × 4 rows, so the **kept-NF4 host-grouped M-tile**, never K21. Attention here is
  **transformers' eager attention**, not serve_paged's paged prefill. These rows read e4b's **expert** arithmetic in
  prefill, not its prefill attention.

**The shared harness is not edited.** P39's `step_decomp.py`, SC1's `sc1_prompts.py` and P42's hook are byte-pinned by
other lanes. `bench/sc2/sc1g_k8.py` runs them unmodified, with two things set in its own process:
- the chat date pin: it replaces `transformers.utils.chat_template_utils.datetime`, which `strftime_now` reads at render
  time, and checks that the rendered prefix carries the pinned date;
- the route record.

**One revision check.** `step_decomp` loads the model by id, with no revision. Before the e4b arms, box I asserts that
the hub's `main` is the pin `6cee5e81`, which it has been since 2025-08-26. Otherwise the e4b arms are refused.

## The route record (a gate: observed, not inferred)

`sc1g_k8.py` writes `routes/<arm>.<pid>.json` at exit, holding e4b#1129's `hot_residency.ROUTE_SEEN` and
`paged_attention.ATTN_SEEN`.
- Only a process that imported e4b's `hot_residency` writes, and it writes atomically.
- **A missing, empty or `null` record FAILS the row.** atexit does not run on SIGALRM, `os._exit` or a kill, and the arms
  run under `perl alarm`. `null` means the build predates the counters.

**Gates:**
- **`e4b_serve` served-shape:** `mxfp4_gemv|le256` ≥ 2,048 steps × 24 layers. No `mxfp4_*|gt256`, and no route outside
  {`mxfp4_gemv|*`, `nf4_mtile_host|*`}.
- **`e4b_serve` prefill-shaped:** at least one `nf4_mtile_host|*`, and **no `mxfp4_*|gt256`** (KEEP_NF4 observed).
- **`e4b_nf4`:** a non-empty record with no `mxfp4_*` route at all.
- **`attn_seen`** is **recorded, not gated.** The prefill-shaped rows run transformers' attention, which #1129 does not
  count, and decode attention is not counted at all.

**The e4b pin** is this registration's merge commit, after e4b#1129 (`7ffe2c05`). Box I's tripwire refuses an e4b without
`ROUTE_SEEN`/`ATTN_SEEN`.

**llama.cpp's control engages or is UNREAD.** The harness does not report which MMQ precision ran. If `llamacpp` and
`llamacpp_q8` give prefill NLLs equal within 1e-9 on both texts, the switch did not take: G3 is UNREAD (NOT_ENGAGED), and
the proof fails on it.

## Predictions (`bench/sc2/sc1g_reduce.py`, self-tested on 9 cases; partly unbased: P44 gives the ordering, the floor the scale)

| # | prediction | basis |
|---|---|---|
| G1 | `e4b_serve` served-shape is within the floor of `vllm` served-shape | P44: the MXFP4 store's KL 0.00192, far under any gpt-oss floor |
| G2 | `e4b_serve` prefill-shaped NLL ≥ its served-shape NLL on both texts (direction only) | P44: NF4 re-quantisation 0.0222 ≫ MXFP4 0.00192. Magnitude not registered; the shapes also differ in attention (eager vs paged) |
| G3 | `llamacpp` prefill-shaped is **not** within the floor of `llamacpp_q8` prefill-shaped: W4A4 moves quality | unbased: no measurement of W4A4 on gpt-oss |
| G4 | `vllm` and `sglang_marlin` are within the floor of each other in both shapes | two W4A16 Marlin forwards on the same bytes |
| G5 | every arm and every row VALID: rc 0, text_sha, scored range, steps, route gates | — |

Every arm's Δ against `vllm`, per shape and text, is also reported against the floor. These carry no bar beyond G1–G4.

## What a result licenses, and its scope

If G1 holds, the SC position rule's quality condition (**COMPARABLE**) holds for e4b's **B = 1 served arithmetic** (the
MXFP4 GEMV). Through G2's rows it also holds for its **host-grouped NF4 prefill experts**.

**It does not cover the paths serve_paged takes under load:**
- K21 for ≤ 256 rows;
- the **captured** NF4 M-tile above 256 rows;
- serve_paged's paged prefill attention.

Those cite P90/P44 (KL on an H100) and are not re-read here. A position sentence drawing on SC2g's B > 1 rows (TPOT under
load, capacity) must carry that scope. **Those rows are not quality-checked by SC1g.**

## Box, proof and budget

**Box I** (`SC1_BOX=I`): box G's CUDA 13 image, installs and pins.
- grouped-nf4-gemm v0.41.0 (`dc8f94ab`).
- openai/gpt-oss-20b @ `6cee5e81`; ggml-org's GGUF @ `ef9b12f2`.
- Box G's NF4 bake (the arena the NF4 control and the kept stacks load).
- Box I exports neither of SC1's prefill route pins.

**Proof** (`sc1g-prove-*`, guard 1.5 h): SC1's common proof, the reducer's self-test, the fetches, the bake, both windows,
and one full-length wikitext scoring per engine path:
- `e4b_serve` served and prefill (chunk 64), and `e4b_nf4` served;
- vLLM, both shapes;
- SGLang native and Marlin, prefill;
- llama.cpp default and q8, prefill.

It is PROVED only if every one of those rows is VALID with its route gate, and the q8 switch engaged.

**Reading** (`sc1g-5090-*`, guard 2.25 h): every arm, both texts, both shapes. The guards come from SC2g's measured phases:
installs plus fetch and bake took about 31 min, and SGLang's first start about 10 min. The deadline drops arms from the end, so
llama.cpp goes first.

**Budget**, including download at about $0.0078/GB (35 GB per box):
- proof ≤ $1.13 + $0.30;
- reading ≤ $1.69 + $0.30;
- **lane ≤ about $3.4**, each run under the $15 no-ask tier.

This is above the campaign's original $2.5 line for SC1g. The difference is the proof that a guard over 1 h
requires.

## Amendment A1 (2026-10-05): in-distribution text; the proof's control kept as descriptive; diagnostics and a kernel check

Registered after `sc1g-prove-1` and before any reading. It was reviewed by the maintainer session, whose design choices are
written in below.

### Why

`sc1g-prove-1` is PROVED (adertha-receipts, OK, **$1.051**). Its own rows refute the registration's premise that the chat
frame makes wikitext in distribution for gpt-oss. Every proof row scored the same ids, sha `506d7ca8`; mean NLL (top-1):

| row | activations | NLL (top-1) |
|---|---|---|
| vLLM prefill / served | Marlin W4A16, bf16 activations | 6.330 / 6.346 (0.083) |
| SGLang marlin prefill | W4A16 | 6.404 (0.080) |
| SGLang native prefill | **MXFP8** | 6.555 (0.066) |
| llama.cpp q8 prefill | **int8** per-32, Q8_0 attention | 7.060 (0.056) |
| llama.cpp default prefill | **W4A4** | 7.776 (0.031) |
| e4b prefill, chunk 64 | NF4, bf16, HF eager attention | 6.188 |
| e4b NF4 served | bf16, paged fp8 KV | 6.437 |
| e4b MXFP4 served (GEMV) | **int8** per-32, paged fp8 KV | 6.674 |

- **Perplexity is about 500–2,400 and top-1 3–8%.** The text is pathological for gpt-oss, as step_decomp's own help text
  warns for bare wikitext.
- **On it, every engine loses NLL as its activations coarsen:** MXFP8 +0.15, int8 per-32 +0.24 (e4b vs its NF4) to +0.66
  (llama.cpp vs vLLM), W4A4 +1.45. Two independent int8 per-32 implementations both pay, so this is the text plus the
  activation scheme, not an e4b-only kernel fault.
- **This contradicts no prior number.** P44's in-distribution short prompts (~29 tokens) read the same GEMV at KL 0.0019 on
  an H100.

### The texts

- **Graded: `conv1`, `conv2`.** These are the first two HuggingFaceH4/ultrachat_200k `test_sft` conversations, by dataset
  index at revision `8049631c`, whose **single** rendering in gpt-oss's chat template (date pinned) reaches 2,561 tokens.
  Assistant turns are rendered in the final channel, and conversations are never concatenated. Each window is that
  conversation's first 2,561 ids, with its sha.
  - The rule selects index 25 (3,019 tokens) and index 108 (2,712 tokens).
  - Their scored targets are 96.7% and 90.1% assistant content (`scored_target_roles`, read from the template's special
    tokens).
  - The shared scorers record mean NLL only, so an assistant-only NLL is not read. The role shares are recorded instead.
- **Descriptive: `wikitext`** (chat-framed, as the proof scored it), named for what it showed: activation-quantization cost
  on gpt-oss's out-of-distribution text. **c4val1 is dropped.**
- **Generated text (gpt-oss's own samples) is not used.** The arm that generated it would score it with its own arithmetic.

### The rule under A1

- **The floor's statistic is unchanged and measured on `conv1`, `conv2`:** F = max over them of |NLL(chunk 64) − NLL(chunk
  128)|. It is not reused from wikitext.
- **G1–G5 are read on `conv1`, `conv2` only.**
- **The wikitext control** is reported with its own chunk-pair gap and every arm's Δ against vLLM, never graded.
- **G6, new:** `gemv_mxfp4_b32` agrees with its exact reference on the 5090.
  - `bench/sc2/sc1g_gemv_check.py` compares the kernel with `dequant(quant_x_rows(x)) @ dequant_mxfp4(W_e)^T`, fp32, then
    **rounded to bf16** (the kernel's output dtype). KERNEL_AGREES iff ≤ 1e-3 relative on every call.
  - It runs on activations the served arms capture (layers 0, 6, 12, 18, 23; 16 steps; gate_up and down) and on synthetic
    rows (normal and 100× outlier channels).
  - A mutation arm, whose references read the wrong expert, must DISAGREE. If it agrees, the check is inert and G6 is
    UNREAD.
  - It also reports the int8 scheme's own error, exact vs raw activations.
  - **The same script already ran on the A2000 (sm_86, $0, correctness only; `bench/sc2/sc1g-a2000/`):** KERNEL_AGREES at
    ≤ 4.4e-5, mutation 1.47, scheme error 0.5% on normal rows and 1.0–1.3% with outliers.
- **The diagnostics** run on `conv1`, `conv2`, are e4b only and descriptive. What each one moving would mean is fixed now:
  - `served at GNF4_PDL=0`: moving means an ordering (PDL) bug in the served T == 1 path.
  - `served with the folds off` (`E4B_FUSE_T1_GLUE=0 E4B_FUSE_T1_GLUE_R2=0 E4B_FUSE_ROUTER_EPI=0`): moving means a fold bug.
  - `--ppl-oracle eager --ppl-chunk 1`, MXFP4 and NF4: the T == 1 expert routes under transformers' attention with a bf16
    cache. Against served, this isolates the paged fp8-KV decode path. Against prefill, it isolates T == 1 vs the M-tile
    route.
- **A hypothesis, registered so the reading is read against it:**
  - e4b's served-vs-prefill gap (+0.25 nats NF4 on the proof's wikitext, against vLLM's +0.016) is carried by the paged
    fp8-KV decode attention on gpt-oss's sinks and sliding windows. The register's
    `e4b.parity.gptoss.paged-vs-own-attention` reads 0.00288 nats, but on a different, earlier stack.
  - It is supported if chunk 1 closes **≥ 0.5** of the served-vs-prefill gap on both graded windows, and opposed if
    < 0.5.
  - Either way it is descriptive: the share is reported, not graded.

### Mechanics

- **e4b arms read the window from its file.** step_decomp builds only wikitext/C4 windows. Under `SC1G_WINDOW_FILE`,
  `sc1g_k8.py` replaces its `_k8_window` with one that returns the file's ids and refuses unless their sha matches. It
  then calls step_decomp's `main()`, unmodified. The shared harness files stay byte-identical to main.
- **The capture** wraps gnf4's `quant_x_rows` / `gemv_mxfp4_b32` in the served process. Nothing in gnf4 is edited.
- **An arm's stack follows SC1's FOLDS,** so the folds-off diagnostic can override them.

### Proof, reading, budget

- **`sc1g-prove-2`** (guard 1.25 h), A1's new paths on `conv1`:
  - the window from its file with capture (e4b served), prefill chunk 64, and eager chunk 1, every route gate read;
  - G6 on the captured activations, synthetic rows and the mutation;
  - one scoring each from vLLM (prefill), SGLang native (prefill) and llama.cpp q8 (prefill).
  - PROVED only if every one is VALID and G6 HOLDS.
- **`sc1g-5090-*`** (guard 2.5 h), in this order: the e4b rows and diagnostics on `conv1`, `conv2`; the e4b rows on
  wikitext; G6; vLLM, SGLang (native, Marlin) and llama.cpp (default, q8) on all three windows. The deadline drops from the
  end.
- **Spend:** `sc1g-prove-1` was $1.051. A1 adds proof ≤ $0.94 + $0.30 and reading ≤ $1.88 + $0.30, so **the lane is about
  $4.5**. That is above the registration's ~$3.4 because of the re-proof A1 needs; each run stays under the $15 no-ask
  tier.

## Amendment A2 (2026-10-05): an e4b-only diagnostic box first, because e4b's own served path reads far worse than its prefill

Registered after `sc1g-prove-2` and before any further run.

### Why

**`sc1g-prove-2` ended HARNESS_ERROR ($1.306).** Its 1.25 h guard ran out on a slow host (machine 145701; the first e4b arm
started 72 min in):

| step | prove-2 | prove-1 |
|---|---|---|
| vLLM install | 23 min | ~5 min |
| SGLang install | 11 min | ~4 min |
| gpt-oss fetch | 9 min | ~3 min |

It still scored two rows on `conv1`, which is in distribution (ppl ≈ 2), and they change the lane's priority:

| e4b row (`conv1`) | NLL |
|---|---|
| served (MXFP4 GEMV, paged fp8 KV) | **0.905** |
| prefill (chunk 64, NF4 host M-tile, transformers' attention) | **0.720** |
| served − prefill | **+0.185 nats** |

That is far beyond any plausible floor, and e4b's own served path is the outlier. prove-1's wikitext rows split a similar gap
in two:
- NF4 served − NF4 prefill **+0.25** (both bf16 activations, so the paged path);
- MXFP4 GEMV − NF4 served **+0.24** (the int8 activations).

On the same text vLLM's served − prefill is +0.016.

**The kernel is not the fault, on sm_86 at least** ($0, correctness only, `bench/sc2/sc1g-a2000/`). The A1 kernel check ran
on prove-2's REAL captured `conv1` decode activations: 160 calls, layers 0/6/12/18/23, gate_up and down.
- `gemv_mxfp4_b32` agrees with its bf16-rounded exact reference **≤ 1.9e-4**; the mutation reads 1.5–2.5.
- The int8 per-32 scheme itself costs **0.45–0.94% mean relative error per GEMV output** (max 1.3%).
- Block crest factors are 7–22 on the gate_up inputs and 23–29 on the down inputs.

A cross-engine reading now would mostly measure this gap without saying where it lives. The diagnostics come first.

### What changed on the decode path since the 0.00288 read (the maintainer, from `git log`)

The register's `paged-vs-own-attention` 0.00288 entered claims.json at `d3e902c3` (#356, 2026-09-03). Its receipt is
private, so the commit it was read on is not known. Changes on the decode path since:
- **#363 / #367:** fp8 key-scale groups, 32-wide with power-of-two counts. For head_dim 64 the count floors at **4 groups,
  16-wide**, which keeps gpt-oss on the **f32 attention compute** path on sm_120. P30 measured 2 groups instead of 4 at
  **+0.108 nats on gpt-oss**, so key precision matters on this model.
- **#757 and the 09-29 fix:** bucketed CUDA-graph decode.
- **#999 / P111:** step-select on by default. It was verified on Qwen3 only.
- **#966:** the unbound fallback keeps sliding windows.
- **#960:** prefill flash, which sinks route around anyway.

**Graphs and step-select are already excluded here.** step_decomp's K8 loop runs at `--b1d-loop eager`, so prove-2's +0.185
was read with no CUDA graph and no step-select. Arms that switch them off would be no-ops. (They remain open for serve_paged
under load, which is SC2g's path, not this one.) **There is no bf16-KV switch:** serve_paged builds `Fp8PagedKV`
unconditionally.

### Box J (`SC1_BOX=J`)

- **What it is:** box I's code, with no vLLM, SGLang or llama.cpp install and no GGUF. **Guard 1.0 h**, so no proof
  precedes it. It installs e4b, fetches gpt-oss and ultrachat, bakes the arena, builds the same pinned windows, and runs only
  e4b arms plus the kernel check.
- **The truth anchor** is e4b's chunk-free full forward with NF4 experts and transformers' attention (`--ppl-oracle full`).
- **The fp8 KV cache is tested directly.** `--ppl-fq` applies the paged kernel's K/V roundings inside that full forward,
  from the prompt on (`--fq-from` default), exactly as the paged decode reads fp8.

**Arms, in priority order** (the deadline drops from the end):

| # | arm (`conv1` unless named) | isolates |
|---|---|---|
| 1 | MXFP4 served (capturing) | the row being explained |
| 2 | NF4 served | the served path with bf16 activations |
| 3 | NF4 eager chunk 1 | T == 1 NF4 route under transformers' attention with a bf16 cache |
| 4 | NF4 full | the truth anchor |
| 5 | NF4 full + fq `kv` (4 key groups, 1 value group: the kernel's) | fp8 K/V rounding alone |
| 6 | NF4 full + fq `k` | keys alone |
| 7 | NF4 full + fq `v` | values alone |
| 8 | MXFP4 eager chunk 1 | the int8 GEMV under the same attention as #3 |
| — | **G6**, the kernel check on this card | |
| 9 | NF4 full + fq `kv`, 16 key groups | finer key scales, modelled |
| 10 | MXFP4 served `--kv-groups 16` | finer key scales on the real kernel |
| 11–12 | NF4 and MXFP4 prefill (chunk 128) | the chunked comparison rows |
| 13–18 | `conv2`: NF4 served, NF4 chunk 1, NF4 full, NF4 fq `kv`, MXFP4 served, MXFP4 chunk 1 | replication |
| 19–20 | MXFP4 served at `GNF4_PDL=0`; with the folds off | ordering and fold bugs |

### Predictions per arm (registered; `sc1g_reduce.py`'s `diag_predictions`, read on `conv1`, `conv2` reported as replication)

| # | prediction | what holding means | basis |
|---|---|---|---|
| J1 | NF4 chunk 1 closes ≥ 0.5 of NF4 served − NF4 full | the paged fp8-KV decode path carries the gap | prove-1: NF4 served − prefill +0.25 with bf16 activations, against vLLM's +0.016 |
| J2 | NF4 full + fq `kv` reproduces ≥ 0.5 of NF4 served − NF4 full | the fp8 K/V rounding is the mechanism | P30: key groups 2 vs 4 cost +0.108 nats on gpt-oss |
| J3 | the `k` rounding costs more than the `v` rounding (each vs full) | keys dominate | the same P30 sensitivity; partly unbased |
| J4 | 16 key groups recover ≥ 0.035 nats (about 2× the arithmetic-order floor), both modelled (fq `kv` 4 → 16) and on the real kernel (served → served `--kv-groups 16`), on `conv1` | finer key scales are a candidate fix. **A HOLDS licenses a registered default read, not a default change** | P30's direction; magnitude unbased |
| J5 | MXFP4 chunk 1 ≥ NF4 chunk 1 (direction only) | the int8 GEMV costs NLL under identical attention | prove-1: +0.24 on wikitext; the A2000 check: 0.45–0.94% per output |
| J6 | PDL=0 within 0.005, and folds-off within 0.035 (about 2× the floor: the folds reorder arithmetic), of MXFP4 served | no ordering or fold bug | P113: PDL value-identical on the int4 GEMV; the folds are not bit-identical by design |

Each prediction is UNREAD if any of its arms is missing or not VALID; route gates apply to every e4b row as registered. G6 is
read as in A1.

- **The reduction** is `sc1g_reduce.py`. G1–G5 read UNREAD here by design: the comparator rows are absent.
- **Cost:** ≤ $0.75 + about $0.15 download (no comparator wheels or GGUF). Lane spend so far is $2.357.
- **Host:** prove-2's slow host (145701) cannot be excluded through the anchor class: its HARNESS_ERROR is a guard timeout,
  not a strict-anchor refusal. Box J's e4b-only installs fit 1 h even on that host, so none is applied. The later full
  reading's proof gets a 2.0 h guard.

### What follows (decided after box J, by its result)

- **If e4b's served path carries a defect** (paged fp8 KV or the T == 1 route), the cross-engine reading waits for the fix.
  Reading it first would grade the defect.
- **If box J finds no e4b-path defect** (the gap is the arithmetic as designed), the A1 reading proceeds with a proof guard
  of 2.0 h. prove-2 showed a slow host needs it.

## Amendment A3 (2026-10-05): box J again, to split the MXFP4 route's cost into weights, route and activations

Registered after `sc1g-diag-1` and before any further run.

### What box J read (`sc1g-diag-1`: adertha-receipts `a6a16350`, OK, **$0.709**; e4b `3e133cf7`, gnf4 `dc8f94ab`)

| row (`conv1`) | NLL |
|---|---|
| MXFP4 served (T == 1 GEMV on int8 activations, paged fp8 KV) | **0.905** |
| MXFP4 eager chunk 1 (the same GEMV, transformers' attention, bf16 cache) | 0.960 |
| NF4 served | **0.736** |
| NF4 eager chunk 1 | 0.736 |
| NF4 prefill (chunk 128) | 0.731 |
| NF4 full (chunk-free) | 0.811 |
| MXFP4 served `--kv-groups 16` | 1.035 |

- **The paged fp8-KV path carries no gap on `conv1`.** NF4 served − NF4 chunk 1 = +0.0003, and NF4 chunk 1 − NF4 prefill =
  +0.005. A2's premise (prove-1's wikitext) does not hold on in-distribution text.
- **The MXFP4 T == 1 route carries it.** MXFP4 − NF4 is **+0.169** served and **+0.225** under identical eager attention
  (chunk 1). J5 HOLDS.
- **The GEMV kernel is exact for its scheme on sm_120 as well (G6 HOLDS).** On 160 captured `conv1` calls it agrees with
  the bf16-rounded exact reference to ≤ 1.9e-4, and the mutation reads 1.47. The int8 per-32 scheme costs 0.70% mean relative
  error per output on those activations.
- **The modelled-fp8 arms are VOID, not REFUTED** (3.3–3.4 nats). `--ppl-fq`'s `_fq_eager_attention` omits gpt-oss's
  attention sinks, so it scores a different attention. J2, J3 and J4's modelled half are not read.
- **The chunk-free full anchor is not a truth anchor here.** NF4 full sits +0.075 above NF4 prefill on `conv1` and +0.013
  above NF4 chunk 1 on `conv2`. step_decomp's own docstring records that the full forward flips 4.5% of gpt-oss's router top-k
  choices against a chunked order. J1 read against it is set aside.
- **The NF4 M-tile at large M is not why** ($0, A2000, `bench/sc2/sc1g-a2000/a3_*`, the maintainer's request).
  - Setup: one layer's real experts (layers 0 and 12, gate_up and down), 2561 tokens × top-4 = 10,244 rows. Routing was
    uniform, and skewed with a largest group of 2,540 rows.
  - One call against 128-token chunks: both read ≤ 3.8e-3 per row against fp32, the bf16 floor, and they are bit-identical
    on all but a few rows.
  - The mutation (the reference reads the next expert) reads 3.4.
- **Finer key groups made the real kernel worse.** Served at `--kv-groups 16` reads **+0.130** over the default 4 groups
  (J4 REFUTED on the kernel).
  - e4b's fp8 pack is correct at 4, 8 and 16 groups (A2000, PACK_OK). The stored keys' error falls from 2.2e-2 to 2.0e-2 to
    1.6e-2 with finer groups, as it should. The mutation reads 1.42.
  - The regression therefore lies in the sm_89+ decode kernel or its call at 16 groups (e4b#1175). Refuse-until-validated
    for `--kv-groups 16` stands.
- **`conv2` bounds what one window can say.** NF4 served − NF4 chunk 1 = **−0.051** there, the largest NF4 path-to-path
  spread box J read. A 0.05 effect on one window sits inside it.
- The deadline dropped A2's arms 16–20 (MXFP4 on `conv2`, PDL=0, folds-off). Nothing was read on them.

**A correction to box J's prefill rows.** MXFP4 prefill (chunk 128) read 0.73058, identical to NF4 prefill to every digit. The
serve stack's `E4B_INT4_KEEP_NF4=1` sends rows above 256 to the kept NF4 stacks, so that row never read MXFP4 weights. A3's
weights arm runs at `KEEP_NF4=0`, where those rows take `mxfp4_grouped_v1` on bf16 activations. It is route-gated: only
`mxfp4_grouped_v1|*` may appear, `|gt256` must, and any `nf4_*` route VOIDs it.

### The question

Is MXFP4's +0.17 the **weights** (MXFP4 blocks against the NF4 requantization), the decode **route**, or the **int8 per-32
activations** the GEMV quantizes at T == 1? `E4B_MXFP4_GEMV=0` keeps the same weights and the same decode rows but runs them
through `mxfp4_grouped_v1` on bf16 activations. The route gate requires `mxfp4_grouped_v1|le256` on every step's 24 layers,
and the GEMV route VOIDs it.

### Box J under A3 (guard 1.0 h; priority order, the deadline drops from the end)

| # | arm | reads |
|---|---|---|
| 1 | MXFP4 served at `E4B_MXFP4_GEMV=0` (`conv1`) | the decode rows on bf16 activations |
| 2–3 | MXFP4 prefill at `KEEP_NF4=0`, and NF4 prefill (chunk 128, `conv1`) | the weights alone |
| 4–5 | MXFP4 served and NF4 served (`conv1`) | the gap, and K4's repeat of box J |
| — | **the attention check** (`sc1g_attn_check.py`, below) | e4b#1175 |
| 6–14 | MXFP4 served, NF4 served and GEMV=0 served on `conv2`, `conv3` and `conv4` | K2 and K5 across windows |
| 15 | MXFP4 served `--kv-groups 4` (`conv1`) | auto picks 4 at head_dim 64: a determinism control |
| 16–17 | MXFP4 served with the folds off; at `GNF4_PDL=0` (`conv1`) | fold and ordering bugs |
| 18–19 | MXFP4 prefill at `KEEP_NF4=0`, NF4 prefill (`conv2`) | K1's replication |

`conv3` and `conv4` are the next two `test_sft` conversations by the registered rule (`--n-conv 4`). `conv1` and `conv2` are
unchanged. Box I keeps two.

### Predictions (registered; `sc1g_reduce.py`'s `a3_predictions`, self-tested on 20 cases)

| # | prediction | what holding means | basis |
|---|---|---|---|
| K1 | MXFP4 prefill (`KEEP_NF4=0`) − NF4 prefill ≤ 0.05 on `conv1` | the weights cost no more than the path spread | MXFP4 is the checkpoint's native format and NF4 a requantization of it; partly unbased |
| K2 | (MXFP4 served − GEMV=0 served) / (MXFP4 served − NF4 served) ≥ 0.5 on `conv1`, and **pooled** over ≥ 3 windows | the int8 activations carry most of the cost | the int8 per-32 scheme's 0.45–0.94% (sm_86) and 0.70% (sm_120) mean error per GEMV output; block crest factors to 29 |
| K3 | folds-off within 0.035 and PDL=0 within 0.005 of MXFP4 served | no fold or ordering bug | J6's bands |
| K4 | MXFP4 served repeats box J's 0.904969 within 1e-4, and `--kv-groups 4` equals it within 1e-9 | the eager K8 loop is deterministic across hosts | the register's cross-box reproducibility of K8 |
| K5 | MXFP4 − NF4 served ≥ **0.10** on `conv1`, and on **every** window read (≥ 3) | the cost is real, not one window's | box J's +0.169; 0.10 is about 2× the 0.051 spread |

- Every row must score VALID and pass its route gate. A prediction missing an arm is UNREAD.
- The across-window reads need at least 3 windows with all three rows VALID. Below that they read UNREAD, never HOLDS.
- The full anchor is excluded from every A3 prediction.

### The attention check (e4b#1175; the maintainer, 2026-10-05: "kvg16 tail step on A3's box: yes")

- **What it runs:** `sc1g_attn_check.py`, through e4b's own `Fp8PagedKV`, on gpt-oss's geometry: 64 query heads, 8 KV heads,
  head_dim 64, sinks. It covers k_groups 4, 8 and 16, window 128 and full attention, and normal keys plus keys with three
  channels × 20.
- **What it compares:** the kernel against a dequantize-then-attend fp32 reference that sees the same stored bytes and the
  same bf16 q. The reference is rounded to bf16.
- **What else it reads:** the stored K and V against their bf16 originals (the pack, which the kernel and `reference_kv`
  share), and the compute mode each call ran. At head_dim 64 every group count is under the fp8 path's 32-wide minimum, so
  f32 is expected throughout.
- **Verdict:**
  - INERT if the no-sink mutation agrees;
  - PACK_BAD if any reconstruction error exceeds 0.1;
  - KERNEL_DISAGREES if the kernel is more than 1e-2 from the reference at any group count;
  - otherwise KERNEL_AGREES.
- **Where it goes:** the verdict is posted on e4b#1175.
- **What it licenses:** KERNEL_AGREES at 16 groups does not lift the refusal on `--kv-groups 16`. The served +0.130 would
  then be unexplained at the attention level, and a served re-read is a separate registration.

### What follows (decided by the result)

- **K2 HOLDS and K5 holds across windows:** the cost is the int8 activation scheme. An e4b issue is filed with these rows.
  The candidates are bf16 activations for the decode rows (the GEMV=0 route exists; its speed cost is SC2g's to read) or a
  finer activation scale. The cross-engine reading (A1) then grades e4b at its default and at GEMV=0, both stated.
- **K1 REFUTED:** the MXFP4 weights carry a cost of their own. It is reported against the comparators' MXFP4 rows, not fixed
  in e4b.
- **K5 REFUTED across windows:** `conv1`'s gap does not replicate. Report it per window and file nothing.
- **K3 or K4 REFUTED:** a fold, ordering or determinism defect. It is filed on e4b before anything else is read.
- **Cost:** ≤ $0.75 + about $0.15 download, the same box as A2. Lane spend so far is **$3.066**
  ($2.357 + `sc1g-diag-1`'s $0.709).

## A3 read (2026-10-05): the cost is the MXFP4 weights, not e4b's route, and it does not replicate across windows

`sc1g-diag-2` (adertha-receipts `28cd4d15`, OK, **$0.726**; e4b `a7891300`, gnf4 `dc8f94ab`). Its receipts are committed
at `bench/h2h-2026-10-02/sc1g/receipts/` with `sc1g-diag-1`'s, and the lines below re-derive from them
(`python bench/sc2/sc1g_reduce.py --dir bench/h2h-2026-10-02/sc1g/receipts/sc1g-diag-2/sc1g`). The host was 145701 again: its
install took 24 min, so 10 of the 19 arms ran and the deadline dropped k11–k19 (conv3's GEMV=0 row, conv4, kvg4, folds-off,
PDL=0, conv2's K1 pair).

**Rows (NLL).**

| window | MXFP4 served | GEMV=0 served (bf16 activations) | NF4 served | MXFP4-weights prefill (`KEEP_NF4=0`) | NF4 prefill |
|---|---|---|---|---|---|
| `conv1` | 0.90497 | 0.87375 | 0.73620 | 0.88718 | 0.73058 |
| `conv2` | 1.74705 | 1.74862 | 1.64969 | — | — |
| `conv3` | 0.91333 | — | 0.94626 | — | — |

**The instrument repeats bit for bit across hosts.** MXFP4 served on `conv1` is 0.904969107589033, box J's value to the last
digit. The three NF4 rows the two runs share are identical too.

**The registered predictions.**

| # | verdict | value |
|---|---|---|
| K1 | **REFUTED** | MXFP4-weights prefill − NF4 prefill = **+0.157** |
| K2 | **REFUTED** | share carried by the int8 activations: **0.185** on `conv1` (GEMV=0 still sits +0.138 over NF4 served), −0.016 on `conv2`; pooled UNREAD (2 windows) |
| K3 | UNREAD | folds-off and PDL=0 dropped at the deadline |
| K4 | UNREAD | kvg4 dropped; the repeat itself is exact (above) |
| K5 | `conv1` HOLDS (+0.169); `conv2` REFUTED (+0.097); **every window REFUTED** | per window +0.169 / +0.097 / **−0.033**; mean +0.078 |

**By A3's registered "what follows":**
- K1 REFUTED: the cost is in the MXFP4 weights. The int8 activations add about +0.03 on `conv1` and nothing on `conv2`.
  It is reported against the comparators' native-MXFP4 rows, not fixed in e4b.
- K5 REFUTED across windows: the gap does not replicate (on `conv3` MXFP4 reads lower). Report it per window and file
  nothing.
- By A2's rule, no e4b-path defect was found, so the cross-engine reading may proceed. Its instrument is amended first
  (below).

**A hypothesis, not a reading: NF4 flattery rather than MXFP4 harm.** P44 (`e4b.serve.p44.gptoss.store-r12.kl-vs-bf16.2026-09-19`)
measured both paths against a bf16 dequantization of the same shipped bytes:
- the native MXFP4 store is KL **0.0019** nats/token from it;
- the NF4 requant is **0.0222**, more than ten times further.

So NF4 is the less faithful path. Its lower teacher-forced NLL on these conversations is **most likely** the entropy flattery
P44 recorded on wikitext. That is a hypothesis resting on P44's KL; this lane has not tested it, and A4's fidelity
instrument is what would. The ultrachat answers are off-policy for gpt-oss: they are rendered without its analysis channel,
and a noisier model spreads probability onto them. **Consequence for this lane, whichever way that hypothesis reads:** ranking engines by
teacher-forced NLL on this text could grade flattery. Amendment A4 reads A1's cross-engine NLL as descriptive only, and adds a fidelity instrument
(KL to a bf16-dequant reference computed once on an 80 GB card). It is registered before any reading run.

**The attention check (e4b#1175): INERT, as registered, and the inertness is a design error.**

| k_groups | kernel vs reference | scheme (deq vs raw) | recon | compute ran |
|---|---|---|---|---|
| 4 | 1.91e-3 | 3.41e-2 | 2.51e-2 | f32 |
| 8 | 1.88e-3 | 3.21e-2 | 2.51e-2 | f32 |
| 16 | 1.72e-3 | 3.15e-2 | 2.51e-2 | f32 |

- The no-sink mutation reads 1.1–1.4e-3. N(0,1) sinks over about 2000 keys carry about 1/T of the softmax mass, so a
  kg16 path that mishandled sinks would hide inside the kernel's own error.
- The re-check uses gpt-oss's learned sinks from the checkpoint, a T sweep from 512 to 2560, and captured activations.
- **The write paths are cleared** ($0, A2000, `bench/sc2/sc1g-a2000/a3read_*`, `sc1g_prompt_append_check.py`).
  - The served loop writes its prompt through `append_prompt` and each decode token through `append_many`.
  - Both are **bitwise equal** to per-layer `append` on 24/24 layers, K and V, at k_groups 4, 8 and 16.
  - That held over a 512-token prompt plus 100 one-token decode steps crossing block boundaries.
  - The mutation (the layers' K reversed) reads 0/24 equal.
- `refuse until validated` for `--kv-groups 16` stands. The open suspect is the kernel at kg16 under real sinks and the
  served lengths.

**Spend:** the lane is at **$3.792** ($3.066 + $0.726).

## Amendment A4 (2026-10-05): grade engines by KL to the model's own function; NLL becomes descriptive

Registered before box R runs and before any box I reading. The maintainer reviewed the design: the named-token estimator,
calibration on R, R as a separate box, and K-A as a prediction rather than a gate.

### Why

A3's read showed that teacher-forced NLL on these conversations can rank the less faithful path first. The NF4 requant reads
lower NLL than the native MXFP4 store on two of three windows, while P44 measured the store ten times closer to the bf16
dequant reference. (That this is entropy flattery remains a hypothesis.) Ranking engines by NLL on off-policy text could
therefore grade flattery.

From A4 on:
- **A1's cross-engine NLL (G1–G5) is read DESCRIPTIVELY only.**
- **Engines are graded by KL from a bf16-dequant reference of gpt-oss-20b** (P44's reference: the shipped MXFP4 bytes through
  `Mxfp4Config(dequantize=True)`, refused if any packed tensor survives), at every scored position of the registered windows.

### The estimator (`bench/sc2/sc1g_kl.py`, self-tested on 12 cases)

**What R stores.** At each scored position, box R stores:
- the reference's top-64 token ids and their fp64 log-probs;
- its rest mass (as its own log-sum-exp);
- the target and its log-prob.

**What each engine returns.** Each engine returns its log-probs on exactly those 64 named tokens. Its rest is 1 − Σ. Then:

  KL65 = Σ_k p_k (log p_k − log q_k) + p_rest (log p_rest − log q_rest)

KL65 is exact on the partition {64 named tokens, rest}. By data processing it is a **lower bound** on the full-vocabulary KL:
truncation biases downward, the direction `kl_fidelity.assert_full_vocab` refuses. Hence the gate below.

**The registered rules:**
- **Rest clamp.** The rest bucket is clamped at **ε = 1e-9** on either side. A window whose engine-side clamps exceed **1 %**
  of its positions is UNREAD.
- **Coverage floor.** A window whose reference top-64 mass averages under **0.99** is UNREAD.
- **Top-K fallback,** for an engine or shape that cannot name tokens; none is planned. The engine's own top-K with
  **K ≥ 256**:
  - coverage = the reference top-64 mass whose tokens it returned, and a window under 0.99 is UNREAD;
  - an uncovered reference token gets the engine's residual spread evenly over the vocabulary it did not return;
  - the bound that gives it the whole residual is reported beside it.
- **Full-vocab fallback.** If it is ever used (fp16 log-probs over all 201,088 tokens, ≈ 0.8 GB per window), those artifacts
  are sha-pinned here before box I launches, box I refuses a mismatch, and SGLang reads UNREAD there.

### Box R (`bench/sc2/sc1g-r/`, one H100 NVL)

**Rate.** H100 NVL at a declared **$3.50/h**, maintainer-approved 2026-10-05 per tc1c's precedent. This is no policy change.
Guard **1.0 h**, so no proof precedes it.

**Inputs.** The five committed windows (`bench/h2h-2026-10-02/sc1g/receipts/sc1g-diag-2/sc1g/k8_window_*.json`), each
refused unless it hashes to its registered sha (`window_shas.json`):

| window | sha |
|---|---|
| conv1 | `3753e997…` |
| conv2 | `9bf650b8…` |
| conv3 | `997bc7c3…` |
| conv4 | `e8415e7a…` |
| wikitext | `506d7ca8…` |

**What R does:**
1. K0 controls on the host (`kl_fidelity.py --controls`).
2. The pinned fetch (gpt-oss-20b @ `6cee5e81`).
3. `sc1g_ref.py`: the reference scored **decode-shaped** (one token per forward over a KV cache, as P44) over the 2,048
   scored positions of every window. Each window's artifact is `ref_<src>.npz`, hashed.
4. The calibration (below).

**The calibration, which is the instrument gate.** Both full distributions exist only on R. KL65 is compared with the
full-vocabulary KL (fp64) on two real perturbations of the reference:
- **self:** the reference prefill-shaped against its decode-shaped self (arithmetic order, ~5e-4 scale). Its full KL is the
  instrument floor **F**, and F must stay under P44's 1e-2.
- **nf4:** the reference with every expert matrix fake-quantised to NF4 in place, against the reference, both
  prefill-shaped (requantisation, ~0.02 scale). The fake-quant is gnf4's `quantize_pack_nf4` + `dequant_ref`, cross-checked
  bit-for-bit against gnf4 on the box.

**Registered:** KL65 / KL_full ≥ **0.90** on both pairs, on every window. Otherwise the bucketed estimator is UNREAD for the
lane, and the full-vocab fallback above is registered instead.

**What 0.90 licenses.** The calibration covers two perturbation types: arithmetic-order reordering and NF4 requantisation. A
comparator whose deviation has a different tail shape (GPTQ-int4 attention, an engine's activation quantisation) is covered
by the lower-bound property, **not by the calibration**.

**R's verdict (`r_verdict.json`) is its own gate.** It is decided on R, before anything consumes the artifacts. Each check
reads OK / UNREAD / VOID:
- K0;
- the windows complete;
- floor F < 1e-2;
- the self ratio;
- the nf4 ratio;
- coverage;
- the NF4 fake-quant matching gnf4: **VOID** unless gnf4 imports on R and its `quantize_pack_nf4` + `dequant_ref` match
  `sc1g_ref.fake_nf4` bit for bit. An import failure must not leave the nf4 pair unverified.

R reads **R_OK** only when every check is OK. Box I starts only from an R_OK set.

**Descriptive, not predicted:** the reference's own NLL per window (decode- and prefill-shaped) and the NF4 fake-quant's.
The flattery hypothesis's direction is read off the reference itself.

### After R: the artifact registration

A short PR commits R's artifacts (about 1 MB per window), `ref_shas.json`, `r_verdict.json` and `r_calib.json` to
`bench/sc2/sc1g-r/ref/`, and adds them to the SC staging lists. Box I stages them into `$W/sc1g_ref` and refuses any KL arm
whose artifact is absent or does not hash to its registered sha (`i_ref`).

### Box I under A4 (supersedes A1's arm list)

**Windows.** Four conversations: `--n-conv 4` adds conv3 and conv4 by the registered rule. Each scored text matches R's sha.

**Arms, in priority order** (the deadline drops from the end):
1. The **named-token KL rows**, engine by engine so each server starts once. Each runs conv1–conv4, then wikitext.
   - e4b served, MXFP4 GEMV and NF4: `sc1g_k8.py`'s torch proxy records step_decomp's own served rows;
   - vLLM served (Marlin W4A16): `logprob_token_ids` = the named ids + the target;
   - SGLang served, flashinfer_mxfp4 then Marlin: `token_ids_logprob`;
   - llama.cpp decode, default MMQ then `MMQ_PREC=q8`: the harness's `--named`.
2. The descriptive prefill-shaped rows.

**When a row counts.** A KL row is VALID only when:
- the arm itself is VALID, route gates included;
- its named record is complete, with no void positions; for e4b the proxy's meta record must exist (missing = VOID, the
  broken-proxy case) and show one log_softmax row per step;
- its target log-probs reproduce the arm's own mean NLL to **1e-9**.

**The proof.** Box I's proof (2.0 h guard, per A2) reads one named KL row per engine path on conv1. These are e4b MXFP4,
vLLM, SGLang native and llama.cpp q8, each VALID against R's artifact (`sc1g_reduce.py --prove-a4`).

### Predictions (registered; `sc1g_reduce.py`'s `a4`, self-tested; pooled = the mean over the graded windows VALID, ≥ 3 needed)

| # | prediction | basis |
|---|---|---|
| K-A | e4b NF4 served KL65 ≥ **3×** e4b MXFP4 served KL65 on every graded window | P44: 0.0222 vs 0.0019 (11.6×) on its 200 prompts. **A prediction, not an instrument gate**: if it fails, that is a finding about NF4 on chat text |
| L1 | e4b MXFP4 served pooled KL65 ≤ **2×** the best comparator's | partly unbased: P44 gives e4b's 0.0019; the comparators were never measured |
| L2 | every engine serving the native MXFP4 weights (e4b MXFP4, vLLM, both SGLang, both llama.cpp) reads pooled KL65 **below R's NF4-requant scale** (the nf4 pair's full KL, pooled) | they serve the checkpoint's own weights; partly unbased for llama.cpp's q8_1 activations |

- Every prediction is UNREAD unless box R read R_OK and every artifact matched its sha.
- **Descriptive:** the per-window KL65 table, the pooled rank, each row against the floor F (within F = indistinguishable
  from the reference's own arithmetic order), and NLL beside each row.

### Cost and order

1. This registration.
2. Box R: ≤ ~$4.
3. The artifact registration.
4. Box I's proof: 2.0 h guard, ≈ $1.7.
5. Box I's reading.

Each run stays under the $15 no-ask tier. The lane is at **$3.792**.

## A4 box R read (2026-10-06): `R_NOT_OK`, so A4's KL grading is UNREAD

**The run.** `sc1g-r-8` (adertha-receipts `2c49f7d5`, $0.5556; runpod:secure H100 NVL; from `9dc7b59a`) ran to its verdict.
The registered rule is unchanged from `8a1513a4`: digest `ee122b74…`, `test_box_r_rule_is_the_registered_one`. The verdict
re-derives from the committed receipt (`sc1g_ref.py --reverdict`, see the receipts README, which lists every attempt, r-1 to
r-8; box R total $0.6904). By the registration, **R is not re-run to get a pass**.

| check | result | measured (conv1 / conv2 / conv3 / conv4 / wikitext) | bar |
|---|---|---|---|
| K0 | OK | all passed on the host | |
| windows complete | OK | all five | |
| NF4 fake-quant matches gnf4 | OK | bit-equal, 1,536 matrices | |
| coverage (reference top-64 mass) | **UNREAD** | 0.961 / 0.929 / 0.978 / 0.966 / 0.773 | 0.99 |
| calib self (KL65 / KL_full) | **UNREAD** | 0.827 / 0.841 / 0.882 / 0.876 / 0.791 | 0.90 |
| calib nf4 (KL65 / KL_full) | **UNREAD** | 0.793 / 0.746 / 0.910 / 0.886 / 0.785 | 0.90 |
| floor F (decode vs prefill full KL) | **UNREAD** | 5.7e-3 / 7.0e-3 / 1.8e-3 / 1.9e-3 / 2.18e-2 | 1e-2 |

**Why, per check:**
- **Coverage and the calibrations.** On these texts gpt-oss's next-token distributions keep 2–7 % of their mass outside the
  top 64, and 10–25 % of the KL lives there. The gate did its job: KL65 would have under-read by that much.
- **The floor.** At 2,560-token windows the reference's own decode-vs-prefill KL (F) is 6–7e-3 on conv1 and conv2. P44 read
  5e-4 at 320 tokens. That is above the ~2e-3 expected of an engine, so even a full-vocabulary KL cannot separate engines below
  about F there. On wikitext, the out-of-distribution control, F is 2.2e-2.
- **The same windows are sensitive to everything.** The NF4 fake-quant's full KL is large on conv1 and conv2 (0.108, 0.139)
  and small on conv3 and conv4 (0.021, 0.023, P44's scale).

**Descriptive, as registered: the true model's NLL beside e4b's served rows.** This supports A3's hypothesis; it does not
settle it.

| window | reference (decode) | e4b MXFP4 served | e4b NF4 served | e4b GEMV=0 served | NF4 fake-quant of the reference (prefill) vs the reference (prefill) |
|---|---|---|---|---|---|
| conv1 | 0.884 | 0.905 | 0.736 | 0.874 | 0.746 vs 0.864: **−0.118**, at full KL 0.108 |
| conv2 | 1.821 | 1.747 | 1.650 | 1.749 | 1.702 vs 1.840: **−0.139**, at full KL 0.139 |
| conv3 | 0.895 | 0.913 | 0.946 | — | 0.933 vs 0.892: +0.041, at full KL 0.021 |
| conv4 | 1.590 | — | — | — | 1.595 vs 1.587: +0.008, at full KL 0.023 |

- **The evidence.** The reference's own NF4 requantisation reads BELOW the true model's NLL exactly where its full KL is
  largest. A less faithful copy scoring a lower NLL is flattery, not fidelity. e4b's NF4 served rows sit below the reference
  on the same windows.
- **MXFP4 served** sits within about 0.02 of the reference on conv1 and conv3. Its −0.074 on conv2 is **unexplained**.

**Next.** Amendment A5 is to be registered before anything runs. It moves to A4's registered full-vocabulary fallback and
makes the floor a per-window gradability rule.

## Amendment A5 (2026-10-06): the full-vocabulary KL, graded per window above its own floor

Registered before box R re-runs and before any box I reading. It applies the full-vocab fallback that A4 registered for exactly
this case. The maintainer reviewed the design on #1222's merge: fp16 storage checked on both pairs, per-window drops with a
bound for an unresolved denominator, an `R_NO_GRADABLE` outcome, the digest pinned beside A4's, a Pool 3 copy, and the vLLM
flag stated.

### Why

A4's box R read `R_NOT_OK` on three checks, and two causes need two different fixes:

- **Coverage and the two calibrations are moot under a full-vocabulary KL.** They measured what the 64-token partition lost.
  Over all 201,088 tokens nothing is lost, so the estimator is the quantity itself and not a lower bound. Coverage,
  `calib_self` and `calib_nf4` are **dropped by name** (`A5_DROPPED`) and stay in R's receipt as descriptive.
- **The floor is not moot.** No estimator separates engines below the reference's own decode-vs-prefill KL. A4 applied
  F < 1e-2 to the lane, and wikitext's 2.2e-2 sank it. A5 applies the bar **per window**: a window is graded only if its
  own F < 1e-2. Wikitext is the out-of-distribution control and is never graded.

### The estimator (`bench/sc2/sc1g_kl.py`, self-tested on 21 cases)

**What R stores.** For each window, R stores the reference's decode-shaped log-softmax over the whole vocabulary:
- format: fp16, computed in fp64 and then cast (`full_rows_fp16`);
- shape: `[2048, 201088]` per window, about 0.82 GB, written as `ref_full_<src>.npy`;
- each file is hashed into the receipt (`full_artifacts`: sha256, shape, dtype, bytes);
- a masked `−inf` is stored exactly as `−inf`. A NaN, or a finite value that became `−inf` in the cast, is refused.

**How an engine is read.** At every scored position, each engine reads `kl_full_support` in fp64 over the full vocabulary.
Both sides are renormalised, and entries where p_ref = 0 stay out of the sum. Each position records four values:
- **the full KL**, KL(p_ref ‖ p_eng). It is `+inf` where the engine masks (`−inf`) a token that the reference gives mass.
- **the common-support KL:** the reference restricted to the engine's finite support and renormalised there, against the
  engine.
- **the masked mass:** the reference mass on the engine's `−inf` tokens.
- **the masked count:** how many tokens the engine sets to `−inf`.

The maintainer's review of #1223 found why this is needed: gpt-oss's 201,088-entry head carries padded or unused ids.
- An engine may mask those ids where R gives them tiny finite mass.
- A read that **raised** on that, as A4's `kl_full_rows` does, would crash the arm, or e4b's server mid-run.

So the read never raises on masking. When a position's read does fail, the position is recorded **void** with its reason,
and the run continues. A read fails on:
- a NaN or `+inf` engine entry;
- a negative KL beyond −1e-9 (misalignment);
- a vLLM position that returned fewer entries than the vocabulary, which would otherwise read as masking.

The same holds at all three call sites:
- e4b's serving proxy;
- vLLM's served loop;
- the llama.cpp harness, which writes five columns per step and NaN for a void position.

The C++ read was checked against the Python one on masked and unmasked rows.

Only the reference side is stored in fp16. The engine side is the engine's own logits as it computes them, and the
instrument adds no rounding to them.

**The fp16 storage check.** It runs on R, on both of R's pairs, because both full distributions exist only there. For each
window and each pair, `storage_error` computes:
- the mean KL from the fp64 reference rows;
- the mean KL from the same rows after the fp16 cast.

The registered check: |KL_fp16 − KL_fp64| ≤ **0.1 × F** (`STORAGE_F_FRACTION`) on every gradable graded window, taking the
max over the two pairs. A window that fails makes the check VOID, and R reads `R_NOT_OK`. Rounding the reference to fp16
must cost at most a tenth of the floor that the window is graded above.

### Box R under A5 (`sc1g_ref.verdict_a5`; inputs, model, fetch and windows unchanged from A4)

**What R does.** The same flow as A4, except:
- it writes the full rows plus their shas;
- it measures the storage error on both pairs;
- it no longer writes the KL65 named-token artifacts.

**Validity checks, each OK / VOID:**
- K0;
- the windows complete;
- the NF4 fake-quant matching gnf4 (OK only when gnf4 imports AND matches bit for bit);
- the full artifacts written and hashed for every window;
- `fp16_storage`.

**Gradability.** Gradable = the graded windows (conv1–conv4) whose own F < **1e-2** (`GRADABLE_F_MAX`).

**Outcomes:**
- **`R_NOT_OK`:** a validity check is not OK. Box I does not run.
- **`R_NO_GRADABLE`:** the checks are OK, but fewer than **3** graded windows are gradable (`A5_MIN_GRADABLE`). This is a
  cost gate, not a failure of the instrument: K-A, L1 and L2 each need ≥ 3 windows, so box I would buy nothing. Box I is
  not launched. `sc1g_ref.py` exits 2 and `sc1g_r_run.sh` finishes with rc 32.
- **`R_OK`:** otherwise. The verdict lists the gradable windows, F per window and the storage error per window.

A4's measured F (conv1 5.7e-3, conv2 7.0e-3, conv3 1.8e-3, conv4 1.9e-3) would make all four conv windows gradable. That is
the same arithmetic on a new run, and it is not assumed here. A5 needs a new box R run because A4's never wrote full rows.
That run is made once; **R is not re-run under A5 to get a pass.**

**The rule pin.** `test_box_r_rule_is_the_registered_one` pins the digest
**`7f307c39c896e456f63e99f09ce33944cd12b4cf25275ae8801b761f152469b7`**. It covers:
- `verdict_a5`, `score`, `load_reference`, `window_ids`, `fake_nf4`, `fake_nf4_experts_` and `gnf4_crosscheck`;
- the constants `SELF_CONSISTENCY_MAX`, `SRCS`, `NF4_LUT`, `BLOCK`, `A5_GRADED`, `A5_MIN_GRADABLE` and `A5_DROPPED`;
- the whole of `sc1g_kl.py`.

The test's mutations show that changing `A5_MIN_GRADABLE`, `STORAGE_F_FRACTION` or `GRADABLE_F_MAX` changes the digest.

A4's digest **`ee122b74…`** covered `score()`, which A5 changes, so it cannot stay a live pin. A4's rule is pinned instead in
two ways:
- its `verdict()` function's own sha, **`0f521ac4d77d3a17daf342ca3cea7d20bf1f2f79b03d5c7adbffef6128c18102`**, unchanged;
- `test_the_a4_read_still_rederives_from_its_committed_receipt`: `--reverdict` on `sc1g-r-8` still reads `R_NOT_OK rule=A4`
  with `matches_recorded=True`.

`--reverdict` dispatches on the receipt's `rule`, so each read re-derives under its own rule. For A5 it also re-hashes the
full rows when they are present.

### Where the full rows live

The rows (~4.1 GB for five windows) stay outside every repository:

1. **R's driver** fetches `ref/full/` separately from the receipt, to `$SC1G_REF_FULL_STORE/<run id>/` on the controller
   (default `~/sc1g-ref-full`). It prints a `FULL_FETCHED` sha line per file. The receipt itself carries only
   `ref/SHA256SUMS` and the shas in `r_calib.json`.
2. **The durable copy** goes to the QNAP's **Pool 3** (HDD bulk), at `/share/ZFS19_DATA/sc1g-ref-full/<run id>/`, with its
   `SHA256SUMS`. It is re-hashed after the copy and must match R's receipt. The path and shas are recorded in the
   registration PR.
3. **Box I** gets the rows staged from the controller copy (`sc1_drive.sh`, `SC1G_REF_FULL_SRC`) into `$W/sc1g_ref_full`.
   Every KL arm re-hashes its window's file against the registered sha (`i_ref_full`) and is REFUSED on any mismatch.

### After R: the sha registration

If R reads `R_OK`, a short PR commits:
- `bench/sc1/sc1g_ref/ref_full_shas.json`, `r_verdict.json` and `r_calib.json` (the rows themselves are not committed);
- R's receipt;
- the attempts list.

`sc1g_ref` is in the SC staging lists (`sc1_drive.sh`, `make_pin.sh`, `test_sc1_staged_pin.py`), so box I gets them at
`$W/sc1g_ref`. The reading is UNREAD unless those files say rule A5 and `R_OK` (`a5_refs`).

### Box I under A5 (supersedes A4's arm list)

**Arms, in priority order** (the deadline drops from the end). Each engine starts once, and each runs conv1–conv4 and then
wikitext.

1. **e4b served, MXFP4 (GEMV) then NF4.** `sc1g_k8.py`'s `full_capture` reads step_decomp's own served rows. A missing meta
   record, or a row count other than one per step, makes the row VOID.
2. **vLLM served (Marlin W4A16, TRITON_ATTN).**
   - **The flag:** `SamplingParams(logprobs=-1)` with **`LLM(max_logprobs=-1)`**, set whenever `SC1_REF_FULL` is set.
   - **Verification:** the first request must cover the whole vocabulary (`full_vocab_cover`), and so must every
     position. Covered means every id 0..V−1 is present. vLLM 0.30.0's V2 runner returns the generated token first
     and then every token, so V + 1 entries are accepted only when the one repeated id is entry 0 and both copies
     carry the identical log-prob (amended before the reading, after `sc1g-prove-a5-6`; see the proof record below).
   - **Failure:** if that does not hold, the KL row is **VOID**. A5 never falls back to top-K or named tokens.
3. **llama.cpp decode** (the published GGUF; default MMQ, then `MMQ_PREC=q8`). The harness reads R's rows as raw fp16
   (`--ref-full`) and writes (KL, target log-prob) per step (`--kl-out`). A size mismatch is refused (rc 3).
4. **The descriptive prefill-shaped NLL rows**, with no KL.

**SGLang native and Marlin are UNREAD by registration.** They have no KL arm, for two reasons:
- SGLang's API returns top-K (`top_logprobs_num`) and named-token (`token_ids_logprob`) log-probs. Asking it to name
  every token would mean 2,048 × 201,088 ≈ 4.1 × 10⁸ log-probs per window as JSON, and no full-distribution path was
  built or proved for it;
- A4 registered that SGLang reads UNREAD under the full-vocab fallback.

SGLang therefore leaves L1's comparators and L2's engines.

**When a row counts.** A full-KL row is VALID only when:
- the arm is VALID, route gates included;
- its record is complete and finite, and carries the support arrays;
- **no position is void;**
- **the engine masks no token that carries reference mass** (masked mass 0 at every position). Masking ids where the
  reference is itself `−inf` is fine; the count is reported;
- the reference sha it recorded is the registered one;
- its target log-probs reproduce the arm's own mean NLL to **1e-9**.

Every row reports its support either way: positions masked, the max and mean masked mass, the max masked count, the
common-support KL, and the void positions.

**The proof.** Box I's proof (2.0 h guard, per A2) reads three rows on conv1, each VALID against R's registered rows
(`sc1g_reduce.py --prove-a5`):
- e4b MXFP4 served;
- vLLM served;
- llama.cpp q8 decode.

**The support rule is decided on the proof, not on the data.** The proof prints, per engine, the reference mass on the
tokens that engine masks (`SC1G_PROVE_A5_SUPPORT`).
- **If it is zero for every engine:** A5 stands as written.
- **If any engine masks reference mass:**
  - that engine's rows are VOID under A5 as written, and the proof reads NOT PROVED (`SC1G_PROVE_A5_SUPPORT_RULE_NEEDED`);
  - before the reading box, an amendment registers a **common-support rule with a mass bound**, set from the proof's
    masked-mass numbers;
  - every row already records its common-support KL, so the rule applies to the proof's own committed receipt without a
    re-run;
  - the rule never looks at the reading's data.

### Predictions (registered; `sc1g_reduce.py`'s `a5`, self-tested on 37 cases with A1–A4's; graded over the gradable windows only)

**When a KL is resolved.** An engine's KL on a window counts as **resolved** only when it exceeds that window's F. At or
below F it cannot be told apart from the reference's own arithmetic order.

| # | prediction | the floor rule |
|---|---|---|
| K-A | e4b NF4 served KL ≥ **3×** e4b MXFP4 served KL on every counted window | NF4 unresolved, so the window is **dropped**. MXFP4 unresolved (with NF4 resolved), so **F is used as MXFP4's upper bound** in the denominator, which makes the test conservative |
| L1 | e4b MXFP4 served pooled KL ≤ **2×** the best comparator's (vLLM, llama.cpp default, llama.cpp q8) | a window whose best comparator is unresolved is **dropped**. An unresolved e4b takes **F as its upper bound** |
| L2 | every native-MXFP4 engine (e4b MXFP4, vLLM, both llama.cpp) reads pooled KL **below R's NF4-requant pooled full KL** | pooled per engine over the windows where both that engine and the NF4 pair are resolved |

- Each prediction needs **≥ 3** counted windows, or it is UNREAD.
- All of them are UNREAD unless R read `R_OK` under A5 and every row matched its registered sha.
- **Descriptive:** the per-window KL table, each row's within-F flag, NLL beside each row, and wikitext's rows as the
  control.

### Cost and order

1. This registration.
2. Box R under A5: H100 NVL at the declared $3.50/h, guard 1.0 h. A4's run took ~10 min. Under A5 it adds the
   full-row write, two storage passes and the ~4.1 GB fetch.
3. The Pool 3 copy and the sha registration.
4. Box I's proof: 2.0 h guard, ≈ $1.7, plus staging ~4.1 GB of rows to the box.
5. Box I's reading.

Each run stays under the $15 no-ask tier. The lane is at **$4.482**.

## A5 box R read (2026-10-06): `R_OK`, four conv windows gradable

**The run.** `sc1g-r5-2` (adertha-receipts `fde4f454`, $0.826) ran on a Vast verified H100 NVL, machine 141791, at $2.07/h,
under the declared $3.50. It launched from A5's merge `dfc5bdaf`.
- The rule digest `7f307c39…` and `staged-r.sha256` were re-checked at that SHA before launch.
- `sc1g-r5-1` was refused at the provider first: no RunPod Secure stock, no instance, $0.
- The verdict re-derives from the committed receipt (`sc1g_ref.py --reverdict`, pinned by
  `test_the_a5_read_rederives_and_the_registered_shas_are_rs`).

| check | result | measured (conv1 / conv2 / conv3 / conv4 / wikitext) | bar |
|---|---|---|---|
| K0 | OK | all passed on the host | |
| windows complete | OK | all five | |
| NF4 fake-quant matches gnf4 | OK | bit-equal | |
| full artifacts | OK | five `[2048, 201088]` fp16 files, sha-recorded | |
| floor F (decode vs prefill full KL) | per window | 5.67e-3 / 7.03e-3 / 1.84e-3 / 1.85e-3 / 2.18e-2 | gradable if < 1e-2 |
| fp16 storage error (max over both pairs) | OK | 6.0e-7 / 4.8e-6 / 3.9e-7 / 9.2e-7 / 8.2e-7 | ≤ 0.1 × F on gradable windows |

- **Gradable: conv1, conv2, conv3, conv4.** Wikitext is the out-of-distribution control, so it is never graded (its F is
  2.18e-2).
- **The fp16 storage margin** is about 150× under the bar at its tightest (conv2: 4.8e-6 against 7.0e-4).

**Where the rows are.** They are registered in `bench/sc1/sc1g_ref/ref_full_shas.json`, with R's `r_verdict.json` and
`r_calib.json`. They live outside git in two places, each re-hashed against R's `SHA256SUMS`:
- the controller copy: `~/sc1g-ref-full/sc1g-r5-2/` on the mini;
- the durable copy: QNAP Pool 3, `/share/ZFS19_DATA/sc1g-ref-full/sc1g-r5-2/`.

**Descriptive: the reference is bit-reproducible across hosts.** Every number this run shares with A4's `sc1g-r-8` is
bit-identical: per window, F, the reference NLL (decode and prefill), the NF4 fake-quant's NLL and its full KL, which is
25 of 25 values. The two runs differ in:
- the H100 NVL card;
- the provider (RunPod, then Vast);
- the driver (580.126.09, then 595.71.05).

So the floor F is the reference's own decode-vs-prefill arithmetic, not run-to-run noise. Box I's engines are graded
against a fixed, repeatable function.

**Next, in order.** Box I's proof first (2.0 h guard, three conv1 rows, `--prove-a5`). Box I installs vLLM and llama.cpp
only, because SGLang has no A5 arm; its PROVED line names the rows the proof read (`kl_full=[e4b_serve vllm llamacpp_q8]`).
The proof prints each engine's masked reference mass:
- if every engine reads zero, the reading box follows;
- if any engine reads non-zero, a common-support rule with a mass bound is registered on the proof's numbers first.

## A5 box I proof record (2026-10-06): two of three rows VALID, no masked reference mass; vLLM's verification fixed

Box I's proof reads three conv1 rows (`--prove-a5`). These are the attempts, each launched from a merged SHA after the
pre-launch re-checks and the store probe:

| attempt | receipt (adertha-receipts) | outcome | $ |
|---|---|---|---|
| `sc1g-prove-a5-1` | `9db5ce6f` | disk 317 < 320: the lane's staged rows; fetch pulled them back → #1231. `host_evidence: false` | 0.241 |
| `sc1g-prove-a5-2` | `419c000e` | disk 317 < 320 again: the launcher orders a fixed 320 GB disk → #1235. `host_evidence: false` | 0.107 |
| `sc1g-prove-a5-3` | `4987fd57` | NOT_RUN: Vast 152169 never accepted the ssh key (host evidence) | 0.026 |
| `sc1g-prove-a5-4` | `81b7b16e` | the disk floor passed (317 + 4 staged); driver 570 < R580 on Vast 40093 (host evidence) | 0.058 |
| `sc1g-prove-a5-5` | `584188ca` | REFUSED at $0: an rc-18 receipt cited in the wrong exclusion class | 0 |
| `sc1g-prove-a5-6` | `24ca184e` | **ran: NOT PROVED (rc 23)**. e4b served and llama.cpp q8 VALID; vLLM VOID on the verification below | 0.598 |
| `sc1g-prove-a5-7` | none written | REFUSED[97] at $0, before `rent.py`: `pod-launch.sh` read `ADERTHA_REPO` (default `adertha-main`), not the checkout the kit had changed into | 0 |
| `sc1g-prove-a5-8` | `e0b46a90` | **PROVED**: all three rows VALID, zero masked reference mass on each; from `5d794520` (#1248), Vast 145701 | 0.955 |

**The support numbers, which are what the support rule is decided on.** In `sc1g-prove-a5-6`, both e4b served and
llama.cpp q8 decode read, on all 2,048 positions:
- masked reference mass 0;
- masked count 0;
- no void positions.

So **no common-support rule is registered for e4b or llama.cpp**. vLLM's support is unread. The proof's KL values are not
recorded here: they grade nothing, and the reading box measures conv1–conv4 itself.

**The instrument defect.** vLLM returned 201,089 entries for the 201,088-token vocabulary. That is not a gap: vLLM
0.30.0's V2 model runner builds `cat((sampled_token_ids, topk_indices))` in `vllm/v1/worker/gpu/sample/logprob.py`
(`compute_topk_scores`), so the generated token comes first and then every token, scored by one gather. A count check
calls that VOID, although every token is present.
- The check is now `full_vocab_cover`: every id present, and for V + 1 entries the single repeat must be entry 0 with an
  identical log-prob. Anything else is still VOID, with no downgrade.
- It applies at request 0 and at every position, on A5's path only. SC1's own arms keep their count check.
- **The rule is unchanged** (A5 reads the full vocabulary or nothing). Only the test of "full" was wrong.

**Next.** The proof re-runs, and must read all three rows VALID. vLLM's masked mass then decides whether it needs the
common-support rule.

## A5 box I reading (2026-10-06): K-A REFUTED, L1 HOLDS, L2 HOLDS

**The run.** `sc1g-5090-a5-1` (adertha-receipts `1ac9c0ff`) ran on a Vast RTX 5090, machine 145701, the same host as the
proof. It cost **$1.557**, ran for 90 minutes against its 2.5 h guard, and teardown is proven. It launched from `5d794520`,
the proof's commit, behind a gate requiring the PROVED proof receipt at that commit.
- Every KL row is VALID: no void positions, and zero masked reference mass on every row.
- SGLang is UNREAD by registration.
- The reading re-derives from the committed receipt: `sc1g_reduce.py --dir` reproduces the box's A5 section, every verdict identical and every number to 1e-12 relative (pinned
  by `test_the_a5_reading_rederives_from_its_committed_receipt`).

**The rows.** Full-vocabulary KL in nats against box R's reference, per scored position, mean over 2,048 positions.

| engine | conv1 | conv2 | conv3 | conv4 | wikitext (control, never graded) |
|---|---|---|---|---|---|
| e4b MXFP4 served | 0.0152 | 0.0430 | 0.0060 | 0.0090 | 0.0673 |
| e4b NF4 served | 0.1645 | 0.2004 | 0.0240 | 0.0226 | 0.1134 |
| vLLM served | 0.0250 | 0.0085 | 0.0018 (within F) | 0.0028 | 0.0334 |
| llama.cpp decode, default | 0.0881 | 0.0961 | 0.0082 | 0.0171 | 0.3008 |
| llama.cpp decode, q8 | 0.0552 | 0.0321 | 0.0057 | 0.0190 | 0.1163 |
| floor F (box R) | 0.0057 | 0.0070 | 0.0018 | 0.0019 | 0.0218 |
| R's NF4 fake-quant (full KL) | 0.1080 | 0.1394 | 0.0212 | 0.0227 | 0.0989 |

### The predictions, by the registered rules (`sc1g_reduce.py`'s `a5`)

**K-A: REFUTED.** NF4 is above 3× MXFP4 on three windows but not the fourth, and the prediction requires every counted window:

| window | NF4 / MXFP4 |
|---|---|
| conv1 | 10.8× |
| conv2 | 4.7× |
| conv3 | 4.0× |
| conv4 | **2.5×** |

No bound was used, because every MXFP4 row is resolved. The basis was P44's 11.6× on 200 short prompts. On these
2,560-token chat windows, e4b's served MXFP4 KL is itself 0.006–0.043, which is 3–23× P44's 0.0019 for MXFP4. That
compresses the ratio.

**L1: HOLDS.** e4b MXFP4 pooled **0.0224** is **1.85×** the best comparator, vLLM at 0.0121, against the bar of 2×.
- Counted windows: conv1, conv2 and conv4. conv3 is dropped, because vLLM, the best comparator there, is within F.
- Per window, e4b is the closer engine on conv1 (0.0152 against vLLM's 0.0250). On conv2 it is 5.1× vLLM (0.0430 against
  0.0085), and on conv4 3.2× (0.0090 against 0.0028).

**L2: HOLDS.** Every native-MXFP4 engine's pooled KL is below R's NF4-requant pooled KL, over the windows where both are resolved:

| engine | pooled KL | NF4 scale | windows |
|---|---|---|---|
| e4b MXFP4 | 0.0183 | 0.0728 | conv1–conv4 |
| vLLM | 0.0121 | 0.0900 | conv1, conv2, conv4 |
| llama.cpp default | 0.0524 | 0.0728 | conv1–conv4 |
| llama.cpp q8 | 0.0280 | 0.0728 | conv1–conv4 |

### Descriptive, not graded

**vLLM's served output is not deterministic run to run, even on one host.** On conv1 on Vast 145701, vLLM read:

| run | KL | NLL |
|---|---|---|
| the proof `-a5-8` | 0.0332 | 0.8317 |
| the reading | 0.0250 | 0.8575 |
| `-a5-6`, another host | n/a | 0.8520 |

Both of these runs relabelled the same 127 of 2,047 rows as prefill-shaped. e4b's and llama.cpp q8's conv1 rows are
**bit-identical** between the proof and the reading.

L1's margin is 0.0027 on one window, and vLLM's conv1 alone moved by 0.0082 between two runs. **L1 holds on the registered
draw, but a single vLLM draw cannot carry a sub-0.003 margin.** Any reading of vLLM against e4b at that resolution needs
repeated draws.

**NLL flattery is confirmed by the KL.** On conv1 and conv2, e4b NF4 reads the lowest NLL of any engine (0.736 and 1.650,
against the reference's 0.884 and 1.821), and the largest KL (0.164 and 0.200). This is A3's hypothesis, which A4's read
supported from the reference side; it now has its grading.
- On conv3 and conv4, e4b NF4 served sits at R's own NF4 fake-quant cost: 0.0240 against 0.0212, and 0.0226 against 0.0227.
- On conv1 and conv2 it is about 1.5× that cost.

**e4b MXFP4's conv2 departure is real.** Its KL there, 0.043, is 6× its own floor and 5× vLLM's. This is where A4's
unexplained −0.074 NLL lives. **Open:** it is the one window where e4b's MXFP4 served path is far from the reference while
vLLM is close.

**llama.cpp's q8 activations** halve the default's KL on conv1 and conv2. They do not do so on conv4.

### Cost

| item | $ |
|---|---|
| this reading | 1.557 |
| box I's proofs (eight attempts) | 1.985 |
| box R under A5 | 0.826 |
| **the lane** | **8.850** |

## Amendment A6 (2026-10-06): box J under A5's instrument, the prompt route graded within the box by the median

Registered before any A6 data. The maintainer reviewed the design and required six changes, all of which are in this text:
- comparisons within the box;
- a statement about the median's floor;
- three-way outcomes;
- a bar for P2;
- consequences registered before data;
- a proof of the new wiring.

**A5 is unchanged and was graded as registered** (K-A REFUTED, L1 HOLDS, L2 HOLDS).

### Why

The observations below were **found after A5's data, at $0, from the committed per-position records** (`kl_*.npz` of
`sc1g-5090-a5-1` and `sc1g-prove-a5-8`). They motivate A6; fresh data tests them.
- **A5's mean is tail-dominated.** The top 1% of positions carry 38–59% of each window's KL, and medians sit 10–40× below
  the means.
- **At the median, e4b MXFP4 served is 4–6× further from the reference than vLLM on all four windows.**

  | window | e4b median | vLLM median |
  |---|---|---|
  | conv1 | 0.00041 | 0.00007 |
  | conv2 | 0.0038 | 0.00097 |
  | conv3 | 0.00094 | 0.00015 |
  | conv4 | 0.0019 | 0.00052 |

  e4b is further at 81–90% of positions. That includes conv1, where A5's mean ranks e4b first: vLLM's conv1 mean is a
  first-quarter burst, which is also where its run-to-run change lives.
- **conv2's e4b excess is front-loaded**, at the median as well as the mean: the first half's median against the second
  half's is 0.00721 against 0.00223, so 3.24×. It shrinks with context, which argues against fp8-KV accumulation.
- **The mechanism this tests.** e4b's served stack keeps NF4 for large row counts (`E4B_INT4_KEEP_NF4=1`). The route
  record shows the 512-token prompt's MoE on `nf4_mtile_host|gt256` and only the decode on `mxfp4_gemv|le256`. vLLM and
  box R both process the prompt with the checkpoint's MXFP4 weights. So e4b's prompt KV carries NF4's requantisation
  error, which is 2.5–11× MXFP4's in A5's rows, and every decode step attends to it.
- **Not the attention.** The arm log loads attention in bf16, and there is no attention-INT4 line.

### The statistic, and what it cannot see

- **Primary:** the per-window **median** of per-position full KL (`kl_full_support`, the same estimator as A5).
- **Descriptive:** the mean.
- **No floor for the median.** Box R kept per-window **means** only, so there is no per-position floor and the median has
  none. R's mean floors (0.0057, 0.0070, 0.0018, 0.0019) sit **above** e4b's medians (4e-4 to 4e-3) and vLLM's (7e-5 to
  1e-3), so these medians may lie near the reference's own arithmetic noise. Consequences:
  - **"Resolved" for A6 means distinguishable within the box.** The noise that counts is e4b's own run-to-run, measured by
    (b') below.
  - **P1's ratio stays meaningful**, because 0.5 is reachable even if (a) bottoms out at the noise. But P1 **cannot tell
    an (a) at noise level from an (a) at vLLM's level**.
  - **The "within 2× of vLLM's A5 median" bar may compare against a noise-level number.** It is descriptive, labelled as
    such, and so is the exploratory "4–6×" above.
  - **A real median floor would take a box R re-run** that keeps per-position floor arrays. Not registered here.

### Box J under A6 (`sc1g_box_i.sh`: `box_j` with `i_arms_a6`; A3's box J kept as `box_j_a3`)

- **The box.** e4b only: no comparator, no GGUF. RTX 5090 class at the declared $0.75/h, guard **1.0 h**.
- **Inputs.** Box R's registered rows (`sc1g-r5-2`), staged from the controller copy and re-hashed per arm (`i_ref_full`).
  Four windows (conv1–conv4).
- **Engine path.** The served rows go through `sc1g_k8.py`'s full capture, the same path box I's proof proved.

**The arms.** Each runs in **its own process**, so no compiled or KV state carries over. The order is: conv1's (b), (a),
(b'), then (b) + (a) on conv2–conv4, then (c). The deadline drops from the end.

| arm | stem | stack | engagement rule (otherwise the row is not read) |
|---|---|---|---|
| (b) baseline served | `e4b_serve_served_<src>` | `SC1G_E4B_SERVE` | A5's served gate |
| (a) the prompt on MXFP4 too | `e4b_a6mx_served_<src>` | `SC1G_E4B_MXPRE`: the served stack with `E4B_INT4_KEEP_NF4=0` and nothing else changed | **zero `nf4_*` routes**; `mxfp4_gemv\|le256` ≥ steps × 24; at least one `mxfp4_*\|gt256` (the prompt on the MXFP4 store) |
| (b') determinism repeat | `e4b_a6rep_served_conv1` | `SC1G_E4B_SERVE` | A5's served gate |
| (c) bf16 activations on the decode | `e4b_a6g0_served_<src>` | `SC1G_E4B_SERVE E4B_MXFP4_GEMV=0` | **no `mxfp4_gemv` route**; `mxfp4_grouped_v1\|le256` ≥ steps × 24; the prompt on `nf4_mtile_host` |

**Within-box determinism.** (b') against (b) on conv1 reads either BIT_IDENTICAL, or DIFFERS with the median's relative
change. If the median moved by **≥ 10%** (`A6_REP_NOISE_MAX`), or the repeat is missing, P1 and P3 read **UNREAD**: noise
of that size is comparable to the bars.

**The proof of the new wiring.** The maintainer's review called box J's A5 helpers new wiring on a different box. Box J's
guard is ≤ 1 h, so the proof is local and costs $0:
- `tests/test_sc1g_a6.py` sources the real box script, stubs only the engine calls, and drives `i_arms_a6` and `box_j`.
  It checks 13 arms in the registered order, each its own call carrying R's re-hashed row file, its sha and its own KL
  record, with KEEP_NF4 and GEMV set only where registered. It also checks that every arm is refused when R's rows are
  absent, and that the A6 reading runs last.
- `sc1g_reduce.py --self-test` covers A6's reading on synthetic rows: HELD, FALSIFIED and PARTIAL on each prediction, the
  noisy-repeat UNREAD, and every engagement gate's refusal (45 cases).

### Predictions (registered; `sc1g_reduce.py`'s `a6`; every ratio is within the box, against (b))

| # | prediction | outcomes |
|---|---|---|
| P1 | The NF4-prefilled prompt carries most of e4b's excess at the median: median(a) / median(b) | **HELD** if ≤ **0.5** on ≥ 3 of 4 windows; **FALSIFIED** if ≥ **0.9** on ≥ 3 of 4; **PARTIAL** otherwise. The 0.5 is a **guess, flagged as such**: the exploratory numbers say that if the prompt route were the whole story, (a) would fall to about 0.2–0.25× of (b). |
| P2 | (a) removes conv2's front-loading: (a)'s conv2 first-half / second-half median ratio | **HELD** if ≤ **1.5**; **FALSIFIED** if ≥ 0.9 × (b)'s own ratio on this box; **PARTIAL** otherwise. UNREAD if (b)'s own ratio on this box is ≤ 1.5 (nothing to remove). |
| P3 | The decode GEMV's int8 activations contribute little at the median: median(c) / median(b) | **HELD** if ≥ **0.9** on ≥ 3 of 4; **FALSIFIED** if ≤ **0.5** on ≥ 3 of 4; **PARTIAL** otherwise. Basis: A3, where the int8 activations contributed little in NLL terms. |

**Descriptive, not graded:**
- median(a) against 2× vLLM's A5 median per window. The pinned values are 6.56e-05, 9.70e-04, 1.46e-04 and 5.18e-04,
  recomputed from the committed npz by a test. This is subject to the noise caveat above.
- the means.
- **(b) against A5's reading**, as an integrity check: bit-identical, or the max |diff| per window. Bit-identity held on
  one host, 145701; another host's driver or cuBLAS heuristics can change bits.

### Consequences (registered before data)

- **P1 HELD: no change to the KEEP_NF4 default on fidelity alone.** KEEP_NF4=1 exists for prefill speed. HELD licenses a
  registered serve A/B on a rented card (not the A2000, which is correctness only) that prices KEEP_NF4=0's prefill cost.
  The default moves only on that A/B.
- **P1 PARTIAL:** recorded, with no default change. The remainder goes to the decode path, and P3 decides the next
  registration.
- **P1 FALSIFIED:** the prompt route is cleared. The open conv2 and median gap moves to the decode path, and P3 decides
  the next registration.
- **P3 FALSIFIED:** the GEMV's int8 activations become the lead suspect, and a kernel-level registration on activation
  precision follows.
- **P3 HELD:** the activations are cleared at the median.
- **P2:** informs where any remaining excess sits. It has no default consequence of its own.

### Cost and order

1. This registration.
2. Box J under A6: one run, from this registration's merge, after the pre-launch checks and the store probe. About $1,
   inside the no-ask tier.
3. The reading.

The lane is at **$8.850**.

**Not registered here:**
- Repeated vLLM draws per window, with the draw spread as vLLM's own floor. That belongs to a later box I amendment, only
  if L1 needs resolving.
- A box R re-run that keeps per-position floor arrays.

## A6 box J reading (2026-10-06): P1 PARTIAL, P2 FALSIFIED, P3 UNREAD

**The run.** `sc1g-diag-a6-1` (adertha-receipts `04e7678f`) ran on a Vast RTX 5090, machine 152440, from `1825cf02`
(#1254). It cost **$0.532**, teardown is proven, and the reading re-derives from the committed receipt (pinned by
`test_the_a6_reading_rederives_from_its_committed_receipt`).
- **Arms run:** (b) on all four windows, (a) on conv1–conv3, and (b') on conv1.
- **Dropped by the deadline:** (a) on conv4, and all four (c) arms.
- **Why:** the 1.0 h guard's setup took about 24 minutes. That is the instance boot, an **8-minute staging of the 4 GB of
  reference rows**, and Phase 0. Our 45–50 minute estimate missed it.

**Determinism and integrity.**
- (b') is **BIT_IDENTICAL** to (b).
- (b) is **bit-identical to A5's reading** on all four windows (max |diff| 0.0), on a different host (152440 against
  145701). e4b's served rows reproduce across these two hosts.

### The predictions, by the registered rules

**P1: PARTIAL.** median(a) / median(b) is **0.70** on conv1, **0.76** on conv2 and **0.58** on conv3; conv4 is unread.
No window is ≤ 0.5 and none is ≥ 0.9.
- The NF4-prefilled prompt carries part of e4b's median excess, about a quarter to two fifths, but not most of it.
- (a)'s medians (2.9e-4, 2.8e-3, 5.5e-4) still sit about 3–4× above vLLM's A5 medians. None is within the 2× descriptive
  bar, which is itself subject to the noise caveat.

**P2: FALSIFIED.** (a)'s conv2 first-half / second-half median ratio is **5.79**, against (b)'s 3.24 on this box.
Moving the prompt onto MXFP4 *increased* conv2's front-loading. So the front-loading is not the NF4 prompt route.

**P3: UNREAD.** No (c) arm ran.

**Registered consequences:**
- P1 PARTIAL means no change to the KEEP_NF4 default, and no serve A/B is licensed.
- The remainder goes to the decode path, and P3 decides the next registration. **P3 is unread**, so the decode path is
  untested.

### Descriptive, not graded: the MXFP4 prompt route on conv2

The means by quarter of the window:

| window | arm | Q1 | Q2 | Q3 | Q4 | whole window |
|---|---|---|---|---|---|---|
| conv2 | (b) | 0.0731 | 0.0570 | 0.0212 | 0.0209 | 0.0430 |
| conv2 | (a) | **0.2517** | 0.0680 | 0.0224 | 0.0109 | **0.0882** |
| conv1 | (b) | | | | | 0.0152 |
| conv1 | (a) | | | | | 0.0109 |
| conv3 | (b) | | | | | 0.0060 |
| conv3 | (a) | | | | | 0.0044 |

- With the prompt on MXFP4, the mean KL falls on conv1 and conv3.
- On conv2 it **doubles**, entirely in the first quarter of decode. The later quarters improve.
- (a)'s conv2 NLL rises from 1.747 to **1.973**, against the reference's 1.821.

So on conv2, the decode steps nearest the prompt depart *further* from the reference when the prompt's MoE runs on the
MXFP4 store's large-row kernel (`mxfp4_grouped_v1|gt256`) than when it runs on the kept NF4 (`nf4_mtile_host`).

**This is a lead on e4b's MXFP4 large-row prefill path, on one window. It is not graded here.** The default stays
KEEP_NF4=1, which on conv2 is the closer of the two.

### Cost and next

- This run: $0.532. **The lane is at $9.382.**
- **Next** (separate registrations, not here):
  1. P3 still needs reading, per A6's registered consequence. That means (b) and (c) on conv1–conv4 within one box. At
     about 2.1 min per arm, that fits a 1.0 h box J.
  2. The conv2 lead is a correctness question about the MXFP4 grouped v1 kernel at more than 256 rows. It is a candidate
     for a $0 kernel check against a dequantise-then-matmul reference before any rented run.

## Out of scope

- Distance to bf16 (P44, P90).
- K21, the captured NF4 path and serve_paged's paged prefill attention (serve_paged under load).
- Speed (SC2g).
- Long contexts.
- gpt-oss-120b.
- Bare-text (non-chat) windows.
- Why the chunk-free full forward reads +0.075 on `conv1` (A3: not the NF4 M-tile; the router-flip floor and transformers' full-forward
  attention remain candidates).
