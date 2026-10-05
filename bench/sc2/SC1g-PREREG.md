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

## Out of scope

- Distance to bf16 (P44, P90).
- K21, the captured NF4 path and serve_paged's paged prefill attention (serve_paged under load).
- Speed (SC2g).
- Long contexts.
- gpt-oss-120b.
- Bare-text (non-chat) windows.
