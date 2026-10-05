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

## Out of scope

- Distance to bf16 (P44, P90).
- K21, the captured NF4 path and serve_paged's paged prefill attention (serve_paged under load).
- Speed (SC2g).
- Long contexts.
- gpt-oss-120b.
- Bare-text (non-chat) windows.
