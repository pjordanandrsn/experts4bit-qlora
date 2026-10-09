# SC5 — same-box serving head-to-head on one RTX 5090: `serve_paged` against current vLLM and SGLang on Qwen3-30B-A3B, at 1, 16 and 64 concurrent requests, with quality against one common bf16 reference — **DRAFT, not registered**

> **Status: DRAFT.** This file is a design draft for experts4bit-qlora#1478 item 4. It registers nothing.
> - It is registered only after the grouped-nf4-gemm 0.45.0 / experts4bit-qlora 0.52.0 release train, so every arm
>   names a pip-installable version.
> - No rental happens before that, and none until the maintainer approves the registration.
> - Numbers marked *(size at registration)* are placeholders.
> - The open questions at the end are decided before registration.

Issue: experts4bit-qlora#1478 (item 4), the serving campaign #846.

**Lineage:**
- P58 (vLLM 0.30.0 ahead 1.087× at B=1 and 1.396× at B=16, engine-level, `bench/p58/`).
- SC1 (engine-level map, `bench/sc1/`).
- SC2 (request-level, open-loop, fixed engine order, `bench/h2h-2026-10-02/sc2/`: vLLM 8 req/s, SGLang 8, e4b 1).
- SC2c and SC2e (e4b only: 4, then 12 req/s at 64 slots).

Nothing has been read against vLLM or SGLang on the same box since SC2. The names SC3 (capacity-frontier engines) and SC4
(a second card class) are taken, so this lane is SC5.

## The question

On one RTX 5090, with each framework at its current release and its own best 4-bit format of the same base checkpoint:
- **where** does `serve_paged` lead, match or trail vLLM and SGLang;
- **in what:** time to first token, time per output token and throughput, at 1, 16 and 64 concurrent requests;
- **at what memory:** each framework's documented default memory setting, and one setting with matched KV capacity;
- **at what quality:** each measured against one common bf16 reference on the same text?

Every cell is reported, including every cell a competitor wins.

## Arms

| framework | version | weights (repo @ revision) | 4-bit format | KV cache |
|---|---|---|---|---|
| e4b `serve_paged` | experts4bit-qlora 0.52.0 + grouped-nf4-gemm 0.45.0, every serving default (int4 experts and int4 attention; the tile-table, wide-tile and slot/bucket `auto` defaults) | `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` (bf16), packed on the box | int4-b32, 4.50 bpw (`bench/sc1/UPSTREAM-NOTES.md` census) | FP8 paged |
| vLLM | the current release at registration *(pin then)* | `Qwen/Qwen3-30B-A3B-GPTQ-Int4` @ `9b534e4318b7ebc3c961a839f13eb18b1833f441` | GPTQ 4-bit g128 (Marlin kernels) | the framework's default |
| SGLang | the current release at registration *(pin then)* | the same GPTQ checkpoint | GPTQ 4-bit g128 (Marlin kernels) | the framework's default |

**The primary rows** (decided 2026-10-09):
- vLLM and SGLang on the one shared official checkpoint, `Qwen/Qwen3-30B-A3B-GPTQ-Int4` @ `9b534e43`, the 4-bit format
  SC2 served on both;
- e4b at its int4 defaults.

**A framework's native-best row** is added only where that framework serves an AWQ or compressed-tensors W4A16
checkpoint of the same base faster. It is reported apart and says so, with the checkpoint's repo and revision named at
registration.

**NVFP4** (`nvidia/Qwen3-30B-A3B-NVFP4`, W4A4) is a different precision class. It is deferred to a follow-up amendment
with its own quality bar, and is not part of this registration.

Quality is measured for every row, so a format difference shows up in the quality columns rather than hiding behind
speed.

## Load

- **Prompts:** SC2's 64 distinct 512-token wikitext-2 rows (`bench/sc2/sc2_prompts.py`).
- **Requests:** every request is greedy, with `max_tokens` = 256 *(size at registration)*, `ignore_eos`, and streaming.
  A request is VALID only when it returns exactly `max_tokens` tokens with finish reason `length` (SC2's rule).
- **Closed loop (a new driver, `bench/sc5/sc5_driver.py`).** C ∈ {1, 16, 64} workers each send their next request as soon as the last one
  finishes, until N requests have completed. N is **48 at C = 1, 160 at C = 16 and 320 at C = 64** (sized 2026-10-09,
  before any SC5 data). Every p95 then rests on at least 47 counted requests, and C = 64 runs five waves.
  The first C requests of each cell are warm-up, excluded from every statistic.
- **Reported per cell:**
  - TTFT p50 and p95, and TPOT p50 and p95 (SC2's definitions);
  - output throughput: completed output tokens over the cell's steady window, which runs from the first counted
    request's send to the last counted request's end;
  - peak GPU memory, sampled from `nvidia-smi` at 1 Hz;
  - the slots and KV-token capacity the framework reports.
- **Open loop (reported only, never gated; budget permitting):** one Poisson point per framework, near its saturation as
  the closed-loop C = 64 cell shows it, at the default memory setting. It reports SC2's SLO attainment. The closed-loop
  cells are the gated ones.

## Memory settings (two per framework)

| setting | e4b | vLLM | SGLang |
|---|---|---|---|
| **default** | `E4B_PAGED_MAX_SEQS=auto`, `E4B_PAGED_MAX_TOKENS_PER_SEQ` default (4096) | default `--gpu-memory-utilization` (0.9), default context length, `--max-num-seqs 64` | default `--mem-fraction-static`, `--max-running-requests 64` |
| **matched** | `E4B_PAGED_MAX_SEQS=64`, `E4B_PAGED_MAX_TOKENS_PER_SEQ=1024` | `--max-num-seqs 64` with KV capped at **65,536 tokens** through the release's KV-size flag *(verify the flag at registration)* | `--max-running-requests 64 --max-total-tokens 65536` |

- **Matched means 65,536 KV tokens:** 64 slots × 1,024 tokens.
- **Why that size:** it holds every request (512 + 256). It also fits beside the 15.8 GiB GPTQ weights on 32 GiB:
  fp16 KV is 96 KiB a token, so 6.0 GiB, where 64 × 2,048 would need 12.0 GiB.
- **Tokens are matched, not bytes:** e4b's KV is FP8 and the others' default dtype is 16-bit. Bytes are reported.
- **A matched block whose reported capacity differs from 65,536 tokens is VOID,** beyond its block-size rounding, which
  is stated per framework.
- **Default-against-default and matched rows are both reported.**

## Order and noise

- **Each block is one framework at one memory setting:** a cold server start, a warm-up excluded from every statistic,
  then the cells C = 1, 16 and 64 in that order.
- **The quiescence gate:** before every start, no GPU compute process may remain (`gpu_free`), and the card must be idle
  (`sc1_run.sh`'s gate).
- **The ABBA order, per draw and memory setting:**

  | draw | block order |
  |---|---|
  | 1 | e4b, vLLM, SGLang, SGLang, vLLM, e4b |
  | 2 | SGLang, vLLM, e4b, e4b, vLLM, SGLang (the reverse) |

  Each framework has two blocks per draw.
- **The noise bound.** For each framework, cell and metric, the two same-framework blocks in a draw bound the noise:
  `|A1 / A2 − 1|`.
- **The bounds**, sized 2026-10-09 before any SC5 data:
  - **5 % for TPOT p50/p95 and output tok/s;**
  - **10 % for TTFT p50/p95.**
- **The basis.** Between cold-started e4b servers running identical request plans, SC2e's two draws moved:
  - TPOT p50 by at most 1.5 % (at most 0.2 % serially, 0.4–1.5 % in 64-request bursts);
  - output tok/s by at most 1.1 %;
  - TTFT p50 and p90 by at most 4.2 % and 4.9 %.
- **Why not tighter.** A captured runner can also carry a whole-life level offset of 3–3.6 % (P124 attempt 1, P126 both
  attempts). The 5 % bound turns such a block into NOISY rather than a label.
- **What does not apply.** SC2e's Poisson rates moved far more (up to 76 %) because each draw drew different arrival
  times. SC5's closed-loop plans are identical across blocks, so that source does not apply here.
- **Labels, per cell and metric, between e4b and a competitor:**
  - **LEADS** or **TRAILS** only when both blocks of each framework clear the bound in the same direction;
  - **WITHIN NOISE** otherwise.

## Quality: one common bf16 reference

- **Text:** W windows of wikitext-2-raw-v1 test, each 512 prompt tokens plus 128 scored positions, taken exactly as
  `bench/p117/p117_box.windows()` takes them. W = 64 *(size at registration)*. The same token ids go to every framework.
  The windows and their sha256 are committed before the run.
- **The reference (new, `sc5_ref.py`).**
  - It is `Qwen/Qwen3-30B-A3B` @ `ad44e777` in bf16 through transformers, with `device_map="auto"` over the card and the
    host RAM, eager attention and teacher forcing.
  - This follows SC1 box B's oracle (`step_decomp.py --ppl-oracle upstream`). That oracle records only a mean NLL, so the
    new script records **per position** the NLL of the true next token and the argmax id.
  - It needs about 98 GB of host RAM (SC1's floor). It may run on a separate box (decided 2026-10-09): it is computed
    **once**, stored with its sha256 in the lane's receipts, and every draw verifies that hash before scoring. What must
    match is the windows and token ids, not the host.
- **e4b:** offline teacher forcing through `p117_box.paged_pass` on the **served** int4 pack and every current default
  (graphs and buckets as served). It records per-position NLL and argmax.
- **vLLM:** through the running server, with `prompt_logprobs=1` on the completions endpoint, one request per window.
  - How the entries are read is pinned: for each position, NLL comes from the true token's entry and the argmax from the
    rank-1 entry.
  - vLLM returns the true token plus the top-k, so a position has 1 entry when they coincide and 2 otherwise.
  - The `logprobs=-1` trap (V+1 entries) is avoided by never requesting the full vocabulary. A unit test pins both shapes
    on recorded fixtures, as SC1's `full_vocab_cover` pinned the full-vocabulary case.
- **SGLang:** through the running server, with `return_logprob`, `logprob_start_len` 0 and `top_logprobs_num` 1 (input
  token logprobs plus input top-1). The pattern is SC1's `sc1_sglang_nll.py`, and the same unit-test pinning applies.
- **Reported per framework:**
  - the mean NLL delta against the reference, in nats per scored position, with its spread over windows;
  - argmax agreement with the reference, as the share of positions;
  - SC1's comparability labels (CLOSE ≤ 0.0095, COMPARABLE ≤ 0.02 nats).
- **Scoring shapes (decided 2026-10-09).** The cross-framework comparison is **prefill-shaped for all three**: vLLM and
  SGLang through `prompt_logprobs`, and e4b through one offline forward over the same window on its served weights and
  kernels.
  - e4b's decode-shaped row (`p117_box.paged_pass` through its served graphs) is **reported**, labelled as the served
    arithmetic.
  - **Cannot-say:** the competitors' decode-shaped arithmetic is not measured.
- **The ordering floor (proposed).** `sc5_ref.py --chunked 256` scores the same windows through the bf16 cache in chunks.
  Its NLL delta and argmax agreement against the full forward are the floor that every argmax agreement is read against.
  Chunking reorders an MoE's arithmetic and flips a few percent of Qwen3's router choices (6.77 %, METHODOLOGY 13.1), so
  even a correct implementation agrees below 100 %.
- **Quality gates no speed row; it sits beside every one.**

## Provenance

- **e4b and grouped-nf4-gemm** are the release wheels, by version and sha256.
- **vLLM and SGLang** each install into their own venv from a lock written at registration: `pip download` with
  `--require-hashes`, a list of every wheel and its sha256.
  - The release-anchor tooling (`bench/ra/`) covers e4b and grouped-nf4-gemm only, so these locks are new.
  - Each tripwire writes the installed versions of torch, triton, flashinfer and the framework.
- **The image** is pinned by digest *(choose at registration; SC2 used `nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04`)*.
- **The pins carried over from SC2 are `max_model_len` and seeds.** Everything else either runs at the framework's own
  default or is stated here.

## VOID

A block or cell is VOID, and is never read, when any of these holds:
- a server never reaches ready (SC2's readiness checks per framework);
- an installed version differs from the lock;
- a request is invalid;
- a matched block's KV capacity is off;
- the quiescence gate fails;
- the quality pass is incomplete for a framework (its quality columns are VOID; its speed rows stand, labelled).

## Budget *(size at registration)*

- **The reading:** one RTX 5090 on a host with at least 98 GB of RAM (for the reference), on the scale of 3–4 h for
  2 draws × 2 memory settings × 6 blocks, plus the reference and three quality passes.
- **#1478's estimate:** about $6–8 for proof plus reading.
- **The proof:** one draw, one memory setting, C = 1 and 16, and all three quality passes on 8 windows.

## Decided before registration (maintainer, 2026-10-09)

1. **Formats.** The shared GPTQ checkpoint is the primary row for vLLM and SGLang, and e4b runs at its int4 defaults.
   AWQ or W4A16 appears only as a reported native-best row where a framework serves it faster. NVFP4 is deferred to its
   own amendment.
2. **Load.** The closed-loop cells at C = 1/16/64 are gated. At most one Poisson point per framework near saturation is
   reported only.
3. **The reference** may be computed once on a separate box. It is stored with its sha256, and the hash is verified on
   every draw.
4. **Noise.** N and the bound are sized from SC2e's spread and written here before any SC5 data. A cell is WITHIN NOISE
   unless its blocks clear the bound in the same direction.

## Work items (zero rental; each lands as reviewed code before registration)

- **`bench/sc5/sc5_driver.py`:** a closed-loop mode, `closed C N`, with tests. It imports `bench/sc2/sc2_driver.py`'s
  request, streaming and summary code and leaves that file byte-identical, because `bench/sc1/staged.sha256` and RA's
  `bench/ra/source-pins.json` both pin it.
- **`bench/sc5/sc5_box.sh`:** the box. It reuses SC2's install, start, stop and readiness pieces with the pinned versions,
  the two memory settings, the ABBA blocks, the GPU memory sampler and the capacity readout.
- **`bench/sc5/sc5_ref.py`:** the bf16 reference, per-position NLL and argmax.
- **`bench/sc5/sc5_quality.py`:** the vLLM and SGLang server scorers (`prompt_logprobs=1`, `top_logprobs_num=1`), with
  fixture tests for both entry shapes and for the V+1 trap.
- **e4b quality:** a thin wrapper on `p117_box.paged_pass` at served defaults, with per-position NLL and argmax.
- **`bench/sc5/sc5_reduce.py`:** cells, noise bounds, labels and quality, with a self-test whose mutants include a noisy
  pair, a capacity mismatch, an invalid request, a missing quality pass and a V+1 response.
- **The rest:** the staged pins and wiring tests, and a changelog fragment.
