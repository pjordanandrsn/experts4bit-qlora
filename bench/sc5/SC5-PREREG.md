# SC5 — same-box serving head-to-head on one RTX 5090: `serve_paged` against current vLLM and SGLang on Qwen3-30B-A3B, at 1, 16 and 64 concurrent requests, with quality against one common bf16 reference — **registered on merge**

> **Status: registered when this file merges** (experts4bit-qlora#1478 item 4; the serving campaign #846).
> - Every arm names a pip-installable release: grouped-nf4-gemm 0.45.0 and experts4bit-qlora 0.52.0 (the wheels, by
>   sha256), vLLM 0.31.0 and SGLang 0.5.21 (hash-locked closures). Nothing is spent before this merges.
> - **The sequence:** the proof (`SC1_BOX=M SC1_PROVE=1`), then the reference (`SC1_SC5_PHASE=ref`, once), then an
>   amendment that commits the reference under `bench/sc5/ref/` and sets its sha256 in `sc5_box_m.sh`
>   (`SC5_REF_SHA256`), then the reading (`SC1_BOX=M`). The reading never launches before that amendment merges.
> - **e4b's out-of-box serving (pip install, no environment) is NF4, and it is not measured here.** The e4b arm is the
>   documented int4 serving configuration below, and every result says so in one plain line.

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
| e4b `serve_paged`, **int4 (documented serving configuration)** | experts4bit-qlora 0.52.0 + grouped-nf4-gemm 0.45.0, the release wheels by sha256 (`locks/e4b-wheels.lock`); SC1's `SPEEDENV` (`E4B_SERVE_EXP_INT4=1 E4B_SERVE_ATTN_INT4=1 E4B_SERVE_ATTN_INT4_CALIB=0 E4B_CALIB_SOURCE=c4 E4B_FUSE_T1_GLUE=1 E4B_FUSE_T1_GLUE_R2=1 E4B_FUSE_ROUTER_EPI=1`) + `E4B_PAGED_FUSE_QKV=1`; every other knob at its 0.52.0 default (slots and buckets `auto`, tile programs `auto`, bulk KV, the prefill graph) | `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` (bf16), packed on the box | int4-b32, 4.50 bpw (`bench/sc1/UPSTREAM-NOTES.md` census) | FP8 paged |
| vLLM | 0.31.0 (tag commit `db9527a4`), `locks/vllm.lock.txt` | `Qwen/Qwen3-30B-A3B-GPTQ-Int4` @ `9b534e4318b7ebc3c961a839f13eb18b1833f441` | GPTQ 4-bit g128 (Marlin kernels) | the framework's default |
| SGLang | 0.5.21 (tag commit `e00930c5`), `locks/sglang.lock.txt` | the same GPTQ checkpoint | GPTQ 4-bit g128 (Marlin kernels) | the framework's default |

**The primary rows** (decided 2026-10-09):
- vLLM and SGLang on the one shared official checkpoint, `Qwen/Qwen3-30B-A3B-GPTQ-Int4` @ `9b534e43`, the 4-bit format
  SC2 served on both;
- e4b at its documented int4 serving configuration (corrected 2026-10-10: at 0.52.0 `E4B_SERVE_EXP_INT4`,
  `E4B_SERVE_ATTN_INT4` and `E4B_PAGED_FUSE_QKV` all default off, so "its int4 defaults" named no configuration;
  `serve_paged.py:265-266, :409`).

**A framework's native-best row** is added only where that framework serves an AWQ or compressed-tensors W4A16
checkpoint of the same base faster. It is reported apart and says so, with the checkpoint's repo and revision named at
registration.

**NVFP4** (`nvidia/Qwen3-30B-A3B-NVFP4`, W4A4) is a different precision class. It is deferred to a follow-up amendment
with its own quality bar, and is not part of this registration.

Quality is measured for every row, so a format difference shows up in the quality columns rather than hiding behind
speed.

## Load

- **Prompts:** SC2's 64 distinct 512-token wikitext-2 rows (`bench/sc2/sc2_prompts.py`).
- **Requests:** every request is greedy, with `max_tokens` = 256, `ignore_eos`, and streaming.
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
| **matched** | `E4B_PAGED_MAX_SEQS=64`, `E4B_PAGED_MAX_TOKENS_PER_SEQ=1024` | `--max-num-seqs 64 --block-size 16 --num-gpu-blocks-override 4096` (**65,536 tokens**; both flags in v0.31.0's `vllm/engine/arg_utils.py`) | `--max-running-requests 64 --max-total-tokens 65536` (v0.5.21's `arg_groups/fields/schedule.py`) |

- **Matched means 65,536 KV tokens:** 64 slots × 1,024 tokens.
- **Why that size:** it holds every request (512 + 256). It also fits beside the 15.8 GiB GPTQ weights on 32 GiB:
  fp16 KV is 96 KiB a token, so 6.0 GiB, where 64 × 2,048 would need 12.0 GiB.
- **Tokens are matched, not bytes:** e4b's KV is FP8 and the others' default dtype is 16-bit. Bytes are reported.
- **A matched block whose reported capacity differs from 65,536 tokens is VOID,** beyond its block-size rounding, which
  is stated per framework.
- **Default-against-default and matched rows are both reported.**
- **The capacity readout and its rounding:** e4b `/health` (`max_seqs × max_tokens_per_seq`, rounding 0); vLLM
  `/metrics` `vllm:cache_config_info` (`num_gpu_blocks × block_size`, rounding one block); SGLang server info
  `max_total_num_tokens` (rounding its `page_size`).
- **Prefix caching is off for vLLM (`--no-enable-prefix-caching`) and SGLang (`--disable-radix-cache`),** as in SC2.
  The closed-loop plans reuse SC2's 64 prompts across up to 320 requests, and e4b has no prefix cache, so a cache hit
  would be prefill that only the competitors skip. vLLM runs with `--seed 0` and SGLang with `--random-seed 0`.
- SGLang serves the GPTQ checkpoint with `--dtype float16`, the checkpoint's need, as in SC2.

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
  `bench/p117/p117_box.windows()` takes them. **W = 64.** The same token ids go to every framework.
  The windows are committed here: `bench/sc5/sc5_windows_w64.json`, windows sha256
  `5f6e00d8c4f7c01f9b6f165feb85288948fceba94911c2bb2dd21bac52417a86`, dataset `Salesforce/wikitext` @ `b08601e0`,
  tokenizer `Qwen/Qwen3-30B-A3B` @ `ad44e777`.
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
- **The ordering floor.** `sc5_ref.py --chunked 256` scores the same windows through the bf16 cache in chunks.
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
- **The locks** are regenerated from PyPI alone by `bench/sc5/make_locks.sh` (`uv pip compile --generate-hashes` for
  x86_64 manylinux_2_34, CPython 3.12; the e4b wheels' sha256 read from PyPI's JSON). Both competitor closures carry
  torch 2.13.0 (CUDA 13), so the host needs an R580+ driver; the e4b venv keeps SC1's torch 2.8.0 cu128.
- **The image** is `nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04@sha256:0230b7f243483cb15969fa3cc724a9459599604427052fc2a0d4291c7c0647dd`
  (its system Python is 3.12, the locks' interpreter). The proof checks that the provider pulls it by digest.
- **The pins carried over from SC2 are the seeds and prefix caching off.** Everything else, including the context
  length at the default setting, runs at the framework's own default or is stated here.

## VOID

A block or cell is VOID, and is never read, when any of these holds:
- a server never reaches ready (SC2's readiness checks per framework);
- an installed version differs from the lock;
- a request is invalid;
- a matched block's KV capacity is off;
- the quiescence gate fails;
- the quality pass is incomplete for a framework (its quality columns are VOID; its speed rows stand, labelled).

## Budget

Every run is one RTX 5090 at the policy's fixed $0.85/h, on a host with at least 98 GB of RAM (the policy's floor), with
the launcher's default download ceiling ($0.011/GB). Each is a single run under #846's standing tier.

| run | what | guard | ceiling |
|---|---|---|---|
| proof | `SC1_PROVE=1`: every install from its lock, one default-setting block per framework at C = 1 and 16, the capacity readouts, every scorer on 8 windows | 2.5 h | $2.13 + 90 GB × $0.011 = **$3.12** |
| reference | `SC1_SC5_PHASE=ref`: the bf16 reference, full and chunked, over the 64 windows | 1.5 h | $1.28 + 70 GB × $0.011 = **$2.05** |
| reading | 2 draws × 2 memory settings × 6 cold-started blocks, then the quality passes | 5.5 h | $4.68 + 90 GB × $0.011 = **$5.67** |

The lane ceiling is $10.84; the expected spend is $6–8, as #1478 estimated. The deadline drops the second draw's matched
blocks first (they run last).

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

## What the registration carries (all CPU-tested; nothing has run on a GPU)

- **`bench/sc5/sc5_driver.py`:** the closed-loop mode, `run --concurrency C --n N`. It imports
  `bench/sc2/sc2_driver.py`'s request, streaming and summary code and leaves that file byte-identical, because
  `bench/sc1/staged.sha256` and RA's `bench/ra/source-pins.json` both pin it.
- **`bench/sc5/sc5_box_m.sh`, SC1 box M,** sourced after `sc2_box_e.sh` (SC2's box E and SC2e's box L are the precedent):
  - `m_install_e4b`: the two release wheels downloaded and verified by sha256 before pip sees them, e4b's `[train]` extra
    with SC1's transformers 5.16.1 pin, and a tripwire of its own. `sc1_run.sh` runs it in place of the git-SHA install,
    whose tripwire asserts the cuts the other boxes registered.
  - `m_install_competitors`: SC1's installers in their lock mode (`SC1_VLLM_WHEEL=lock`, `SGLANG_LOCK`), which install
    the whole closure with `pip install --require-hashes --no-deps`; every other box's install is unchanged.
  - `m_block`: one cold-started block: the quiescence gate (`gpu_free`, `quiesce`), the server, the capacity readout, a
    4-request warm-up excluded from every statistic, the three cells, a 1 Hz `nvidia-smi` memory sampler, `block.json`.
  - `box_m` (the reading, in the ABBA order above), `box_m_ref` (`SC1_SC5_PHASE=ref`) and `prove_m` (`SC1_PROVE=1`).
- **The wiring** at every site that enumerates box letters: `sc1_run.sh` (the gate, the GNF4 record pin, the default
  routes, BASEPY, the install branch, the sourcing, PROVE_NEEDS, the proof and the dispatch), `sc1_drive.sh` (the gate,
  the staging list, the name-to-source case, the forwarded `SC1_SC5_PHASE`), `make_pin.sh` and a regenerated
  `staged.sha256`, and the tests that pin letters (`test_sc1_run_shape`, `test_sc1_staged_pin`, `test_sc2_box`,
  `test_sc2b_box`, `test_sc2g_box`, `test_sc2d_box`, `test_sc2e_box`, `test_sc1b`, `test_sc1g_box`, `test_sc1g_a6`).
- **Quality:** `sc5_windows.py` and the committed windows; `sc5_ref.py` (full forward and `--chunked 256`);
  `sc5_quality.py` (`score` through a running vLLM or SGLang server with the pinned parsers, `compare` against a
  reference verified by its sha256); `sc5_e4b_quality.py` (`run`: the engine `serve_paged.build_engine` builds from the
  server's own environment, prefill-shaped and decode-shaped).
- **The record and the reading:** `sc5_record.py` assembles the box's files into the record `sc5_reduce.py` reads; the
  reducer's self-test covers a noisy pair, a capacity mismatch, an invalid request, a missing cell, a missing quality
  pass and a V+1 response.
- **Provenance:** `bench/sc5/make_locks.sh` and `bench/sc5/locks/`.

**Cannot-say, stated before any data:**
- the competitors' decode-shaped arithmetic (not measured);
- whether e4b's server replays its first prefill piece through the same kernels as the eager scoring pass. It is
  expected, because the prefill graph replays the captured forward, but this registration does not measure it;
- e4b's NF4 out-of-box serving.
