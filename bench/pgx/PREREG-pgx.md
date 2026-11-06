# PGX: Paging stage 1 — e4b's tiered serving against SGLang Paged Experts and the vLLM Expert Pager at matched expert-VRAM budgets, on one RTX 4090 (descriptive; registered before any run)

Status: REGISTRATION DRAFT (not merged, so nothing is registered yet). Written 2026-11-05 (clock read 05:17Z;
revision 1 at 05:33Z on the maintainer's review: E-hyb places from a measured routing profile, E-hyb-u is a control,
and the attribution gains a placement branch; revision 2 at 05:47Z: E-hyb's expectation is the solver's own placement
on the box's calib.json, and Q7 moves to 0.2; revision 3 at 05:49Z: the CPU is a recorded, scoped condition, and
E-hyb's hit rate is reporting only; revision 4 at 15:00Z, in-tree: Z3 measured the checkpoints' bytes per expert, so
S4's K is 33/67/101 (was 34/68/102 from an inferred ~2.45 MB) and its predictions are recomputed at that K; the FP8
revision is pinned; the c1 and c4 warm passes carry separate prompt sets; revision 5 at 15:34Z: the scored prompt
sets are built from the four sources the maintainer approved, and their shas are registered below; revision 6 at
15:45Z on the registration review: Q3 argues from the profile placement, E-hyb's token invariance is reported and not
ruled, and Q2 excepts S8's c4 cell at 4 GB). The lane number is
claimed at merge;
`PGX` is the lane's working name, not a reservation. Under the 2026-11-04 paging-baselines ruling (B): E-hyb is e4b's
stage-1 entry, and E-pipe is never a matched arm.


## The question

At the same budget of expert-attributable VRAM on one RTX 4090, with the rest of the experts in pinned host RAM, how fast
do three paged MoE servers decode and prefill Qwen3-30B-A3B? The three are SGLang's Paged Experts, the vLLM Expert Pager,
and e4b's `serve_paged` at the solver placement (VRAM-hot experts on the GPU, the rest computed on the host CPU). Each is
measured at concurrency 1 and 4, warm, with one cold and one content-shifted cell. Alongside speed, the run records how
often each server's VRAM tier holds the experts a token routes to.

It is **descriptive**. It reports per-cell numbers and ratios beside their draw spread. It licenses no default, and no
claim beyond the registered cells.

## The arms

| arm | engine (pinned at registration) | checkpoint (pinned revision) | expert format, kernel | how the budget is set |
|---|---|---|---|---|
| **S4** | SGLang 0.5.23 (tag 7df5dd0, commit 297e54fd), Paged Experts | `Qwen/Qwen3-30B-A3B-GPTQ-Int4` @9b534e43 (P235's) | GPTQ int4, Marlin MoE | `--paged-experts-num-resident K` |
| **S8** | SGLang 0.5.23, Paged Experts | `Qwen/Qwen3-30B-A3B-FP8` @d206ba73 (block-128 FP8, `weight_scale_inv`) | FP8, Triton fused_moe | `--paged-experts-num-resident K` |
| **V8** | vLLM 0.29.0 + vllm-expert-pager e9264fda (0.1.1), own venv | the same FP8 snapshot (one copy on disk) | FP8, Triton FP8 MoE | `VLLM_EXPERT_PAGER_CACHE_SLOTS=K`, RAM tier = every expert (its default), no SSD file |
| **E-hyb** | e4b + grouped-nf4-gemm at the pins in force at registration; `python -m experts4bit_qlora.serve_paged` | `Qwen/Qwen3-30B-A3B` bf16 @ad44e777, baked to an NF4 arena on the box | NF4, gnf4 kernels; DRAM experts on the host CPU | `E4B_PAGED_PLACEMENT=solver`, `E4B_PAGED_VRAM_GB` = the budget in GiB, `E4B_PAGED_DRAM_GB=16` (holds every non-VRAM expert, so the NVMe tier is empty), `E4B_PAGED_PROFILE` = the held-out fit profile (below; e4b#2169) |

**The fit profile (E-hyb's routing mass, fixed before any run).** The solver places by a measured routing mass, not
uniform-assumed mass: a uniform-placed static set against two adaptive LRU baselines would mostly measure a placement
handicap (the maintainer's ruling on revision 0). The profile is `bench/pgx/fit_profile.jsonl`, an
`engines/expert_profile` JSONL written at $0 by `pgx_fit_profile.py` from the #1469 decode traces (grouped-nf4-gemm
`bench/cold-engine/routing-trace/qwen3_{code,dialogue,math,prose}.jsonl`, pinned by sha256):
- the same model on e4b's own NF4 path (the census ran `load_moe_4bit_streaming` and the pipelined engine), so routing
  carries E-hyb's numerics;
- the four content kinds, 512 greedy decode steps each; `tokens_routed` = routed picks per (layer, expert) over all
  steps; `layer_id` = the MoE ordinal, which for Qwen3-30B-A3B (every decoder layer MoE) is the trace's layer index and
  the arena id;
- held out: the four capture prompts and their continuations are excluded from every scored prompt set (the kit
  sha-checks the scored sets against them);
- its limits, stated: one prompt per kind, and a decode-only mass.
- Its sha256 is `0a46edd68005b46db6bb2174542ec948df2843071efce2ecb36099fda04c626a`; the traces' sha256 prefixes are
  code `64647ca1`, dialogue `1dc7aea0`, math `b432a869`, prose `1293f390` (in the profile's meta row).
- The receipt records the profile's sha256, and `/health`'s `levers.placement_profile` must name it (kind `measured`)
  on every E-hyb lifetime, or the pass is VOID.

**Controls (reported, never matched):**
- **E-hyb-u** (8 GB only): E-hyb without `E4B_PAGED_PROFILE` (uniform-assumed mass, `serve_paged` as shipped before
  e4b#2169). E-hyb against E-hyb-u is what the measured profile buys.
- **E-allvram-g** (e4b's ceiling): `serve_paged` at `E4B_PAGED_PLACEMENT=all-vram`, graphs `auto`. Every NF4 expert is in VRAM (about 16.3 GB), so it sits over every budget.
- **E-allvram-e**: the same with `E4B_PAGED_GRAPHS=0`. The difference from E-allvram-g isolates what eager decode costs e4b, which E-hyb pays at every budget (graphs refuse at `solver`).
- **E-pipe is not an arm** (ruling B). The pipelined engine keeps its packed base weights for its T>1 fallback, so its VRAM is near all-resident at any K. It needs a lean prefill before it can be matched.

**Arms struck from stage 1:** the zero-residency `--cpu-offload-gb` controls and the static-hot e4b control go to a later stage. The A2000 is struck throughout (ruling 1).

## The matched budget

- **The budget is expert-attributable VRAM in decimal GB: 4, 8 and 12.** K follows per format as
  `K = floor(budget / (48 × bytes_per_expert))`. The registered table is computed from the checkpoints' safetensors
  headers (Z3, `bench/pgx/pgx_bytes.py`, committed as `bench/pgx/bytes_per_expert.json`) and, for NF4, P233's
  placement record. The box measures each engine's loaded slot bytes before the first timed cell; where that gives a
  different K, the measured K replaces the table's, recorded in the receipt.

| budget | FP8 K (S8, V8) | int4 K (S4) | NF4 (E-hyb) |
|---|---|---|---|
| 4 GB | 17 | 33 | 31 experts' worth; `E4B_PAGED_VRAM_GB=3.725` |
| 8 GB | 35 | 67 | 62; `7.451` |
| 12 GB | 52 | 101 | 94; `11.176` |

  Bytes per expert: FP8 4,719,744 (three 768×2048 E4M3 projections and their 128-block scales); GPTQ int4 g128
  2,470,912 (qweight, qzeros, scales, g_idx), or 2,451,456 without the g_idx, which is trivial at `desc_act: false`
  (both give the same K at every budget); NF4 2,654,208 (P233). Every checkpoint has 48 × 128 experts of one size.
- **The solver reads GiB** (`vram_gb * 2**30`), so E-hyb's knob is the decimal budget converted. The solver may place
  fewer than the budget's bytes; the placement proof (below) records what it placed, and an E-hyb cell whose VRAM-tier
  expert bytes exceed the decimal budget is VOID.
- **Each arm's non-slot VRAM is reported, not charged against the budget.** That covers the pager's all-E scales and
  its W-row working slab, SGLang's buffers, and e4b's non-expert weights. Peak VRAM total and the KV blocks available
  are reported per cell. The budget matches expert capacity, not total memory.
- **SGLang's graph limit is a measured output.** SGLang captures decode only up to `K // top_k` sequences: 2 at FP8
  K=17, so S8's c4 cell at 4 GB decodes eagerly. When `topk_ids.numel() > K` (S8 c4 at 4 GB: 32 > 17), it plans on the
  host in waves. Graph coverage, the share of decode steps replayed, is recorded per cell for every arm.

## Workload

- **Prompts:** fixed token ids, identical in every engine, under P235's `prompts_sha` contract (each engine's driver
  sends token-id lists; the receipt carries the sha of the list per pass). Prompt length 1,024 tokens (p1024),
  tokenized once with the Qwen3-30B-A3B tokenizer (byte-identical `tokenizer.json`, sha256 `aeb13307…`, in all three
  checkpoints). Content kinds as locality-1469: code, dialogue, math, prose. Three sets, each disjoint from the
  four #1469 capture prompts and from the other two:
  - **warm_c1** (P1, P2): 8 requests, 4 code and 4 dialogue;
  - **warm_c4** (P3): 16 requests, 8 code and 8 dialogue;
  - **shift** (P4): 8 requests, 4 math and 4 prose.

  The sets are built by `bench/pgx/pgx_prompts.py` and committed under `bench/pgx/prompts/` with a manifest of every
  source file's sha256. Each kind's corpus is tokenized once (no special tokens), and window i is tokens
  [8,192 i, 8,192 i + 1,024). Every kind is rendered as raw text, with no chat template, so every engine is sent the
  same ids. The kit test asserts that no window contains a line or sentence of the four #1469 capture prompts
  (`bench/pgx/capture_prompts.json`), and that no window is in two sets.

  | kind | source (pinned) | rendering | licence |
  |---|---|---|---|
  | prose | `Salesforce/wikitext` @b08601e0, `wikitext-2-raw-v1` test | non-empty rows joined by blank lines (SC2's) | CC-BY-SA-3.0 / GFDL |
  | code | `python/cpython` @8183fa5e (tag v3.13.9): `Lib/{argparse, dataclasses, functools, inspect, typing, collections/__init__, pathlib/_local, enum, textwrap, difflib}.py` | the files joined by blank lines | PSF-2.0 |
  | math | `EleutherAI/hendrycks_math` @21a56338, test, the seven subjects in sorted order | `Problem: …` / `Solution: …` per row | MIT |
  | dialogue | `HuggingFaceH4/ultrachat_200k` @8049631c, `test_sft` | `User: …` / `Assistant: …` lines per conversation | MIT |

  | set | requests | `prompts_sha` |
  |---|---|---|
  | warm_c1 | 8 | `d45648005719bb7ff6979e814f0c9cde5824e85d12b60f5d1cd0e107b838f554` |
  | warm_c4 | 16 | `37589f7c124653b62f321a6641d76ab5162840d25e618321f574025704b11b12` |
  | shift | 8 | `c4477b9ad77d5c02a2a30eade3ea1021e04e21299f9c43ce4bed31717403a295` |

  Training-data overlap with these public texts does not matter here: the reading is routing and paging speed, not
  quality.
- **Output:** 256 tokens per request, greedy, `ignore_eos` and `min_tokens = max_tokens` (P235/SC2), temperature 0.
- **Concurrency:** closed loop at c = 1 (8 requests per pass) and c = 4 (16 requests per pass, 4 waves).
- **One server lifetime per (arm, budget, draw)**, passes in this order:
  Controls run one lifetime per draw: E-hyb-u at 8 GB (all four passes), E-allvram-g and E-allvram-e (P1-P3).
  1. **P1 (cold)**: c1 over the warm set on a fresh server. Recorded; READ as the arm's cold cell at 8 GB only.
  2. **P2 (warm, c1)**: the same requests again.
  3. **P3 (warm, c4)**: the warm set at c4.
  4. **P4 (shifted, at 8 GB only)**: c1 over the shift set, on the cache P1-P3 warmed.
- **Draws:** 3 fresh-process lifetimes per (arm, budget). Arm order within a draw is rotated (draw d starts at arm d
  of S4, S8, V8, E-hyb), so no arm always runs first on a cold host page cache. Budgets ascend within an arm.

## Measures per pass

- **Speed:** output tok/s, TTFT p50/p99, ITL p50/p99, all from the driver's own clocks (SSE token arrival), the same
  driver for every engine.
- **Residency:**
  - S4/S8: VRAM hit rate and page-ins per token, computed exactly from the per-layer per-step `topk_ids` dumped by the
    counter shim, replayed through SGLang's own `LRUPolicy` and `plan_waves` (PE/residency.py at 297e54fd). On the
    shakedown, check the hit rate against a hook counting the device path's `count` writes.
  - V8: the pager's device hit/miss counters per tier, dumped at the end of each pass (they log only on eager calls).
    Bytes = misses × row bytes.
  - E-hyb: the VRAM-tier hit rate and the DRAM-tier rows computed on the CPU per token, from `E4B_PAGED_STEP_TRACE`
    and the tier counters. **E-hyb pages nothing in**: a DRAM expert is computed in place.
- **Graph coverage** per pass; **peak VRAM** and **KV blocks**; **host CPU utilization** (1 s samples); routed-expert
  agreement against S8's routing on the same prompts (routing differs across formats, so a hit-rate gap can partly be
  routing).
- **Provenance:** versions, revisions, `prompts_sha`, the measured bytes per expert and K, the card (name, cc, SMs,
  power limit, clocks from a 20 s `card_clock` burn, recorded and not ruled), CPU model and cores, min(MemTotal,
  cgroup) RAM.

## Validity and the rule (`bench/pgx/pgx_reduce.py`, self-tested before data)

**VOID a pass** when any of these holds (each check fails closed; a missing record is a fault):
- a version, revision, card class or `prompts_sha` mismatch;
- an arm's K, or E-hyb's placed VRAM-tier bytes, differ from the registered table as the box measured it;
- fewer than the registered requests completed, or any request returned an error or a short output;
- a missing counter dump (S: no topk dump; V8: no counter dump; E-hyb: no step trace or placement proof);
- V8: any pager give-up during the pass (gather.py:44-47, "one token's output … wrong");
- E-hyb: the placement proof (per-layer tier byte and expert counts read after the build) is missing, or any NVMe-tier
  expert is non-zero;
- E-hyb: `/health`'s `levers.placement_profile` is not kind `measured` with the fit profile's sha256 (E-hyb-u: not
  kind `uniform-assumed`).

**Token invariance (reported per arm; it rules only where stated):**
- P2 against P1 within one lifetime: same prompts, different cache state, so the greedy tokens must be identical for
  S8 and V8 and the two all-VRAM controls (GPU-only, deterministic kernels). A difference VOIDs that lifetime's passes
  and is reported as a paging finding.
- E-hyb and E-hyb-u P2 against P1 is REPORTED, never ruled. The rule exists to show that paging does not change
  outputs, and they page nothing: the placement is static, and the DRAM tier is multithreaded host arithmetic, so a
  difference there would be CPU nondeterminism, not paging.
- Across budgets within an arm: S8 and V8 should be token-identical (Triton). A difference is reported as a finding.
  E-hyb across budgets is NOT expected to be bitwise (the DRAM tier's host arithmetic differs; P233 saw 7/10 prompts
  match), so it is reported, never ruled.
- S4 (Marlin) is reported, never ruled: `use_atomic_add = (fp16 or cc >= 9) and not mxfp4`, and the GPTQ checkpoint
  runs in fp16, so atomics are likely on at cc 8.9 (the box records which). Token-order nondeterminism (#48150) also
  applies.

**Stability:** a cell's three draws must have max/min ≤ **1.03** on its median ITL and on its output tok/s (P235's
self-pair rule, applied to the three). Otherwise the cell reads UNSTABLE, is reported, and stays out of every headline
ratio.

**The box's verdict:** READ when every registered pass READs or is UNSTABLE. A VOID pass is reported with its faults and
never dropped to rescue the rest.

## Readings

- **The headline:** per budget and concurrency (warm), E-hyb's output tok/s and median ITL against each of S4, S8 and
  V8, and the best of the three, with TTFT p99, peak VRAM and host CPU beside them. Every E-hyb row also carries the
  placement preview's VRAM share for that budget, from the box's calib.json. Losses and wins are reported together.
  ITL and tok/s are the decisive measures for every arm.
- **Same-format check:** S8 against V8 (FP8 against FP8). It is the instrument sanity row and needs no e4b.
- **Residency:** each arm's hit rate per cell against the predictions below, and S8/V8 against each other (both are
  LRU at the same K). **For E-hyb (and E-hyb-u) the hit rate is REPORTING only, never a comparator:** the solver trades
  VRAM hits for CPU-tier compute by design, so a lower hit rate is not a worse placement.
- **Cold and shifted:** per arm at 8 GB, P1 and P4 against P2.
- **Attribution of an E-hyb loss** (from the step trace and the controls, stated now):
  - **The capture-legal hybrid tier** is named when E-hyb's ITL gap is mostly the eager tax. That tax is read as
    E-allvram-e minus E-allvram-g, and the rule is that it is ≥ 50 % of E-hyb's ITL gap to the best baseline at that
    cell.
  - **The lean-prefill / GPU-paging path** is named when the gap is mostly DRAM-tier host compute, read from the step
    trace's host-tier time, by the same ≥ 50 % rule.
  - **Placement quality** is named when E-hyb-u's ITL minus E-hyb's at 8 GB is ≥ 50 % of E-hyb-u's gap to the best
    baseline: the measured profile closed most of what uniform placement lost, so the remaining placement lever is a
    better profile (broader, or fitted per workload), not an engine build.
  - Otherwise the branches that pass are named, each with its share stated.
  - This is the "a loss names the engine work" consequence. It names work; it does not size it.

## Predictions (written before any stage-1 data; expectations, not sizing)

Decode hit rates from the per-layer replay (e4b `bench/train-routing-trace/replay.py --per-layer-k` at beaa5018) on
the #1469 Qwen3-30B-A3B decode traces. Those are B=1 decode on the traces' own prompts, scored on steps 128-511. They
predict the c1 warm cells only. This run's prompts differ, so a band of ±0.05 is the expectation, not a bar.

| arm (policy) | budget | K | code | dialogue | math | prose |
|---|---|---|---|---|---|---|
| S8 (lru_seeded) / V8 (lru) | 4 GB | 17 | 0.578 | 0.503 | 0.587 | 0.630 |
| | 8 GB | 35 | 0.769 | 0.747 | 0.814 | 0.846 |
| | 12 GB | 52 | 0.878 | 0.929 | 0.930 | 0.945 |
| S4 (lru_seeded) | 4 GB | 33 | 0.752 | 0.718 | 0.793 | 0.828 |
| | 8 GB | 67 | 0.933 | 0.983 | 0.977 | 0.986 |
| | 12 GB | 101 | 0.988 | 0.997 | 0.998 | 0.998 |

**E-hyb is a static placement, so neither LRU nor lfu_protect predicts it.** Its VRAM hit rate is the coverage of
the experts the solver places from the fit profile; E-hyb-u's, of those it places under uniform-assumed mass. Two
proxies, computed on the same traces (scored on steps 128-511):

| E-hyb | 4 GB (K≈31) | 8 GB (K≈62) | 12 GB (K≈94) |
|---|---|---|---|
| index-ordered static set (experts 0..K-1 per layer) | 0.23-0.26 | 0.48-0.50 | 0.74-0.75 |
| a fit-window profile set (top-K over steps < 128) | 0.58-0.77 | 0.84-0.94 | 0.96-0.98 |

**These proxies are NOT E-hyb's expectation (revision 2).** The solver balances GPU against CPU completion time, so it
keeps hot experts on the DRAM tier when the host CPU is fast. Run at $0 on the fit profile (`pgx_place_preview.py`,
the solver exactly as `build_engine` calls it), the VRAM tier's share of the profile's routed mass depends on the
calibration blob's CPU scatter bandwidth:

| CPU scatter (GB/s) | E-hyb 4 / 8 / 12 GB | E-hyb-u 8 GB |
|---|---|---|
| 496.5 (the sc1 5090 host's measured blob, a 96-core EPYC) | 0.523 / 0.709 / 0.760 | 0.493 |
| 100 | 0.576 / 0.813 / 0.921 | 0.493 |
| 60 | 0.582 / 0.825 / 0.939 | 0.493 |
| top-by-mass ceiling (any host) | 0.591 / 0.843 / 0.967 | |

**E-hyb's registered expectation is therefore a method, not a number:** after the shakedown's `calibrate.py` writes the
box's `calib.json`, `pgx_place_preview.py` runs on it and its shares are E-hyb's and E-hyb-u's expected VRAM hit
rates, posted (and receipted) before the reading launches. They are on the profile's own prompts, so on this run's
held-out prompts E-hyb should sit at or below them. A fast-class desktop CPU (dual-channel memory) is expected near
the 60-100 GB/s rows.

**lfu_protect (ColdTier) is not engaged in stage 1:** RAM holds every expert, so E-hyb's NVMe tier is empty. Its
replay predictions belong to the SSD sub-grid. They carry the ascending-order caveat (the traces store routed sets
sorted, so its page-ins above misses depend on that order) and, since grouped-nf4-gemm#590, page-ins = misses.

| # | prediction |
|---|---|
| Q1 | at every budget, S4's hit rate > S8's ≈ V8's (int4's K is about twice FP8's) |
| Q2 | S8 and V8 hit rates agree within 0.01 at every cell except S8's c4 cell at 4 GB, where S8 plans on the host in waves and the pager does not (Q5) (both are LRU at the same K; only tie-breaks differ) |
| Q3 | E-hyb's c1 warm ITL is longer than S4's at 4 GB and 8 GB (by the $0 placement preview over the 60-496 GB/s CPU rows, about 42-48 % of the profile's routed mass is computed on the host CPU tier at 4 GB and 17-29 % at 8 GB, where S4's LRU holds about 0.72-0.83 and 0.93-0.99 of routed experts in VRAM) |
| Q4 | E-hyb's eager tax (E-allvram-e minus E-allvram-g) is a visible share of its ITL at 12 GB (P66: 8.05 against 1.184 ms/token fixed tax) |
| Q5 | S8's c4 cell at 4 GB runs eager and on the host wave path, and its ITL lead over V8 at that cell shrinks or reverses |
| Q6 | P2 against P1 is token-identical for S8, V8 and E-hyb on every lifetime (ruled for S8 and V8; reported for E-hyb) |
| Q7 | E-hyb's VRAM hit rate at 8 GB is at least 0.2 above E-hyb-u's, and its c1 warm ITL is shorter (the $0 preview gives +0.22 to +0.33 over the CPU range above) |

Cross-engine speed order at each cell is NOT predicted, only reported.

## Consequence, registered now

- **READ:** `RESULTS-pgx.md`, and a row in `docs/REPRODUCE.md`: "e4b serve_paged (solver placement) against SGLang
  Paged Experts and the vLLM Expert Pager at matched expert-VRAM budgets, Qwen3-30B-A3B, one RTX 4090". Wins and losses
  appear together, with the attribution above.
- **Next, from the attribution:** stage 2 (the remaining arms, the SSD sub-grid, quality against bf16) or one of the two
  named engine builds. That choice is a new decision; this page registers none.
- No default moves.

**Scope:** one model, one card class, decode at p1024 and c ≤ 4, host RAM holding every expert. E-hyb's placement and
speed depend on the host CPU (its scatter bandwidth sets how much hot mass the solver keeps on the CPU tier). This
reading is for the recorded CPU, and nothing is claimed for another CPU class. Nothing about
prefill-heavy load, larger batches, SSD tiers, other models or other cards. Nothing about quality against bf16 (P233
covers E-hyb's: NLL +0.0021 against all-VRAM).

## The $0 path first (no A2000, nothing rented)

- **Z1, driver and reducer on CPU.** A stub OpenAI-compatible SSE server per engine flavour drives every registered
  pass. The reducer's self-test covers each VOID rule, the stability rule, the token-invariance rules and the
  attribution rule on fixture records, including a fixture written from a REAL shakedown record once one exists (the
  "gates tested only on handwritten fixtures" lesson).
- **Z2, counter shims on CPU.**
  - The SGLang `topk_ids` dump replayed through PE/residency.py's `LRUPolicy` + `plan_waves`, with msgspec stubbed as
    in the per-layer differential, on a synthetic routing sequence with a hand-computed answer, including the wave
    case.
  - The pager's end-of-run counter dump against `table.ExpertTable` on CPU torch.
- **Z3, byte accounting from the safetensors headers** of the three checkpoints (header-only range reads; done:
  `pgx_bytes.py`, `bytes_per_expert.json`, `tests/test_pgx_bytes.py`). It gives the registered K table above.
- **Z4, E-hyb config checks on CPU:** `PagedServeConfig.from_env` at each budget (the graphs/solver refusals, the
  hot-rows floor = 0 with the NVMe tier empty), and the solver run on a stub calibration to check the placement
  proof's fields.
- **Z4b, the placement preview:** `pgx_place_preview.py` and its test (every registered placement within its decimal
  budget with the NVMe tier empty; profile > uniform <= the top-by-mass ceiling; the CPU-speed dependence).
- **Z6, the fit profile:** `pgx_fit_profile.py` writes `fit_profile.jsonl` from the pinned traces, and its test
  checks the row format against `placement.load_routing_mass` and `serve_paged.placement_profile` (48 x 128, positive
  mass in every layer), so the profile E-hyb reads is the one registered.
- **The scored prompt sets** (`pgx_prompts.py`, `prompts/`, `tests/test_pgx_prompts.py`): sizes, kinds, exact
  1,024-token windows, the shas against this page, no window in two sets, and no capture-prompt text.
- **Z5, the kit test** (`tests/test_pgx_prereg.py`): this page's K table, budgets, arms, controls, passes, stability
  rule and checkpoint pins against the kit's constants, and the fit profile's sha against the one named here, so the
  page and the kit cannot drift apart.

## Run ids, order and the stop

- **The shakedown first: `pgx-shake-1`** (retry ids `pgx-shake-r1` to `-r3`). Never read.
  - It covers: the three venvs and e4b install; the fetch; the NF4 bake; grouped-nf4-gemm `bench/calibrate.py` writing
    E-hyb's `calib.json` on this host (E-hyb's solver refuses to guess bandwidths, and a host-specific blob is
    required); the measured bytes per expert and K table; the card-clock burn.
  - Then one lifetime per arm at 8 GB with P1 and P2 at c1; each counter shim on a real record; peak host RSS per arm.
  - It measures the per-lifetime wall per arm, from which the reading's guard is fit.
  - If any arm cannot serve at its budget: STOP and report. That is a finding about the engine, not a retry.
- **Then the reading: `pgx-4090-1`** (retry ids `pgx-4090-r1` to `-r3` for a box that does not start).
  - **The split rule (P238's form):** the reading runs in the fewest K of 1, 2 or 3 boxes whose largest box fits the
    box cap (worst ≤ USD 15, guard ≤ 18 h under the per-run override, or ≤ 6 h without it).
  - **Boxes are whole draws, never split by arm**, so every cross-arm ratio is within one box.
  - **Every box is pinned to box 1's machine.** If it cannot be rented within 24 h, the reading is labelled
    `cross_host`, its stability band is reported as a cross-host band, and its ratios stay within one box.
  - The plan is decided from the shakedown's receipt and stated before the reading launches.
- Two strikes end the lane. Host refusals, before `STARTED`, are not strikes.

## Budget (INFERRED estimates, all to be refit from the shakedown)

**The card and host floors:**
- **Card:** one RTX 4090 on Vast, verified/secure, direct SSH, cc 8.9, ≥ 128 SMs, VRAM ≥ 24,000 MiB.
- **CPU:** a fast-class CPU (`p237_host.vast_cpu_class`), and no narrower bound: a desktop-only or server-only floor on
  top of the others risks an empty market (P237). E-hyb's DRAM tier computes on the host CPU, so its numbers depend on
  the CPU. The CPU is a recorded, scoped condition: the box's `calibrate.py` blob (scatter GB/s, cores, model) and the
  placement preview's shares are receipted before the reading, every arm of a draw shares the CPU, and the K=2 split
  pins both reading boxes to box 1's machine.
- **Host RAM:** listed ≥ 62 GiB and measured min(MemTotal, cgroup) ≥ 60 GiB.
  - Per arm, the need is: S8/V8 pin about 31 GB of FP8 experts; S4 about 17 GB; E-hyb streams the bf16 load plus a
    DRAM tier of up to about 15 GiB (P233's floor for the same load was 60 GiB).
  - Arms run one at a time, so the floor is the largest single arm, measured at the shakedown, and a pre-reading
    amendment moves it if the measurement says so.
  - P237's census found the 4090 fast class almost entirely at 30-62 GiB listed; a 128 GiB floor would empty it.
- **Disk:** ≥ 250 GB, for the three checkpoints (about 61 + 17 + 31 GB), the NF4 arena (16.3 GB) and three venvs.
- **Rate cap:** ≤ USD 0.85/h. P237 saw 4090 offers at USD 0.61-0.80; price was never the binding floor.
- **Downloads:** ≤ USD 0.02/GB, about 120 GB per box.
- **No power floor:** every ratio is within one box. The power limit and clocks are recorded.

**The fit (per box; × 1.25 for the guard):**
- **Setup:** installs 1,800 s; fetch 2,750 s (110 GB at 40 MB/s); bake 500 s; calibrate 300 s. Total 5,350 s.
- **Lifetimes, estimated:**
  - Baseline lifetimes about 700 s each: load plus P1-P3, with P4 at 8 GB.
  - E-hyb about 1,300 s (eager, with the CPU tier); E-hyb-u the same.
  - The two all-VRAM controls about 600 s each.
  - **One draw** = 9 × 700 + 3 × 1,300 + 1,300 (E-hyb-u) + 2 × 600 + 4 × 300 (P4) = 13,900 s.
- **Reserve:** 600 s.

| plan | box guard | box worst (guard × 0.85 + 120 GB × 0.02) |
|---|---|---|
| shakedown: setup + 4 lifetimes × 900 s + reserve | (5,350 + 3,600 + 600) × 1.25 = 11,940 s = 3.3 h | **USD 5.22** |
| reading K = 1 (3 draws, one box) | (5,350 + 41,700 + 600) × 1.25 = 59,562 s = 16.5 h | USD 16.46, over the USD 15 box cap |
| reading K = 2 (draws {1,2} and {3}) | 11.7 h and 6.9 h (both under the per-run override) | **USD 12.36 + USD 8.26 = USD 20.62** |
| reading K = 3 (one draw a box) | 6.9 h each (over the 6 h global cap: the per-run override) | USD 8.26 each, **USD 24.78** |

**At the estimate, the plan is K = 2.**
- **Lane worst:** shakedown + reading + 3 retries at about USD 0.25 each = 5.22 + 20.62 + 0.75 = **USD 26.59**
  (USD 30.75 if the shakedown's fit forces K = 3).
- That is under the USD 35 per-run cap, and each box is under USD 15. Expected spend is about half the worst.
- The daily rail (USD 100) is shared with the other lanes' boxes.

## Launch

- One exact launch commit per box (this registration's merge or a later amendment's), through pod-launch with
  `--validate` passing, the host pinned, rent-watch armed; then the maintainer's Approved naming that commit and the
  kit's hash. The kit's approval reader fails closed.
- The shakedown launches after Z1-Z6 are green, e4b#2169 and the registration have merged, and the pins name a
  release (or commit) that carries `E4B_PAGED_PROFILE`.

## What was seen before this page (stated, not hidden)

- No number from any arm on this card, model and workload.
- The per-layer replay predictions (above) and the static-coverage proxies, computed on the #1469 traces on CPU.
- From source only: SGLang PE's residency, wave and capture rules at 297e54fd; the pager's tiers, counters and give-up
  path at e9264fda; `serve_paged`'s solver path, its graph refusals and its hot-rows floor at e4b main.
- grouped-nf4-gemm#590 (ColdTier two-pass): page-ins = misses on the same traces. Not exercised here.

## What this lane cannot say

- Anything about quality against bf16, SSD tiers, concurrency above 4, prompt lengths other than 1,024, prefill-heavy
  load, other models or other cards.
- Anything about E-pipe or a lean prefill: they are not built.
- Why an engine is slower, beyond the registered attribution rule.
- Anything about a power-capped card's absolute speed: power is recorded, not ruled.

## Receipts

Per box, fetched and committed to `bench/pgx/receipts/<run>/`: every pass's driver record, the counter dumps, the step
traces, the placement proofs, `calib.json`, the bytes-per-expert and K table, `versions.txt`, `logs/` and `SHA256SUMS`.
The machine and instance ids stay in the private receipt store only.

## Open items for the reviewer

1. **The workload is two concurrencies and one prompt length (stage-1 scope as directed).** The design's c16 and
   p4096 cells are left for stage 2.
2. **RESOLVED (revision 4):** the FP8 checkpoint is `Qwen/Qwen3-30B-A3B-FP8` @d206ba73, and S8 and V8 share one
   snapshot on disk.
3. **The RAM floor at 62 GiB listed is INFERRED;** the shakedown measures it. If E-hyb's bf16 streaming load needs more,
   a fast-class 4090 at that floor may not exist (P237), and the fallback is a DRAM tier sized to fit, stated before the
   reading.
4. **RESOLVED (revision 5):** the scored prompt sets come from the four pinned public sources in the Workload table,
   approved by the maintainer; their shas are registered there.
5. **RESOLVED (revision 1):** E-hyb places from the held-out fit profile (e4b#2169); uniform placement is the E-hyb-u
   control. Open within it: the fit is one prompt per kind and decode-only. A broader fit (more prompts per kind, or
   prefill mass too) needs routing captured on a GPU, so it is not $0; the shakedown could capture it at no extra box
   if the reviewer prefers that to the traces.

## Amendment 1 (2026-11-05): the shakedown's box side

Written 2026-11-05 (clock read 17:56Z). The registration merged as 582514dc with no box side: no controller driver, no box
runner, no staged pins. So pod-launch had no command to run. This amendment adds them for **the shakedown only**. Its merge is
the shakedown's launch commit. The reading's box plan is registered in a later amendment, after the shakedown's receipt.
Nothing above changes, except the shakedown's fit and its worst case (below).

**What it adds:**
- `bench/pgx/pgx_drive.sh`, the controller side. It is P238's driver with this lane's run ids (`pgx-shake-1`, retry ids
  `pgx-shake-r1` to `-r3`), and it sources the shared `bench/common/drive_tail.sh`. Its fetch is `rsync --no-links`.
- `bench/pgx/pgx_run.sh`, the box. `bench/pgx/staged-pgx.sha256` pins its 22 staged files. Two of them come from shared
  code: SC1's SGLang venv installer (`bench/sc1/sglang/install.sh`) and P39's NF4 bake.
- `pgx_tripwire.py`, `pgx_box.py` (the lifetime record) and `pgx_shake.py` (the verdict).
- `tests/test_pgx_kit.py`. It runs, at $0 against this commit, every box step that needs no GPU: the staging, the tripwire,
  both self-tests, both hooks, the header reader and the exit codes.

**The box (one RTX 4090; the host floors above):**
- **Refusals, before STARTED:**

  | code | cause |
  |---|---|
  | 18 | CUDA unusable, or a driver below 580 (SGLang 0.5.23 pins torch 2.14.1, whose wheels are CUDA 13) |
  | 10 | torch will not import |
  | 28 | the image's torch is not `2.8.0+cu128` |
  | 15 | the card: not a 4090, not cc 8.9, < 128 SMs or < 24,000 MiB |
  | 13 | disk < 250 GB |
  | 16 | measured RAM, min(MemTotal, cgroup), < 60 GiB |
  | 19 | download < 40 MB/s |
  | 41 | the guard fit |

  Only 13 and 18 name the machine to the launcher.
- **Installs:**
  - e4b at the launch commit, from git, in the image's python. 0.58.1 predates e4b#2169, so no release carries
    `E4B_PAGED_PROFILE`.
  - grouped-nf4-gemm 0.51.0 from PyPI. Its `bench/calibrate.py` is checked out from the v0.51.0 tag commit 30adb1bf.
  - SGLang 0.5.23 in its own venv.
  - vLLM 0.29.0, with `vllm-expert-pager` installed from git at e9264fda, in its own venv.
  - Each has a tripwire. A failure is rc 9.
- **Then, in order:**
  1. The kit's self-tests (rc 25).
  2. The three checkpoints are fetched at their revisions (rc 11).
  3. **The bytes per expert are read from the box's own snapshots.** Their headers must hash to the registered ones (rc 9
     otherwise), and the K table is recomputed from them.
  4. The NF4 bake (rc 12).
  5. `calibrate.py` (rc 33), then `pgx_place_preview.py` on the box's `calib.json`. This is E-hyb's expectation, receipted.
  6. A 20 s card-clock burn, recorded and never ruled.
- **Lifetimes:** one per arm at 8 GB, in draw 1's order: S4, S8, V8, E-hyb. Each runs P1 and P2 at c1 over `warm_c1`.
  Every server runs with prefix or radix caching off, at a context of 2,048 and at most 4 sequences.
- **Each counter shim on a real record:**
  - **S4 and S8:** an untimed instrument lifetime before the arm's timed one. It runs with decode and prefill graphs
    `disabled`, `PGX_ROUTE_DUMP` on, and the P1 prompts once (pass `I1`). The dump replays through SGLang's rule
    (`pgx_residency.sglang_residency`), and it must self-check: the engine's own host-path loads equal the replay
    (`host_path_check`, 0 mismatches). The registration asked for one such pass; the shakedown runs one per SGLang arm.
  - **V8:** `VLLM_EXPERT_PAGER_LOG_INTERVAL=1`, and one eager probe before P1, between P1 and P2, and after P2. A probe is
    one 1,024-token prefill with 1 output token, longer than any captured size. Each pass's counts are the difference of
    the cumulative counters at its two probes.
  - **E-hyb:** `/health`'s `levers.placement_profile`, the step trace, and **the placement proof from a new in-process
    hook** (`bench/pgx/hook_e4b/sitecustomize.py`). `serve_paged` reports the profile but not the tiers. The hook wraps
    `enable_hybrid_tier` and records the manifest `build_engine` hands it, as per-layer expert counts per tier.
- **Per lifetime:** wall time, peak host RSS (the server's process group) and peak VRAM, sampled each second.

**A finding from the $0 path, before any box: the registered SGLang hook could not have loaded.** `bench/pgx/hook/` is
loaded as `usercustomize`. Python runs `usercustomize` only when `site.ENABLE_USER_SITE` is true, and a venv without
system site-packages sets it false. Every engine on the box runs in such a venv, so no SGLang process would have dumped
anything. `bench/pgx/hook/sitecustomize.py` now imports it; `sitecustomize` loads in every interpreter.
`tests/test_pgx_kit.py` shows both cases in a real venv: the hook loads with the shim, and does not load without it.

**The shakedown's verdict (`pgx_shake.py`):**
- **STOP:** an arm did not serve. Its server did not start, or a pass is missing, has a request fault or has the wrong
  `prompts_sha`. As registered: "If any arm cannot serve at its budget: STOP and report."
- **SHAKE_FAULTS:** every arm served, but a lifetime carries a fault the reducer would VOID it for (`pgx_reduce.lifetime_faults`).
  The reading's kit needs an amendment first.
- **SHAKE_OK:** neither of the above.

The box finishes 0 only on SHAKE_OK, and 27 otherwise. It is never read. P2 against P1 token identity is recorded and not
ruled.

**The shakedown's fit and worst case (they replace the shakedown row above):**
- setup 5,350 s + 4 lifetimes x 900 s + 2 instrument lifetimes x 450 s + reserve 600 s = 10,450 s;
- x 1.25 = **13,062 s = 3.63 h**;
- worst = 3.63 h x USD 0.85 + 120 GB x USD 0.02 = **USD 5.49**, against USD 5.22 before.
- The lane worst becomes 5.49 + 20.62 + 0.75 = **USD 26.86**, under the USD 35 per-run cap.

**Open risks, stated now (each would read SHAKE_FAULTS or STOP, never a reading):**
- S4's Marlin MoE is compiled by SGLang's JIT with the image's nvcc (CUDA 12.8), while SGLang 0.5.23's torch is CUDA 13.
- The pager may not log on the probe if its MoE call runs inside a captured piece.
- E-hyb's engine-loaded slot bytes are read from the placement proof. S4's, S8's and V8's slot bytes are read only from
  their logs and peak VRAM: the K check uses the checkpoint headers measured on the box.

## Amendment 2 (2026-11-06): pgx-shake-1's bake ran in the wrong interpreter

Written 2026-11-06 (clock reads 01:19Z and 01:49Z). Nothing above changes except what is listed here.

**What happened: a defect in this lane's kit, found by the shakedown.** `pgx-shake-1` passed:
- every refusal check (driver 580.95, torch 2.8.0+cu128, cc 8.9 with 128 SMs, 62 GiB, 70 MB/s);
- the guard fit (14,398 s left against the 13,062 s it needs);
- the four installs and all their tripwires;
- the three fetches, with K measured at 67/35/35, as registered.

It then ended rc 12 in the bake's first second, at USD 0.959.

The cause:
- After the SGLang install, `pgx_run.sh` sourced SC1's `sglang.env` in its own top-level shell. That file puts
  `venv-sglang/bin` first on `PATH`.
- From that line on, every bare `python` in the runner was SGLang's venv, which has torch 2.14.1 and no e4b.
- `k8_bake.py` therefore failed on `from experts4bit_qlora.loader import ...`. Its own handler caught the error, printed
  `BAKE FAILED` and exited 0. With no arena written, the runner finished 12.
- The same failure is reproduced at $0 with the real `k8_bake.py`, a stand-in venv python and the same `export`, byte
  for byte: a 12-byte `bake.log`, no arena, and `ModuleNotFoundError: No module named 'experts4bit_qlora'` in
  `bake.json`.
- It was not the bake alone. Seventeen bare `python` call sites followed the source line: calibrate, the placement preview, the
  card clock, the E-hyb server, the K and driver helpers, `pgx_box` and `pgx_shake`. The ones that happened to work ran on
  the wrong interpreter. SC1 itself sources the file only inside its installer and server scripts, never in a lane
  runner's top shell.

**1. The fix.**
- The runner binds the image python by absolute path before anything can change `PATH`
  (`PY_IMG=$(command -v python)`), and every image-python step runs `"$PY_IMG"`. Every other python call is an explicit
  venv path.
- Nothing sources `sglang.env` in the runner's shell. It is sourced only inside the SGLang subshells, the tripwire and the
  server, as SC1's `server.sh` does.
- The vLLM venv is now created from `"$PY_IMG"`. Before, it was created from SGLang's venv python.

**2. The guard.** Right before the bake, calibrate, the placement preview and the E-hyb server start, `img_guard` runs
`"$PY_IMG" -c 'import experts4bit_qlora, nvme_arena'`, the interpreter the step itself runs, and stops at rc 9 on failure.
`logs/interpreters.txt` records what a bare `python` and `$PY_IMG` resolve to at each phase.

**3. The tripwire runs the bake's and calibrate's chain, at rc 9, before the fetch.**
- `pgx_tripwire.py --box ... --bake k8_bake.py --calib calibrate.py` reads every import in the staged `k8_bake.py` from its
  source and executes each one. It binds the bake's calls to `load_moe_4bit_streaming` and `bake_expert_tensors` against
  the installed signatures.
- It imports grouped-nf4-gemm's `calibrate.py` and builds its CPU microbench with calibrate's own `build_microbench`.
- It runs in `"$PY_IMG"`, the interpreter the bake and calibrate now run.

**4. Observability.**
- On either bake failure, the runner appends `work/bake.json`'s status, error and traceback to `logs/bake.log`.
- The driver fetches `work/*.json`. The NF4 snapshot and the arena stay on the box.
- The bake runs with `PYTHONPATH` cleared, as the tripwire does.
- `env_image.txt` also records `PYTHON*`, `PIP_*`, `LD_LIBRARY_PATH` and `PATH`.
- `logs/sys_path.txt` records the image python's `sys.path`.

**5. The launch slack, registered.**
- The shakedown's guard is **4.05 h**: Amendment 1's fit of 13,062 s plus 1,500 s of launch slack.
  - rent.py hands the box a deadline of launch + wallclock, so a guard equal to the fit leaves less than the fit at the
    box's check, and every box refuses rc 41.
  - pgx-shake-1 read 14,398 s left against 13,062 s needed.
- The registered rc 41 rule is unchanged.
- Worst case per box: 4.05 h x USD 0.85 + 120 GB x USD 0.02 = **USD 5.85**.
- The maintainer ruled this for pgx-shake-1 in its Approved; this registers it.

**6. The stop: one exemption, for pgx-shake-1's exact record.** `pgx-shake-1`'s record is exempt from "a started box that
did not finish 0 stops for a person", for every later id, when all of these hold:
- it STARTED;
- its exit code is 12;
- `logs/bake.log` begins with the line `BAKE FAILED`;
- its `outer.log` says `BAKE FAIL: no arena`;
- nothing ran after the bake (no `logs/calibrate.log` and no `shake/`).

`pgx-shake-1` still counts as one STARTED box and one strike. Every other stop is unchanged:
- host refusals before STARTED (13, 15, 16, 18, 19) are not strikes, and the four ids bound them, as registered. So `r2`
  may follow an `r1` refused before STARTED, and `r3` an `r2`;
- the next box that STARTS and does not finish 0 is strike two and ends the lane;
- any other record of `pgx-shake-1` reads the ordinary stop.

The kit implements this, with stub cases for each way the exemption refuses. Two of them: `r2` after `r1` rc 18 before
STARTED opens; `r2` after `r1` STARTED and not 0 stops.

**7. Pins, hygiene only.** The image-python install line now pins `accelerate==1.15.0`, `safetensors==0.8.0` and
`huggingface_hub==1.33.0`. These are the versions an unpinned install resolved to on 2026-11-06 under transformers 5.17.0's
bounds. The tripwire checks every package on that line at its pin. These pins are not the fix.

**8. The kit test, at $0.** `tests/test_pgx_kit.py`:
- **static:** no top-level source in `pgx_run.sh`; `sglang.env` only in subshells; no bare `python` invocation;
  `PY_IMG` bound before the first probe and before the SGLang install; each e4b step guarded by its own interpreter;
- **dynamic, in pgx-shake-1's order:** bind `PY_IMG` with this runner's own line, install a stand-in SGLang venv through
  SC1's `sglang.env` form, then import e4b in the bake's interpreter. Amendment 1's sequence reproduces the box's
  `ModuleNotFoundError`; this runner's does not. All four static and dynamic checks fail against Amendment 1's runner;
- each new tripwire step on this commit, and on a fixture for each way it fails;
- `bake_why`, with and without a record;
- the driver's rsync rules, on a fixture tree;
- the bake's exact load call on a shardless snapshot, which must log the host and the checkpoint and stop only at the
  first shard read.

The suite also runs once in a CPU venv at torch 2.8.0 + transformers 5.17.0 with the pins in 7, before the kit is cut.

**The worst cases, after this amendment:**
- the shakedown per box: USD 5.85;
- the lane: pgx-shake-1's USD 0.959 + the next started shakedown box's USD 5.85 + the reading's USD 20.62 + three
  refused retries at about USD 0.25 each = **USD 28.18**, under the USD 35 per-run cap.

## Amendment 3 (2026-11-06): the reading's box plan

Written 2026-11-06 (clock read 06:40Z) after the shakedown's receipt, as the registration requires ("The plan is decided from
the shakedown's receipt and stated before the reading launches"). Nothing above changes except what is listed here.

**The shakedown's record.** `pgx-shake-r1` finished 0 with `SHAKE_OK`: all four arms served at 8 GB, with 0 faults and
K measured 67/35/35 as registered. Its files are committed at `bench/pgx/receipts/pgx-shake-r1/`, and `pgx-shake-1`'s (the record
Amendment 2 rule 6 names) at `bench/pgx/receipts/pgx-shake-1/`. Each has a README and `SHA256SUMS`. The shakedown is never
read, and no rule below is chosen from its speed numbers. The fit is necessarily computed from them: that is what the
shakedown is for. Every choice made after seeing the receipt is named where it is made, marked **(after the receipt)**.

**1. The fit (`bench/pgx/pgx_schedule.py`; every constant is read back from the receipt by the kit test).**
- **Setup:** STARTED to the first lifetime, measured: 950 s (installs 234, fetch 584, bake 42, calibrate + preview + clock
  ~90). The margin: the fetch is re-fit at the runner's 40 MB/s refusal floor (2,750 s for about 110 GB) in place of the
  measured 584 s, **(after the receipt)**. Setup fit = 950 - 584 + 2,750 = **3,116 s**. Each box pays it, since every box is a
  new instance.
- **Measured at 8 GB, c1, over warm_c1** (8 requests x 256 tokens): pass walls S4 15.5 s, S8 48.1 s, V8 44.2 s, E-hyb
  412 / 397 s (median TPOT 6.1, 20.1, 18.4 and 190.7 ms). A lifetime's load is its wall minus its two passes.
- **4 and 12 GB were NOT measured on the shakedown.** They are scaled, INFERRED, by TPOT ratios:
  - **S4, S8, V8:** TPOT = a fixed part + misses x bytes per expert / link (calibrate's 24.07 GB/s h2d). The fixed part is
    each arm's 8 GB TPOT minus its predicted page-in time. The model fits all three at 8 GB: S8's predicted page-ins
    are 18.2 of its 20.1 ms. At 4 and 12 GB the hit rate is the registered prediction minus its own 0.05 band (more
    page-ins), **(after the receipt)**. Ratios to 8 GB: S4 2.78 / 1.10, S8 2.00 / 0.64, V8 2.10 / 0.61 at 4 / 12 GB.
  - **E-hyb, E-hyb-u (INFERRED):** TPOT = P66's eager GPU decode (8.05 ms) + the CPU tier's routed mass x a per-mass cost
    fit at 8 GB. All of E-hyb's excess over the eager decode is charged to the CPU tier. The CPU mass comes from the box's
    own placement preview (`place_preview.json`): 0.431 / 0.200 / 0.099 at 4 / 8 / 12 GB, and 0.507 for E-hyb-u.
    Ratios: E-hyb 2.11 / 0.51; E-hyb-u 2.47.
  - **The two all-VRAM controls** were never run: **300 s a lifetime, INFERRED.**
- **Passes:** P1 and P2 are the shakedown's c1 passes scaled. P3 (c4, 16 requests) is twice the c1 pass, i.e.
  serial-equivalent: c4 is assumed to buy no throughput (E-hyb's CPU tier reads each expert once per step, so a batch of 4
  should cost no more per sequence than one). P4 is the P2 pass.
- **The instrument lifetimes** (below, 2) run the same passes, slowed by the shakedown's measured graphs-off ratio I1 / P1
  (S4 5.53, S8 2.97). S4's first SGLang start on a box adds its Marlin JIT once: 353 s (496 - 86 - 57, measured).
- **Per draw** (s): instrument lifetimes 4,378; S4 + S8 + V8 2,525; E-hyb 6,281; E-hyb-u 4,967; controls 600; total
  **18,751**. E-hyb and E-hyb-u are 60 % of it.
- **A box's need** = (setup + JIT + its draws) x 1.25. Its guard = the need + 1,500 s of launch slack (Amendment 2 rule 5).
  Worst = guard x USD 0.85 + 120 GB x USD 0.02.

**2. S4/S8 residency at every cell: an instrument lifetime per (S arm, budget, draw)** (the maintainer's ruling). Each
timed S lifetime is preceded by an untimed graphs-off instrument lifetime at the same budget and draw. That lifetime runs
the same pass sequence (P1-P3, + P4 at 8 GB) with the routing dump, so every timed SGLang cell has its own replay. The
attribution needs residency at c4, and the predictions cover c1 only. The server's own warmup is tagged `W0`: it is
replayed for the cache state and never scored.
- **(i) The assumption, stated:** the graphs-off instrument lifetime forms the same decode batches as the graphs-on timed
  one. The loop is closed at c = 4 with fixed `max_tokens` and `ignore_eos`, so each wave of 4 starts and ends together.
- **(ii) The check:** SGLang 0.5.23 logs `Decode batch, #running-req: N` every 40 decode tokens at its default level, and
  both lifetimes log it. Per pass, the two logged sequences are compared inside the runner's pass windows (start and end
  epochs, the boundary seconds excluded). Equal means `BATCH_MATCHED`; otherwise that cell's residency reads
  `APPROXIMATE`, reported and never a VOID. No server flag changes.

**3. The split (the registered rule: the fewest K of 1, 2 or 3 boxes whose largest box fits the box cap).** The box cap is
min(18 h, (USD 15 - 2.40) / 0.85 = 14.82 h) = 53,364 s.

| K | boxes (draws) | guard | worst | fits |
|---|---|---|---|---|
| 1 | {1, 2, 3} | 76,152 s = 21.15 h | USD 20.38 | no |
| **2** | **{1, 2}; {3}** | **52,713 s = 14.64 h; 29,274 s = 8.13 h** | **USD 14.85 + 9.31 = 24.16** | **yes** |
| 3 | {1}; {2}; {3} | 29,274 s = 8.13 h each | USD 9.31 each, 27.93 | yes |

- **K = 2.** The kit carries it as a constant (`pgx_schedule.PLAN_K`) and nothing recomputes it at launch. Box 1 fits with
  651 s to spare. **K = 3 is the fallback**, and only a later amendment can move to it, from a new receipt-based fit.
  A launch-time number never does.
- **Run ids:** box i of 2 runs as `pgx-4090-1-2-<i>`. A box that does not start is retried as `pgx-4090-r<n>-2-<i>`, which
  names the box it replaces, with n = 1 to 3 across both boxes (three retry ids in all, as registered). The runner and the
  driver accept exactly these ids.
- Boxes are whole draws, as registered. Box 2 is pinned to box 1's machine; if that machine cannot be rented within 24 h of
  box 1's receipt, the reading is labelled `cross_host`.
- **The per-run override:** both boxes are over the 6 h cap. The rental policy's Teardown Policy carries the owner's
  standing approval (2026-10-26): a run may declare `wallclock_override` up to 18 h when its estimate, the launcher's worst
  at the guard with downloads, is at most USD 15. The launcher enforces it per box. Each box's manifest declares it; the
  launcher checks it against that box's estimate; the receipt records it; and the create disclosure states the box's
  hours and estimate against USD 15. No further grant is needed.
- **Each box reduces its own draws** (`pgx_reduce.py --mode part`, P238's form): its lifetimes, passes and ruled token
  identity. It finishes 0 with `PART_OK`, else 27. The reading is the controller's merge of both boxes' directories.

**4. The host RAM floor stays: listed >= 62 GiB, measured >= 60 GiB.** The shakedown's peak host RSS was 49.96 GiB (S8),
so 60 GiB leaves 10.04 GiB (20 % of the peak). S8's and V8's RAM tier holds every expert at every budget, so their RSS
does not grow at 4 or 12 GB. E-hyb's largest DRAM tier is at 4 GB, 4,638 experts x 2.65 MB = 12.3 GB against 8.3 GB at
8 GB, so about 23 GiB against its measured 19.13. The largest arm stays S8.

**5. Two registered measures had no producer, found while building the kit (not from any reading's data).**
- **E-hyb's DRAM-tier host time** was to be "read from the step trace's host-tier time". At this commit e4b's step trace
  has no tier split: a decode step is `{"seg": {"dec_prep", "dec_issue", "dec_sync", ...}, "step_ms"}`, and `dec_issue`
  (155-217 ms a step on the shakedown) covers both tiers. The reducer's field existed only in a handwritten fixture.
- **E-hyb's VRAM-tier hit rate** was to come "from E4B_PAGED_STEP_TRACE and the tier counters". Neither the step trace
  nor `/health` carries tier counts. e4b counts tier activations only in `_HybridTier.amort`, which `serve_paged` leaves
  off.
- The shakedown's lesson: its E-hyb `counters_ok` checked only that the trace was non-empty. In the reading kit, each
  lifetime's counters check covers every quantity the reducer can VOID on: S, the per-pass dump and its host-path
  self-check; V8, every pass between two probes; E arms, the placement proof, the profile and the trace. The tier
  quantities below carry their own invariant and read "not measured", never VOID.

**The replacement source: tier counters in the E-hyb hook** (`bench/pgx/hook_e4b/sitecustomize.py`, already loaded in
every E-hyb server for the placement proof). It wraps the five `_HybridTier` methods a routed row can take:
- `forward` (the per-layer entry): every routed slot;
- `_cold_contrib`: every non-VRAM row;
- `_dram_contrib` (the DRAM bus on the host): DRAM rows and a `perf_counter_ns` wall;
- `_dram_on_gpu` (the thin and offload route): DRAM rows;
- `_cold_on_cpu`: host NVMe rows, empty in stage 1.

Every count is the `numel()` of an index the method already holds, so it adds no sync. A call of at most 4 rows is decode.
A daemon thread appends the cumulative counters with a wall epoch once a second. Per pass, over the runner's windows:
- **the DRAM-tier host time** = the decode `_dram_contrib` wall per decode token. It includes the method's own host copy
  of the routed ids, as e4b's own `amort["dram_ns"]` does. With no record for a cell, the attribution's `dram_host`
  branch reads "not measured", reported and never a VOID; the other two branches are unaffected;
- **the VRAM-tier hit rate** = 1 - non-VRAM rows / routed slots, reported only, under an invariant: the routed slots must
  equal the pass's decode tokens x MoE layers x top_k, the last two read from the box's `/health`. If they are unequal, a
  route was missed, and that cell's hit rate reads "not measured". Beside it, never ruled: the preview's predicted VRAM
  share on the same box (the shakedown's preview gave 0.800 at 8 GB).
- **Q7's hit-rate half** reads from this measure. If no E-hyb or E-hyb-u cell at 8 GB passes the invariant, Q7 reads
  `NOT_TESTED`, not a fail.
- **The cost, bounded:** `bench/pgx/pgx_dram_bench.py` times each wrapper around a no-op original against the no-op, and
  one append. At $0, on e4b's real `_HybridTier` under Python 3.14, it measured 309 + 286 + 622 ns a layer for forward,
  `_cold_contrib` and `_dram_contrib`. That is 48 x 1.22 us = 58 us a decode step, 0.031 % of the shakedown's 190.7 ms
  E-hyb step, plus 140 us a second for the append (0.014 %): **0.045 %, under the 0.1 % bound.** At 12 GB's modelled
  98 ms step it is about 0.07 %. Each reading box reruns the bench on its own image python before any lifetime, and a
  bound over 0.1 % or a wrap not in place is a premise failure (25).
- **Symmetry, stated:** the counters load only in the e4b arms (E-hyb, E-hyb-u). The baselines carry nothing equivalent,
  and the bound above is why that is acceptable.
- **Tested on the real producer:** the bench and the kit test import e4b's real `hybrid` module through the hook. They
  run `_dram_contrib`'s real body (its production `amort is None` branch, into a counting stub of the inner kernel) and
  enter the other four real bodies, which stop at their first tensor op on CPU. They check the decode/prefill split on a
  mixed set of calls, and that per-pass deltas from the hook's own output match a direct count, with the invariant
  catching a deliberately missed route. A hook that omits any one of the five routes fails. The route list is pinned to
  `_HybridTier`'s methods at this commit, so a new method fails the test until it is inspected.

**6. P2 against P1 for S4, registered for the reading.** The shakedown recorded S4's P2 differing from its P1, and S8's,
V8's and E-hyb's identical. As registered, S4 (Marlin) is reported and never ruled: at the reading, its P2-P1 token
identity is reported per lifetime, and it never VOIDs a pass or moves a cell. No rule is added from the shakedown.

**7. The reading's box side** (staged and hash-pinned with the rest, 25 files in `staged-pgx.sha256`):
- `bench/pgx/pgx_read.sh` is sourced by `pgx_run.sh` on a read box after the shared setup. It runs, in order:
  1. the tier-counter bench (25 on failure);
  2. the measured K at every budget;
  3. the part reduction's registration record;
  4. each lifetime in `pgx_schedule.py`'s order, started only if its modelled need x 1.25 fits what is left (a skipped
     lifetime leaves its cells to VOID);
  5. the part reduction.
- `pgx_run.sh`: the reading ids; its rc 41 fit is `pgx_schedule.py --need` for the box; the whole registered K table and
  E-hyb's GiB knob per budget; drive at c1 or c4 with each lifetime's own step alarm; the E arms' servers per arm
  (E-hyb-u without the profile, the all-VRAM controls at `all-vram`, graphs auto or 0).
- `pgx_box.py`: any arm, budget, draw and pass list; V8's n + 1 probes; per-pass S residency; the batch check; the tier
  counters.
- `pgx_reduce.py`: per-cell residency with its label; the hit-rate sanity line; Q7's hit-rate half; "not measured"; part
  mode.
- `pgx_drive.sh` accepts the reading ids.
- The interpreter rules of Amendment 2 hold in `pgx_read.sh`: no bare `python`, no `source` of an env file, every e4b step
  behind `img_guard` in its own interpreter.
- **The $0 kit test** is `tests/test_pgx_reading.py`, with `test_pgx_kit.py` unchanged and passing. It covers:
  - the schedule;
  - the plan and its pinned K;
  - every fit constant read back from the committed receipt;
  - the ids, both patterns;
  - the tier counters on the real module, with the five route mutants;
  - the readers;
  - part mode on the shakedown's real lifetimes: their P1 and P2 rows carry no fault, and every other registered row of
    draw 1 faults as missing;
  - a stubbed run of box 2 end to end, in schedule order, with the skip path.

**The worst cases, after this amendment:**
- the reading: two boxes, USD 14.85 + 9.31 = **USD 24.16**, each under USD 15;
- the lane: spent USD 2.36 (pgx-shake-1 0.96 + pgx-shake-r1 1.40) + the reading 24.16 + three non-starting retries at about
  USD 0.25 = **USD 27.27**, under the USD 35 per-run cap. At the K = 3 fallback it would be 2.36 + 27.93 + 0.75 = USD 31.04.

## Amendment 4 (2026-11-06): how a reading box ends

Written 2026-11-06 (clock read 09:58Z), after Amendment 3's merge and before any reading box. It fixes a defect in how the
registered stop and Amendment 3 combine. It is not chosen from any reading's data: there is none yet.

**The defect.** Amendment 2 says "the next box that STARTS and does not finish 0 is strike two and ends the lane".
Amendment 3 says a reading box "finishes 0 with `PART_OK`, else 27". Together, one VOID cell anywhere in box 1's 14.64 h
(one request fault, or one ruled token-identity miss) gives 27. That is strike two (`pgx-shake-1` is strike one), so the
lane ends and box 2 never runs. Draw 3 is then missing, the three-draw stability rule cannot read, and box 1's spend is
wasted. Strikes exist to stop boxes that fail to run, not boxes that ran and recorded VOID cells.

**1. A reading box ends in one of three classes** (`bench/pgx/pgx_box_end.py`, run by the controller on the fetched run
directory):
- **(a) `PART_OK`:** rc 0 with a part record that verifies and reads `PART_OK`. Not a strike.
- **(b) `PART_FAULTS`:** rc 27 with a complete, verifiable part record that reads `PART_FAULTS`. **It is not a strike.**
  Its faulted cells are VOID under the registered rules, unchanged, and the next box proceeds. A class (b) box is not
  retried: retry ids are for boxes that do not start, as registered.
- **(c) anything else after `STARTED`:** no part record, an incomplete or unverifiable one, a crash, or any other code.
  **That is a strike, as registered.**

**What "verifies" means.** The controller reruns the box's own part reduction (`pgx_reduce.py`, part mode) on the fetched
`read/` directory, with the box's own `registration.json`, for the draws the box's run id names. The result must equal
the box's `part_verdict.json` field for field. So the record covers every scheduled lifetime of those draws, skipped ones
included (a skipped lifetime reduces as missing), and it belongs to the directory it came with. Also required:
- the box ran the registered schedule (`box_schedule.txt` equals `pgx_schedule.py --box K-i`);
- the exit code is the one in the box's file for its own nonce, and it agrees with the code the launcher read. The
  nonce comes from the box, so it must be the driver's 16-hex form before it is used in a path; anything else is
  class (c);
- rc 0 carries the nonce's success marker.

A 27 without such a record is class (c). A 27 can also come from a reduction that crashed: `pgx_read.sh` maps any
non-zero reducer exit to 27. An rc 0 without a verified `PART_OK` record is also class (c). The box side cannot produce
one, so it fails closed.

**Only a missing `STARTED` reads as never started.** Not started is the permissive outcome, so any other error reading
`STARTED` (a permission error, say) is not absence. The classifier raises, and the kit stops for a person.

**Strikes count lane-wide.** `pgx-shake-1` is strike one (Amendment 2, rule 6). `pgx-shake-r1` finished 0. So any class
(c) reading box is strike two and ends the lane. Host refusals before `STARTED` stay non-strikes, as registered.

**2. The shakedown's "STOP and report", carried into the reading.** Box 1's part record can show an arm that served NO
lifetime at some budget. That means each of the arm's scheduled lifetimes at that budget either recorded a startup
failure or completed no request in any pass. If so, the lane **stops for a person before box 2**. That is a finding about
the engine, not a retry. A lifetime skipped for time is not a serve failure. The classifier lists these as `no_serve`
and exits 11.

**3. This changes no reading rule.** VOID, READ, the stability rule and the attribution are unchanged. It changes only
whether the next box may launch. It is made before any reading box.

**4. Its tests, at $0** (`tests/test_pgx_box_end.py`):
- class (b) on the shakedown's real lifetimes (r1, committed), reduced in part mode as box 2-1 would reduce them;
- class (a) on a whole-draw fixture built from r1's real records, under the plain and the retry ids;
- class (c): 17 record mutants (a missing, truncated or edited record, a lifetime removed after the reduction, a wrong
  schedule, nonce, id or registration) and every other code after `STARTED`. Each test names the check that caught it;
- six malformed nonces, `../x` among them, each with a readable 27 planted where the bad nonce would lead: class (c);
- a `STARTED` that cannot be stat'ed raises; it never reads as not started;
- the `no_serve` rule:
  - a startup failure, and every request faulted, each stop;
  - one serving lifetime of two does not stop;
  - a lifetime skipped for time does not stop;
  - a control that served nothing stops;
- the box schedule's timed lifetimes equal the reducer's schedule for each box's draws.

Code mutants of the classifier were also run. Each of these fails at least one test:
- skipping the verification;
- skipping the schedule check;
- skipping the success marker;
- reading `no_serve` from any one lifetime instead of all;
- counting a skipped lifetime as a serve failure;
- skipping the exit-code check;
- testing `STARTED` with `os.path.exists` (which reads any stat error as absent);
- skipping the nonce check, or matching it with `re.match` in place of `re.fullmatch`.

**The reading kit is cut from this amendment's merge.** That merge is the launch commit. The kit's stop calls
`pgx_box_end.py` on each started reading box:
- (a) and (b) add no strike;
- (b) with `no_serve` stops for a person;
- (c) adds a strike;
- any exit outside the five classes stops for a person.

Nothing staged to the box changes: `staged-pgx.sha256` is unchanged.
