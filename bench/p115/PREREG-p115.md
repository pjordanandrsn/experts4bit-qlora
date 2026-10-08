# P115 — does the registered B=1 fused stack (`E4B_PAGED_FUSE_QKV=1` + `E4B_FUSE_T1_GLUE=1` + `E4B_FUSE_T1_GLUE_R2=1` + `E4B_FUSE_ROUTER_EPI=1`) decode the default `serve_paged` server faster, at no measurable quality cost? On one RTX 5090 (registered 2026-10-07, before any run)

Issue: experts4bit-qlora#1313 (the B1 work item: single-stream decode on the default NF4 server). Lane number claimed by
`prereg/p115` (pushed 2026-10-07T17:01:04Z). Follows P109–P111 (the default graph server) and P54 / P58 / P88 (the fused
stack on the int4 route).

## Why this lane

**The default server's B=1 step is 2.3× e4b's own int4 route.** On one RTX 5090, Qwen3-30B-A3B NF4 through the default
graph server decodes one request at 9.05–10.06 ms per token (`bench/p109/RESULTS-p109.md`, `bench/p111/RESULTS-p111.md`);
the licensed int4 configuration serves 4.14 ms (SC2c). SV2's kernel census of the graphed NF4 step
(`bench/hybrid-g9/sv2/RESULTS-sv2-device-census.md`) puts about 1.7 ms in glue: elementwise / index / casts 0.938 ms over
692 launches, the router's top-k and sort 0.311 ms over 96, norms and residuals ~0.43 ms over ~200.

**The folds remove that glue, and they are opt-in.** bo7 (`bench/hybrid-g9/throughput-20260904/bo7/`) timed Qwen3's B=1
step at 8.68 ms bare (`qwen3_b1_nf4.json`) and 6.16 ms with the three folds (`qwen3_b1_folds.json`), both without fused
q/k/v, on the harness's graph window. bo7's README says the folds alone have no K8 on record. On bo5 they read
FAIL-by-improving (−0.073 ppl, one text, inside that arm's own 0.0095-nat floor): the K8 gate is two-sided for an
uncalibrated arm. At the time the fused router epilogue returned fp32 routing weights where upstream returns bf16
(#726); 0.37.5 made it return the model's dtype (P70: INDISTINGUISHABLE). Nobody has read the folds on the NF4 route
since. Fused q/k/v has never been timed on the bf16 route (bo7's fold arms ran `--no-fuse-qkv`; P54 read it on int4).

**Why a quality instrument, not token identity.** The fused stack changes bf16 arithmetic: one GEMM where there were
three (cuBLAS picks kernels by N; P54's B=16 tokens diverged on the int4 route for the same reason), a fused norm, a
fused residual-and-norm. Token agreement says how fast two bf16 orders part, not whether either is worse. P110's
teacher-forced floor answers that, and this lane reuses it. Determinism of the fused stack itself (F1 against F1) is
read bitwise.

## Subject

The shipped default server: `PagedServeConfig.from_env()` + `build_engine`, with only the model, arena and calibration
set. It captures bucketed decode graphs (`E4B_PAGED_GRAPHS=auto`, `max_seqs` 16, buckets 1–16, all-VRAM), with bulk KV
bookkeeping (#1200) and one KV-table selection per step (P111) at their defaults.
- **Model:** `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`, NF4 arena baked on the box by P39's
  `k8_bake.py` (P109's subject).
- **Stack:** e4b at the launch commit; grouped-nf4-gemm v0.42.0 (`b4f93f1c62d1e3436ed45bec8ccd608c90433737`, e4b CI's
  pin at registration); transformers 5.17.0.

## Phase A — speed (`bench/p115/p115_box.py`, P111's protocol)

Four arms, in ABBA order, each a fresh process. Only the four fusion knobs differ; `build_engine` reads them at one
assembly point (`serve_paged._apply_fusions`).

| arm | knobs |
|---|---|
| F0a, F0b | all four unset: the shipped default |
| F1a, F1b | `E4B_PAGED_FUSE_QKV=1 E4B_FUSE_T1_GLUE=1 E4B_FUSE_T1_GLUE_R2=1 E4B_FUSE_ROUTER_EPI=1` |

**Workloads** are P109's: W16 is 16 distinct 512-token wikitext prompts at once, W1 is row 0 alone; 32 and 160 new
tokens, one warm pass and 3 timed passes; the decode throughput is p37's slope.

**Recorded per arm:** P111's record (every pass's token digest, the last pass's tokens per row, `graph_status`,
`graph_stats`, memory) plus `fuse_qkv` from the config and the fusion census from `parts.info`.

**The census is known before the box.** SC1's 95 receipts (`bench/h2h-2026-10-02/sc1/receipts/`) record the fused stack
on Qwen3-30B-A3B as `fuse_qkv_n / fuse_t1_glue_n / fuse_t1_glue_r2_n / fuse_router_epilogue_n` =
**48 / 193 / [48, 48] / 48**, and on Granite-3.1-3b-a800m (the folds only) 0 / 65 / [32, 32] / 32. The reducer's
`census_for` encodes both; `tests/test_p115_quality_box.py` reproduces them on tiny models.

## Phase B — quality (`bench/p115/p115_quality.py`, P110's instrument)

Two processes on the default server built eager (P110's build: the passes make their own runners):
- **OFF** (knobs unset) scores R, its floor and the mutant, and writes R's fp32 log-probs per (text, group) to the box's
  `work/ref` (they stay on the box);
- **ON** (the four knobs as F1) builds the fused stack exactly as the server does and scores against R.

**R is the default graph server's arithmetic:** device grouping on and every decode step through the bucketed path with
`capture=False`, so 12 rows pad to bucket 16 (P110's P; P109 read the replay bit-identical to it).

| arm | what it is |
|---|---|
| R | the reference: the default server's arithmetic, prompts prefilled in one 512 chunk |
| rep | R again (first group only); a floor draw if not bit-identical |
| half | floor: the group decoded as two halves of 6, each padded to bucket 8 |
| chunk | floor: prompts prefilled in 256-token chunks |
| mutant_scale | R with the decode softmax scale halved (P108's mutant); must fail the bar |
| **ON** | **the subject: R's arithmetic on the fused model** |

(P110's `rev` floor read bit-identical to R; it is dropped.)

**Texts**, 48 windows each of 512 prompt tokens and 128 teacher-forced positions, in 4 groups of 12, scored on fp32
log-probs (the true token's NLL, the argmax, KL against R):
- **wikitext:** wikitext-2-raw test, window k from token k·4096 (P97's `wikitext_windows`);
- **c4val1:** K8's c4 validation corpus (the first 2000 documents of `allenai/c4`
  `en/c4-validation.00001-of-00008.json.gz`, joined as K8 joins them), window k from token k·4096.

**Engagement, counted per pass:**
- P110's: decode attention calls = 127 × 48 (`half` ×2), device grouping on, every decode step an eager padded step of
  the group's bucket with no replays;
- **glue-kernel calls:** every grouped-nf4-gemm glue kernel the folds bind is wrapped in `int4_b32` before
  `build_engine` runs (the folds bind them at patch time). Per decode step on Qwen3 the fused stack must call
  `rmsnorm_rows / rmsnorm_resid_rows / rope_norm_heads / router_epilogue` = **49 / 48 / 96 / 48** (the input norms and
  the final norm; the residual + post-attention norm fold; q/k norm + rotary per attention; one router epilogue per
  MoE layer), and no other glue kernel. OFF passes call none. The reducer's `per_step` encodes the table (Granite's
  proof: 33 / 32 / 32 scaled residual adds / 64 rotary-only / 32);
- **forwards and `qkv_proj` calls:** with fused q/k/v, every forward calls each of the 48 `qkv_proj` modules once.

## Phase C — engagement under `auto` (registered now; its scripts land in an amendment before its box)

The default this lane would license is `auto` on all four knobs: apply where the structure licenses, never raise on a
family a fold does not match, `1` still refuses a vacuous enable, `0` the escape. That code is a separate pull request
(e4b#1313's PR-B, default unchanged). Phase C reads it on two other families, SC2d's shape, at `build_engine` level:
- **Models:** `openai/gpt-oss-20b` on SC2g's e4b path and `Qwen/Qwen3.6-35B-A3B` on P98's arena (pins as SC2d's).
- **Builds per model:** OFF (`0` on all four), ON (`auto` on all four), and EXPLICIT (`1` on all four), which must raise.
- **Passes:** 16 prompts × 32 tokens through `p109_box.run_pass`; OFF twice.
- **Gates:** SERVED; ENGAGED (the ON census equals the prediction below; OFF all zero); EXPLICIT_RAISE; DETERMINISM
  (OFF ≡ OFF); **SANE** (added in review, before any box). IDENTITY ON against OFF is reported, not gated.
- **SANE.** `auto` changes these families' bf16 arithmetic, and the engagement counts alone cannot see a fold that matches
  a family's structure but computes the wrong thing (#726's class). Phase B's instrument (`p115_quality.py`) therefore
  scores ON against OFF on each Phase C model at reduced size: 12 wikitext windows of 512 prompt tokens and 128
  teacher-forced positions, no floor arms. SANE holds iff |mean d_ON| ≤ **0.02 nats** and the argmax agreement with OFF
  is ≥ 0.95. This is a gross-error gate, not a quality reading: a broken rotary or norm fold moves the NLL by tenths of a
  nat. If the instrument cannot build a model, the amendment says so, and that model fails SANE (so FLIP_HELD).
- **Predictions:** gpt-oss-20b `0 / 49 / [24, 0] / 24` (its attention carries `sinks`, so the rotary fold refuses);
  Qwen3.6-35B-A3B `0 / 0 / [0, 0] / 40` (its norms are centered `(1 + w)` and fail the probe; its router is the
  softmax-top-k-renormalise kind); EXPLICIT raises "matched no attention module" on both.
- **FLIP_LICENSED** iff both models pass every gate; otherwise **FLIP_HELD**.

## The rule (`bench/p115/p115_reduce.py`, self-tested on 27 cases)

The verdict is the first of these that applies.

1. **VOID.** Any of:
   - an arm or a quality record is missing or not ok;
   - another e4b or grouped-nf4-gemm commit, or another model revision;
   - different prompts or lengths across arms; a void slope;
   - wrong speed engagement: every arm reports `graph` for every bucket (1–16); F0's census all zero and `fuse_qkv`
     false; F1's census `census_for(family, layers)` and `fuse_qkv` as registered;
   - wrong quality engagement: a text short of 48 windows (`rep`: one group); the phases scored different windows; a
     pass with the wrong decode calls, grouping flag or bucket statistics; an OFF pass that called a glue kernel; an ON
     pass whose glue-kernel calls are not `per_step` × the decode steps, or whose `qkv_proj` calls are not
     48 × its forwards; the OFF or ON census wrong;
   - `mutant_scale` passes the bar on either text (the gate cannot fail).
2. **NOISY.** A self-pair, F0b/F0a or F1b/F1a decode tok/s on either workload, falls outside [0.96, 1.04].
3. **FUNCTION_FAIL.** F0b's tokens differ from F0a's, or F1b's from F1a's, on any row of either workload at either
   length; or an arm's timed reps do not all digest the same. F1 differing from F0 is expected and reported.
4. **QUALITY_FAIL.** On either text, ON fails P110's bar against the floor:
   - mean d_ON ≤ B_floor + 0.01 nats, and mean |d_ON| ≤ 2 × max(S_floor, 0.005), where d_X(w) is X's mean continuation
     NLL minus R's, the floor is `half` and `chunk` (plus `rep` if R did not repeat bit for bit), B_floor the largest
     |mean d_f| and S_floor the largest mean |d_f|;
   - or, on **wikitext**, the K8 perplexity (exp of the mean NLL over the 48 × 128 positions) moves by more than
     0.05 in either direction (`experts4bit_qlora.k8_gate`'s budget for an uncalibrated arm).
   **c4val1's K8 move is reported, not gated.** At c4val1's perplexity (~16.6) 0.05 ppl is 0.0030 nats, below this
   instrument's own neutral-perturbation floor (P110 measured biases of 0.0005–0.0018 nats with an SE of ~0.002): a
   two-sided 0.05 gate there would fail an arithmetically neutral change by noise, which is bo5's trap. On wikitext
   (~6.3) 0.05 ppl is ~0.0079 nats, about four standard errors.
5. **SLOWER.** g1 = min(F1a/F0a, F1b/F0b) on W1 is below **1.10**, or g16, the same on W16, is below **1.00**. A default
   that changes bf16 arithmetic should buy a clear single-stream gain (P113 took 1.02 for a bitwise switch); it may not
   cost 16 requests.
6. **DEFAULT_AUTO** otherwise.

**Reported:** both pair ratios per workload and their geometric means; ms per step per arm; F1-vs-F0 token agreement
(rows identical, first divergence); every quality arm's bias, spread, SE, max |d|, KL and argmax agreement per text;
both texts' K8 perplexities.

## Predictions (written before any data)

| # | prediction | basis |
|---|---|---|
| Q1 | every arm captures every bucket with the fused stack; engagement exact in every pass | SC1's receipts; bo7's fold arms ran the graph loop; `tests/test_fused_glue_decode_graphs_gpu.py` |
| Q2 | F0b ≡ F0a and F1b ≡ F1a bitwise; F1 ≠ F0 on some rows | P111's S0b ≡ S0a; P54 |
| Q3 | **g1 ∈ [1.25, 1.60]**; F1's W1 step 5.8–7.2 ms | bo7's folds saved 2.52 of 8.68 ms (×1.41); on P111's ~9.05 ms host step that is ×1.39, plus 0.2–0.5 ms if fused q/k/v removes 96 of SV2's 145 cuBLAS GEMV launches |
| Q4 | **g16 ∈ [1.03, 1.20]** | bo7's B=16 folds ×1.130 on an older, slower step; P54's +2 % for fused q/k/v at 16 rows |
| Q5 | self-pairs within [0.98, 1.02] | P111 0.997–1.002 |
| Q6 | wikitext: ON bias in [−0.004, +0.004] nats, spread ≤ 0.015, \|ΔK8\| ≤ 0.03 ppl; c4val1: bias in [−0.006, +0.006], \|ΔK8\| ≤ 0.10; the floor's B ≤ 0.003 and S ≤ 0.015 on both; the mutant's bias > +0.3 | P110's floor and P's +0.0004; bo5's c4val1 floor 0.0095 nats |
| Q7 | the verdict is DEFAULT_AUTO | — |
| Q8 | each speed arm ≤ 3 min; the quality phase ≤ 25 min; peak ≤ 23 GiB; ~8 GB of reference log-probs on disk | P111: four arms in 4.7 min; P110: 576 s for 29 group-passes (here 42) |

## Consequence, registered now

- **DEFAULT_AUTO, then Phase C FLIP_LICENSED:** a separate pull request makes the four knobs default to `auto` in
  `serve_paged` (`0` on each restores the unfused server; the library's fold functions keep unset = off, because the
  bench harnesses call them directly and their unfused reference arms must not move). `docs/SERVING.md`,
  `docs/STATUS.md` and the changelog say so, with the gain and the quality read. Register rows:
  `e4b.serve.p115.fused-stack-speed.qwen3.5090.<date>` (g1), `e4b.serve.p115.fused-stack-quality.qwen3.5090.<date>`
  (wikitext ON bias; c4val1 in the claim text), and Phase C's engagement row. **Scope:** speed and quality read on
  Qwen3-30B-A3B NF4; other families engage by structure (Phase C's counts) without a quality reading of their own.
  The default PR's changelog and `docs/SERVING.md` list, per family, what was read (added in review):
  - Qwen3-30B-A3B: speed and quality;
  - gpt-oss-20b and Qwen3.6-35B-A3B: engagement and the SANE gate;
  - every other family a fold engages on by structure: nothing.

  The list also names `0` on each knob as the way back.
- **DEFAULT_AUTO, then Phase C FLIP_HELD:** a family-scoped default only under a new registration.
- **SLOWER or QUALITY_FAIL:** the knobs stay opt-in; the ratios and biases are recorded; QUALITY_FAIL names a follow-up
  lane with one knob per arm.
- **FUNCTION_FAIL:** a determinism defect to find before anything moves.
- **NOISY or VOID:** no consequence; an amendment or a rerun inside the ceiling.

## The premise and the proving rental

**The premise** runs on the card before anything is fetched (rc 25): `tests/test_decode_graph_buckets.py`,
`tests/test_kv_step_select.py` and `tests/test_fused_glue_decode_graphs_gpu.py`, **14 passed**, none skipped. The new
file captures and replays glue rounds 1 and 2 on a tiny dense Qwen3 and asserts the replay decodes exactly as the
padded eager step, with the fused kernels called. The router epilogue and fused q/k/v need a MoE attention and router;
the reading's engagement counts cover them on the served model.

**The proving rental** runs the whole box end to end on `ibm-granite/granite-3.1-3b-a800m-instruct` @
`a02780686e08a03fe0d2679a293b5c74a90efa89` at 8 / 24 tokens and 1 rep, 12 windows of 32 positions per text, with F1 =
the three folds (Granite has no Qwen3-MoE attention, so `E4B_PAGED_FUSE_QKV=1` would refuse a vacuous fusion). It is
**PROVED** iff the lane exits 0 with a verdict other than VOID. The proof's verdict is not a reading.

## Budget and STOP rules

- **Proof:** one RTX 5090 (Vast verified/secure), **guard 0.75 h** at ≤ $0.75/h, about $0.6 with its download.
- **Reading:** one RTX 5090, **guard 1.5 h** at ≤ $0.75/h, about $1.8 with the checkpoint download at about $0.011/GB
  (P113's pricing). Expected time about 50 minutes: install and premise 4, fetch 6–10, bake 2, prompts 1, four speed
  arms about 12, quality about 25.
- **Lane ceiling for Phases A and B:** $5.00 (the proof, the reading and one rerun of each). Phase C's amendment states
  its own budget; the lane's hard stop over all phases is $10. Every run is inside the owner's $15 no-ask tier.
- **STOP-1:** the refusals run before any install: dud box (10), card class (15), disk < 150 GB (13), host RAM < 60 GiB
  (16), premise (25).
- **STOP-2:** every time-left check (fetch, bake, each arm, each quality phase) fits inside its guard, enforced by
  `tests/test_p115_staged_pin.py`.
- **STOP-3:** a VOID or NOISY reading is not retried inside the same launch.
- **STOP-4:** the driver refuses a dirty tree or a staged file that differs from `bench/p115/staged.sha256`.
- **STOP-5:** the quality phase does not start when a speed arm failed; the reducer VOIDs.

## What was seen before this page (stated, not hidden)

- **No P115 data exists.** bo7's and bo5's fold arms, SC1's census counts and P110's floor are cited above from their
  receipts.
- **Locally (CPU):** the box's self-test (12 cases), the quality box's (11) and the reducer's (27) pass;
  `tests/test_p115_quality_box.py` runs both quality phases on a tiny Qwen3-MoE (stand-in attention, R unpadded) and
  reproduces the census and per-step tables on tiny Qwen3-MoE (2 and 3 layers) and GraniteMoe models with the glue
  kernels stood in. Writing that test caught a defect in the counter before any box: a bare `*args` wrapper hid
  `rmsnorm_resid_rows`'s `scale` parameter, which glue round 2 probes to license GraniteMoe's scaled fold; the
  wrappers now keep their kernels' signatures.
- **The NAS A2000** cannot run the fp8 paged KV or the bucketed path (sm_86), so the box has had no GPU rehearsal; the
  proving rental is its first run.

## What this lane cannot say

- Nothing about the int4 route, where the fused stack already runs (SC1's configuration).
- Nothing about eager decode, other cards or other hosts (B=1 ratios travel between hosts, absolutes do not), batch
  sizes between 1 and 16, or TTFT.
- Nothing about the quality of the fused stack on families other than Qwen3-30B-A3B; Phase C reads engagement only.

## Receipts

Fetched to the run directory's `p115/` and committed to `bench/p115/receipts/<run>/`:
- `arm_{F0a,F1a,F1b,F0b}.json`, `quality_off.json`, `quality_on.json`, `verdict.json`, `summary.txt`, `forensics.txt`,
  `versions.txt`, `prompts.json`, `bake.json`;
- the logs and the teardown proof;
- `SHA256SUMS`.

The reference log-probs (`work/ref`) stay on the box. `RESULTS-p115.md` is written from those files.

## Amendment 1 (2026-10-07T18:02Z, after `p115-5090-1` refused before any reading, before any reading box ran)

**STOP-1's dud-box refusal becomes the registered host floor.** `p115-5090-1` drew Vast machine 34887, whose image torch
could not use the GPU. The runner exited 10 ("DUD BOX"), a code that names no machine, so the launcher recorded
HARNESS_ERROR and a relaunch could buy the same host. That is the machine TC1's `tc1-5090-119` hit, which led TC1 to
amendment 61. `p115_run.sh` now probes as TC1 does:
- torch imports but cannot use the GPU: exit **18**, with a REFUSAL line, so the launcher names the machine;
- torch does not import (the image's fault): still 10.

Nothing else changes: no rule, prediction or budget. `tests/test_p115_staged_pin.py` pins one 18, and only in that
branch.

## Amendment 2 (2026-10-07, Phase C's scripts, budget, one corrected prediction and Granite's quality read; before any Phase C box)

Phase C was registered above with its scripts to follow in an amendment. This is that amendment. It also adds
Granite-3.1-3b-a800m's full Phase B read, which the maintainer asked for in review. No Phase C box has run. Phase A/B's
rule, predictions and budget are unchanged.

**Written before Phase A/B's data; stated, not hidden.** Phase C's scripts, rule and predictions below were fixed before
any Phase A/B number existed. Since then, Phase A/B's reading `p115-5090-8` (2026-10-07T19:32Z) read DEFAULT_AUTO:
g1 1.429 and g16 1.230, with wikitext ON bias +0.00098 nats. That reading changes nothing in Phase C: no gate, bar or
prediction here depends on it. The budget line below was corrected after it.

**Scripts.** `bench/p115/p115c_run.sh` (box), `p115c_drive.sh` (controller), `p115c_box.py`, `p115c_reduce.py`, and
`staged-c.sha256`, which pins every staged file. Phase A/B's files run at their registered bytes: the reducer, the
quality box, P109's, P110's, P108's and P97's boxes, P39's bake and calibration, and the GPU premise tests. P98's
`p98_bake.py` runs at P98's bytes. `tests/test_p115c_staged_pin.py` checks all of it in CI.

**One box, three models, in order Granite, gpt-oss, Qwen3.6.** Granite is the cheapest. A later model's fetch or bake
failure still leaves the earlier reads. Per model:
1. Fetch at the pinned revision. gpt-oss skips `original/` and `metal/`, as SC2g did.
2. Bake the NF4 arena. Granite and gpt-oss use P39's `k8_bake.py` from the pinned snapshot; Qwen3.6 uses P98's
   `p98_bake.py`.
3. Build P109's 16 wikitext prompts in the model's own tokenizer.
4. Run fresh processes of `p115c_box.py`. Each builds the shipped default server with `PagedServeConfig.from_env()` +
   `build_engine`, and only the four knobs differ:
   - **serve off** (all `0`): the census, then 16 prompts × 32 tokens, twice;
   - **serve on** (all `auto`): the census, then the prompts once;
   - **serve explicit** (all `1`): `build_engine` must refuse, and the refusal is the record;
   - gpt-oss and Qwen3.6: **SANE off** and **SANE on**, below;
   - Granite: **quality off** and **quality on**, below.
5. Run `p115c_reduce.py` over the records.

gpt-oss is served on SC2g's e4b path (`SC2G_E4B_ENV`: native MXFP4 decode, NF4 prefill). Granite and Qwen3.6 are served
on the server's defaults.

**SANE, as implemented (gpt-oss, Qwen3.6).** The default server is built eager (`E4B_PAGED_GRAPHS=0`), all-vram. Phase
B's instrument, `p115_quality.measure_phase`, runs at reduced size: 12 wikitext windows, 512 prompt tokens, 128
teacher-forced positions, one group of 12, P110's padded bucket arithmetic. The off process scores R and saves its fp32
log-probs. The on process scores ON against them. SANE holds iff |mean d_ON| ≤ 0.02 nats and mean argmax agreement
≥ 0.95, as registered. Neither family has been through this instrument before, so `tests/test_p115c_sane_families.py`
runs both phases on tiny gpt-oss and Qwen3.5-MoE hybrid models on CPU. That check confirms the instrument builds, steps
every window and writes records the reducer reads. Its stand-in attention drops the sinks, so its scores are not the
real kernel's.

**Granite's quality read (added in review).** `auto` changes Granite's arithmetic: the folds engage at
`0 / 65 / [32, 32] / 32`. SANE's 0.02-nat bar cannot settle a cost of the size the proof hinted at (below), so Granite
gets Phase B's full read instead of SANE:
- **Instrument:** Phase B's, at Phase B's size. The default server is built eager under `0` (off) and `auto` (on). Both
  texts, 48 windows each in groups of 12, 512 prompt tokens and 128 teacher-forced positions. The off process scores R,
  the `rep`, `half` and `chunk` floors and `mutant_scale`; the on process scores ON. The glue-kernel call counters are
  installed before `build_engine`, and the per-forward `qkv_proj` counter runs too. The records are `p115_quality.py`'s.
- **Rule:** Phase B's, unchanged, read with `p115_reduce.py`'s own functions:
  - VOID on any integrity fault: the census, the window counts, or any pass's engagement (decode attention calls, bucket
    stats, the per-step kernel calls of `per_step("granite", 32)`). VOID also if the mutant passes the bar.
  - On both texts, bias_ON ≤ B_floor + 0.01 and spread_ON ≤ 2 × max(S_floor, 0.005).
  - On wikitext, |K8 Δppl| ≤ 0.05. c4val1's K8 is reported, not gated.
- **Verdict:** Granite gets its own, which does not enter FLIP_LICENSED / FLIP_HELD. **GRANITE_LICENSED** iff Granite
  passes SERVED, ENGAGED, EXPLICIT_RAISE, DETERMINISM and the quality rule. **GRANITE_HELD** otherwise. **VOID** when its
  harness failed or its records fail the integrity checks.
- **Consequence:** the flip PR's `auto` allowlist takes Granite only on GRANITE_LICENSED.
- **What was seen before this amendment (stated, not hidden).** The Granite proof `p115-prove-2` read 12 windows × 32
  positions per text. It is not a reading. c4val1's ON bias was +0.0151 nats (bar 0.0184, SE 0.0054), with K8 +0.18
  ppl. Wikitext's was −0.0009 nats, with K8 −0.0075.
- **Prediction:** census and per-step counts exact. Wikitext ON bias within ±0.004 nats and |K8 Δ| ≤ 0.03. c4val1 ON
  bias between 0.000 and +0.015 nats; the proof's hint is not evidence. GRANITE_LICENSED about 60 %.

**One corrected prediction (found before data, in `tests/test_fusion_modes.py`, #1315).** With all four knobs at `1`,
glue round 1's own refusal fires on Qwen3.6 before the q/k/v check is reached. The centered norms fail its probe:
"E4B_FUSE_T1_GLUE=1 patched no RMSNorm modules … refusing a vacuous enable". The registered message, "matched no
attention module", therefore holds for gpt-oss and Granite only. EXPLICIT_RAISE accepts any of the four knobs' own
vacuous-enable refusals, provided the message names the knob at `=1` (`EXPLICIT_RE`). The same phrase raised by another
feature, such as the int4 expert plan or the int4 attention, does not pass the gate.

**Premise.** On the card, before anything is fetched: Phase A/B's 14 GPU tests plus `tests/test_fusion_modes.py`'s 34,
**48 passed**, none skipped (rc 25).

**Proof** (`p115c-prove-<n>`). Granite alone runs every process kind: serve ×3, SANE ×2 and quality ×2. It uses 8 new
tokens, and 12 windows × 32 positions for SANE and for quality. Expected census `0 / 65 / [32, 32] / 32`; the explicit
arm refuses the q/k/v fusion. A VOID from the reducer fails the proof (rc 27), and so do quality records that fail the
integrity checks. The proof's verdict is not a reading.

**Refusals (STOP-1).** CUDA unusable 18 (Amendment 1's host floor), card class 15, disk < 200 GB 13 (checkpoints of 6, 13
and 72 GB with their snapshots and arenas; each snapshot is deleted once its model's processes finish), host RAM
< 60 GiB 16, premise 25.

**STOP-2.** Every step checks the time left. Reading: fetch 1500 s, bake 900 s, each process 600 s. Proof: 300 s,
300 s and 240 s. Each check leaves 600 s for the fetch-back and fits inside its guard.

**Budget.**
- **Proof:** one RTX 5090, **guard 0.75 h**, `--download-gb 7`; about $0.7 at the launcher's policy rate.
- **Reading:** one RTX 5090, **guard 2.0 h**, `--download-gb 92`; about $2.7 at the policy rate. Expected time is about
  1.5 h: install and premise 5 min, fetches 15–40 min, bakes 7 min, Granite's five processes about 12 min, and ten
  processes for the other two models at about 3–4 min each.
- **Limits.** Phase C's ceiling is **$5.00**: the proof, the reading and one rerun of the proof. The lane's registered
  hard stop over all phases, $10, stands. Every run sits inside the owner's standing no-ask tier for a single run
  under $15, and anything over $15 needs the maintainer lane's approval first. Phases A and B cost $0.952.

**Receipts.** The box's records are fetched to the run directory's `p115c/` and committed to
`bench/p115/receipts/<run>/p115c/`:
- `serve_<m>_{off,on,explicit}.json`, `sane_<m>_{off,on}.json`, `quality_granite_{off,on}.json`, and `harness_<m>.json`
  where one was written;
- `verdict_c.json`, `summary.txt`, `forensics.txt`, `versions.txt`, `prompts_<m>.json`, `logs/`, `work_<m>/bake.json`.

The arenas, the snapshots and the reference log-probs (SANE's and Granite's) stay on the box.

## Amendment 3 (2026-10-08, after Phase C read FLIP_HELD; Phase D, the combined read, and the family-scoped default; before any Phase D box)

**Why.** Phase C read FLIP_HELD (#1342): gpt-oss-20b fails the SANE gate on argmax agreement (0.924 < 0.95). Qwen3.6
passes, and Granite reads GRANITE_LICENSED. The registered consequence is "a family-scoped default only under a new
registration". This amendment is that registration. It carries the read the maintainer asked for in #1318's review.
grouped-nf4-gemm 0.43.0 made P116's bandwidth decode GEMV the default (`GNF4_GEMV_BW=auto`, at Qwen3-30B-A3B's two expert
shapes on ≥ 160-SM parts). The fused stack (Phases A and B, at 0.42.0) and that GEMV (P116) were each read alone; the
combination never was.

**The family-scoped default (the code PR this registration licenses).** Under the default, the four fusion knobs
engage only on a family with a registered passing read. The allowlist, by `model_type`:

| `model_type` | read | registered by |
|---|---|---|
| `qwen3_moe` | Phases A and B: speed and quality | #1328 |
| `qwen3_5_moe` | Phase C: engagement and SANE (the router epilogue only) | #1342 |
| `granitemoe` | Phase C: Phase B's quality read, GRANITE_LICENSED | #1342 |

Every other family stays unfused unless set otherwise. That includes gpt-oss (`gpt_oss`), which gets a
one-knob-per-arm follow-up with its own neutral floor first.

**Mechanism: (B), unset resolves per family; explicit `auto` structural, with a warning off the allowlist** (the
maintainer's decision on the bus, 2026-10-08T18:25Z):
- An **unset** knob resolves to `auto` on an allowlisted `model_type` and to `0` everywhere else. `/health` reports each
  knob's resolution and its source (`default-allowlisted` or `explicit`).
- **Explicit `auto`** stays structural, exactly as Phase C measured it. On a `model_type` outside the allowlist it logs
  one warning naming the read that family lacks or failed, for gpt-oss: Phase C's SANE, argmax 0.924 < 0.95 (#1342).
  `docs/SERVING.md` says in one line that explicit `auto` is structural and quality-licensed only on the allowlist.
- `1` and `0` keep their meanings. No pinned file changes: `tests/test_fusion_modes.py`, which Phase C staged, still
  asserts structural `auto`.

The measurement below does not depend on the mechanism: on Qwen3-30B-A3B both the default and explicit `auto` engage
all four knobs.

### Phase D — the combined read on Qwen3-30B-A3B

**The subject.** The shipped default server at the launch commit: graphs `auto`, buckets 1–16, all-vram, bulk KV, one
KV-table selection per step, and **`max_seqs` 16, named** (`E4B_PAGED_MAX_SEQS=16`; main's default is now `auto`).
It runs Qwen3-30B-A3B @ `ad44e777bcd1…`, with the NF4 arena baked on the box by P39's `k8_bake.py`. grouped-nf4-gemm is
at **`6ee2e10`** (v0.43.0, a registered constant). The decode GEMV is at its default: the bandwidth route at Qwen3's
shapes, never forced. transformers is 5.17.0.

**The arms.** The arms differ only in the four fusion knobs:
- **D0:** all four `0`.
- **D1:** all four `auto` (named explicitly in both arms).

**The SANE read (the gate).**
- **Instrument:** Phase B's (`p115_quality.measure_phase`), at P115's registered bytes, on the default server built
  eager, at **one window per pass** (group 1). That is bucket 1, `T == 1`, the served one-request arithmetic: the only
  place the bandwidth GEMV and the folds meet, since `hot_residency._collapsed_grouping` sends `T > 1` to the M-tile.
  Phase C's SANE ran 12 windows as one group (`T == 12`), which would never reach the GEMV.
- **Size:** Phase C's, 12 wikitext windows of 512 prompt tokens and 128 teacher-forced positions.
- **Phases:** D0 scores R and saves its fp32 log-probs; D1 scores ON against them.
- **Gate:** Phase C's, unchanged: |mean d_ON| ≤ 0.02 nats and argmax agreement ≥ 0.95.
- **Engagement:** grouped-nf4-gemm's dispatch tally is recorded over each phase's measurement. Both phases must
  dispatch `bw_prmt32` and never dot-pad.

**The speed read (reported, not ruled).** Four fresh processes, ABBA: **D0a D1a D1b D0b** on the graph server.
- **Workloads:** P109's, at P109's registered bytes. W16 is 16 distinct 512-token wikitext prompts at once; W1 is row 0
  alone. 32 and 160 new tokens, 1 warm pass then 3 timed passes, p37's slope.
- **Records:** the dispatch tally after the build and after each workload.
- **Reported:** g1 = min(D1a/D0a, D1b/D0b) at W1 and g16 at W16. That is the stack's gain on top of the GEMV, which
  the default PR quotes in place of Phase A's 0.42.0 ratio.

### The rule (`bench/p115/p115d_reduce.py`, self-tested on 23 cases)

First rung that applies:
1. **VOID:**
   - a record is missing or not ok;
   - another e4b or grouped-nf4-gemm commit, or another model revision;
   - a speed arm with a bucket not captured;
   - the arms differ in prompts or lengths;
   - a slope is void;
   - a fusion census off its registration: D0 all zero; D1 `48 / 193 / [48, 48] / 48` on Qwen3, as Phases A/B read it;
   - the modes not the arm's;
   - the bandwidth GEMV not where the registration puts it: on the reading, every W1 speed workload and both SANE
     phases dispatch `bw_prmt32` and no dot-pad; on the proof's Granite, whose shapes the route does not cover, the
     scalar GEMV and no `bw_*`;
   - a SANE phase not at one window per pass, or short of its 12 windows.
2. **FUNCTION_FAIL:** a self-pair (D0b/D0a, D1b/D1a) decodes different tokens on any row, workload or length, or one
   arm's timed reps digest differently. D1 ≠ D0 is expected and reported.
3. **COMBINED_FAIL:** the SANE gate fails on the combination.
4. **COMBINED_SANE:** otherwise.

### Predictions (written before any data)

| # | prediction |
|---|---|
| D1 | engagement exact: D1's census `48 / 193 / [48, 48] / 48` and D0's zero; `bw_prmt32` on every W1 workload and both SANE phases, dot-pad never; every bucket captured |
| D2 | D0b ≡ D0a and D1b ≡ D1a bitwise; D1 ≠ D0 on some rows |
| D3 | SANE: bias within ±0.005 nats, argmax agreement ≥ 0.96 (Phase B's ON read 0.961 at T = 12; P116 read the GEMV within P110's bar at T == 1) |
| D4 | reported speed: **g1 ∈ [1.30, 1.60]**. The bandwidth GEMV's W1 step is about 8.2 ms (P116), and the stack removed about 3.2 ms at Phase A; its glue, router and q/k/v work does not overlap the GEMV's. **g16 ∈ [1.15, 1.30]** (Phase A's 1.23; T > 1 never reaches the GEMV) |
| D5 | COMBINED_SANE about 90 % |
| D6 | each speed arm ≤ 3 min; each SANE phase ≤ 15 min; peak ≤ 23 GiB |

### Consequence, registered now

- **COMBINED_SANE:** the family-scoped default PR may proceed on the allowlist above, quoting D4's ratios as the stack's
  gain at grouped-nf4-gemm 0.43.0 (the register row `e4b.serve.p115.fused-stack-combined.qwen3.5090.<date>`, SANE bias
  as its value, g1 and g16 in the claim text). Its changelog and `docs/SERVING.md` list per family what was read (Phase
  A/B's rule) and name `0` on each knob as the way back.
- **COMBINED_FAIL:** no default on Qwen3-30B-A3B. The combination is examined one knob per arm at T == 1 under its own
  registration before any flip. Qwen3.6 and Granite are unaffected by this phase, because the GEMV route covers neither
  family's shapes, so their entries may still proceed.
- **FUNCTION_FAIL:** a determinism defect, found first.
- **VOID:** no consequence; one rerun inside the ceiling, then an amendment.

### The premise and the proving rental

**Premise**, on the card before anything is fetched (rc 25). `tests/test_decode_graph_buckets.py`,
`tests/test_kv_step_select.py`, `tests/test_fused_glue_decode_graphs_gpu.py` and `tests/test_gemv_bw_served_gpu.py`:
**19 passed**, none skipped. They show:
- the bucket graphs replay as the padded eager step;
- the fused glue captures under the graphs;
- the bandwidth GEMV reaches the served `T == 1` route only and replays bitwise as eager.

**Proof** (`p115d-prove-<n>`): the whole box on Granite-3.1-3b-a800m, at 8 / 24 tokens, 1 rep and 12 windows × 32
positions. Its GEMV is the scalar route, since Granite's shapes are not in the table. A VOID from the reducer fails the
proof (rc 27). Its verdict is not a reading.

**Order on the box:**
1. Refusals, then install and the tripwire. The tripwire checks:
   - the pins;
   - the knobs opt-in at this commit and `auto` parsing;
   - `GNF4_GEMV_BW` at `auto` with Qwen3's two shapes in `_BW_SHAPES`;
   - the `T == 1` / `T > 1` boundary.
2. The reducer, box and quality self-tests, then the premise.
3. Fetch, bake and prompts.
4. The four speed arms, then SANE D0 and D1. SANE does not start when an arm failed.
5. The reducer.

### Budget and STOP

- **Proof:** one RTX 5090, **guard 0.75 h**, `--download-gb 7`, about $0.65 at the launcher's policy rate.
- **Reading:** one RTX 5090, **guard 1.5 h**, `--download-gb 61`, about $2.0 at the policy rate.
- **Ceiling:** Phase D's ceiling is **$4.00** (the proof, the reading and one rerun), inside P115's registered $10 hard
  stop. P115 has spent $2.455 over Phases A–C. Every run sits inside the owner's standing no-ask tier for a single run
  under $15; anything over $15 needs the maintainer lane's approval first.
- **STOP:** STOP-1 to STOP-4 as Phase C's: the refusals (18, 10, 15, 13 at < 150 GB, 16, 25), every time-left check
  inside its guard, no in-launch retry, and the driver refusing a dirty tree or a staged file that differs from
  `bench/p115/staged-d.sha256`.

**Receipts.** The run directory's `p115d/` is committed to `bench/p115/receipts/<run>/`:
- `arm_{D0a,D1a,D1b,D0b}.json`, `sane_off.json`, `sane_on.json`, `verdict_d.json`;
- `summary.txt`, `forensics.txt`, `versions.txt`, `prompts.json`, `bake.json`;
- `logs/` (added with `git add -f`) and `SHA256SUMS`.

The reference log-probs and the arena stay on the box.
