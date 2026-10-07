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
  (OFF ≡ OFF). IDENTITY ON against OFF is reported, not gated.
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
