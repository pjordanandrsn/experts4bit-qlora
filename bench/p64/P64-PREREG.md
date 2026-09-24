# P64 — what the int4 serve lane's int8 activation step costs at decode: a decode-scored KL A/B over the same int4 bytes (registered 2026-09-23, before the run)

Record: [#709](https://github.com/pjordanandrsn/experts4bit-qlora/issues/709). Owner authorization: the work and a
later rental (Jordan, 2026-09-23, chat), within the standing caps. The rental is not part of this change.

## Question

Every int4 expert decode route quantises its activations per 32-block to int8 before the GEMV. `quant_x_rows`
(grouped-nf4-gemm `kernel/int4_b32.py:41-66`) is called at `experts4bit_qlora/engines/hot_residency.py:326`, `:359`
and `:371` (e4b `94842a2`, identical to the issue's `c2637e2` there). At T = 1 the route is `:371`: the
singleton-groups GEMV. Prefill and verify on the same store run bf16 activations: the dequant branch at `:376`.

- The quantise's **time** is priced: 0.153 / 0.318 ms per step at B = 1 / B = 16 (gnf4
  `kernel/PREREG-k17-fused-splitk-gemv.md:22`; P57's census reads 0.158 ms over 192 calls at B = 1, 96 of them
  experts).
- Its **quality** cost is not priced anywhere. The kernel's error figure is against "the fp32 reference of the same
  int4 values and int8 activations" (gnf4 `docs/solutions/int4-decode-gemv.md:64`): accumulation exactness, not
  the quantise. Every decode-scored KL on record bundles the step with the int4 weight grid.

**On the served stack, at B = 1, what does routing the T = 1 expert calls through bf16 activations over the same
int4 bytes change, in nats, against this instrument's own arithmetic-order floor?** Secondary: the same for the
attention projections' int8 step, and for both together.

**A correction the issue needs, found while building this.** The issue says the serving KLs of P44 and P59 are
decode-scored, "so the int8 step is inside them". That holds for P44's decode scorer at B = 1: every forward there
is T = 1. It does not hold for P59's expert calls:

- `kl_b16.py` scores B = 16 rows together, so each decode forward is T = 16.
- `hot_residency.DEVICE_GROUPING` is set only by `step_decomp.py`'s timed stages. Neither `serve_stack.py` nor
  `kl_b16.py` nor the P42 hook sets it.
- So in P59 the int4 experts took the host-grouped **dequant + bf16** branch (`:376`) at decode. The attention took
  K16 at 16 rows, also bf16 activations.
- P59's 0.0044 nats/token therefore contains **no int8 activation step at all**. The timed B = 16 arms it licensed
  run the device-grouped GEMV on int8 activations (`:326`).
- Shown by code reading here, and by a counted probe in the rehearsal below. B = 16 is outside this lane (see "What
  this lane cannot say").

## Instrument

**The lever:** `E4B_INT4_DECODE_A16=1`, a new environment flag, default off. It is exposed to a harness as
`hot_residency.DECODE_A16`, the `FORCE_SINGLETON_GROUPS` pattern: the environment sets the default at import and a
harness may flip it in-process.

- **On:** the int4-b32 store's singleton-groups calls take the prefill branch. That is T = 1 in every default
  configuration. Each routed expert is dequantised (`int4_pack_ref.dequant_int4_ref`) and bf16-matmul'd. The bytes
  are the same int4 bytes, the activations are bf16, and `quant_x_rows` is never called. There is no new kernel.
- **Not covered:** the device-grouped batched-decode routes (`:326`, `:359`) and the MXFP4 store.
- **Eager only:** the dequant loop reads the expert ids on the host, so under CUDA-graph capture the flag refuses
  with a sentence.
- **Off:** byte-identical to today. `tests/test_int4_decode_a16.py` pins both states on CPU with the int4 kernels
  stubbed by torch references and the real pack and dequant. With the flag on, T = 1 never calls `quant_x_rows` or
  the GEMV, dequantises 2 × top-k per layer, and equals the prefill branch on the same rows bit for bit; T > 1 is
  unchanged either way. With the flag off, T = 1 is exactly `quant_x_rows` → GEMV per projection. It also pins the
  device-grouped route's non-coverage and the capture refusal.
- **Off, against `main`:** a scratch comparison of `main`'s `_fused_over_stack` against this branch's, flag off,
  over 25 randomized cases on five routes (int4 T = 1, int4 prefill, int4 batched-decode GEMV, NF4 grouped, NF4
  singleton), is `torch.equal` with the same kernel-call sequence in every case.

**The scorer:** P59's, reused unchanged. `bench/p59/kl_b16.py:batched_teacher_forced` is called with ONE row at a
time (`bench/p64/kl_a16.py`), so every decode forward is T = 1.

- Each row prefills its first 384 tokens in chunks of 128 (the harness's prefill step; T > 1, the same dequant +
  bf16 route in every pass).
- It then decodes the remaining 128 positions one token per forward, with a carried KV cache and the ground-truth
  token fed back.
- Rows: the harness's own window, `step_decomp._k8_window`, 16 rows × 512 tokens, on **two texts**:
  - wikitext-2 test (P59's rows, digest `f67e7e4d…`);
  - C4 validation shard 1, K8's `c4val1`.
- **2,048 decode positions per text**, the count P59 scored and the length of K8's window.
- Logits are kept per pass and text. KL is kl_fidelity's (fp64, full vocabulary, token-weighted), read through
  `p59_reduce.kl`. NLL is the ground-truth next token's, at the 127 of 128 positions per row that have one (2,032).

**The stack:** Qwen3-30B-A3B @ `ad44e777…`, the family P59 scored and the one with a licensed int4 serving position.

- The expert pack is **P55x's licensed pack**, `sha256:0c9955a9…`. The box rebuilds it with P55x's command
  verbatim: `step_decomp.py` licbuild, the streamed 64k C4 recipe, calibrated attention, folds, and the wikitext K8
  from the live stores (P55x read 6.36709). Four builds on ≥ 3 boxes gave these bytes.
- The fingerprint is the check. A different fingerprint is recorded, and the read is labelled "not the licensed
  pack".
- The pack is loaded by fingerprint (`E4B_INT4_ARTIFACT_DIR` + `E4B_INT4_EXPECTED_FINGERPRINT`), with C4-calibrated
  int4 attention (192 `Int4Linear`), round-1/2 folds and the router epilogue. That is bo6c's configuration, built
  by P44's `serve_stack.build_served_model`, `--no-fuse-qkv` as the licence rows were.
- **One build, one process, every int4 pass.** The calibrated attention is re-derived at every load (#674), so a
  process per pass would compare two attention packs. Here every pass reads the same bytes by construction, and the
  attention pack's sha256 is recorded.

**Proof of execution, per pass and text.**

- **Counted:**
  - `int4_b32.quant_x_rows` and `int4_pack_ref.dequant_int4_ref`, patched after the build: only the expert
    dispatch imports them at call time;
  - every `Int4Linear`'s own `_qx`;
  - split by phase: a forward with one token is decode.
- **Required at decode, exactly:**

  | pass | expert quantises | expert dequants | attention quantises |
  |---|---|---|---|
  | a8 | 2 · L per step | 0 | 192 per step |
  | a16 | 0 | 2 · L · top-k per step | 192 per step |
  | a16_all | 0 | 2 · L · top-k per step | 0 |

- **Refused otherwise:** a pass whose counts differ is refused by the scorer (rc 40) and by the reducer.

## Arms

| pass | experts at T = 1 | attention at one row | prefill chunk | role |
|---|---|---|---|---|
| `a8` | GEMV on int8 (shipped) | GEMV on int8 (shipped) | 128 | the served stack |
| `a16` | dequant + bf16 (`DECODE_A16`) | GEMV on int8 | 128 | **the lever** |
| `a16_all` | dequant + bf16 | cached-bf16 branch (`Int4Linear.GEMV_ROWS_MAX = SMALLM_ROWS_MAX = 0` for the pass, lane-side) | 128 | the whole int8 step |
| `a8_rep` | as `a8`, after the toggles | as `a8` | 128 | determinism control |
| `a8_pc64` | as `a8` | as `a8` | 64 | arithmetic-order floor |
| `a8_pc384` | as `a8` | as `a8` | 384 (one forward) | arithmetic-order floor |
| `nf4` | NF4 (own process, no folds) | bf16 | 128 | anchor (informational) |

- **Order of the clock:** the primary pair on both texts first, then `a8_rep`, `a8_pc64`, `a16_all`, `a8_pc384`,
  then the anchor.

**What `a16` is, stated so it is not over-read.** It is the prefill branch's arithmetic, not an exact reference.

- The dequantised weight is rounded to bf16: a 4-bit level times an fp16 scale has up to 14 significant bits
  against bf16's 8, about 0.2 % rms relative.
- The GEMV keeps the weights exact and rounds the activations instead: per-32 int8, about 0.5 % of the row's σ for
  Gaussian rows.
- By that estimate a16's own error is about half a8's, in rms. So the A/B is the served decode arithmetic against
  the served prefill arithmetic on the same bytes, which is what #709 asks. It is not int8 against exact.
- `a16_all`'s attention branch has the same arithmetic class: bf16-rounded dequantised weight, cuBLAS. K16 at
  M = 1, the attention speed candidate #709 names, sits within one bf16 ulp of it by its own contract. K16 is not
  run here.

## Predictions (written before the data)

Every KL below is nats/token over 2,048 decode positions per text. F(t) is the mean of the two floor pairs' KL.

- **P1 — determinism.**
  - Predicted: `KL(a8 ‖ a8_rep)` is exactly 0 on every position of both texts.
  - Basis: P59b's fresh-process rebuild read exactly 0.
  - Refuted by any non-zero position. Nothing is then read (validity).
- **P2 — the flag touches no T > 1 forward.**
  - Predicted: the prefill-last logits of `a8`, `a16`, `a16_all` and `a8_rep` are bit-identical on both texts.
  - Basis: by construction.
  - Refuted: a design fault, and nothing is read.
- **P3 — the floor.**
  - Predicted: F(t) ∈ [0.001, 0.010] on both texts, central ~0.004.
  - Basis: P59b's reorder through this scorer on Qwen3 read 0.0044 (fused q/k/v through the 128-row prefill,
    wikitext rows). METHODOLOGY §13.1's chunk-vs-full floor for Qwen3 is KL 0.0035.
  - Refuted: outside the band. Reported; not voiding unless F = 0 (then the on-record 0.0044 stands in, stated).
- **P4 — the expert int8 step (primary).**
  - Predicted: `G_exp(t) = KL(a16 ‖ a8) ≤ 2 · F(t)` on both texts (INDISTINGUISHABLE), central ~0.002–0.004, top-1
    ≥ 0.97.
  - Basis, three:
    - gpt-oss's `store_r12` reads **0.0019** nats/token, top-1 0.981 (P44-b). That is the same per-32 int8 quantise
      on a single-row expert GEMV, bundled with fold reorders, against a bf16-activation reference over the same
      weight bytes. It sits 13× below gpt-oss's own 0.0256 floor.
    - The int4 grid, with its int8 decode step, costs +0.007 ppl on experts at the 8k gate (about +0.001 nats), per
      `int4_b32`'s docstring.
    - The per-element estimate above.
  - Refuted by `G_exp > 2F` on either text.
- **P5 — the sign.**
  - Predicted: |dNLL_exp(t)| = |NLL(a8) − NLL(a16)| < 0.0095 nats on both texts (the family's registered K8 floor),
    and its paired row-bootstrap 95 % interval includes 0.
  - Refuted by an interval excluding 0 with |dNLL| ≥ 0.0095.
- **P6 — attention.**
  - Predicted: `G_attn(t) = KL(a16_all ‖ a16) ≤ 2 · F(t)` on both texts.
  - Basis: the attention grid, with its int8 step, reads −0.006 ppl at the 8k gate (`int4_b32`).
- **P7 — anchor (informational, not gated).**
  - Predicted: `KL(nf4 ‖ a8)` ∈ [0.03, 0.10], and |KL(nf4 ‖ a16) − KL(nf4 ‖ a8)| ≤ 0.005.
  - Basis: P59b's anchor with RTN int4 was 0.067.
- **P8 — the flag's price (informational, never a speed number).** a16's seconds per decode step ≥ 1.5× a8's: the
  dequant loop runs per routed expert per step.

## Decision rule (nats, against a measured floor)

Per text t and statistic G ∈ {G_exp (primary), G_all, G_attn}:

- **Classes:**
  - **BELOW FLOOR:** G ≤ F;
  - **NOT DISTINGUISHABLE:** F < G ≤ 2F;
  - **DISTINGUISHABLE:** G > 2F.
- **Why factor 2 over a measured floor, not a constant:** METHODOLOGY §13.1. Two correct arithmetic orders of an
  MoE model disagree because rounding flips router choices, and "a delta below the floor means indistinguishable —
  never 'a small cost'". The floor here is measured in this instrument, on this box, per text. It is not the 0.01
  constant a pre-registered KL gate was falsified by (METHODOLOGY §13).
- **MATERIAL on t:** DISTINGUISHABLE, and dNLL(t) > F_NLL(t) (the floor pairs' own mean |dNLL|), and dNLL's 95 %
  interval excludes 0. The int8 step then makes the model measurably worse, not merely different.
- **Lane verdict per statistic:**
  - MATERIAL if material on either text;
  - else DISTINGUISHABLE, NOT MATERIAL if distinguishable on either;
  - else INDISTINGUISHABLE.
- **Validity first:** P1, P2, the census and identical rows. Any failure → **nothing is read**. The receipt is kept
  and the fault filed.

What follows:

- **G_exp MATERIAL** → the speed half is registered as its own lane: an A16 expert decode route, with this lane's
  G_exp as its quality bar. #709's "experts have none yet" candidates are a small-M int4 GEMM over the expert store
  (K16's in-register dequant at M = 1) or folding the quantise out. If G_attn is also MATERIAL, K16 at M = 1 for
  attention is registered with it. No default moves here; the flag stays an instrument.
- **DISTINGUISHABLE, NOT MATERIAL** → a register row records the int8 step as a distribution change in nats that
  does not cost NLL on these texts. No A16 route is motivated on quality grounds, and #709 closes with the receipt.
- **INDISTINGUISHABLE** → a register row: the expert int8 step is below Qwen3's arithmetic-order floor on both
  texts. #709 closes, W4A8 decode stays, and its 0.153 ms/step is its only cost on record.
- **Pack ≠ licensed fingerprint** → the verdict is a reading on the box's pack, and says so. Nothing is attached to
  the licensed stack.

## What this lane cannot say

- **Nothing about B = 16.** The flag does not cover the device-grouped GEMV (`:326`), and P59's B = 16 scorer never
  ran that route (above). A B = 16 read needs a scorer that sets `DEVICE_GROUPING` and a flag that covers `:326`:
  its own lane.
- **No bf16 reference.** Qwen3-30B-A3B in bf16 is 61 GB and does not fit a 5090. The question is a difference
  between two arithmetic paths of one stack, P59's framing. NF4 is the anchor, disclosed as such.
- **Not the paged fp8 KV path.** Like P44-b and P59, the scorer runs the served stack's MoE and projections under
  HF's bf16 cache, not `step_decomp`'s paged decode.
- **Scope:** one family, two texts, 2,048 positions per text; nothing about speed.

## What was run before this registration was final (disclosed; none of it is the reading)

**Rehearsal on the NAS RTX A2000 12 GB (sm_86)**, `bench/p64/rehearsal-a2000/` (README: NOT a reading).

- **The runner itself,** `p64_run.sh`, with its knobs off the registered defaults so it marks itself REHEARSAL:
  - OLMoE-1B-7B-0924 **base** (P44's family, the NAS copy; P44 pins the Instruct model);
  - calibration 8 × 512 C4 tokens, Hessian budget 2 GB;
  - install skipped (the branch tree on `PYTHONPATH`, gnf4 v0.33.0 = the registered cut);
  - RTX A2000 accepted as the class;
  - the licensed fingerprint unset.
- **Image:** `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel` (torch 2.8.0+cu128, triton 3.4.0, transformers 5.16.1).

REHEARSAL_RESULTS

## Box and cost

**Two rentals, in order.** The compute rule in force says a guard over 1 h needs a proving rental first (≤ $0.15,
≤ 10 min). This lane cannot fit 1 h: the pack build alone is ~30 min. So:

1. **`p64-prove-1`, the proving rental:** one RTX 5090, **10 min guard**, ≈ 6 min expected, **≈ $0.07**
   (hard stop $0.15). `p64_drive.sh` with `P64_PROVE=1`: the same staging, nonce handshake, class and disk
   refusals, pinned install, tripwire, scorer self-test and K0 as the reading, then
   `kl_a16.py --prove-flag` and a 50 MB HF CDN range probe. `--prove-flag` checks the flag on the box's own sm_120
   kernels at Qwen3's expert shapes:
   - off runs `quant_x_rows` + the GEMV;
   - on runs 2 × 8 dequants and equals the prefill branch bit for bit;
   - the off route still captures and replays bit-identical to eager;
   - on refuses under capture.

   It exits 0 with `PROVED`. There is no model, no pack and no KL. Download: the pip packages only (~0.5 GB) plus
   50 MB. It proves the path the reading takes, on the class, before a 2 h guard is spent. A failure there (any rc
   but 0) stops the lane before the reading rents anything.
2. **`p64-5090-1`, the reading:** launched only after `p64-prove-1` returns rc 0 with its receipts fetched.

- **Box:** one RTX 5090, Vast verified/secure, image `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`. No power floor:
  nothing is timed.
- **Guard: 2.0 h.** Estimate ≈ ESTIMATE_MIN min, **≈ $ESTIMATE_USD at $0.66/h**. The lane ceiling is $2 and the hard
  stop $3, both under the $35 cap.
- **Expected shape:**
  - install ~8 min, K0 ~1, fetch 6–15, bake ~5, prompts ~1;
  - the pack build ~30 (P55x's licbuild: 1,832 s);
  - the served process: load + attention calibration ~5, then the passes (TIMING_BASIS);
  - anchor ~4, reduce ~3.
- **Pins:**
  - e4b = the merge of this change (the launch manifest's `heads.e4b`);
  - grouped-nf4-gemm `5ca1897585f9f456f99ea504b2a1be0ea91db496` (v0.33.0, the consumer CI pin);
  - transformers 5.16.1, bitsandbytes 0.50.1;
  - the P39/P42/P44/P59 pieces staged byte-identical (`staged.sha256`, `tests/test_p64_staged_pin.py`).
- **Downloads (all on the box, from the internet):**
  - `Qwen/Qwen3-30B-A3B` @ `ad44e777…`: safetensors, json and tokenizer files, ~61 GB;
  - datasets: `allenai/c4` `en/c4-validation.00000-of-00008.json.gz` (calibration, ~40 MB) and
    `…00001-of-00008.json.gz` (the c4val1 rows, ~40 MB), plus `Salesforce/wikitext` `wikitext-2-raw-v1` (a few MB);
  - pip: e4b and grouped-nf4-gemm from git at their pins, transformers, bitsandbytes, datasets, accelerate,
    sentencepiece, tiktoken, safetensors, huggingface_hub, ~0.5 GB;
  - the HF token is staged from `~/.config/hf/token` as a file, never on a command line.
- **Nothing large leaves the box:** the logits (~0.6 GB per pass and text) and the pack's 15.2 GiB of payloads stay.
  The pack is P55x's, already held.
- **Disk floor:** 200 GB (P55x's working set plus ~9 GB of logits).

**STOP rules.** Refusals come before anything is fetched: a card that is not a 5090 (rc 15); under 200 GB free on
`/root` (rc 13, host-limited). A build with no calibration chunk in 1,500 s is killed (rc 30, host-limited). A step
that cannot finish 10 min before teardown is skipped, never shortened, and the scorer skips passes the same way. No
second box on a disappointing result.

## Receipts and exit codes

- **Fetched:**
  - `summary.txt` (KNOBS line, pack line, every pass's census line), `forensics.txt`, `versions.txt`, `k0.json`;
  - `pack.json`; `artifact1/manifest.json` + the identity/assignment payloads (not the layer payloads);
  - `build_ppl_wikitext.json` (the build's K8, beside P55x's 6.36709);
  - `prompts_{wikitext,c4val1}.json`;
  - `out/build.json`, `out/*.census.json`, `out_nf4/*.census.json`;
  - `RESULTS-p64-generated.md` and `p64_rep.json` (the reducer ran on the box);
  - `logs/`, `work/bake.json`, and the launcher's teardown proof.
- **Filed:** into `bench/p64/receipts/` beside the register rows `e4b.serve.p64.qwen3.*`, with `RESULTS-p64.md`
  quoting only them.

| rc | meaning |
|---|---|
| 0 | the registered read can be made (validity VALID, the primary pair + determinism + a floor sample on both texts); in the proving run, every proving check passed |
| 9 | stage / install / tripwire |
| 10 | no CUDA |
| 11 / 12 | fetch / bake failed |
| 13 | disk below floor (host-limited) |
| 15 | wrong GPU class |
| 16 | K0 controls failed (no KL row) |
| 19 | prompt dump failed |
| 20 | no pack that verifies (`verify_artifact`) after the build, or no time left for it |
| 21 | the scorer's CPU self-test failed |
| 27 | proving run only: the real-kernel flag check failed |
| 30 | build host-limited (no calibration chunk in time) |
| 40 | a pass's counts are not the registered ones (the lever did not engage) |
| 41 | validity not met (P1, P2, census or rows) |
| 42 | a required pass is missing or not engaged |
| 43 | the reducer failed |
| 44 | int4 leaked into the NF4 anchor |
| 78 | configuration refusal (a required variable missing or malformed) |
| 130 | interrupted |

- **Controller codes** (`p64_drive.sh`): 20–25 are staging, start, fetch, stale nonce, no `TP_DONE`, and lane died
  on the box. A non-zero lane rc is passed through.
- **Handshake:** B393's. `P64_RUN_NONCE` first; `P64_EXIT_CODE.<nonce>` + `TP_DONE.<nonce>` on every exit;
  `P64_SUCCESS.<nonce>` only at rc 0.

Amendments, dated, go below this line before any data they touch.
