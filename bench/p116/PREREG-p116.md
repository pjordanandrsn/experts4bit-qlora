# P116 — does the bandwidth-targeted NF4 decode GEMV (`GNF4_GEMV_BW=1`) decode the default `serve_paged` server faster at one request, at no measurable quality cost? One RTX 5090

Lane number claimed by `prereg/p116` (pushed 2026-10-07T18:58:20Z). Issue: e4b#1313 (single-stream decode on the default
NF4 server). The kernel: grouped-nf4-gemm#500 (`_gemv_nf4_bw`, opt-in behind `GNF4_GEMV_BW`). Its microbenchmark:
grouped-nf4-gemm K33 (#501 registered, #502 the read), **LEVER**.

## Why now

**What K33 measured** (`k33-5090-1`, one RTX 5090, grouped-nf4-gemm `5a60c37`). Qwen3-30B-A3B's single-row expert
projections ran over all 48 layers in one CUDA graph per projection: 8 rows, the top-8 experts.
- `bw_prmt32` took 0.961 ms for the gate_up + down pair against dot-pad's 2.818 ms (×0.341), at 0.74 / 0.63 of the copy
  floor.
- `prmt32` was bitwise the exact tree decode, and the tolerance contract held.
- With PDL on (`bw_pdl`, the served default caps PDL at 8 rows) the pair was 1.010 ms.
- K33's registered consequence is this lane: the served read, under P110's quality bar.

**Where the served server reaches the switch.** The graph server sets `hot_residency.DEVICE_GROUPING`.
`hot_residency._collapsed_grouping` then sends a decode step at **T == 1** to singleton groups: `gemm_4bit_grouped`'s
single-row decode branch, where dot-pad runs today and `_gemv_nf4_bw` runs with the switch on. A step at **T > 1** takes
device grouping, the captured M-tile GEMM, which the switch does not touch.
- **W1 (one request) is therefore the subject.**
- **W16 (16 requests at once) is a control**: no engagement, so no change is expected.
- The same holds for P110's teacher-forced instrument at its group of 12 (T = 12), so the quality read here runs one
  window per pass.

**What the step can gain.** SV2's census put the served NF4 expert GEMV at 2.469 ms of the graphed Qwen3 B=1 step. K33's
pair saves about 1.86 ms of kernel time per step (2.818 − 0.961), or about 1.8 ms with PDL as served. K6b measured how
much of a kernel win reaches the served step: about 0.55. That puts roughly 1 ms on P111's 9.05 ms W1 step, about ×0.89,
so g1 near 1.12 (the maintainer's basis in the K33 read's review). Q3's band spans a transfer between about 0.35 and
0.9.

## The subject

The shipped default server at the launch commit:
- graphs `auto`, `max_seqs` 16, buckets 1–16, all-vram, bulk KV, one KV-table selection per step;
- the four fusion knobs at their launch-commit default, recorded per arm, and the same in every arm;
- `GNF4_PDL` at its default (on, capped at 8 rows).

It runs Qwen3-30B-A3B @ `ad44e777bcd1…`, with the NF4 arena baked on the box by P39's `k8_bake.py`. grouped-nf4-gemm is
at **`5a60c37`** (0.42.0 + #500 + #501, K33's measured cut, a registered constant in `p116_run.sh` and the reducer), with
transformers 5.17.0 and the image's torch 2.8.0.

## The arms

Four fresh processes, ABBA: **B0a B1a B1b B0b**. The arms differ only in grouped-nf4-gemm's decode GEMV switch, which is
read at call time:
- **B0:** `GNF4_GEMV_BW` unset, the shipped default. Dot-pad runs at Qwen3's shapes on ≥ 160-SM parts.
- **B1:** `GNF4_GEMV_BW=1` with `GNF4_GEMV_BW_PLAN` set to **K33's selected plan per shape**, from
  grouped-nf4-gemm `kernel/receipts-k33/5090/k33.json`. Qwen3 gate_up 1536 × 2048 runs BLOCK_N 16, KC 1024, 4 warps,
  split-K 1. Down 2048 × 768 runs 16 / 256 / 4 / 1. Granite's and OLMoE's plans are listed in `p116_box.BW_PLANS`.
  K33's default plan is 12 % slower on Qwen3's gate_up, so the lane reads the plans that would ship. On a DEFAULT read the
  kernel repository writes them in as a shape table.

**Workloads** are P109's, from `p109_box.py` at its registered bytes:
- W16: 16 distinct 512-token wikitext prompts at once. W1: row 0 alone.
- 32 and 160 new tokens, 1 warm pass then 3 timed passes, p37's slope.

**Engagement** is read from grouped-nf4-gemm's dispatch tally (`nf4_grouped.dispatch_counts()`), recorded after the build
and after the runs:
- B1 dispatches every single-row decode GEMV to `bw_prmt32`, with none to dot-pad, the scalar GEMV, the tree or split-K.
- B0 dispatches none to `bw_*` and some to dot-pad.

## The quality read

P110's teacher-forced instrument (`p115_quality.measure_phase`, at P115's registered bytes) runs on the default server
built eager, **one window per pass**: bucket 1, T == 1, the served W1 arithmetic. It covers both texts (wikitext-2 test
and c4val1), **24 windows each** of 512 prompt tokens and 128 teacher-forced positions.
- **OFF (B0)** scores R and saves its fp32 log-probs. It also scores `rep` (window 0 again), `chunk` (256-token prefill
  chunks) and `mutant_scale` (the attention scale halved).
- **ON (B1)** scores ON against R.
- **Floor:** `half` does not exist at one window per pass. The floor is drawn from `chunk`, and from `rep` when R did not
  repeat bit for bit.
- **Bars, P110's:**
  - bias_ON ≤ B_floor + 0.01 and spread_ON ≤ 2 × max(S_floor, 0.005), on both texts;
  - |K8 Δppl| ≤ 0.05 on wikitext, while c4val1's K8 is reported.
- **Why 24 windows:** every decode step in this instrument is eager. At one window per pass, P115's 48 windows would take
  about an hour on this box. 24 keeps the read inside the guard. The floor-relative bar scales with the noise.

**Integrity**, or VOID:
- every pass has (128 − 1) × 48 decode attention calls and 127 eager bucket-1 steps, with no padding and no replay;
- the glue-kernel and q/k/v call counts are equal between each R pass and its ON pass;
- the dispatch rule above holds on each phase's measurement.

## The rule (`bench/p116/p116_reduce.py`, self-tested on 24 cases)

First rung that applies:
1. **VOID:**
   - a record is missing or not ok;
   - another e4b or grouped-nf4-gemm commit, or another model revision;
   - a bucket was not captured;
   - the arms differ in prompts, lengths or fusion census;
   - a slope is void;
   - the engagement rule fails, on a speed arm or a quality phase;
   - the quality integrity fails;
   - the scale mutant passes the quality bar.
2. **NOISY:** a self-pair, B0b/B0a or B1b/B1a, falls outside [0.96, 1.04] on either workload.
3. **FUNCTION_FAIL:** B0b ≠ B0a or B1b ≠ B1a on any row, workload or length, or one arm's timed reps digest differently.
   B1 ≠ B0 is expected at W1, from another reduction order, and is reported.
4. **QUALITY_FAIL:** either text fails P110's bias or spread bar, or wikitext's K8.
5. **SLOWER:** g1 = min(B1a/B0a, B1b/B0b) at W1 **< 1.03**, or g16 at W16 **< 0.99**.
6. **DEFAULT_ON:** otherwise.

## Predictions (written before any data)

| # | prediction |
|---|---|
| Q1 | engagement exact: B1 dispatches only `bw_prmt32`, B0 only dot-pad; every bucket captured in every arm |
| Q2 | B0b ≡ B0a and B1b ≡ B1a bitwise; B1 ≡ B0 at W16 (the switch is never reached there); B1 ≠ B0 on some W1 rows |
| Q3 | **g1 ∈ [1.06, 1.22]**: the W1 step falls by about 1.0–1.6 ms from P111's 9.05 ms |
| Q4 | **g16 ∈ [0.99, 1.01]** |
| Q5 | self-pairs within [0.98, 1.02] |
| Q6 | wikitext ON bias within ±0.003 nats and \|K8 Δ\| ≤ 0.02; c4val1 within ±0.005 nats; the chunk floor ≤ 0.003; the mutant > +0.3 nats |
| Q7 | DEFAULT_ON, about 70 % |
| Q8 | speed arms ≤ 3 min each; quality OFF ≤ 25 min and ON ≤ 10 min; peak ≤ 23 GiB |

## Consequence, registered now

- **DEFAULT_ON:**
  - grouped-nf4-gemm fills `_BW_SHAPES` with Qwen3-30B-A3B's two shapes, writes K33's plans in as a shape table and makes
    `GNF4_GEMV_BW=auto` its default on ≥ 160-SM parts, with the `test_m3_defaults` trio and a release;
  - experts4bit-qlora floors `[fast]` on that release, adds a `compatibility` record in `docs/system-manifest.json` and
    updates `docs/SERVING.md`;
  - the register row is `e4b.serve.p116.gemv-bw.qwen3.5090.<date>` (g1).
  - **Scope:** speed and quality are read on Qwen3-30B-A3B NF4 at one request. Granite and OLMoE run the scalar GEMV at
    B=1 today, and K33 read 0.22–0.27× there. Their served read is its own lane.
- **SLOWER:** the switch stays opt-in, and the ratios are recorded.
- **QUALITY_FAIL:** opt-in; the failing text is examined before any second round.
- **FUNCTION_FAIL:** a determinism defect to find first.
- **NOISY or VOID:** no consequence; one rerun inside the ceiling, then an amendment.

## The premise and the proving rental

**Premise**, on the card before anything is fetched (rc 25):
- grouped-nf4-gemm's `kernel/test_nf4_gemv_bw.py` at `5a60c37`, compiled, **27 passed**, none skipped. That is the
  kernel's contract on this card, from a source clone of the pinned commit.
- `tests/test_decode_graph_buckets.py`, `tests/test_kv_step_select.py` and the new `tests/test_gemv_bw_served_gpu.py`,
  **18 passed**, none skipped. The new file shows four things on a tiny all-hot NF4 store under the served collapse with
  device grouping:
  - T == 1 with the switch on dispatches only `bw_prmt32`, and with it off none to `bw_*`;
  - both routes read the reference within the served tolerance;
  - the switch's T == 1 step captures in a CUDA graph and replays bitwise as eager;
  - T > 1 never reaches the decode GEMV.

**Proof** (`p116-prove-<n>`): the whole box on Granite-3.1-3b-a800m, at 8 / 24 tokens, 1 rep, and 12 windows × 32
positions. Its incumbent is the scalar GEMV. A VOID from the reducer fails the proof (rc 27). Its verdict is not a reading.

**Order on the box:**
1. Refusals, then install and the tripwire. The tripwire checks the pins, the switch's API, `_BW_SHAPES` empty and the
   switch off, `prmt32` as the decode, K33's plans parsing, and the T == 1 / T > 1 boundary in `_collapsed_grouping`.
2. The three self-tests, then the premise.
3. Fetch, bake and prompts.
4. The four arms, then quality OFF and ON. Quality does not start when an arm failed (STOP-5).
5. The reducer.

## Budget and STOP

- **Proof:** one RTX 5090, **guard 0.75 h**, about $0.7 at the launcher's policy rate.
- **Reading:** one RTX 5090, **guard 1.5 h**, `--download-gb 61`, about $1.95 at the policy rate. Expected about 55 minutes:
  install and premise 5, fetch 6–10, bake 2, prompts 1, four arms about 12, quality about 30.
- **Lane ceiling $3.00, hard stop $4.00.** That is the owner's per-lane cap for this program: the proof, the reading and one
  rerun. A launch goes ahead only while the lane's actual spend plus that run's launcher estimate is ≤ $4.00.
- **STOP-1:** the refusals run before any install: CUDA unusable 18 (the host floor), dud box 10, card class 15, disk
  < 150 GB 13, host RAM < 60 GiB 16, premise 25.
- **STOP-2:** every time-left check fits inside its guard, enforced by `tests/test_p116_staged_pin.py`.
- **STOP-3:** a VOID or NOISY reading is not retried inside the same launch.
- **STOP-4:** the driver refuses a dirty tree or a staged file that differs from `bench/p116/staged.sha256`.
- **STOP-5:** the quality phase does not start when a speed arm failed; the reducer VOIDs.

## What was seen before this page (stated, not hidden)

- **No P116 data exists.** No served run has ever set `GNF4_GEMV_BW=1`.
- **K33's 5090 numbers** are quoted above from grouped-nf4-gemm#502's receipts.
- **Locally (CPU):** the box's self-test (14 cases) and the reducer's (24) pass. `tests/test_p116_staged_pin.py` passes.
  `tests/test_gemv_bw_served_gpu.py` skips without CUDA.
- **On the RTX A2000** (sm_86, the NAS card; correctness only under #1133, no timing taken): with grouped-nf4-gemm
  `5a60c37`, `tests/test_gemv_bw_served_gpu.py` **5 passed**. The run exercised the `prmt32` decode, T == 1 engagement and
  the bitwise capture replay. Its neighbours `tests/test_capture_quantized_moe.py` and `tests/test_hot_residency.py`
  passed unchanged (19). The first attempt failed at fixture setup, because `monkeypatch.setitem` cannot patch a
  one-element list. The fixture now saves and restores the two grouping flags itself.

## What this lane cannot say

- **Nothing about B=16 or any T > 1 step:** the switch is not reached there.
- **Nothing about other families or other cards**, or about the tree decode served.
- **Nothing about PDL against no PDL with the switch on:** both arms run PDL at its default.

## Receipts

The run directory's `p116/` is fetched and committed to `bench/p116/receipts/<run>/`:
- `arm_{B0a,B1a,B1b,B0b}.json`, `quality_off.json`, `quality_on.json`, `verdict.json`;
- `summary.txt`, `forensics.txt`, `versions.txt`, `prompts.json`, `logs/` and `work/bake.json`;
- `SHA256SUMS`.

The reference log-probs (`work/ref`), the arena and the gnf4 source clone stay on the box. `RESULTS-p116.md` is written
from those files.
