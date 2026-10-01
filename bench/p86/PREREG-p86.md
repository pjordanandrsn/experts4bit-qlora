# P86 — where do e4b's and vLLM's decode steps go, kernel by kernel, on one box? (registered 2026-09-30, before any run)

Issue: experts4bit-qlora#564 (the B=16 gap, decomposed). P58 (`bench/p58/RESULTS-p58.md`) measured vLLM 0.30.0 at
**1.087×** e4b's int4 stack at B=1 and **1.396×** at B=16, end to end on one RTX 5090.

We know where e4b's B=16 step goes. P57's census on P58's stack: 11.58 ms per step, of which
- the int4 expert GEMV takes 6.34 ms, about 77 % of its byte roofline at the measured routing (#564; about 1.4 ms of
  headroom);
- the attention projections take 1.30 ms;
- routing and gather glue take 0.95 ms;
- fp8 paged attention takes 0.68 ms;
- the output head's bf16 GEMM takes about 0.4 ms;
- the rest is reduce, quantize, norm folds and launch gaps.

**We do not know where vLLM's 8.3 ms goes.** P37 and P58 timed vLLM end to end only. So the 3.3 ms gap cannot be
assigned. vLLM may win on the expert kernel (Marlin MoE), where our GEMV has measured headroom, or on everything
around it: attention projections, glue, norms fused by `torch.compile`, launch gaps. The next kernel lane depends on
which. This lane answers that, descriptively.

Rule: the owner's standing no-ask tier for a single run under $15 (2026-09-26), with the usual mechanics:
- this page merged before the launch;
- a proving rental before a guard over 1 h;
- receipts and ledger rows;
- proven teardown.

The owner asked to push ahead on throughput (2026-09-30).

## Question

On one RTX 5090, with the same prompt token ids, how many ms per decode step does each engine spend in each
**role**, and which role has the largest e4b − vLLM gap at B=16?

The roles:
- **quantized linear:** experts and attention projections;
- **routing and gather glue;**
- **attention and the KV write;**
- **norm, rope and activation;**
- **dense bf16 GEMM:** the output head;
- **other kernels;**
- **host and launch:** the step minus its kernels.

## Instrument

**The box:** one RTX 5090, Vast verified/secure, `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`, ≥ 200 GB of disk.
- **The driver must be ≥ 580.** vLLM 0.30.0's wheels are torch 2.13 on CUDA 13.0, which P58 ran on driver 580.119.
  The runner refuses an older driver with rc 18, before any install.
- Any CPU vendor: both engines run on the same box, so this lane compares no float with another machine's.

**The e4b side.**
- **Stack:** the launch commit (e4b 0.37.8 plus this lane's files) and grouped-nf4-gemm `9407d49` (v0.33.7).
- **Env:** P58's (RTN int4 experts, uncalibrated int4 attention, glue r1/r2, router epilogue), with fused q/k/v at both
  batches (P54's B=1 default, P59's B=16 default).
- **Harness:** P58's exact pieces (`bench/p39/step_decomp.py`, `bench/p42` hook).
- **Timing:** the graph-replay window, `step_ms_clean`.
- **Census:** P42's protocol. `--replay-profile-out` profiles 8 replays after the timed window, outside it, and the
  table is parsed by `bench/p42/p42_reduce.py`.

**The vLLM side: 0.30.0, P58's comparator,** serving `Qwen/Qwen3-30B-A3B-GPTQ-Int4` @ `9b534e43` (Marlin) with P37's
graph_r1 settings.
- **Timing:** P37's slope arm, unchanged (`bench/h2h-20260905/p37/p37_vllm.py`): generate 32 and 128 tokens and divide
  the extra wall by the extra tokens, min-of-3.
- **Census:** `p86_vllm_census.py` applies the same slope to the GPU's kernel record.
  - The engine runs in-process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`), so `torch.profiler` (CUPTI) records the model
    runner's kernels, including the kernels of CUDA-graph replays.
  - Each length is warmed once, then generated once under the profiler.
  - Per kernel name, (128-token total − 32-token total) / 96 is its time per decode step. Prefill, load and prompt
    scheduling cancel.
  - Only kernels with **more calls in the long run** count as decode. A kernel with the same count in both runs ran
    only in prefill or setup, and its difference is run-to-run jitter. Those are excluded and reported as
    `vllm_prefill_noise_ms`. The rehearsal showed why: a prefill GEMM at 0.00 calls per step read 101.5 µs per step.
  - The profiled runs are not timed.

**Arms, in order:** e4b B=16 and B=1, with census → vLLM timing at B=16 and B=1 → vLLM census at B=16 and B=1 → the
second e4b draws → the second vLLM timing draws. Each step time is the median of its two draws.

**Families and roles.** Kernels are sorted by registered name maps, first match wins (`p86_reduce.py`):
- e4b's are P57's families plus the fp8 KV append;
- vLLM's are Marlin MoE, Marlin dense, MoE routing, attention, norm/rope/act, `torch.compile` Triton, and bf16 GEMM.

The families map to the roles above. The read lists each family's top kernels, so the map can be checked against
the names.

## Reading rule (`p86_reduce.py`, 13-case self-test)

1. **VOID** if any of these holds:
   - an arm is missing or failed;
   - a census covers no decode steps;
   - the vLLM build is not 0.30.0;
   - either engine's census sums to more than 1.10× its own step time, so the census is not measuring the step.
2. **NOT_READ** if more than 10 % of either engine's B=16 kernel time falls in "other". The name map does not cover
   that engine, so any ranking would rank the map. The read then lists the unmapped kernels, and a re-reading needs a
   registered map amendment.
3. **Otherwise READ.** The roles are ranked by gap (e4b − vLLM, ms per step) at B=16. The largest positive gap names
   the next lane's target. B=1 is reported beside it. B=1 is host-bound on any box, so it does not rank.

**What follows:**
- **Quantized linear** largest: a kernel lane on the B=16 expert GEMV (#564's 1.4 ms of measured headroom), with
  vLLM's Marlin MoE time as the benchmark.
- **Glue** largest: fold the routing gathers into the GEMV's row addressing.
- **Host and launch** largest: the graph's launch structure (launches per step, gaps).
- **Anything else:** that role's lane, named in the read.

## Predictions (written before the data)

- **Stated expectation, not the rule:** quantized linear is the largest gap. vLLM's Marlin MoE plus dense Marlin
  should take about 5–6 ms per step at B=16, against e4b's 7.0 ms for GEMV, K16, reduce and quantize. Glue should be
  second.
- **Both steps reproduce P58** within its host spread: e4b near 11.6 ms and vLLM near 8.3 ms at B=16. These are
  reported, not gated, because the box differs.

## Box and cost

Two rentals, in order. The guard exceeds one hour, so the compute rule requires a proving rental first.

1. **`p86-prove-<n>`**: one RTX 5090, **0.5 h guard at ≤ $0.75/h (≤ $0.375)**. `P86_PROVE=1` runs:
   - the refusals;
   - both installs (e4b with its tripwire; vLLM 0.30.0 in its venv);
   - the reducer self-test and an egress probe;
   - **the census arm itself on this build**: `p86_vllm_census.py` on vLLM 0.30.0 with `Qwen/Qwen3-0.6B`, B=4, an
     8 → 24-token slope.

   The proof passes only if the census ran in-process, covered 16 decode steps, and recorded at least 10 decode
   kernels, at least 3 of them once per layer per step (28 layers). This proves the profiler on vLLM 0.30 / CUDA 13 /
   sm_120, which the A2000 cannot. No 30B model is fetched. At most three attempts.
2. **`p86-5090-<n>`**: only after a proof returns rc 0 with its receipts fetched. **Guard 2.0 h at ≤ $0.75/h
   (≤ $1.50).**
   - P58 registered 3.0 h for 18 arms and ran in 42 minutes on a fast host. This lane runs 10 arms.
   - The deadline guard skips any arm that cannot finish 10 minutes before teardown (P58's 15-minute admission per arm),
     and a skipped arm reads VOID.
- **Lane ceiling $2.50; hard stop $3.00**; both under the $35 per-run cap.
- **Launch only when the cheapest eligible offers have a driver ≥ 580.** The launcher cannot exclude a machine on the
  lane's own refusal, so the offer list is checked read-only first, as for P85.

## Rehearsal

The home A2000 (driver 575, sm_86) cannot run vLLM 0.30's CUDA 13 wheels or hold either 30B model, so the rehearsal
tests the method, not this build. vLLM 0.11.0 (torch 2.8, CUDA 12.8) serves `Qwen/Qwen3-0.6B` in a throwaway
container, and `p86_vllm_census.py` runs unchanged (short slope 8 → 24 by its rehearsal knobs) at B=1 and B=4. A
control is the same census with `enforce_eager=True`: if graph replays were invisible to the profiler, the graph
arm's per-step kernel time would collapse while the eager arm's would not.

Results (2026-09-30; the card idle at 0 %, 1,374 MiB, before the first arm; records in
[`rehearsal-a2000/`](rehearsal-a2000/)):

| arm | cudagraph mode | kernels (decode / all) | ms/step, decode kernels | ms/step, all kernels | per-layer kernels |
|---|---|---:|---:|---:|---|
| graph, B=1 | `FULL_AND_PIECEWISE` | 33 / 40 | 5.53 | 5.50 | 28 / 56 calls per step (28 layers) |
| graph, B=4 | `FULL_AND_PIECEWISE` | 34 / 38 | 5.79 | 5.60 | 28 / 56 calls per step |
| eager control, B=4 | `NONE` | 25 / 28 | 7.40 | 7.59 | 28 / 56 / 112 calls per step |

- **Graph replays are recorded.** The decode kernels of the full-graph replays appear once per layer per step, under
  the names the eager control shows, plus `torch.compile`'s fused Triton kernels. The eager control's separate
  `rms_norm`, `fused_add_rms_norm`, `act_and_mul` and `rotary_embedding` kernels are the ones the graph arm fuses. A
  profiler blind to replays would have shown a graph-arm sum near zero.
- **One refinement came from it:** the decode-kernel filter above. The graph arm had 4 kernels with negative per-step
  differences and some at 0.00 calls per step, all prefill jitter over a 16-step slope. The lane's slope is 96 steps.
- **The first attempt failed in the rehearsal's environment, not the method.** vLLM 0.11.0 with the newest transformers
  (5.x, unpinned) raised `Qwen2Tokenizer has no attribute all_special_tokens_extended`. Pinning `transformers<5` fixed
  it. The lane installs vLLM 0.30.0 with its own resolved dependencies, as P58 did.

Also: the driver's dry run, and the CI tests (`tests/test_p86_staged_pin.py`), which pin:
- the staged bytes, and P58's harness and comparator bytes;
- the comparator version and checkpoint, and the gnf4 pin;
- the refusals coming before any install;
- the arm order and settings;
- the census's in-process engine and P37's settings.
