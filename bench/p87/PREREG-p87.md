# P87 — does K19 make e4b's int4 decode faster on an RTX 5090 without moving its quality? (registered 2026-10-01, before any run)

Issue: experts4bit-qlora#564 (the B=16 gap). P86 (`bench/p86/RESULTS-p86.md`) put vLLM 0.30.0's B=16 lead in the
expert kernel:
- e4b's split-K int4 GEMV takes 6.98 ms per step, plus 0.39 reduce and 0.35 quantize;
- Marlin MoE takes 4.78;
- that is 2.86 of the 3.32 ms gap.

The GEMV serves every routed row on its own. At B=16 that is 128 rows over about 80 distinct experts. Rows that share
an expert re-read its weights, and the int8 activation quantise and the split-K reduce are separate launches.

**K19** (grouped-nf4-gemm#419, merged `3351c9d`) is `int4_smallm.gemm_int4_b32_grouped_smallm`:
- K16's arithmetic (bf16 activations, in-register int4 dequant with per-32 scales, bf16 MMA, fp32 accumulate), one
  `tl.dot` per 128-wide K chunk;
- run over the expert-major 16-row tile table e4b already builds (`build_group_tiles_fused`);
- the first projection's gather folded in;
- no split-K, so no reduce, and no activation quantise;
- bit-identical to K16 per expert.

**Evidence so far** (A2000 only, P60's recorded B=16 routing, the expert projections alone): served GEMV 79.3 ms per
step, K16 per expert 73.3, K19 41.9 (1.89×) (`kernel/receipts-k19-a2000-probe/` in grouped-nf4-gemm). Exploratory, not
a claim, and not the 5090.

**The e4b route.** `E4B_INT4_GROUPED_SMALLM=1` (#803) sends the int4 store's decode rows through K19. #804 extends it
to T == 1, so B=1 decode, and K8, reach K19 too. Asked for and absent is a refusal, and without the opt-in nothing
changes. K19 changes the arithmetic (bf16 activations instead of int8), so the opt-in stays opt-in until a registered
read licenses a default. **This lane is that read.**

Rule: the owner's standing no-ask tier for a single run under $15 (2026-09-26), with the usual mechanics:
- this page merged before the launch;
- a proving rental before a guard over 1 h;
- receipts and ledger rows;
- proven teardown.

The owner asked to push ahead on throughput (2026-09-30).

## Question

On one RTX 5090, with the opt-in **on** against **off**:
1. **Speed:** how much faster is the decode step at B=16, and at B=1?
2. **Quality:** does K8 on the licensed int4 recipe move by more than the family's 0.0095-nat floor?

## The premise that lets a B=1 K8 stand for B=16

K8 scores through the T == 1 loop, one token at a time. A B=16 step puts 128 rows into tiles of up to 16 rows of one
expert. If K19's output for a row depended on the other rows in its tile, the B=1 K8 would not read the B=16 rows'
arithmetic.

K19 has no split-K and a fixed plan (BLOCK_N, KC), so it should not depend on them.
`tests/test_k19_row_exact_gpu.py` (#804) asserts it with `torch.equal`, on the real kernel at Qwen3-30B-A3B's expert
shapes:
- a token's rows are bit-equal whether it decodes alone or inside a B=16 step;
- the T == 1 route also captures in a CUDA graph and replays bit-equal to eager.

It passes on the A2000 (sm_86). **The lane runs it on the 5090 (sm_120) before anything is fetched. If it fails, the
lane stops (rc 25), and the reducer VOIDs any reading without it.**

## Instrument

**The box:** one RTX 5090, Vast verified/secure, `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`, ≥ 200 GB of disk.
- Any driver the image runs on. This lane installs no vLLM, so P86's ≥ 580 requirement does not apply.
- Any CPU vendor. Every comparison is ON against OFF on the same box. The licensed float is read on AMD hosts (P84
  amendment 1), so the OFF K8 is compared with it only as a report, never a gate.

**Stack:** the launch commit (e4b, carrying #803 and #804) and grouped-nf4-gemm `3351c9d` (K19's merge). The runner's
tripwire refuses an install that lacks:
- K19;
- the route;
- the T == 1 extension (`_collapsed_grouping`);
- the K16 attention route;
- the #405 pack knobs.

It also refuses a fused GEMV reduce that is defaulted on.

**Speed arms: P86's, unchanged except the opt-in.**
- **Env:** P58's RTN int4 experts, uncalibrated int4 attention, glue r1/r2, router epilogue.
- **Harness:** P58's pieces (`bench/p39/step_decomp.py`, `bench/p42` hook), byte-identical to P86's.
- **Command line:** `--batch {16,1} --prompt-len 512 --gen-tokens 128 --b1d-loop graph --b1d-timed --fuse-qkv`.
- **Timing:** `step_ms_clean`, the graph-replay window.
- **Census:** P42's protocol. `--replay-profile-out` profiles 8 replays after the timed window, parsed by
  `bench/p42/p42_reduce.py`.

**Quality arms: K8 on the licensed recipe.**
- **Env:** calibrated experts and calibrated int4 attention, C4 calibration over 128 sequences, the three folds:
  P85's `lic` env.
- **K8 arguments:** P85's: `--batch 1 --prompt-len 512 --gen-tokens 16 --ppl-steps 2048 --b1d-loop eager --no-fuse-qkv
  --ppl-source wikitext`, window `9ef10d760ad9`.
- **The arms:**
  - one **build** (opt-in OFF) calibrates the experts and dumps the pack, `verify_artifact`-checked;
  - **OFF** and **ON** then load that pack by fingerprint, and the attention is calibrated live, as P70 and P85 did.
- **The build is reported beside OFF.** P85 read a build and its reload bit-equal. It is not gated.

**Arms, in order:**
1. speed B=16 OFF, ON; B=1 OFF, ON — the first draws, censused;
2. K8 build (OFF), K8 OFF, K8 ON;
3. speed B=16 ON, OFF; B=1 ON, OFF — the second draws.

Each step time is the median of its two draws.

## Reading rule (`p87_reduce.py`, 17-case self-test)

1. **VOID** if any of these holds:
   - the on-box premise did not hold or did not run;
   - an arm is missing or failed, or a census is missing;
   - two draws of one arm differ by more than 3 %;
   - **engagement fails at either B:**
     - ON did not run K19 96 times per step (48 layers × gate_up + down);
     - the expert GEMV's 96 calls per step did not all leave the GEMV. At B=1 the GEMV also serves the attention
       projections, so the count is OFF − ON = 96.
     - OFF ran K19 at all;
   - a K8 is off its window or step count;
   - **K8 ON equals K8 OFF to the bit.** K19 changes the arithmetic, so a bit-equal reading means it did not run.
2. **QUALITY_FAIL** if |K8(ON) − K8(OFF)| > 0.0095 nats. K19 stays opt-in, whatever its speed.
3. **LICENSED** if quality holds and B=16 is faster: median ON / median OFF ≤ 0.95.
4. **NOT_FASTER** if quality holds and B=16 is not.

B=1 is reported beside the verdict: FASTER (≤ 0.97), SLOWER (≥ 1.03) or NEUTRAL. It decides the scope of the default
a LICENSED read proposes.

**What follows:**
- **LICENSED:** a PR defaults `E4B_INT4_GROUPED_SMALLM` on for int4 decode rows, with `=0` kept as the escape hatch.
  - All decode rows if B=1 is FASTER or NEUTRAL.
  - Only rows above T == 1 if B=1 is SLOWER.
  - Plus a register row for each repo (e4b's step, gnf4's K19 on the 5090) and a gnf4 release carrying K19 (0.34.0).
- **NOT_FASTER:** K19 stays opt-in. The censuses say where the time went: tile padding, the tile build, or K19 itself.
  The next kernel lane is named in the read.
- **QUALITY_FAIL:** K19 stays opt-in. The read examines whether the move is in the experts' bf16 activations (P64's
  question, read INDISTINGUISHABLE at T == 1 on the dequant path) or in K19's own arithmetic.

## Predictions (written before the data)

- **B=16:** faster, ratio 0.75–0.85.
  - The expert time drops from 6.98 ms toward 4 ms, the A2000 probe's 1.89× applied to P86's split.
  - The 0.39 ms reduce and the 0.35 ms quantize go away. At B=16 all 96 of each per step are the experts' (P86's census).
  - ON adds the tile build, one fused launch per layer. The OFF GEMV route builds no grouping at all.
  - OFF should land near P86's 12.06 ms, reported, not gated.
- **B=1: NEUTRAL** (within ±3 %).
  - The 8 rows are 8 one-row tiles, so the MMA wastes 15 of 16 rows. Each expert's weights are still read once, as the
    GEMV reads them, and B=1 is weight-bandwidth-bound.
  - The quantize and the reduce go away; one tile-build launch per layer is added.
- **K8:** |Δ| < 0.003 nats. P64 read bf16 expert activations INDISTINGUISHABLE at T == 1. K19's error against an fp32
  oracle was 0.0048 relative on the A2000, against the GEMV's 0.0131.
- **The build equals OFF to the bit**, as in P85.

## Box and cost

Two rentals, in order. The guard exceeds one hour, so the compute rule requires a proving rental first.

1. **`p87-prove-<n>`**: one RTX 5090, **0.5 h guard at ≤ $0.75/h (≤ $0.375)**. `P87_PROVE=1` runs:
   - the refusals, the install with its tripwire, and the reducer self-test;
   - **the premise test** on sm_120;
   - **grouped-nf4-gemm's K19 and K16 contract tests**, compiled on the card (`test_int4_grouped_smallm_interp.py`,
     `test_int4_smallm_interp.py` at `3351c9d`; 20/20 compiled on the A2000). K19 has never run on sm_120.
   - an egress probe.

   No model is fetched. At most three attempts.
2. **`p87-5090-<n>`**: only after a proof returns rc 0 with its receipts fetched. **Guard 2.5 h at ≤ $0.75/h
   (≤ $1.875).**
   - P85's reading took 72 minutes on the box (two bakes, a calibrated build, five K8s), and P86's took 53.
   - This lane fetches one model, bakes once, runs one calibrated build, two K8s and eight speed arms.
   - The deadline guard skips any arm that cannot finish 10 minutes before teardown, and a skipped arm reads VOID.
   - A build that has not finished its first calibration chunk in 1,500 s is killed as host-limited (rc 30; P85
     amendment 1).
- **Lane ceiling $3.00; hard stop $3.50**; both under the $35 per-run cap.

## Rehearsal

The home A2000 (sm_86) cannot run step_decomp's decode loops: the fp8 paged-KV kernel needs sm_89 or newer (P64's
rehearsal). It cannot hold Qwen3-30B-A3B either. So the rehearsal covers the pieces below the model, on real CUDA,
with gnf4 at `3351c9d`:

- **The route and the premise.** Run in a throwaway `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel` container: 19/19 pass
  across `tests/test_k19_row_exact_gpu.py`, P63's `tests/test_p63_row_exact_gpu.py` and
  `tests/test_int4_grouped_smallm_route.py`.
  - T == 1 reaches K19 (2 calls: gate_up, down).
  - Tokens 0, 5 and 15 of a B=16 step are bit-equal alone vs inside the step.
  - The captured T == 1 route replays bit-equal to eager on two other tokens.
  - Relative error against an fp32 oracle: K19 0.0048, the GEMV 0.0131.
- **The census engagement counts.** `p87_reduce.py`'s parser reads P86's 5090 censuses
  (`bench/p86/receipts/p86-5090-3/logs/`) as 96 GEMV calls per step at B=16 and 192 at B=1. That is what the OFF arms
  must show, and what the ON arms must drop by 96.
- **The runner's proving path, end to end** ([`rehearsal-a2000/`](rehearsal-a2000/)). `p87_run.sh` ran with `P87_PROVE=1` under
  rehearsal knobs (class A2000, 20 GB disk), with the files staged as the driver stages them and e4b installed from
  GitHub at #804's head. The pins, the reducer self-test (17 cases), the install and tripwire, the premise (3 passed),
  K19's and K16's contract tests compiled (20 passed) and the egress probe all passed: lane rc 0, `PROVED`.
- **The driver's** syntax and dry run.

Also the CI tests (`tests/test_p87_staged_pin.py`), which pin:
- the staged bytes, and P58/P86's harness bytes;
- the gnf4 pin;
- the refusals and the premise coming before any fetch;
- the proof's contents;
- the arm order and settings.

**What the rehearsal changed in the design** (before registration):
- **Writing this reducer found the T == 1 gap.** Its engagement check expected 96 K19 calls per step at B=1 too. #803's
  route engaged only at T > 1, so K8 would have read the GEMV with the opt-in on and off alike. #804 fixed the route
  and added the premise test.
- **The quality arms moved to the licensed recipe**, as the lane was first described to the owner. The speed arms stay
  on P86's RTN recipe, so their numbers sit beside P86's and P58's.
