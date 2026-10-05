# TC1c — the same matched set on one H100 NVL: the 5090 position reverses (lane TC1c of #835; 2026-10-02)

Pre-registration: [`../../tc1/TC1C-PREREG.md`](../../tc1/TC1C-PREREG.md) — TC1's `qwen3` token unchanged on an H100 NVL
(`TC1_GPU_CLASS="H100 NVL"`), the same harness at e4b `09b0f6a` (0.38.1; `experts4bit/` byte-identical to TC1's 079a422),
grouped-nf4-gemm `846b512`, the same tokens (sha `bfc742f67e37`), the same per-slot init (`f7832488926eda91`), fp32 adapters in both
frameworks, Unsloth 2026.9.14 on torch 2.12.1+cu130 with `grouped_mm` engaged on every step. Receipts: `receipts/tc1c-h100-2/`
(private store `receipts/experts4bit-qlora/2026-10-02/tc1c-h100-2/tc1/`); one box, Vast verified-secure instance 53802633
(AMD EPYC 9534, 224 vCPU, 1.58 TB host RAM, driver 595.71.05), $4.88. The first draw (`tc1c-h100-1`) was refused at $0 before any
instance existed (its manifest carried the 5090 pre-flight exclusion receipts, not same-class for an H100).

## Amendment 9 (2026-10-05): the same-stack H100 box reads UNTESTED: e4b's draws 18 % and 29 % apart on a RunPod container given 18 vCPUs and run with 72 threads

Pre-registration: [`../../tc1/TC1C-PREREG.md`](../../tc1/TC1C-PREREG.md), amendment 9. TC1 amendment 25's same-stack family on one H100 NVL.
Both frameworks run on torch 2.12.1+cu130 / transformers 5.5.0, 60 load-gated steps, e4b @ `1607441` (0.48.0), grouped-nf4-gemm `f127981`
(0.41.0), Unsloth 2026.9.14. Read: [`RESULTS-tc1c-samestack-h100.md`](RESULTS-tc1c-samestack-h100.md); receipts
[`receipts/tc1c-h100-22/`](receipts/tc1c-h100-22/).

**The box.** `tc1c-h100-22` ran on **RunPod Secure**, not Vast: pod `lj6bxnvudzxiv7`, H100 NVL, Intel Xeon Platinum 8452Y (144 host CPUs, 18
vCPUs allotted to the pod), driver 580.159.04, 141 GB RAM. It was the first lane to complete on the launcher's RunPod adapter.

- No Vast H100 NVL qualified when it launched. `tc1c-h100-18` (Vast) failed pre-flight on a 150 GB overlay for a 320 GB order ($0.008).
- `-19` and `-20` (RunPod) stopped because the image lacked `rsync` and then `git` (adertha-agents #169, #171; $0.29).
- `-21` was refused at $0 for lack of stock.
- **Budget, disclosed:** the amendment registered a $2.80/h GPU ceiling. This pod's GPU rate was $3.19/h, run on the owner's direct go for
  the H100.
- **Spend, disclosed:** the launcher booked $3.48 from billing records that covered 3,869 of the pod's 7,548 s. At its own rate the box cost
  about $6.78. adertha-agents #174 makes such partial records book the estimate.

| arm | s/step, two draws | peak |
|---|---|---|
| e4b `fused_attn4_m`, venv-unsloth | 2.968 / 3.557 (**18.1 % apart**) | 27.49 GB |
| e4b `fused_attn4_m_t28`, venv-e4b | 3.428 / 2.568 (**28.7 % apart**) | 27.51 GB |
| Unsloth `ckpt_unsloth_m` | 3.127 / 3.021 (3.5 %) | 24.27 GB |

- **P27 UNTESTED** and **P28 UNTESTED**: e4b's draws are unstable on both of its arms. **P29 HELD**: `grouped_mm` with `GNF4_TRAIN_GEMM`
  unset on every e4b receipt. Every arm is VALID.
- **By amendment 9's rule nothing is quoted.** Amendment 8's 1.061 stays the H100 position of record.
- **Why the e4b draws moved (a candidate, not a reading).** The lane sets `OMP_NUM_THREADS` to the host's physical cores. That was 72 here,
  for a pod RunPod allotted 18 vCPUs (`runpod_vcpu_count` on the receipt), four threads per vCPU. Its load gate also read the host's load
  average, 7.5–42 over the standing attempts, where the gate is 6.0. e4b's step is host-bound and Unsloth's less so, and only e4b's draws
  came apart. Both frameworks also stepped slower than amendment 8's Vast host (2.43 / 2.55 s): 2.6–3.6 s here. A lane that sizes its
  threads to the container's allotment would separate the two causes. Whether Vast containers carry a CPU quota is not recorded on any box
  yet.

## Amendment 8 (2026-10-04): the H100 at default settings on 0.45.0 — e4b faster per step, Unsloth/e4b 1.061 (P24, P25, P26 HELD): the H100 position of record

Pre-registration: [`../../tc1/TC1C-PREREG.md`](../../tc1/TC1C-PREREG.md), amendment 8. Amendment 7's box once more, the last re-ask under these rules.

**The box.** `tc1c-h100-15`: an H100 NVL (instance 54181574, AMD EPYC 9V84, driver 580.159.03), e4b 0.45.0's code (`3888fca`), grouped-nf4-gemm v0.37.0,
nothing set, HF and axolotl skipped. Vast invoiced $2.44. Receipts: [`receipts/tc1c-h100-15/`](receipts/tc1c-h100-15/). Three earlier draws produced no box:
`tc1c-h100-12` and `-14` were refused before any rental ($0; the first at a price that had risen to $3.01/h, the second on an exclusion
receipt the launcher does not accept), and `-13` failed pre-flight on a 60 GB disk ($0.008). The guard ceiling was raised from $2.80/h to
$3.10/h, a budget change, not a change to the box.

| | e4b `fused_attn4_m` (default) | Unsloth `ckpt_unsloth_m` |
|---|---|---|
| s/step, two draws | 2.428 / 2.367 (**2.5 % apart**) | 2.545 / 2.543 |
| peak VRAM | 27.21 / 27.19 GB | 24.27 GB |
| energy per step | 492.9 / 488.8 J | 410.4 / 389.1 J |
| held-out at N = 20 | 0.8491 / 0.8484 | 0.8487 / 0.8494 |

- **P24 HELD.** The MATCHED POSITION is **Unsloth/e4b 1.061 [1.047, 1.075]**, inside [0.95, 1.12]: e4b is faster per step.
- **P25 HELD.** `route_ab` names `grouped_mm` with `GNF4_TRAIN_GEMM` unset on every fused arm (16,896 forward and 7,680 dgrad calls), and none on
  the reference.
- **P26 HELD.** The reference and Unsloth are both EQUIVALENT to the fused arm.

**Decision, as registered.** This box is the H100 position at default settings: `e4b.train.h2h.unsloth.qwen3.h100.release-0.45.0`. It
supersedes amendment 1's 0.817 as the default-settings row; that row stays as the fused kernels' reading. On the H100, e4b is now faster than
Unsloth with nothing set, at 2.93 GB more peak VRAM and about ×1.23 Unsloth's energy per step.

## Amendment 7 (2026-10-04): the H100 at default settings on 0.45.0 — `auto` takes the route (P22 HELD), but the box is not quoted (P21 UNTESTED, P23 FALSIFIED)

Pre-registration: [`../../tc1/TC1C-PREREG.md`](../../tc1/TC1C-PREREG.md), amendment 7.

**The box.** `tc1c-h100-11`: an H100 NVL (instance 54150922, AMD EPYC 9534, driver 595.71.05), e4b 0.45.0's code (`3171621`), grouped-nf4-gemm
v0.37.0 (`71185d6`), nothing set: no `TC1_E4B_ENV`, `GNF4_TRAIN_GEMM` unset. HF and axolotl were skipped. Vast invoiced $2.15. Receipts:
[`receipts/tc1c-h100-11/`](receipts/tc1c-h100-11/).

| | e4b `fused_attn4_m` (default) | Unsloth `ckpt_unsloth_m` |
|---|---|---|
| s/step, two draws | 2.540 / 2.414 (**5.1 % apart**) | 2.537 / 2.552 |
| peak VRAM | 27.21 / 27.19 GB | 24.27 GB |
| held-out at N = 20 | 0.8484 / 0.8474 | 0.8545 / 0.8510 |

- **P22 HELD.** With nothing set, `route_ab` names `grouped_mm` on every e4b fused arm, with amendment 6's 16,896 forward and 7,680
  dgrad calls, and none on the reference arm. This is the first time `auto`'s capability check ran on a card that takes the route.
- **P21 UNTESTED.** e4b's two draws differ by 5.1 %, over the 5 % stability rule, so no position is quoted. The pooled ratio would be
  about 1.03, as in amendment 6, but it is reported, not quoted.
- **P23 FALSIFIED.** Unsloth's first draw reads COMPARABLE: held-out Δ 0.0061 against a band of 0.0054. The band is narrow because
  e4b's fused-vs-reference Δ was small on this box. The reference is INSIDE-DRAW-NOISE, and Unsloth's second draw sits at 0.0027 to e4b.
- **The receipt reads ALARM.** The workload finished (its success marker is present), but the teardown went through the emergency
  retry loop. The destroy call answered 404, and the listing then proved the instance absent. The arm receipts are unaffected.

**Decision, as registered.** The box is not quoted. Amendment 1's 0.817 stays the default-settings row (it read the fused
kernels); amendment 6's `.route-v2` (1.030) stays labelled beside it. Amendment 8 re-asks the same box once.

## Amendment 6 (2026-10-04): with the dequant at bandwidth the grouped_mm route makes e4b faster than Unsloth on the H100 — 1.030 alone, 1.325 with MoE-keep — P18, P19 and P20 HELD

Pre-registration: [`../../tc1/TC1C-PREREG.md`](../../tc1/TC1C-PREREG.md), amendment 6.

**The boxes.** Both ran amendment 4's two boxes again on an H100 NVL (AMD EPYC 9534, driver 595.71.05), now with e4b `e775eff` and
grouped-nf4-gemm `81706a9` (the merge of #452, whose dequant kernel is bit-equal to the old one). HF and axolotl were skipped.
Receipts are in [`receipts/tc1c-h100-9/`](receipts/tc1c-h100-9/) and [`receipts/tc1c-h100-10/`](receipts/tc1c-h100-10/).

- **Box R** (`tc1c-h100-9`, instance 54128342, $2.65 invoiced): every e4b arm with `GNF4_TRAIN_GEMM=grouped_mm`.
- **Box K** (`tc1c-h100-10`, instance 54128460, $2.76 invoiced): the same plus `E4B_MOE_KEEP_LAYERS=all`, the compact delta and
  host reuse.

Engagement is verified on every fused arm. `route_ab` names `grouped_mm` with 16,896 forward and 7,680 dgrad calls on box R, and
9,216 and 7,680 on box K, the same counts as amendment 4.

| box | e4b s/step (d1 / d2) | Unsloth s/step | **Unsloth/e4b** | amendment 4 (old dequant) | its default counterpart | peak e4b / Unsloth | verdict |
|---|---|---|---|---|---|---|---|
| R (route) | 2.512 / 2.450 (2.481) | 2.557 | **1.030** [1.016, 1.045] | 0.664 | 0.817 (amendment 1) | 27.19 / 24.27 GB | **P18 HELD** (0.90–1.25) |
| K (keep + route) | 1.941 / 1.910 (1.925) | 2.552 | **1.325** [1.296, 1.356] | 0.934 | 1.100 (amendment 2) | 34.08 / 24.27 GB | **P19 HELD** (1.15–1.65) |

**P20 HELD.** On box R e4b's reference and Unsloth are EQUIVALENT to the fused arm. On box K the reference is inside the fused arm's
draw noise and Unsloth is EQUIVALENT. Held-out losses at N = 20: box R e4b 0.8467 / 0.8495, Unsloth 0.8513 / 0.8515; box K e4b
0.8531 / 0.8499, Unsloth 0.8532 / 0.8503.

**Why it is faster now.** The profiled e4b arm on box R reads device-busy 0.454, against amendment 4's 0.855:

- `_dequant_groups_kernel` took **256 ms of device time per step**: 1,152 calls at **0.222 ms each**, against amendment 4's 2,178 ms
  at 1.89 ms (×8.5). The grouped GEMMs took 247 ms per step, as before.
- Dequant plus grouped GEMMs come to about 0.50 s per step, against the fused kernels' 1.42 s in amendment 1.
- Box K's dequant took 170 ms per step over 768 calls (0.221 ms each), against amendment 4's 1,456 ms; its device is busy 0.433 of
  the profiled step.
- e4b's step on this box is now host-bound: 1.51 s of device time per step against an unprofiled step of 2.48 s (the profiled
  step takes 3.32 s).

**The default rule.** Amendment 4 registered three conditions for making the route grouped-nf4-gemm's default on sm_90. All three hold:

- P20 HELD.
- Box R 1.030 ≥ 0.858 (amendment 1's 0.817 + 5 %).
- Box R's e4b held-out, 0.8481 over the two draws (0.84667 / 0.84945), is within 0.005 of amendment 1's 0.8521 (0.85228 / 0.85201):
  Δ 0.0041, lower. The margin is thin. Draw against draw the four differences are 0.0026, 0.0028, 0.0053 and 0.0056, so two of the
  four pairings would miss 0.005. The rule is read on the two-draw means, the statistic this page quotes in bold for every step time,
  and the route's held-out sits with the other arms on the box: the reference 0.8502, Unsloth 0.8513 / 0.8515.

**What follows.**

- The route qualifies as grouped-nf4-gemm's default on compute capability 9.0. The change is grouped-nf4-gemm#454
  (`GNF4_TRAIN_GEMM=auto`; `GNF4_TRAIN_GEMM=fused` keeps the fused kernels; other cards unchanged).
- Both readings are LABELLED rows, `e4b.train.h2h.unsloth.qwen3.h100.2026-10-04.route-v2` (1.030) and `...moe-keep-route-v2` (1.325),
  until a default-settings box re-reads the H100 position on the release that carries the new default.
- The next lever on this card is host time, not device time.

## Amendment 5 (2026-10-04): the fused kernels' own configs on the H100 change nothing worth taking — a kernel replay, not a position

Pre-registration: [`../../tc1/TC1C-PREREG.md`](../../tc1/TC1C-PREREG.md), amendment 5.

**The box.** `tc1c-h100-8`: Vast instance 54116273, H100 NVL (capability 9.0), torch 2.8.0+cu128. Vast invoiced $0.33: GPU $0.324,
storage $0.007, download $0.001 for 4.4 GB. Receipts are in [`receipts/tc1c-h100-8/`](receipts/tc1c-h100-8/), with the per-config table
in `FUSEDSWEEP.json`. The code was e4b `4ce22eb` and grouped-nf4-gemm `951a97f`. Token `fusedsweep`, with no model:
[`../../tc1/fused_sweep.py`](../../tc1/fused_sweep.py) ran the same 128 unique recorded calls as amendment 3.

**Forward** (25 configs; the default is variant 1, groups 1, BLOCK_N 128, 4 warps, 3 stages):

- Nothing beats the default. Re-run explicitly it read 0.985, and every other variant-1 config was 0.999–1.82×. All of them were
  bit-identical to the default.
- BLOCK_K 128 (`prefill_groups=2`) now fits in the H100's 227 KB, unlike the A2000's 101 KB, but reads 1.03–1.78×.
- The bf16-MMA mainloop (variant 3) is **2.3–3.5× slower**, at 0.0024 relative Frobenius off the default. With BLOCK_K 128 it still
  does not fit.
- **P15 FALSIFIED:** the best bit-identical config is 0.985 against a bound of 0.90.
- **P16 FALSIFIED:** the best config of any kind is 0.985 against a bound of 0.60.

**dgrad** (9 configs; the default is BLOCK_M 32, BLOCK_N 64, BLOCK_K 64, 2 warps):

- (BLOCK_M 64, BLOCK_N 128, BLOCK_K 64, 4 warps) reads **0.853**.
- It is not bit-identical to the default: a 5e-5 relative Frobenius difference, a different accumulation order. The best bit-identical
  config reads 0.988.
- **P17 HELD** (bound 0.90).

**What follows.** The fused kernels' gap to `torch._grouped_mm` on sm_90 is structural, not a configuration choice: amendment 3 found
the grouped GEMM alone at 0.20 / 0.16 of them. The decision rule turned a P17 win into a default only for a bit-identical config, and
this one is near-exact, not identical. It is recorded, not taken. It would matter only if the grouped_mm route, whose dgrad reads 0.494,
does not become the H100 default after amendment 4. This box quotes no position and changes no register row.

## Amendment 4 (2026-10-04): the grouped_mm route makes e4b slower on the full step — its dequant kernel, not the route — P12 and P13 FALSIFIED, P14 HELD

Pre-registration: [`../../tc1/TC1C-PREREG.md`](../../tc1/TC1C-PREREG.md), amendment 4.

**The boxes.** Both ran TC1c's token on an H100 NVL (AMD EPYC 9534, driver 595.71.05) with e4b `8208f5e` and grouped-nf4-gemm `e7ec90a`
(#450). HF and axolotl were skipped. Receipts are in [`receipts/tc1c-h100-6/`](receipts/tc1c-h100-6/) and
[`receipts/tc1c-h100-7/`](receipts/tc1c-h100-7/).

- **Box R** (`tc1c-h100-6`, instance 54119104, $2.49 invoiced): every e4b arm with `GNF4_TRAIN_GEMM=grouped_mm`.
- **Box K** (`tc1c-h100-7`, instance 54119272, $2.49 invoiced): the same plus `E4B_MOE_KEEP_LAYERS=all`, the compact delta and host
  reuse.

Engagement is verified on every fused arm. `route_ab` names `grouped_mm` with 16,896 forward and 7,680 dgrad calls on box R, and
9,216 and 7,680 on box K. Box K's `keep_ab` records 48 layers.

| box | e4b s/step (d1 / d2) | Unsloth s/step | **Unsloth/e4b** | its counterpart | peak e4b / Unsloth | verdict |
|---|---|---|---|---|---|---|
| R (route) | 3.956 / 3.908 (3.932) | 2.612 | **0.664** [0.650, 0.678] | 0.817 (amendment 1, e4b 3.146 s) | 27.20 / 24.27 GB | **P12 FALSIFIED** (0.85–1.20) |
| K (keep + route) | 2.806 / 2.735 (2.771) | 2.588 | **0.934** [0.916, 0.953] | 1.100 (amendment 2, e4b 2.343 s) | 34.08 / 24.27 GB | **P13 FALSIFIED** (1.15–1.60) |

**P14 HELD.** On both boxes e4b's reference is inside the fused arm's draw noise. Unsloth is inside it on R and EQUIVALENT on K. The
route's training numerics are fine.

**Why the step got slower.** The profiled e4b arm on box R reads device-busy 0.855:

- `_dequant_groups_kernel`, grouped-nf4-gemm's own dequant for the route, took **2,178 ms of device time per step**: 1,152 calls at
  **1.89 ms each**. The grouped GEMMs took 250 ms per step.
- Amendment 1's fused forward + dgrad kernels took 1,424 ms per step.
- Box K's dequant took 1,456 ms per step over 768 calls; keeping activations removes the recompute's calls.
- Amendment 3's replay, whose ratios this box's predictions relied on, timed **bitsandbytes'** `dequantize_4bit` at about 0.46 ms
  per call, and the route does not use that.
- On an RTX A2000 the route's dequant is 2.9–4.1× slower than bitsandbytes' on the same stacks (51–75 against about 210 GB/s;
  outputs bit-equal).

So the falsification is the route's dequant kernel, not the dequant-then-grouped-GEMM idea. At bitsandbytes' speed the route would
take about 0.8 s per step against the fused kernels' 1.42 s.

**What follows.**

- By the decision rule the route does not become the sm_90 default. It stays opt-in, and both readings become LABELLED rows:
  `e4b.train.h2h.unsloth.qwen3.h100.2026-10-04.route` (0.664) and `...moe-keep-route` (0.934), quoted beside amendments 1 and 2,
  never in place of them.
- A faster dequant kernel is the next change, with its own registered box.

## Amendment 3 (2026-10-04): on the H100, dequantize + `torch._grouped_mm` runs the recorded GEMM calls in 0.50–0.60 of the fused kernels' time — a kernel replay, not a position

Pre-registration: [`../../tc1/TC1C-PREREG.md`](../../tc1/TC1C-PREREG.md), amendment 3.

**The box.** `tc1c-h100-5`: Vast instance 54112861, H100 NVL (capability 9.0), driver 595.71.05, torch 2.8.0+cu128, bitsandbytes 0.50.2,
$0.31 invoiced: GPU $0.305, storage $0.007 (corrected 2026-10-04; the receipt's GPU-time figure was $0.21). Receipts in [`receipts/tc1c-h100-5/`](receipts/tc1c-h100-5/), with the per-call table in `ROUTEBENCH.json`. The code was e4b
`69e7962` and grouped-nf4-gemm `4509307`. Token `routebench`: no model, no Unsloth venv.
[`../../tc1/route_bench.py`](../../tc1/route_bench.py) replays the 128 unique fused-GEMM calls of e4b's training step from
[`../../tc1/routecalls-qwen3.json`](../../tc1/routecalls-qwen3.json). Those were recorded through the real checkpoint's router on a
4-layer Qwen3-30B-A3B slice, at Qwen3-30B-A3B's shapes. Each call is timed by CUDA events at the median of 10. The whole-stack
dequant matched `dequant_ref` bitwise before anything was timed.

| calls (64 each) | fused (ms) | dequantize_4bit (ms) | torch._grouped_mm (ms) | (dequant + grouped_mm) / fused | grouped_mm / fused |
|---|---|---|---|---|---|
| forward | 75.11 | 29.45 | 15.29 | **0.596** | 0.204 |
| dgrad | 89.20 | 29.42 | 14.63 | **0.494** | 0.164 |

By shape:

| | gate_up (N 1536, K 2048) | down (N 2048, K 768) |
|---|---|---|
| forward | 0.569 | 0.646 |
| dgrad | 0.493 | 0.495 |

- **P9 HELD:** forward 0.596, band 0.20–0.80.
- **P10 HELD:** dgrad 0.494.
- **P11 HELD:** the route's relative Frobenius error against the fused output is at most 0.0024 on every call (bound 0.005).
- **The dequant dominates.** The dequantize is about two thirds of the route's time (≈ 0.46 ms per call, a 128-expert bf16 stack
  written each time). The grouped GEMM alone is 0.16–0.20 of the fused kernels. On sm_90 the fused decode-in-the-mainloop kernels are
  slower than decoding once and running the card's native grouped GEMM, and the profiles' device-time gap matches.

**What follows.** By the decision rule, the route goes into grouped-nf4-gemm as an opt-in for sm_90 (`GNF4_TRAIN_GEMM=grouped_mm`: a
Triton dequant of the present experts, bit-equal to `dequant_ref`, then `torch._grouped_mm` for the forward and the dgrad). Its value on
the full training step is a separate registered box (amendment 4). Not bit-identical to the fused kernels, so a training A/B decides
it. This box quotes no position and changes no register row.

## Amendment 2 (2026-10-04): with e4b keeping all 48 layers' MoE activations it is faster per step on the H100 too — 1.100, a labelled row (register `e4b.train.h2h.unsloth.qwen3.h100.2026-10-04.moe-keep`)

Pre-registration: [`../../tc1/TC1C-PREREG.md`](../../tc1/TC1C-PREREG.md), amendment 2.

**The box.** `tc1c-h100-4`: Vast instance 54103635 on the same machine as amendment 1's box (AMD EPYC 9534, H100 NVL, driver
595.71.05), $2.97 invoiced: GPU $2.89, storage $0.07, download $0.02 for 73.6 GB (corrected 2026-10-04; the receipt's GPU-time figure was $2.80). Receipts in [`receipts/tc1c-h100-4/`](receipts/tc1c-h100-4/). It used TC1c's token. Every e4b arm ran with
`TC1_E4B_ENV="E4B_MOE_KEEP_LAYERS=all NF4_QLORA_COMPACT_DELTA=1 GNF4_HOST_REUSE=1"`. In each of the 48 decoder layers attention alone
is checkpointed, and the MoE activations are kept rather than recomputed; gradients are identical by construction. HF and axolotl
were skipped (`not_run`). The code was e4b `8846764` and grouped-nf4-gemm `ac84818`. Unsloth's arms were unchanged.

**Engagement.** Each e4b fused arm's `keep_ab` records 48 layers kept with the compact delta on, and its `reuse_ab` records the flag on
with 8,640 upload hits.

| | e4b `fused_attn4_m`, MoE activations kept | Unsloth `ckpt_unsloth_m` | reading |
|---|---|---|---|
| s/step, median of steps 11..20, two draws | 2.333 / 2.353 (**2.343**) | 2.593 / 2.560 (**2.577**) | **Unsloth/e4b 1.100 [1.088, 1.111]**: e4b faster per step; both STABLE |
| peak VRAM | 34.08 GB | **24.27 GB** | Unsloth lower by 9.81 GB |
| energy per step | 475.1 / 443.8 J | 480.1 / 451.1 J | about equal (Unsloth ×1.013) |
| held-out at N = 20 | 0.8486 / 0.8500 | 0.8507 / 0.8483 | the matched set inside the draw noise (P3 HELD) |

- **P7 HELD.** 1.100 lies in the registered [0.85, 1.30]. The interval is wholly above 1.0, so by the registered ordering reading
  **e4b with this setting is faster per step on this card**, at 9.8 GB more peak memory.
- **P8 HELD.** e4b's reference arm and Unsloth are both inside the fused arm's draw noise.
- **Against amendment 1 on the same machine.** e4b's step went from 3.146 to 2.343 s (×0.745). Unsloth's went from 2.571 to 2.577 s.
- **The profiled e4b arm.** Device-busy is 0.659, with 68,452 device events and 455,685 CPU-side ops per step (amendment 1: 97,851 and
  619,192). Unsloth reads 0.306 / 81,780 / 515,750. e4b still spends about 1.8× Unsloth's device time per step.

**How to read the two H100 rows.** Amendment 1's 0.817 is e4b with its default settings, and stays the H100 position for those
defaults. This row is e4b with an opt-in memory-for-time setting, and is quoted beside it, never in place of it. The setting costs
peak memory: e4b keeps 34.08 GB here against Unsloth's 24.27 GB. The 5090 measured the same trade at 16–32 kept layers
(`e4b.train.moe-keep.qwen3.5090.2026-10-04`).

## Amendment 1 (2026-10-04): the position again with e4b after TC1 amendments 10–15 — Unsloth still faster, by 1.22× instead of 1.61× (register `e4b.train.h2h.unsloth.qwen3.h100.2026-10-04`)

Pre-registration: [`../../tc1/TC1C-PREREG.md`](../../tc1/TC1C-PREREG.md), amendment 1.

**The box.** `tc1c-h100-3`: Vast instance 54091206, AMD EPYC 9534, H100 NVL, driver 595.71.05, $4.42 invoiced: GPU $4.30, storage $0.10, download $0.02 for 73.6 GB (corrected 2026-10-04; the receipt's GPU-time figure was $4.33). Receipts in
[`receipts/tc1c-h100-3/`](receipts/tc1c-h100-3/). The token is the same as before. e4b `e1837cf` carries every default from TC1
amendments 10–15 (#945's single-read grouping and pinned ring, the trimmed LoRA delta, the cost tile rule, the fused RMSNorm and rotary),
and grouped-nf4-gemm is at `00929a4`. No environment variables were set. The comparator was unchanged: Unsloth 2026.9.14, torch
2.12.1+cu130, `grouped_mm` engaged.

| | e4b `fused_attn4_m` | Unsloth `ckpt_unsloth_m` | reading |
|---|---|---|---|
| s/step, median of steps 11..20, two draws | 3.188 / 3.104 (**3.146**) | 2.548 / 2.594 (**2.571**) | **Unsloth/e4b 0.817 [0.799, 0.836]**: Unsloth faster per step by 1.22× (was 1.61×); both STABLE |
| peak VRAM | 27.26 GB | **24.27 GB** | Unsloth lower by 2.98 GB |
| energy per step | 548.9 / 521.2 J | **435.7 / 412.0 J** | Unsloth ×0.79 |
| held-out at N = 20 | 0.8523 / 0.8520 | 0.8472 / 0.8504 | the matched set EQUIVALENT (P3 HELD) |

- **P5 HELD.** 0.817 lies in the registered [0.70, 1.20]. The interval is wholly below 1.0, so by the registered ordering reading
  **Unsloth is still faster on this card**.
- **P6 HELD.** e4b's reference is inside the fused arm's draw noise, and Unsloth is EQUIVALENT.
- e4b's step fell from 4.097 to 3.146 s (×0.768). Unsloth's moved from 2.546 to 2.571 s. The point estimate from the 5090 factors
  was ≈ 0.87; the H100 gave a little less.

**Where each step goes now** (the profiled arms on this box, descriptive):

| profiled matched arm | s/step | device-busy | device time / step | device events / step | CPU-side ops / step |
|---|---|---|---|---|---|
| e4b `fused_attn4_m_prof` | 3.182 | **0.665** (was 0.562) | ≈ 2.1 s | 97,851 (was 139,181) | 619,192 (was 744,465) |
| Unsloth `ckpt_unsloth_prof` | 2.861 | 0.305 | ≈ 0.87 s | 81,780 | 515,750 |

e4b's host work has fallen, so on this card it is now mostly device-bound: it spends about 2.4× Unsloth's device time per step.
That is the remaining gap. On sm_90 the comparator's dequantize-then-dense-grouped-GEMM path does the MoE arithmetic in far less
device time than e4b's fused NF4 kernels, and e4b's gradient checkpointing also recomputes every MoE forward. The 5090 positions
are unaffected (e4b faster there; TC1 amendment 19).

**The other arms.**
- axolotl 0.20.0 trained on this box this time: one VALID draw, axolotl/e4b 1.299, e4b faster. Its second draw did not run, so
  this is a reported row, not a position.
- HF + PEFT hit its alarm again with no training step (P4 of TC1c stays UNTESTED).

## The 2026-10-02 position, e4b before #945 (register `e4b.train.h2h.unsloth.qwen3.h100.2026-10-02`)

| | e4b `fused_attn4_m` | Unsloth `ckpt_unsloth_m` | reading |
|---|---|---|---|
| s/step, median of steps 11..20, two draws | 4.130 / 4.065 (**4.097**) | 2.539 / 2.553 (**2.546**) | **Unsloth/e4b 0.621 [0.615, 0.628]** over the four cross-draw ratios — **Unsloth faster per step by 1.61 ×**; both STABLE (1.6 % / 0.5 %) |
| tokens/s | 370.4 | 491.1 | |
| peak VRAM | 27.86 GB | **24.27 GB** | Unsloth lower by 3.59 GB |
| energy per step | 742.5 J | **466.8 J** | Unsloth ×0.63 |
| held-out loss at N = 20 | 1.9468 → 0.8476 / 0.8490 | 1.9305 → 0.8517 / 0.8498 | Δ +0.0041: **EQUIVALENT** (band 0.0078 = 3 × the in-draw fused-vs-reference Δ; floor 0.0019); step-0 Δ −0.0163 reads NEAR (the two quantisers' bytes on Hopper's kernels; ≤ 0.05) |

e4b's fused path against its own reference on the same box: Δ final 0.00414, median step |Δ| 0.00218 — PASS; ×12.58 faster than
the reference (51.94 s/step). The ratio's sign is the finding: **on the card where `torch._grouped_mm` is first-class, Unsloth's
dequant-then-dense-grouped-GEMM path beats e4b's fused kernel per step, at lower VRAM and lower energy, with the same loss.**
TC1c's registered P1 (Unsloth/e4b in [0.5, 2.0] and lower than on the 5090) holds on the number; its decision rule — below 1.0 the
public "e4b faster" position is withdrawn for this card class — applies. The 5090 position (1.437) stands as a 5090 position.

## Why the two cards disagree (what the profiles show)

| profiled matched arm | box | s/step | device-busy | device events / step | CPU-side ops / step |
|---|---|---|---|---|---|
| e4b `fused_attn4_m_prof` | RTX 5090 (Threadripper 3975WX) | 6.169 | 0.480 | 140,809 | 753,129 |
| e4b `fused_attn4_m_prof` | H100 NVL (EPYC 9534) | 4.306 | 0.562 | 139,181 | 744,465 |
| Unsloth `ckpt_unsloth_prof` | RTX 5090 | 9.868 | 0.234 | 612,179 | 7,430,931 |
| Unsloth `ckpt_unsloth_prof` | H100 NVL | 2.828 | 0.308 | 81,780 | 515,683 |

e4b issues the same ~139 k device events per step on both cards; Unsloth issues 612 k on the RTX 5090 and 82 k on the H100 — 7.5 ×
fewer — with 14 × fewer CPU-side ops. On sm_120 (torch 2.12.1+cu130) the grouped GEMM Unsloth routes through is not one launch per
call; on sm_90 it is. That is consistent with the two positions: on the 5090 both paths are launch-bound and e4b's single fused launch
wins; on the H100 Unsloth's path becomes a handful of large GEMMs and the dequant-then-GEMM arithmetic beats the fused NF4 kernel.
Stated as what the profiles show; which torch code path sm_120 takes is not established here.

## The other arms on this box

- HF + PEFT (bf16 experts, `target_parameters`, double-quant on the attention): loaded in 23.5 s at 62 GB resident — it fits on 80 GB,
  as TC1c predicted — then produced no training step inside its registered 1,800 s alarm: **ALARM, not measured** (P4 UNTESTED; the
  alarm was sized for the 5090's OOM, not for bf16 experts training through PEFT's weight-side fold). No HF position on this card.
- axolotl 0.20.0: its venv installed (amendment 3) and the arm died after load in transformers 5.17.0's Qwen3-MoE router **Corrected 2026-10-02 (TC1 amendment 4): the harness's no-autocast loop met axolotl's fp32 `.gate` router; not an axolotl result.**
  (`F.linear(hidden_states, self.weight)`: bf16 vs fp32) before writing a receipt — HARNESS_ERROR here, an UNSUPPORTED row with that
  message once TC3 amendment 4's wrapper is on the box. The same crash on the 24 GB box (`tc3-4090-1`).
- The mb1 secondary pair did not run (HF did not OOM); the native rows and the torch-2.8 row are not part of this token.

## Predictions scored (TC1's P-set on this box, mechanically; TC1c's own P1–P4 in the registration)

P1 (TC1's band) FALSIFIED at 0.621; P2 HELD (draws); P3 HELD (EQUIVALENT); P5 UNTESTED (ALARM, not OOM); P6 UNTESTED (HARNESS_ERROR);
P8 FALSIFIED (device-busy 0.308); P9 HELD (parity). TC1c: P1 HELD on the number ([0.5, 2.0], lower than the 5090) with its decision
rule applied; P2 HELD; P3 HELD; P4 UNTESTED (HF ALARM).
