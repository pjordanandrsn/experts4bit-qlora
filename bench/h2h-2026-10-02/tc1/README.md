# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, matched init and matched adapter precision (lanes TC1 and TC1b of #835; 2026-10-01/02)

Pre-registrations, read first: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md) (with its three dated amendments) and
[`../../tc1/TC1B-PREREG.md`](../../tc1/TC1B-PREREG.md). The harness is `bench/tc1/` at e4b `079a422` (0.38.0; the boxes'
`versions.txt`), grouped-nf4-gemm `846b512` (0.34.0). Upstream facts read from source: [`../../tc1/UPSTREAM-NOTES.md`](../../tc1/UPSTREAM-NOTES.md).
Design review and its disposition: [`../../tc1/DESIGN-REVIEW.md`](../../tc1/DESIGN-REVIEW.md). Receipts landed privately first (the
adertha receipt store, `receipts/experts4bit-qlora/2026-10-0{1,2}/<run id>/tc1/`) and are copied here under `receipts/<run id>/`
(the per-arm JSONs, `box.json`, `summary.txt`, `versions.txt`, the box-side `RESULTS-tc1.md`, `outer.log`, `logs/`; the tokenised
fixtures and the Unsloth compile cache are left in the private store).

| box | token | host (Vast verified-secure) | what | cost |
|---|---|---|---|---|
| `tc1-5090-16` | `qwen3` | instance 53773622, AMD Threadripper PRO 3975WX, driver 580.178.04 | the matched set, two interleaved draws, HF, the profiled pair, the mb1 secondary | $1.27 |
| `tc1-5090-14` | `qwen3native` | instance 53773863, Xeon E5-2698 v4, driver 595.71.05 | the labelled / native-best rows against the box's own e4b matched arm | $0.93 |
| `tc1-5090-15` | `qwen3curve` (TC1b) | instance 53772765, i9-13900, driver 595.91.07 | the 200-step matched curves, the as-shipped curve, the tp2/P38 anchor pair, the t1 and r64 scaling pairs | $0.56 |
| `tc1-5090-20` | `qwen3axolotl` (amendment 3) | instance 53812706, AMD EPYC 7663 56-Core Processor, driver 595.71.05 | the axolotl rows re-asked on their own box with a working venv, plus the HF torch-2.14 grouped_mm row at micro-batch 1 | $0.32 |
| `tc1-5090-33` | `qwen3nativebest` (amendment 5) | instance 53916526, AMD EPYC 7663 56-Core Processor, driver 580.95.05 | each framework's native-best, two draws each, e4b's matched arm as the anchor; P13 UNTESTED (a refused scattermoe second draw, e4b's pair 6.1 % apart) | $0.51 |
| `tc1-5090-34` | `qwen3nativebest` (amendment 7) | instance 53927504, AMD EPYC 9334 32-Core Processor, driver 590.48.01 | the same token again, with clocks recorded; P13 scored here | $0.44 |
| `tc1-5090-35` | `qwen3nativebest200` (amendment 8) | instance 53935033, Intel Xeon E5-2698 v4, driver 595.71.05 | e4b shipped vs axolotl scattermoe over 200 steps, two draws each, the 200-step matched anchor; P14 HELD, axolotl faster at steady state | $1.76 |
| `tc1-5090-36` | `qwen3nativebest200` (amendment 9) | instance 53971676, AMD Ryzen 9 3900X 12-Core Processor, driver 595.71.05 (machine 45501) | P14's box again on a second host, e4b pinned at 7c31b88; P15 HELD (0.901 [0.892, 0.910]) | $0.93 |
| `tc1-5090-38` | `qwen3syncab` (amendment 10, #945) | instance 53991731, AMD EPYC 7663 56-Core Processor, driver 580.95.05 | e4b against itself: 13 host syncs per MoE layer pass vs 1; P16 0.866, P17 0.847, both HELD | $0.42 |
| `tc1-5090-40` | `qwen3nativebest200` (amendment 11, #945) | instance 53999231, Intel Xeon E5-2696 v4 @ 2.20GHz, driver 595.91.07 | e4b shipped vs axolotl scattermoe over 200 steps after the sync fix; P18 UNTESTED (both pairs unstable, 12.0 % / 9.1 %); [read](RESULTS-tc1-steady945.md) | $1.50 |
| `tc1-5090-41` | `qwen3prof945` (amendment 12, #945) | instance 54000851, Intel Core Ultra 9 285K, driver 595.91.07 | e4b profiled after the sync fix, the legacy path as the before-picture; P19 HELD (matched device busy 0.595 → 0.740); [read](RESULTS-tc1-prof945.md) | $0.27 |
| `tc1-5090-42` | `qwen3leanab` (amendment 13, #945) | instance 54006898, AMD Ryzen 9 7900 12-Core Processor, driver 580.178.04 | e4b against itself: gnf4's previous padded LoRA delta vs its trimmed body (#440); P20 0.939, P21 0.911, both HELD; [read](RESULTS-tc1-leanab.md) | $0.33 |
| `tc1-5090-43` | `qwen3tileab` (amendment 14, #945) | instance 54011443, AMD Ryzen 9 7950X 16-Core Processor, driver 580.159.03 | e4b against itself: gnf4's max-keyed prefill M-tile vs the cost rule (#441); P22 0.924, P23 0.968, both HELD; [read](RESULTS-tc1-tileab.md) | $0.41 |
| `tc1-5090-44` | `qwen3rmsab` (amendment 15, #945) | instance 54018557 | HARNESS_ERROR before the command ran: the guard's Vast auth probe got one HTTP 429 (adertha-agents#138) | $0.00 |
| `tc1-5090-45` | `qwen3rmsab` (amendment 15, #945) | instance 54018669, AMD EPYC 7663 56-Core Processor, driver 580.95.05 | e4b against itself: the HF RMSNorm composite vs e4b's fused training RMSNorm (#961); P24 0.924, P25 0.959, P26 held-out within 0.0021, all HELD; [read](RESULTS-tc1-rmsab.md) | $0.39 |
| `tc1-5090-46` | `qwen3nativebest200` (amendment 16) | instance 54024744, AMD EPYC 7663 56-Core Processor, driver 580.95.05 | e4b shipped (after amendments 10-15) vs axolotl scattermoe over 200 steps; P27 HELD, axolotl / e4b 1.238 [1.231, 1.246], e4b faster at steady state; [read](RESULTS-tc1-steady16.md) | $0.99 |
| `tc1-5090-47` | `qwen3nativebest200` (amendment 17) | instance 54037643, machine 142284 again (AMD EPYC 7663) | the same box; a same-machine repeat, P28 UNTESTED by rule; reads 1.253 [1.248, 1.258]; [read](RESULTS-tc1-steady18.md) | $0.98 |
| `tc1-5090-48` | `qwen3nativebest200` (amendment 18) | instance 54051454, machine 150527, AMD EPYC 7C13 64-Core Processor, driver 580.95.05 | the same box with machine 142284 excluded by evidence; P29 HELD, 1.146 [1.129, 1.164]: e4b faster at steady state on a second host; [read](RESULTS-tc1-steady18.md) | $1.23 |
| `tc1-5090-49` | `qwen3` (amendment 19) | instance 54075516, AMD EPYC 7C13 64-Core Processor, driver 580.95.05 | the matched set again with e4b after amendments 10-15: Unsloth/e4b 1.997 (P30 HELD), matched set inside the draw noise (P31 HELD); [read](RESULTS-tc1-matched19.md) | $1.15 |
| `tc1-5090-50` | `qwen3axolotl` (amendment 19) | instance 54075633, AMD Ryzen 9 9950X3D 16-Core Processor, driver 610.57.04 | the axolotl matched rows again: axolotl/e4b 2.775 (P32 HELD); [read](RESULTS-tc1-matched19.md) | $0.33 |
| `tc1-5090-59` | `qwen3denseab` (amendment 22) | instance 54177463, AMD EPYC 9454P | e4b against itself: grouped-nf4-gemm's fused kernels vs its dense route: dense/fused 2.947, far slower (P39 FALSIFIED); [read](RESULTS-tc1-denseab.md) | $0.45 |
| `tc1-5090-62` | `mixtraldenseab` (amendment 22) | instance 54179030, Intel Core Ultra 9 285K | the same A/B on Mixtral-8x7B resident (`E4B_ABSMAX_DQ=1` both sides): dense/fused 0.651 (P38 HELD); held-out within 0.01 on both families (P40 HELD); [read](RESULTS-tc1-denseab.md) | $0.51 |
| `tc1-5090-65` | `qwen3memcensus` (amendment 23) | instance 54195008, AMD EPYC 9454P | a memory census, no speed read: e4b fp32 absmax, e4b `E4B_ABSMAX_DQ=1`, Unsloth at micro-batch 1; e4b dq − Unsloth +0.43 GB, all transient (P41, P42 HELD; P43 FALSIFIED); [read](RESULTS-tc1-memcensus.md) | $0.38 |
| `tc1-5090-66` | `qwen3bmmab` (amendment 24) | instance 54201179, AMD EPYC 7B13 | a bmm replay (no model), then e4b in its field environment vs Unsloth's: the matched arm 0.882× in torch 2.12.1+cu130 (P47 HELD); the fp32 bmm's cost is per new shape and the same on both torch versions (P44, P46 FALSIFIED; P45 HELD); [read](RESULTS-tc1-bmmab.md) | $1.26 |
| `tc1-5090-67` | `qwen3samestack` (amendment 25) | instance 54209084, Intel Core Ultra 9 285K | both frameworks on one stack: the matched set EQUIVALENT, parity PASS (P52 HELD); Unsloth's draws 18.2 % apart and e4b's field-image draws 8.4 % apart, so no ratio is read (P50, P51 UNTESTED); one re-draw registered (amendment 27); [read](RESULTS-tc1-samestack-box1.md) | $0.61 |
| `tc1-5090-68` | `qwen3samestack` (amendment 27, the re-draw) | instance 54214853, AMD EPYC 7663 | the matched set INSIDE-DRAW-NOISE, parity PASS (P52 HELD again); e4b's same-stack draws 7.8 % apart, so no ratio is read (P50, P51 UNTESTED, final); amendment 29 re-asks them over 60 steps; [read](RESULTS-tc1-samestack-box2.md) | $1.86 |
| `tc1-5090-69` | `qwen3prebindab` (amendment 26) | instance 54223080, Intel Core Ultra 9 285K | e4b against itself, prebound Triton launches off vs on: matched arm 0.973 (P54 HELD); the shipped arm's flags-off draws 8.3 % apart (P53, P55 UNTESTED); the flags stay opt-in; amendment 30 re-asks the shipped pair over 60 steps; [read](RESULTS-tc1-prebindab.md) | $0.43 |
| `tc1-5090-70` | `qwen3dqab` (amendment 28) | instance 54223173, AMD EPYC 7C13 (Vast machine 45511) | e4b against itself, the expert absmax fp32 vs double-quantized: peak 27.15 → 25.82 GB and held-out +0.0026, but both speed pairs unstable under host load (P56 UNTESTED); amendment 31 re-asks it over 60 steps; [read](RESULTS-tc1-dqab.md) | $1.13 |
| `tc1-5090-71` | `mixtraldqab` (amendment 28) | instance 54223342, AMD EPYC 7B13 (Vast machine 145701) | the same on Mixtral-8x7B resident: dq1/dq0 1.023, peak 31.07 → 29.03 GB, held-out −0.0028 (P57 HELD); [read](RESULTS-tc1-dqab.md) | $1.27 |
| `tc1-5090-72` | `qwen3samestack` (amendment 29, 60 steps) | instance 54227048, AMD EPYC 7C13 (Vast machine 45511) | the same-stack pair over 60 steps: e4b's same-stack draws 13.1 % apart and its field-image draws 6.4 %, host load1 6–23, so no ratio is read (P50, P51 UNTESTED, final); amendment 33 re-asks with load-gated draws; [read](RESULTS-tc1-samestack-box3.md) | $1.75 |
| `tc1-5090-73` | `qwen3prebindab` (amendment 30, 60 steps) | instance 54227619, AMD EPYC 7B13 (Vast machine 145701) | the shipped arm's prebind pair over 60 steps: `_pb1`/`_pb0` 0.980 [0.957, 1.003] (P53 HELD), held-out +0.0011; with amendment 26's P54 the prebound launches become the default |
| `tc1-5090-75` | `qwen3dqab` (amendment 31, 60 steps) | instance 54238515, AMD EPYC 7B13 (Vast machine 145701) | Qwen3-30B-A3B's absmax pair over 60 steps on a quiet host: dq1/dq0 1.014, peak 27.44 → 26.10 GB, held-out −0.0021 (P56 HELD; with P57 and P58, amendment 28's rule makes the double-quantized absmax the default); [read](RESULTS-tc1-dqab-qwen3-60.md) | $1.03 |
| `tc1-5090-74` | `qwen3tritonab` (amendment 32, 60 steps) | instance 54237146, AMD EPYC 7B13 (Vast machine 145701) | one variable, triton 3.4 vs 3.7.1 in venv-e4b: matched arm 0.992 (P59 FALSIFIED), shipped arm 0.971 (P60 HELD), held-out within 0.001 (P61 HELD); the environment gain is not triton's on this host-bound step; amendment 34 splits it; [read](RESULTS-tc1-tritonab.md) | $1.28 |
| `tc1-5090-76` | `qwen3samestack` (amendment 33, 60 steps, load-gated) | instance 54239673, AMD EPYC 7B13 (Vast machine 145701) | both frameworks on one stack, every pair stable: Unsloth/e4b 2.352 (P50 HELD), e4b same-stack / field-image 0.900 (P51 HELD); three draws voided for host load and run again; by amendment 25's rule 2.352 becomes the quoted Qwen3-30B-A3B position; [read](RESULTS-tc1-samestack-box4.md) | $1.76 |
| `tc1-5090-78` | `qwen3envsplit` (amendment 34, 60 steps, load-gated) | instance 54248510, AMD EPYC 7B13 (Vast machine 145701) | e4b's matched arm in three environments: transformers 5.5 vs 5.18 on torch 2.8 1.005 (P62 FALSIFIED), torch 2.12 + triton 3.7 vs torch 2.8 + triton 3.4 0.905 (P63 HELD), the whole environment 0.909 (P64 HELD); the environment gain is torch's; [read](RESULTS-tc1-envsplit.md) | $1.73 |
| `tc1-5090-79` | `qwen3prebind37` (amendment 35, 60 steps, load-gated) | instance 54255834, AMD EPYC 7B13 (Vast machine 145701) | amendment 26's prebind A/B under triton 3.7.1 in venv-unsloth: shipped 0.996 (P66 HELD), matched 0.986 (P67 HELD), held-out within 0.003 (P68 HELD); triton 3.7 stays in the prebound path's supported versions; [read](RESULTS-tc1-prebind37.md) | $1.68 |
| `tc1-5090-80` | `qwen3compactab` (amendment 36, 60 steps, load-gated) | instance 54259219, AMD EPYC 7B13 (Vast machine 145701) | grouped-nf4-gemm's compact padded LoRA delta off vs on in venv-unsloth: the matched peak ROSE 0.23 GB (P69 FALSIFIED), and the step got faster than registered, matched 0.969 and shipped 0.970 (P70, P71 FALSIFIED); held-out within 0.001 (P72 HELD); it stays opt-in; [read](RESULTS-tc1-compactab.md) | $1.48 |
| `tc1-5090-83` | `qwen3compactab2` (amendment 37, 60 steps, load-gated) | instance 54275103, AMD EPYC 7702P (Vast machine 45379) | the compact delta again with grouped-nf4-gemm#473's backward, on another host: matched peak −0.288 GB (P73 HELD), matched 0.967 (P74 HELD), shipped 0.948, faster than its band (P75 FALSIFIED), held-out within 0.003 (P76 HELD); by the rule it stays opt-in pending its own registration; [read](RESULTS-tc1-compactab2.md) | $0.78 |
| `tc1-5090-85` | `qwen3compactab3` + `mixtralcompactab` (amendment 38, 60 steps, load-gated) | instance 54292473, AMD EPYC 9655 (Vast machine 150700) | the compact delta's default decision on a fast host: slower on every arm, Qwen3 matched 1.016 and shipped 1.013, Mixtral 1.013 (P77, P78, P80 FALSIFIED), while the peaks held (Qwen3 −0.312 GB, Mixtral −0.037; P79, P81 HELD); it stays opt-in; [read](RESULTS-tc1-compact-default.md) | $2.64 |
| `tc1-5090-86` | `qwen3samestack4k` (amendment 39, packed 4,096-token rows, 30 steps, load-gated) | instance 54297512, AMD EPYC 7B13 (Vast machine 145701) | the packed regime on one stack: every e4b arm OOMed at step 1 allocating 2.32 GiB, the fp32 copy of the full-vocabulary logits (P86 FALSIFIED), while Unsloth trained at 24.86 GB (its draws 7.7 % apart; P84, P85 UNTESTED); an e4b loss in that regime; [read](RESULTS-tc1-packed4k.md) | $1.37 |
| `tc1-5090-91` | `qwen3samestack4kce` (amendment 40, packed 4,096-token rows, e4b with `E4B_CHUNKED_LM_LOSS=1`, 40 steps) | instance 54323622, AMD EPYC 7K62 (Vast machine 152440) | with the chunked loss every e4b arm trained the packed rows (peak 32.44–32.57 GB, no OOM), but each is VOID under TC1's no-loop rule: grouped-nf4-gemm's `auto` took the per-expert LoRA loop on ~1.5 % of its delta calls (padded blocks over its 2 GiB limit); P87–P89 UNTESTED; the unquotable readings say Unsloth/e4b 1.43; [read](RESULTS-tc1-packed4k-chunked.md) | $1.43 |
| `tc1-5090-94` | `qwen3samestackh2` (amendment 42, 60 steps, load-gated) | instance 54328898, AMD EPYC 7K62 (Vast machine 152440) | amendment 33's same-stack box on a second host, the current code: Unsloth/e4b 2.468 (P94 HELD) and the environment 0.870 (P95 HELD); 2.352 stays the position to quote, now read on two hosts; [read](RESULTS-tc1-samestack-host2.md) | $1.25 |

Thirteen earlier draws were refused or stopped before producing a row (driver floor, pre-flight bandwidth, a controller-slot
race, the cu130 pip resolver — TC1 amendments 1 and 2) for about $0.57 in total, and the first axolotl box (`tc1-5090-19`) was
lost at 12 minutes when its controller was killed on the operator's side (the launcher's guard destroyed the instance on heartbeat
loss; about $0.14, no receipt); every one is a receipt or a guard record in the private store and none is quoted. Verdicts, validity, positions, equivalence bands and the prediction scoring are the reducer's
([`../../tc1/tc1_reduce.py`](../../tc1/tc1_reduce.py)) and are reproduced here from the receipts:
[`RESULTS-tc1-combined.md`](RESULTS-tc1-combined.md) (the four boxes in one pass, the amendment-3 reducer) and
[`RESULTS-tc1b-vs-tc1.md`](RESULTS-tc1b-vs-tc1.md) (TC1b read against the matched box with `--tc1-dir`).

## Amendment 46 (2026-10-06): grouped-nf4-gemm's decoded route shows no measurable step-time saving on OLMoE (1.005 [0.979, 1.031]) and costs Qwen3-30B-A3B 6.6 %; `auto` stays as it is

Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendment 46, after lane RD1's per-call read
([`../../moegen/rd1/RESULTS-rd1.md`](../../moegen/rd1/RESULTS-rd1.md)).
- **The box:** one RTX 5090 (`tc1dec-5090-4`, AMD EPYC 7713, Vast machine 55913).
- **Software:** venv-e4b (torch 2.8.0+cu128, triton 3.4.0, RD1's software), with e4b `ac3adf9` and grouped-nf4-gemm `aaefbf8` (#487's merge).
- **Steps:** 60 load-gated steps on a quiet host. Every arm ran on its first attempt, at load1 medians of 1.9–3.1.
- **Arms:** `_dec0` is the fused kernels (`GNF4_TRAIN_GEMM=fused`); `_dec1` is the decoded route (`=decoded`, 256 MiB cap).
- **Read:** [`RESULTS-tc1-decoded.md`](RESULTS-tc1-decoded.md).

| family | `_dec0` s/step | `_dec1` s/step | `_dec1` / `_dec0` | peak `_dec0` → `_dec1` | prediction |
|---|---|---|---|---|---|
| OLMoE-1B-7B (`olmoedecab`) | 1.040 / 1.026 | 1.018 / 1.058 | **1.005** [0.979, 1.031] | 7.67 → 7.67 GB | P108 **FALSIFIED** (≤ 0.95 on the median and every cross-draw ratio) |
| Qwen3-30B-A3B (`qwen3decab`) | 3.682 / 3.647 | 3.943 / 3.868 | **1.066** [1.050, 1.081] | 27.45 → 27.45 GB | P109 HELD ([0.97, 1.25]) |

- **P107 HELD: the gate ran first and passed on this card.** grouped-nf4-gemm's compiled tests for the route at `aaefbf8` ran 24 + 38
  tests with no failure, error or skip, and every required test passed (`receipts/tc1dec-5090-4/decgate.json`).
- **P110 HELD:** held-out at N moved +0.0021 (OLMoE) and −0.0015 (Qwen3-30B-A3B).
- **P111 HELD:** the matched peaks are unchanged.
- **Engagement.** Every arm is VALID.
  - Each `_dec1` arm counted decoded forward and dgrad calls, and no dense ones: OLMoE 16,384 / 7,680, Qwen3-30B-A3B 49,152 / 23,040.
  - Each `_dec0` arm counted none.
- **Decision, as registered: P108 FALSIFIED.** grouped-nf4-gemm's `auto` does not change, and `GNF4_TRAIN_GEMM=decoded` stays opt-in.
- **An observation, not a registered line.** RD1's per-call win on OLMoE's expert shapes (0.79 × the fused kernels at 512 skewed rows)
  shows no measurable gain on the step: the interval [0.979, 1.031] cannot exclude a saving of about 2 %, nor a loss of 3 %, and
  `_dec1`'s two draws are 3.9 % apart. It does exclude the registered 5 % gain.
  On Qwen3-30B-A3B the step's cost (1.066) sits at the low end of RD1's per-call loss at 512 rows (1.05 skewed, 1.35 uniform). Nothing here
  measures where the step's time goes, so why the per-call win is lost stays open.
- **Attempts.**
  - `tc1dec-5090-1` and `-2` were refused at $0 by the receipt store, with no box.
    - `-1`: SV2's receipt commit was unpushed because the mini's GitHub https credentials had failed.
    - `-2`: this lane's own commit was briefly unpushed.
  - `-3` was NOT_RUN ($0.007): a Vast API SSL timeout in pre-flight.
  - `-4` is the reading: $1.07 invoiced, of which $0.52 was the 80 GB download.
  - While `-4` ran, its gate checkout's nested `receipt.json` reached the receipt store and refused other launches. #1220 keeps the
    checkout off every fetch, and adertha#178 makes the reconciler read only `<date>/<run>/receipt.json`.

## Amendment 57 (2026-10-07): e4b's 1.92 GB training-phase excess on packed rows is checkpoint activations kept on the GPU, the combine backward's fp32 temporaries and the bucketed delta's block (P149–P152 HELD)

Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendment 57. One RTX 5090 (`tc1-5090-114`, AMD EPYC 7713, Vast
machine 55913). e4b `c07ea7f`, grouped-nf4-gemm `0e6bff3`, Unsloth 2026.9.14, every arm in venv-unsloth (torch 2.12.1). This is amendment
47's census on packed 4,096-token rows with no evaluation inside the census window. Read: [`RESULTS-tc1-memc4kt.md`](RESULTS-tc1-memc4kt.md).

| arm | training-phase peak | where | e4b − Unsloth at the peak, by class |
|---|---|---|---|
| e4b, `E4B_ABSMAX_DQ=0` | 28.14 GB | `s17.mb3.backward` | transient +1.92, expert absmax +1.35 |
| e4b, `E4B_ABSMAX_DQ=1` | 26.79 GB | `s10.mb3.backward` | transient +1.91 |
| Unsloth (`--grad-ckpt unsloth`) | 24.86 GB | `s2.mb2.backward` | — |

- **P149 HELD:** 100 % of each peak is attributed. **P150 HELD:** e4b fp32 is 3.274 GB above Unsloth (band [2.0, 4.5]). **P151 HELD:** e4b
  with the double-quantized absmax is 1.922 GB above (≤ 2.5). **P152 HELD:** every census peak falls in a training backward. The peaks
  equal amendment 56's phase peaks to the MB.
- **What the 1.91 GB is.** Every static class matches Unsloth's at the peak, so the excess is all transient. e4b's largest non-static groups
  are:
  - **checkpoint activations, 0.789 GB:** `modeling_qwen3_moe.py:352`, each decoder layer's output (`residual + hidden_states`), 47 blocks
    of 16.8 MB, i.e. the inputs that non-reentrant checkpointing saves for the next layer. e4b runs Hugging Face's checkpointing; TC1's
    Unsloth arm runs Unsloth's own (`unsloth`), and its census holds no such group.
  - **the routed-expert combine's backward, 1.07 GB across three lines:** `engines/fast.py:429`, `:430` and `:435`. These are fp32
    copies of the incoming gradient expanded to `[tokens × top-k, hidden]` (268 MB each), its reordered copy, and the weight-gradient
    product. This is e4b's own code and the largest single function.
  - **grouped-nf4-gemm's bucketed delta, 0.90 GB:** `nf4_qlora.py:672`, `:673`, the zero-padded fp32 input block and its products.
- **By amendment 57's rule** the next registration targets the largest e4b-only group, the checkpoint activations. The combine backward is
  larger as a function and is e4b's own; it materialises two fp32 copies where one indexed read would do. Both are candidates for the next
  box.

## Amendment 56 (2026-10-07): the double-quantized absmax costs packed rows 0.2 % for 1.35 GB; e4b's training phase peaks 1.92 GB above Unsloth's with it (P144–P146, P148 HELD; P147 FALSIFIED)

Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendment 56. One RTX 5090 (`tc1-5090-113`, AMD EPYC 7713, a 61-CPU
quota, Vast machine 19317). e4b `52acaa3`, grouped-nf4-gemm `0e6bff3`, Unsloth 2026.9.14, every arm in venv-unsloth (torch 2.12.1, triton
3.7.1). Packed 4,096-token rows, 40 load-gated steps, every attempt first time at load1 1.3–1.4. Each arm's peak was split by phase
(`--phase-peaks 1`). Read: [`RESULTS-tc1-dqpack.md`](RESULTS-tc1-dqpack.md).

| arm | s/step (two draws) | run peak | setup | evaluation | training |
|---|---|---|---|---|---|
| e4b, fp32 expert absmax (`a0`, the library default) | 10.513 / 10.526 | 28.23 GB | 21.86 | 28.23 | 28.14 |
| e4b, `E4B_ABSMAX_DQ=1` (`a1`) | 10.530 / 10.543 | 26.88 GB | 21.86 | 26.88 | 26.78 / 26.79 |
| Unsloth `ckpt_unsloth_m_pp` (one draw) | 14.085 | 24.86 GB | 21.14 | 21.85 | 24.86 |

- **P144 HELD:** the double-quantized absmax steps **1.002** [1.000, 1.003] of the fp32 absmax's time (≤ 1.02). **P145 HELD:** the run
  peak falls 1.351 GB (≥ 1.2). **P146 HELD:** held-out at N moves +0.0002.
- **P147 FALSIFIED:** with the double-quantized absmax, e4b's training-phase peak is **26.79 GB against Unsloth's 24.86: +1.92 GB**
  (registered ≤ +1.0).
- **P148 HELD, narrowly:** on every e4b draw the evaluation phase peaks above the training phase, but only by about 0.09 GB (28.23 against
  28.14 with the fp32 absmax).
- **This corrects how amendment 55 was read.** e4b's run peak is its evaluation, as the census found, but its training steps reach almost
  the same height. The packed-row gap to Unsloth lives in training, not in the evaluation's logits. Unsloth's own evaluation peaks at
  21.85 GB, 0.71 GB over its setup.
- **By amendment 56's rules:**
  - P144–P146 HELD: a library PR makes `enable_fast_train` compress the expert absmax by default for resident training, with the
    trainer's guards and the stated evidence scope (one model, one RTX 5090, torch 2.12 / triton 3.7; the field image's torch 2.8 is unread).
  - P147 FALSIFIED: the next registration is a census of the training phase, to name the 1.92 GB.

## Amendment 55 (2026-10-06): on packed rows e4b's run peak is its held-out evaluation, not its training step; with the double-quantized absmax the gap to Unsloth is 2.01 GB (P141–P143 HELD)

Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendment 55. One RTX 5090 (`tc1-5090-112`, AMD EPYC 7K62, a 23-CPU
quota, Vast machine 152440). e4b `dde06a9`, grouped-nf4-gemm `71847e5`, Unsloth 2026.9.14, every arm in venv-unsloth (torch 2.12.1).
Amendment 47's census box at the current defaults: packed 4,096-token rows, 20 steps, one draw per arm, `--mem-census 1`. Read:
[`RESULTS-tc1-memc4kb.md`](RESULTS-tc1-memc4kb.md).

| arm | peak allocated | where the peak falls | expert absmax at the peak | largest non-static groups at the peak |
|---|---|---|---|---|
| e4b defaults | 28.23 GB | `s20.eval`, the held-out evaluation after step 20 | 1.81 GB (fp32) | the stock LM loss: `ForCausalLMLoss` 2.49 GB (the fp32 logits), `cross_entropy` 2.49 GB, the head's bf16 output 1.28 GB |
| e4b, `E4B_ABSMAX_DQ=1` | 26.88 GB | `s20.eval` | 0.46 GB | the same three |
| Unsloth | 24.86 GB | `s2.mb2.backward`, a training step | 0.46 GB | `nf4_dequant_triton` 0.81 GB, autograd 0.29 GB |

- **P141 HELD:** the census attributes 100 % of each arm's peak. **P142 HELD:** e4b's defaults peak 3.364 GB above Unsloth's (band
  [2.0, 5.0]). **P143 HELD:** with the double-quantized absmax the gap is 2.013 GB (≤ 2.5).
- **The bucketed delta is not the remaining excess.** By class at the peak, e4b − Unsloth is transient +4.58 GB and expert absmax
  +1.35 GB, offset by adapter gradients −2.57 GB. Unsloth's peak falls in a backward pass, where the gradients are live; e4b's falls in
  the evaluation after them. grouped-nf4-gemm's groups hold under 0.001 GB at e4b's peak.
- **Why e4b peaks in evaluation.** e4b's chunked LM loss takes training forwards only. A `torch.no_grad` evaluation runs the stock
  forward on purpose: the held-out loss stays the stock path's bit for bit, and a caller that reads the logits still gets them. On a
  4,096-token row that stock forward holds the fp32 logits, cross-entropy's working copy and the bf16 head output at once: about 6.26
  GB on top of the static 21.99 GB. So the 28.23 GB packed-defaults peak (amendment 51) is e4b's evaluation, set against Unsloth's
  training step. This census keeps only the snapshot that set the run's peak, so e4b's training-phase peak is not recorded here. (Amendment 56 recorded it: 28.14 GB, about 0.09 GB below the evaluation. The gap is in training.)
- **By amendment 55's rule** (P143 HELD), the next registration asks whether the library should default to the double-quantized absmax
  on packed rows, reading its speed there the way amendment 28 read it at the field recipe. The same box should record each arm's
  training-phase and evaluation peaks separately, so the packed memory comparison can be made phase for phase.

## Amendment 54 (2026-10-06): neither remedy moves the step on a host where torch 2.8 is not host-bound (P137, P138 FALSIFIED); the ladder does remove `bmm`'s per-shape host cost

Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendment 54. One RTX 5090 (`tc1-5090-111`, AMD Ryzen Threadripper
PRO 7965WX, a 23-CPU quota, Vast machine 151831). e4b `aeb5bd6`, grouped-nf4-gemm `71847e5` (#498, the ladder), packed 4,096-token rows,
40 load-gated steps, every attempt first time at load1 3.2–4.6. It ran e4b's matched arm in venv-e4b (torch 2.8.0, triton 3.4.0) at its
defaults, profiled on steps 3–5. Read: [`RESULTS-tc1-ladder28.md`](RESULTS-tc1-ladder28.md).

| side | s/step, timed (two draws) | peak | device ms per step | `aten::bmm` CPU self time per call | nvidia-smi median util |
|---|---|---|---|---|---|
| `c0`: the defaults | 10.507 / 10.492 | 28.18 GB | 10,402 / 10,410 | ~145 / ~150 µs | 98 % |
| `c1`: `CUBLASLT_HEURISTICS_CACHE_CAPACITY=262144` | 10.477 / 10.446 | 28.18 GB | 10,403 / 10,451 | ~136 / ~139 µs | 98 % |
| `c2`: `NF4_QLORA_PAD_BUCKETS_LADDER=1` | 10.584 / 10.632 | 28.50 / 28.53 GB | 10,583 / 10,646 | ~14 / ~17 µs | 99 % |

- **P137 FALSIFIED: the ladder steps 1.010** [1.007, 1.013] of the defaults' time (registered ≤ 0.95).
- **P138 FALSIFIED: the bigger cuBLASLt cache steps 0.996** [0.994, 0.999] (registered ≤ 0.97).
- **P139 HELD:** held-out at N moves −0.0002 (`c1`) and −0.0001 (`c2`). **P140 HELD:** the ladder's matched peak rises 0.335 GB (≤ 1.5).
- **This host was not host-bound at the defaults, so neither remedy had host time to recover.** `c0`'s device time is 99 % of its timed
  step, and nvidia-smi reads 98 %. The same configuration with the same code read 12.55 s on amendment 53's host, a Threadripper PRO
  3955WX with a 15-CPU quota: 0.850 of the step was device time there. Here it is 10.50 s. Device time differs by 2.5 % between the two,
  so nearly all of the difference is host idle. These are cross-host comparisons, reported, not scored. Torch 2.8's extra time at e4b's
  defaults on packed rows depends on the host.
- **The ladder does what it was built for**, reported, not scored. It cuts `aten::bmm`'s CPU self time per call about tenfold, from ~145–150 µs
  to ~14–17 µs, over the same ~26,700 calls a step. On this GPU-bound host the saved time reappears as waiting: `cudaMemcpyAsync` self time
  rises 3.3 s per profiled step. That fits amendment 24's per-new-shape cost and amendment 53's caution that self time counts waits on a
  full queue. The ladder's extra padding costs about 2 % of device time, which is the 1.010.
- **By amendment 54's rule** (P137 and P138 both FALSIFIED), the next candidate is fewer, wider buckets. The rule assumed a falsification
  would mean shape novelty is not the in-training cost. The profile says the opposite: the per-shape cost is real, and this host had no host
  bottleneck for it to matter. So the next registration takes the registered candidate and gates validity on the defaults being host-bound,
  so that either remedy can be read where the cost exists. No default changed; the ladder stays opt-in.

## Amendment 53 (2026-10-06): 60 % of torch 2.8's extra time at e4b's defaults is not device time, most visibly in the bucketed delta's batched matmuls; 40 % is grouped-nf4-gemm's own kernels

Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendment 53. One RTX 5090 (`tc1-5090-110`, AMD Ryzen Threadripper
PRO 3955WX, a 15-CPU quota, Vast machine 26157). e4b `3c8793f`, grouped-nf4-gemm 0.42.0 (`427a771`), packed 4,096-token rows, 40 load-gated
steps, every attempt first time at load1 1.9–2.2. It ran e4b's matched arm at its defaults with the profile instrument (steps 3–5) in
three environments. Read: [`RESULTS-tc1-prof28.md`](RESULTS-tc1-prof28.md).

| side | s/step, timed (two draws) | device ms per step (CUPTI) | device / timed step | nvidia-smi median util |
|---|---|---|---|---|
| `q212`: venv-unsloth (torch 2.12.1, triton 3.7.1), buckets `auto` | 9.924 / 9.914 | 9,616 / 9,597 | 0.969 | 96 % |
| `q28`: venv-e4b (torch 2.8.0, triton 3.4.0), buckets `auto` | 12.562 / 12.535 | 10,669 / 10,665 | 0.850 | 82 % |
| `q28k0`: venv-e4b, single block | 12.337 / 12.375 | 11,920 / 11,927 | 0.965 | 98 % |

- **P134 HELD: environment ratio 0.790** [0.789, 0.792] (≤ 0.92). On this host torch 2.8 at e4b's defaults steps 2.63 s slower than torch
  2.12. That is less than amendment 51's 0.739, more than amendment 43's 0.915 without buckets.
- **P135 HELD: 59.7 % of that gap is not device time** (≥ 0.5). Torch 2.8 adds 2,630 ms per timed step and 1,061 ms of device time. So
  1,569 ms is the host's, with the device waiting on it.
- **P136 HELD:** in torch 2.8 the buckets lower the device's share of the step from 0.965 (single block) to 0.850 (drop 0.115 ≥ 0.03).
  They also take 1,256 ms of device time off the step, and the host gives most of it back.
- **The host time is `aten::bmm`.** The profiler's CPU self time by family puts `matmul` first by a wide margin: +4,817 ms per profiled
  step from `q212` to `q28`, against +22 for the next family. Within it, `aten::bmm` makes the same number of calls in both torches, about
  26,750 per step. The single block makes 3,012, so the buckets' per-bucket products add about 23,750. Its self time per call is about
  86 µs in torch 2.12 and about 268 µs in torch 2.8. Profiled times carry the profiler's overhead in both torches, so they are reported,
  not scored. Launch time barely moved (`cudaLaunchKernel` 525 against 558 ms per profiled step, over 79,786 and 89,567 calls). cuBLAS
  picks the same kernel for the fp32 products in both (`cutlass_80_simt_sgemm_64x64_8x5_tn_align1`, about 8,270 a step). So the added
  host time is inside the batched-matmul call, not in launching its kernel. One caution on the size of it (maintainer review): CPU self
  time also counts any wait on a full CUDA launch queue. The single block's 3,012 `aten::bmm` calls read about 1.4 ms of self time each
  in torch 2.8 while its device is 96.5 % busy, which is mostly back-pressure, not host work. So 268 µs per call is an upper bound on the
  bmm's host work in `q28`, and P135's 59.7 % (not device time) is the measured share. Profiled per-family totals can rank the host
  families; they cannot price them.
- **The device time is grouped-nf4-gemm's own Triton kernels**, and it does not depend on the buckets. Torch 2.8's device time grows by
  1,061 ms per step, 989 of them in `fused_kernel`:
  - `_gemm_nf4_grouped` takes 1,995 ms per step in `q212` and 2,439 in `q28`;
  - `_dgrad_nf4_grouped` takes 1,038 and 1,582;
  - with the single block in torch 2.8 they read the same, 2,445 and 1,584.

  The environments differ in torch and triton together, so this box does not split the two. Triton's code generation is the candidate.
  Amendment 32 read triton 3.7.1 alone in venv-e4b at the field recipe, where the step is host-bound: 0.992 matched and 0.971 shipped.
- **Against amendment 52**, reported, not scored. On a different host, `q28` steps 12.55 s against amendment 52's bucketed matched arm
  at 12.05, and `q28k0` 12.36 against its single block at 12.25. The nvidia-smi medians repeat amendment 52's pattern: 82 % against 98 %
  here, 87 % against 97 % there.
- **By amendment 53's rule** the next registration targets the CPU family that grew most: the bucketed delta's batched matmuls. The
  registered candidate is a shape-stable bucket ladder in grouped-nf4-gemm. Bucket widths would be rounded up to a fixed set, so repeated
  calls give the batched matmul repeated shapes. It would be read on packed rows in both torches. The device-side 40 % is a second lead:
  amendment 32's triton A/B, on packed rows. No default changed on this box.

## Amendment 52 (2026-10-06): bucketed padding costs torch 2.8 nothing on packed rows; P127's gap lies elsewhere

Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendment 52. One RTX 5090 (`tc1-5090-109`, AMD EPYC 7713, no CPU
quota, Vast machine 55913), amendment 48's packed box in the field image's venv-e4b (torch 2.8.0+cu128, triton 3.4.0). e4b `a41c857`,
grouped-nf4-gemm 0.42.0 (`b4f93f1`), packed 4,096-token rows, 40 load-gated steps, e4b's defaults otherwise (the chunked loss `auto`),
`NF4_QLORA_PAD_BUCKETS=0` (`_k0`) against `=1` (`_k1`), two draws a side in ABBA order. Read:
[`RESULTS-tc1-padbk28.md`](RESULTS-tc1-padbk28.md).

| arm | `_k0` s/step (two draws) | `_k1` s/step (two draws) | `_k1` / `_k0` | peak `_k0` → `_k1` |
|---|---|---|---|---|
| matched (fp32 adapters) | 12.219 / 12.287 | 12.129 / 11.971 | **0.983** [0.974, 0.993] | 32.42 → 28.18 GB |
| shipped (bf16 adapters) | 9.598 / 9.760 | 9.071 / 9.106 | **0.939** [0.929, 0.949] | 26.92 → 26.92 GB |

- **P130 HELD** (matched ≤ 1.02) and **P131 HELD** (shipped ≤ 1.02): under torch 2.8 the buckets are no slower than the single block on
  packed rows. **P132 HELD**: the matched peak drops 4.239 GB (≥ 3.0). **P133 HELD**: held-out at N moves −0.0002 (matched) and +0.0002
  (shipped). Every arm VALID; the `_k1` arms bucketed every padded call (about 31,750 a run), the `_k0` arms none.
- **By amendment 52's rule** grouped-nf4-gemm's `auto` default stands for torch 2.8 too, and P127's environment ratio (0.739, amendment 51)
  is not the buckets' cost.
- **Load.** Five attempts ran above the 6.0 gate and were run again; one standing attempt is above it (matched `_k0` second draw, load1
  7.11 on its last retry). Its draw is within 0.6 % of its pair.
- **Where to look next (reported, not scored).** The GPU traces show the buckets saving GPU work under torch 2.8 and giving most of it back
  as idle time. Over each arm's training window the median GPU utilisation was 97 % on both `_k0` arms and 87 % on both `_k1` arms (median
  power 485–488 W against 424–434 W); the shipped arms read 97 % against 96 %. Amendment 51's host shows the same pattern larger: e4b at
  its defaults ran at 97–98 % in torch 2.12 and 75–76 % in torch 2.8. Without buckets, torch 2.8 cost amendment 43's box 9 % (0.915);
  with them, torch 2.12 gains about 10 % (amendment 48) and torch 2.8 here 1.7 % (matched). So the open question is host-side time in
  the bucketed delta under torch 2.8 / triton 3.4. An RTX A2000 can count that host-side work at no rental cost (launches, syncs,
  compilations per step under each torch), but not time it: the A2000 is a correctness testbed, so any timing has to come from a rented
  box. Those are cross-host comparisons, so they are a lead, not a reading.

## Amendment 51 (2026-10-06): at e4b's defaults the packed position is Unsloth/e4b 1.453 on one stack -- the regime amendment 39 lost

Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendment 51. One RTX 5090 (`tc1-5090-108`, Intel Xeon W-2145, a
15-CPU quota, Vast machine 36546), amendment 43's packed same-stack box with nothing set. e4b `7c4bdd7`, grouped-nf4-gemm `1cea661` (#492),
Unsloth 2026.9.14, packed 4,096-token rows, 40 load-gated steps, every attempt first time at load1 1.3–1.5. Read:
[`RESULTS-tc1-packed4k-defaults.md`](RESULTS-tc1-packed4k-defaults.md).

| arm | s/step (two draws) | peak |
|---|---|---|
| e4b `fused_attn4_m`, venv-unsloth, defaults | 10.070 / 10.050 | 28.23 GB |
| e4b `fused_attn4_m_t28`, venv-e4b (the field image: torch 2.8.0, triton 3.4), defaults | 13.606 / 13.636 | 28.18 GB |
| Unsloth 2026.9.14 `ckpt_unsloth_m` | 14.619 / 14.609 | 24.86 GB |

- **P126 HELD: Unsloth/e4b on one stack 1.453** [1.451, 1.455] (band [1.25, 1.80]). **P128 HELD**: every e4b arm trained resident.
  **P129 HELD**: e4b's matched peak 28.23 GB (≤ 29.5). Held-out at N: e4b 0.9544, Unsloth 0.9543.
- **Both defaults engaged by themselves**, as validity required. Each e4b arm made 160 chunked forwards with `E4B_CHUNKED_LM_LOSS` unset.
  grouped-nf4-gemm resolved `auto` with the variable unset, and bucketed every padded call (about 31,750 a run). The per-expert loop took
  about 1.5 % of calls.
- **By amendment 51's rule** 1.453 is Qwen3-30B-A3B's packed 4,096-token position at e4b's defaults
  (`e4b.train.h2h.unsloth.qwen3.5090.2026-10-06.packed-4k-defaults`). It supersedes amendment 39's out-of-memory row as the default-settings
  reading; that row stays as the record of the code before #1203 and #492. Amendment 43's labelled 1.278 stays as the reading with the
  chunked loss set by hand and no buckets.
- **P127 FALSIFIED: the environment ratio is 0.739** (registered [0.84, 0.98]). In the field image's torch 2.8 / triton 3.4, e4b steps
  13.62 s against 10.06 in torch 2.12. Amendment 43 read the same ratio at 0.915 with the chunked loss and no buckets. Across the two boxes
  (different hosts, so only suggestive), e4b in torch 2.12 went from 11.20 to 10.06 s and e4b in torch 2.8 from 12.24 to 13.62 s. That
  points at bucketed padding being slower under torch 2.8. Amendment 48 and 50 read the buckets in venv-unsloth (torch 2.12) only, and the
  default applies to every torch. That is the next registration.

## Amendment 50 (2026-10-06): under `auto` the field recipe never buckets; `auto` becomes grouped-nf4-gemm's default

Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendment 50. One RTX 5090 (`tc1-5090-107`, AMD Ryzen Threadripper
PRO 3955WX, a 15-CPU quota, Vast machine 26157), TC1's field recipe, e4b `9d3732a` with grouped-nf4-gemm `706d84f` (#491), venv-unsloth,
e4b's defaults, `NF4_QLORA_PAD_BUCKETS=0` against `auto` (buckets only where a call carries at least 16,384 routed rows). Read:
[`RESULTS-tc1-fieldauto.md`](RESULTS-tc1-fieldauto.md).

- **P123 HELD: the gate never fired.** Every `auto` arm resolved `auto` with its 16,384-row gate, and ran its 49,152 delta calls on the
  single block, exactly as many as the `0` arms, with no bucketed call.
- **P124 HELD.** Held-out at N moved +0.0022 (matched) and +0.0002 (shipped).
- **P125 HELD.** The matched peak moved −0.012 GB (27.49 / 27.51 → 27.48 / 27.49).
- **Speed, reported, not scored.** Matched 3.175 / 3.213 s/step at `0` against 3.184 / 3.195 at `auto`; shipped 2.513 / 2.466 against
  2.446 / 2.448. This host was quiet (load1 2.1–2.3, every attempt first time), and the two sides ran the same ops.
- **Decision, as registered.** With P123–P125 HELD and amendment 48's re-ask HELD, amendment 50's rule licenses `auto` as grouped-nf4-gemm's default: packed
  4,096-token rows bucket (0.893 of the step on the matched arm, 4.29 GB lighter; 0.933 shipped), and the field recipe runs the single block.
  The flip is a grouped-nf4-gemm PR citing amendments 47–50, not yet made. It carries amendment 50's conditions: the evidence scope
  (one model, one card), correctness across every expert geometry, and a release with `NF4_QLORA_PAD_BUCKETS=0` as the way back.

## Amendment 49, re-ask (2026-10-06): UNTESTED again; the census places the field recipe's delta calls at 9,040 routed rows at most

Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendment 49 and its re-ask note. One RTX 5090 (`tc1-5090-106`,
Intel Xeon Platinum 8347C, no CPU quota, Vast machine 36544), the first box's design with `TC1_PAD_CENSUS=1` on every arm, e4b `3556f46`,
grouped-nf4-gemm `d3e7788`. Two earlier draws failed on the provider's side: one host stuck `loading`, one network error ($0.03). Read:
[`RESULTS-tc1-fieldbk2.md`](RESULTS-tc1-fieldbk2.md).

- **P119–P122 UNTESTED again.** As on the first box, the single block's draws were unstable: matched 5.359 / 6.129 s/step (13.4 % apart),
  shipped 4.575 / 4.222 (8.0 %). The bucketed draws were stable: matched 5.997 / 6.205, shipped 5.717 / 5.567. The host was quiet (load1
  1.7–3.0, every attempt first time).
- **Recorded, not read.** On both boxes the bucketed shipped arm was slower than either single-block draw; the matched arm's bucketed draws
  fell inside the single block's spread here.
- **What the census measured** (49,152 delta calls per arm, the same on both sides). Each call carried 3,968 routed rows at the median and
  **9,040 at most**. Its single block was 20,520 rows at the median, 60,495 at p99 and about 104,000 at most. On the fp32 matched arm that is
  0.29 / 0.87 / 1.49 GB at the gate_up projection. On packed 4,096-token rows every call carries exactly 32,768 routed rows (4,096 × top-8).
- **So** a gate on routed rows at 16,384 would never fire at the field recipe and always fire on packed rows, with nearly 2× margin each
  way. Under it the field recipe runs today's single block by construction, and the field speed question this amendment could not settle
  does not arise. That gate is the next registration, as amendment 48 anticipated.

## Amendment 49 (2026-10-06): buckets at the field recipe read UNTESTED -- the single block's own draws came 12-21 % apart

Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendment 49. One RTX 5090 (`tc1-5090-103`, AMD Ryzen Threadripper
3990X, a 61-CPU quota, Vast machine 55583), TC1's field recipe, e4b `9170c92` with grouped-nf4-gemm `d3e7788` (#490), venv-unsloth, e4b's
defaults, `NF4_QLORA_PAD_BUCKETS=0` against `=1`. Read: [`RESULTS-tc1-fieldbk.md`](RESULTS-tc1-fieldbk.md).

- **P119–P122 UNTESTED.** The single block's draws were unstable on both arms: matched 3.524 / 4.342 s/step (20.8 % apart), shipped 2.970 /
  3.358 (12.3 %). The bucketed draws were stable: matched 4.218 / 4.159, shipped 3.685 / 3.709. Every arm VALID. The engagement held: 49,152
  bucketed calls and no single-block call on the bucketed arms, the reverse on the others, no loop on either.
- **Recorded, not read: the direction is not established.**
  - On the shipped arm, both bucketed draws (3.685, 3.709) were slower than both single-block draws (2.970, 3.358). But they ran at load1
    4.38 and 4.68 (the second a re-run after a VOID at 78.11), against 1.82 and 1.65.
  - On the matched arm, the slower single-block draw (4.342, load1 3.78) was slower than both bucketed draws (4.218, 4.159). The one
    load-matched pair, 4.159 against 3.524 at load1 1.82 each, has buckets slower.
  - If buckets do cost the host-bound field step, it is through the host-side bucket plan and the extra launches. The re-ask is what can
    show it.
  - The matched peak read 27.19 GB bucketed against 27.49 / 27.51.
- **By amendment 49's rule they stay opt-in pending a re-ask.** The re-ask will record each call's single block, its rows and bytes, so
  a size gate can be placed between the field recipe's blocks and the packed rows' from measured sizes.

## Amendment 48, re-ask (2026-10-06): bucketed padding takes 4.29 GB off e4b's packed-row peak and makes the step 11 % faster

Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendment 48 and its re-ask note. One RTX 5090 (`tc1-5090-102`, Intel
Xeon W-2145, Vast machine 36546), the first box's design on the fixed arm (`_lora_loop_share` counts every path), e4b `0afa032` with
grouped-nf4-gemm `d3e7788` (#490), packed 4,096-token rows, venv-unsloth, e4b's defaults. Read: [`RESULTS-tc1-padbk2.md`](RESULTS-tc1-padbk2.md).

| arm | one block s/step | buckets s/step | buckets / one block | peak one block → buckets | prediction |
|---|---|---|---|---|---|
| matched (fp32 adapters) | 11.306 / 11.338 | 10.112 / 10.118 | **0.893** [0.892, 0.895] | 32.52 → **28.23 GB** | P115 HELD (drop ≥ 3.0); P116 HELD (≤ 1.02) |
| shipped (bf16 adapters) | 8.742 / 8.748 | 8.166 / 8.154 | **0.933** [0.932, 0.934] | 26.97 → 26.97 GB | P117 HELD (≤ 1.02) |

- **All four HELD.** Held-out at N moves −0.0004 (matched) and +0.0000 (shipped) (P118). The host was quiet: every attempt ran first time,
  at load1 1.3–1.4.
- **Engagement as recorded** (`lean_ab.lora_path_calls`, whole run). Every bucketed arm made 31,755–31,822 bucketed calls and no
  single-block call, and the reverse on the other side. The per-expert loop took 0.8–2.9 % of a step's calls on both sides, peaking at
  2.6–2.9 % (amendment 43's 5 % rule reads the peak); over the whole run it was 1.3–1.6 %. `auto`'s rule still sizes the single block.
- **It is faster, not only lighter.** At 4,096 tokens the single block computes and moves about 11× the routed rows; the buckets at most
  2×. The fp32 arm gains most: its block is twice the bytes. The shipped arm's peak does not move: with bf16 adapters its peak is set
  elsewhere in the step.
- **Decision, as registered.** Buckets buy the packed regime with no speed or quality cost there. Before they can be a default, the field
  recipe's short rows must be shown not to pay for the extra launches. That is the next registration.

## Amendment 48, first box (2026-10-06): UNTESTED -- the arm's loop share did not count grouped-nf4-gemm's new bucketed calls

Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendment 48. One RTX 5090 (`tc1-5090-101`, AMD Ryzen 9 7950X, Vast
machine 130223), packed 4,096-token rows, e4b `087e045` with grouped-nf4-gemm `d3e7788` (#490), `NF4_QLORA_PAD_BUCKETS=0` against `=1`. Read:
[`RESULTS-tc1-padbk.md`](RESULTS-tc1-padbk.md).

- **Every `_pk1` arm reads VOID, so P115–P118 are UNTESTED.** The reducer applied amendment 43's rule, a per-expert loop above 5 % of a step's
  delta calls, and the bucketed arms recorded `lora_loop_share` 1.000 on every step.
- **That share is an instrument bug, not the route.** `tc1_arm.py` divided the loop's calls by the sum of loop, padded and grouped_mm calls.
  grouped-nf4-gemm#490's new `padded_bucketed` counter was not in that sum. Over the matched `_pk1` arm's whole run
  (`lean_ab.lora_path_calls`, warm-up included) the counters read 494 loop calls against 31,762 bucketed calls, a 1.5 % share. Amendment
  43's rule reads the largest per-step share. Recomputed from `kernel_calls_all` with every counter, that is 2.6–2.9 % on the `_pk1` arms
  and on the `_pk0` arms, which the old sum already covered. Every arm is under the 5 % line. The fix counts every `lora_path_*` counter
  (`_lora_loop_share`, with a self-test case for this receipt's numbers).
- **The VOID arms' speeds and peaks are not read here.** The `_pk0` arms were VALID on a quiet host: load1 1.1–1.3, every attempt first
  time.
- **Next, as registered** (any UNTESTED, none FALSIFIED: a re-ask is allowed). The same box runs again on the fixed arm. That is recorded in
  amendment 48 before it runs.

## Amendment 47 (2026-10-06): on packed rows e4b's peak is 7.47 GB above Unsloth's, and 6.1 GB of it is the padded LoRA delta padding every expert to the hottest

Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendment 47. One RTX 5090 (`tc1-5090-100`, AMD EPYC 7B13, Vast
machine 145701), amendment 23's census (`--mem-census 1`) on amendment 39's packed rows (4,096 real tokens, micro-batch 1 × accum 4), 20
steps, one draw per arm in venv-unsloth (e4b `dfc5bda`, grouped-nf4-gemm `a93b80c`). No speed is read. Read:
[`RESULTS-tc1-memc4k.md`](RESULTS-tc1-memc4k.md). The first draw, `tc1-5090-99`, was refused before any rental: another run's partial
copy of grouped-nf4-gemm's source held a `receipt.json` the store could not reconcile.

| peak allocated (GB) | e4b defaults | e4b, `E4B_ABSMAX_DQ=1` + `NF4_QLORA_COMPACT_DELTA=1` | Unsloth |
|---|---|---|---|
| static classes (all equal but the absmax) | 24.49 | 23.14 | 23.14 |
| expert absmax (inside the static line) | 1.812 | 0.460 | 0.460 |
| transient (live at the peak) | **7.82** | **6.40** | 1.73 |
| **peak** | **32.34** | **29.54** | **24.86** |

- **P112 HELD.** The census attributes at least 99.94 % of each arm's peak to named groups.
- **P113 HELD: e4b's defaults peak 7.47 GB above Unsloth's.** 1.35 GB is the fp32 expert absmax. The other 6.10 GB is transient,
  almost all of it grouped-nf4-gemm's padded LoRA delta (`_lora_delta_padded`: 3.35 GB at its zero-padded input block, 2.47 GB at its
  products).
- **P114 FALSIFIED: with both levers e4b is still 4.68 GB above** (registered ≤ 3.0). The double-quantized absmax removed its 1.35 GB.
  The compact delta took only 1.42 GB off the transient. What is left is the compact node's own forward: its zero-padded input block
  `[G·widest, K]` (1.17 GB, `nf4_qlora.py:490`) and its padded output `[G, widest, N]` (3.11 GB, `:493`), fp32 on this matched arm.
- **Why the block is that wide.** Both paths pad every expert's rows to the hottest expert's count (`widest`). The sizes fit the down
  projection (K 768, N 2048) at about 380,000 padded rows against 32,768 routed rows (4,096 tokens × top-8): about **11.6× padding**.
  The same output unpadded is about 0.27 GB. The compact delta changes what is saved for backward, not how wide the block is. At the
  field recipe the hottest expert is far less hot, which is why amendment 23 saw 0.43 GB there.
- **As registered, this is a measurement.** The next registration targets the padding itself. A delta that pads each expert only to
  its own group's widest (bucketed by row count) would, if this arithmetic holds, cut the block several-fold at 4,096 tokens.

## Amendment 45 (2026-10-05): the container held to 31 CPUs ran 128 threads, but the threads A/B reads UNTESTED on the busiest host

Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendment 45. One RTX 5090 (`tc1-5090-98`, AMD EPYC 7B13, Vast
machine 145701). e4b's and Unsloth's matched arms run in venv-unsloth (e4b `ee9d179`, grouped-nf4-gemm `054be19`), 60 load-gated steps:
`OMP_NUM_THREADS` at the host's physical cores (`_om0`) against the container's CPU allotment (`_om1`). Read:
[`RESULTS-tc1-ompab.md`](RESULTS-tc1-ompab.md).

- **The fact this box recorded first (#1196).** The container's cgroup `cpu.max` was **3071999 / 100000, a quota of 31 CPUs**. Its
  cpuset and affinity showed all 256, and the lane ran **128 threads** (the physical cores), four per allotted CPU. Every earlier box on
  this machine ran that way, among them amendments 41 and 43's.
- **P104, P105 and P106 UNTESTED.** The host's load1 medians ran 4–75 against a gate of 6.0, so most arms used all three attempts.
  e4b's `_om1` draws came 13.6 % apart (3.513 / 4.024 s/step), and Unsloth's `_om0` draws 5.2 % (8.416 / 8.868). The 4 h guard ended before
  e4b's last `_om0` draw ran. Every receipt ran the thread count its side names (torch's pool equal to `OMP_NUM_THREADS`: 128 or 31).
- **Unquotable, for the record.** e4b's one `_om0` draw (3.464 s at load 4.2) was no slower than its `_om1` draws (3.513 at load 33, 4.024
  at 54). Unsloth's sides were interchangeable (8.416 / 8.868 at 128 threads, 8.866 / 8.437 at 31). These are confounded by host load
  (each side's draws ran at different loads), so they say nothing either way about the thread count. P104's bound (at least 3 %
  faster) stays open.
- **By amendment 45's rule** a re-ask is allowed. The harness keeps the physical cores, and no position changes.

## Amendment 44 (2026-10-05): `E4B_CHUNKED_LM_LOSS=auto` costs the field recipe nothing (0.992 / 0.999) and its gate never fired; it becomes e4b's default

Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendment 44. One RTX 5090 (`tc1-5090-97`, AMD Ryzen Threadripper PRO
3955WX, 32 CPUs, Vast machine 26157), every arm in venv-unsloth (torch 2.12.1+cu130, triton 3.7.1) with e4b `298224f` (#1178's `auto`) and
grouped-nf4-gemm `f127981` (0.41.0), TC1's Qwen3-30B-A3B tokens and field recipe, 60 load-gated steps. `_ca0` is the stock loss, and `_ca1`
sets `E4B_CHUNKED_LM_LOSS=auto` (chunk a training forward only when its stock fp32 logits would reach 1 GiB). Read:
[`RESULTS-tc1-chunkauto.md`](RESULTS-tc1-chunkauto.md). The first draw, `tc1-5090-96`, refused on a host whose driver was below the lane's
floor ($0.007).

| arm | `_ca0` s/step | `_ca1` s/step | `_ca1` / `_ca0` | peak `_ca0` → `_ca1` | prediction |
|---|---|---|---|---|---|
| shipped | 2.483 / 2.421 | 2.429 / 2.435 | **0.992** [0.978, 1.006] | 24.673 → 24.673 GB | P100 HELD (≤ 1.02) |
| matched | 3.135 / 3.142 | 3.139 / 3.131 | **0.999** [0.997, 1.001] | 27.496 → 27.494 GB | P101 HELD (≤ 1.02); P102 HELD |

- **P99 HELD: the gate never fired on the field recipe.** Every `auto` arm ran its 240 training forwards on the stock path (`small_calls`
  240, `chunked_calls` 0), as the shapes said it would.
- **P103 HELD.** Held-out at N moves +0.0001 (matched) and −0.0001 (shipped).
- **The host was quiet:** every attempt ran first time, under the gate, at load1 medians 1.1–1.9.
- **Decision, as registered.** With P99–P103 HELD and amendment 43's P98 HELD, `auto` becomes e4b's default in `enable_fast_train` and the
  CLI trainer. Packed 4,096-token rows chunk (2.32 GiB of logits a row, over the gate), and amendment 43 measured that path. The field
  recipe runs the stock loss, and this box measured that. `E4B_CHUNKED_LM_LOSS=0` keeps the stock loss everywhere, and `1` chunks every
  training forward.

## Amendment 43 (2026-10-05): on packed 4,096-token rows, with its chunked loss, e4b is 1.278× Unsloth's speed on one stack (labelled, opt-in)

Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendment 43. One RTX 5090 (`tc1-5090-95`, AMD EPYC 7B13, Vast
machine 145701): amendment 40's box again. Both frameworks run on torch 2.12.1+cu130 / transformers 5.5.0 on packed rows of exactly
4,096 real tokens (micro-batch 1 × accum 4, `TC1_FREE_OUTPUTS=1`), 40 load-gated steps. e4b @ `da897e3` with `E4B_CHUNKED_LM_LOSS=1`
on every e4b arm and grouped-nf4-gemm `f127981` (0.41.0); grouped-nf4-gemm's per-expert LoRA loop is read as a recorded route up to 5 %
of a step's delta calls. Read: [`RESULTS-tc1-packed4k-chunked2.md`](RESULTS-tc1-packed4k-chunked2.md).

| arm | s/step (two draws) | peak | per-expert loop (max share) |
|---|---|---|---|
| e4b `fused_attn4_m`, venv-unsloth, chunked loss | 11.188 / 11.220 | 32.47 / 32.53 GB | 2.6 % |
| e4b `fused_attn4_m_t28`, venv-e4b, chunked loss | 12.236 / 12.257 | 32.45 / 32.46 GB | 2.9 % |
| Unsloth 2026.9.14 `ckpt_unsloth_m` | 14.266 / 14.366 | 24.86 GB | — |

- **P96 HELD: Unsloth/e4b on one stack 1.278** [1.271, 1.284] (band [1.25, 1.65]). **P97 HELD:** e4b's environment ratio is
  **0.915** [0.913, 0.917]. **P98 HELD:** every e4b arm trained resident with 160 chunked forwards and no fallback. Held-out at N:
  e4b 0.9548, Unsloth 0.9544.
- **By amendment 43's rule** 1.278 is recorded as Qwen3-30B-A3B's packed 4,096-token position, labelled "e4b with
  `E4B_CHUNKED_LM_LOSS=1`, opt-in; grouped-nf4-gemm's loop route at the recorded share"
  (`e4b.train.h2h.unsloth.qwen3.5090.2026-10-05.packed-4k-chunked`). It sits beside amendment 39's row, where e4b at its defaults ran
  out of memory. At e4b's defaults the packed regime is still an e4b loss. Amendment 44 registers `auto` (#1178) as the way to close
  that, and this box's P98 is its packed side.
- **Memory is where e4b gives ground here:** 32.5 GB against Unsloth's 24.9, about 1.1 GB under the card's 33.7 GB (31.36 GiB). Recorded, not read.
- **The host was the busy one** amendment 41 ran on. The standing attempts ran at load1 medians 3.6–52.2, against a gate of 6.0. On an
  11 s, device-bound step the draws stayed within 0.3 % (e4b) and 0.7 % (Unsloth).
- **Recorded for later, not read.** Amendment 40's box (EPYC 7K62) gave e4b the same 11.20 s step but Unsloth 16.03 s. Here Unsloth
  stepped 14.32 s. Positions are within-box readings of their host.

## Amendment 41 (2026-10-05): at the field recipe e4b's chunked LM loss costs the shipped arm 4.9 % of its step; it stays opt-in

Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendment 41. One RTX 5090 (`tc1-5090-89`, AMD EPYC 7B13, Vast
machine 145701), every arm in venv-unsloth (torch 2.12.1+cu130, triton 3.7.1, transformers 5.5.0) with e4b `054cb8c` and
grouped-nf4-gemm `ccf4de9` (0.41.0), TC1's Qwen3-30B-A3B tokens and field recipe, 60 steps, load-gated draws. `_ce0` is the default,
`_ce1` sets `E4B_CHUNKED_LM_LOSS=1` (512-token chunks). Read: [`RESULTS-tc1-chunkab.md`](RESULTS-tc1-chunkab.md).

| arm | `_ce0` s/step | `_ce1` s/step | `_ce1` / `_ce0` | peak `_ce0` → `_ce1` | prediction |
|---|---|---|---|---|---|
| shipped | 2.918 / 2.878 | 2.967 / 3.113 | **1.049** [1.017, 1.082] | 24.673 → 23.508 GB | P91 FALSIFIED (≤ 1.01) |
| matched | 3.688 / 3.717 | 3.896 / 3.692 (5.4 % apart: UNSTABLE) | not read | 27.498 → 27.145 GB | P90, P92 UNTESTED |

- **Engagement.** Every `_ce1` arm ran 240 chunked training forwards (one per micro-batch, 60 steps × 4) with no run-time fallback;
  every `_ce0` arm ran none. All eight arms VALID; held-out at N moves −0.0015 (shipped) and −0.0007 (matched): P93 UNTESTED only
  because its matched side is unstable.
- **By amendment 41's rule `E4B_CHUNKED_LM_LOSS` stays opt-in.** P91 fell on the slow side of its bound on the shipped arm. The
  default decision needed P90–P93 all HELD, so no re-draw of the matched pair can change it, and none is registered.
- **The host was loaded throughout.** Machine 145701 carried other sessions' boxes; every attempt of every arm ran above the 6.0 load
  gate (load1 medians 7.0–32.9), so each arm's retries ran out and its last attempt stands, as registered (standing attempts at
  7.0–22.5). The cost does not come from the load alone: the least-loaded cross pair, `_ce1` at load 7.0 against `_ce0` at 8.3, still
  reads 1.017.
- **What it costs and buys.** At ~1,000–1,400 real tokens per step the fp32 logits the chunking avoids are small; it takes 1.17 GB off
  the shipped arm's peak and 0.35 GB off the matched arm's, and pays for the lm_head recomputed in backward and the extra launches per
  chunk, on a step that is host-bound. That matches #1142's 2–6 % on an RTX A2000 slice. On packed 4,096-token rows the same flag is
  what lets e4b train at all (amendment 40); amendment 43's box reads that regime with it set explicitly.

## Amendment 40 (2026-10-05): with its chunked loss e4b trains the packed 4,096-token rows, but the box reads UNTESTED on TC1's no-loop rule

Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendment 40. One RTX 5090 (`tc1-5090-91`, AMD EPYC 7K62, Vast
machine 152440; three earlier launches died on their hosts for $0.24: one went offline, one never authenticated ssh, one had a driver below
the harness's floor): amendment 39's box with `E4B_CHUNKED_LM_LOSS=1` on every e4b arm, 40 steps. Read:
[`RESULTS-tc1-packed4k-chunked.md`](RESULTS-tc1-packed4k-chunked.md).

| arm | verdict | s/step (two draws) | peak | per-expert loop |
|---|---|---|---|---|
| e4b `fused_attn4_m`, venv-unsloth, chunked loss | VOID | 11.188 / 11.210 | 32.49 / 32.57 GB | every step, 1.5 % of delta calls (max 2.9 %) |
| e4b `fused_attn4_m_t28`, venv-e4b, chunked loss | VOID | 12.562 / 12.553 | 32.52 / 32.44 GB | every step, 1.5 % (max 2.9 %) |
| Unsloth `ckpt_unsloth_m` | VALID | 16.027 / 16.030 | 24.86 GB | — |

- **The chunked loss did what it was for.** Every e4b arm trained the packed rows to step 40: 160 chunked training forwards each, no
  run-time fallback, no OOM, peak 32.4–32.6 GB. Amendment 39's e4b arms OOMed at step 1 on the same rows.
- **P87, P88 and P89 UNTESTED.** Each e4b arm is VOID under TC1's registered rule that a fused arm's per-expert LoRA loop never runs
  (`lora_path_loop_steps`). Here grouped-nf4-gemm's `auto` route took the loop on every step, for about 1.5 % of its delta calls: those
  whose padded block would exceed its 2 GiB limit (`NF4_QLORA_PAD_BYTES_LIMIT`), which a hot expert's block does at 4,096 tokens. That
  is e4b's default route doing what it was built to do (the route changes, never the result); the rule was written for the field recipe,
  where the loop never runs.
- **Unquotable, for the record.** Read as if VALID, the box says Unsloth/e4b **1.43** on one stack (16.03 against 11.20 s/step) and the
  environment **0.892**. Neither is a reading under amendment 40's instrument.
- **The host was shared** with this campaign's `tc1-5090-94` (amendment 42) from 14:19Z; the load gate voided no draw.
- **Next.** A registration that reads the packed regime's loop share as a recorded route rather than a VOID, made with these numbers in view
  and said so.

## Amendment 42 (2026-10-05): the same-stack position replicates on a second host, 2.468

Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendment 42. One RTX 5090 (`tc1-5090-94`, AMD EPYC 7K62, Vast
machine 152440; earlier launches cost $0.006 and $0: a host below the driver floor, and a receipts-store race): amendment 33's box on the
current code (e4b `5c90886`, grouped-nf4-gemm `ccf4de9`), off amendment 33's machine. Read:
[`RESULTS-tc1-samestack-host2.md`](RESULTS-tc1-samestack-host2.md).

| arm | s/step (two draws, 60 steps) | peak |
|---|---|---|
| e4b `fused_attn4_m`, venv-unsloth | 4.143 / 4.073 (1.7 % apart) | 27.50 GB |
| Unsloth `ckpt_unsloth_m`, venv-unsloth | 10.155 / 10.120 (0.3 % apart) | 24.27 GB |
| e4b `fused_attn4_m_t28`, venv-e4b | 4.734 / 4.709 (0.5 % apart) | 27.45 GB |

- **P94 HELD.** Unsloth/e4b on one stack reads **2.468** [2.443, 2.493], against amendment 33's 2.352 on an EPYC 7B13: 5 % apart, both inside
  [1.9, 2.9]. Held-out COMPARABLE (Δ −0.0012); Unsloth peaks 3.22 GB lower and spends ×1.41 e4b's energy per step.
- **P95 HELD.** The environment reads **0.870** [0.860, 0.880]. The prebound launches now cover triton 3.7, so both of e4b's sides take them;
  amendment 33's 0.900 had them on one side only, and its read bounded the symmetric value at about 0.876–0.900.
- **By amendment 42's rule 2.352 stays the position to quote**, now read on two hosts (2.352 and 2.468). Both steps ran longer on this host
  (e4b 4.11 s, Unsloth 10.14 s, against 3.49 and 8.22 s) and the ratio moved 5 %.
- **The host was shared** with this campaign's `tc1-5090-91` (amendment 40) until 15:30Z; the load gate voided no draw.
- **Two host models, by draw, not by design.** Amendment 42 excluded machine ids but did not register a CPU model (#1157: it
  auto-merged before that review point was applied). This draw's EPYC 7K62 differs from amendment 33's EPYC 7B13, so "two hosts"
  here is also two host models. The code moved too (e4b `5c90886`), so the 5 % between them is not attributed to either.

## Amendment 39 (2026-10-05): on packed 4,096-token rows e4b at its defaults runs out of memory where Unsloth trains

Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendment 39. One RTX 5090 (`tc1-5090-86`, AMD EPYC 7B13, Vast
machine 145701): the token `qwen3samestack4k`, amendment 25's same-stack family on packed rows of exactly 4,096 real tokens (tokens sha
`d2a501eba57d`, the same as the harness PR's local build), micro-batch 1 × accum 4, 30 steps, `TC1_FREE_OUTPUTS=1`, load-gated draws.
Read: [`RESULTS-tc1-packed4k.md`](RESULTS-tc1-packed4k.md).

| arm | status | s/step (two draws) | peak |
|---|---|---|---|
| e4b `fused_attn4_m`, venv-unsloth, defaults | **OOM at step 1** (both draws) | — | 30.32 GB at the failure |
| e4b `fused_attn4_m_t28`, venv-e4b, defaults | **OOM at step 1** (both draws) | — | 30.27 GB at the failure |
| Unsloth `ckpt_unsloth_m` | OK, VALID | 15.164 / 14.040 (7.7 % apart) | 24.86 GB |

- **P86 FALSIFIED.** All four e4b arms failed at step 1 allocating **2.32 GiB**: 4,096 × 151,936 × 4 bytes, the fp32 copy of the
  full-vocabulary logits that Hugging Face's causal-LM loss makes. At that moment 29.5 GiB of the card's 31.36 GiB were in use.
- **P84 and P85 UNTESTED.** No e4b arm ran. Unsloth's two draws are 7.7 % apart; its first draw stood at a host load of 16.2 after the
  gate's two re-runs (the voided attempts read 14.19 and 14.55 s/step).
- **By amendment 39's rule this is an e4b loss in the packed regime**, recorded with the peak where it failed (row
  `e4b.train.h2h.unsloth.qwen3.5090.2026-10-05.packed-4k`). Unsloth trains the same rows resident at 24.86 GB: it computes its loss in
  chunks and never materialises the full logits.
- **The lever is a chunked loss for e4b**, its own registration. experts4bit-qlora#1142 adds one, opt-in: on an RTX A2000, a 4-layer
  slice of the same checkpoint hits the stock path's exact 2.32 GiB failure at 4,096 tokens and trains through it with the loss chunked.
- **The host was busy** (load1 6.8–22.5 during Unsloth's draws). That costs the speed reading here, not the memory one.

## Amendment 38 (2026-10-05): on a fast host the compact delta is 1.3–1.6 % slower; it stays opt-in

Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendment 38. One RTX 5090 (`tc1-5090-85`, AMD EPYC 9655, Vast
machine 150700), every arm in venv-unsloth with e4b `e1faf47` and grouped-nf4-gemm `0e393f0` (after #473), 60 steps, load-gated draws
(none voided). Read: [`RESULTS-tc1-compact-default.md`](RESULTS-tc1-compact-default.md).

| family, arm | `_cd0` s/step | `_cd1` s/step | `_cd1` / `_cd0` | peak `_cd0` → `_cd1` | prediction |
|---|---|---|---|---|---|
| Qwen3-30B-A3B, matched | 2.167 / 2.178 | 2.215 / 2.199 | **1.016** [1.010, 1.022] | 27.501 → 27.189 GB | P77 FALSIFIED (≤ 0.99); P79 HELD |
| Qwen3-30B-A3B, shipped | 1.716 / 1.720 | 1.742 / 1.740 | **1.013** [1.011, 1.015] | 24.673 → 24.673 GB | P78 FALSIFIED (≤ 0.99) |
| Mixtral-8x7B, matched (resident, defaults) | 3.326 / 3.330 | 3.374 / 3.371 | **1.013** [1.012, 1.014] | 31.234 → 31.197 GB | P80 FALSIFIED (≤ 1.01); P81 HELD |

- **Held-out** moves +0.0008 / 0.0000 (Qwen3) and +0.0008 (Mixtral): P82 and P83 HELD.
- **By amendment 38's rule the flag stays opt-in**: P77, P78 and P80 fell on the slow side. No further box is registered for it under
  these rules.
- **The peak held everywhere; the speed depends on the host.** With #473's backward the flag lowers the fp32 arm's peak by 0.29–0.31 GB
  on both hosts that measured it, and never raises a peak. Its speed tracks how host-bound the step is:

  | box | host | e4b's Qwen3 matched step | `_cd1` / `_cd0` matched / shipped |
  |---|---|---|---|
  | `tc1-5090-80` (amendment 36) | EPYC 7B13 | 3.43 s | 0.969 / 0.970 |
  | `tc1-5090-83` (amendment 37) | EPYC 7702P | 3.89 s | 0.967 / 0.948 |
  | `tc1-5090-85` (this box) | EPYC 9655 | 2.17 s | 1.016 / 1.013 |

  The compact node trades host work (one autograd node instead of about ten per projection) for device work (the padded block rebuilt in
  backward: +0.7 ms per layer's backward on an RTX A2000, grouped-nf4-gemm#445). Where the host is slow the first wins; on this host the
  same step is 37–44 % shorter, less of it is host time, and the second wins.
- **Recorded for later, not read here.** e4b's Qwen3 step on this host is 2.17 s against 3.4–3.9 s on the other two, the same code and
  card. Positions are within-box readings of their host, and this box ran no Unsloth arm.

## Amendment 37 (2026-10-05): with its backward releasing early, the compact delta lowers the matched peak 0.29 GB and is 3–5 % faster on another host

Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendment 37. One RTX 5090 (`tc1-5090-83`, AMD EPYC 7702P, Vast
machine 45379; two earlier launches cost $0.007 and $0: a host below the harness's driver floor, then an avoid receipt the launcher does not
accept): amendment 36's box, `NF4_QLORA_COMPACT_DELTA` 0 vs 1, every arm in venv-unsloth with e4b `64afc6d` and grouped-nf4-gemm `9622144`
(after #473), 60 steps, load-gated draws (none voided). Read: [`RESULTS-tc1-compactab2.md`](RESULTS-tc1-compactab2.md).

| arm | `_cd0` s/step | `_cd1` s/step | `_cd1` / `_cd0` | peak `_cd0` → `_cd1` | held-out Δ | prediction |
|---|---|---|---|---|---|---|
| matched (fp32 adapters) | 3.914 / 3.860 | 3.752 / 3.763 | **0.967** [0.959, 0.975] | 27.477 → **27.189 GB** | +0.0029 | P73, P74 HELD |
| shipped (bf16 adapters) | 3.026 / 2.970 | 2.824 / 2.864 | **0.948** [0.933, 0.964] | 24.673 → 24.673 GB | −0.0003 | P75 FALSIFIED (below [0.95, 0.99]) |

- **The peak moved the way the A2000 said.** Amendment 36's box read the matched peak +0.229 GB with the flag on; with #473's backward it
  reads −0.288 GB (P73 HELD, band [−0.05, 0.50]). The shipped arm's peak is elsewhere in its step and does not move.
- **The speed replicated, and the shipped arm more than registered.** Matched 0.967 against amendment 36's 0.969 (P74 HELD); shipped 0.948
  against 0.970, just below the band (P75 FALSIFIED on the fast side).
- **P76 HELD.** Held-out at N moves +0.0029 (matched) and −0.0003 (shipped).
- **By amendment 37's rule the flag stays opt-in pending its own registration**: a ratio below 0.95 is the "otherwise" branch. The bands
  were two-sided, and the reading fell outside on the side the decision wanted. A default needs a registration whose bands allow that.
- **Two hosts now agree on the direction** (EPYC 7B13 and EPYC 7702P): 3–5 % off the step on both arms, with outputs and gradients
  bit-identical by construction.

## Amendment 36 (2026-10-05): the compact padded LoRA delta is 3 % faster, not lighter — the matched peak rose 0.23 GB

Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendment 36. One RTX 5090 (`tc1-5090-80`, AMD EPYC 7B13, Vast
machine 145701): the token `qwen3compactab`, `NF4_QLORA_COMPACT_DELTA=0` (`_cd0`) against `=1` (`_cd1`) on the shipped and the matched
arm, two draws a side in ABBA order, every arm in venv-unsloth with e4b `23edeff` and grouped-nf4-gemm `c4a683b`, 60 steps, load-gated
draws. Read: [`RESULTS-tc1-compactab.md`](RESULTS-tc1-compactab.md).

| arm | `_cd0` s/step | `_cd1` s/step | `_cd1` / `_cd0` | peak `_cd0` → `_cd1` | held-out Δ |
|---|---|---|---|---|---|
| matched (fp32 adapters) | 3.431 / 3.428 | 3.348 / 3.299 | **0.969** [0.962, 0.977] | 27.490 → **27.719 GB** | +0.0004 |
| shipped (bf16 adapters) | 2.919 / 2.858 | 2.793 / 2.810 | **0.970** [0.957, 0.983] | 24.673 → 24.673 GB | −0.0006 |

- **Engagement.** Every arm resolved the flag its tag names, kept no layer's MoE activations and ran the padded LoRA path (49,152 padded
  calls each).
- **P69 FALSIFIED.** The matched arm's peak rose by 0.229 GB instead of falling by [0.3, 2.0] GB. The shipped arm's did not move.
- **P70 and P71 FALSIFIED, on the fast side.** The step got faster than the registered band allowed, 0.969 (matched) and 0.970 (shipped,
  0.9699 before rounding) against [0.97, 1.02]. One autograd node in place of about ten per projection is less host work, and this step is
  host-bound.
- **P72 HELD.** Held-out at N moves +0.0004 and −0.0006.
- **By amendment 36's rule the compact delta stays opt-in.** The registered memory reason did not hold, and a speed default needs its own
  registration.
- **Why the peak rose (from the code, not yet measured).** `_CompactPaddedDelta.backward` keeps the padded output gradient `[G, widest, N]`
  referenced until it returns, while it rebuilds the input block and computes the input's gradient. Autograd's separate nodes release that
  gradient after the second product's backward. With whole-layer checkpointing, the matched arm's peak falls inside a MoE layer's backward,
  where the extra block sits. The shipped arm's peak did not move, so it is elsewhere in that step. A grouped-nf4-gemm change that
  releases each intermediate at its last use is being measured on an RTX A2000.
- **The gate.** The shipped arm's first draws were voided (load 8.2) and run again; the first attempts read 0.966 for the shipped
  arm, the same verdict.
- **The host was shared with this campaign's own boxes** (`tc1-5090-79` until 06:13Z).

## Amendment 35 (2026-10-05): under triton 3.7.1 the prebound launches take 1.4 % off the matched arm and 0.4 % off the shipped arm

Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendment 35. One RTX 5090 (`tc1-5090-79`, AMD EPYC 7B13, Vast
machine 145701): the token `qwen3prebind37`, amendment 26's eight arms in its ABBA order, every one in venv-unsloth (torch 2.12.1+cu130,
transformers 5.5.0, triton 3.7.1) with e4b `3cc7f10` and grouped-nf4-gemm `c4a683b` (both after #1108 / #471), 60 steps, load-gated
draws. Read: [`RESULTS-tc1-prebind37.md`](RESULTS-tc1-prebind37.md).

| arm | `_pb0` s/step | `_pb1` s/step | `_pb1` / `_pb0` | held-out Δ | prediction |
|---|---|---|---|---|---|
| shipped (bf16 adapters) | 2.876 / 2.902 | 2.839 / 2.913 | **0.996** [0.978, 1.013] | −0.0026 | P66 HELD ([0.97, 1.00]) |
| matched (fp32 adapters) | 3.444 / 3.418 | 3.408 / 3.360 | **0.986** [0.975, 0.997] | +0.0009 | P67 HELD ([0.97, 1.00]) |

- **Engagement.** Every `_pb1` arm counted 216,335 prebound e4b launches and 72,162 grouped-nf4-gemm ones per process, every `_pb0` arm
  none, all under triton 3.7.1 and torch 2.12.1.
- **P68 HELD.** Held-out at N moves −0.0026 (shipped) and +0.0009 (matched), inside 0.005.
- **By amendment 35's rule** triton 3.7 stays in the prebound path's supported versions in both repositories; row
  `e4b.train.prebind.triton37.qwen3.5090.2026-10-05`.
- **What it is worth.** The matched interval excludes 1.0; the shipped interval does not. Under triton 3.4 the same flags read 0.973
  (matched) and 0.980 (shipped); triton 3.7.1's own launch is cheaper, so there is less to take off.
- **The gate.** Five draws were voided for host load and run again; their files are in
  [`receipts/tc1-5090-79/loadvoid/`](receipts/tc1-5090-79/loadvoid/). The shipped arm's first draws exhausted the retries and stood at a
  median load1 of 8.17 (`_pb0`) and 8.24 (`_pb1`). Reading the first attempts instead gives 0.980 (shipped) and 0.995 (matched), inside
  the same bands.
- **The host was shared with this campaign's own boxes.** `tc1-5090-78` (until 05:23Z) and `tc1-5090-80` (from 05:11Z) ran on the same
  machine, so part of the load the gate saw was the campaign's.

## Amendment 34 (2026-10-05): the 5090's environment gain is torch 2.12's, not transformers 5.5's

Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendment 34. One RTX 5090 (`tc1-5090-78`, AMD EPYC 7B13, Vast
machine 145701): the token `qwen3envsplit`, the matched arm in three environments, two draws each in ABC CBA order, 60 steps, load-gated
draws, the prebound launches off everywhere. Read: [`RESULTS-tc1-envsplit.md`](RESULTS-tc1-envsplit.md).

| side | torch | transformers | triton | s/step (two draws) | stable |
|---|---|---|---|---|---|
| `_e0` venv-e4b | 2.8.0+cu128 | 5.18.0 | 3.4.0 | 3.840 / 3.816 | yes (0.6 %) |
| `_e1` venv-e4b-tf55 (built on the box) | 2.8.0+cu128 | 5.5.0 | 3.4.0 | 3.821 / 3.871 | yes (1.3 %) |
| `_e2` venv-unsloth + e4b | 2.12.1+cu130 | 5.5.0 | 3.7.1 | 3.507 / 3.452 | yes (1.6 %) |

- **P62 FALSIFIED.** transformers 5.5 over 5.18 on torch 2.8 reads **1.005** [0.995, 1.014], outside [0.90, 1.00]. The router casts the
  A2000 profile pointed at do not show in the 5090's step. By amendment 34's rule nothing follows from P62.
- **P63 HELD.** torch 2.12 + triton 3.7 over torch 2.8 + triton 3.4, at transformers 5.5, reads **0.905** [0.892, 0.918]. Amendment 32
  read triton 3.7.1 alone (on torch 2.8) at 0.992 on this arm, so nearly all of it is torch 2.12's.
- **P64 HELD.** The whole environment reads **0.909** [0.899, 0.919]; amendment 24 read 0.882, amendment 33 0.900.
- **P65 HELD.** Held-out at N moves +0.0013 (`_e1`) and −0.0010 (`_e2`).
- **By amendment 34's rule (P63 HELD)** the install section now says that torch 2.12 runs e4b's host-bound training step faster on an
  RTX 5090. Row `e4b.train.env-split.qwen3.5090.2026-10-05`.
- **The gate.** Six draws were voided for host load and run again; their files are in
  [`receipts/tc1-5090-78/loadvoid/`](receipts/tc1-5090-78/loadvoid/). Reading the first attempts instead gives 1.004, 0.907 and 0.911:
  no verdict changes. `_e0`'s second draw stood on its third attempt at a median load1 of 9.15 (the last attempt stands whatever its
  load); it read 3.816 against its first draw's 3.840 at 4.2.
- **The host was shared with this campaign's own boxes.** From 04:45Z and 05:11Z, `tc1-5090-79` and `tc1-5090-80` ran on the same machine
  (145701). Their steps are host-bound too, so part of the load the gate saw was this campaign's. The ABC CBA order puts each side's draws
  on both halves of the run.
- **Not registered, recorded.** Step-0 held-out on `_e1` (torch 2.8, transformers 5.5) is 1.9733 against `_e0`'s 1.9441 and `_e2`'s
  1.9478 (NEAR, 0.029). The same transformers on torch 2.12 does not move it, so it is the combination, not transformers 5.5 alone. At N=60 the
  three sides are within 0.0014.

## Amendment 30 (2026-10-05): the shipped prebind pair over 60 steps (P53 HELD); with P54 and P55 HELD the prebound launches become the default

Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendments 26 and 30.
- **The box.** One RTX 5090 (`tc1-5090-73`, AMD EPYC 7B13, Vast machine 145701, not machine 151350 as the amendment
  requires; $0.97 invoiced): the token `qwen3prebindab` with `TC1_STEPS=60`, the shipped arm only, two draws a side in ABBA
  order; e4b `306dfa9c`, grouped-nf4-gemm `f0c1ece`, triton 3.4.0.
- **Read:** [`RESULTS-tc1-prebindab-60.md`](RESULTS-tc1-prebindab-60.md), the box's own mechanically scored file.

| `fused_attn4_shipped` | `_pb0` (flags off) | `_pb1` (flags on) |
|---|---|---|
| s/step, two draws (60 steps) | 3.092 / 3.047 (1.5 % apart) | 3.056 / 2.960 (3.2 % apart) |
| held-out at N | 0.7575, 0.7556 | 0.7589, 0.7565 |

- **P53 HELD.** `_pb1`/`_pb0` reads 0.980 [0.957, 1.003 over four cross-draw ratios], inside [0.90, 0.98], at its upper
  edge, as the reducer scores it. On the `_pb1` side, 216,335 e4b launches and 72,162 grouped-nf4-gemm launches were
  prebound.
- **P55 HELD.** The shipped half moves held-out by +0.0011, inside 0.005. The matched half is amendment 26's −0.0012
  (`tc1-5090-69`). This box ran no matched arm, so its own file scores P54 and P55 UNTESTED, and the combined verdict is
  amendment 30's registered rule.
- **P54 HELD** on amendment 26's box (matched 0.973 [0.958, 0.988]).
- **By the rule, both flags become defaults.** They already have: e4b #1099 and grouped-nf4-gemm #470 flipped them at
  ~01:50Z, citing this box before its read was on `main`. This read was committed afterwards by the maintainer session,
  from the run's output in the private receipts store (`tc1-5090-73`, committed there as `b43ec238`), under the standing
  permission to commit another lane's finished receipts. The gain is small (2–3 %), and the shipped interval reaches 1.0.

## Amendment 33 (2026-10-05): on one stack, with load-gated draws, Unsloth/e4b 2.352 and the environment gain 0.900 (P50, P51 HELD)

Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendments 25, 29 and 33. One RTX 5090 (`tc1-5090-76`, AMD EPYC
7B13, Vast machine 145701): the token `qwen3samestack` with `TC1_STEPS=60`, no reference arm, `TC1_LOAD_GATE=6.0` and
`TC1_LOAD_RETRIES=2`. Read: [`RESULTS-tc1-samestack-box4.md`](RESULTS-tc1-samestack-box4.md).

| arm | s/step (two draws, 60 steps) | stable | host load1 median per standing draw |
|---|---|---|---|
| e4b `fused_attn4_m`, venv-unsloth (torch 2.12.1+cu130, transformers 5.5.0, triton 3.7.1) | 3.496 / 3.492 | yes (0.1 %) | 4.41 / 5.83 |
| e4b `fused_attn4_m_t28`, venv-e4b (torch 2.8.0+cu128, transformers 5.18.0, triton 3.4.0) | 3.855 / 3.913 | yes (1.5 %) | 2.94 / 5.85 |
| Unsloth `ckpt_unsloth_m`, venv-unsloth | 8.208 / 8.228 | yes (0.3 %) | 4.08 / 5.22 |

- **P50 HELD.** On one stack Unsloth/e4b reads **2.352** [2.348, 2.356 over 4 cross-draw ratios], inside [1.9, 2.9]. Unsloth peaks
  3.22 GB lower (24.27 vs 27.49 GB); its energy per step is ×1.37 e4b's. Held-out at N=60: e4b 0.7569, Unsloth 0.7557, COMPARABLE
  (|Δ| 0.0012; no reference arm, so EQUIVALENT cannot be read).
- **P51 HELD.** e4b's matched arm in venv-unsloth over venv-e4b reads **0.900** [0.893, 0.907], inside [0.80, 0.95]. Amendment 24 read
  0.882 on another host; the environment gain replicates.
- **By amendment 25's rule** (P50 HELD here, P52 HELD on amendment 25's two boxes), 2.352 becomes the Qwen3-30B-A3B 5090 position to
  quote, register `e4b.train.h2h.unsloth.qwen3.5090.2026-10-05.same-stack`. Amendment 19's 1.997 stays as the reading with e4b in the
  field image's environment.
- **The gate.** Three draws ran over a median load1 of 6.0 and were run again; their files are in
  [`receipts/tc1-5090-76/loadvoid/`](receipts/tc1-5090-76/loadvoid/). The voided draws read 3.899 and 4.119 s/step (e4b's field-image
  second draw, load 6.26 and 8.44) and 7.928 (Unsloth's second draw, load 6.33): the voided Unsloth draw was the faster one. Reading the
  first attempts instead gives 2.309 and 0.901, inside the same bands, so the gate changed no verdict.
- **A difference between the sides that the registration did not name.** The box ran e4b `c8925bb`, after the prebound Triton launches
  became the default (#1099) for triton 3.4 and 3.6. They engaged on the venv-e4b arm (216,335 prebound launches) and not on the
  venv-unsloth arm (triton 3.7.1, not covered at that commit; 0). grouped-nf4-gemm's prebound launches were in neither (its pin predates
  them). Amendment 26 read e4b's and grouped-nf4-gemm's flags together at 0.973 on the matched arm, so with e4b's flag off on both sides
  P51 would sit between about 0.876 and 0.900: inside the band either way. That is arithmetic, not a measurement. Amendment 34's box runs
  every arm with both flags off.
- P52 is not re-asked (no reference arm); it held on amendment 25's two boxes.

## Amendment 32 (2026-10-05): triton 3.7.1 alone is not the 5090's environment gain on the matched arm (0.992), and is 2.9 % on the shipped arm

Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendments 32 and 34. One RTX 5090 (`tc1-5090-74`, AMD EPYC 7B13,
Vast machine 145701, host load1 3.0–9.4). venv-e4b (torch 2.8.0) with its own triton 3.4 (`_tr0`) against triton 3.7.1 put first on the
arm's `PYTHONPATH` (`_tr1`), 60-step runs, prebound launches off. Read: [`RESULTS-tc1-tritonab.md`](RESULTS-tc1-tritonab.md).

| arm | `_tr0` s/step | `_tr1` s/step | `_tr1`/`_tr0` | prediction |
|---|---|---|---|---|
| matched (fp32 adapters) | 3.827 / 3.871 | 3.894 / 3.746 | 0.992 [0.968, 1.017] | **P59 FALSIFIED** ([0.82, 0.95]) |
| shipped (bf16 adapters) | 3.117 / 3.023 | 2.976 / 2.986 | **0.971** [0.955, 0.988] | **P60 HELD** ([0.85, 0.98]) |

- **P61 HELD.** Held-out moves −0.0008 on the matched arm and +0.0005 on the shipped arm.
- **By the registered rule the A2000 decomposition does not transfer.** On the A2000 the slice ran device-bound, and triton 3.7.1's faster
  code for grouped-nf4-gemm's kernels was the whole gain. On the 5090's host-bound step that device time is not what binds, and the
  matched arm does not move. The shipped arm gains 2.9 %.
- **What is left.** Amendment 24's 0.882 must come from torch 2.12 and/or transformers 5.5. Both removed host work on the A2000:
  attention-mask handling in torch, and the router's dtype round trip in transformers. Amendment 34 splits them on the 5090. Row
  `e4b.train.triton37.qwen3.5090.2026-10-05`.


## Amendment 31 (2026-10-05): Qwen3-30B-A3B's absmax pair, read over 60 steps; the double-quantized absmax becomes the default

Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendments 28 and 31. One RTX 5090 (`tc1-5090-75`, AMD EPYC 7B13,
Vast machine 145701; host load1 3.8–4.9 throughout): the token `qwen3dqab` with `TC1_STEPS=60`. Read:
[`RESULTS-tc1-dqab-qwen3-60.md`](RESULTS-tc1-dqab-qwen3-60.md).

| | `_dq0` (fp32 absmax) | `_dq1` (double-quantized) |
|---|---|---|
| s/step, two draws (60 steps) | 3.751 / 3.727 (0.6 % apart) | 3.777 / 3.805 (0.7 % apart) |
| peak | 27.44 GB | 26.10 GB |
| held-out at N | 0.7597, 0.7583 | 0.7567, 0.7572 |

- **P56 HELD.** `_dq1`/`_dq0` reads 1.014 [1.007, 1.021], inside [0.97, 1.03], and the peak falls 1.339 GB, inside [1.25, 1.45]. The
  double-quantized absmax costs 1.4 % of the step on Qwen3-30B-A3B and 2.3 % on Mixtral.
- **P58 HELD.** Held-out moves −0.0021 on Qwen3 and −0.0028 on Mixtral, both inside 0.005.
- **By amendment 28's rule, with P57 already HELD, the double-quantized absmax becomes the default for resident training.** That is
  e4b's own PR, citing these boxes. The TC1 harness keeps `E4B_ABSMAX_DQ` explicit, because its boxes are registered instruments.
- On a quiet host the 60-step pairs were stable. Amendment 28's Qwen3 box lost both pairs on a busy one.

## Amendment 29 (2026-10-05): the same-stack pair over 60 steps, unstable again on a busy host; amendment 33 gates the draws on host load

Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendments 29 and 33. One RTX 5090 (`tc1-5090-72`, AMD EPYC 7C13,
Vast machine 45511): the token `qwen3samestack` with `TC1_STEPS=60` and no reference arm. Read:
[`RESULTS-tc1-samestack-box3.md`](RESULTS-tc1-samestack-box3.md).

| arm | s/step (two draws, 60 steps) | stable | host load1 median per draw |
|---|---|---|---|
| e4b `fused_attn4_m`, venv-unsloth | 3.979 / 3.490 | **no** (13.1 %) | 22.9 / 11.0 |
| e4b `fused_attn4_m_t28`, venv-e4b | 3.834 / 4.087 | **no** (6.4 %) | 8.8 / 6.4 |
| Unsloth `ckpt_unsloth_m` | 8.759 / 8.500 | yes (3.0 %) | 13.6 / 14.8 |

- **P50 and P51 UNTESTED, final under amendment 29.** Unsloth and e4b's field-image arm are COMPARABLE to e4b's same-stack arm at N=60.
  There is no reference arm, so EQUIVALENT cannot be read.
- **The box ran on the machine whose load had already spoiled amendment 28's Qwen3 box.** That was found only after this box was queued.
  The 60-step window alone does not survive a busy host: the slow e4b draw ran at twice the other's load.
- **Not readings.** Across the three same-stack boxes, the medians sit near 2.3–2.45 for Unsloth/e4b and 0.90–0.94 for the environment.
- **Amendment 33** adds load-gated draws: an arm whose median host load1 exceeds 6.0 is set aside and run again, at most twice. It re-asks
  P50 and P51 once more, off the busy machines. Its reading is final.

## Amendment 28 (2026-10-05): the double-quantized expert absmax costs Mixtral 2.3 % of its step for 2.04 GB; Qwen3's speed pair was unstable

Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendments 28 and 31. Two RTX 5090s, e4b against itself on the
matched arm, resident, `E4B_ABSMAX_DQ=0` (`_dq0`) against `=1` (`_dq1`), two draws a side in ABBA order. Read (both boxes in one pass):
[`RESULTS-tc1-dqab.md`](RESULTS-tc1-dqab.md).

| family | `_dq0` s/step | `_dq1` s/step | `_dq1`/`_dq0` | peak `_dq0` → `_dq1` | held-out Δ | prediction |
|---|---|---|---|---|---|---|
| Mixtral-8x7B (`tc1-5090-71`) | 3.807 / 3.917 | 3.932 / 3.970 | **1.023** [1.004, 1.043] | 31.07 → **29.03** GB (−2.04) | −0.0028 | **P57 HELD** |
| Qwen3-30B-A3B (`tc1-5090-70`) | 3.801 / 4.019 (5.6 % apart) | 3.947 / 5.014 (23.8 % apart) | not read | 27.15 → 25.82 GB (−1.33) | +0.0026 | **P56 UNTESTED** |

- **Mixtral (P57 HELD).** Double-quantizing the absmax costs 2.3 % of the step and saves 2.04 GB. The registered arithmetic predicted
  2.10 GB.
- **Qwen3 (P56 UNTESTED).** Both speed pairs were unstable. The box's host load average tracked the draws: 6.6 and 7.3 on the `_dq0`
  draws, 12.1 and 19.4 on the `_dq1` draws. Its peak and held-out readings do not depend on the host and sit inside their bands, but
  P56 is one prediction, so it stays UNTESTED.
- **P58 UNTESTED** for the same reason: its Mixtral half reads −0.0028, its Qwen3 half is unread.
- **By the registered rule `E4B_ABSMAX_DQ` stays opt-in for now.** Amendment 31 re-asks Qwen3's pair over 60 steps on a quieter machine.
- **Why boxes go unstable.** The step is host-bound, and on multi-tenant hosts the other tenants' load shows up directly in it:
  - this box (load 6.6–19.4);
  - `tc1-5090-66`'s shipped pair (load 4 vs 19);
  - `tc1-5090-68` (load 30–37 throughout).

  Machine 151350 is the exception: it lost pairs at a load near 1.

## Amendment 26 (2026-10-05): prebound Triton launches take 2.7 % off the matched arm's step; the shipped pair was unstable, so the flags stay opt-in

Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendments 26 and 30. One RTX 5090 (`tc1-5090-69`, Intel Core
Ultra 9 285K, Vast machine 151350). The box ran e4b against itself in venv-e4b (triton 3.4), comparing `E4B_TRITON_PREBIND` and
`GNF4_TRITON_PREBIND` both 0 (`_pb0`) against both 1 (`_pb1`), two draws a side in ABBA order. Read: [`RESULTS-tc1-prebindab.md`](RESULTS-tc1-prebindab.md).

| arm | `_pb0` s/step | `_pb1` s/step | `_pb1`/`_pb0` | held-out at N, `_pb0` / `_pb1` |
|---|---|---|---|---|
| matched (fp32 adapters) | 2.555 / 2.505 | 2.447 / 2.476 | **0.973** [0.958, 0.988] | 0.8524, 0.8493 / 0.8513, 0.8480 |
| shipped (bf16 adapters) | 1.926 / 1.772 (8.3 % apart) | 1.972 / 1.970 | not read | 0.8135, 0.8149 / 0.8113, 0.8116 |

- **P54 HELD.** On the matched arm the prebound launches take 2.7 % off the step. The launches were prebound as registered: the `_pb1`
  arm counted 73,591 prebound e4b launches and 24,546 prebound grouped-nf4-gemm launches, against 313 and 30 through Triton's own path.
  The `_pb0` arm counted none.
- **P53 UNTESTED.** The shipped arm's flags-off draws were 8.3 % apart.
- **The shipped pair, unread, points the other way.** Its `_pb1` draws (1.972, 1.970) are slower than both `_pb0` draws. Had the pair
  been stable, P53 ([0.90, 0.98]) would have read FALSIFIED at about 1.07. Host noise or a real cost on this arm: this box cannot tell.
- **P55 UNTESTED** for the shipped half. The matched half moved 0.0012 nats.
- **By the registered rule the flags stay opt-in.**
- **One machine, several boxes.** This box, `tc1-5090-67` and three earlier stable ones (`-55`, `-62`, `-63`) all ran on one machine.
  The launcher ranks it near the top. Two of its last three boxes lost a pair to instability. The re-asks avoid it.
- **Amendment 30** re-asks the shipped pair over 60 steps (50-step medians) on another machine. Its reading is final.

## Amendment 27 (2026-10-05): the re-draw, also unstable on one pair; amendment 29 re-asks P50 and P51 over 60 steps

Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendments 25, 27 and 29. One RTX 5090 (`tc1-5090-68`, AMD EPYC
7663), the same token on a machine box 1 did not use. Read: [`RESULTS-tc1-samestack-box2.md`](RESULTS-tc1-samestack-box2.md).

| arm | s/step (two draws) | stable |
|---|---|---|
| e4b `fused_attn4_m`, venv-unsloth | 4.060 / 3.755 | **no** (7.8 %) |
| Unsloth `ckpt_unsloth_m` | 8.989 / 9.246 | yes (2.8 %) |
| e4b `fused_attn4_m_t28`, venv-e4b | 4.388 / 4.314 | yes (1.7 %) |

- **P52 HELD again.** Unsloth and e4b's reference read INSIDE-DRAW-NOISE against e4b's fused arm, and parity PASSES.
- **P50 and P51 UNTESTED, final under amendment 25.** This time e4b's same-stack pair was the unstable one.
- **The instrument, not a prediction.** Across the two boxes, three different pairs lost stability. Each draw's median is over 10 steps.
- **Amendment 29** re-asks P50 and P51, with their bands unchanged, over 60-step runs (50-step medians) on a third machine; its reading
  is final.

## Amendment 25, first box (2026-10-04): on one stack the matched set holds; the speed pairs were unstable, so one re-draw is registered

Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendments 25 and 27. One RTX 5090 (`tc1-5090-67`, Intel Core
Ultra 9 285K). The box ran:

- e4b's matched set in Unsloth's venv (torch 2.12.1+cu130, transformers 5.5.0);
- Unsloth as in TC1;
- e4b's field-image arm (`_t28`) alongside.

Read: [`RESULTS-tc1-samestack-box1.md`](RESULTS-tc1-samestack-box1.md).

| arm | s/step (two draws) | stable |
|---|---|---|
| e4b `fused_attn4_m`, venv-unsloth | 2.220 / 2.188 | yes (1.5 %) |
| Unsloth `ckpt_unsloth_m` | 4.913 / 5.899 | **no** (18.2 %) |
| e4b `fused_attn4_m_t28`, venv-e4b | 2.305 / 2.507 | **no** (8.4 %) |
| e4b `reference_attn4_m`, venv-unsloth | 29.582 (one draw) | — |

- **P52 HELD.** On one stack Unsloth and e4b's reference read EQUIVALENT to e4b's fused arm (held-out Δ 0.0039 and 0.0035 against a
  band of 0.0105), and e4b's parity PASSES (Δ final 0.0038).
- **P50 and P51 UNTESTED.** Two of the three speed pairs were unstable. The GPU logs show steady SM clocks and a host load average near
  1.2, so they do not explain it.
- **No ratio is read.** The medians would read about 2.45 for Unsloth/e4b and 0.92 for the environment, but those are not readings and
  are not quoted.
- **Amendment 27** registers one re-draw on another machine with amendment 25's predictions unchanged. Its reading is final.

## Amendment 24 (2026-10-04): the 5090's fp32 `bmm` costs host time per new shape, not per torch version; in Unsloth's environment e4b's matched arm steps 0.882×

Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendment 24. One RTX 5090 (`tc1-5090-66`, AMD EPYC 7B13). The box
ran two parts:

1. A replay with no model. `bmm_bench.py` timed the padded LoRA delta's batched products at the 96 recorded real-router shapes, with
   the queue drained before each call.
2. e4b against itself in two environments, on TC1's qwen3 tokens and recipe, two draws a side in ABBA order:
   - `_tv0`: the field image's torch 2.8.0+cu128, transformers 5.18.0, triton 3.4.0;
   - `_tv1`: Unsloth's venv, torch 2.12.1+cu130, transformers 5.5.0, triton 3.7.1.

Read: [`RESULTS-tc1-bmmab.md`](RESULTS-tc1-bmmab.md).

**The replay** (median host µs per forward `bmm`):

| torch | fp32, recorded shapes | bf16 | fp32, new shapes | fp32, repeated shapes | fp32, one fixed shape |
|---|---|---|---|---|---|
| 2.8.0+cu128 | 88.9 | 38.3 | **119.4** | **37.9** | 37.4 |
| 2.12.1+cu130 | 93.4 | 38.3 | **120.9** | **37.9** | 35.9 |

- **P44 FALSIFIED.** Outside training, the fp32 call costs 89 µs at the recorded shapes, 2.3× bf16's. The profiles' 197–324 µs
  (under the profiler, mid-training) does not reproduce.
- **P45 HELD.** What does reproduce is a per-shape cost: a shape the process has not seen costs 119 µs, and a repeated one 38 µs,
  the same as bf16. The routing makes nearly every call's padded shape new.
- **P46 FALSIFIED.** torch 2.12.1+cu130 pays the same cost (120.9 µs).
- The cublasLt and TF32 rows are descriptive and cost more on new shapes (223–269 µs). The bucket rows are not a clean first
  exposure: the cold passes had already used most of the bucketed widths.

**The environment A/B:**

| arm | `_tv0` s/step | `_tv1` s/step | `_tv1`/`_tv0` | held-out at N, `_tv0` / `_tv1` |
|---|---|---|---|---|
| matched (fp32 adapters) | 3.915 / 3.895 | 3.495 / 3.394 | **0.882** [0.867, 0.897] | 0.8522, 0.8531 / 0.8467, 0.8483 |
| shipped (bf16 adapters) | 3.009 / 3.171 (5.3 % apart) | 2.950 / 2.897 | not quoted | 0.8160, 0.8104 / 0.8124, 0.8133 |

- **P47 HELD.** In Unsloth's environment e4b's matched arm steps 0.882× as long, with held-out 0.005 lower. The peak is unchanged at 27.2 GB.
- **P48 and P49 UNTESTED.** The shipped arm's torch 2.8 draws were 5.3 % apart, so its pair cannot be quoted. Its medians read 0.946.
- The replay shows the `bmm` is not the cause: torch 2.12 pays the same per call. The environment changes torch, transformers and
  triton together, so this box does not say which one gives the gain.

**What follows, as registered.**

- P44 failed, so grouped-nf4-gemm does not change.
- P46 failed, so the docs do not change.
- The rule that would have named an environment asymmetry in STATUS needed P48 as well, and P48 went untested.
- What stands is P47's reading. Every 5090 position so far ran e4b in the field image's environment and Unsloth in torch 2.12.1's.
  e4b requires only `torch>=2.2` and `transformers>=5.0`, so both are e4b environments.
- A position with both frameworks on one stack needs its own registration. Rows: `e4b.train.env-ab.qwen3.5090.2026-10-04` and
  `e4b.train.bmm-host-replay.5090.2026-10-04`.

## Amendment 23 (2026-10-04): a memory census of e4b against Unsloth — with the absmax double-quantized, e4b's peak is 0.43 GB above Unsloth's, all of it transient

Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendment 23. One RTX 5090 (`tc1-5090-65`), Qwen3-30B-A3B's matched
set at micro-batch 1 × accum 8, one draw per arm. `tc1_arm.py --mem-census 1` recorded the allocator's history from before the load,
and reduced its snapshots on the box. The census slows the step, so no speed is read. Read: [`RESULTS-tc1-memcensus.md`](RESULTS-tc1-memcensus.md).

| live at the peak (GB) | e4b, fp32 absmax (default) | e4b, `E4B_ABSMAX_DQ=1` | Unsloth |
|---|---|---|---|
| frozen expert weights (NF4) | 14.496 | 14.496 | 14.496 |
| expert absmax | **1.812** | 0.460 | 0.460 |
| other frozen (bf16 1.270, NF4 attention 0.468) | 1.738 | 1.738 | 1.738 |
| adapters / their grads / optimizer state | 2.570 / 2.570 / 1.305 | 2.570 / 2.570 / 1.305 | 2.570 / 2.570 / 1.305 |
| transient | **1.532** | **1.536** | 1.095 |
| **peak allocated** | **26.022** | **24.676** | **24.244** |

- **The census measures what it should (P41 HELD).** It finds e4b's fp32 absmax at exactly the analytic 1.8119 GB (29.0 B expert
  parameters / 64 × 4 bytes), and the double-quantized absmax at 0.4602 GB, 3.94× smaller.
- **It accounts for nearly everything (P42 HELD).** At least 99.96 % of each arm's peak goes to named groups.
- **The gap is smaller than registered (P43 FALSIFIED, below [0.5, 2.5] GB).** Every static class is byte-for-byte the same in both
  frameworks except the absmax. With it double-quantized, e4b's peak sits **0.43 GB** above Unsloth's, and the whole gap is transient.
- **What the transients are.** e4b's largest are grouped-nf4-gemm's padded LoRA delta (`kernel/nf4_qlora.py:_lora_delta_padded`):
  - the zero-padded input block it allocates in the adapters' dtype, fp32 on this arm (0.62 GB, 2 live);
  - the batched products' outputs (0.45 GB, 3 live).

  Unsloth's largest transient is its NF4 dequant (0.81 GB).
- **At e4b's defaults the gap is 1.78 GB, and 1.35 GB of it is the fp32 absmax.** Those defaults are what this lane has quoted;
  `E4B_ABSMAX_DQ=1` removes the absmax part.
- **As registered, this is a measurement.** Each fix the read names gets its own A/B. The candidates are the absmax default and the
  padded block's dtype and size on fp32 adapters. Row `e4b.train.memory-census.qwen3.5090.2026-10-04`.

## Amendment 22 (2026-10-04): grouped-nf4-gemm's dense route against its fused kernels — far faster on Mixtral, far slower on Qwen3-30B-A3B

Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendment 22. Each box ran e4b against itself on the matched arm,
`GNF4_TRAIN_GEMM=fused` against `=dense` (grouped-nf4-gemm#459: one present expert dequantized at a time, its GEMM through `torch.mm`).
There were two draws a side in ABBA order, with grouped-nf4-gemm at #459's merge. Read: [`RESULTS-tc1-denseab.md`](RESULTS-tc1-denseab.md),
both boxes in one pass.

| family | box, host | fused s/step | dense s/step | **dense/fused** | held-out Δ (dense − fused) | prediction |
|---|---|---|---|---|---|---|
| Mixtral-8x7B (2 of 8 experts), resident, `E4B_ABSMAX_DQ=1` both sides | `tc1-5090-62`, Core Ultra 9 285K | 5.491 / 5.541 | 3.592 / 3.589 | **0.651** [0.648, 0.654] | −0.0021 | **P38 HELD** ([0.55, 0.90]) |
| Qwen3-30B-A3B (up to 128 present experts) | `tc1-5090-59`, EPYC 9454P | 2.949 / 2.976 | 8.674 / 8.784 | **2.947** [2.915, 2.979] | −0.0001 | **P39 FALSIFIED** ([0.85, 1.15]) |

**P40 HELD:** both families' held-out stay within 0.01, and the peaks do not move (Mixtral 29.03 / 29.02 GB).

**Why the two families split.** The dense route trades the fused kernels' device time for launches. Each present expert costs a dequant and
a GEMM, in every projection call. On Mixtral that is 8 experts per call, and the expert GEMMs are large (about 1,000 rows each), so the
device-side gain dominates. An RTX A2000 replay put the dense forward at about 0.3× and its dgrad at about 0.13× of the fused kernels on
those shapes. On Qwen3-30B-A3B a step makes about 1,150 projection calls with up to 128 experts each, roughly 295,000 extra launches. On a
5090, where this step is already launch-bound, that cost dominates.

**Step-0 held-out on Qwen3 moves with every GEMM implementation**, not only this route: the fused kernels read 1.9714 on an H100 and
1.9441 on a 5090, the reference loop 1.9551 / 1.9508, the grouped_mm route 1.9614, and the dense route 1.9672. All of them train to the
same held-out (here Δ −0.0001).

**Decision, as registered.** P38 and P40 held, so grouped-nf4-gemm's `auto` takes the dense route on cards other than sm_90 for calls
with at most 16 present groups (grouped-nf4-gemm#463). Qwen3-30B-A3B's calls stay on the fused kernels. Rows
`e4b.train.dense-route.mixtral.5090.2026-10-04` and `e4b.train.dense-route.qwen3.5090.2026-10-04`. No position against another framework is
read here; the Mixtral position with the new default needs its own box.

## The position (register `e4b.train.h2h.unsloth.qwen3.5090.2026-10-02`)

On one RTX 5090 at the tp4 field recipe — alpaca, seq 2048, micro-batch 2 × accum 4, r 16 / alpha 16, lr 2e-4, AdamW-8bit, linear
warm-up 5, N = 20 — with **the same 642,514,944 trainable parameters, the same per-slot LoRA A init (`matched:3407`, name-free
sha `f7832488926eda91` in every matched arm, 12,480 / 12,480 slots), fp32 adapters in both frameworks, the same tokens
(sha `bfc742f67e37`)**, and Unsloth 2026.9.14 on the route its installer names for Blackwell (torch 2.12.1+cu130, the `grouped_mm`
backend engaged on every step: 384 `torch._grouped_mm` calls per step, 0 loop calls):

| | e4b `fused_attn4_m` | Unsloth `ckpt_unsloth_m` | reading |
|---|---|---|---|
| s/step, median of steps 11..20, two draws | 5.692 / 5.683 (**5.688**) | 8.160 / 8.183 (**8.171**) | **Unsloth/e4b 1.437 [1.434, 1.440]** over the four cross-draw ratios; both STABLE (0.2 % / 0.3 %) |
| tokens/s | 265.2 | 176.4 | |
| peak VRAM | 27.83 GB | **24.27 GB** | Unsloth lower by 3.57 GB |
| energy per step (dmon) | 921.3 J | **660.7 J** | Unsloth ×0.72 |
| held-out loss at N = 20 (8 rows) | 1.9505 → 0.8516 / 0.8487 | 1.9583 → 0.8482 / 0.8476 | Δ −0.0034 ± 0.0028 SE: **EQUIVALENT** (band 0.0123 = 3 × the in-draw fused-vs-reference Δ; draw-noise floor 0.0030); step-0 Δ 0.0078 SAME-BYTES-CLASS |
| frozen base | C1 bit-exact, the real-storage flip control detects | C1 bit-exact, control detects | attention projections SAME-BYTES (nf4/64 + double-quant); expert stacks N-A (e4b nf4/64, Unsloth nf4/64 + dq: its loader exposes no double-quant knob) |

e4b's fused path against its own per-expert reference on the same box: Δ final train loss 0.00041, median step |Δ| 0.00141 — PASS;
×10.39 faster per step than the reference (5.692 vs 59.171 s) at ×1.03 its peak. The secondary pair (micro-batch 1 × accum 8, run
because HF OOMed) reads 1.725 the same way (15.021 vs 8.706 s, single draws).

**The standing position is refuted and superseded.** `e4b.train.h2h.unsloth.qwen3.5090.2026-09-19` quoted 4.490 (6.47 vs 29.05 s/step)
at this recipe. That Unsloth arm ran on the field image's torch 2.8.0+cu128, where `torch._grouped_mm` is sm_90-only and Unsloth's
loader silently selects its `native_torch` per-expert loop; the comparison was e4b's fused kernel against Unsloth's fallback. TC1's
registered prediction P1 put the matched ratio in [2.0, 5.0] with "below 1.5 refutes the standing position": it reads **1.437,
FALSIFIED**, and this bundle's row supersedes the 2026-09-19 row (its `.quality-n20`, `.e4b-internal-parity` and `.secondary-mb1`
rows likewise). The loop is still what a torch-2.8 user gets: on the native box the same Unsloth install on torch 2.8 (`ckpt_unsloth_t28`,
backend `native_torch`, 96 loop calls per step) takes 61.3 s/step, 6.565 × e4b's matched arm on that host.

## What the ratio measures, and what it does not

- **Both steps are host-bound on a 5090, and e4b's is the less host-bound of the two.** The profiled matched arms on the same box:
  e4b device-busy 0.480, 141 k device events and 753 k CPU-side ops per step; Unsloth device-busy 0.234, 612 k device events and
  7.43 M CPU-side ops per step (P8, which predicted ≥ 0.5 on the grouped_mm arm, is FALSIFIED). The fused path records 4.3 × fewer
  device events and 9.9 × fewer CPU-side ops per step than the dequant-then-dense-grouped-GEMM path, and that dispatch cost — not the
  GEMM — is where the gap is on this card.
- **Absolute s/step does not travel between hosts; the ratio does.** The same e4b matched arm takes 3.16 s/step on the i9-13900 host,
  5.69 on the Threadripper host and 9.34 on the Xeon E5-2698 v4 host (same card class, same code, 0 recompiles); the Unsloth matched
  arm tracks it (4.50 / 8.17). TC1b's P3 ("the 200-step and 20-step medians agree within 10 %") is FALSIFIED mechanically because the
  reducer compares across those two hosts (−44 %); within one draw the 11..20 and 11..200 windows agree within 3.5 %. Nothing on this
  page divides numbers from different boxes; every ratio is within one box.
- **e4b on the cu130 torch is 14 % faster per step** (`fused_attn4_m_t212`: e4b + grouped-nf4-gemm installed into the comparator's
  torch 2.12.1+cu130 venv, 8.069 vs 9.336 s on the native box; VALID, EQUIVALENT-class quality). The position above is e4b on the
  field image's torch 2.8; a position with both frameworks on cu130 is not quoted here (one draw, one host).
- **Unsloth's footprint rows are in its favour** (peak VRAM −3.57 GB, energy ×0.72 per step at the matched configuration). The
  matched configuration carries fp32 expert adapters on e4b's side by design; e4b as shipped (bf16 expert adapters) peaks at 24.58 GB
  and 843 J/step on the native box — the same footprint as Unsloth's — so the gap is the matched set's precision choice, not the kernel.
  Peak VRAM and J/step are quoted beside the ratio wherever it is quoted.

## The native and labelled rows (box `tc1-5090-14`, each against the box's own e4b matched arm, single draws)

| row | s/step | vs e4b matched (9.336) | held-out at N (e4b matched 0.8481) | note |
|---|---|---|---|---|
| Unsloth native-best (grouped_mm, speed tilt, native init, fp32) | 15.245 | 1.633 | 0.8467 | vs e4b as shipped (6.267): **2.433** — the "both at their own defaults" row, never the position; P7 (≥ 2 vs e4b matched) FALSIFIED |
| Unsloth torch-2.8 (`native_torch` loop, tp4's venv) | 61.295 | 6.565 | 0.8471 | 23.9 tok/s, 3885 J/step; P1b (within 15 % of tp4's 29.05) FALSIFIED on this slower host |
| Unsloth `unsloth_triton` backend | 14.754 | VOID | — | requested backend never engaged (`moe_backend_selected grouped_mm`, 0 triton calls): a row, not a position |
| e4b as shipped (bf16 expert adapters, N(0, 1/r) init) | 6.267 | **0.671** | **0.8108** | 1.49 × faster than the matched e4b arm and 0.037 nats LOWER held-out at N = 20; P4 (1.05–1.3 × and ≥ 0.01 lower) FALSIFIED on the speed half |
| e4b `dgrad=False` (`enable_fast_train`'s default) | 9.248 | 0.991 | 0.8512 | the dgrad kernel is not where the step goes at this shape |
| e4b on torch 2.12.1+cu130 | 8.069 | 0.864 | 0.8506 | see above |
| axolotl native-best (scattermoe) | — | — | — | INSTALL_FAILED here (the harness's uv index strategy, amendment 3); on the re-ask box REFUSED at load: the KernelsPlugin fetches `kernels-community/rotary` and the harness runs arms with the Hub offline — a harness limit, not a reading |
| HF on torch 2.14 + `grouped_mm` at mb1 | — | — | — | NOT_RUN here (its gate reads the matched box's OOM); on the re-ask box OOM at load (31.35 GB in use): bf16 expert stacks do not fit the 32 GB card under any experts implementation |

HF + PEFT (transformers 5.18.0, bf16 expert stacks, `target_parameters`) OOMs on the 32 GB card at micro-batch 2 and at micro-batch 1
(31.34 GB in use at a 20 MiB allocation), and so does the torch-2.14 `experts_implementation="grouped_mm"` variant at micro-batch 1 on the
re-ask box: P5 HELD. No HF position exists on this card at this recipe.

> **Correction (2026-10-02, TC1 amendment 4).** The axolotl failure below was this harness's, not axolotl's. axolotl's loader keeps every
> `*.gate` router in fp32 on purpose (`loaders/model.py:611-616, 1449-1451`) because its own trainer runs the forward under bf16 autocast;
> this harness runs no autocast, so the fp32 router met bf16 activations. The receipts show the router in float32 on every failing axolotl
> arm and in bfloat16 on the HF and e4b arms. `e4b.train.h2h.unsloth.qwen3.5090.2026-10-02.axolotl-unsupported` is retired, and the arm is
> re-asked with the routers cast to bf16 after load (what autocast computes per call).

> **The re-run (`tc1-5090-30`, 2026-10-02, $0.43; `RESULTS-tc1-5090-30-axolotl.md`).** With the routers cast, axolotl trains the matched set:
> 7.759 / 7.642 s/step against e4b's 5.408 / 5.464 — **axolotl/e4b 1.416 [1.398, 1.435]**, both pairs stable, e4b faster per step; axolotl's peak
> 26.88 GB against 27.81 and its energy ×2.31; held-out COMPARABLE (about 0.01 nats). Its 96 expert stacks are NF4 (axolotl's parametrized
> `quantize_moe_experts`, double-quantised statistics), and its step-0 loss sits 0.026 above e4b's: the two quantisers' bytes differ. axolotl's
> scattermoe native-best trains too, at **0.929 × e4b's matched step** and 25.84 GB (one draw, its own init — a labelled row). TC1 P6 FALSIFIED:
> axolotl trains, faster than the [1.5, 6] band. Register `e4b.train.h2h.axolotl.qwen3.5090.2026-10-02` and `.scattermoe-native`.

> **Native-best against native-best (`tc1-5090-33` and `tc1-5090-34`, amendments 5-7; `RESULTS-tc1-nativebest.md`).** Each framework as its
> users run it, on one box, two draws each. On `tc1-5090-34` **Unsloth native-best / e4b as shipped is 1.794 [1.790, 1.797]**, both pairs
> stable: P13's Unsloth half HELD (register `.native-vs-native`, a labelled row). axolotl's scattermoe draws were 10.8 % apart, with
> 216, 61-65 and 76-81 s warm-up steps inside the window, so P13 is UNTESTED and no "fastest" statement is made. Amendment 8 re-asks the
> axolotl half over steps 101..200 of a 200-step run (P14).
>
> **P14 (`tc1-5090-35`, amendment 8).** Over steps 101..200 of a 200-step run, **axolotl scattermoe / e4b as shipped is 0.911
> [0.902, 0.921]**: P14 HELD, and axolotl is faster at steady state (register `.native-steady-state`). e4b still finishes the 200-step run
> first, because of axolotl's warm-up, and axolotl's held-out matches the matched curve while e4b as shipped sits 0.024-0.027 above it.
>
> **P15 (`tc1-5090-36`, amendment 9).** The same box on a second host (Ryzen 9 3900X, machine 45501): **0.901 [0.892, 0.910]**, P15 HELD.
> axolotl's steady-state lead over e4b as shipped holds on two hosts with different CPU classes, and e4b again finishes the 200-step run first.

**axolotl 0.20.0 on Qwen3-30B-A3B at its pins (torch 2.14.0+cu130, transformers 5.17.0, peft 0.21.0) does not train** (box `tc1-5090-20`,
with the venv installing after amendment 3): both matched draws loaded (`quantize_moe_experts` packed all 96 stacks, PEFT's
`target_parameters` on the expert stacks, the attention projections by module) and died in the first forward inside transformers'
`Qwen3MoeTopKRouter` — `F.linear(hidden_states, self.weight)`: bf16 activations against an fp32 router weight — the same exception on
the 24 GB and H100 boxes (five attempts, three cards). The router was not a target; the HF arm's own loader (transformers 5.18.0, the
same model, the same quantizer) builds a bf16 router and trains; the router code is identical in 5.17.0 and 5.18.0, so the fp32 weight
is a property of the model axolotl's `ModelLoader` hands over — whether its post-quantisation conversion or 5.17.0's bnb path leaves it,
this bundle does not establish. On those boxes the harness wrote no receipt for the crash (it died after load, outside the loader's
own handler — TC3 amendment 4 makes such a crash a row), so the mechanical verdict is HARNESS_ERROR and P6 reads UNTESTED; by the
logs the row is UNSUPPORTED, and no axolotl position is quoted on this family. axolotl trains Granite's 4-bit path in lane TC2 (its
own bundle), so this is the family, not the framework.

## TC1b — the long curve re-asked with matched init and precision (box `tc1-5090-15`)

200 steps, 16 held-out rows every 40 steps, paired by row:

| eval step | e4b matched | Unsloth matched | e4b as shipped | Δ Unsloth − e4b | Δ shipped − matched |
|---|---|---|---|---|---|
| 0 | 1.9607 | 1.9604 | 1.9607 | −0.0004 | 0 |
| 40 | 0.7880 | 0.7885 | 0.7894 | +0.0005 | +0.0014 |
| 80 | 0.7709 | 0.7724 | 0.7741 | +0.0015 | +0.0032 |
| 120 | 0.7665 | 0.7669 | 0.7674 | +0.0004 | +0.0009 |
| 160 | 0.7626 | 0.7646 | 0.7671 | +0.0020 | +0.0045 |
| 200 | 0.7687 | 0.7688 | 0.7945 | +0.0001 | **+0.0258** |

- **EQUIVALENT at every eval** (largest paired |Δ| 0.0020, at step 160; TC1b P1 HELD): with init, precision, tokens and schedule matched,
  the two 4-bit MoE training paths produce the same curve to 200 steps. s/step over steps 11..200: 3.162 vs 4.505 (1.425 within this
  box, consistent with the position's 1.437).
- **The as-shipped e4b curve ends 0.0258 nats above the matched one at step 200** while being faster per step (2.527 s, 1.25 ×) — the
  direction P38 reported against e4b, now reproduced on e4b's own side with the comparator out of the picture: the loader's defaults
  (bf16 expert adapters, N(0, 1/r) init) are the cause, not the kernel. It is NOT P38's shape (TC1b P2 FALSIFIED): the shipped curve
  is above the matched one at step 40 too (+0.0014), with no early lead; on the N = 20 instrument (8 rows) it reads 0.037 LOWER.
  The two instruments disagree at step 20–40 and agree from 80 on; an open item for `load_moe_4bit_streaming`'s adapter defaults.
- **The tp2/P38 anchor pair does not reproduce** (TC1b P4 FALSIFIED): at that fixture (clinical text, seq 512, 1 × 1, r 8, N 60) the ratio
  is 2.428 with Unsloth on cu130 grouped_mm (1.095 vs 0.451 s/step) and 5.147 with Unsloth on torch 2.8 (2.322 s/step) against
  tp2's 1.457 and P38's 1.413 (e4b 0.35.0 / grouped-nf4-gemm 0.30.0 vs Unsloth 2026.9.2). Both frameworks moved between those
  cuts and these; the old anchors stand as dated measurements on their cuts and are not repeated as current.
- Scaling points, never positions: at seq 2048 × micro-batch 1 × accum 1 the matched ratio is 1.918 (1.073 vs 0.559 s); at r 64 /
  alpha 64 both frameworks OOM at step 1 on the 32 GB card (31.9 / 32.4 GB).

## Predictions, scored mechanically

TC1 (`RESULTS-tc1-combined.md`): P1 FALSIFIED (1.437, below 1.5), P1b FALSIFIED (host), P2 HELD (draws), P3 HELD (equivalence), P4
FALSIFIED (shipped 1.49 ×, not 1.05–1.3 ×; the quality half held), P5 HELD (HF OOM, both implementations), P6 **UNTESTED** (on the matched and native boxes the
INSTALL_FAILED was this harness's own install line; on the axolotl box the framework's crash left no receipt — UNSUPPORTED by the logs,
not a mechanical reading), P7 FALSIFIED (1.633 < 2 within its box), P8 FALSIFIED (device-busy 0.234), P9 HELD (parity),
P10 UNTESTED (the expert slots are different regimes by construction). TC1b: P1 HELD, P2 FALSIFIED, P3 FALSIFIED (cross-host),
P4 FALSIFIED.

## Harness defects found by the boxes (fixed in amendment 3, `../../tc1/TC1-PREREG.md`)

1. The axolotl venv never installed: uv's first-index strategy took `packaging` from PyTorch's cu130 index, where axolotl's
   `packaging==26.0` does not exist. Reproduced off-box and fixed with `--index-strategy unsafe-best-match` (dry-resolved: axolotl 0.20.0,
   torch 2.14.0+cu130, transformers 5.17.0, peft 0.21.0). The axolotl rows are re-asked on their own box (`qwen3axolotl`); this bundle
   quotes no axolotl row.
2. The reducer read the profiled arms' kernel-table sidecars as HARNESS_ERROR rows (cosmetic; skipped now).
3. The native box's HF torch-2.14 row's gate could not be read on that box (re-asked with the axolotl rows: OOM).
4. Found by the re-ask box and fixed after it (TC3 amendment 4): an exception after load left no receipt; the scattermoe row needs the
   KernelsPlugin's kernels fetched before the Hub goes offline (open).

## Not claimed here

No position on any other family (lane TC2) or under a memory budget (lane TC3) — each has its own registration and bundle; no axolotl
position on this family (it does not train it at its pins, above). No cross-box ratio. **The H100 NVL reading is the opposite sign** — Unsloth/e4b 0.621, Unsloth faster — see
[`../tc1c/README.md`](../tc1c/README.md): the position above is a 5090 position, where both paths are launch-bound and Unsloth's grouped
GEMM is not one launch per call.
