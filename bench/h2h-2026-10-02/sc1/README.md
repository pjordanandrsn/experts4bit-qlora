# SC1: Qwen3-30B-A3B served on three RTX 5090 boxes. No position quoted; P13 holds on all three boxes (lane SC1 of #846; 2026-10-03)

Pre-registration: [`../../sc1/SC1-PREREG.md`](../../sc1/SC1-PREREG.md), with amendments A1–A13. A12 ([#933](https://github.com/pjordanandrsn/experts4bit-qlora/pull/933))
is the SGLang prefill scorer below. Each box's arms ran under the registered driver at the box's own commit. All three are
read at read time by **main's reducer** (`87506a3`, sha256 `89ffa3fc20d12072`), which carries A10's census rule and A11's
cross-box reading. Each box's own on-box reduction is kept beside it as `RESULTS-sc1-<box>-onbox.md`.

| box | run (receipt) | e4b | engines | host (driver; board power limit, max clock) | $ |
|---|---|---|---|---|---|
| A | `sc1a-5090-2` (adertha-receipts `db915fc`) | `32d424e` (before A10) | e4b licensed pack (window + scheduler), vLLM 0.30.0 | AMD EPYC 7663, 224 vCPU (580.95.05; 600 W, 3210 MHz) | 2.9558 |
| B | `sc1b-5090-3` (`8b4e2c9`) | `9dd712b` (A10) | e4b RTN anchors, vLLM, llama.cpp `552f18f`, ExLlamaV3 1.5.3 cu128, LMDeploy 0.18.0 (UNSUPPORTED) | AMD EPYC 7663, 224 vCPU (580.95.05; 600 W, 3210 MHz) | 0.9034 |
| C | `sc1c-5090-5` (`94a8f5e`) | `9dd712b` (A10) | e4b RTN anchors, vLLM, SGLang 0.5.20, ExLlamaV3 1.5.3 cu132 (labelled native row) | Intel Xeon E5-2698 v4, 40 vCPU (595.71.05; 575 W, 3090 MHz) | 1.2626 |
| A, third draw | `sc1a-5090-4` (`c4915d3`) | `c19dd52` (A13) | as box A; the int4 prefill route pinned to `loop` | AMD Ryzen 9 7950X, 32 vCPU (580.159.03; 500 W, 3105 MHz) | 1.9335 |

Every box used an RTX 5090 32 GB, the checkpoint Qwen/Qwen3-30B-A3B (GPTQ-Int4 for vLLM, SGLang and
LMDeploy, Q4_K_M and IQ4_XS GGUF, EXL3 4.0 bpw), the same token ids (`a8e6ea1`, `cd70a14`), the same K8 windows
(`9ef10d760ad9`, `4bcb55179b96`) and the bf16 oracle measured on box B. Files: `RESULTS-sc1-cross-box.md` (P13 and the
licence), and `receipts/<run>/RESULTS-sc1-<box>.md` plus `verdict.json` per box, with every arm receipt, log and forensic
file beside them. The launcher's own receipts stay in the private receipt store, at the commits named in the table.

**Box A's third draw** (`sc1a-5090-4`, amendment A13) was run after the first read. The owner approved it to read the
predictions box A's second draw left unread. It ran at A13's merge with every arm VALID and none skipped (3 h 37 min,
inside a 6.0 h guard).
- A13 pins e4b's int4 prefill route to `loop`, the route every other SC1 row ran, because #937 made `auto` (k19) the
  default afterwards.
- #918's fix of the row-index cache (#913) is in this draw and in no other. SC1's scheduler rows are speed only, and the
  fix changes which rows are gathered, not the work.
- A13 registered a 7.5 h guard. The launcher's compute policy caps a run at 6.0 h, so `sc1a-5090-3` was refused at $0
  and this draw ran at 6.0 h.

`RESULTS-sc1-cross-box-a3.md` is the three-box read with this draw as box A. `RESULTS-sc1-cross-box.md` is the first
read, with `sc1a-5090-2` as box A. Re-reducing boxes B and C against the third draw changes only the substitution-licence
figure printed beside their rows: +0.2 % / +0.2 % instead of −0.1 % / +0.1 %. e4b's quality rows are bit-identical
across the two box A draws, so their `RESULTS` files are left as first read.

## The outcome by the registered rule: no position on any box

Box A's licence reads **QUALITY_FAIL** on all three draws (`sc1a-5090-1`, `sc1a-5090-2`, `sc1a-5090-4`), with the same
NLLs to every digit. The K8 gate is calibrated, budget
0.05, with calibration domain c4val1. The licensed pack against its NF4 base:

| arithmetic | wikitext Δppl | c4val1 Δppl | verdict |
|---|---|---|---|
| `auto` (B=1) | −0.01438 | **+0.05601** | FAIL |
| `=1` (K19+K23, B=16) | −0.01067 | **+0.04030** | FAIL |

The pack improves wikitext and worsens the calibration text. The calibrated rule requires an improvement of the same sign on
at least two texts, one of them outside the calibration domain. The registration: "QUALITY_FAIL → … no position is quoted
from the RTN rows either." A11 item 2 carries that block to boxes B and C. **SC1 therefore quotes no position, and P1, P2,
P3, P4 and P5 read UNREAD.** Under the decision rules, nothing moves in the register. The P58 comparator
(`e4b.serve.h2h.vllm-0.30.0.p58.qwen3.{b1,b16}.5090.2026-09-22`) stands as measured.

Every ratio below is a **measurement, not a position**: both arms VALID and stable, labelled.

## What was measured: decode tok/s on the scheduler axis (comparator / e4b `int4_sched`, same box)

| engine (contract) | box | B=1 tok/s | ratio [interval] | B=16 tok/s | ratio [interval] | served quality vs e4b |
|---|---|---|---|---|---|---|
| e4b `int4_sched` (RTN, 4.50 bpw) | B | 222.1 | 1 | 1662.2 | 1 | anchor |
| e4b `int4_sched` | C | 209.6 | 1 | 1541.5 | 1 | anchor |
| vLLM 0.30.0 GPTQ Marlin + graphs (4.15 bpw) | B | 267.2 | **1.203** [1.191, 1.216] | 1948.2 | **1.172** [1.159, 1.185] | CLOSE ×2 (box A's rows) |
| vLLM | C | 244.5 | **1.167** [1.157, 1.176] | 1814.1 | **1.177** [1.172, 1.182] | (same) |
| vLLM, over e4b `lic_sched` | A (third draw) | 258.1 | **1.187** [1.183, 1.191] | 1913.5 | **1.167** [1.165, 1.170] | CLOSE ×2 |
| SGLang 0.5.20 GPTQ Marlin, matched | C | 265.9 | **1.268** [1.263, 1.274] | 1836.6 | **1.191** [1.187, 1.196] | CLOSE ×2 |
| SGLang native (labelled) | C | 265.6 | 1.267 [1.263, 1.272] | 1833.3 | 1.189 [1.186, 1.193] | CLOSE ×2 |
| llama.cpp Q4_K_M (4.86 bpw), np 1 / 16 | B | 329.6 | **1.484** [1.451, 1.518] | 1036.5 | **0.624** [0.618, 0.629] | CLOSE ×2 |
| llama.cpp IQ4_XS (4.29 bpw) | B | 325.5 | single draw | — | — | not scored |
| ExLlamaV3 4.0 bpw, cu128 | B | 199.7 | **0.899** [0.881, 0.932] | 108.6 / 112.8 / 104.9 | UNSTABLE (7.5 %) | COMPARABLE ×2 (better) |
| ExLlamaV3 4.0 bpw, cu132 (native row) | C | 117.7 | single draw | 65.7 | single draw | not scored on C |
| LMDeploy 0.18.0 TurboMind W4A16 | B | — | **UNSUPPORTED** | — | UNSUPPORTED | UNSUPPORTED |

Read as measurements:
- **vLLM, SGLang and llama.cpp at B=1 decode faster than e4b's scheduler.** Their served quality is CLOSE to e4b's
  served rows on both texts. At B=16, vLLM and SGLang lead by 17–19 %, and llama.cpp's 16-slot path runs at 0.62×.
- **SGLang over vLLM on box C:** 1.087 at B=1 and 1.012 at B=16, matched; native is the same to within 0.2 %. Both are inside
  P3's 10 % / 15 % band on the number. This is not a P3 reading, because P3 reads quoted positions.
- **llama.cpp Q4_K_M** sits at 1.484 at B=1, above P4's [0.8, 1.15], and 0.624 at B=16, inside [0.35, 0.7].
- **ExLlamaV3** sits at 0.899 at B=1, just under P5's [0.9, 1.4]. Its B=16 is unstable at about 109 tok/s, against e4b's
  window at 1677.

The registered route to naming the cause of each gap is SC1b's per-kernel census. That is out of scope here, and no
README sentence changes before it.

**Box A's own rows** (the licensed pack). The second draw ran before A10; the third ran at A13:

| row | second draw B=1 | B=16 | third draw B=1 | B=16 |
|---|---|---|---|---|
| e4b window `lic` | 237.0, 237.4 | 1686.1, 1685.7 | 231.2, 231.2 | 1648.4, 1648.5 |
| e4b window `rtn` | 237.3 | 1683.4 | 230.8, 230.7 | 1644.4, 1644.7 |
| e4b window `nf4_ctrl` | 104.1 | 499.4 | 100.4, 100.4 | 478.5, 480.6 |
| e4b scheduler `lic_sched` | 220.5, 310.2 (UNSTABLE) | 2390.4, 1552.1 (UNSTABLE) | 217.6, 217.3 | 1639.2, 1638.8 |
| vLLM `gptq_graph` | 267.0, 266.4 | 1942.5, 1944.6 | 257.4, 258.7 | 1909.1, 1917.8 |
| vLLM `gptq_fp8kv` | 284.2 (single) | 2037.9 (single) | 281.3, 275.5 | 2121.1, 2125.4 |
| vLLM `gptq_graph_nodetok` | skipped | skipped | 262.1 (single) | 1906.5 (single) |

- The window ratio vLLM / e4b-`lic` (the kernel ceiling) is 1.124 / 1.153 on the second draw and 1.116 / 1.161 on the
  third. P58 read 1.087 on the window.
- The second draw's scheduler arm read **UNSTABLE**: 40.7 % at B=1, 54.0 % at B=16. It ran the registered wall-slope
  estimator, which A10 item 7 replaced: each rep's wall carries e4b's ~2 s paged prefill, so a few percent of prefill
  jitter became 10–25 % of the 0.45 s difference the 32 → 128-token slope is read from. Its 310 and 2390 tok/s are faster
  than the window the scheduler wraps (237 and 1686), a sign of that artefact.
- The third draw ran A10's decode-only estimator and reads STABLE (0.1 % / 0.0 %), as boxes B and C do (1.8 % / 0.8 % and
  0.5 % / 0.4 %).

## Predictions (the reducer's mechanical reading; `verdict.json` per box)

| P | reading | evidence |
|---|---|---|
| P1, P2 | UNREAD | No quoted position: the licence is QUALITY_FAIL on all three box A draws. |
| P3 | UNREAD | No quoted position. Measured SGLang/vLLM on box C is 1.087 / 1.012. |
| P4, P5 | UNREAD | No quoted position. Measured values above. |
| P5b | **HOLD** on its stated alternative | LMDeploy UNSUPPORTED at both B (below). |
| P6 | **REFUTED** | Served − prefill NLL: e4b +0.0048 / +0.0025 nats on every box A draw (predicted > 0.005). vLLM −0.0017 / −0.0006 (second draw, inside 0.002) and +0.0007 / −0.0022 (third: c4val1 just outside 0.002). |
| P7 | **REFUTED** (third draw) | SAMEPROMPT faster than distinct: vLLM ×1.62 (holds, ≥ 1.3); e4b scheduler ×1.18 (predicted ≥ 1.3). |
| P8 | **REFUTED** (third draw) | fp8-KV / kv-auto ×1.079 at B=1 (predicted within 5 %), ×1.110 at B=16 (faster, holds). Its Δ_bf16 is larger by +0.0027 / +0.0043 nats (< 0.01, holds). |
| P9 | **REFUTED** | The pack does not license: QUALITY_FAIL at `auto` and at `=1`, on all three draws. |
| P10 | **REFUTED** | TTFT-4096 vLLM / e4b: 0.011 (A second draw), 0.026 (A third draw), 0.011 (B), 0.008 (C); band [0.5, 1.0]. |
| P11 | **REFUTED** (third draw) | J/token at B=1: vLLM 1.041 against e4b 1.386, ratio 0.751 (predicted within 15 %). The B=16 clause needs a quoted position and is unread. |
| P12 | **HOLD** | `lic` / `rtn`: −0.1 % / +0.1 % (second draw), +0.2 % / +0.2 % (third). |
| P13 | **HOLD** | B=1: A 1.187, B 1.203, C 1.167 (gap 3.1 %). B=16: A 1.167, B 1.172, C 1.177 (gap 0.8 %). Limit 5 %. |
| P14 | **HOLD** (third draw) | nodetok / matched 1.016 at B=1 (within 3 %), 0.996 at B=16 (within 1 %). The first draw read 0.968 at B=16 (A5). |

**P13 holds across all three boxes** once box A's third draw measures its own ratio. The first read held it on B and C
only, because box A's scheduler anchor was UNSTABLE. The three boxes are three hosts:

| box | CPU | board | driver |
|---|---|---|---|
| A | Ryzen 9 7950X | 500 W | 580.159 |
| B | EPYC 7663 | 600 W | 580.95 |
| C | Xeon E5-2698 v4 | 575 W | 595.71 |

The absolute rates move by 5–9 % between hosts (box C's vLLM runs 8.5 % slower than box B's at B=1), and the ratio holds
within 3.1 %. A11's reading of P13 on measured rather than quoted ratios stays flagged for review.

The B=16 clause of P11 is measured, though not read: vLLM uses 0.240 J/token against e4b's 0.350 and is the faster
engine, as the clause predicts.

## Quality (served shape, nats against the bf16 oracle; Δ_pair against e4b's served rows on the identical window)

| engine | Δ_bf16 wikitext / c4val1 | Δ_pair wikitext / c4val1 | band |
|---|---|---|---|
| e4b licensed pack (box A; carried to B and C under A11) | +0.0142 / +0.0241 | 0 | anchor |
| vLLM GPTQ (box A) | +0.0195 / +0.0296 | +0.0053 / +0.0055 | CLOSE |
| SGLang GPTQ (box C) | +0.0191 / +0.0274 | +0.0048 / +0.0033 | CLOSE |
| llama.cpp Q4_K_M (box B) | +0.0165 / +0.0169 | +0.0023 / −0.0072 | CLOSE |
| ExLlamaV3 4.0 bpw (box B) | −0.0010 / +0.0101 | −0.0152 / −0.0141 | COMPARABLE (closer to bf16 than e4b) |

- Box A's third draw re-reads vLLM: served +0.0194 / +0.0281, Δ_pair +0.0052 / +0.0040, CLOSE. Its fp8-KV served rows
  are +0.0222 / +0.0324, Δ_pair +0.0080 / +0.0082, CLOSE.
- vLLM's quality rows run on box A only. A11 carries e4b's and the oracle's rows across boxes, not vLLM's, so vLLM's
  positions on B and C print "quality UNREAD".
- **SGLang's prefill-shaped rows are HARNESS_ERROR,** and they are box C's only non-zero exit.
  - The scorer asserted that `input_token_logprobs[0]` is a bare None. SGLang 0.5.20 prepends None to the values and zips
    them with the ids, so entry 0 is `(None, ids[0], None)`.
  - The CPU fake made the same misreading, so the defect never surfaced on CPU.
  - A12 ([#933](https://github.com/pjordanandrsn/experts4bit-qlora/pull/933)) fixes the scorer and the fake.
  - These rows bear on no registered prediction: P6's pair is e4b's and vLLM's, and every position reads the served shape.
    So they are not re-run, and SGLang's served − prefill engagement reading is UNREAD.

## TTFT: e4b's paged prefill is the loss, and it depends on the host

Median of 3 warm, uncached, `max_tokens=1` requests:

| engine | box A 512 / 4096 | box A third draw | box B 512 / 4096 | box C 512 / 4096 |
|---|---|---|---|---|
| e4b scheduler (int4 store, `loop` route) | 2.010 / 16.386 s | **0.823 / 7.010 s** | 2.027 / 16.321 s | **3.492 / 27.615 s** |
| vLLM | 0.041 / 0.173 s | 0.030 / 0.182 s | 0.041 / 0.174 s | 0.066 / 0.212 s |
| SGLang | — | — | — | 0.098 / 0.213 s |
| llama.cpp Q4_K_M | — | — | 0.045 / 0.363 s | — |
| ExLlamaV3 | — | — | 0.333 / 0.779 s | 0.537 / 1.191 s |

- e4b's TTFT is 27–130× vLLM's, so P10 is REFUTED on every box.
- **The cause is P100's** ([#920](https://github.com/pjordanandrsn/experts4bit-qlora/pull/920)). At `max_seqs` 1, each
  prefill chunk runs the int4 store's host-grouped `loop` route: one `dequant_int4_ref` per routed expert per projection,
  8,390 per 512-token chunk, 73 % of the chunk.
- Its cost moves with the host:
  - 0.55 s at 512 tokens on a Ryzen 9 9950X3D (P100);
  - 0.82 s on box A's third draw (Ryzen 9 7950X);
  - 2.0 s on the EPYC 7663 boxes A and B;
  - 3.5 s on box C's Xeon E5-2698 v4.

  P102's VOID first draw ran on the same Xeon class and read the loop at 30.1 s for 4096 tokens; box C read 27.6 s. Decode
  barely moves across the same hosts: e4b's window is 237 / 231 / 235 / 231 tok/s at B=1 (box A second draw, box A third draw, B, C). The prefill loop is host-bound and the
  decode is not.
- **P102** ([#931](https://github.com/pjordanandrsn/experts4bit-qlora/pull/931)) read `DEFAULT=k19` for that route: TTFT-4096
  from 7.21 s to 1.37 s (5.25×) on its host, inside the K8 rule.
- Main's default is still `loop`, and SC1 measured `loop`. SC1's decision rules move no kernel default, so the flip is
  #916's. A cross-host figure, not a reading: P102's 1.37 s against SC1's vLLM at 0.17–0.21 s would still sit at 0.12–0.15,
  outside P10's band.

## Controls, energy, resources

- **Degraded control PASS** on both box A draws: `lic_degraded_b16` runs ×1.175 (second draw) and ×1.183 (third) slower
  than `lic_b16`, with K19 absent from its census and present in `lic_b16`'s. The pack's K19 lever is engaged and carries
  the speed.
- **Method pair** (window / sched − 1): A third draw +6.3 % / +0.6 %; B +6.0 % / +0.9 %; C +10.4 % / +1.8 %. The
  registered expectation is +3 to +15 % at B=1. Box A's second-draw −10.6 % is the wall-slope artefact above. The third
  draw runs #918's fix and the other boxes don't; at B=1 its method pair (+6.3 %) sits between theirs, and at
  B=16 it is 0.3–1.2 points under both.
- **Energy** (J/token, board power, trapezoid over the window):

  | engine | B=1 | B=16 |
  |---|---|---|
  | e4b scheduler (box A, second draw) | 1.740 | 0.391 |
  | e4b scheduler (box A, third draw) | 1.386 | 0.350 |
  | vLLM (box A, third draw) | 1.041 | 0.240 |
  | llama.cpp Q4_K_M (box B, whole arm) | 1.333 | — |
  | ExLlamaV3 (box B, whole arm) | 3.391 | — |

  The second draw skipped vLLM's windows. The third draw reads P11.
- **Resident VRAM:**

  | engine | reserved GB |
  |---|---|
  | vLLM | 27.6–28.6 (policy reservation 28.22) |
  | ExLlamaV3 | 15.3 at B=1, 18.1 at B=16 |

  e4b's scheduler and SGLang recorded no reading in this column.

## LMDeploy: UNSUPPORTED on sm_120 (A10 item 2)

LMDeploy 0.18.0's TurboMind W4A16 cannot run on an RTX 5090.
- Its host dispatch runs the SM80 GEMM kernels on sm_120, and each kernel body compiles only where
  `Kernel::Arch::is_compatible(__CUDA_ARCH__)` holds. `Sm80` is `Arch<800, 900>`, so at 1200 every 4-bit GEMM is an empty
  kernel (`arch.h:25/53`, `gemm_universal.h:174-178`).
- All 11 arms on `sc1b-5090-1` aborted with rc 134 on their first real request.
- Box B records the rows as UNSUPPORTED instead of installing LMDeploy. P5b reads HOLD on that stated alternative.
- Not tried: LMDeploy's PyTorch backend, or a source build that widens `Sm80`.

## Box C's ExLlamaV3 row is confounded, not a finding

The cu132 wheel on box C read 117.7 tok/s at B=1 and 65.7 at B=16, one draw each. Box B's cu128 wheel read about 200 and
about 109. The two differ in wheel, torch and host at once, and each single draw is unmeasured for stability. The row is
reported as labelled; no cause is attributed.

## Every SC1 rental (40 receipts in `receipts/experts4bit-qlora/2026-10-0{2,3}/`, $15.9047)

| run | status | $ | cause / amendment |
|---|---|---|---|
| `sc1a-5090-1` | HARNESS_ERROR (rc 1) | 2.5786 | Its scheduler arms failed on the doubled lever hook (A4). vLLM B=1 failed at engine initialisation, energy read no header, SAMEPROMPT read a sha mismatch (A5). Its licence read QUALITY_FAIL. |
| `sc1a-5090-2` | **OK** | 2.9558 | Box A's read. Its host-limited deadline skipped the third scheduler draws, the second fp8-KV draws, nodetok and vLLM energy. |
| `sc1b-5090-1` | HARNESS_ERROR (rc 134) | 1.0741 | llama.cpp `-lv`, LMDeploy sm_120, ExLlamaV3 version read, census rule (A10 items 1–4). |
| `sc1b-5090-2` | NOT_RUN | 0.0905 | Instance stuck `loading` after 600 s. |
| `sc1b-5090-3` | **OK** | 0.9034 | Box B's read. |
| `sc1c-5090-1` | HARNESS_ERROR (rc 22) | 2.1541 | Stalled after its first SGLang arm, and the box became unreachable, so the draw was lost (A10 items 5 and 8). The cause was never established because the logs were lost. On `sc1c-5090-5`, every bounded SGLang transition completed. |
| `sc1c-5090-2` | HARNESS_ERROR (rc 18) | 0.0264 | Refused: driver 575 < 580. |
| `sc1c-5090-3` | NOT_RUN | 0.0434 | Download bandwidth 10.1 MB/s < 40. |
| `sc1c-5090-4` | NOT_RUN | 0.0350 | HTTP 429 attaching the ssh key. |
| `sc1c-5090-5` | HARNESS_ERROR (rc 1) | 1.2626 | Box C's read. rc 1 is the two SGLang prefill scorings (A12). |
| `sc1a-5090-3` | REFUSED | 0.0000 | A13's 7.5 h guard exceeds the launcher policy's 6.0 h cap; no rental. |
| `sc1a-5090-4` | **OK** | 1.9335 | Box A's third draw (A13): every arm VALID, none skipped. |
| 28 proofs (`sc1{a,b,c}-prove-*`) | 8 OK, 10 HARNESS_ERROR, 9 NOT_RUN, 1 REFUSED | 2.8473 | A1–A3, A6–A9; `sc1a-prove-12` ($0.1146) proved A13. |

- Full-run receipts total **$13.0574** across 12 receipts, $2.18 over the $10.88 registered for three draws. The excess
  is $0.24 of re-runs the amendments registered plus the $1.93 box A redraw the owner approved. Every run stayed under its own guard: the largest was
  $2.96, against box A's $4.13 guard.
- Proving receipts total $2.8473. The lane total is **$15.9047** across 40 receipts.

## Stated limits

- One card class, one prompt distribution and one request shape.
- n = 2–3 draws, with in-process loops on both sides of every engine-level ratio.
- Three different hosts. The host-CPU share is recorded, not matched.
- Box A's second draw's scheduler reading used the estimator A10 replaced; the third draw's is the one P13 reads.

These are the reasons SC2 (request-level) and SC4 (a second card class) exist.
