# P82 — P81 re-measured on the fixed graph path, with the licensed build's fp32 router: does bucketed CUDA-graph decode beat the eager `PagedModelRunner` on the licensed int4 stack, decoding the eager function exactly? And does the router cast account for #674's K8 gap? (registered 2026-09-29, before any run)

Issues: experts4bit-qlora#511, #674 and #777. P81 (`bench/p81/RESULTS-p81.md`) read CONFIRMED on the licensed int4
stack, but on the path #777 later found wrong at bucket 1: its ratios stand **corrected, not superseded**, until the
stack is re-measured on the fixed path. B771b (`bench/b771b/RESULTS-b771b.md`) re-measured NF4 on the fixed path and
found the eager runner, the bucket step and the graph replay bit-identical. This lane does the same for the int4 stack.

It also isolates one difference for #674. P81's build read wikitext K8 **6.33015**, and P55x, P64 and P70 read
**6.36709**. The one arithmetic change known to sit on the recipe's decode path between those builds is 0.37.5's
router-epilogue default: `E4B_FUSE_ROUTER_EPI=1` now rounds `softmax_topk` routing weights to bf16 (P70's read).
- The rest of e4b 0.37.4 → 0.37.8 is off this path or default-off: 0.37.6's attention pack dump/load, 0.37.7's
  opt-in graphs, and 0.37.8's graph-only #777.
- grouped-nf4-gemm moved 0.33.0 (P70) → 0.33.5 (P81) → 0.33.7 (here). Its changelog records no arithmetic change on
  this path. 0.33.7's append is the graph path's; the eager K8 appends through `quantize_kv_fp8`.

So every process in this lane runs the licensed builds' fp32 router (`E4B_ROUTER_EPI_CAST=0`). Two K8 arms then read
the same packs with and without the cast, on one box.

Rule: the owner's standing no-ask tier for a single run under $15 (2026-09-26), with the usual mechanics: this page
merged before the launch, a proving rental before a guard over 1 h, receipts and ledger rows, proven teardown. The
authorization permalink is posted on #511 before the launch.

## Questions

1. **(#511 / #777, decided.)** P81's claim on the fixed path: *on the p37 Qwen3 prompts under the registered
   changing-active-set trace, CUDA-graph bucket replay exceeds its eager `PagedModelRunner` aggregate output-token rate
   by more than the 1.03 self-pair band*, on the licensed int4 stack (GPTQ-calibrated int4 experts, calibrated int4
   attention, the round-1/2 T = 1 folds, the fused router epilogue at fp32 weights), **and the eager runner, its bucket
   step and the graph replay decode one identical token stream**.
2. **(#674, reported.)** With the fp32 router, does the build read the licensed 6.36709? Does a process that loads
   both packs by fingerprint read the same? And does the same process with the cast read P81's 6.33015? These are
   facts, not a verdict. The attention pack's fingerprint is also compared with P81's (`d7cfa1f4…`), which gives a
   cross-box reproduction datum.

## Instrument

- **Model and weights.** `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`, baked to an NF4 arena by
  P39's `k8_bake.py`, as in P70 and P81.
- **The router, everywhere.** `E4B_ROUTER_EPI_CAST=0` is exported after the lane's unset list, so the build and every
  arm inherit it. Only K16 removes it. The tripwire refuses the box unless `router_epilogue.CAST_WEIGHTS[0] is False`
  in an importing process.
  - After each process, the runner stamps what `router_epilogue` reads under that process's env into a sidecar,
    `<receipt>.router.json`: the env value, `CAST_WEIGHTS`, and whether `softmax_topk` casts. The receipt itself stays
    as the harness wrote it.
  - The stamp reads what the arm read because the module takes `CAST_WEIGHTS` from the env at import, and neither the
    staged harness nor the hook sets either. `tests/test_p82_staged_pin.py` pins both facts.
- **The build (one process, P81's).** `step_decomp` with P70's recipe: streamed GPTQ-calibrated int4 experts (C4, 128
  sequences, 10 layers per pass), calibrated int4 attention, `E4B_FUSE_T1_GLUE`, `E4B_FUSE_T1_GLUE_R2` and
  `E4B_FUSE_ROUTER_EPI`. It reads wikitext K8 over 2,048 steps at B = 1, eager, `--no-fuse-qkv`, and dumps both packs
  (`artifact1`, `attn1`). Both must pass `verify_artifact`, or the run stops (rc 20) with no arm.
  - The packs cannot depend on the router knob. Calibration runs at load, before `fuse_router_epilogue`, and on
    sequences far above the fused path's 64-row ceiling. P70's P2 showed the cast touches no forward above 64 rows.
- **Harness and hook: P81's, referenced unchanged.** `bench/p81/step_decomp.py` and hook v7
  (`bench/p81/hook/usercustomize.py`) are staged by P82's driver and pinned by both lanes' tests. The dynb receipts
  therefore carry `"lane": "P81"` and print `P81_DYNB`. That names the harness copy, not the lane.
- **The trace, the arms, the timing.** Exactly P81's: `_dynb_plan(16, 32)`, so 16, 8, 4, 2, 1 rows for 32 steps each,
  160 decode-only steps and 992 decode tokens. Arms run A1 eager, B1 graph, B2 graph, A2 eager, P padded, one process
  and one receipt each. Every arm passes `--dynb-grouping device` and installs both packs from their artifacts by
  fingerprint.
- **The K8 arms (after the five, reported only).**
  - **K32** reads wikitext K8 through both packs loaded by fingerprint, fp32 router, with the build's command line
    otherwise.
  - **K16** is identical with `E4B_ROUTER_EPI_CAST` unset (0.37.5's default: `softmax_topk` casts).
  - Each is skipped, and says so, if less than 900 s of guard would remain.
  - The decisive reduction runs **before** them and again after, so a K arm cannot cost the verdict.
- **Pins.**
  - grouped-nf4-gemm **v0.33.7** at `9407d499a4d1e0fe8c22b050878a9f869b385b45` (#413's append).
  - e4b at the launch commit, a `main` that contains this page (0.37.8 or later: #777).
  - The tripwire refuses a cut without #777's bucket-1 routing (`_g_sel` in `paged_attention_forward`) or #413's
    append (`_e4m3_group` in `fp8_kv`).
- **Verdict.** `p82_reduce.py` computes it from the five receipts and their sidecars. Its self-test pins the rule and
  the K8 table on 21 synthetic cases, and it runs on the box and in CI.

## Departures from P81, and why

- **(i) The eager runner's identity to its bucket step is decisive (A1 ≡ P).** In P81, A ≠ P was "reported, not
  decisive", and it hid two bugs: grouped-nf4-gemm#413's append and #777's bucket-1 slot. With both fixed, B771b found
  A ≡ P ≡ B bitwise on NF4. The int4 stack shares everything that differed there, so a divergence now is a bug, not
  rounding (finding_an_oracle_sharing_the_code_cannot_see_its_bug). With all five arms on the device grouping, A1 is
  the same function as the bucket step at every row count: the trace pads nothing (bucket = active rows).
- **(ii) A router gate.** Every arm's sidecar must read `router_epi_cast_weights == false`. Otherwise the read is VOID.
- **(iii) The K8 arms.** They are new, and read after the verdict.
- **Unchanged from P81:** the pack gate, the device-grouping gate, the engagement gates, the self-pair band, and B ≡ P.

## Predictions (written before the data)

- **P1 (the claim).** B1/A1 > 1.03 **and** B2/A2 > 1.03 in aggregate decode tok/s.
- **P2 (correctness).** A1, P, B1 and B2 decode identical token streams, bitwise, in every row.
- **P3 (engagement).** Every bucket captures (`graph_status` = `graph` for 1, 2, 4, 8, 16), and no candidate step runs
  eagerly.
- **P4 (the router).** Every arm, the build and K32 ran the fp32 router weights; K16 ran the cast.
- **Stated expectation, not the rule.** B/A is of P81's order (13.95–14.29 on its EPYC 7K62). It is host-dependent,
  because the int4 eager step is host-bound. The one-row phase costs slightly more than P81's, because the fixed path
  attends over a context that keeps growing, where the buggy path's froze (B771b's direction).
- **#674, expected, not decided: "THE CAST IS THE GAP".**
  - The build and K32 read **6.36709** to five decimals, and K16 reads **6.33015**.
  - K32 equals the build's K8 exactly: the packs loaded by fingerprint serve the build's function.
  - The expert pack is `0c9955a9…` again. The attention pack is P81's `d7cfa1f4…`: the attention fingerprint covers
    only the identity keys and payloads, so #772's informative field does not move it.

## Decision rule (`p82_reduce.py`)

1. **VOID** (NOT_RUN; neither confirms nor refutes):
   - A missing arm receipt (not an OOM).
   - An arm that ran the wrong mode, a trace other than the registered one, or no device grouping.
   - Arms that did not all serve the same two packs from their artifacts.
   - An arm whose router stamp is not the fp32 weights (`router_epi_cast_weights != false`, missing included).
   - A control arm carrying graph state, or P capturing graphs.
   - A1 ≠ A2 tokens.
   - A self-pair A2/A1 or B2/B1 outside [1/1.03, 1.03].
2. **REFUTED**:
   - A candidate arm OOMs.
   - Any bucket falls back to eager, or any B step runs eagerly.
   - **A1, P, B1 and B2 do not decode identical tokens** (any of B1–P, B2–P, B1–B2, A1–P).
   - With everything valid, B1/A1 ≤ 1.03 or B2/A2 ≤ 1.03.
3. **CONFIRMED**: everything valid, the four streams identical, and B1/A1 > 1.03 and B2/A2 > 1.03.

A CONFIRMED read supersedes P81's register row (`e4b.serve.p81.qwen3.licensed-int4.dynb.graph-buckets.5090.2026-09-29`)
with a row that names the fp32 router. A REFUTED read on identity opens an issue before anything else is read.

**The #674 table (reported; `k8_reading`).** It is read to five decimals, on the registered window
(`9ef10d760ad9`), and only when the router stamps match: the build and K32 fp32, K16 casting. Otherwise it reads
INCOMPLETE.

| build, K32 | K16 | reading |
|---|---|---|
| 6.36709, 6.36709 | 6.33015 | **THE CAST IS THE GAP**: the fp32 router reads the licensed K8 and the cast reads P81's, on the same packs |
| 6.36709, 6.36709 | other | **THE FP32 ROUTER READS THE LICENSED K8**, but the cast does not read P81's here |
| either ≠ 6.36709 | any | **THE CAST IS NOT THE WHOLE GAP**; `cast_effect_K16_minus_K32` is its share |

Also reported: `K32_equals_build_exactly`, `cast_effect_K16_minus_K32`, and `attention_pack_equals_p81`.

**What follows from the #674 reading.**
- **If the cast is the gap:** the licensed reading 6.36709 belongs to the fp32 router, and the recipe reproduces it
  with `E4B_ROUTER_EPI_CAST=0`. #674's quality statement must name the router setting, and #782 (retiring `=0`) has
  its data.
  - The cast's −0.037 ppl is ln(6.33015 / 6.36709) = −0.0058 nats. That is under the family's 0.0095 K8 floor, as
    P70's INDISTINGUISHABLE read predicts. So the cast is not a quality regression, but it is a different function.
- **Otherwise:** the gap is somewhere else, and the next lane bisects the software on the build: e4b 0.37.4 → 0.37.8
  and grouped-nf4-gemm 0.33.0 → 0.33.7.

## Box and cost

Two rentals, in order. The guard exceeds one hour, so the compute rule requires a proving rental first.

1. **`p82-prove-<n>`**: one RTX 5090, **0.2 h guard at ≤ $0.75/h (≤ $0.15)**. `P82_PROVE=1` runs:
   - the refusals and the install at the pins;
   - the tripwire: #777, #413, the router at fp32, the #754 dump/load entry points, the #405 knobs, and the hook
     importable as `usercustomize`;
   - the reducer self-test and an egress probe.

   No model. At most three attempts (≤ $0.45).
2. **`p82-5090-<n>`**: only after a proof returns rc 0 with its receipts fetched.
   - One RTX 5090, Vast verified/secure, image `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`, ≥ 200 GB of disk.
   - **Guard 3.0 h at ≤ $0.75/h (≤ $2.25).**
   - P81 used 2.23 h of 2.5 on a slow host: its build took 97 min against P70's 29. This lane adds two K8 arms of about
     5–8 min each. The 1,500 s first-calibration-chunk watchdog still ends a host that cannot build (rc 30).

**Lane ceiling $2.75; hard stop $3.75**; both under the $35 per-run cap.

## Rehearsal

The home A2000 (sm_86) cannot run the fp8 paged KV (native e4m3 needs sm_89+), so no arm can be rehearsed there. The
rehearsal is:
- the runner's `P82_PROVE=1` path in a throwaway A2000 container, with the card class lifted in a local copy, plus two
  mutation arms: the router left unset, and a cut without #777's routing. Each must be refused by the tripwire.
- the driver's dry run;
- the CI tests: the pins, the reducer's rule and K8 table, the router export and the stamp's soundness, the tripwire
  markers, the order of reduction and K arms, and both packs in the arms' load env.

It is not a reading.

## What this lane cannot say

- **Families and prefill.** Nothing about any family but Qwen3, or about prefill.
- **Arrivals.** Nothing about a server loop with arrivals: the trace admits nothing after the first step.
- **Absolute throughput.** It holds only for one host. No absolute tok/s is compared to another box's.
- **Which attention P55x licensed.** P55x's calibrated attention was never dumped, so a K8 match is a functional
  match on one window, not a byte-identification.
- **Quality beyond K8.** Nothing beyond the three wikitext K8 readings. The dynb arms are not scored.
- **The cast on other kinds.** Nothing about `topk_softmax` routers (gpt-oss, GraniteMoe) or other families.
- **The pack bytes are not kept.** Both packs' payloads stay on the box. Their manifests travel.

## Receipts

`receipts/experts4bit-qlora/<date>/p82-{prove,5090}-<n>/` in the adertha receipt store:

- The launcher's `receipt.json` and `teardown-proof.json`.
- The fetched `p82/`: `p82_{A1,B1,B2,A2,P}.json`, `k8_{K32,K16}.json`, every `*.router.json`, `verdict.json`,
  `packs.json`, `build_ppl_wikitext.json`, both pack manifests, `summary.txt`, `versions.txt`, `forensics.txt`,
  `logs/` and `work/bake.json`.

The read lands as `bench/p82/RESULTS-p82.md` with the small receipts.
