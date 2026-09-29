# P81 — does bucketed CUDA-graph decode beat the eager `PagedModelRunner` on the licensed int4 stack, Qwen3-30B-A3B, one RTX 5090? And the licensed attention pack, by fingerprint (registered 2026-09-28, before any run)

Issues: experts4bit-qlora#511 and #674. P80 (`bench/p80/RESULTS-p80.md`) confirmed the graph claim on **NF4 only**, and
its "cannot say" section names licensed int4 serving as the open case. #674's mechanism shipped in #754 (a hash-pinned
attention artifact, `int4_b32.attn.v1`), and its remaining step is a licensed build that dumps one. This lane does both
in one rental, because the build is the same process either way. Rule: the owner's standing no-ask tier for a single run
under $15 (2026-09-26), with the usual mechanics: this page merged before the launch, a proving rental before a guard
over 1 h, receipts and ledger rows, proven teardown. The authorization permalink is posted on #511 before the launch.

## Questions

1. **(#511, decided.)** P80's claim on the configuration actually quoted for serving: *on the p37 Qwen3 prompts under
   the registered changing-active-set trace, CUDA-graph bucket replay exceeds its eager `PagedModelRunner` aggregate
   output-token rate by more than the 1.03 self-pair band, and decodes the identical token stream as its padded eager
   oracle*, with the **licensed int4 stack**: GPTQ-calibrated int4 experts, calibrated int4 attention, the round-1/2
   T=1 folds and the fused router epilogue.
2. **(#674, reported.)** The licensed build's attention pack: its fingerprint, and whether it installs by fingerprint in
   every arm. The lane records whether the build's expert pack equals the licensed one (`sha256:0c9955a9…`, P55x) and
   whether the build's wikitext K8 reads P55x/P64/P70's **6.36709**. These are facts, not a verdict. A pack that does
   not install voids question 1 (below), because the arms would not be serving the same bytes.

## Instrument

- **Model and weights.** `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`, baked to an NF4 arena by
  P39's `k8_bake.py`, as in P70.
- **The build (one process, P70's).** `step_decomp` with P70's recipe: streamed GPTQ-calibrated int4 experts (C4, 128
  sequences, 10 layers per pass), calibrated int4 attention, `E4B_FUSE_T1_GLUE`, `E4B_FUSE_T1_GLUE_R2` and
  `E4B_FUSE_ROUTER_EPI`. It reads wikitext K8 over 2,048 steps at B = 1, eager, `--no-fuse-qkv`. It dumps **both**
  packs: the experts to `artifact1` (`E4B_INT4_DUMP_ARTIFACT_DIR`, #405) and the attention to `attn1`
  (`E4B_SERVE_ATTN_INT4_DUMP`, #754), each after its calibrated enable and before any fusion. Both must pass
  `verify_artifact` against the pinned revision, or the run stops (rc 20) with no arm.
- **Hook v7** (`bench/p81/hook/usercustomize.py`): P42's hook, with its calibrated-attention block routed through
  `int4_attn_calib.enable_from_env`. That is the only entry point that honours `E4B_SERVE_ATTN_INT4_ARTIFACT` /
  `_FINGERPRINT` / `_DUMP`.
- **Harness.** `bench/p81/step_decomp.py` is P80's lane copy plus two additions, pinned by
  `tests/test_p81_staged_pin.py`:
  1. The P81 stage merges the model's pack provenance into every arm receipt (`merge_provenance_into_receipt`), so each
     receipt names both fingerprints and how each pack was installed.
  2. `--dynb-grouping device` (see departure (i)).
- **The trace, the arms, the timing.** Exactly P80's: `_dynb_plan(16, 32)`, so 16, 8, 4, 2, 1 rows for 32 steps each,
  160 decode-only steps and 992 decode tokens, with `TRACE MISMATCH` refusing any other. Arms run A1 eager, B1 graph,
  B2 graph, A2 eager, P padded, one process and one receipt each. Pass 0 warms and pass 1 is timed. Buckets are
  (1, 2, 4, 8, 16), captured on scratch slots only.
- **Every arm installs both packs from their artifacts by fingerprint** (`E4B_INT4_ARTIFACT_DIR` +
  `E4B_INT4_EXPECTED_FINGERPRINT`; `E4B_SERVE_ATTN_INT4_ARTIFACT` + `E4B_SERVE_ATTN_INT4_FINGERPRINT`). No arm
  calibrates, and a mismatch refuses rather than rebuilding. The folds and the router epilogue are on in every arm.
- **Verdict.** `p81_reduce.py` computes it from the five receipts. Its self-test pins the rule on 13 synthetic cases,
  and it runs on the box and in CI.

## Departures from P80, and why

- **(i) The eager control runs the device grouping too.** P80's A ran the engine's default T > 1 grouping (off). With
  the int4 expert store, grouping off sends a T > 1 decode to `hot_residency._fused_over_stack`'s **prefill / verify**
  branch: dequantise each routed expert and matmul, with a host loop over groups. That is not how the licensed stack
  serves a batch. The register's `e4b.serve.b16.qwen3-30b.int4.5090` row was measured with the grouping on. An A on the
  prefill branch would be a strawman control. So every arm passes `--dynb-grouping device`, and A is eager, unpadded,
  grouping on: the int4 split-K GEMV at ≤ 256 routed rows. The receipt records `device_grouping`. An arm without it
  voids the read.
- **(ii) A pack gate.** All five receipts must carry the same `pack_fingerprint` and `attn_pack_fingerprint`, with
  `attn_pack_source == "artifact"`. Otherwise the arms did not serve the same function, and the read is VOID.
- **Unchanged from P80:** the identity gate is B ≡ P, not B ≡ A. A-vs-B is reported only. It may well hold here: the
  int4 GEMVs quantise and reduce each row independently. But the attention and the bf16 remainder still see different
  row counts (finding_bf16_gemm_varies_with_row_count).

## Predictions (written before the data)

- **P1 (the claim).** B1/A1 > 1.03 **and** B2/A2 > 1.03 in aggregate decode tok/s.
- **P2 (correctness).** B1, B2 and P decode identical token streams, bitwise, in every row.
- **P3 (engagement).** Every bucket captures (`graph_status` = `graph` for 1, 2, 4, 8, 16), and no candidate step runs
  eagerly.
- **Stated expectation, not the rule.** B/A is at least P80's ×2.32. Two reasons:
  - The int4 stack issues more launches per step than NF4 (an activation quantise before every int4 GEMV, in the
    experts and the attention).
  - Its GEMMs are shorter, so the host share of an eager step is larger.
  - The licensed stack has been captured before at B = 1 and B = 16 (the P54/P57 `--b1d-loop graph` arms), with the
    dense cache. Buckets 2, 4 and 8 have not, and a capture failure there would refute, as the memo registered.
- **#674, expected, not decided.** The expert pack equals `0c9955a9…` (it reproduced byte-for-byte across two boxes and
  two releases in P55x and again in P70), and the build's K8 reads 6.36709. If both hold, this build calibrated the
  licensed stack, and its attention pack is the first fingerprint of the licensed attention. Cross-box reproduction of
  that pack is a second build on another box: not this lane.

## Decision rule (`p81_reduce.py`)

1. **VOID** (NOT_RUN; neither confirms nor refutes):
   - A missing arm receipt (not an OOM).
   - An arm that ran the wrong mode, a trace other than the registered one, or no device grouping.
   - Arms that did not all serve the same two packs from their artifacts.
   - A control arm carrying graph state.
   - A1 ≠ A2 tokens.
   - A self-pair A2/A1 or B2/B1 outside [1/1.03, 1.03].
2. **REFUTED**:
   - A candidate arm OOMs.
   - Any bucket falls back to eager, or any B step runs eagerly.
   - B1, B2 and P do not decode identical tokens.
   - With everything valid, B1/A1 ≤ 1.03 or B2/A2 ≤ 1.03.
3. **CONFIRMED**: everything valid and B1/A1 > 1.03 and B2/A2 > 1.03.

`--licensed-fp` records whether the served expert pack is the licensed one (`expert_pack_is_licensed`). It is reported,
not decisive. A build whose pack differs still serves one consistent stack to all five arms, and the read says which.

## Box and cost

Two rentals, in order. The guard exceeds one hour, so the compute rule requires a proving rental first.

1. **`p81-prove-<n>`**: one RTX 5090, **0.2 h guard at ≤ $0.75/h (≤ $0.15)**. `P81_PROVE=1` runs the refusals, the
   install at the pins, the tripwire (the #754 dump/load entry points, the #405 knobs, the hook importable as
   `usercustomize`), the reducer self-test and an egress probe, with no model. At most three attempts (≤ $0.45).
2. **`p81-5090-<n>`**: only after a proof returns rc 0 with its receipts fetched.
   - One RTX 5090, Vast verified/secure, image `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`, ≥ 200 GB of disk (P70's
     floor).
   - grouped-nf4-gemm at **`fb15cf5f5b9f2fb107fa1a39218fd9d910b34d94`** (v0.33.5, P80's); e4b at the launch commit (a
     `main` that contains this page).
   - **Guard 2.5 h at ≤ $0.75/h (≤ $1.875).**
   - Expected about 75 min: a ~15 min fetch, a ~1 min bake, a ~30 min build (P70's took 29), then five arms of a model
     load, two pack installs and two ~10 s passes each.

**Lane ceiling $2.50; hard stop $3.50**; both under the $35 per-run cap.

## Rehearsal

The home A2000 (sm_86) cannot run the fp8 paged KV (native e4m3 needs sm_89+), so no arm can be rehearsed there. The
rehearsal is the runner's `P81_PROVE=1` path in a throwaway A2000 container (with the card class lifted in a local
copy), the driver's dry run, and the CI tests: the pins, the reducer's rule, the lane copy's diff against P80's, both
packs in the arms' load env, and the trace. It is not a reading.

## What this lane cannot say

- **Families and prefill.** Nothing about any family but Qwen3, or about prefill.
- **Arrivals.** Nothing about a server loop with arrivals: the trace admits nothing after the first step.
- **Across boxes.** Nothing about the attention pack reproducing on another box. It shows one build's pack loading by
  fingerprint on the box that built it.
- **Quality.** Nothing about quality beyond the build's single K8 reading. The arms are not scored.
- **Absolute throughput.** It holds only for one host. So no absolute tok/s is compared to another box's, and the graph
  gain is not compared numerically to P80's except as the stated expectation.
- **The pack bytes are not kept.** Both packs' payloads stay on the box. Their manifests (every payload's sha256 and
  the root fingerprint) travel with the receipts. The claim is the fingerprint and the recipe that produced it.

## Receipts

`receipts/experts4bit-qlora/<date>/p81-{prove,5090}-<n>/` in the adertha receipt store:

- The launcher's `receipt.json` and `teardown-proof.json`.
- The fetched `p81/`: `p81_{A1,B1,B2,A2,P}.json`, `verdict.json`, `packs.json`, `build_ppl_wikitext.json`,
  `artifact1/manifest.json`, `attn1/manifest.json`, `summary.txt`, `versions.txt`, `forensics.txt`, `logs/` and
  `work/bake.json`.

The read lands as `bench/p81/RESULTS-p81.md` with the small receipts.
