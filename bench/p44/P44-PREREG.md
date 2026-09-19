# P44 — SERVING QUALITY INSTRUMENTS: what would license the measured-but-unlicensed levers on OLMoE, Gemma-4, gpt-oss and Mixtral (pre-registered 2026-09-19, before any box is rented)

Work item: adertha-agents#110 (throughput + training campaign). Lineage: bo7 (`e4b.serve.census.bo7.*`) measured every family's fastest arm but could license only Qwen3's and Granite's positions; the register's own reading of the gap (`finding_throughput_census_bo7`): *speed parity is measured everywhere, licensing lags — the next levers are quality instruments, not kernels.* This lane builds the instruments and runs the gates. It changes no kernel and no default.

## What is measured but unlicensed, and why (from the register)

| family | fastest measured arm | over NF4 (B=1 / B=16) | why unlicensed |
|---|---|---|---|
| OLMoE-1B-7B | `int4all` (RTN int4 experts + calibrated attention + folds) | ×2.07 / ×2.29 | one K8 text (wikitext) only; the two-text clause for calibrated packs (#386) never applied; c4val1 never scored |
| Gemma-4-26B-A4B | `int4_r1epi` | ×1.71 / ×1.70 | **no instrument**: K8 unreadable on this family (±0.1–0.27 "chaos band", `e4b.parity.gemma4.no-reference`, #359) |
| gpt-oss-20b | `store_r12` (native MXFP4 GEMV for single rows) | ×1.29 / — | **no instrument**: wikitext K8 *improves* under quantisation (OOD flattery, −0.13…−0.17 nats), so a pass there is not evidence |
| Mixtral-8x7B | `all` | ×2.60 / ×1.96 | K8 FAIL (+0.116 c4val1; calibrated stack FAIL +0.077 wikitext) |
| Granite-3.1-3B | `calibexp_r12epi` | ×2.16 / ×2.09 | K8 FAIL (c4val1 +0.387; RTN int4 experts +0.063 > 0.05) |

## Two lanes

### P44-a — OLMoE two-text K8, and the per-expert residual census for Granite / Mixtral (one RTX 5090)

The K8 machinery as bo5/bo6/bo7 ran it (`bench/hybrid-g9/step_decomp.py --ppl-source {wikitext,c4val1}`, `k8_bake.py`, the registered 8192-step window, `experts4bit_qlora.k8_gate.verdict` with the 0.05 ppl budget: two-sided for RTN, one-sided for calibrated packs and the negative side trusted only when the calibration text differs from the scoring text — unchanged).

Arms (each a K8 row on BOTH texts):
1. OLMoE `nf4` (control, the licensed position) — wikitext + c4val1.
2. OLMoE `int4all` (RTN) — wikitext + c4val1. **Licenses** iff |Δ| ≤ 0.05 ppl on both.
3. OLMoE `calibexp_all` — the streamed 64k sequential calibration that licensed Qwen3 (bo6c's recipe, `finding_sequential_calibration_closes_the_int4_expert_gap`), calibrated on wikitext-train, scored on wikitext + c4val1. **Licenses** iff Δ ≤ +0.05 on both (one-sided); a negative Δ on wikitext is not trusted (same-domain), a negative Δ on c4val1 is recorded as the OOD-flattery tell.
4. Granite `calibexp_r12epi` and Mixtral `all` — **not re-gated** (their FAIL rows stand; no re-registration out of dissatisfaction). Instead a **per-expert residual census**: for every expert, the GPTQ (or RTN) reconstruction error of its packed rows against the bf16 expert on the calibration activations (`||W_q X − W X||_F / ||W X||_F`, per expert, per layer), written as a row per expert. This is the data a per-expert NF4 fallback (a third method in the recorded gptq/rtn assignment, #530) would be built on; **no fallback is built or gated here**.

### P44-b — a KL-from-bf16 instrument for the families K8 cannot read (one 80 GB-class card)

The instrument exists and is validated (`bench/kl_fidelity.py`, K0 controls: self-KL 0.0, single-byte perturbation detected, analytic second-order agreement 1.0000; `bench/KL-FINDINGS.md`): `KL(P_ref ‖ P_test)` token-weighted over the full vocabulary, fp64, teacher-forced on the frozen 200-prompt set (`bench/kl_prompts.py`, sha256 `eaa7792260b3f10d…`; strata general 72 / technical 55 / code 55 / longctx 18, **reported per stratum**). What this lane adds is a driver (`bench/p44/kl_serve.py`) that scores e4b's SERVING stacks — the arms bo7 timed — against the family's **bf16 reference resident on the same card**, and a registered reading rule.

Arms (each a KL row per stratum against the bf16 reference):
- Gemma-4-26B-A4B-it: `nf4` (control), `r1epi` (the licensed position: exact arithmetic, so its KL must equal `nf4`'s to fp64 noise — an instrument control), `int4_r1epi` (the candidate), `calattn_r1epi`.
- gpt-oss-20b: the reference is the **dequant path of the same shipped MXFP4 bytes** (there is no bf16 original; `kl_paths.py`'s rule, stated on every row): `nf4_r12` (control), `store_r12` (the candidate).
- Mixtral-8x7B-Instruct (bf16 reference 93 GB does NOT fit an 80 GB card resident): **NOT_RUN by registration**; its K8 FAIL stands.

**Reading rule (registered):** a candidate arm **licenses** iff, on every stratum, `KL(bf16 ‖ candidate) ≤ 1.10 × KL(bf16 ‖ nf4 control)` and pooled `KL(bf16 ‖ candidate) − KL(bf16 ‖ nf4) ≤ 0.005` nats/token. The band is relative to the family's own NF4 (the licensed position), so it asks "does the lever cost more knowledge than NF4 already costs", not "is 4-bit free". KL is a fidelity instrument, not a downstream predictor (`KL-FINDINGS.md`): a licence from this rule is labelled `licensed_by: kl-vs-bf16` in the register, distinct from a K8 licence, and the site renders the label.

## Registered predictions (falsifiable)

- **P1** (OLMoE `int4all`): FAILS the two-text gate on c4val1 (|Δ| > 0.05) — RTN int4 experts failed it on Qwen3 (+0.063) and Granite (+0.063); OLMoE's 1B-active experts are in the class Granite's are.
- **P2** (OLMoE `calibexp_all`): PASSES both texts one-sided (the Qwen3 recipe transfers) — if so, OLMoE's licensed position moves from NF4 to the calibrated stack (bo7 measured ×2.07 / ×2.29 for `int4all`; the calibrated pack's speed is the same kernels — "calibration buys quality not speed").
- **P3** (Granite census): the residual distribution is heavy-tailed — ≤ 10 % of experts carry ≥ 50 % of the summed reconstruction error (the premise of a per-expert fallback). Refuted → a per-expert fallback cannot work and the lever is closed.
- **P4** (Gemma-4 `r1epi` control): KL equals `nf4`'s to < 1e-4 nats on every stratum (exact arithmetic).
- **P5** (Gemma-4 `int4_r1epi`): licenses under the reading rule (per-stratum ≤ 1.10× and pooled Δ ≤ 0.005).
- **P6** (gpt-oss `store_r12`): licenses — the MXFP4 store serves the released bytes exactly for single rows, so its KL against the dequant reference should be at the instrument's floor (< 1e-3 nats) — and the row is the first quality verdict for this family that is not OOD flattery.

## Decision rules

Every licence here is a register row with its receipt, `licensed_by` naming the instrument, and the position document moves only when a row licenses; a FAIL is a row. P3 refuted closes the per-expert-fallback lever (LATER item retired with a receipt). No gate, threshold or existing value moves; the KL reading rule is NEW and applies only to families with no K8 instrument.

## Budget and STOP rules

| run | class | $/h ceiling | guard | estimate |
|---|---|---|---|---|
| `p44-a-olmoe` | RTX 5090 | 0.65 | 4 h | $2.60 |
| `p44-b-kl80` | H100 NVL (≥ 80 GB) | 3.10 | 3 h | $9.30 |

Both under the $35 cap. STOP: a card not of the class (refused before install); egress < 20 MB/s (refused, host-limited); the K0 controls failing on the box (no KL row is produced; the instrument's own rule); a fetch alarm. A second host-limited draw on a lane → stop and report.

## Receipts

`receipts/experts4bit-qlora/<date>/<run_id>/`; the register rows under `e4b.serve.p44.*`; `bench/p44/RESULTS-p44.md` quoting only the rows. Harness: `bench/p44/kl_serve.py` (P44-b driver; PR after this pre-registration), the census script `bench/p44/expert_residuals.py`, box/controller runners in the K14 pattern. Amendments dated below, before the data they touch.

## Amendments

(none yet)
