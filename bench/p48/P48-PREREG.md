# P48 — WHICH LAYERS: one NF4 layer at a time on Gemma-4-26B-A4B-it, and whether the weights predict it (pre-registered 2026-09-19 ~09:00Z, before any box is rented)

Work item: adertha-agents#110; defect [#597](https://github.com/pjordanandrsn/experts4bit-qlora/issues/597). Lineage: P47 (`bench/p47/RESULTS-p47.md`, run `p47-gemma4diag`): the served stack is innocent (training loader 1.084 vs served 1.077 nats from the bf16 checkpoint), and the cost sits in depth — NF4 experts in layers 0–14 alone read **1.058**, in layers 15–29 alone **0.133**, the halves additive. P47's bf16-experts arm was refused by the loader (amendment 1), so e4b's Gemma-4 modelling cost is not yet bounded.

## Registered input: the weight probe (free, CPU, LAN copy of the checkpoint — read BEFORE this prereg was written)

`bench/p48/g4_expert_probe.py` streamed every expert tensor of `/share/models/gemma-4-26B-A4B-it` (the pinned checkpoint; 30 layers × {`gate_up_proj` [128, 1408, 2816], `down_proj` [128, 2816, 704]}, all BF16, zero non-finite values) and computed, per layer and per expert: rms, absmax/rms, and the **block-64 NF4 round-trip relative error** (bitsandbytes' codebook, absmax scaling, the loader's block size). Result (`g4_expert_probe.jsonl`, committed): **the per-expert NF4 relative error is 0.0922–0.0931 in every layer and both projections** (median over experts; first-half mean 0.0927, second-half 0.0925), absmax/rms medians 7–11 (gate_up) and 10–18 (down), rising gently WITH depth — the opposite direction from the KL. **In weight space the early layers are not harder to quantise than the late ones.** So whatever makes NF4 in layers 0–14 cost a nat is not the weights' distribution; it is how those layers' quantisation error propagates (activation scale, residual sensitivity, routing), or how the loader handles those layers specifically. P4 below registers this as a prediction on the per-layer KL.

## Instrument (unchanged) and the arms

`bench/p44/kl_serve.py` on family **`gemma4layer`** (`bench/p44/serve_stack.py`): **30 arms `L00`…`L29`**, arm `L<i>` = builder `loader_only_<i>` = `load_moe_4bit_streaming` with `quantize_layers={i}` — NF4 experts in exactly ONE text-decoder layer, the checkpoint's bf16 experts everywhere else; the nf4 lever set (no int4 store, no fusion) for every arm; one child per arm; proof of execution per row: `verify_moe_4bit` must report exactly 1 quantised / 29 unquantised stacks or the row is refused (`_builder_check`). No served arm → no arena bake (`needs_arena`). Same prefill scorer chosen by control (i), same cached bf16 reference, same 200 prompts as P44-b/P47. The lane is P47's (`p47_run.sh` / `p47_drive.sh` with `P47_FAMILY=gemma4layer`).

## Registered predictions (falsifiable; reducer `bench/p48/p48_reduce.py`)

- **P1 (near-additivity):** the sum of the 30 single-layer rows is within **[0.5×, 1.5×]** of P47's all-layers loader row (1.084). Outside → the layers interact strongly and the per-layer profile is a ranking, not a decomposition (reported as such).
- **P2 (concentration):** the five largest single-layer rows carry **≥ 50 %** of the sum; **P2b:** layers 0–14 carry **≥ 80 %** (P47's split, per layer). Refuted → the cost is spread across the early half (then the mechanism is a property of "early" rather than of particular layers).
- **P3 (the modelling bound):** the SMALLEST single-layer row is **≤ 0.02 nats** — e4b's Gemma-4 model with 29 of 30 expert stacks in bf16 matches the checkpoint to within the other families' NF4 floor. Refuted (≥ 0.02 everywhere) → a modelling component exists beside quantisation and is at least that large (the P3 that P47 could not read).
- **P4 (the weights do not predict it):** Spearman |ρ| between the per-layer KL and the probe's per-layer NF4 relative error is **< 0.5** (the probe is flat; the KL is not). Refuted → the tiny weight-space differences track the KL after all and the probe deserves a finer look (per-expert, per-block).

## Decision rules

- P2 ∧ P3 hold → the defect is **specific layers' expert quantisation**, and the next step is mechanistic on the top-5 layers: per-expert NF4-vs-bf16 output error on real activations (the P44-a census instrument with the KL prompts as its source), plus an FP4 / int8 expert-format arm on those layers only, to say whether ANY 4-bit format survives there or those layers need to stay bf16 (`quantize_layers` = all but top-k: an e4b-side per-family default, with its own K8/KL gate before any position is quoted).
- P2 refuted ∧ P3 holds → an "early layers are sensitive" story: register a depth-scaling arm (NF4 in 0–4, 0–9, 0–14) and read the growth curve.
- P3 refuted → the modelling component is real and bounded below by the smallest row; the T2b per-layer hidden-state sweep runs on the `L29`-style model to name the module.
- P4 refuted → re-probe per expert / per block before any of the above.
- Nothing here changes a default or the register: Gemma-4 stays unquoted.

## Budget and STOP rules

| run | class | $/h ceiling | guard | estimate |
|---|---|---|---|---|
| `p48-gemma4layer` | H100 NVL (≥ 80 GB) | 3.10 | 1.5 h | $4.65 |

Timing basis (P47): fetch 7.5 min, K0 < 2 min, reference pass 40 s, each loader arm ~80–85 s including load → 30 arms ≈ 42 min; ≈ 55 min total. STOP: a box not of the class (15); egress < 20 MB/s (14); K0 failing (16); a fetch alarm → not_run; a refused builder row is an honest hole (the others run); the deadline-derived arm alarm ends the sweep partial — P1/P2 need all 30 rows (NOT_READ otherwise), P3/P4 are reported on what was measured and marked partial; a second host-limited draw → stop and report.

## Receipts

`receipts/experts4bit-qlora/<date>/p48-gemma4layer/p47/` (the P47 lane directory name) — `gemma4layer_kl.json` (30 rows or holes), `k0.json`, `versions.txt`, `forensics.txt`, `logs/`; read by `python bench/p48/p48_reduce.py <dir> --probe bench/p48/g4_expert_probe.jsonl`. Results → `RESULTS-p48.md`; #597 updated.

## Staged files

`bench/p44/serve_stack.py` and `bench/p44/kl_serve.py` are staged by P44-b, P47 and this lane; `bench/p44/staged-b.sha256` and `bench/p47/staged.sha256` are re-pinned in the PR that adds `gemma4layer`, and both drivers dry-run before the manifest is written.

## Amendments

### Amendment 1 (2026-09-19 ~09:00Z, after run 1, before the redraw) — a disk floor on the box

Run `p48-gemma4layer` (H100 NVL, instance 51560608): K0 passed, then the Gemma-4 fetch died `ENOSPC` — the instance's container overlay was **32 GB** (P47's box had 494 GB, P44-b's 2.4 TB; the launcher orders ≥ 320 GB of MACHINE disk, which is not what the instance gets). No arm ran; $0.49; a host-limited draw, receipted as HARNESS_ERROR. The runner now refuses BEFORE any fetch when `/root` has under `P47_MIN_DISK_GB` (default 120) GB free — exit 13, a refusal row, the same class as the VRAM and egress floors — and P44-b's runner gets the same check. Predictions, rules and budget are unchanged; the redraw is `p48-gemma4layer-2` (same manifest, the new e4b head).
