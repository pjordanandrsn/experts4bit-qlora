# P47 — WHERE GEMMA-4'S 1.08 NATS COME FROM: the served stack, the NF4 experts, or e4b's Gemma-4 modelling (pre-registered 2026-09-19, before any box is rented)

Work item: adertha-agents#110 (throughput + training campaign); defect [#597](https://github.com/pjordanandrsn/experts4bit-qlora/issues/597). Lineage: P44-b (`bench/p44/P44-PREREG.md`, read 2026-09-19, `RESULTS-p44.md`): e4b's served Gemma-4-26B-A4B-it NF4 stack is **1.077 nats/token** from the bf16 checkpoint (top-1 0.638) under the scorer control (i) admits (prefill on both sides; the decode scorer is refused on this family at 0.279 nats reference self-KL), lever-independent (+0.05..+0.20 on top for the folds / int4 experts / calibrated attention), reproduced to four digits across runs 2, 3 and 5 — where the same instrument reads gpt-oss NF4 at 0.022 and OLMoE / Granite / Mixtral NF4 at 0.02–0.1. P43 T2b (`bench/p43/P43-PREREG.md`, read) measured the TRAINING-loader NF4 model per layer against the HF oracle: rms(hidden − oracle) grows 0.0044 → 0.42 by layer 24 on BOTH e4b paths (fused and reference, ratio within [0.9, 1.1]) — the two paths agree with each other and not with the checkpoint. Nothing in the register quotes a Gemma-4 serving position; this lane decides what the defect IS, not whether to quote one.

## Question

The 1.08 nats is a composition of three things this lane can separate on one card: **(a) the serving stack** (arena bake → placement solver → `enable_hybrid_tier` all-VRAM → hot-residency dispatch), **(b) NF4 quantisation of THIS family's experts** (block-64 NF4 over the fused `gate_up_proj` / `down_proj` stacks), **(c) e4b's Gemma-4 modelling** (the loader's tower / expert-fusion assumptions, dense parallel MLP, softcapping, layer types, router). Each arm removes one layer of the composition; the arms are ONE model family, ONE reference, ONE prompt set, ONE scorer.

## Instrument (unchanged) and the arms (new builders only)

`bench/p44/kl_serve.py` (P44-b's instrument: K0-gated, 200 committed prompts / 4 strata, full-vocabulary fp64 KL from the cached bf16 reference, `--scorer auto` = control (i) picks the scorer per family, one child process per arm, proof-of-execution census per row) on the new family `gemma4diag` in `bench/p44/serve_stack.py`: the SAME nf4 lever set (no int4 store, no fusion; `arm_env` is the nf4 control's for every arm) on five **builders** (`serve_stack.BUILDERS`, `build_arm_model`):

| arm | builder | what it holds |
|---|---|---|
| `served_nf4` | `served` — P44-b's `build_served_model` | the anchor: arena + placement + hybrid tier, all-VRAM (the 1.077) |
| `loader_nf4` | `loader` — `load_moe_4bit_streaming` alone, `quantize_layers=None` | the model a TRAINING step holds (T2b's fixture): NF4 experts in all 30 layers, no arena / tier / hot residency |
| `loader_bf16experts` | `loader_unquant` — `quantize_layers=set()` | e4b's Gemma-4 modelling with the checkpoint's bf16 experts (`_place_unquantized_experts`: fused-on-disk, left in place) |
| `loader_nf4_lo` | `loader_lo` — `quantize_layers = {0..14}` | NF4 experts in the first half of the layers, bf16 in the rest |
| `loader_nf4_hi` | `loader_hi` — `quantize_layers = {15..29}` | NF4 experts in the second half only |

Proof of execution per row: `verify_moe_4bit`'s quantised / unquantised stack counts must equal the builder's expectation (`kl_serve._builder_check`), else the row is refused — a half that did not quantise is not a half. `reference_pass.self_consistency` (control (i)) is recorded again (P5). The lane runs `p47_run.sh` (BOX; K0 first, pinned fetch, arena bake, `kl_serve --family gemma4diag --scorer auto`) under `p47_drive.sh` (CONTROLLER; staged hashes in `staged.sha256`).

## Registered predictions (falsifiable; reducer `p47_reduce.py`)

- **P1 (the anchor reproduces):** `served_nf4` reads in **[0.95, 1.25]** nats. Refuted → the P44-b number was not stable and this lane's other rows are read against ITS anchor only, disclosed.
- **P2 (the model, not the stack):** |`loader_nf4` − `served_nf4`| **< 0.05** nats — the training loader's NF4 model carries the same gap, so arena / placement / hybrid tier add nothing (T2b's 0.42-rms late-layer divergence on the loader model predicts this). **Registered alternative:** `loader_nf4` **< 0.15** while the anchor holds → the serving STACK is the defect (a Gemma-4-specific bake / dispatch fault); `MIXED` otherwise (both carry part; the split is reported).
- **P3 (modelling faithful; NF4 of these experts is the cost):** `loader_bf16experts` **< 0.05** nats — with bf16 experts e4b's Gemma-4 model matches the checkpoint, so the nat is NF4 on THIS family's experts specifically. **Registered alternative:** **≥ 0.50** → a modelling defect independent of quantisation (the tower / fusion / dense-MLP / softcap path). `PARTIAL` in between (reported, not read as either).
- **P4 (depth):** if one half carries **> 70 %** of (`lo` + `hi`) the cost is localised in depth; alternative: spread. Read only if the halves ADD (|lo + hi − `loader_nf4`| / `loader_nf4` ≤ 0.30); a non-additive pair is reported as interaction.
- **P5 (the reference's own inconsistency replicates):** control (i) reads **0.20–0.36** nats again (0.279 in runs 3 and 5) — the HF Gemma-4 decode-vs-prefill self-disagreement is a property of the checkpoint under transformers 5.16.1, not of a run.

## Decision rules (what each outcome registers next; nothing is fixed in this lane)

- P2 ∧ P3 hold → **NF4 of Gemma-4's experts is the defect.** Next lane: the per-expert residual census (P44-a's `expert_residuals.py`) on Gemma-4 + an NF4-vs-FP4-vs-int8 expert-format arm on the loader model, and an inspection of the fused `gate_up_proj` block layout on this family (a block that straddles the gate/up boundary, or an outlier channel scale, would produce exactly a lever-independent constant). Until it closes, Gemma-4 stays unquoted AND the training-parity failure (#558) is read as the same cause.
- P2 holds ∧ P3 refuted → **e4b's Gemma-4 modelling is wrong regardless of quantisation.** Next: T2b's per-layer sweep on the `loader_bf16experts` model (hidden-state rms vs the oracle, hooks on every layer) to name the module; the fix is a loader / modelling correction with its own parity gate.
- P2 refuted (stack) → the arena bake or hybrid-tier dispatch is at fault ON THIS FAMILY; next: a bake round-trip check (dequantise the arena's stacks against the loader model's) before any dispatch inspection.
- P4 holding narrows either of the above to a layer region; P4 refuted says per-layer.

## Budget and STOP rules

| run | class | $/h ceiling | guard | estimate |
|---|---|---|---|---|
| `p47-gemma4diag` | H100 NVL (≥ 80 GB) | 3.10 | 2 h | $6.20 |

Under the $35 cap. Timing basis (run 5): fetch 7.5 min, K0 < 2 min, bake ~6 min, reference prefill pass ~4 min, each served arm ~6 min; the loader arms load without a bake (~3–5 min each); the bf16-experts arm holds ~52 GB (fits the 94 GB card with the reference freed). STOP: a box not of the class (refused, exit 15); egress below 20 MB/s (14); K0 failing on the box (16 — no KL row by the instrument's rule); a fetch alarm → `not_run`; a refused builder row is an honest hole, the other arms still run; a second host-limited draw → stop and report.

## Receipts

`receipts/experts4bit-qlora/<date>/p47-gemma4diag/p47/` — `gemma4diag_kl.json` (five rows or their `not_measured` holes, `reference_pass.self_consistency`, `controls`), `k0.json`, `versions.txt`, `forensics.txt`, `logs/`; read by `bench/p47/p47_reduce.py` (P1–P5 exactly as above; a missing row is NOT_READ). Results → `RESULTS-p47.md`; #597 updated; the register gains no row (a diagnostic licenses nothing) unless a reading later supports one.

## Staged files

`bench/p44/serve_stack.py` and `bench/p44/kl_serve.py` are staged by BOTH lanes (P44-b's `staged-b.sha256` and this lane's `staged.sha256`); both pins are re-pinned in the PR that changes them and both drivers dry-run before a manifest is written (the P44-b run-4 rule).

## Amendments

(none yet)
