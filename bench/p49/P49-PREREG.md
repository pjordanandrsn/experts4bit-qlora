# P49 — WHICH STORE SURVIVES LAYER 0, AND WHY LAYER 0: expert-format arms on Gemma-4's first layer, and the activation probe (pre-registered 2026-09-19 ~11:00Z, before any box is rented)

Work item: adertha-agents#110; defect [#597](https://github.com/pjordanandrsn/experts4bit-qlora/issues/597). Lineage: P48 (`bench/p48/RESULTS-p48.md`): NF4 experts in layer 0 ALONE put the model 0.892 nats from the bf16 checkpoint (all 30 layers: 1.084); the profile is monotone in depth (0.10 at layer 14, 0.0056 at layer 27); e4b's Gemma-4 modelling is faithful (P3); and two weight-space probes are flat — block-64 NF4 does the same ~9 % relative damage (per expert and per block) in every layer. So the question is no longer WHERE but (a) **which store survives the early layers** and (b) **what turns a uniform 9 % weight error into 0.9 nats in layer 0 and 0.006 in layer 27.**

## Instrument and arms

Two instruments on one H100 NVL, one lane (`bench/p47/p47_run.sh` with `P47_FAMILY=gemma4fmt P47_ACT_PROBE=1`):

1. **KL vs the bf16 checkpoint** (`bench/p44/kl_serve.py`, prefill scorer by control (i), the same 200 prompts) on family **`gemma4fmt`** (`bench/p44/serve_stack.py`), every arm the nf4 lever set with ONE quantised layer, the builder naming the store `loader_only_<i>:<quant_type>[:b<blocksize>]`:

| arm | layer | store | role |
|---|---|---|---|
| `L00_nf4` | 0 | NF4, block 64 | the P48 anchor (0.892) |
| `L00_fp4` | 0 | FP4, block 64 | the other 4-bit codebook |
| `L00_int8` | 0 | int8, blockwise 64 | the 8-bit store the loader already ships |
| `L00_fp8` | 0 | fp8 (e4m3), blockwise 64 | the other 8-bit store |
| `L00_nf4b32` | 0 | NF4, block 32 | does a smaller block rescue it? (the loader gains a `blocksize` knob for this row) |
| `L07_nf4` `L14_nf4` `L21_nf4` `L27_nf4` | 7 / 14 / 21 / 27 | NF4, block 64 | the depth arms for the activation probe (their KL re-read from P48 as a cross-check) |

Proof of execution per row: `verify_moe_4bit` must report exactly 1 quantised / 29 bf16 stacks AND the quantised stack's `quant_type` and `blocksize` must equal the builder's (`kl_serve._builder_check`, extended), else the row is refused.

2. **The activation probe** (`bench/p49/act_probe.py`, P5): on the first 8 prompts (2 per stratum) the HF bf16 reference runs with hooks on every text layer, saving per layer the MoE branch's raw output `experts(hidden_states_2)` and its post-norm `post_feedforward_layernorm_2(raw)` (Gemma-4 normalises the MoE branch BEFORE adding it to the residual — `modeling_gemma4.Gemma4TextDecoderLayer.forward`), plus per-layer rms of the residual input, the raw MoE output, its post-norm and the dense branch. Then each depth arm is built and its quantised layer's raw / post-norm outputs are captured on the same tokens (layers below are bf16 in both, so the layer's inputs are identical) and compared: `rel_err_raw`, `rel_err_norm2`, `amplification = rel_err_norm2 / rel_err_raw`.

## Registered predictions (falsifiable; reducer `bench/p49/p49_reduce.py`)

- **P0 (anchor):** `L00_nf4` ∈ [0.75, 1.05] (P48 read 0.892).
- **P1 (int8 survives):** `L00_int8` **≤ 0.02** nats — an 8-bit blockwise store on layer 0 is at the other families' NF4 floor; the remedy candidate is "8-bit experts in the first k layers". **Registered alternative:** ≥ 0.20 → even 8-bit does not rescue layer 0; the early layers stay high-precision (bf16). In between: INCONCLUSIVE, reported.
- **P2 (fp8):** `L00_fp8` ≤ 0.05.
- **P3 (fp4 is no better):** `L00_fp4` ≥ 0.50 (the 4-bit codebook is not the variable).
- **P4 (block size does not rescue):** `L00_nf4b32` ≥ 0.55 × `L00_nf4` (the weights are uniform, so halving the block cannot buy much). Refuted → smaller blocks are a lever on this family after all.
- **P5 (post-norm amplification):** across the NF4 depth arms, `rel_err_raw` stays within **2×** of each other (the weights are the same everywhere) while `rel_err_norm2` tracks the KL: **Spearman(KL, rel_err_norm2) ≥ 0.8** and `rel_err_norm2(L0) / rel_err_norm2(L27)` **≥ 5**. **Registered alternative:** `rel_err_raw` itself spreads > 2× and tracks the KL → the damage is already in the raw expert sum (cancellation among the top-k weighted expert outputs makes the SUM small relative to its terms in early layers), not in the post-norm. Either is activation-side; they name different remedies (per-layer output scaling vs router/top-k-aware calibration).

## Decision rules

- P1 holds → e4b's remedy is a **per-layer store map** (`quantize_layers` today is a set; the loader gains "these layers at int8, the rest NF4") for Gemma-4's first k layers, k from the P48 profile (layers 0–14 carry 96 %; the memory cost of int8 there is 2× NF4, not 4×); it ships behind its own K8 + KL gate and only then does a Gemma-4 position become quotable.
- P1 refuted → the first k layers stay bf16 (`quantize_layers` = all but 0..k−1); same gate; the VRAM cost (~1.5 GB per layer) is stated beside any position.
- P5 holds → the write-up names the post-norm; P5 alternative → names the expert-sum cancellation; either goes to #597 and to the Gemma-4 modelling notes so no future lever is measured without it.
- Nothing here changes a default or the register; Gemma-4 stays unquoted.

## Budget and STOP rules

| run | class | $/h ceiling | guard | estimate |
|---|---|---|---|---|
| `p49-gemma4fmt` | H100 NVL (≥ 80 GB) | 3.10 | 1.5 h | $4.65 |

Timing basis (P48-3): fetch ~7 min, K0 < 2 min, reference pass 40 s, nine one-layer arms ≈ 13 min; the probe adds a hooked reference pass (~2 min) and one model build per depth arm (~5 × 80 s). ≈ 35 min. STOP: not the class (15); egress < 20 MB/s (14); free disk < 120 GB (13); K0 failing (16); a fetch alarm → not_run; a refused store row (wrong quant_type/blocksize) is an honest hole; the probe failing leaves P5 NOT_READ and the KL rows stand; a second host-limited draw → stop and report.

## Receipts

`receipts/experts4bit-qlora/<date>/p49-gemma4fmt/p47/` — `gemma4fmt_kl.json`, `gemma4fmt_act.json`, `k0.json`, `versions.txt`, `forensics.txt`, `logs/`; read by `bench/p49/p49_reduce.py`. Results → `RESULTS-p49.md`; #597 updated.

## Staged files

`bench/p44/serve_stack.py`, `bench/p44/kl_serve.py`, `bench/p47/p47_run.sh`, `bench/p49/act_probe.py` are staged; `bench/p44/staged-b.sha256` and `bench/p47/staged.sha256` are re-pinned in the PR that adds them and both drivers dry-run before the manifest is written.

## Amendments

(none yet)
