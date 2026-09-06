## P40 — tp2: TRAINING-SIDE MIRROR OF THE SERVING BUILD-OUT — e4b vs Unsloth PER SUPPORTED FAMILY, SAME BOX, ONE FIXTURE (pre-registered 2026-09-05 21:45Z, before any box is rented)

**User directive (2026-09-05 21:40Z): "let's mirror this parity run for the training side re model support and parity, etc."** The serving side has a per-family census (bo7), a licence gate (bo6c) and a same-box competitor head-to-head (P37). The training side has the six-family internal parity matrix (tp1) and a competitor head-to-head on ONE family (P38, Qwen3). tp2 completes the mirror: **support + parity + resources per family, e4b vs Unsloth, one box, one fixture**, feeding `training_support` per path and the register exactly as tp1/P38 did. It is evidence build-out under the phase directive, not an optimisation campaign: no kernel, gate or threshold changes.

### Claim under test
For each supported MoE family, on the identical QLoRA problem: (a) does each framework train it in 4-bit at all (support: OK / REFUSED / OOM / install_failed / load_fault — a refusal is a row, never coerced), (b) is quality comparable at N steps (held-out loss), (c) what does each cost (s/step, peak VRAM, tok/s, J/step, adapter size). The headline per family is the **s/step ratio Unsloth/e4b** from the primary pair, quoted only when both arms are VALID and the quality reading is COMPARABLE; otherwise the family's row states why.

### Families (tp1's six, tp1's pinned revisions unless stated) and the e4b path tp1 licensed
| family | e4b primary arm (tp1 disposition) | Unsloth primary arm | notes |
|---|---|---|---|
| ibm-granite/granite-3.1-3b-a800m-instruct | `fused_attn4` (tp1 fused PASS ×5.86) | `ckpt_unsloth` | small; expect host-bound both |
| allenai/OLMoE-1B-7B-0924-Instruct | `fused_attn4` (tp1 fused PASS ×3.22) | `ckpt_unsloth` | |
| openai/gpt-oss-20b | **REFUSED on the fast path (tp1)** → secondary `attn_only` reference row only | `ckpt_unsloth` (whatever Unsloth does with MXFP4 weights is the row) | e4b's honest position = refused; no coercion through NF4/ExpertsLoRA |
| Qwen/Qwen3-30B-A3B @ ad44e777 | `fused_attn4` (P38 primary) | `ckpt_unsloth` (P38 primary) | repeat of P38's primary pair on this box = the cross-lane anchor (P38 read 1.413) |
| google/gemma-4-26b-a4b-it | `fused_attn4` (tp1 fused PASS ×2.37) | `ckpt_unsloth` | Unsloth Gemma-4 MoE 4-bit support unknown → the load is the probe |
| mistralai/Mixtral-8x7B-Instruct-v0.1 | `fused_attn4` with expert offload (tp1 fused PASS under offload) | `ckpt_unsloth` resident (Unsloth has no expert offload) | Unsloth predicted OOM at seq 512 on 32 GB — an OOM is a row |

### Fixture — identical for both frameworks per family (P38's, generalised)
Clinical instruction set, sha-pinned (`ds_manifest.json`); **tokenised once per family with that family's tokenizer at the pinned revision** → `tokens_<fam>.json` (sha in every receipt); seq 512, train 1,200 / held-out 48 rows; LoRA r 8, α 16, dropout 0, bias none, targets q/k/v/o/gate/up/down (+ the experts through each framework's MoE path: e4b ExpertsLoRA, Unsloth's `Enabling LoRA on MoE parameters`), AdamW lr 2e-4, batch 1 × accum 4, bf16 autocast, seed 0, N = 60 steps, eval every 20 on the held-out 48, gradient checkpointing on (e4b HF non-reentrant; Unsloth `use_gradient_checkpointing="unsloth"`, with `hf` as a secondary arm on the anchor family only); trainable-parameter count asserted equal across frameworks per family (P38: 321,257,472 for Qwen3). Adapters saved (dtype recorded — P38 found e4b saves bf16 expert LoRA vs Unsloth fp32; recorded, not corrected).

### Environments
e4b in the image python at the **shipped** cut: experts4bit-qlora 0.35.1 + grouped-nf4-gemm 0.30.2 from PyPI (what a user installs today; versions in every receipt). Unsloth in its own venv: `unsloth[cu128-torch280]` + `unsloth_zoo` at the latest release at launch, **no transformers/bnb/peft pins** (P38 amendment 1); if the tripwire fails on torchao, torchao is removed (P38 amendment 2); every version recorded. Both read the same cached snapshot; the e4b loader needs `refs/main` (P38 amendment 2 / e4b#404): fetch unpinned and ASSERT the staged sha equals the pin, else write `refs/main` from the pin only after proving `main == pin`. The Unsloth branch resolves the snapshot directory directly (P38 amendment 3). U8 evaluates the bnb4bit predicate on the innermost experts module (P38 amendment 4).

### Arms per family (each one process, one JSON `<fam>_<fw>_<arm>.json`, one alarm; order = family by family, small to large; Mixtral last)
1. `e4b/reference_attn4` — the e4b reference path (dense expert loop) with attn-4bit: the internal parity control (tp1's rule: fused vs reference |Δ final| ≤ 0.05 and median step |Δ| ≤ 0.05 → PASS).
2. `e4b/fused_attn4` — PRIMARY e4b arm (`enable_fast_train(dgrad=True)` + attention 4-bit); gpt-oss: skipped as REFUSED (tp1 row cited), `attn_only` runs instead as the secondary row.
3. `unsloth/ckpt_unsloth` — PRIMARY Unsloth arm.
4. (anchor family only) `unsloth/ckpt_hf`, and the 200-step curves `e4b/fused_attn4_200` + `unsloth/ckpt_unsloth_200` are NOT repeated (P38 owns them).
Mixtral: e4b arms with `--offload 1` (tp1's setting); the Unsloth arm resident (its only mode).

### Validity rules (VOID never enters a ratio)
e4b fused: `n_patched == n_layers` and `kernel_calls_per_step_min ≥ 2·n_layers` (Granite 32, OLMoE 16, Qwen3 48, Gemma-4 30, Mixtral 32 per tp1), `n_attn4 == 4·n_layers` when attn-4bit; C1 frozen-bytes bit-exact. Unsloth: census `Params4bit_expert_stacks ≥ 2·n_layers`, `experts_forward_calls_per_step_min ≥ n_layers`, `n_bnb4bit_unwrapped ≥ n_layers`, the "Enabling LoRA on MoE parameters" banner; C1 bit-exact. Both: step count == N, same tokens sha, same trainable count.

### Verdict and reading per family (pre-registered)
- **Support**: OK / REFUSED (framework says no, or 0 patched) / OOM / install_failed / load_fault / harness_error / alarm — every attempt a row.
- **Position** (both VALID): s/step ratio Unsloth/e4b (median of steps 11..N), peak-VRAM Δ, J/step ratio, tok/s. **Quality reading**: held-out |Δ| at N=60 ≤ 0.05 nats reads COMPARABLE (a reading threshold, not a gate; P38's). A family whose quality reading is NOT comparable gets its ratio reported with that flag, never as a clean position.
- **e4b internal parity** per family (fused vs reference): tp1's rule, informational (tp1 owns the licence).
- **Cross-lane anchor**: Qwen3's ratio here vs P38's 1.413 — a difference beyond ±10% is a finding about box/stack variance, stated, not averaged away.

### Pre-registered predictions (falsifiable; competitor estimates biased UP per the standing correction)
P1 Granite & OLMoE: both frameworks train; ratio in [0.9, 1.3] (host-bound at ≤1B active). P2 Qwen3: ratio within ±10% of 1.413. P3 Gemma-4: Unsloth loads and engages its 4-bit MoE path (uncertain); ratio in [1.2, 1.6]. P4 Mixtral: Unsloth resident OOMs at seq 512; e4b offload trains (tp1) — no ratio. P5 gpt-oss: e4b fused REFUSED (tp1); Unsloth either refuses MXFP4 4-bit training or dequantises to bf16 (no 4-bit ratio either way). P6 quality COMPARABLE at N=60 on every family where both train. P7 e4b internal parity PASS on the five fused families (tp1).

### Out of scope
200-step curves beyond P38; time-to-target (no per-family target is registrable without data); Unsloth expert offload (does not exist); any change to tp1 verdicts; serving.

### Box, order, cost, failure classes
One Vast.ai RTX 5090 (tp1's spec: ≥ 320 GB disk, ≥ 98 GB host RAM, ≥ 40 MB/s pre-flight; cuda-12.8 image; hard-kill guard 10 h; HARD $35 cap, expected ≈ 6–7 h ≈ $5). Downloads per family, freed after the family (tp1's pattern). Failure classes write stubs with the vocabulary above; the lane never coerces a refusal. Amendments are dated entries below, before the data they touch.

### Amendments
- **Amendment 1 (2026-09-05 23:00Z, before any box is rented):** the Fixture line above wrote `AdamW lr 2e-4, batch 1 × accum 4, bf16 autocast`; P38 as run (and its pre-registration) used **lr 1e-4, batch 1, accum 1, bf16 compute with no autocast context**. tp2's fixture is P38's exactly, for every family (`tp2_run.sh` defaults `TP2_LR=1e-4 TP2_ACCUM=1 TP2_AUTOCAST=0`), so the Qwen3 anchor is a same-fixture repeat and P2 reads against 1.413 as registered. Wall estimate drops to ≈ 2 h (one row per step). No other line changes.

### Launch log
- 22:33Z launched from the mini (`/tmp/tp2`, queue → `tp2_all.sh`). First box **50004573** (84.238.141.9): pre-flight 41 MB/s on the first sample, **39 MB/s on the run-length sample → SLOW BOX, destroyed (teardown proven, remaining = [])**; the launcher exits after a re-roll (rc=1) rather than looping, so the queue was re-run by hand at 22:40Z: box **50004890** rented. No amendment (nothing in the lane changed).
- 22:45Z box **50004890** (81.39.139.61, 56 MB/s): **BOX REFUSED by the train anchor (rc=3)** — `h2d.self_pair 1.0985 > 1.03` (host-to-device bandwidth not repeatable within the strict tolerance); flops 191.56 TFLOPS (1.034× ref), launch self-pair 1.0069, class `pcie-full/launch-fast`. The registered anchor-strict rule refused it; box destroyed (proven), guard killed; anchor receipts kept as `refused-50004890.*`. Re-rolled (attempt 3). If a third box is refused, the next step is an amendment decision (relax the h2d self-pair tolerance for a TRAINING lane, where H2D is not on the measured path) — not a silent retry loop.
