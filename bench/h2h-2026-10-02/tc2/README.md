# TC2 — the other families at matched work on the current cuts: Granite-3.1-3B-A800M, OLMoE-1B-7B, gpt-oss-20b (box A); Qwen3.6-35B-A3B and Mixtral-8x7B (box B) — lane TC2 of #835, 2026-10-02

Pre-registration: [`../../tc1/TC2-PREREG.md`](../../tc1/TC2-PREREG.md) (with its amendment 1). The TC1 harness (`bench/tc1/`) at e4b
`09b0f6a` (0.38.1) for box A, grouped-nf4-gemm `846b512`; the same matched-work rules as TC1 (one per-slot init, fp32 adapters, the same
tokens per family, validity predicates, verdicts, two draws for a quoted position, within-box ratios only). Receipts under `receipts/<run id>/`
(private store `receipts/experts4bit-qlora/2026-10-02/<run id>/tc1/`).

| box | token | host (Vast verified-secure) | families | cost |
|---|---|---|---|---|
| `tc1-5090-17` | `tc2small` (N = 60, 48 held-out rows every 20 steps) | instance 53800083, AMD EPYC 9755, driver 580.119.02 | granite, olmoe, gptoss | $0.90 |
| `tc1-5090-22` (box B) | `tc2big` (the field recipe, N = 20) | instance 53828163, machine 117847 — **stopped by its host at 11:41Z** with the run's receipts unfetched (below) | qwen3_5, mixtral | $2.42 (`tc1-5090-18` and `-21` before it failed pre-flight on one dead machine, $0.03) |

## What the small families say

**The per-step edge e4b holds on Qwen3-30B-A3B on a 5090 does not generalise to small experts.** On both families where another
framework trained the experts in 4-bit or in bf16, the matched-work ratio is at or below parity in speed, while e4b keeps a VRAM edge
against bf16 experts and loses it against Unsloth's 4-bit stacks on OLMoE.

### Granite-3.1-3B-A800M (`ParallelExperts`; 32 layers)

| arm | s/step | ratio vs e4b `fused_attn4_m` (1.484 s, two draws) | peak VRAM | J/step | held-out at N = 60 (e4b 0.8311) | regime |
|---|---|---|---|---|---|---|
| HF + PEFT (`target_parameters`, two draws) | 1.441 | **0.971 [0.965, 0.978]** (HF faster) | 8.50 GB (e4b 4.54) | 334.9 (e4b 343.8) | 0.8286 (Δ −0.0024, COMPARABLE) | **bf16 experts** — bitsandbytes' Linear-only quantizer leaves `ParallelExperts` unquantised |
| axolotl 0.20.0 (`quantize_moe_experts`) | 1.340 | **0.903 [0.901, 0.905]** (axolotl faster; one draw) | 4.08 GB | 351.1 | 0.8295 (Δ −0.0016, EQUIVALENT) | bf16 experts too: its parametrisation found no stack to pack on this family (`Params4bit stacks 0`) |
| Unsloth 2026.9.14 (both target lists) | — | VOID | — | — | — | attention-only: 5,242,880 trainable vs e4b's 99,614,720 — it adapts no expert parameter (as registered; `coverage` row of tp4 stands) |
| HF on torch 2.14 + `experts_implementation="grouped_mm"` | 1.347 | 0.908 (labelled) | — | — | 0.8278 | the same bf16 experts, a different dispatch |
| e4b as shipped (bf16 expert adapters, N(0, 1/r)) | 1.247 | 0.840 (labelled) | 3.78 GB | — | 0.8329 (Δ +0.0018) | |

e4b's parity control PASSES (Δ final 0.00153, median 0.00168; ×5.81 vs its reference at ×1.09 the peak). Read: on an 800 M-active MoE
whose experts nobody else quantises, e4b's fused 4-bit experts train at 0.90–0.97 × the speed of bf16-expert training and at 0.5 × its
VRAM (4.54 vs 8.50 GB) — a footprint position, not a speed position. TC2 P1 (HF/e4b in [1.1, 1.6]) is FALSIFIED.

### OLMoE-1B-7B (16 layers, 64 experts)

| arm | s/step | ratio vs e4b `fused_attn4_m` (1.105 s, two draws) | peak VRAM | J/step | held-out at N = 60 (e4b 0.8424) | regime |
|---|---|---|---|---|---|---|
| Unsloth (grouped_mm, one draw) | 1.327 | **1.201 [1.199, 1.204]** (e4b faster) | **5.97 GB** (e4b 8.18) | **231.9** (e4b 326.2, ×0.71) | 0.8426 (Δ +0.0002, EQUIVALENT) | 4-bit expert stacks (+ double-quant) + bnb-4bit attention |
| HF + PEFT (two draws) | 1.387 | **1.255 [1.253, 1.258]** (e4b faster) | 16.32 GB | 490.0 (×1.50) | 0.8370 (Δ −0.0054, COMPARABLE) | bf16 experts |
| HF on torch 2.14 + `grouped_mm` | 1.316 | 1.191 (labelled) | 16.3 GB | — | 0.8376 | bf16 experts |
| axolotl 0.20.0 | — | UNSUPPORTED | — | — | — | its loader asserts `Original QKV code not found` on this family (both arms) |
| e4b as shipped | 0.902 | 0.816 (labelled) | 6.62 GB | — | 0.8373 (Δ −0.0051) | |

e4b's parity control PASSES (Δ final 0.00095, median 0.00159; ×6.08 vs its reference). Read: e4b is 1.20 × faster per step than Unsloth's
engaged 4-bit path and 1.26 × faster than bf16-expert HF, but **Unsloth trains this family at 2.2 GB less peak VRAM and 0.71 × the energy
per step, with the same loss**. TC2 P2 (Unsloth in [1.2, 3.0] — 1.201 sits on the edge; HF in [1.5, 2.5]) is FALSIFIED on the HF half.
The attention bytes differ between the two quantisers on this family (`q_proj` DIFFERENT, the control detecting), unlike Qwen3-30B-A3B where
they are SAME-BYTES: recorded, not resolved here.

### gpt-oss-20b (MXFP4 experts in the checkpoint)

No common adapter set, as registered: e4b has no expert path on this family (`enable_fast_train` patches 0 modules; the experts are
built bare — tp1/tp2 cited) and trains the attention only (3.820 s/step, 7,962,624 trainable, 17.82 GB, two draws STABLE). Unsloth's
bnb-4-bit load adapts the experts but left no `Params4bit` stack (VOID: 0 of 48, a silent fallback); its 16-bit load with the experts kept
packed in MXFP4 trained (`ckpt_unsloth_mxfp4`, two draws) but is VOID by the step-0 rule — its held-out loss at step 0 is 3.2109 against
the e4b anchor's 3.1100 (Δ 0.10 > 0.05): the two arms do not start from the same bytes, so no equivalence and no ratio. HF and axolotl refuse
the family at load (`The model is quantized with Mxfp4Config but you are passing a BitsAndBytesConfig`). TC2 P3 is FALSIFIED. No position.

## Box B: lost with its instance (TC2 amendments 2 and 3; re-runs registered)

`tc1-5090-22` ran from the TC3 amendment-4 merge (`39834a7`) for 4 h 40 min and stopped answering ssh at 11:42Z, inside Mixtral's Unsloth
micro-batch-1 secondary. The provider then listed the instance as exited and **stopped by the host** (memory use 85 % at the stop; its last
duration ending about 11:41Z), the ssh gateway refused every connection, the controller's deadline fetch failed and the guard destroyed the
instance with its disk (teardown complete, 12:20:45Z). The controller's heartbeat log holds the status of the cells it sampled, never a
number, so **nothing from that box is registered**; the private store's receipt for the run records the loss. What the heartbeats show, as
statuses only and with cells the sampling missed: Qwen3.6-35B-A3B — e4b's matched arms OOM at the field recipe (both draws) and at
micro-batch 1, e4b as shipped OK, Unsloth's three matched arms OK, HF OOM, axolotl's scattermoe arm refused; Mixtral-8x7B — e4b under
offload OK (both draws and at micro-batch 1), e4b as shipped OK, **both Unsloth matched draws out of their 2,400 s alarm with no step
taken** (still loading at about 30 GB on the card), HF OOM, axolotl's scattermoe arm refused. Two amendments follow from it:
amendment 2 (`tc2mixtral`, merged as `31cb04d`) re-asks Mixtral's P5 pair with the Unsloth alarm at 7,200 s, and amendment 3 (merged as
`0933d4e`) re-runs the Qwen3.6 half on its own box and gives the Mixtral box a 192 GB host-RAM floor, because Unsloth's loader
materialises the 93 GB bf16 checkpoint in host memory before it quantises. Both boxes (`tc1-5090-23`, `-24`) launch after this read;
their rows, P4, P5 and the Mixtral P6/P7 are a follow-up read. In this read P4 and P5 are **UNTESTED**.

## Predictions scored (box A)

P1 granite FALSIFIED; P2 olmoe FALSIFIED; P3 gptoss FALSIFIED; P6 (e4b parity on every family with a reference) HELD; P7 (matched sets
EQUIVALENT where two frameworks train one adapter set) FALSIFIED on the HF rows (COMPARABLE, inside 0.006 nats, outside the 0.005 band) and
held on the axolotl (granite) and Unsloth (olmoe) rows. P4 / P5 are UNTESTED here (box B lost; the re-runs above).

## Harness notes from this box

axolotl's scattermoe row (`ckpt_axolotl_best`) refused at load on every family: the KernelsPlugin fetches `kernels-community/rotary` and the
harness runs its arms with the Hub offline — a harness limit, open; its plain 4-bit arm ran on Granite. The `hf_peft_m_t214` rows ran in the
axolotl venv's torch 2.14 and transformers 5.17.0 and dispatched as recorded in each receipt.
