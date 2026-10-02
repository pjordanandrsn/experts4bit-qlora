# TC3 — the memory frontier: Qwen3-30B-A3B at the field recipe on a 24 GB RTX 4090 (rented) and on the owned 12 GB RTX A2000, every framework with its own memory levers — lane TC3 of #835, 2026-10-02

Pre-registration: [`../../tc1/TC3-PREREG.md`](../../tc1/TC3-PREREG.md) with its four dated amendments (pip in the hand run's venvs; C1 reading
e4b's offload home; the as-shipped resident fit row on the 24 GB token; every post-load exception a row). The TC1 harness at e4b `39834a7`
(the amendment-4 merge; `experts4bit/` byte-identical to TC1's 079a422 for the training path), grouped-nf4-gemm `846b512`, TC1's tokens
(sha `bfc742f67e37`) and per-slot init (`f7832488926eda91`). Receipts under `receipts/<run id>/`; the owned card's run is a hand run (no rental,
no private-store receipt) whose receipts ship here in full, labelled as such. `RESULTS-tc3-4090-8-vs-tc1.md` is the reducer's read of the
complete 24 GB box with `--tc1-dir` pointed at TC1's resident box (`tc1-5090-16`).

| box | token | host | cost |
|---|---|---|---|
| `tc3-4090-1` | `qwen3frontier` (24 GB, first draw) | RTX 4090, instance 53804549, AMD EPYC 7B13, driver 595.84 — the pre-amendment-2 harness: its e4b offload rows refused themselves in the C1 hasher | $0.28 |
| `tc3-4090-8` | `qwen3frontier` (24 GB, the complete box) | RTX 4090, instance 53830054, AMD EPYC 7642, driver 595.58.03 | $0.75 |
| `local-20261002T042928Z` | `qwen3frontier12` (12 GB, hand run) | the owned RTX A2000 12 GB in a container (Xeon W-1250, 6 cores, driver 575.64; the card shared with on-demand sidecars holding 1.37 GB at start), e4b at `516c87c` (amendment 2); the HF arm re-run once under amendment 5 (10:18–10:31Z) | $0 |

Between `tc3-4090-1` and `-8` six draws of the verified-secure RTX 4090 pool were refused at pre-flight or by the box itself (two slow links at
24 and 12 MB/s, two hosts that never answered ssh, one of each again, one driver 570.86 under the 580 floor): $0.14 in total, every one a
receipt, the machines excluded by those receipts as the launcher's rules allow.

> **Correction (2026-10-02, TC1 amendment 4).** The axolotl plain-arm failures in both tables below was this harness's, not axolotl's. axolotl's loader keeps every
> `*.gate` router in fp32 on purpose (`loaders/model.py:611-616, 1449-1451`) because its own trainer runs the forward under bf16 autocast;
> this harness runs no autocast, so the fp32 router met bf16 activations. The receipts show the router in float32 on every failing axolotl
> arm and in bfloat16 on the HF and e4b arms. `e4b.train.h2h.unsloth.qwen3.5090.2026-10-02.axolotl-unsupported` is retired, and the arm is
> re-asked with the routers cast to bf16 after load (what autocast computes per call).

## The 24 GB card: e4b trains the matched set under expert offload at half the card; Unsloth trains it resident (register `e4b.train.frontier.qwen3.4090-24gb.2026-10-02`)

The complete box, `tc3-4090-8` (every row one draw; the fit table as registered — a fit table, not a position):

| framework | arm | lever | verdict | peak VRAM | host RAM high-water | s/step | tok/s | J/step | held-out 0 → 20 |
|---|---|---|---|---|---|---|---|---|---|
| **e4b** | `fused_attn4_m_offload` | **expert offload** (`--offload 1`): the 48 layers' NF4 expert stacks pinned in host RAM, streamed per layer | **OK · VALID** | **11.881 GB** | 113.5 GB (the cgroup figure, the 61 GB checkpoint's page cache included) | **10.521** | 144.7 | 1325.4 | 1.9542 → **0.8548** |
| e4b | `reference_attn4_m_offload` | expert offload, the reference loop (the parity control) | OK · VALID | 13.890 GB | 86.2 GB | 77.387 | 19.3 | 4603.8 | 1.9697 → 0.8503 |
| e4b | `fused_attn4_m` | resident | **OOM** at step 1 | 24.449 GB | — | — | — | — | — |
| e4b | `fused_attn4_m_mb1` | resident, micro-batch 1 × accum 8 | **OOM** at step 2 | 24.530 GB | — | — | — | — | — |
| e4b | `fused_attn4_shipped` | resident, as shipped (bf16 expert adapters, N(0, 1/r) init; amendment 3's fit row) | **OOM** at step 18 | 24.408 GB | — | — | — | — | — |
| **Unsloth** 2026.9.14 | `ckpt_unsloth_m` | resident (`use_gradient_checkpointing="unsloth"`, `grouped_mm` engaged) | **OK · VALID** | **24.219 GB** | 62.9 GB | **10.032** | 138.5 | 1145.3 | 1.9515 → 0.8445 |
| Unsloth | `ckpt_unsloth_m_mb1` | micro-batch 1 × accum 8 | OK · VALID | 24.192 GB | 62.9 GB | 18.366 | 80.7 | 1922.0 | 1.9515 → 0.8418 |
| HF 5.18 + PEFT 0.21.2 | `hf_peft_m` | resident (bf16 experts) | **OOM** at load (23.50 GB in use at a 20 MiB allocation) | — | — | — | — | — | — |
| HF | `hf_peft_m_offload` | accelerate `device_map="auto"` + `max_memory` | **REFUSED** (`Tensor.item() cannot be called on meta tensors`) | — | — | — | — | — | — |
| axolotl 0.20.0 | `ckpt_axolotl_m` | resident (`quantize_moe_experts`) | **UNSUPPORTED** (its fp32 router against bf16 activations, as on every box of the campaign) | — | — | — | — | — | — |
| axolotl | `ckpt_axolotl_m_layeroffload` | `layer_offloading` | **UNSUPPORTED** (`invalid argument to getCurrentStream` under the harness's driving of the trainer mixin) | — | — | — | — | — | — |
| axolotl | `ckpt_axolotl_m_zero3` | DeepSpeed ZeRO-3 parameter offload | **UNSUPPORTED** (the registered refused stub: ZeRO-3 cannot be driven outside axolotl's trainer) | — | — | — | — | — | — |

Read, within the box:

- **e4b fits the field recipe under expert offload at 11.88 GB peak — 0.49 × Unsloth's resident peak — and steps in 10.52 s against Unsloth's
  resident 10.03 s.** Two single-draw rows beside each other, as the lane registered (no position is quoted from one draw each; TC1 owns the
  5090 position). Resident, e4b needs more than this card has: 24.45 GB at the first step at both recipes (the fp32 matched adapters and
  their AdamW-8bit state over 12,480 slots), and the as-shipped bf16-adapter arm reaches step 18 before it too OOMs at 24.41 GB.
- **Unsloth trains the matched set resident on 24 GB** (24.22 GB peak against 25.25 GB on the card): TC3 P1's clause (ii) is **FALSIFIED**,
  and that row is the finding. Its first draw on `tc3-4090-1` (an EPYC 7B13 host) read 8.43 s/step at the same 24.22 GB — host variance of the
  kind TC1 recorded (`.host-variance`), which is why nothing is quoted across the two boxes.
- **The offload path keeps the trajectory** (TC3 P3 **HELD**): against TC1's resident e4b arm on the 5090 (same tokens, init, precision and N),
  the median per-step |Δ| is 0.0025 (band 0.02), |Δ held-out at N| 0.0032, the step-0 losses SAME-BYTES-CLASS (0.0037); the in-box parity
  control (fused offload vs reference offload) is EQUIVALENT at 0.0028 / 0.0045. Unsloth's held-out at N sits 0.0103 under the anchor —
  EQUIVALENT by the lane's band.
- **TC3 P4 HELD**: 10.521 s/step on the 4090 under offload beside 2 × 5.688 = 11.376 from TC1's resident 5090 reading — two measurements with
  the registered factor applied, never a ratio.
- HF neither fits nor offloads here (its accelerate path dies on a meta tensor before the first step); axolotl's three levers are three
  refusals. P1's clauses (i), (iii-a) and (iii-b) hold; (iv) is vacuous (the ZeRO-3 arm did not run).

## The 12 GB card: e4b trains the matched set under expert offload at micro-batch 1 (register `e4b.train.frontier.qwen3.a2000-12gb.2026-10-02`)

| arm | lever | result | peak VRAM | host RAM high-water | s/step | held-out at N = 20 |
|---|---|---|---|---|---|---|
| e4b `fused_attn4_m_offload` (field recipe, mb 2 × accum 4) | expert offload | **OOM at step 2** (after C1, eval and one step) | 10.47 GB | 42.1 GB | — | — |
| e4b `fused_attn4_m_offload_d2` | expert offload | OOM at step 2 | — | — | — | — |
| e4b `reference_attn4_m_offload` | expert offload, the reference loop | OOM | — | — | — | — |
| e4b `fused_attn4_m` resident | — | PHASE_ALARM: the prologue's 420 s budget (a share of the arm's 1,200 s alarm) ran out inside `load_weights` (991 s on this CPU) — not a reading | — | — | — | — |
| **e4b `fused_attn4_m_offload_mb1`** (mb 1 × accum 8, the registered secondary) | expert offload | **OK · VALID** | **10.46 GB** | 25.3 GB | **69.9** (22 tok/s; 3,273 J/step) | **1.9533 → 0.8483** |
| e4b `reference_attn4_m_offload_mb1` | expert offload, the reference loop | OOM at step 1 (10.78 GB at a 20 MiB allocation) | 10.78 GB | — | — | — |
| e4b `fused_attn4_shipped_offload` (bf16 expert adapters) | expert offload | PHASE_ALARM: `load_weights` ran 2,511 s against the prologue's 2,520 s budget — not a reading | — | — | — | — |
| Unsloth 2026.9.14 on torch 2.8 `ckpt_unsloth_m_mb1` | none (its loader has no expert-offload lever) | **REFUSED at load**: bitsandbytes dispatches modules to the CPU on this card (`Some modules are dispatched on the CPU or the disk`) — UNSUPPORTED, not an OOM reading | — | — | — | — |
| HF 5.18 + PEFT 0.21.2 `hf_peft_m_mb1` | none (bf16 experts resident) | **OOM at load** — amendment 5's re-run with a 7,200 s alarm: 11.04 GiB in use at a 20 MiB allocation after 761 s of `load_weights`; the first pass had refused itself at 621 s against the 630 s prologue budget its 1,800 s alarm implied (a reading of this host's disk, not of HF; kept under `first-pass-alarm/`) | — | — | — | — |
| axolotl 0.20.0 `ckpt_axolotl_m` | — | the driver-refused stub (its loader's fp32 router; the campaign's row) | — | — | — | — |

The secondary trains the same 642,514,944 parameters from the same init as every matched arm of the campaign, with the fused kernel on all 48
layers on every step (1,536 kernel calls per step, 0 loop calls) and C1 bit-exact over 337 frozen tensors, 192 of them read from the offload
handle's pinned-host home (amendment 2). Its held-out loss at N = 20 (0.8483) sits inside the 5090's matched pair (0.8516 / 0.8487) and beside
the 4090's offload reading (0.8548): the offload path changes where the bytes live, not the arithmetic. The cost is the card: 70 s/step on an
A2000 streaming 48 layers of 4-bit experts over PCIe each step, with 1,872 s of nf4 quantisation on a 6-core Xeon before step 1. The 24 GB box
says what the field-recipe OOM on this card was: the same recipe peaks at 11.88 GB on the 4090, which is more than the 10.3 GB this shared card
had free — the margin, not the recipe; an unshared 12 GB card is marginal at micro-batch 2 and fits at micro-batch 1 (10.46 GB).
Two arms are not readings on this host: the resident e4b arm and the as-shipped offload arm ran out their prologue phase budgets inside
`load_weights` (991 s and 2,511 s of safetensors reads on this CPU with the sidecars active) — an open harness item for the slow owned card, said as such.
**TC3 P2 HELD** (`RESULTS-tc3-a2000-vs-tc1.md`, the reducer on the first pass with the amendment-5 HF receipt in place of its ALARM stub —
the stub and its log sit under `first-pass-alarm/`, the re-run's outputs under `amendment-5-rerun/`): the e4b fused offload arm completed at its
registered secondary after the field-recipe OOM, and every other framework's arm is OOM (HF) or UNSUPPORTED (Unsloth's CPU dispatch, axolotl's
driver floor — said as not OOM readings). P3 is not scored on this token: its field-recipe anchor OOMed and the secondary is not the anchor the
rule names; the secondary's held-out loss beside the 5090's pair above is the reading, not a scored prediction.

## Harness notes

- The C1 hasher counted e4b's evicted expert stacks (0-element GPU placeholders; the bytes live in the offload handle's pinned-CPU `home`) as
  empty frozen tensors and refused the first 24 GB box's offload rows; amendment 2 (`516c87c`) reads the home copy, and `tc3-4090-8` and the
  hand run ran from it. Amendment 4 (`39834a7`) wraps every arm so a post-load exception is a row with its phase and exception type — the
  axolotl rows above exist because of it.
- The RTX 4090 pool at or under $0.49/h gave one host passing the registered pre-flight (40 MB/s download, ssh within 180 s, driver ≥ 580)
  in eight draws; the pool finding is recorded in the private store's receipts and nothing in this read depends on the refused draws.
- HF's accelerate offload path (`device_map="auto"` + `max_memory`) is a harness-driven lever, not HF's documented QLoRA route; its meta-tensor
  refusal is recorded as UNSUPPORTED on this harness, not as a property of HF.
