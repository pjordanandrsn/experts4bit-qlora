# A/B: routed-subset vs whole-layer staging (2026-07-10, 3090, quarantined)

> **CONVERGENCE VERDICT AT SCALE (2026-07-10/11, Qwen3-30B-A3B, pre-registered): FAIL for
> training.** The full convergence A/B this file was "pending" ran at published-config scale
> (150 steps, seq 2048 + packing, seed 42, bracket whole_a → routed → whole_b) against
> [`conv_ab_prereg.json`](conv_ab_prereg.json). Final eval: whole_a **1.509**, whole_b **1.511**,
> routed **1.572**. Measured floor |whole_a−whole_b| = **0.002** (exactly the archival calibration
> prior — session not drifted); routed sits **0.062 off the whole mean = 31× the floor**, and the
> direction is one-sided: routed's train loss is higher on **146/150 steps** (signed mean +0.0634
> vs whole-vs-whole +0.0019, a coin-flip 97/150). Both pre-registered criteria fail; both validity
> controls passed (no staging-flag bleed; whole_a reproduced the 1.510 anchor). Wall-clock showed
> **no speed win either**: routed 1187 s vs whole 1044/1230 s — inside arm-to-arm variance, as
> expected at the measured **rf ≈ 0.97** (layer-0 staged 124/128; at seq 2048 × top-k 8 the routed
> union approaches the full stack, so there were barely any bytes to save). Full logs/configs/env:
> [`conv-ab-qwen30b/`](conv-ab-qwen30b/).
>
> **Standing decision:** `whole_layer` is the only training-validated staging. Routed-subset is
> demoted to **forward/decode only** (where it remains bit-identical-validated and where low rf
> actually pays); its training path is experimental **with a failed A/B at scale**, not "pending."
> The 15-step OLMoE pass below did not transfer — small-scale/short-horizon equivalence is not
> evidence of at-scale convergence. Mechanism consistent with the earlier zeros→12 result: the
> training path is sensitive to un-routed row content, and the deterministic fill makes that error
> quiet and reproducible, not absent. Next step per the pre-registered failure path: reproduce
> cheaply (OLMoE/3090 or CPU, longer horizon) and test dose-response — does the gap scale with
> fill mass (1−rf)? The low-rf mini-pair (seq 256, 50 steps) was killed post-verdict ~7 min into
> its step-0 eval (staging-bound at low seq → projected ~$3 more on the $1.89/hr pod); **low-rf
> remains PENDING**, to run on cheap silicon with the exact synced configs
> ([`conv-ab-qwen30b/configs/`](conv-ab-qwen30b/configs/)). Full reduction:
> [`conv-ab-qwen30b/reduction.json`](conv-ab-qwen30b/reduction.json).

**Fix history (2026-07-10, superseded as a training claim by the verdict above):** the original
A/B caught routed diverging (loss 11.75); root cause was per-expert ADDRESSING (n_experts read the
packed `shape[0]` = flat byte count under bnb `quantize_moe_experts`, not the real 64). Fixed to
byte-range addressing with the real count (forward bit-identical) + deterministic real-expert fill
for un-routed rows. Verified then: routed test suite green; step-0 loss 0.7038==0.7038;
routed-vs-whole grad_norm 5.4% <= whole-vs-whole 6.7%; 15-step OLMoE loss tracked whole (mean
0.018) — **but the convergence A/B above shows that did not hold at 150-step/30B scale.** The
forward/decode bit-identity claims stand. The cheap-3090 A/B + debug did their job; so did the
at-scale A/B.

---
_Original verdict (superseded):_ whole-layer wins; routed diverges. The cheap-3090 A/B did exactly its job: caught a broken design before it reached
expensive silicon.

## Design under test

Routed-subset staging (new, `expert_offload_staging=routed`): stage only the experts a forward
routes to (distinct `top_k_index` union) instead of the whole layer — `read_fraction` x the
staging traffic. Whole-layer kept as the default arm. Correctness by construction: un-routed rows
never indexed → bit-identical output.

## What the A/B found

| eff=64 | s/step | loss₀ | status |
|---|---|---|---|
| whole-layer f=1.0 / 0.5 / 0.0 | 2.20 / 3.75 / 5.86 | **1.214** | ✅ correct (matches the surface) |
| routed f=1.0 / 0.5 / 0.0 | 1.18 / 1.34 / 1.38 | **11.75** | ❌ **DIVERGES** (fast because wrong) |

The routed "speedup" (46–77% faster, penalty collapsing +166%→+17%) was an artifact of a **broken
forward** — loss 11.75 vs 1.214, constant from step 0. It is not a real result.

## Root cause — FOUND and FIXED (forward), 2026-07-10

The divergence was **per-expert addressing**, not the staging concept. `n_experts` read the packed
tensor's `shape[0]`, which under axolotl `quantize_moe_experts` (bnb-parametrized) is the **flat
byte count** (67M/134M for OLMoE), not 64. So `fetch_expert` read ~3 bytes at offset `e*3` →
garbage → random logits (11.75 ≈ ln(vocab)). The kit's `Experts4bit` is `[E,…]` (`shape[0]==64`),
so the CPU fake passed; the real bnb layout is flat, so it failed.

**Fix:** get the real expert count from the owning module (`ParametrizedOlmoeExperts.num_experts=64`)
and address experts as contiguous **byte ranges** (`per = total_bytes // n_experts`) — correct for
both layouts. **The forward is now bit-identical to whole-layer: real-OLMoE training step-0 loss
`0.7038 == 0.7038` (exact).**

**So routed-subset is CORRECT for inference/decode** — the thesis's low-eff-tokens regime, forward
only. **Training remains blocked:** the gradient-checkpoint backward reads un-routed experts
(`zeros` → loss explodes to 12; `torch.empty` reuses freed memory ≈ real weights → close but
non-deterministic), so the trained trajectory diverges past step 0. Making the backward provably
sparse is the remaining work. Gated experimental (`AXOLOTL_EXPERT_OFFLOAD_ROUTED_EXPERIMENTAL=1`).

## Prior status (superseded)

- The staging is provably correct on a **lazy** fake MoE (CPU test `TestRoutedSubsetStaging`:
  output + grads bit-identical, both stores).
- `args[1]` on the real hooked module IS the correct `top_k_index` (64×8 int64, verified), so the
  staged union is correct.
- Yet the real axolotl e4b forward diverges. The bug is in the staging ↔ parametrized-forward
  interaction — unresolved. Additionally the per-expert copy loop is a Python-level perf
  bottleneck (a probe staging all experts hung at step 0).

Routed is now gated behind `AXOLOTL_EXPERT_OFFLOAD_ROUTED_EXPERIMENTAL=1` (raises otherwise) with a
loud warning. It needs **two** fixes before it's viable — correctness (the divergence) and
performance (batched multi-expert staging) — neither a quick tweak.

## Decision for the bare-metal (GEX131) session

**Bring whole-layer** — correct, validated (the quarantined surface stands), performant (async-H2D
fix). Routed-subset is a sound idea worth fixing (it isn't perishable), but a diverging + slow
design must not consume the expensive host. Fix routed on cheap silicon first, then re-A/B.
