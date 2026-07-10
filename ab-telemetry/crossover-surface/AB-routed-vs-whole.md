# A/B: routed-subset vs whole-layer staging (2026-07-10, 3090, quarantined)

**Verdict: whole-layer WINS. Routed-subset diverges on the real e4b forward — NOT ready for the
bare-metal run.** The cheap-3090 A/B did exactly its job: caught a broken design before it reached
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
