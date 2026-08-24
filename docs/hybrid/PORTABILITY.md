# Portability and naming — extending the hybrid engine to new MoE families

Written at the owner's direction (2026-08-24) while the campaign is
Qwen3-30B-centric, so the conventions are recorded BEFORE a second
family arrives.

## The three-layer rule

1. **Engine components carry no model names.** `SlotController`,
   `Fp8PagedKV.append_many`, `_HybridTier.swap_expert`,
   `batched_append`, the residency classes: all geometry (L, E, H, D,
   top-k) comes from the modules, the ctor, or the arena index — never
   from a family-specific config key at a call site. Audited
   2026-08-24: the only family word on the engine surface is the
   `gptoss` MECHANISM flag (per-expert bias epilogue + clamped GLU),
   detected structurally, with DeepSeek-V4's clamp-without-bias
   handled separately. **Rename candidate** when a third bias-carrying
   family lands: `gptoss` → `bias_epilogue` (+ the existing
   `clamp_limit`), so the flag names the mechanism, not the first
   model that had it.
2. **Bench instruments are family-flexible via flags, not forks.**
   Config-key lookups go through `_routed_topk(cfg)` (alias list —
   extend it, never hardcode a key at a call site); the decoder-layer
   tree is `--layers-attr` (default `model.layers`; latent families
   like Kimi use `model.language_model.layers` — the bake tool's
   `--prefix` is the same idea); the arena bake carries `--prefix` and
   `--moe` for tensor naming.
3. **Model-specific artifacts are data, clearly labeled, never
   reused.** Priors, routing profiles, calibrations, and every
   receipts-* directory are bound to (model, operating point, host
   class). Extending to a new family means NEW profile passes and NEW
   registrations that cite the old ones as method precedents — the
   tailvar/co-routing results are Qwen3-30B measurements, not
   constants of nature (rates are content- and model-conditional; that
   is the certified finding).

## Per-family onboarding checklist

* top-k config key present in `_routed_topk`'s aliases; expert count
  readable as `module.num_experts` (the loader sets it).
* Decoder-layer tree path for `--layers-attr`.
* Expert tensor naming for the bake (`--prefix`, `--moe`); latent-MoE
  geometry comes from the arena index by design
  (`expert_geometry_from_arena`), never the config.
* Bias/GLU mechanism: does the family carry per-expert biases
  (gptoss-mechanism) or clamps (DeepSeek-V4-mechanism)? `swap_expert`
  and the DRAM stacks follow the flags automatically.
* Fresh routing profile + prior artifacts; fresh registration for any
  claim (rates and constants do not transfer across families — the
  co-routing and tailvar receipts are the precedent).
