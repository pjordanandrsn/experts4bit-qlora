# Layer-major beyond Qwen3: Mixtral, OLMoE, Gemma-4 — design packet (before any code)

**Goal.** `layer_major_step` (`experts4bit_qlora/engines/layer_major.py`) is licensed on Qwen3-30B-A3B under expert offload
at 2.365× (RTX 4090, P137) and 2.311× (RTX 5090, P140). Its gate is `SUPPORTED = ("Qwen3MoeForCausalLM",)`. Mixtral,
OLMoE and Gemma-4 also train under offload in e4b. This packet says what each family needs before the gate can include
it. Nothing is claimed for speed: a family's gain is read only by its own registered run.

## What layer-major assumes today

`layer_major.py` reproduces the model's forward around the decoder layers, so it is exact only where its reproduction is:

| step | today (Qwen3) |
|---|---|
| preamble | `model.model.embed_tokens(ids)`; positions `arange(T)`; one causal mask, or a sliding-window mask when `config.sliding_window` is set (read as an attribute); one `rotary_emb(h, position_ids)` |
| layers | each of `layers[:num_hidden_layers]` called as `layer(h, attention_mask=, position_ids=, past_key_values=None, use_cache=False, position_embeddings=)`, returning a tensor |
| head and loss | the model's own `forward(inputs_embeds=last_h, labels=…)` with `layers` swapped for an empty list: norm, `lm_head` and the model's loss |
| refusals | `output_router_logits` (the aux loss), dropout > 0 in training (config keys containing "dropout" and `nn.Dropout` modules) |
| guard | on a model's first call, every micro-batch's loss must be `torch.equal` to the stock forward's, and a repeat of micro-batch 0 must be reproducible, both before the one backward (`LayerMajorDriftError`, `LayerMajorNotReproducibleError`) |

The offload side is architecture-agnostic. The pre- and post-hooks sit on each `ExpertsLoRA`, `hold_forward_residency` is
per handle, and `offload_handles` finds the experts at `mlp.experts` (Mixtral, OLMoE) or at `layer.experts` (Gemma-4,
whose `mlp` is dense). All three families get the same `ExpertsLoRA(Experts4bit)` from the loader.

## Per family

Line references are to transformers' modeling files. They were read at 5.13 locally; the $0 legs run on the A2000's
install, and the pinned stack for any paid read is P137's (5.18).

### OLMoE: one fix, then the Qwen3 path

- **Fix:** `OlmoeConfig` has no `sliding_window` attribute, so today's preamble raises `AttributeError` on it. Read it as
  `getattr(cfg, "sliding_window", None)`. OLMoE builds a plain causal mask.
- **Unchanged otherwise:** one rotary, a layer signature identical to Qwen3's, a tensor return. The aux loss is gated on
  `output_router_logits` (refused). `clip_qkv` clamps in the attention, so it runs inside the layer either way.
- **Refusals:** `attention_dropout` only, caught by today's check. There are no shared experts.

### Mixtral: one new refusal, then the Qwen3 path

- **Same preamble and loop as Qwen3.** One rotary, the same sliding-window rule (`sliding_window` defaults to None), and a
  layer that takes the same keywords (its `use_cache` goes to `**kwargs`) and returns a tensor.
- **New RNG source.** `router_jitter_noise` multiplies the router input by random noise in training. Today's dropout check
  misses it because the key has no "dropout" in it, and the drift guard's repeat of micro-batch 0 would catch it only as
  "not reproducible". Refuse it by name when it is above 0 in training, with its own message.
- **Aux loss:** gated on `output_router_logits` (refused). There are no shared experts.

### Gemma-4 (`Gemma4ForCausalLM`, the `gemma4_text` model the loader builds): a family-specific preamble

- **Embedding.** The scale by sqrt(hidden) happens inside `embed_tokens`, so calling it reproduces it.
- **Per-layer embeddings (PLE).** When `hidden_size_per_layer_input` is set (default 256), `per_layer_inputs` are built
  from `input_ids` and passed per layer. The head-only call cannot rebuild them from a hidden state.
  - **First cut:** refuse a model with PLE on, by name.
  - **Later**, if a released checkpoint needs it: precompute `per_layer_inputs` once per micro-batch and pass each layer
    its slice.
- **Masks.** A dict `{"full_attention": causal, "sliding_attention": sliding_window}` is chosen per layer by
  `config.layer_types[i]` (default pattern 5 sliding to 1 full, the last layer full). The preamble builds both, once per
  micro-batch.
- **Rotary.** One `Gemma4TextRotaryEmbedding` evaluated per layer type: `rotary_emb(h, pos, layer_type)` for each type in
  use.
- **Layer call.** `layer(h, per_layer_input, shared_kv_states=, position_embeddings=pe[type], attention_mask=mask[type],
  position_ids=, past_key_values=None)`, returning a tensor.
  - **`shared_kv_states`** is a per-forward dict that the last layer of each type writes, and that KV-shared layers read.
  - Layer-major runs micro-batch m through layer L before L+1, so each micro-batch keeps **its own** dict for the whole
    step. Sharing one dict across micro-batches would hand a layer another micro-batch's keys.
- **Dense MLP and experts in parallel.** `layer.mlp` (dense) and `layer.router` with `layer.experts` are summed after their
  own norms, then scaled by `layer_scalar`. This all happens inside the layer, so layer-major changes nothing there. The
  offload handle is on `layer.experts`, and dense layers have none.
- **Head.** `final_logit_softcapping` and the loss run in the model's own forward, so the head-only call reproduces them.
  There is no aux loss.
- **Refusals:** `attention_dropout` (caught today), and PLE on (above).

## The change

1. `SUPPORTED` gains `MixtralForCausalLM`, `OlmoeForCausalLM` and `Gemma4ForCausalLM`.
2. The preamble and layer call become a small per-family adapter: `_preamble(model, mb)` returns
   `(h, per_layer_kwargs_fn)`, where the function returns layer L's keyword arguments for that micro-batch.
   - **Qwen3, Mixtral and OLMoE** share one adapter: today's code, with `sliding_window` read by `getattr`.
   - **Gemma-4** gets its own: both masks, rotary per layer type, a `shared_kv_states` dict per micro-batch, and
     `per_layer_input=None` with PLE refused.
3. **New refusals, before any work:** `router_jitter_noise > 0` in training (Mixtral), and PLE on (Gemma-4). The drift
   guard and the reproducibility pre-check stay as they are, and run for every family on its first call.
4. **Unchanged:** the head-and-loss call, the `output_router_logits` and dropout refusals, offload, the hold, the staging
   counts, and the trainer (`LAYER_MAJOR=1` implies `FAST_TRAIN=1`; the family must also be a `fast_train` family, which
   all three are).

## The $0 legs (A2000, tiny models; correctness only, no timing)

Per family, on a tiny model loaded through the real loader with `offload=True` and `enable_fast_train(dgrad=True)`,
gradient checkpointing first. Each check is STRICT:

- **Drift-guard equality.** Step 0's per-micro-batch losses are `torch.equal` to the stock loop's. This is the guard
  itself, run and passed. A self-pair of the direct call runs first.
- **Gradients.** The adapters' gradients after one layer-major step match the stock loop's within the band
  `test_layer_major.py` already uses for Qwen3 (the same sum in another order).
- **Stagings.** 2n − 1 per steady pass with layer-major, and GRAD_ACCUM × (2n − 1) without, with n the family's MoE
  layers (Gemma-4's dense layers stage nothing).
- **Same start.** The adapters are equal at the step's entry.
- **Refusals:** Mixtral with `router_jitter_noise > 0`; Gemma-4 with PLE on; each by name, before any forward.
- **Gemma-4 masks.** The tiny Gemma-4 mixes sliding and full layers (`sliding_window` below the sequence length), so a
  wrong mask or a shared `shared_kv_states` fails the equality check rather than passing by accident.

Tiny builders already exist: `tests/test_loader_architectures.py` (`_olmoe`, `_mixtral`, `_gemma4`; set PLE to 0 for
Gemma-4), `tests/test_chunked_lm_loss.py` (`_config(family)`), and PLE-off Gemma configs with a sliding/full mix
(`tests/test_p108_box.py`, `tests/test_fam_box.py`).

## Paid reads (not part of this packet)

One registered read per family, P137's design at its Amendment 3 with that family's model and card, the bar S ≥ 1.10. Each
is run only if a user-facing number is wanted for that family. Until then the docs say layer-major *runs* on these
families and cite only the Qwen3 numbers.

## If deferred

Mixtral, OLMoE and Gemma-4 users under offload keep the stock accumulation loop: GRAD_ACCUM × the stagings per step.
