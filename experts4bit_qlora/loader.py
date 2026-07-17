"""Streaming 4-bit loader for fused-MoE checkpoints (OLMoE, Qwen3-MoE / Qwen3.5-MoE, Gemma-4, GraniteMoe).

Streams the checkpoint tensor-by-tensor straight onto the GPU, quantizing each fused expert stack
to NF4 on the way and dropping the bf16 source immediately, so the full bf16 model is never
materialized in CPU *or* GPU memory. Each layer's fused ``experts`` module is swapped for a frozen
4-bit :class:`Experts4bit` base wrapped in trainable per-expert :class:`ExpertsLoRA` adapters.

Supports SwiGLU fused-MoE architectures. Experts may be stored on disk either **per-expert**
(``...experts.{e}.{gate,up,down}_proj.weight`` — OLMoE, Qwen3-MoE) or already **fused**
(``...experts.{gate_up,down}_proj`` — Gemma-4, GraniteMoe); both are handled. The experts module
sits under the MLP for OLMoE/Qwen3, directly on the layer (beside a parallel dense MLP) for
Gemma-4, and under a ``block_sparse_moe`` block for GraniteMoe — whose Hub checkpoints use legacy
tensor spellings (``input_linear``/``output_linear``, see :data:`LEGACY_KEY_RENAMES`). Requires
transformers>=5.0.
"""

import json
import os

from accelerate import init_empty_weights
from huggingface_hub import snapshot_download
from safetensors import safe_open
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.activations import ACT2FN

from . import Experts4bit, ExpertsNbit, normalize_quant_type
from .gptoss import GptOssExperts4bit
from .lora import ExpertsLoRA
from .mxfp4 import dequantize_mxfp4
from .offload import enable_expert_offload, enable_inference_prefetch
from .util import log

# model_type -> experts submodule path relative to `model.layers.{i}`.
# OLMoE/Qwen3 nest experts under the MLP; Gemma-4 puts them beside a parallel dense MLP; GraniteMoe
# nests them under a `block_sparse_moe` block (router + experts; no parallel dense branch).
SUPPORTED_ARCHITECTURES = {
    "olmoe": "mlp.experts",
    "qwen3_moe": "mlp.experts",
    "qwen3_5_moe": "mlp.experts",
    "gpt_oss": "mlp.experts",  # GPT-OSS: on-disk MXFP4 blocks/scales + per-proj biases (see gptoss.py)
    "gemma4": "experts",  # multimodal top-level config
    "gemma4_text": "experts",  # the text tower (what a text-only QLoRA loads)
    "granitemoe": "block_sparse_moe.experts",  # IBM Granite MoE (granite-3.0-*-a*m, PowerMoE-3b)
}
SUPPORTED_MODEL_TYPES = set(SUPPORTED_ARCHITECTURES)

# model_type -> ((legacy on-disk spelling, name in the transformers>=5 module tree), ...).
# GraniteMoe checkpoints on the Hub predate the standardized fused-experts interface: the fused
# stacks are stored as `block_sparse_moe.input_linear.weight` [num_experts, 2*inter, hidden]
# (gate+up pre-fused, gate first — the module chunks the projection in half and activates the first
# half, matching Experts4bit's has_gate convention) and `block_sparse_moe.output_linear.weight`
# [num_experts, hidden, inter]; the router weight sits one module deeper, at `router.layer.weight`.
# transformers' own from_pretrained applies exactly these renames (conversion_mapping.py,
# "granitemoe"); this loader reads shards directly, so it must apply them itself. Substring
# renames over the whole key set: a checkpoint already saved with the current names matches
# nothing and passes through unchanged.
LEGACY_KEY_RENAMES = {
    "granitemoe": (
        ("block_sparse_moe.input_linear.weight", "block_sparse_moe.experts.gate_up_proj"),
        ("block_sparse_moe.output_linear.weight", "block_sparse_moe.experts.down_proj"),
        ("block_sparse_moe.router.layer.weight", "block_sparse_moe.router.weight"),
    ),
}


def _assign(model, name, tensor):
    """Place a real (GPU) tensor into a meta-initialized module by dotted name."""
    *path, attr = name.split(".")
    mod = model.get_submodule(".".join(path)) if path else model
    if attr in mod._parameters:
        mod._parameters[attr] = torch.nn.Parameter(tensor, requires_grad=False)
    elif attr in mod._buffers:
        mod._buffers[attr] = tensor
    else:
        setattr(mod, attr, tensor)


def load_moe_4bit_streaming(
    model_id, device, dtype, r, alpha, offload=False, pin=True, prefetch=False, quant_type="nf4"
):
    """Stream the checkpoint onto the GPU, quantizing fused experts to Experts4bit on the way.

    Peak memory stays low: the model is built on ``meta`` (no allocation), then each tensor is read
    one at a time directly to the GPU. The big fused-expert stacks are quantized to NF4 (~3.5x
    smaller) and their bf16 source is dropped immediately, so the full bf16 model never exists.

    When ``offload`` is set, each layer's frozen 4-bit experts are moved to (pinned, if ``pin``) CPU
    RAM *immediately after that layer is built* — inside the per-layer loop, never in a post-load
    pass (which would require every layer's experts GPU-resident first, defeating the purpose). A
    forward pre-hook streams a layer's experts back to the GPU just-in-time and a post-hook evicts
    them, so only one layer's experts are GPU-resident at a time (see :mod:`experts4bit_qlora.offload`).

    ``prefetch=True`` (with ``offload``) additionally links the layers for inference prefetch: during
    ``no_grad`` forwards each layer starts the next layer's H2D copy on a side stream, overlapping
    transfer with compute at a bounded cost of two layers resident instead of one. Training forwards
    are unaffected. See :func:`experts4bit_qlora.offload.enable_inference_prefetch`.
    """
    # Validate + canonicalize the scheme FIRST: a bad quant_type must fail here, before any config
    # fetch, snapshot download, or shard read — and the Experts4bit-vs-ExpertsNbit class dispatch
    # below must only ever see canonical names (an unnormalized alias would silently pick the
    # wrong class).
    quant_type = normalize_quant_type(quant_type)
    if prefetch and not offload:
        raise ValueError(
            "prefetch=True requires offload=True: prefetch overlaps the H2D copy of offloaded "
            "experts; without offload there is nothing to prefetch."
        )
    config = AutoConfig.from_pretrained(model_id)
    model_type = getattr(config, "model_type", None)
    if model_type not in SUPPORTED_ARCHITECTURES:
        raise NotImplementedError(
            f"Unsupported model_type={model_type!r}. This streaming loader handles SwiGLU fused-MoE "
            f"checkpoints: {sorted(SUPPORTED_ARCHITECTURES)}. The Experts4bit primitive itself is "
            "model-agnostic — see the README 'Scope' note to adapt another architecture."
        )
    expert_rel = SUPPORTED_ARCHITECTURES[model_type]
    # Multimodal configs (e.g. Gemma-4's `gemma4`) nest the language model under `text_config` and
    # prefix its checkpoint tensors with `model.language_model.` (vision lives under `model.vision_tower.`).
    # Build + size the text tower from that sub-config, and strip the prefix so keys match the text
    # CausalLM we build (dropping the vision weights we don't need for a text-only QLoRA).
    lm_config = getattr(config, "text_config", None) or config
    ckpt_prefix = "model.language_model." if lm_config is not config else ""
    act_name = getattr(lm_config, "hidden_activation", None) or getattr(lm_config, "hidden_act", "silu")
    try:
        activation = ACT2FN[act_name]
    except KeyError:
        # gpt_oss uses its own clamped GLU inside GptOssExperts4bit; the generic
        # activation is unused on that path, so a missing ACT2FN entry is fine.
        activation = None

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(lm_config, dtype=dtype)

    snap = (
        model_id
        if os.path.isdir(model_id)
        else snapshot_download(
            model_id,
            allow_patterns=["*.safetensors", "*.json", "tokenizer*", "*.model", "*.txt"],
        )
    )
    index_path = os.path.join(snap, "model.safetensors.index.json")
    if os.path.exists(index_path):
        raw_map = json.load(open(index_path))["weight_map"]
    else:
        # Small checkpoints ship unsharded with no index (e.g. granite-3.0-1b-a400m-instruct is a
        # single model.safetensors): synthesize the weight map from the file's own key list so the
        # streaming path below works unchanged. Anything else is unstreamable — fail loudly.
        single_path = os.path.join(snap, "model.safetensors")
        if not os.path.exists(single_path):
            raise FileNotFoundError(
                f"{model_id!r}: found neither 'model.safetensors.index.json' (sharded) nor "
                "'model.safetensors' (single-file) in the snapshot — nothing this loader can stream."
            )
        with safe_open(single_path, framework="pt", device="cpu") as f:
            raw_map = dict.fromkeys(f.keys(), "model.safetensors")
    if ckpt_prefix:
        weight_map = {"model." + k[len(ckpt_prefix) :]: f for k, f in raw_map.items() if k.startswith(ckpt_prefix)}
        orig_key = {"model." + k[len(ckpt_prefix) :]: k for k in raw_map if k.startswith(ckpt_prefix)}
    else:
        weight_map, orig_key = raw_map, {k: k for k in raw_map}
    # Normalize legacy tensor spellings (see LEGACY_KEY_RENAMES) so the expert-fusing loop and the
    # non-expert `_assign` pass below only ever see the module names the built model actually has;
    # `orig_key` keeps pointing at the on-disk spelling the shard must be read with.
    for old, new in LEGACY_KEY_RENAMES.get(model_type, ()):
        for key in [k for k in weight_map if old in k]:
            renamed = key.replace(old, new)
            weight_map[renamed] = weight_map.pop(key)
            orig_key[renamed] = orig_key.pop(key)
    handles = {f: safe_open(os.path.join(snap, f), framework="pt", device=device) for f in set(weight_map.values())}

    def get(name):
        return handles[weight_map[name]].get_tensor(orig_key[name])

    n_layers = lm_config.num_hidden_layers
    n_exp = getattr(lm_config, "num_local_experts", None) or getattr(lm_config, "num_experts", None)
    log(f"  fusing + quantizing experts (up to {n_layers}x{n_exp}) to {quant_type} (streaming)...")
    expert_keys = set()
    offload_handles = []
    n_moe = 0
    for i in range(n_layers):
        epfx = f"model.layers.{i}.{expert_rel}."  # "...mlp.experts." / "...experts." / "...block_sparse_moe.experts."
        if f"{epfx}gate_up_proj_blocks" in weight_map:
            # GPT-OSS: experts on disk as MXFP4 (blocks/scales) + per-projection biases.
            # Dequantize the exact released bytes, then build a faithful NF4 expert
            # (biases + clamped GLU) — see gptoss.py. Built bare (no ExpertsLoRA):
            # GPT-OSS-aware training LoRA is a separate change.
            if offload:
                raise NotImplementedError("gpt_oss + expert offload is not yet supported")
            gate_up = dequantize_mxfp4(get(f"{epfx}gate_up_proj_blocks"), get(f"{epfx}gate_up_proj_scales"), dtype=dtype)
            down = dequantize_mxfp4(get(f"{epfx}down_proj_blocks"), get(f"{epfx}down_proj_scales"), dtype=dtype)
            gu_bias = get(f"{epfx}gate_up_proj_bias").to(dtype)
            dn_bias = get(f"{epfx}down_proj_bias").to(dtype)
            expert_keys.update({
                f"{epfx}{k}" for k in (
                    "gate_up_proj_blocks", "gate_up_proj_scales", "gate_up_proj_bias",
                    "down_proj_blocks", "down_proj_scales", "down_proj_bias",
                )
            })
            n_moe += 1
            experts = GptOssExperts4bit.from_gptoss(
                gate_up, gu_bias, down, dn_bias, quant_type=quant_type, compute_dtype=dtype
            ).to(device)
            parent, leaf = epfx.rstrip(".").rsplit(".", 1)
            setattr(model.get_submodule(parent), leaf, experts)
            del gate_up, down
            continue
        if f"{epfx}gate_up_proj" in weight_map:
            # Already fused on disk (Gemma-4; GraniteMoe after the legacy rename above):
            # [num_experts, 2*inter, hidden] / [num_experts, hidden, inter].
            gate_up = get(f"{epfx}gate_up_proj").to(dtype)
            down = get(f"{epfx}down_proj").to(dtype)
            expert_keys.update({f"{epfx}gate_up_proj", f"{epfx}down_proj"})
        elif f"{epfx}0.gate_proj.weight" in weight_map:
            # Per-expert Linears on disk (OLMoE, Qwen3): fuse gate_up[e] = cat([gate, up]).
            gate_up_rows, down_rows = [], []
            for e in range(n_exp):
                g, u, d = (
                    get(f"{epfx}{e}.gate_proj.weight"),
                    get(f"{epfx}{e}.up_proj.weight"),
                    get(f"{epfx}{e}.down_proj.weight"),
                )
                gate_up_rows.append(torch.cat([g, u], dim=0))  # [2*inter, hidden]
                down_rows.append(d)  # [hidden, inter]
                expert_keys.update({f"{epfx}{e}.{p}.weight" for p in ("gate_proj", "up_proj", "down_proj")})
            gate_up = torch.stack(gate_up_rows).to(dtype)
            down = torch.stack(down_rows).to(dtype)
        else:
            continue  # dense layer (no experts here — e.g. Qwen3 mlp_only_layers, or a dense Gemma-4 layer)
        n_moe += 1
        # Instantiate the most-specific class for the scheme: 4-bit loads stay `Experts4bit`
        # instances, so downstream `isinstance(x, Experts4bit)` checks keep working exactly as
        # they did before the ExpertsNbit fold.
        base_cls = Experts4bit if quant_type in ("nf4", "fp4") else ExpertsNbit
        base = base_cls.from_float(
            gate_up, down, has_gate=True, activation=activation, quant_type=quant_type, compute_dtype=dtype
        )
        experts = ExpertsLoRA(base, r=r, alpha=alpha, dtype=dtype).to(device)
        if offload:
            # Move this layer's packed experts to (pinned) CPU now, before the next layer is built,
            # so the GPU never holds more than one layer's experts at a time during load.
            offload_handles.append(enable_expert_offload(experts, device, pin=pin))
        parent, leaf = epfx.rstrip(".").rsplit(
            ".", 1
        )  # ("model.layers.i.mlp","experts") or ("model.layers.i","experts")
        setattr(model.get_submodule(parent), leaf, experts)
        del gate_up, down
    if n_moe == 0:
        raise RuntimeError(
            f"no fused expert stacks found in {model_id!r} (model_type={model_type!r}): expected "
            f"'model.layers.<i>.{expert_rel}.gate_up_proj' (fused) or "
            f"'model.layers.<i>.{expert_rel}.0.gate_proj.weight' (per-expert) tensors in the "
            "checkpoint. Refusing to return a model with zero quantized expert layers — silently "
            "skipping the experts is the exact failure this loader exists to prevent."
        )
    log(f"  quantized experts on {n_moe}/{n_layers} MoE layers ({n_exp} experts each)")

    if offload_handles:
        pinned = all(h.pinned for h in offload_handles)
        log(
            f"  offloaded {len(offload_handles)} layers' 4-bit experts to {'pinned ' if pinned else ''}CPU RAM "
            "(streamed to GPU one layer at a time during train/eval)"
        )
        if prefetch:
            # Handles were appended in layer order above, which is what the circular linking needs.
            enable_inference_prefetch(offload_handles)
            log("  inference prefetch ON: next layer's experts copy on a side stream during no_grad forwards")
        from .offload import _arena_enabled, _stats_enabled, report_offload_environment

        if _arena_enabled():
            log("  offload arena ON (E4B_OFFLOAD_ARENA): experts staged as consolidated per-dtype copies")
        if _stats_enabled():  # A2: name the PCIe bus + H2D ceiling these per-layer figures ride
            report_offload_environment(device, log)

    log("  loading non-expert weights (attention/embeddings/router/norms/dense-mlp)...")
    for name in weight_map:
        if name not in expert_keys:
            _assign(model, name, get(name))

    # Non-persistent buffers (rotary inv_freq) aren't in the checkpoint — recompute every rotary module
    # the model has (some architectures, e.g. Gemma, use more than one). Generic; no per-model import.
    # Use `lm_config` (the text tower's config): a multimodal top-level config (Gemma-4's `Gemma4Config`)
    # lacks the rotary fields (`max_position_embeddings`, rope_theta) that live on `text_config`.
    for name, module in list(model.named_modules()):
        if type(module).__name__.endswith("RotaryEmbedding"):
            parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
            setattr(parent, name.rsplit(".", 1)[-1], type(module)(lm_config).to(device))
    # Tie lm_head if the checkpoint relied on weight tying.
    if model.lm_head.weight.is_meta:
        model.lm_head.weight = model.model.embed_tokens.weight

    stray = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]
    if stray:
        raise RuntimeError(f"unmaterialized meta tensors remain: {stray[:8]}")
    return model, config


# Backwards-compatible alias (was OLMoE-only).
load_olmoe_4bit_streaming = load_moe_4bit_streaming
