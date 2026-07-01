"""Streaming 4-bit loader for fused-MoE checkpoints (OLMoE, Qwen3-MoE / Qwen3.5-MoE).

Streams the checkpoint tensor-by-tensor straight onto the GPU, quantizing each fused expert stack
to NF4 on the way and dropping the bf16 source immediately, so the full bf16 model is never
materialized in CPU *or* GPU memory. Each layer's fused ``experts`` module is swapped for a frozen
4-bit :class:`Experts4bit` base wrapped in trainable per-expert :class:`ExpertsLoRA` adapters.

Supports fused-MoE architectures that store experts **per-expert on disk** under
``model.layers.{i}.mlp.experts.{e}.{gate,up,down}_proj.weight`` with a SwiGLU gate — verified
identical on OLMoE-1B-7B and Qwen3-30B-A3B. (Gemma-4 differs: experts live at ``layers.{i}.experts``
beside a parallel dense MLP with a custom router — a separate adaptation, not handled here.)
Requires transformers>=5.0.
"""

import json
import os

from accelerate import init_empty_weights
from huggingface_hub import snapshot_download
from safetensors import safe_open
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.activations import ACT2FN

from . import Experts4bit
from .lora import ExpertsLoRA
from .util import log

# model_types whose experts are stored per-expert on disk under
# `model.layers.{i}.mlp.experts.{e}.{gate,up,down}_proj.weight` with a SwiGLU gate — handled identically.
SUPPORTED_MODEL_TYPES = {"olmoe", "qwen3_moe", "qwen3_5_moe"}


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


def load_moe_4bit_streaming(model_id, device, dtype, r, alpha):
    """Stream the checkpoint onto the GPU, quantizing fused experts to Experts4bit on the way.

    Peak memory stays low: the model is built on ``meta`` (no allocation), then each tensor is read
    one at a time directly to the GPU. The big fused-expert stacks are quantized to NF4 (~3.5x
    smaller) and their bf16 source is dropped immediately, so the full bf16 model never exists.
    """
    config = AutoConfig.from_pretrained(model_id)
    model_type = getattr(config, "model_type", None)
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise NotImplementedError(
            f"Unsupported model_type={model_type!r}. This streaming loader handles per-expert fused-MoE "
            f"checkpoints (mlp.experts.{{e}}.{{gate,up,down}}_proj, SwiGLU): {sorted(SUPPORTED_MODEL_TYPES)}. "
            "The Experts4bit primitive itself is model-agnostic — see the README 'Scope' note to adapt "
            "another architecture (e.g. Gemma-4, whose experts + router are laid out differently)."
        )
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, dtype=dtype)

    snap = (
        model_id
        if os.path.isdir(model_id)
        else snapshot_download(
            model_id,
            allow_patterns=["*.safetensors", "*.json", "tokenizer*", "*.model", "*.txt"],
        )
    )
    weight_map = json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"]
    handles = {f: safe_open(os.path.join(snap, f), framework="pt", device=device) for f in set(weight_map.values())}

    def get(name):
        return handles[weight_map[name]].get_tensor(name)

    n_layers = config.num_hidden_layers
    n_exp = getattr(config, "num_local_experts", None) or config.num_experts
    log(f"  fusing + quantizing experts (up to {n_layers}x{n_exp}) to NF4 (streaming)...")
    expert_keys = set()
    n_moe = 0
    for i in range(n_layers):
        # Checkpoint stores per-expert Linears; v5 fuses them. gate_up[e] = cat([gate, up]).
        epfx = f"model.layers.{i}.mlp.experts."
        # Skip dense layers (e.g. Qwen3 mlp_only_layers / decoder_sparse_step): only fuse where experts exist.
        if f"{epfx}0.gate_proj.weight" not in weight_map:
            continue
        n_moe += 1
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
        gate_up, down = (
            torch.stack(gate_up_rows).to(dtype),
            torch.stack(down_rows).to(dtype),
        )
        base = Experts4bit.from_float(
            gate_up,
            down,
            has_gate=True,
            activation=ACT2FN[config.hidden_act],
            quant_type="nf4",
            compute_dtype=dtype,
        )
        model.get_submodule(f"model.layers.{i}.mlp").experts = ExpertsLoRA(base, r=r, alpha=alpha, dtype=dtype).to(
            device
        )
        del gate_up, down, gate_up_rows, down_rows
    log(f"  quantized experts on {n_moe}/{n_layers} MoE layers ({n_exp} experts each)")

    log("  loading non-expert weights (attention/embeddings/router/norms)...")
    for name in weight_map:
        if name not in expert_keys:
            _assign(model, name, get(name))

    # Non-persistent buffers (rotary inv_freq) aren't in the checkpoint — recompute the model's own
    # rotary module (works across architectures; no per-model import).
    if getattr(model.model, "rotary_emb", None) is not None:
        model.model.rotary_emb = type(model.model.rotary_emb)(config).to(device)
    # Tie lm_head if the checkpoint relied on weight tying.
    if model.lm_head.weight.is_meta:
        model.lm_head.weight = model.model.embed_tokens.weight

    stray = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]
    if stray:
        raise RuntimeError(f"unmaterialized meta tensors remain: {stray[:8]}")
    return model, config


# Backwards-compatible alias (was OLMoE-only).
load_olmoe_4bit_streaming = load_moe_4bit_streaming
