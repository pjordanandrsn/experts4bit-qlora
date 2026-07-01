"""Streaming 4-bit loader for fused-MoE checkpoints (built/verified on OLMoE-1B-7B).

Streams the checkpoint tensor-by-tensor straight onto the GPU, quantizing each fused expert
stack to NF4 on the way and dropping the bf16 source immediately, so the full bf16 model is
never materialized in CPU *or* GPU memory. Each fused ``experts`` module is swapped for a frozen
4-bit :class:`Experts4bit` base wrapped in trainable per-expert :class:`ExpertsLoRA` adapters.

Requires transformers>=5.0 (fused ``OlmoeExperts``/``Qwen3MoeExperts`` layout).
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


def load_olmoe_4bit_streaming(model_id, device, dtype, r, alpha):
    """Stream the checkpoint onto the GPU, quantizing fused experts to Experts4bit on the way.

    Peak memory stays low: the model is built on ``meta`` (no allocation), then each tensor is read
    one at a time directly to the GPU. The big fused-expert stacks are quantized to NF4 (~3.5x
    smaller) and their bf16 source is dropped immediately, so the full bf16 model never exists.
    """
    config = AutoConfig.from_pretrained(model_id)
    if getattr(config, "model_type", None) != "olmoe":
        raise NotImplementedError(
            "This streaming loader is OLMoE-specific (fused-expert key layout, OlmoeRotaryEmbedding, "
            f"OlmoeAttention); got model_type={getattr(config, 'model_type', None)!r}. The Experts4bit "
            "primitive itself is model-agnostic — see the README 'Scope' note to adapt another fused-MoE."
        )
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, dtype=dtype)

    snap = snapshot_download(
        model_id,
        allow_patterns=["*.safetensors", "*.json", "tokenizer*", "*.model", "*.txt"],
    )
    weight_map = json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"]
    handles = {f: safe_open(os.path.join(snap, f), framework="pt", device=device) for f in set(weight_map.values())}

    def get(name):
        return handles[weight_map[name]].get_tensor(name)

    n_layers = config.num_hidden_layers
    n_exp = getattr(config, "num_local_experts", None) or config.num_experts
    expert_keys = set()
    log(f"  fusing + quantizing {n_layers}x{n_exp} experts to NF4 (streaming)...")
    for i in range(n_layers):
        # Checkpoint stores per-expert Linears; v5 fuses them. gate_up[e] = cat([gate, up]).
        epfx = f"model.layers.{i}.mlp.experts."
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

    log("  loading non-expert weights (attention/embeddings/router/norms)...")
    for name in weight_map:
        if name not in expert_keys:
            _assign(model, name, get(name))

    # Non-persistent buffers (rotary inv_freq) aren't in the checkpoint — recompute on GPU.
    from transformers.models.olmoe.modeling_olmoe import OlmoeRotaryEmbedding

    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = OlmoeRotaryEmbedding(config).to(device)
    # Tie lm_head if the checkpoint relied on weight tying.
    if model.lm_head.weight.is_meta:
        model.lm_head.weight = model.model.embed_tokens.weight

    stray = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]
    if stray:
        raise RuntimeError(f"unmaterialized meta tensors remain: {stray[:8]}")
    return model, config
