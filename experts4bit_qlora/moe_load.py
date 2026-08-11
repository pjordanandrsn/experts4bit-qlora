"""Execute a validated MoE/dense load plan against a model built on ``meta``.

:mod:`~experts4bit_qlora.moe_plan` proves the checkpoint and the module tree
agree. This module carries out that plan: read each tensor, fuse the per-expert
ones into the stacks the tree declares, and place everything on the target
device. Planning first is what makes execution boring — by the time a byte is
read, every key already has a proven home.

Design notes that are load-bearing rather than stylistic:

* **Streaming placement.** Each tensor moves to its final device as it is read,
  so peak host RAM is one tensor (plus one layer's experts mid-fusion) rather
  than the whole dequantized model. This is what lets a card hold a model the
  host could not.
* **Expert stacks are built per layer and released.** A layer's per-expert
  tensors are gathered, fused, assigned, then dropped, so the transient is one
  layer's experts — not the whole MoE.
* **Computed buffers are rebuilt, not loaded.** Rotary ``inv_freq`` is derived
  from the config and shipped by no checkpoint format; on a meta-built model it
  stays meta and the first forward dies with "Cannot copy out of meta tensor".
  That failure cost a live GPU run to find, so it is handled here for every
  model rather than per-family.
* **The final check is against the TREE.** After execution nothing in the model
  may still be on ``meta``. Checking what we *promised* can only ever find
  holes we already knew about.
"""
from __future__ import annotations

import torch

from .fp8_blocks import dequantize_fp8_blocks, fp8_block_scale_shape
from .compressed_int import dequantize_compressed_int
from .mxfp4 import dequantize_mxfp4
from .moe_conventions import MoEConventionError, fuse_experts, stack_experts


def _assign(model: torch.nn.Module, name: str, tensor: torch.Tensor) -> None:
    """Place `tensor` at dotted name, replacing a meta parameter or buffer.

    Meta tensors have no storage to write into, so they must be REPLACED, not
    copied into. Buffers matter as much as parameters here: a router's
    correction bias is a real weight that lives as a buffer.
    """
    parent, _, leaf = name.rpartition(".")
    mod = model.get_submodule(parent) if parent else model
    current = getattr(mod, leaf, None)
    if current is None:
        raise MoEConventionError(f"no such parameter or buffer on the model: {name}")
    if tuple(current.shape) != tuple(tensor.shape):
        raise MoEConventionError(
            f"{name}: checkpoint gives {tuple(tensor.shape)}, model declares "
            f"{tuple(current.shape)} — refusing to place a mis-shaped tensor")
    if leaf in mod._parameters:
        mod._parameters[leaf] = torch.nn.Parameter(tensor, requires_grad=False)
    else:
        mod._buffers[leaf] = tensor


def _materialize_computed_buffers(model: torch.nn.Module, device) -> list:
    """Rebuild buffers derived from config that no checkpoint ships.

    Uses each module's own rope initializer (the call its constructor makes)
    rather than reimplementing the formula, so a rope-scaling config keeps
    whatever scaling it declares.
    """
    rebuilt = []
    for name, mod in model.named_modules():
        inv = getattr(mod, "inv_freq", None)
        if inv is None or not inv.is_meta:
            continue
        cfg = getattr(mod, "config", None)
        rope_type = getattr(mod, "rope_type", "default")
        init_fn = None
        if rope_type != "default":
            try:
                from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
                init_fn = ROPE_INIT_FUNCTIONS[rope_type]
            except Exception:
                init_fn = None
        if init_fn is None:
            init_fn = getattr(mod, "compute_default_rope_parameters", None)
        if init_fn is not None and cfg is not None:
            fresh, attn_scale = init_fn(cfg, device)
            mod.register_buffer("inv_freq", fresh, persistent=False)
            if getattr(mod, "original_inv_freq", None) is not None:
                mod.register_buffer("original_inv_freq", fresh.clone(),
                                    persistent=False)
            if attn_scale is not None and hasattr(mod, "attention_scaling"):
                mod.attention_scaling = attn_scale
            rebuilt.append(f"{name}.inv_freq")
            continue
        # Vision/audio rotary towers (Qwen3-VL's Qwen3VLMoeVisionRotaryEmbedding)
        # have no config and no rope_type: their __init__ computes the plain
        # default inv_freq = 1 / theta**(arange(0, dim, 2)/dim) from `dim` and
        # `theta`/`base` attributes it kept. Recompute from those exactly —
        # this is the same formula, not a re-derivation, so a config-driven
        # rope above is never routed here.
        dim = getattr(mod, "dim", None)
        theta = getattr(mod, "theta", getattr(mod, "base", None))
        if isinstance(dim, int) and isinstance(theta, (int, float)):
            fresh = 1.0 / (theta ** (
                torch.arange(0, dim, 2, dtype=torch.float, device=device) / dim))
            mod.register_buffer("inv_freq", fresh, persistent=False)
            rebuilt.append(f"{name}.inv_freq")
            continue
        raise MoEConventionError(
            f"{name}.inv_freq is on meta and the module exposes neither a rope "
            f"initializer (config + rope_type) nor dim/theta to rebuild it from")
    return rebuilt


def execute_moe_plan(
    plan,
    model: torch.nn.Module,
    read_tensor,
    *,
    device="cpu",
    dtype: torch.dtype = torch.bfloat16,
    strict: bool = True,
) -> dict:
    """Carry out `plan` against `model`.

    ``read_tensor(checkpoint_key) -> torch.Tensor`` is the only I/O this module
    performs, so the same executor serves safetensors shards, a GGUF reader, an
    NVMe arena, or a test fixture.

    Returns a report: assigned / fused / rebuilt-buffer counts, plus any
    parameters still on ``meta`` (which raises under ``strict``).
    """
    def read(key):
        """Read a checkpoint tensor, dequantizing block-FP8 or MXFP4 in place.

        Doing this in the read path rather than at each call site means every
        consumer below — passthrough, expert fusion, tied heads — gets dense
        tensors without knowing the checkpoint was quantized. ``key`` is the
        MAPPED key: the real weight for FP8, or the synthesized base for MXFP4
        (whose primary/scale companions live only in the dequant map).
        """
        spec = plan.scales.get(key)
        if spec is None:
            t = read_tensor(key)
        else:
            kind, primary_key, scale_key, extra_key = spec
            primary = read_tensor(primary_key)
            scale = read_tensor(scale_key)
            if kind == "compressed_int":
                # compressed-tensors pack-quantized: packed + scale + shape.
                shape = read_tensor(extra_key)
                t = dequantize_compressed_int(primary, scale, shape, dtype=dtype)
            elif kind == "fp8":
                want = fp8_block_scale_shape(tuple(primary.shape))
                if tuple(scale.shape) != tuple(want):
                    raise MoEConventionError(
                        f"{key}: block-scale {scale_key} has shape "
                        f"{tuple(scale.shape)}, but a {tuple(primary.shape)} "
                        f"weight implies {tuple(want)} — refusing to dequantize "
                        f"with a mismatched scale")
                t = dequantize_fp8_blocks(primary, scale, dtype=dtype)
            elif kind == "mxfp4":
                # blocks [..., rows, G, B] with scales [..., rows, G]: they must
                # agree on every axis but the last (the packed byte axis).
                if tuple(primary.shape[:-1]) != tuple(scale.shape):
                    raise MoEConventionError(
                        f"{key}: mxfp4 blocks {tuple(primary.shape)} and scales "
                        f"{tuple(scale.shape)} disagree (blocks[:-1] must equal "
                        f"scales) — refusing to dequantize a mismatched pair")
                t = dequantize_mxfp4(primary, scale, dtype=dtype)
            else:
                raise MoEConventionError(f"{key}: unknown dequant kind {kind!r}")
        # Applied AFTER any dequant: a pre-fused family may store this stack
        # transposed vs the module. .transpose is a view; make it contiguous so
        # the parameter owns its storage rather than aliasing the read buffer.
        if plan.transforms.get(key) == "transpose_last2":
            t = t.transpose(-1, -2).contiguous()
        return t

    assigned = 0
    for ckpt_key, param in plan.passthrough.items():
        _assign(model, param, read(ckpt_key).to(dtype).to(device))
        assigned += 1

    fused = 0
    for layer, roles in plan.experts.items():
        first_name, down_name = plan.expert_targets[layer]
        n = len(roles["down"])
        down = [read(roles["down"][e]).to(dtype).to(device) for e in range(n)]
        up = [read(roles["up"][e]).to(dtype).to(device) for e in range(n)]
        if "gate" in roles:
            gate = [read(roles["gate"][e]).to(dtype).to(device) for e in range(n)]
            first, down_stack = fuse_experts(gate, up, down)
            del gate
        else:
            # Non-gated (nemotron_h): up_proj stacks on its own, no gate to fuse.
            first, down_stack = stack_experts(up, down)
        del up, down                           # release the per-expert transient
        _assign(model, first_name, first)
        _assign(model, down_name, down_stack)
        fused += 2

    # Heads the checkpoint omits because the model ties them. The plan only
    # lists a tie whose SOURCE was loaded, so this can never point at a meta
    # tensor — and doing it here, rather than trusting the model to have tied
    # itself, is what keeps the planner's exemption honest: the parameter is
    # excused from needing a key precisely because it gets real values now.
    loaded_from_disk = set(plan.passthrough.values())
    for target, source in plan.tied_params.items():
        src = model.get_parameter(source)
        parent, _, leaf = target.rpartition(".")
        mod = model.get_submodule(parent) if parent else model
        if target in loaded_from_disk:
            # The checkpoint shipped this head AND the config says it is tied.
            # Normally the two agree and the tie is a no-op on values. If they
            # disagree, the config and the checkpoint contradict each other and
            # either answer is a guess: tying discards weights the publisher
            # shipped, not tying ignores the config. Refuse rather than pick.
            on_disk = getattr(mod, leaf)
            if not torch.equal(on_disk.to(src.dtype), src):
                raise MoEConventionError(
                    f"{target}: config declares tie_word_embeddings=True, but the "
                    f"checkpoint ships a {target} that differs from {source} "
                    f"(max|diff|={float((on_disk.to(src.dtype) - src).abs().max()):.3e}) "
                    f"— tying would discard shipped weights, not tying would "
                    f"ignore the config; refusing to guess")
        # Bind the SAME Parameter object, not a copy of it. A copy would read
        # identically and hide the difference at inference time, then diverge
        # under training: the two halves would take independent gradient steps
        # and the model would stop being tied at all.
        mod._parameters[leaf] = src

    rebuilt = _materialize_computed_buffers(model, device)

    still_meta = sorted(n for n, t in model.state_dict().items() if t.is_meta)
    if strict and still_meta:
        raise MoEConventionError(
            f"{len(still_meta)} tensors still on meta after load, e.g. "
            f"{still_meta[:4]} — the checkpoint is incomplete or the plan drifted")
    return {"assigned": assigned, "fused_stacks": fused,
            "tied": len(plan.tied_params), "fp8_dequantized": sum(1 for v in plan.scales.values() if v[0]=="fp8"),
            "mxfp4_dequantized": sum(1 for v in plan.scales.values() if v[0]=="mxfp4"),
            "compressed_int_dequantized": sum(1 for v in plan.scales.values() if v[0]=="compressed_int"),
            "rebuilt_buffers": len(rebuilt), "still_meta": still_meta}
