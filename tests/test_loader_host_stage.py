# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""With offload on CUDA the streaming loader reads each layer's per-expert rows into host memory and fuses them there
(``loader._stage_expert_rows_on_host``); only the fused stacks go to the device, where they are quantized as before. Read
straight to the device, a layer's rows and its fused stacks coexist there, about twice the layer's experts.

The stored experts must be the same bits either way. For two per-expert families (Qwen3-MoE as ``mlp.experts.{e}.*_proj``,
Mixtral as ``block_sparse_moe.experts.{e}.w{1,2,3}``) the packed 4-bit bytes and the absmax are compared bit for bit between a
staged and a direct load:

* on a CUDA host, the offload load (staged by default) against the same load with the staging switched off;
* on any host, a resident load with the staging forced on against the default resident load (direct), and the double-quantized
  absmax ``compress_expert_absmax_`` derives from each (it refuses offloaded layers, so the offload loads carry none).

A spy on the convention's fuser records where its rows were, so a staged load that silently read to the device fails.
"""
from __future__ import annotations

import gc
import json
import os

import pytest

from quant_guard import load_or_skip

torch = pytest.importorskip("torch")
pytest.importorskip("bitsandbytes")
pytest.importorskip("safetensors")

CUDA = torch.cuda.is_available()
DEVICE = "cuda" if CUDA else "cpu"
DTYPE = torch.bfloat16 if CUDA else torch.float32


def _qwen3():
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
    return Qwen3MoeForCausalLM(Qwen3MoeConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128, moe_intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, num_experts=8, num_experts_per_tok=2, decoder_sparse_step=1,
        mlp_only_layers=[], norm_topk_prob=True, tie_word_embeddings=False, max_position_embeddings=64))


def _mixtral():
    pytest.importorskip("transformers.models.mixtral", reason="this transformers has no mixtral")
    from transformers.models.mixtral.configuration_mixtral import MixtralConfig
    from transformers.models.mixtral.modeling_mixtral import MixtralForCausalLM
    return MixtralForCausalLM(MixtralConfig(
        hidden_size=64, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        num_local_experts=8, num_experts_per_tok=2, vocab_size=128))


# How each family's released checkpoints spell one expert's projections, and its router, on disk.
_SPELLING = {
    "qwen3_moe": ("mlp.experts.", ("gate_proj", "up_proj", "down_proj"), None),
    "mixtral": ("block_sparse_moe.experts.", ("w1", "w3", "w2"), "block_sparse_moe.gate.weight"),
}


def _write_per_expert(model, d):
    """Save ``model`` per expert, as the family's released checkpoints are spelled (the fused tree split back: gate first)."""
    from safetensors.torch import save_file
    container, (g, u, dn), router = _SPELLING[model.config.model_type]
    new = {}
    for k, v in model.state_dict().items():
        if k.endswith("mlp.experts.gate_up_proj"):
            base = k[: -len("mlp.experts.gate_up_proj")] + container
            for e in range(v.shape[0]):
                gate, up = v[e].chunk(2, dim=0)
                new[f"{base}{e}.{g}.weight"], new[f"{base}{e}.{u}.weight"] = gate, up
        elif k.endswith("mlp.experts.down_proj"):
            base = k[: -len("mlp.experts.down_proj")] + container
            for e in range(v.shape[0]):
                new[f"{base}{e}.{dn}.weight"] = v[e]
        elif router is not None and k.endswith("mlp.gate.weight"):
            new[k[: -len("mlp.gate.weight")] + router] = v
        else:
            new[k] = v
    new = {k: v.to(DTYPE).contiguous().clone() for k, v in new.items()}
    save_file(new, os.path.join(d, "model.safetensors"))
    json.dump({"weight_map": {k: "model.safetensors" for k in new}}, open(os.path.join(d, "model.safetensors.index.json"), "w"))
    model.config.save_pretrained(d)


def _flatten(value, key, out):
    """Tensors under ``key`` (nested dicts and sequences walked, e.g. a module's extra state); other leaves kept as they are."""
    if isinstance(value, torch.Tensor):
        out[key.rstrip(".")] = value.detach().to("cpu")
    elif isinstance(value, dict):
        for k, v in value.items():
            _flatten(v, f"{key}{k}.", out)
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            _flatten(v, f"{key}{i}.", out)
    else:
        out[key.rstrip(".")] = value


def _stored_experts(model):
    """Every 4-bit expert stack's stored tensors (packed bytes, absmax, and any double-quant state), on the host.

    Read through ``state_dict()``: under offload the module holds 0-element placeholders between forwards, and the offload
    handle's state-dict hook substitutes its host copies, which are the bytes the device stages."""
    from experts4bit_qlora import ExpertsNbit
    out = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, ExpertsNbit):
            continue
        _flatten(mod.state_dict(), f"{name}.", out)
        for attr, val in vars(mod).items():                   # plain tensor attributes (not registered)
            if isinstance(val, torch.Tensor) and val.numel():
                out.setdefault(f"{name}.{attr}", val.detach().to("cpu"))
    empty = [k for k, v in out.items() if isinstance(v, torch.Tensor) and not v.numel()]
    assert not empty, empty
    return out


def _load(path, monkeypatch, *, stage, offload, dq):
    import experts4bit_qlora.arch.moe_conventions as conv
    import experts4bit_qlora.loader as loader
    seen = []
    real_fuse, real_stack = conv.fuse_experts, conv.stack_experts

    def fuse(gate, up, down):
        seen.append({t.device.type for t in (*gate, *up, *down)})
        return real_fuse(gate, up, down)

    def stack(up, down):
        seen.append({t.device.type for t in (*up, *down)})
        return real_stack(up, down)

    monkeypatch.setattr(conv, "fuse_experts", fuse)
    monkeypatch.setattr(conv, "stack_experts", stack)
    if stage is not None:
        monkeypatch.setattr(loader, "_stage_expert_rows_on_host", lambda device, offload, s=stage: s)
    torch.manual_seed(0)
    model, _ = load_or_skip(path, DEVICE, DTYPE, r=4, alpha=8, offload=offload, pin=False, quant_type="nf4")
    if dq:
        from experts4bit_qlora.absmax_dq import compress_expert_absmax_
        assert compress_expert_absmax_(model) > 0
    got = _stored_experts(model)
    del model
    gc.collect()
    if CUDA:
        torch.cuda.empty_cache()
    monkeypatch.undo()
    return got, seen


def _assert_same_bits(a, b):
    assert a and a.keys() == b.keys(), (sorted(a)[:4], sorted(b)[:4])
    for k in a:
        if not isinstance(a[k], torch.Tensor):
            assert a[k] == b[k], k
            continue
        assert a[k].dtype == b[k].dtype and a[k].shape == b[k].shape, k
        assert torch.equal(a[k].view(torch.uint8) if a[k].dtype.is_floating_point else a[k],
                           b[k].view(torch.uint8) if b[k].dtype.is_floating_point else b[k]), k


@pytest.fixture(params=[_qwen3, _mixtral], ids=["qwen3_moe", "mixtral"])
def ckpt(request, tmp_path):
    torch.manual_seed(0)
    _write_per_expert(request.param().to(DTYPE), str(tmp_path))
    return str(tmp_path)


def test_staging_is_the_cuda_offload_path_only():
    from experts4bit_qlora.loader import _stage_expert_rows_on_host as stage
    assert stage("cuda", True) and stage(torch.device("cuda", 0), True)
    assert not stage("cuda", False) and not stage("cpu", True) and not stage("cpu", False)


@pytest.mark.skipif(not CUDA, reason="the offload staging is a CUDA path")
def test_offload_load_staged_and_direct_store_the_same_bits(ckpt, monkeypatch):
    staged, seen_s = _load(ckpt, monkeypatch, stage=None, offload=True, dq=False)     # the default: staged
    direct, seen_d = _load(ckpt, monkeypatch, stage=False, offload=True, dq=False)
    assert seen_s and all(s == {"cpu"} for s in seen_s), seen_s
    assert seen_d and all(s == {"cuda"} for s in seen_d), seen_d
    _assert_same_bits(staged, direct)


@pytest.mark.parametrize("dq", [False, True], ids=["fp32-absmax", "double-quant-absmax"])
def test_resident_load_staged_and_direct_store_the_same_bits(ckpt, monkeypatch, dq):
    direct, seen_d = _load(ckpt, monkeypatch, stage=None, offload=False, dq=dq)       # the default: direct
    staged, seen_s = _load(ckpt, monkeypatch, stage=True, offload=False, dq=dq)
    assert seen_d and all(s == {DEVICE} for s in seen_d), seen_d
    assert seen_s and all(s == {"cpu"} for s in seen_s), seen_s
    _assert_same_bits(staged, direct)
