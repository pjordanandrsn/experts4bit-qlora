# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""CUDA-graph capture of a QUANTISED MoE, on both serving engines (#527).

``test_capture.py`` builds every GPU case from a dense tiny Llama: no
``Experts4bit``, no engine. So the capture machinery was covered only on a model
the product does not serve, and the two engines' opposite answers went
untested:

* ``enable_pipelined_residency`` dispatches experts by device-side id, so the
  decode step is capturable, and replay must match eager decode;
* ``enable_fast``'s grouped path sizes its launch from host-side per-expert
  counts, so it cannot be captured. It now refuses by name instead of failing
  inside ``torch.bincount`` three frames down.

The gate is ``probe_capture``: token-stream equality against eager decode
under the same argmax rule, not "it ran".
"""
import json
import os
import types

import pytest
import torch

from quant_guard import load_or_skip

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

E, TOP_K = 8, 2


# ---------------------------------------------------- the refusal, anywhere --

def test_the_fast_path_refuses_inside_a_capture_by_name(monkeypatch):
    """Runs on CPU CI: the check is a decision over two facts (a CUDA tensor,
    a capturing stream), so it is pinned without a GPU."""
    from experts4bit_qlora.engines import fast

    cuda_like = types.SimpleNamespace(is_cuda=True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="enable_pipelined_residency"):
        fast._refuse_under_capture(cuda_like)
    # not capturing, or a CPU tensor: nothing to refuse
    fast._refuse_under_capture(types.SimpleNamespace(is_cuda=False))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    fast._refuse_under_capture(cuda_like)


def test_both_grouped_inference_forwards_call_the_refusal():
    """A refusal wired into one forward and not the other is how the
    three-frames-deep error comes back. Structural, so it runs on CPU."""
    import inspect

    from experts4bit_qlora.engines import fast

    for fn in (fast.fused_experts_forward, fast.fused_experts_lora_forward):
        src = inspect.getsource(fn)
        assert "_refuse_under_capture(hidden_states)" in src, fn.__name__
        # before any fallback: the reference path cannot be captured either
        assert src.index("_refuse_under_capture") < src.index("_e4b_fast_ref"), fn.__name__


# ------------------------------------------------------- a tiny real model --

def _qwen3_moe_checkpoint(d):
    """A random 2-layer Qwen3-MoE in its on-disk (per-expert) layout, at
    sizes the NF4 blocksize and the grouped kernel accept."""
    pytest.importorskip("transformers.models.qwen3_moe")
    from safetensors.torch import save_file
    from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM

    torch.manual_seed(0)
    model = Qwen3MoeForCausalLM(Qwen3MoeConfig(
        hidden_size=128, intermediate_size=256, moe_intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=32, num_experts=E, num_experts_per_tok=TOP_K, vocab_size=256,
        decoder_sparse_step=1, mlp_only_layers=[]))
    new = {}
    for k, v in model.state_dict().items():
        if k.endswith("experts.gate_up_proj"):
            base = k[: -len("gate_up_proj")]
            for e in range(v.shape[0]):
                g, u = v[e].chunk(2, dim=0)
                new[f"{base}{e}.gate_proj.weight"] = g
                new[f"{base}{e}.up_proj.weight"] = u
        elif k.endswith("experts.down_proj"):
            base = k[: -len("down_proj")]
            for e in range(v.shape[0]):
                new[f"{base}{e}.down_proj.weight"] = v[e]
        else:
            new[k] = v
    new = {k: v.to(torch.bfloat16).contiguous().clone() for k, v in new.items()}
    save_file(new, os.path.join(d, "model.safetensors"))
    with open(os.path.join(d, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": {k: "model.safetensors" for k in new}}, f)
    model.config.save_pretrained(d)


def _load(tmp_path):
    pytest.importorskip("nf4_grouped", reason="needs grouped-nf4-gemm (the [fast] extra)")
    _qwen3_moe_checkpoint(str(tmp_path))
    model, _ = load_or_skip(str(tmp_path), "cuda", torch.bfloat16, r=4, alpha=8)
    return model.eval()


def _n_moe_modules(model):
    from experts4bit_qlora.verify import verify_moe_4bit
    rep = verify_moe_4bit(model)
    assert rep["n_quantized"] > 0, "the fixture loaded no quantised expert layers"
    return rep["n_quantized"] + rep["n_unquantized"]


IDS = [[1, 17, 42, 99, 7, 128, 3, 200]]


@requires_cuda
def test_pipelined_engine_captures_a_quantised_moe_and_replays_as_eager(tmp_path):
    from experts4bit_qlora import enable_pipelined_residency, probe_capture

    model = _load(tmp_path)
    n = _n_moe_modules(model)
    patched = enable_pipelined_residency(model, [[] for _ in range(n)],
                                         device="cuda", k_slots=TOP_K)
    assert patched == n, f"the engine patched {patched} of {n} MoE modules"
    rep = probe_capture(model, torch.tensor(IDS, device="cuda"), max_new_tokens=12)
    assert rep["captured"], f"capture failed on the capturable engine: {rep['error']}"
    # A replay that differs from eager only on an exact tie is not a capture
    # defect; probe_capture decides that by teacher forcing, not by the gap.
    tied = rep.get("teacher_forced", {}).get("max_abs_delta") == 0.0
    assert rep["matches_eager"] or tied, (
        f"replay diverged from eager:\n  eager   ={rep['eager']}\n  replayed={rep['replayed']}\n"
        f"  teacher_forced={rep.get('teacher_forced')}")


@requires_cuda
def test_enable_fast_refuses_capture_by_name_and_still_decodes_eagerly(tmp_path):
    from experts4bit_qlora import enable_fast, probe_capture

    model = _load(tmp_path)
    patched = enable_fast(model)
    assert patched > 0, "enable_fast patched nothing; the refusal below would be untested"
    rep = probe_capture(model, torch.tensor(IDS, device="cuda"), max_new_tokens=6)
    assert rep["eager"] is not None and len(rep["eager"]) == 6, rep
    assert not rep["captured"], "the grouped path captured, which #527 says it cannot"
    assert "cannot run under CUDA-graph capture" in (rep["error"] or ""), rep["error"]
    # and the context survived the refused capture
    assert torch.ones(4, device="cuda").sum().item() == 4.0
