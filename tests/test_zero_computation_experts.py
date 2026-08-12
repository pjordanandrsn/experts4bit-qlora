# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Identity ("zero-computation") experts must be refused, not half-loaded.

LongCat-Flash routes over `n_routed_experts + zero_expert_num` slots; indices at or
above the routed count pass the token through `nn.Identity` scaled by its router
weight instead of through a SwiGLU expert. Those surplus slots carry `gate_up` rows
the forward never reads and no `down_proj` at all.

Before this refusal the per-expert reader consumed experts `0..n_routed-1`, left the
surplus `gate_proj`/`up_proj` keys orphaned, and the generic weight walk then died on
`get_submodule(".../experts.10")` with `AttributeError: ExpertsLoRA has no attribute
'10'` — an error that names neither the architecture nor the actual limitation. This
runs on CPU because the gate is config-only, which is the point: it fires before any
weight is read, so CI catches a regression without a GPU.
"""
import pytest

pytest.importorskip("torch")


def _load(config):
    import torch
    from experts4bit_qlora.loader import load_moe_4bit_streaming
    import unittest.mock as m
    with m.patch("transformers.AutoConfig.from_pretrained", return_value=config):
        return load_moe_4bit_streaming("ignored/path", "cpu", torch.bfloat16, 8, 16)


class _Cfg:
    def __init__(self, **kw):
        self.model_type = "longcat_flash"
        self.__dict__.update(kw)


def test_zero_computation_experts_are_refused_with_the_counts_named():
    with pytest.raises(NotImplementedError) as e:
        _load(_Cfg(n_routed_experts=512, zero_expert_num=256))
    msg = str(e.value)
    assert "512" in msg and "256" in msg, msg
    assert "768" in msg, f"the message should name the full routed space: {msg}"
    assert "identity" in msg.lower(), msg


def test_a_model_without_identity_experts_is_not_refused_by_this_gate():
    """The gate must not fire on `zero_expert_num = 0` or a config that lacks the field.

    Arming matters: a refusal that triggers on every MoE would pass the test above while
    breaking every supported model.
    """
    for cfg in (_Cfg(n_routed_experts=64, zero_expert_num=0), _Cfg(n_routed_experts=64)):
        with pytest.raises(Exception) as e:
            _load(cfg)
        # It must fail LATER (no such checkpoint), never at this gate. Accepting any
        # exception would make this vacuous -- the first version of this test did
        # exactly that and passed on a TypeError from a wrong call signature, proving
        # nothing about the gate at all.
        assert not isinstance(e.value, NotImplementedError), (
            f"gate fired on a model with no identity experts: {e.value}")
        assert "identity" not in str(e.value).lower(), e.value
