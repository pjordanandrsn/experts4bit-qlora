# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Routed top-k under every family's config spelling, including Gemma-4's
``text_config.top_k_experts`` (P24-GEN-B: bake OK, every arm refused
before any lane because the key was unknown)."""
from types import SimpleNamespace

import pytest

from experts4bit_qlora.engines.int4_experts import _top_k


def _model(cfg):
    return SimpleNamespace(config=cfg)


@pytest.mark.parametrize("cfg,want", [
    (SimpleNamespace(num_experts_per_tok=8), 8),                                   # qwen3_moe / olmoe / mixtral
    (SimpleNamespace(text_config=SimpleNamespace(top_k_experts=8)), 8),            # gemma4 (multimodal wrapper)
    (SimpleNamespace(top_k_experts=4), 4),
    (SimpleNamespace(moe_top_k=2), 2),
])
def test_top_k_aliases(cfg, want):
    assert _top_k(_model(cfg)) == want


def test_unknown_key_refuses():
    with pytest.raises(RuntimeError, match="cannot read routed-experts-per-token"):
        _top_k(_model(SimpleNamespace(num_experts=128)))
