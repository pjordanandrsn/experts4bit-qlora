# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Q235's layer-streamed bf16 reference (``bench/q235/instrument/layer_stream_ref.py``) computes what transformers' resident model
computes: on a small random Qwen3-MoE saved by ``save_pretrained``, the streamed forward and the resident forward agree bit for bit
(per-layer outputs, router top-k, loss), and both match the model loaded by ``from_pretrained`` (its per-layer hidden states, its
router's top-k and its loss) on padded, variable-length rows."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("safetensors")

ROOT = Path(__file__).resolve().parents[1]


def _tool():
    spec = importlib.util.spec_from_file_location("layer_stream_ref", ROOT / "bench" / "q235" / "instrument" / "layer_stream_ref.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
    cfg = Qwen3MoeConfig(vocab_size=256, hidden_size=64, intermediate_size=128, moe_intermediate_size=32, num_hidden_layers=3,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16, num_experts=8, num_experts_per_tok=2,
                         decoder_sparse_step=1, mlp_only_layers=[], norm_topk_prob=True, tie_word_embeddings=False,
                         max_position_embeddings=128)
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("q3tiny")
    Qwen3MoeForCausalLM(cfg).to(torch.bfloat16).save_pretrained(path)
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(3, 256, (2, 24), generator=g)
    mask = torch.ones(2, 24, dtype=torch.long)
    mask[1, 17:] = 0                                           # one right-padded row
    ids = ids.masked_fill(mask == 0, 0)
    return str(path), ids, mask, ids.masked_fill(mask == 0, -100)


def test_streamed_equals_resident_bit_for_bit(tiny):
    path, ids, mask, labels = tiny
    t = _tool()
    s = t.run(path, ids, mask, labels, "cpu")
    r = t.run(path, ids, mask, labels, "cpu", resident=True)
    assert s["layer_sha256"] == r["layer_sha256"] and s["loss"] == r["loss"]
    assert all(torch.equal(a, b) for a, b in zip(s["router_topk"], r["router_topk"]))


def test_the_reference_matches_transformers_resident_model(tiny):
    from transformers import AutoModelForCausalLM
    path, ids, mask, labels = tiny
    s = _tool().run(path, ids, mask, labels, "cpu")
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).eval()
    picks = []
    hooks = [lay.mlp.gate.register_forward_hook(lambda m, i, o: picks.append(o[2])) for lay in model.model.layers]
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=mask, labels=labels, output_hidden_states=True)
    for h in hooks:
        h.remove()
    assert all(torch.equal(a, b) for a, b in zip(s["router_topk"], picks))
    hs = out.hidden_states                                     # embeddings, then each layer but the last, then the normed last
    for i in range(len(model.model.layers) - 1):
        got = hs[i + 1]
        import hashlib
        assert hashlib.sha256(got.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() == s["layer_sha256"][i], i
    assert abs(float(out.loss) - s["loss"]) <= 1e-6 * max(1.0, abs(s["loss"])), (float(out.loss), s["loss"])


def test_truncated_depth_applies_the_head_after_layer_k(tiny):
    path, ids, mask, labels = tiny
    t = _tool()
    s2 = t.run(path, ids, mask, labels, "cpu", layers=2)
    full = t.run(path, ids, mask, labels, "cpu")
    assert s2["layers"] == 2 and s2["layer_sha256"] == full["layer_sha256"][:2] and s2["loss"] != full["loss"]
