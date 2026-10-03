# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane P107's quality instrument on CPU (e4b#960): ``bench/p107/p107_box.py``'s ``served_prefill_nll``.

It scores a window through the paged PREFILL path in 512-token chunks, the way a served prefill runs. On a tiny Qwen3
that must equal the same window scored in ONE non-paged forward, for both attention routes, at the lane's registered
shape (a 512-token prompt, then 2,048 scored positions, 512-token chunks). The scorer must also leave nothing behind:
no staged K/V, the context unbound, the runner's mode switched back -- on success and when a forward raises.
"""
import copy
import importlib.util
import pathlib

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from experts4bit_qlora.engines import paged_attention as pa  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
VOCAB = 96


def _box():
    spec = importlib.util.spec_from_file_location("p107_box", REPO / "bench" / "p107" / "p107_box.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _models():
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
    cfg = Qwen3Config(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=8,
                      num_key_value_heads=2, head_dim=16, vocab_size=VOCAB, max_position_embeddings=4096)
    torch.manual_seed(0)
    ref = Qwen3ForCausalLM(cfg).eval()
    paged = copy.deepcopy(ref)
    pa.register(paged)
    assert ref.config._attn_implementation != pa.IMPL_NAME          # the reference stays non-paged
    return ref, paged


def _ids(box):
    g = torch.Generator().manual_seed(1)
    return torch.randint(0, VOCAB, (box.PROMPT_LEN + box.STEPS + 40,), generator=g)


def _reference_nll(ref, ids, box):
    end = box.PROMPT_LEN + box.STEPS
    with torch.no_grad():
        lg = torch.log_softmax(ref(input_ids=ids[None, :end]).logits[0, box.PROMPT_LEN:end].float(), -1)
    tgt = ids[box.PROMPT_LEN + 1:end + 1]
    return float(-lg[torch.arange(box.STEPS), tgt].mean())


@pytest.mark.parametrize("route", ["math", "flash"])
def test_the_chunked_served_prefill_equals_one_whole_forward(monkeypatch, route):
    box = _box()
    ref, paged = _models()
    ids = _ids(box)
    want = _reference_nll(ref, ids, box)
    monkeypatch.setenv("E4B_PAGED_PREFILL_ATTN", route)
    calls, ctx = [], pa.PagedAttentionContext(kv=None, slots=[3])
    nll, n = box.served_prefill_nll(torch, paged, ctx, pa.set_context, calls.append, ids, "cpu")
    assert n == box.STEPS
    assert abs(nll - want) < 1e-4, (route, nll, want)
    assert calls == [True, False] and ctx.staging == {} and ctx.mode == "decode" and pa._CTX is None


def test_a_forward_that_raises_leaves_nothing_behind():
    box = _box()
    _ref, paged = _models()
    seen = []

    def failing(**kw):
        seen.append(int(kw["position_ids"][0, 0]))
        if len(seen) == 2:
            raise RuntimeError("boom")
        return paged(**kw)

    calls, ctx = [], pa.PagedAttentionContext(kv=None, slots=[0])
    with pytest.raises(RuntimeError, match="boom"):
        box.served_prefill_nll(torch, failing, ctx, pa.set_context, calls.append, _ids(box), "cpu")
    assert seen == [0, box.CHUNK]                                     # the first chunk staged K/V before the raise
    assert calls == [True, False] and ctx.staging == {} and ctx.mode == "decode" and pa._CTX is None
