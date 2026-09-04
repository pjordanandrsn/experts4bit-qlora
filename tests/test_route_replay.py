# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Matched routing: the reference forward's expert choices, recorded per
layer and position, replayed into another path scoring the same tokens.
A tiny OLMoE on the CPU pins the mechanism: replay must FORCE the recorded
choices (so a forward on different tokens routes like the recorded one),
must serve exactly the rows it was asked for and no more, and a chunked
scorer under replay must route exactly like the full forward it came
from."""
import importlib.util
import os
import sys

import pytest
import torch

transformers = pytest.importorskip("transformers")

_HERE = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location(
    "step_decomp", os.path.join(_HERE, "..", "bench", "hybrid-g9", "step_decomp.py"))


def _load_harness():
    mod = importlib.util.module_from_spec(_spec)
    sys.modules["step_decomp"] = mod
    _spec.loader.exec_module(mod)
    return mod


def _tiny_olmoe():
    from transformers import OlmoeConfig, OlmoeForCausalLM
    cfg = OlmoeConfig(vocab_size=97, hidden_size=32, intermediate_size=48,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, num_experts=4, num_experts_per_tok=2,
                      max_position_embeddings=256, eos_token_id=1)
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    return OlmoeForCausalLM(cfg).eval()


def _capture_routing(sd, model, ids):
    """Every router's chosen indices per row, as the model actually used them."""
    seen = {}
    handles = []
    for layer, m in sd._router_modules(model):
        def rec(_m, _i, out, layer=layer):
            seen.setdefault(layer, []).append(out[2].detach().clone())
        handles.append(m.register_forward_hook(rec))
    with torch.no_grad():
        model(input_ids=ids[None], use_cache=False)
    for h in handles:
        h.remove()
    return {layer: torch.cat(v) for layer, v in seen.items()}


def test_record_then_replay_forces_the_recorded_choices(tmp_path):
    sd = _load_harness()
    model = _tiny_olmoe()
    torch.manual_seed(3)
    ids_a = torch.randint(0, 97, (30,))
    ids_b = torch.randint(0, 97, (30,))
    nat_b = _capture_routing(sd, model, ids_b)
    assert sd._route_install(model, "record") == 2
    with torch.no_grad():
        model(input_ids=ids_a[None], use_cache=False)
    rec = sd._route_save(str(tmp_path / "r.pt"), {"text_sha": "abc"})
    sd._route_clear()
    assert set(rec) == {0, 1} and rec[0]["idx"].shape == (30, 2)
    # different tokens: natural routing differs from A's on some rows
    assert any(not torch.equal(nat_b[layer], rec[layer]["idx"]) for layer in rec)
    loaded, meta = sd._route_load(str(tmp_path / "r.pt"))
    assert meta["text_sha"] == "abc"
    sd._route_install(model, "replay", loaded)
    got_b = _capture_routing(sd, model, ids_b)      # routers now serve A's choices
    # the hook returns the replaced output; the model consumed the recorded
    # choices, and the capture hook (registered after) sees them too
    assert sd._ROUTE["served"] == 60 and sd._ROUTE["passed"] == 0
    for layer in rec:
        assert torch.equal(got_b[layer], rec[layer]["idx"])
    # beyond the recorded range the hook passes through and counts it
    with torch.no_grad():
        model(input_ids=ids_b[None, :5], use_cache=False)
    assert sd._ROUTE["passed"] == 10
    sd._route_clear()
    assert sd._ROUTE["mode"] is None and sd._ROUTE["handles"] == []


def test_chunked_scorer_under_replay_routes_like_the_full_forward(tmp_path):
    """Consumption counters: a chunked scorer serves the prompt, then chunks
    of the continuation, in order -- every row must line up with the full
    forward's record, and the NLL must equal the full forward's."""
    sd = _load_harness()
    model = _tiny_olmoe()
    torch.manual_seed(4)
    ids = torch.randint(0, 97, (80,))
    prompt_len, steps = 16, 40
    sd._route_install(model, "record")
    ref = sd.ppl_oracle_score_full(model, ids, prompt_len, steps, device="cpu")
    rec = sd._route_save(str(tmp_path / "r.pt"), {"text_sha": "w"})
    sd._route_clear()
    n_pos = prompt_len + steps + 1
    assert rec[0]["idx"].shape[0] == n_pos
    sd._route_install(model, "replay", rec)
    got = sd.ppl_oracle_score(model, ids, prompt_len, steps, chunk=7, device="cpu")
    # the chunked scorer feeds prompt + steps rows (it never needs the last
    # logit the full forward computes), all of them matched, none passed
    assert sd._ROUTE["passed"] == 0
    assert sd._ROUTE["served"] == 2 * (prompt_len + steps)
    assert abs(got - ref) < 1e-4, (got, ref)
    sd._route_clear()


def test_layer_diff_reference_does_not_consume_replay_rows(tmp_path):
    """The per-layer diff runs an eager reference forward while replay is
    armed; that forward must not advance the replay counters or the
    remaining decode steps replay the wrong positions (Bugbot, #365)."""
    sd = _load_harness()
    model = _tiny_olmoe()
    torch.manual_seed(5)
    ids = torch.randint(0, 97, (40,))
    sd._route_install(model, "record")
    with torch.no_grad():
        model(input_ids=ids[None], use_cache=False)
    rec = sd._route_save(str(tmp_path / "r.pt"), {})
    sd._route_clear()
    sd._route_install(model, "replay", rec)
    for layer in rec:
        sd._ROUTE["consumed"][layer] = 10
    sd._layer_diff_reference(model, ids[:25], {})
    assert all(sd._ROUTE["consumed"][layer] == 10 for layer in rec)
    assert sd._ROUTE["served"] == 0 and sd._ROUTE["passed"] == 0
    assert sd._ROUTE["mode"] == "replay"
    sd._route_clear()


class _GraniteShapedRouter(torch.nn.Module):
    """Returns (index, weights, logits) like GraniteMoeTopKRouter."""
    def __init__(self, hidden, E, k):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(E, hidden) * 0.1)
        self.k = k
    def forward(self, x):
        logits = torch.nn.functional.linear(x, self.weight)
        top, idx = logits.topk(self.k, dim=-1)
        return idx, torch.softmax(top, dim=-1), logits


def test_router_outputs_are_located_by_dtype_not_position():
    sd = _load_harness()
    r = _GraniteShapedRouter(8, 6, 2)
    x = torch.randn(5, 8)
    out = r(x)
    assert sd._router_out_positions(out) == (1, 0)
    std = (torch.randn(5, 6), torch.randn(5, 2), torch.randint(0, 6, (5, 2)))
    assert sd._router_out_positions(std) == (1, 2)
    assert sd._router_out_positions((torch.randn(5, 6),)) is None
    # record + replay through the Granite-shaped order
    sd._ROUTE.update(mode="record", store={}, consumed={}, served=0, passed=0)
    h = r.register_forward_hook(sd._route_hook(0))
    r(x)
    rec = {0: {"idx": torch.cat(sd._ROUTE["store"][0]["idx"]), "w": torch.cat(sd._ROUTE["store"][0]["w"])}}
    assert torch.equal(rec[0]["idx"], out[0]) and torch.allclose(rec[0]["w"], out[1])
    sd._ROUTE.update(mode="replay", store=rec, consumed={0: 0}, served=0, passed=0)
    y = torch.randn(5, 8)
    got = r(y)
    assert torch.equal(got[0], out[0]) and torch.allclose(got[1], out[1]), "replayed into the module's own order"
    assert got[2].shape == (5, 6) and sd._ROUTE["served"] == 5
    h.remove()
    sd._route_clear()

