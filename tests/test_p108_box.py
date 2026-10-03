# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane P108's measurement (``bench/p108/p108_box.py``) on CPU, before any GPU is rented.

``measure()`` runs on a tiny Gemma-4 MoE (five sliding layers at window 16, one full layer, so the window binds on
40-token prompts), with the decode attention stood in by SDPA over the pool's own fp8 bytes:
- every arm scores every window;
- the reference repeats bit for bit;
- the subject's decode attention is counted exactly: every layer every step, the sliding layers at the model's window;
- the mutants reach the attention (the scale mutant halves the softmax scale; the window mutant passes window 0);
- the box's record reduces with the lane's reducer.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

LANE = Path(__file__).resolve().parents[1] / "bench" / "p108"
P97 = Path(__file__).resolve().parents[1] / "bench" / "p97"


def _load(name, path):
    for d in (str(LANE), str(P97)):
        if d not in sys.path:
            sys.path.insert(0, d)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tiny(seed=0):
    from transformers import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM
    cfg = Gemma4TextConfig(vocab_size=256, hidden_size=128, intermediate_size=128, num_hidden_layers=6,
                           num_attention_heads=4, num_key_value_heads=2, head_dim=32, global_head_dim=64,
                           num_global_key_value_heads=1, layer_types=["sliding_attention"] * 5 + ["full_attention"],
                           sliding_window=16, enable_moe_block=True, num_experts=4, top_k_experts=2,
                           moe_intermediate_size=64, attention_k_eq_v=True, hidden_size_per_layer_input=0,
                           vocab_size_per_layer_input=256, max_position_embeddings=512)
    torch.manual_seed(seed)
    return Gemma4ForCausalLM(cfg).eval()


@pytest.fixture(scope="module")
def record():
    pytest.importorskip("transformers.models.gemma4", reason="needs transformers with Gemma-4")
    pytest.importorskip("fp8_kv", reason="needs grouped-nf4-gemm's fp8 KV")
    box = _load("p108_box", LANE / "p108_box.py")
    g = torch.Generator().manual_seed(1)
    windows = [torch.randint(0, 256, (40 + 8,), generator=g).tolist() for _ in range(4)]
    seen = []
    inner = box.stand_in_attention

    def spy(self, layer, q, slots=None, sm_scale=None, window=None, **kw):
        seen.append((int(layer), window, sm_scale))
        return inner(self, layer, q, slots=slots, sm_scale=sm_scale, window=window, **kw)

    box.stand_in_attention = spy
    try:
        rec = box.measure(_tiny(), windows, prompt=40, cont=8, chunk=16, group=2, device="cpu", stand_in=True)
    finally:
        box.stand_in_attention = inner
    return rec, seen


def test_every_arm_scores_every_window_and_the_reference_repeats(record):
    rec, _ = record
    per = rec["per_window"]
    assert {a: len(v) for a, v in per.items()} == {"R": 4, "rep": 2, "oneshot": 4, "chunk": 4, "batch": 4, "paged": 4,
                                                   "mutant_scale": 4, "mutant_window": 4}
    assert rec["engagement"]["rep_identical"] is True
    assert all(x["kl"] == 0.0 for x in per["rep"])
    assert all("kl" in x and x["kl"] >= -1e-6 for arm in ("oneshot", "chunk", "batch", "paged") for x in per[arm])


def test_the_subjects_decode_attention_is_counted_with_its_window(record):
    rec, seen = record
    e = rec["engagement"]
    assert e["groups"] == 2 and e["expected_calls"] == 2 * 7 * 6 and e["expected_sliding_calls"] == 2 * 7 * 5
    assert e["calls"] == e["expected_calls"]
    assert e["by_window"] == {"16": e["expected_sliding_calls"], "0": e["expected_calls"] - e["expected_sliding_calls"]}
    # three paged arms (subject, scale mutant, window mutant) went through the attention
    assert len(seen) == 3 * e["expected_calls"]
    scales = sorted({s for _l, _w, s in seen})
    assert len(scales) == 2 and scales[0] == pytest.approx(scales[1] / 2)
    assert {w for layer, w, _s in seen if layer < 5} == {16, 0}         # the subject's 16, the window mutant's 0


def test_the_record_reduces(record, tmp_path):
    rec, _ = record
    red = _load("p108_reduce", LANE / "p108_reduce.py")
    rec = dict(rec, loaded_commit=red.REV)
    (tmp_path / "box.json").write_text(json.dumps(rec))
    (tmp_path / "summary.txt").write_text("premise ok\n")
    out = red.reduce(tmp_path)
    assert out["verdict"] == "VOID" and any("windows, registered" in r for r in out["reasons"])   # 4 of 32 windows
