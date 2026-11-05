"""PGX Z3: bytes per expert from safetensors headers, and the registered K table (CPU, no network)."""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bench", "pgx"))
import pgx_bytes as pb  # noqa: E402

TABLE = os.path.join(ROOT, "bench", "pgx", "bytes_per_expert.json")


def _hdr(layers=48, experts=128, extra=None):
    """A synthetic FP8-shaped header: three projections plus their block scales, and one non-expert tensor."""
    h = {"__metadata__": {"format": "pt"}, "model.embed_tokens.weight": {"dtype": "BF16", "shape": [151936, 2048]}}
    for L in range(layers):
        for e in range(experts):
            p = f"model.layers.{L}.mlp.experts.{e}."
            h[p + "gate_proj.weight"] = {"dtype": "F8_E4M3", "shape": [768, 2048]}
            h[p + "up_proj.weight"] = {"dtype": "F8_E4M3", "shape": [768, 2048]}
            h[p + "down_proj.weight"] = {"dtype": "F8_E4M3", "shape": [2048, 768]}
            h[p + "gate_proj.weight_scale_inv"] = {"dtype": "F32", "shape": [6, 16]}
            h[p + "up_proj.weight_scale_inv"] = {"dtype": "F32", "shape": [6, 16]}
            h[p + "down_proj.weight_scale_inv"] = {"dtype": "F32", "shape": [16, 6]}
    h.update(extra or {})
    return h


def test_expert_bytes_sum_every_expert_tensor_and_nothing_else():
    per = pb.expert_bytes({"a.safetensors": _hdr()})
    assert pb.uniform(per) == 3 * 768 * 2048 + 3 * 96 * 4 == 4_719_744


def test_a_partial_or_uneven_checkpoint_is_refused():
    with pytest.raises(SystemExit):
        pb.uniform(pb.expert_bytes({"a": _hdr(layers=47)}))
    odd = {"model.layers.0.mlp.experts.0.gate_proj.bias": {"dtype": "F32", "shape": [768]}}
    with pytest.raises(SystemExit):
        pb.uniform(pb.expert_bytes({"a": _hdr(extra=odd)}))


def test_k_is_the_floor_of_the_decimal_budget_over_48_experts_bytes():
    assert pb.k_for(4, 4_719_744) == 17 and pb.k_for(8, 2_654_208) == 62
    assert pb.k_for(12, 10**9 // 4) == 1          # the budget is exactly one expert per layer


def test_the_committed_table_is_the_registered_k_table():
    with open(TABLE, encoding="utf-8") as f:
        rec = json.load(f)
    assert rec["fp8"]["bytes_per_expert"] == 4_719_744
    assert rec["int4"]["bytes_per_expert"] == 2_470_912
    assert rec["int4"]["bytes_per_expert_without_g_idx"] == 2_451_456
    want = {"S8": {"4": 17, "8": 35, "12": 52}, "V8": {"4": 17, "8": 35, "12": 52},
            "S4": {"4": 33, "8": 67, "12": 101}, "E-hyb": {"4": 31, "8": 62, "12": 94}}
    assert rec["k_table"] == want == pb.k_table(rec)
    # desc_act=false: dropping the trivial g_idx moves no K
    assert all(pb.k_for(b, rec["int4"]["bytes_per_expert_without_g_idx"]) == want["S4"][str(b)]
               for b in pb.BUDGETS_GB)
    for fmt, (repo, rev) in pb.CHECKPOINTS.items():
        assert (rec[fmt]["repo"], rec[fmt]["revision"]) == (repo, rev)
