# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane P110's measurement (``bench/p110/p110_box.py``) on CPU, before any GPU is rented.

``measure()`` runs on a tiny Qwen3-MoE with the decode attention stood in by SDPA over the pool's own fp8 bytes, for
every arm whose path runs on CPU: R, rep, the floor (half, chunk, rev) and D. P and the mutant decode through the
bucketed path, which needs the fused KV append (a GPU kernel); the proving rental runs them on the card.
- every arm scores every window, and the reference repeats bit for bit;
- the decode attention is counted exactly (``half``: two calls per layer per step);
- the half-batch and reversed-slot arms map their rows back to the right windows (they match R);
- the grouping switch is in force during D's pass and restored afterwards;
- the scale mutant reaches the attention and moves the scores.
"""
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
LANES = [ROOT / "bench" / d for d in ("p110", "p108", "p97")]


def _load():
    for d in LANES:
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
    spec = importlib.util.spec_from_file_location("p110_box", LANES[0] / "p110_box.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tiny():
    from transformers import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM
    cfg = Qwen3MoeConfig(vocab_size=256, hidden_size=128, intermediate_size=128, moe_intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=32, num_experts=4, num_experts_per_tok=2,
                         max_position_embeddings=512, decoder_sparse_step=1, mlp_only_layers=[])
    torch.manual_seed(0)
    return Qwen3MoeForCausalLM(cfg).eval()


P, C, GROUP = 24, 6, 4
CPU_ARMS = ("R", "rep", "half", "chunk", "rev", "D")


@pytest.fixture(scope="module")
def run():
    pytest.importorskip("transformers.models.qwen3_moe", reason="needs transformers with Qwen3-MoE")
    box = _load()
    g = torch.Generator().manual_seed(1)
    windows = [torch.randint(0, 256, (P + C,), generator=g).tolist() for _ in range(2 * GROUP)]
    model = _tiny()
    rec = box.measure(model, windows, prompt=P, cont=C, chunk=P, floor_chunk=8, group=GROUP, device="cpu", stand_in=True,
                      arms=CPU_ARMS)
    return box, model, windows, rec


def test_every_arm_scores_every_window_and_r_repeats(run):
    _box, _m, windows, rec = run
    per = rec["per_window"]
    assert [len(per[a]) for a in CPU_ARMS] == [8, GROUP, 8, 8, 8, 8]
    assert rec["rep_identical"] is True and rec["groups"] == 2 and rec["layers"] == 2


def test_decode_attention_is_counted_exactly(run):
    _box, _m, _w, rec = run
    for arm in CPU_ARMS:
        for e in rec["engagement"][arm]:
            assert e["decode_calls"] == (C - 1) * 2 * (2 if arm == "half" else 1), (arm, e)


def test_the_floor_and_d_map_their_rows_back_to_the_right_windows(run):
    _box, _m, _w, rec = run
    per = rec["per_window"]
    r = {x["window"]: x["nll"] for x in per["R"]}
    for arm in ("half", "rev", "D"):                    # on this HF MoE the grouping switch changes nothing: same rows
        for x in per[arm]:
            assert abs(x["nll"] - r[x["window"]]) < 1e-4, (arm, x)
            assert x["argmax_agree"] == 1.0
    assert all(abs(x["nll"] - r[x["window"]]) < 5e-2 for x in per["chunk"])


def test_the_grouping_switch_is_in_force_during_d_and_restored(run):
    _box, _m, _w, rec = run
    from experts4bit_qlora.engines import hot_residency as hr
    assert all(e["grouping_flags_in_pass"]["device_grouping"] for e in rec["engagement"]["D"])
    assert not any(e["grouping_flags_in_pass"]["device_grouping"] for a in ("R", "half", "rev") for e in rec["engagement"][a])
    assert hr.DEVICE_GROUPING[0] is False and hr.FORCE_SINGLETON_GROUPS[0] is False


def test_the_scale_mutant_reaches_the_attention(run):
    box, model, windows, _rec = run
    ws = windows[:GROUP]
    ref, _ = box.paged_pass(model, ws, P, C, P, "cpu", stand_in=True)
    mut, e = box.paged_pass(model, ws, P, C, P, "cpu", stand_in=True, mutant="scale")
    assert e["decode_calls"] == (C - 1) * 2
    moved = max((a[1:] - b[1:]).abs().max().item() for a, b in zip(ref, mut))   # decode positions only (prefill untouched)
    assert moved > 1e-3
    assert all(torch.equal(a[0], b[0]) for a, b in zip(ref, mut))


def test_the_padded_arm_names_its_requirement_on_cpu(run):
    box, model, windows, _rec = run
    with pytest.raises(RuntimeError, match="fused KV append"):
        box.paged_pass(model, windows[:GROUP], P, C, P, "cpu", stand_in=True, device_grouping=True, padded=True)
