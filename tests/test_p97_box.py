# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane P97's measurement (``bench/p97/p97_box.py``) run whole on CPU, before any GPU is rented: the reference, the
paged runner, the per-step comparison and the engagement counts, on tiny random Qwen3.5-MoE models with the
rehearsal stand-in for the fp8 decode kernel.

A hybrid ([linear, attention, linear, linear]) and an all-attention control go through ``measure()``. Each must count
exactly the kernel calls and linear-state stores its shape implies -- the counts the reducer holds a reading to --
the hybrid's pooled state at the layer before its first attention layer must match transformers' own cache (the
lane's state gate, in miniature), and its KL must stay within 2x the control's. A count that came back zero, a harness
that compared the reference with itself (a control KL of zero), or a mutant pass (decode write-back rotated by one
slot) that the state gate would have let through fails here.
"""
import importlib.util
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers.models.qwen3_5_moe", reason="needs transformers with Qwen3.5-MoE")

LIN, ATT = "linear_attention", "full_attention"
BOX = Path(__file__).resolve().parents[1] / "bench" / "p97" / "p97_box.py"


def _box():
    spec = importlib.util.spec_from_file_location("p97_box", BOX)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _model(layer_types):
    from transformers import Qwen3_5MoeTextConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM
    cfg = Qwen3_5MoeTextConfig(vocab_size=128, hidden_size=64, num_hidden_layers=len(layer_types), num_attention_heads=4,
                               num_key_value_heads=2, head_dim=32, moe_intermediate_size=32,
                               shared_expert_intermediate_size=32, num_experts=4, num_experts_per_tok=2,
                               linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
                               linear_value_head_dim=16, linear_conv_kernel_dim=4, layer_types=list(layer_types),
                               max_position_embeddings=256)
    torch.manual_seed(0)
    return Qwen3_5MoeForCausalLM(cfg).eval()


def _windows(n, length):
    g = torch.Generator().manual_seed(5)
    return [torch.randint(0, 128, (length,), generator=g).tolist() for _ in range(n)]


def _engagement(rec, layer_types, n, prompt_chunks, cont):
    e = rec["engagement"]
    attn = [i for i, t in enumerate(layer_types) if t == ATT]
    lin = [i for i, t in enumerate(layer_types) if t == LIN]
    assert e["attn_layers"] == attn and e["linear_layers"] == lin
    assert e["kv_pool_layers"] == len(attn)
    assert e["decode_attention_calls"] == e["expected_decode_attention_calls"] == (cont - 1) * len(attn)
    assert e["decode_attention_pool_layers"] == list(range(len(attn)))
    assert e["linear_state_stores"] == e["expected_linear_state_stores"] == len(lin) * (n * prompt_chunks + cont - 1)
    assert e["linear_layers_with_state"] == lin == e["linear_state_store_layers"]
    assert e["linear_state"] is bool(lin)
    assert rec["rehearsal"] == {"stand_in_attention": True}
    assert rec["steps"] == n * cont


def test_the_p97_measurement_counts_its_engagement_and_holds_the_hybrid_to_its_control():
    box = _box()
    P, C, CHUNK, N = 24, 6, 10, 3                        # 24 = 10 + 10 + 4: a short last chunk
    out = {}
    for name, lt in (("hybrid", [LIN, ATT, LIN, LIN]), ("control", [ATT] * 4)):
        out[name] = box.measure(_model(lt), _windows(N, P + C), prompt=P, cont=C, chunk=CHUNK, device="cpu",
                                stand_in_attention=True)
        _engagement(out[name], lt, N, 3, C)
    h, c = out["hybrid"], out["control"]
    print({k: (h[k], c[k]) for k in ("mean_kl", "argmax_agree", "mean_d_nll", "prefill_max_abs_logprob_diff")})
    # the control's only error is the fp8 KV: a zero here would mean the harness compared the reference with itself
    assert c["mean_kl"] > 0
    assert h["mean_kl"] <= 2 * c["mean_kl"], (h["mean_kl"], c["mean_kl"])
    assert h["prefill_max_abs_logprob_diff"] < 1e-4      # chunked prefill with carried state vs one shot, no fp8 read
    # tiny random models sit near argmax ties, so agreement is held loosely; the lane's rule holds it to the control
    assert h["argmax_agree"] >= 0.8 and c["argmax_agree"] >= 0.8
    # G1's quantity: the linear layer before the first attention layer sees the same tokens on both paths, so its pooled
    # state matches transformers' cache to arithmetic precision; layers after attention carry the fp8 KV's effect
    assert c["state"] is None and c["mutant"] is None
    st = h["state"]
    assert st["pre_attention_linear_layers"] == [0] and sorted(st["rel_err"], key=int) == ["0", "2", "3"]
    assert st["pre_attention_max_rel_err"] < 1e-5, st
    assert st["all_linear_max_rel_err"] > st["pre_attention_max_rel_err"]
    # the mutant (decode write-back rotated by one slot) must fail the state gate and move the model
    m = h["mutant"]
    assert m["linear_state_stores"] == h["engagement"]["linear_state_stores"] and m["steps"] == h["steps"]
    assert m["state"]["pre_attention_max_rel_err"] > 0.5, m["state"]
    assert m["mean_kl"] > 10 * h["mean_kl"], (m["mean_kl"], h["mean_kl"])
