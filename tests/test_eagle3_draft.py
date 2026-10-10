# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The EAGLE-3 draft in the paged server (lane SD2, ``bench/sd2/PREREG-sd2.md`` B3, B4).

:class:`Eagle3Drafter` repeats SD1's arithmetic with a KV cache, for one speculating sequence. Its chains must equal
SD1's step-by-step reference, ``bench/sd1/sd1_eagle3.py``'s ``chain_at``, through a prompt and several verify cycles of
every accept count. The true stream follows the drafts up to the accept, as a real verify's does. :class:`AuxStates`
must copy each layer's input residual stream, what ``output_hidden_states`` reports, for prefill chunks and decode
rows. :func:`load_head` must refuse a head that is not the pinned one.
"""
import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch

from experts4bit_qlora.engines import eagle3_draft as ed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench" / "sd1"))
import sd1_eagle3 as ref  # noqa: E402

H, NH, HD, NKV, V, DV, INTER = 16, 4, 8, 2, 50, 20, 24


def _head(seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    return {"d2t": torch.randint(0, V - DV, (DV,), generator=g), "embed_tokens.weight": r(V, H), "fc.weight": r(H, 3 * H) / 4,
            "layers.0.hidden_norm.weight": 1 + 0.1 * r(H), "layers.0.input_layernorm.weight": 1 + 0.1 * r(H),
            "layers.0.mlp.down_proj.weight": r(H, INTER) / 4, "layers.0.mlp.gate_proj.weight": r(INTER, H) / 4,
            "layers.0.mlp.up_proj.weight": r(INTER, H) / 4, "layers.0.post_attention_layernorm.weight": 1 + 0.1 * r(H),
            "layers.0.self_attn.k_proj.weight": r(NKV * HD, 2 * H) / 4, "layers.0.self_attn.o_proj.weight": r(H, NH * HD) / 4,
            "layers.0.self_attn.q_proj.weight": r(NH * HD, 2 * H) / 4, "layers.0.self_attn.v_proj.weight": r(NKV * HD, 2 * H) / 4,
            "lm_head.weight": r(DV, H), "norm.weight": 1 + 0.1 * r(H)}


def _pair(k, dtype=torch.float32, max_positions=96):
    t = _head()
    geo = dict(n_heads=NH, n_kv=NKV, head_dim=HD, dtype=dtype)
    return (ed.Eagle3Drafter(t, k=k, max_positions=max_positions, device="cpu", **geo),
            ref.Eagle3Draft(t, **geo))


@pytest.mark.parametrize("k", [1, 2, 3])
def test_the_cycle_equals_the_reference_chains(k):
    """A prompt, then verify cycles at every accept count: after each, the drafter's chain equals ``chain_at`` at the
    last accepted position, over the true stream so far."""
    drafter, oracle = _pair(k)
    g = torch.Generator().manual_seed(7 + k)
    P = 9
    x = torch.randint(0, V, (P + 1,), generator=g).tolist()      # the prompt, then the first generated token x[P]
    aux = torch.randn(P, 3 * H, generator=g)                       # the target's states at positions 0..P-1
    drafts = drafter.prefill(torch.tensor(x[1:]), aux)
    assert drafts.tolist() == oracle.chain_at(torch.tensor(x), aux, P - 1, K=k)
    base = P                                                       # x[base] is emitted, in neither KV
    for cycle, a in enumerate([0, k, 1 % (k + 1), k, 0, min(2, k)]):
        n = k + 1
        # the verify: row i's argmax is the true x[base + i + 1] up to the accept; rows past it are rejected work
        g_rows = torch.randint(0, V, (n,), generator=g)
        g_rows[:a] = drafts[:a]
        if a < k and int(g_rows[a]) == int(drafts[a]):
            g_rows[a] = (int(drafts[a]) + 1) % V                    # the first mismatch is a mismatch
        aux_v = torch.randn(n, 3 * H, generator=g)
        x = x + g_rows[:a + 1].tolist()                             # the stream grows by the accepted tokens
        aux = torch.cat([aux, aux_v[:a + 1]])
        drafts = drafter.extend_and_draft(g_rows, aux_v, torch.tensor(base), torch.tensor(a))
        t = base + a
        assert drafts.tolist() == oracle.chain_at(torch.tensor(x), aux, t, K=k), f"cycle {cycle}, a = {a}"
        # ids are a coarse oracle at a 20-id vocabulary: the cache's context entries 0..t must also be the
        # reference's keys and values for the true stream, at the true positions (a shifted position or RoPE shows
        # here when the argmax happens to survive it)
        _, k_ref, v_ref, _ = oracle._qkv(torch.tensor(x[1:t + 2]), oracle.fc(aux[:t + 1]), torch.arange(t + 1))
        torch.testing.assert_close(drafter.kc[:t + 1], k_ref, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(drafter.vc[:t + 1], v_ref, rtol=1e-5, atol=1e-5)
        base = t + 1


def test_bf16_drafts_track_the_reference_closely():
    """In the served dtype the cache's and the reference's sums can round apart. Report agreement, and require most."""
    k = 3
    drafter, oracle = _pair(k, dtype=torch.bfloat16)
    g = torch.Generator().manual_seed(3)
    P = 12
    x = torch.randint(0, V, (P + 1,), generator=g).tolist()
    aux = torch.randn(P, 3 * H, generator=g)
    got = drafter.prefill(torch.tensor(x[1:]), aux).tolist()
    want = oracle.chain_at(torch.tensor(x), aux, P - 1, K=k)
    assert sum(a == b for a, b in zip(got, want)) >= k - 1


def test_prefill_refuses_a_context_past_the_cache():
    drafter, _ = _pair(2, max_positions=10)
    with pytest.raises(ValueError, match="pass the cache"):
        drafter.prefill(torch.zeros(9, dtype=torch.long), torch.zeros(9, 3 * H))


def test_a_head_missing_a_tensor_is_refused():
    t = _head()
    del t["fc.weight"]
    with pytest.raises(ValueError, match="lacks"):
        ed.Eagle3Drafter(t, k=1, max_positions=8, device="cpu")


def test_aux_layers_is_the_default_convention():
    assert ed.aux_layers(48) == (2, 24, 45)
    with pytest.raises(ValueError):
        ed.aux_layers(4)


# ------------------------------------------------------------------------------------------------ the head --

CONFIG = {"speculators_model_type": "eagle3", "norm_before_residual": True, "eagle_aux_hidden_state_layer_ids": None,
          "draft_vocab_size": 64000,
          "transformer_layer_config": {"num_hidden_layers": 1, "num_attention_heads": 32, "num_key_value_heads": 4,
                                       "head_dim": 128, "rope_theta": 10000.0, "rms_norm_eps": 1e-6, "hidden_size": 2048,
                                       "rope_scaling": None, "hidden_act": "silu", "attention_bias": False,
                                       "mlp_bias": False}}


def test_head_geometry_reads_the_pinned_config():
    assert ed.head_geometry(CONFIG) == {"n_heads": 32, "n_kv": 4, "head_dim": 128, "rope_theta": 10000.0, "eps": 1e-6,
                                        "hidden": 2048, "draft_vocab": 64000}


@pytest.mark.parametrize("edit,why", [
    (lambda c: c.update(norm_before_residual=False), "norm_before_residual"),
    (lambda c: c.update(eagle_aux_hidden_state_layer_ids=[1, 2, 3]), "eagle_aux_hidden_state_layer_ids"),
    (lambda c: c["transformer_layer_config"].update(num_hidden_layers=2), "draft layers"),
    (lambda c: c["transformer_layer_config"].update(rope_scaling={"type": "yarn"}), "rope_scaling"),
    (lambda c: c.update(speculators_model_type="eagle"), "eagle3"),
])
def test_head_geometry_refuses_what_is_not_implemented(edit, why):
    c = json.loads(json.dumps(CONFIG))
    edit(c)
    with pytest.raises(ValueError, match=why):
        ed.head_geometry(c)


def test_load_head_checks_size_and_digest_through_a_symlink(tmp_path):
    from safetensors.torch import save_file
    blob = tmp_path / "blobs" / "abc"
    blob.parent.mkdir()
    save_file({"x": torch.zeros(4)}, str(blob))
    snap = tmp_path / "snap"
    snap.mkdir()
    link = snap / "model.safetensors"
    try:
        link.symlink_to(blob)
    except OSError:
        pytest.skip("this filesystem cannot make a symlink")
    (snap / "config.json").write_text(json.dumps(CONFIG), encoding="utf-8")
    data = blob.read_bytes()
    size, digest = len(data), hashlib.sha256(data).hexdigest()
    tensors, geo = ed.load_head(str(snap), sha256=digest, size=size)
    assert list(tensors) == ["x"] and geo["n_heads"] == 32
    with pytest.raises(ValueError, match="bytes"):
        ed.load_head(str(snap), sha256=digest, size=size + 1)
    with pytest.raises(ValueError, match="sha256"):
        ed.load_head(str(snap), sha256="0" * 64, size=size)


# ------------------------------------------------------------------------------------------ the aux states --

def _tiny_qwen3_moe(layers=8):
    from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3MoeConfig(hidden_size=32, intermediate_size=48, moe_intermediate_size=16, num_hidden_layers=layers,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8, num_experts=4, num_experts_per_tok=2,
                         vocab_size=64, max_position_embeddings=128)
    return Qwen3MoeForCausalLM(cfg).eval()


def test_aux_states_copy_each_layers_input_residual_stream():
    m = _tiny_qwen3_moe()
    aux = ed.AuxStates(m, max_rows=4, max_prompt=32, dtype=torch.float32)
    assert aux.install() == 3 and aux.layers_idx == (2, 4, 5)
    ids = torch.randint(0, 64, (1, 7))
    with torch.no_grad():
        aux.mode = ("prefill", 0)
        hs = m(input_ids=ids, output_hidden_states=True).hidden_states
    want = torch.cat([hs[i][0] for i in aux.layers_idx], -1)
    assert torch.equal(aux.pre[:7], want)
    # a later chunk lands at its offset
    with torch.no_grad():
        aux.mode = ("prefill", 7)
        hs2 = m(input_ids=ids[:, :3], output_hidden_states=True).hidden_states
    assert torch.equal(aux.pre[7:10], torch.cat([hs2[i][0] for i in aux.layers_idx], -1))
    # decode rows: [b, 1] inputs fill rows 0..b-1
    with torch.no_grad():
        aux.mode = "decode"
        hs3 = m(input_ids=torch.randint(0, 64, (3, 1)), output_hidden_states=True).hidden_states
    assert torch.equal(aux.dec[:3], torch.cat([hs3[i][:, 0] for i in aux.layers_idx], -1))
    # off: nothing is copied
    before = aux.dec.clone()
    with torch.no_grad():
        aux.mode = None
        m(input_ids=torch.randint(0, 64, (2, 1)))
    assert torch.equal(aux.dec, before)
    aux.remove()
    with pytest.raises(RuntimeError, match="already installed"):
        aux.install()
        aux.install()


def test_aux_states_need_a_decoder_stack():
    with pytest.raises(ValueError, match="model.layers"):
        ed.AuxStates(torch.nn.Linear(2, 2), max_rows=1, max_prompt=1)
