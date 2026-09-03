# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The fake-quantised eager attention is transformers' eager attention
with the fp8 paged kernel's roundings applied -- and NOTHING else. Off, it
must equal eager bit-for-bit-ish; on, it must perturb only the rows and
layers it was told to; its quantiser must match the kernel's own; and the
custom attention name must still get a causal mask (a name with no mask
entry attends non-causally and no error is raised)."""
import importlib.util
import os

import pytest
import torch

transformers = pytest.importorskip("transformers")

_HERE = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location(
    "step_decomp", os.path.join(_HERE, "..", "bench", "hybrid-g9", "step_decomp.py"))


def _load_harness():
    import sys
    mod = importlib.util.module_from_spec(_spec)
    sys.modules["step_decomp"] = mod
    _spec.loader.exec_module(mod)
    return mod


class _Mod:
    layer_idx = 0
    sliding_window = None
    is_sliding = False
    training = False
    num_key_value_groups = 2


class _SlidingMod(_Mod):
    sliding_window = 8
    is_sliding = True


def _rand(B=1, Hq=4, Hkv=2, T=12, D=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, Hq, T, D, generator=g) * 0.7
    k = torch.randn(B, Hkv, T, D, generator=g)
    v = torch.randn(B, Hkv, T, D, generator=g)
    mask = torch.full((T, T), torch.finfo(torch.float32).min).triu(1)[None, None]
    return q, k, v, mask


def _plain(q, k, v, mask):
    from transformers.models.llama.modeling_llama import eager_attention_forward
    out, _ = eager_attention_forward(_Mod(), q, k, v, mask, scaling=q.shape[-1] ** -0.5,
                                     dropout=0.0)
    return out


def test_off_equals_transformers_eager():
    sd = _load_harness()
    q, k, v, mask = _rand()
    sd._FQ.update(spec="", kg=4, vg=1, layers="all", frm=0)
    out, w = sd._fq_eager_attention(_Mod(), q, k, v, mask, scaling=q.shape[-1] ** -0.5)
    assert w is None and out.shape == (1, 12, 4, 32)
    assert torch.allclose(out, _plain(q, k, v, mask), atol=1e-5, rtol=1e-5)


def test_from_rows_untouched_rest_perturbed():
    sd = _load_harness()
    q, k, v, mask = _rand()
    ref = _plain(q, k, v, mask)
    sd._FQ.update(spec="qkvp", kg=4, vg=1, layers="all", frm=6)
    out, _ = sd._fq_eager_attention(_Mod(), q, k, v, mask, scaling=q.shape[-1] ** -0.5)
    assert torch.equal(out[:, :6], ref[:, :6]), "rows before `from` must be the bf16 path"
    rel = ((out[:, 6:] - ref[:, 6:]).norm() / ref[:, 6:].norm()).item()
    assert 1e-4 < rel < 0.2, rel


@pytest.mark.parametrize("spec", ["q", "k", "v", "kv", "qkvp"])
def test_each_letter_perturbs(spec):
    sd = _load_harness()
    q, k, v, mask = _rand()
    ref = _plain(q, k, v, mask)
    sd._FQ.update(spec=spec, kg=4, vg=1, layers="all", frm=0)
    out, _ = sd._fq_eager_attention(_Mod(), q, k, v, mask, scaling=q.shape[-1] ** -0.5)
    rel = ((out - ref).norm() / ref.norm()).item()
    assert 1e-5 < rel < 0.2, (spec, rel)


def test_p_alone_is_a_no_op_because_it_needs_the_v_scales():
    sd = _load_harness()
    q, k, v, mask = _rand()
    sd._FQ.update(spec="p", kg=4, vg=1, layers="all", frm=0)
    out, _ = sd._fq_eager_attention(_Mod(), q, k, v, mask, scaling=q.shape[-1] ** -0.5)
    assert torch.allclose(out, _plain(q, k, v, mask), atol=1e-5, rtol=1e-5)


def test_layer_filter_selects_by_sliding_attribute():
    sd = _load_harness()
    q, k, v, mask = _rand()
    ref = _plain(q, k, v, mask)
    s = q.shape[-1] ** -0.5
    sd._FQ.update(spec="kv", kg=4, vg=1, layers="full", frm=0)
    assert torch.allclose(sd._fq_eager_attention(_SlidingMod(), q, k, v, mask, scaling=s)[0], ref)
    assert not torch.allclose(sd._fq_eager_attention(_Mod(), q, k, v, mask, scaling=s)[0], ref)
    sd._FQ.update(spec="kv", kg=4, vg=1, layers="sliding", frm=0)
    assert not torch.allclose(sd._fq_eager_attention(_SlidingMod(), q, k, v, mask, scaling=s)[0], ref)
    assert torch.allclose(sd._fq_eager_attention(_Mod(), q, k, v, mask, scaling=s)[0], ref)


@pytest.mark.parametrize("group", [None, 16, 8])
def test_quantiser_is_bit_exact_with_the_kernel_package(group):
    fp8_kv = pytest.importorskip("fp8_kv")
    sd = _load_harness()
    x = torch.randn(3, 2, 5, 64, generator=torch.Generator().manual_seed(3))
    x[0, 0, 0] = 0.0                         # the all-zero row pins scale 1.0
    mine, _ = sd._fq_quant_dequant(x, group)
    qq, ss = fp8_kv.quantize_kv_fp8(x, group)
    ref = fp8_kv.dequant_kv_fp8_ref(qq, ss, dtype=torch.float32)
    assert torch.equal(mine, ref)


def test_register_installs_attention_and_mask_and_rejects_bad_letters():
    sd = _load_harness()
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    name = sd._fq_register("kv", 4, 1, "all", 3)
    assert ALL_ATTENTION_FUNCTIONS[name] is sd._fq_eager_attention
    assert ALL_MASK_ATTENTION_FUNCTIONS[name] is ALL_MASK_ATTENTION_FUNCTIONS["eager"]
    # the class-level mapping is what _preprocess_mask_arguments consults
    assert ALL_MASK_ATTENTION_FUNCTIONS._global_mapping[name] is ALL_MASK_ATTENTION_FUNCTIONS["eager"]
    assert sd._FQ == {"spec": "kv", "kg": 4, "vg": 1, "layers": "all", "frm": 3}
    with pytest.raises(ValueError):
        sd._fq_register("kx", 4, 1, "all", 0)


def _tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=97, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=256)
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg).eval()


def test_full_forward_under_the_custom_name_is_causal_and_fq_moves_it():
    """The integration that matters: the model run under `eager_fq` with the
    roundings OFF scores exactly what `eager` scores (so the custom name
    got its causal mask), and with them ON it scores something else."""
    sd = _load_harness()
    model = _tiny_llama()
    torch.manual_seed(1)
    ids = torch.randint(0, 97, (120,))
    prompt_len, steps = 16, 40
    with torch.no_grad():
        ref = sd.ppl_oracle_score_full(model, ids, prompt_len, steps, device="cpu")
        name = sd._fq_register("", 4, 1, "all", prompt_len)
        model.config._attn_implementation = name
        off = sd.ppl_oracle_score_full(model, ids, prompt_len, steps, device="cpu")
        sd._fq_register("qkvp", 4, 1, "all", prompt_len)
        on = sd.ppl_oracle_score_full(model, ids, prompt_len, steps, device="cpu")
    assert abs(off - ref) < 1e-5, (off, ref)
    assert on != off and abs(on - ref) < 0.5, (on, ref)


def test_missing_mask_is_refused_not_attended_non_causally():
    sd = _load_harness()
    q, k, v, _ = _rand()
    sd._FQ.update(spec="", kg=4, vg=1, layers="all", frm=0)
    with pytest.raises(RuntimeError, match="no attention mask"):
        sd._fq_eager_attention(_Mod(), q, k, v, None, scaling=q.shape[-1] ** -0.5)
    # a single-row decode step legitimately carries no mask
    out, _ = sd._fq_eager_attention(_Mod(), q[:, :, :1], k[:, :, :1], v[:, :, :1], None,
                                    scaling=q.shape[-1] ** -0.5)
    assert out.shape == (1, 1, 4, 32)



def test_layer_diff_hooks_and_compare_on_a_tiny_model():
    """Eager vs eager at the same position must read as zero at every part;
    a different position must not. Exercises the hook installer, the
    reference forward (context cleared, impl forced to eager) and the
    comparison -- the paged half needs a GPU and runs on the lane."""
    sd = _load_harness()
    model = _tiny_llama()
    torch.manual_seed(2)
    ids = torch.randint(0, 97, (40,))
    cap_a, cap_b = {}, {}
    sd._layer_diff_reference(model, ids[:20], cap_a)
    sd._layer_diff_reference(model, ids[:20], cap_b)
    rows, summ = sd._layer_diff_compare(cap_a, cap_b, sd._layer_kinds(model))
    assert len(rows) == 2 and all(r["attn"] == 0.0 and r["mlp"] == 0.0 and r["out"] == 0.0 for r in rows)
    assert [r["layer"] for r in rows] == [0, 1], "the layer key is the INDEX, never the error"
    assert summ["first_layer_over_5pct"] is None and summ["attn_rel_full"] == 0.0
    cap_c = {}
    sd._layer_diff_reference(model, ids[:21], cap_c)
    rows, summ = sd._layer_diff_compare(cap_a, cap_c, sd._layer_kinds(model))
    assert all(r["out"] > 0 for r in rows) and summ["out_rel"] > 0
    assert summ["first_layer_over_5pct"] in (0, 1, None)


def test_layer_diff_reference_restores_a_custom_attention_name():
    """HF modules share the model's config: the restore must put back the
    name that was there BEFORE the reference forward, on every module,
    not the 'eager' that a per-module save would have recorded."""
    sd = _load_harness()
    model = _tiny_llama()
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.llama.modeling_llama import eager_attention_forward
    ALL_ATTENTION_FUNCTIONS["paged_stub_for_test"] = eager_attention_forward
    model.config._attn_implementation = "paged_stub_for_test"
    ids = torch.randint(0, 97, (30,))
    sd._layer_diff_reference(model, ids[:12], {})
    names = {m.config._attn_implementation for m in model.modules() if hasattr(m, "config")}
    assert names == {"paged_stub_for_test"}, names
